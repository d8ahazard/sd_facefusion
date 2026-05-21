import importlib
import inspect
import os
import pkgutil
import time
from typing import List, Union
from PIL import Image

import gradio as gr

from facefusion import state_manager
from facefusion.args import apply_args, collect_step_args
from facefusion.core import route
from facefusion.download import conditional_download
from facefusion.filesystem import output_dir, get_output_path_auto
from facefusion.memory import tune_performance
from facefusion.processors.core import get_processors_modules
from facefusion.program import create_program
from facefusion.program_helper import validate_args
from facefusion.uis.core import load_ui_layout_module, reload_all_settings, get_reload_outputs
from facefusion.workers.core import get_worker_modules
from facefusion.uis.components.instant_runner import create_and_run_job
from facefusion.filesystem import TEMP_DIRECTORY_PATH
from facefusion.ffmpeg import print_ffmpeg_capabilities, detect_nvenc_encoder
from modules import script_callbacks, scripts, scripts_postprocessing
from modules.shared import cmd_opts

# export CUDA_MODULE_LOADING=LAZY
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'


def run_pre_checks(package):
    def find_submodules(package):
        if hasattr(package, '__path__'):
            for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
                module = importlib.import_module(modname)
                if hasattr(module, 'pre_check') and inspect.isfunction(module.pre_check):
                    module.pre_check()
                if hasattr(module, "MODELS"):
                    for model in module.MODELS:
                        model_path = os.path.dirname(module.MODELS[model]["path"])
                        if model_path.endswith("models"):
                            model_path = model_path[:-7]
                        model_url = module.MODELS[model]["url"]
                        if "inswapper" in model_url:
                            continue
                        conditional_download(model_path, [model_url])
                find_submodules(module)
    find_submodules(package)


def run_preloads(_, __):
    all_processors = get_processors_modules()
    all_workers = get_worker_modules()
    for processor in all_processors:
        print(f"Preloading processor {processor.display_name}")
        processor.pre_load()
    for worker in all_workers:
        print(f"Preloading worker {worker.display_name}")
        worker.pre_load()


def _init_facefusion_state():
    from facefusion import logger, globals, state_manager as sm
    from onnxruntime import get_available_providers

    providers = get_available_providers()
    missing_providers = [p for p in ['CUDAExecutionProvider', 'TensorrtExecutionProvider'] if p not in providers]
    if missing_providers:
        logger.warn(
            f"Warning: The following execution providers are not available, please force-reinstall onnxruntime-gpu: {missing_providers}.",
            __name__,
        )
    globals.output_path = output_dir
    sm.init_item('output_path', output_dir)
    program = create_program()
    og_args = vars(program.parse_args())
    program.add_argument_group('processors')
    all_processors = get_processors_modules()
    all_workers = get_worker_modules()
    for processor in all_processors:
        processor.register_args(program)
        processor.apply_args(og_args, sm.init_item)
    for worker in all_workers:
        worker.register_args(program)
        worker.apply_args(og_args, sm.init_item)

    globals_dict = {}
    if validate_args(program):
        args = vars(program.parse_args())
        ff_args = {key: args[key] for key in args if key not in og_args}
        globals_dict.update(ff_args)
        if sm.get_item('command'):
            logger.init(sm.get_item('log_level'))
            route(args)

    for key in globals.__dict__:
        if not key.startswith('__') and key not in globals_dict:
            globals_dict[key] = globals.__dict__[key]

    ff_ini = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'facefusion.ini'))
    globals_dict['config_path'] = ff_ini
    with open(ff_ini, 'r') as f:
        for line in f:
            if "=" not in line or line.startswith("#"):
                continue
            key, value = line.strip().split('=')
            if value != 'None' and value != "" and value != "''":
                print(f"Setting {key} to {value} from facefusion.ini")
                globals_dict[key] = value
    apply_args(globals_dict, False)
    sm.init_item("config_path", ff_ini)

    # Saved ui_defaults must be applied last — globals reset auto_padding etc. to module defaults.
    try:
        from facefusion.user_data import apply_saved_defaults_over_state
        apply_saved_defaults_over_state()
    except ImportError:
        pass

    if sm.get_item('auto_padding_model') is None:
        sm.init_item('auto_padding_model', 'None')
    if sm.get_item('auto_padding_confidence') is None:
        sm.init_item('auto_padding_confidence', 0.5)
    if sm.get_item('auto_padding_intersection_threshold') is None:
        sm.init_item('auto_padding_intersection_threshold', 50)
    if sm.get_item('auto_padding_mask_areas') is None:
        sm.init_item('auto_padding_mask_areas', ['upper-face', 'lower-face', 'mouth'])
    if sm.get_item('face_mask_blur') is None:
        sm.init_item('face_mask_blur', 0.3)
    if sm.get_item('face_mask_padding') is None:
        sm.init_item('face_mask_padding', (0, 0, 0, 0))

    if sm.get_item('target_paths') is None:
        tp = sm.get_item('target_path')
        sm.init_item('target_paths', [tp] if tp else [])
    if sm.get_item('active_target_index') is None:
        sm.init_item('active_target_index', 0)
    if sm.get_item('remove_target_on_job_completion') is None:
        sm.init_item('remove_target_on_job_completion', True)
    if sm.get_item('preview_update_seconds') is None:
        sm.init_item('preview_update_seconds', 2.0)

    if not sm.get_item('deep_swapper_model'):
        sm.init_item('deep_swapper_model', 'druuzil/elon_musk_320')
    if not sm.get_item('background_remover_model'):
        sm.init_item('background_remover_model', 'modnet')

    try:
        from facefusion.user_data import load_media_targets
        load_media_targets()
    except ImportError:
        pass

    tuned_threads, _, _ = tune_performance()
    current_threads = sm.get_item('execution_thread_count')
    if current_threads is None or current_threads < 1:
        sm.set_item('execution_thread_count', tuned_threads)
    elif current_threads > tuned_threads:
        print(
            f"WARNING: execution_thread_count={current_threads} exceeds VRAM-tuned maximum "
            f"{tuned_threads}, capping to tuned value"
        )
        sm.set_item('execution_thread_count', tuned_threads)

    nvenc_encoder = detect_nvenc_encoder()
    if nvenc_encoder:
        current_encoder = sm.get_item('output_video_encoder')
        if current_encoder in ['libx264', 'libx265', None]:
            sm.set_item('output_video_encoder', nvenc_encoder)
            print(f"Auto-selected NVENC encoder: {nvenc_encoder}")

    print_ffmpeg_capabilities()


def _render_classic_ui():
    with gr.Tabs():
        with gr.Tab(label="File"):
            default_layout = load_ui_layout_module("default")
            default_layout.render()
            default_layout.listen()
        with gr.Tab(label="Live"):
            live_layout = load_ui_layout_module("webcam")
            live_layout.render()
            live_layout.listen()
        with gr.Tab(label="Benchmark"):
            bench_layout = load_ui_layout_module("benchmark")
            bench_layout.render()
            bench_layout.listen()
    return "RD FaceFusion"


def _render_modern_ui():
    settings_layout = load_ui_layout_module("settings")
    media_layout = load_ui_layout_module("media")
    map_layout = load_ui_layout_module("map")
    queue_layout = load_ui_layout_module("queue")

    with gr.Tabs():
        with gr.Tab(label="Settings"):
            settings_layout.render()
        with gr.Tab(label="Media"):
            media_layout.render()
        with gr.Tab(label="Map") as map_tab:
            map_layout.render()
        with gr.Tab(label="Queue") as queue_tab:
            queue_layout.render()

    # Listen after all tabs render so shared components (e.g. preview_frame_slider) exist.
    settings_layout.listen()
    media_layout.listen()
    map_layout.listen()
    queue_layout.listen()

    map_layout.register_tab_select(map_tab)
    queue_layout.register_tab_select(queue_tab)
    return "FaceFusion"


def _extension_css() -> str:
    css_path = os.path.join(os.path.dirname(__file__), '..', 'style.css')
    if os.path.isfile(css_path):
        with open(css_path, encoding='utf-8') as css_file:
            return css_file.read()
    return ''


def load_facefusion():
    _init_facefusion_state()
    use_classic = getattr(cmd_opts, 'ff_classic_ui', False)
    state_manager.init_item('ff_classic_ui', use_classic)

    try:
        from facefusion.face_analyser import ensure_inference_pools_ready
        ensure_inference_pools_ready()
    except Exception as exc:
        print(f'[FaceFusion] Face analyser warmup skipped: {exc}', flush=True)

    with gr.Blocks(css=_extension_css()) as ff_ui:
        tab_label = _render_classic_ui() if use_classic else _render_modern_ui()

        reload_outputs = get_reload_outputs()
        post_job_reload_outputs = []
        try:
            from facefusion.uis.ui_sync import get_post_job_reload_outputs
            post_job_reload_outputs = get_post_job_reload_outputs()
        except Exception:
            post_job_reload_outputs = reload_outputs or []

        def _on_facefusion_tab_load():
            run_preloads(None, None)
            if reload_outputs:
                return reload_all_settings()
            return []

        def _on_post_job_reload():
            from facefusion.uis.ui_sync import reload_ui_after_job
            return reload_ui_after_job()

        ff_ui.load(fn=_on_facefusion_tab_load, outputs=reload_outputs or [])

        gr.Button(visible=False, elem_id='ff_settings_reload').click(
            fn=_on_post_job_reload,
            outputs=post_job_reload_outputs or reload_outputs or [],
            show_progress=False,
        )

        return ((ff_ui, tab_label, "ff_ui_clean"),)


script_callbacks.on_ui_tabs(load_facefusion)


def update_source_faces(file_paths: List[str]) -> None:
    if not file_paths:
        print("No source faces provided")
        return
    source_dict = state_manager.get_item('source_frame_dict')
    if not source_dict:
        source_dict = {}
    source_dict[0] = file_paths
    state_manager.set_item('source_paths', file_paths)
    state_manager.set_item('source_frame_dict', source_dict)
    print(f"Updated source_frame_dict: {source_dict}")


def process_internal(is_ff_enabled, image, source_paths=None):
    if not is_ff_enabled:
        print("FaceFusion is disabled")
        return
    if not source_paths or not any(source_paths):
        print("No source faces selected")
        return

    print("FaceFusion is enabled")
    temp_dir = TEMP_DIRECTORY_PATH
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    temp_name = f"facefusion_{time.time()}"
    temp_file = os.path.join(temp_dir, f"{temp_name}.jpg")
    image.save(temp_file)
    print(f"FaceFusion processing image: {temp_file}")

    output_dir_path = os.path.join(temp_dir, "output")
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    output_path = os.path.join(output_dir_path, f"{temp_name}_output.jpg")

    update_source_faces(source_paths)
    step_args = collect_step_args()
    step_args['target_path'] = temp_file
    step_args['output_path'] = output_path
    step_args['face_selector_mode'] = 'one'

    success = create_and_run_job(step_args, keep_state=True)

    if success and os.path.exists(output_path):
        print(f"FaceFusion succeeded: {output_path}")
        with Image.open(output_path) as img:
            result_image = img.copy()
        os.remove(temp_file)
        os.remove(output_path)
        try:
            os.rmdir(output_dir_path)
        except OSError:
            pass
        return result_image
    print("FaceFusion failed")
    return None


class FaceFusionScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.is_ff_enabled = False
        self.source_paths = []

    def title(self):
        return "FaceFusion"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("FaceFusion", open=False):
            with gr.Row():
                enable = gr.Checkbox(
                    label="Process with FaceFusion",
                    value=False,
                    visible=True,
                )
            with gr.Row():
                source_images = gr.Files(
                    label="Source Face(s)",
                    file_types=["image"],
                    visible=True
                )
        return [enable, source_images]

    def postprocess_image(self, p, pp, enable, source_files):
        self.is_ff_enabled = enable
        self.source_paths = [f.name for f in source_files] if source_files else []
        result_image = process_internal(self.is_ff_enabled, pp.image, self.source_paths)
        if result_image:
            pp.image = result_image


class FaceFusionPostProcessing(scripts_postprocessing.ScriptPostprocessing):
    name = "FaceFusion"
    order = 1999

    def __init__(self):
        super().__init__()
        self.is_ff_enabled = False
        self.source_paths = []

    def ui(self):
        with gr.Accordion("FaceFusion", open=False):
            with gr.Row():
                enable = gr.Checkbox(
                    label="Process with FaceFusion",
                    value=False,
                    visible=True,
                )
            with gr.Row():
                source_images = gr.Files(
                    label="Source Face(s)",
                    file_types=["image"],
                    visible=True
                )
        return {
            "is_ff_enabled": enable,
            "source_files": source_images
        }

    def process(self, pp, *, is_ff_enabled, source_files):
        self.is_ff_enabled = is_ff_enabled
        self.source_paths = [f.name for f in source_files] if source_files else []
        result_image = process_internal(self.is_ff_enabled, pp.image, self.source_paths)
        if result_image:
            pp.image = result_image
        else:
            print("FaceFusion failed")
