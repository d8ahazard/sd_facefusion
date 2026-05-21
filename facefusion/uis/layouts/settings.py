import gradio

from facefusion import state_manager
from facefusion.uis.components import (
    age_modifier_options,
    background_remover_options,
    deep_swapper_options,
    common_options,
    execution,
    execution_queue_count,
    execution_thread_count,
    expression_restorer_options,
    face_buffer_options,
    face_debugger_options,
    face_detector,
    face_editor_options,
    face_enhancer_options,
    face_landmarker,
    face_masker,
    face_selector_settings,
    face_swapper_options,
    frame_colorizer_options,
    frame_enhancer_options,
    lip_syncer_options,
    output_options,
    preview_options,
    processors,
    settings_defaults,
    style_changer_options,
    temp_frame,
)


def pre_check() -> bool:
    return True


def render() -> gradio.Blocks:
    with gradio.Blocks() as layout:
        with gradio.Row():
            with gradio.Column(scale=4):
                settings_defaults.render()
                with gradio.Blocks():
                    processors.render()
                with gradio.Blocks():
                    age_modifier_options.render()
                with gradio.Blocks():
                    expression_restorer_options.render()
                with gradio.Blocks():
                    face_debugger_options.render()
                with gradio.Blocks():
                    face_editor_options.render()
                with gradio.Blocks():
                    face_enhancer_options.render()
                with gradio.Blocks():
                    face_swapper_options.render()
                with gradio.Blocks():
                    deep_swapper_options.render()
                with gradio.Blocks():
                    background_remover_options.render()
                with gradio.Blocks():
                    frame_colorizer_options.render()
                with gradio.Blocks():
                    frame_enhancer_options.render()
                with gradio.Blocks():
                    lip_syncer_options.render()
                with gradio.Blocks():
                    style_changer_options.render()
                with gradio.Blocks():
                    execution.render()
                    execution_thread_count.render()
                    execution_queue_count.render()
                    preview_options.render()
                with gradio.Blocks():
                    temp_frame.render()
                with gradio.Blocks():
                    output_options.render()
                with gradio.Blocks():
                    common_options.render()
            with gradio.Column(scale=4):
                with gradio.Blocks():
                    face_selector_settings.render()
                with gradio.Blocks():
                    face_detector.render()
                with gradio.Blocks():
                    face_landmarker.render()
                with gradio.Blocks():
                    face_masker.render()
                with gradio.Blocks():
                    face_buffer_options.render()
    return layout


def listen() -> None:
    settings_defaults.listen()
    processors.listen()
    age_modifier_options.listen()
    expression_restorer_options.listen()
    face_debugger_options.listen()
    face_editor_options.listen()
    face_enhancer_options.listen()
    face_swapper_options.listen()
    deep_swapper_options.listen()
    background_remover_options.listen()
    frame_colorizer_options.listen()
    frame_enhancer_options.listen()
    lip_syncer_options.listen()
    style_changer_options.listen()
    execution.listen()
    execution_thread_count.listen()
    execution_queue_count.listen()
    preview_options.listen()
    temp_frame.listen()
    output_options.listen()
    common_options.listen()
    face_selector_settings.listen()
    face_detector.listen()
    face_landmarker.listen()
    face_masker.listen()
    face_buffer_options.listen()


def run(ui: gradio.Blocks) -> None:
    ui.launch(favicon_path='facefusion.ico', inbrowser=state_manager.get_item('open_browser'))
