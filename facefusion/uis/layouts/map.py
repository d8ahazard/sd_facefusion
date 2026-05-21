import gradio

from facefusion import process_manager, state_manager
from facefusion.ff_status import FFStatus
from facefusion.args import collect_step_args
from facefusion.core import process_step
from facefusion.filesystem import get_output_path_auto, is_directory
from facefusion.jobs import job_execution, job_helper, job_manager
from facefusion.target_registry import is_under_uploads, register_target_for_job
from facefusion.uis.components import (
    face_selector,
    media_targets,
    preview,
    queue_panel,
    source_picker,
    trim_frame,
)
from facefusion.uis.ui_helper import suggest_output_path
from facefusion.user_data import sync_active_target_path

MAP_PROCESS_START_BUTTON = None
MAP_PROCESS_QUEUE_BUTTON = None
MAP_PROCESS_CLEAR_BUTTON = None
MAP_PROCESS_STATUS = None


def pre_check() -> bool:
    return True


def render() -> gradio.Blocks:
    with gradio.Blocks() as layout:
        with gradio.Row():
            with gradio.Column(scale=7):
                gradio.Markdown('### Preview')
                with gradio.Blocks():
                    preview.render()
                with gradio.Blocks():
                    trim_frame.render()

                media_targets.render_map_dropdown()

                gradio.Markdown('### Sources')
                gradio.Markdown('Pick a source thumbnail then **Add** to map it. **Remove** drops the focused mapping row.')
                with gradio.Blocks():
                    source_picker.render()

            with gradio.Column(scale=5):
                gradio.Markdown('### Face mapping')
                gradio.Markdown(
                    'Click a detected face above, then use **Add** on a source row to assign it. '
                    'Assignments are exclusive across rows.'
                )
                with gradio.Blocks():
                    face_selector.render()
                gradio.Markdown('### Process')
                with gradio.Row(elem_classes=['ff-map-process-row']):
                    global MAP_PROCESS_START_BUTTON, MAP_PROCESS_QUEUE_BUTTON, MAP_PROCESS_CLEAR_BUTTON, MAP_PROCESS_STATUS
                    MAP_PROCESS_START_BUTTON = gradio.Button('Start', variant='primary')
                    MAP_PROCESS_QUEUE_BUTTON = gradio.Button('Queue')
                    MAP_PROCESS_CLEAR_BUTTON = gradio.Button('Clear', variant='stop')
                MAP_PROCESS_STATUS = gradio.Textbox(
                    label='Map process status',
                    value='',
                    interactive=False,
                    max_lines=2,
                )
    return layout


def _refresh_process_buttons() -> tuple:
    running = job_execution.is_busy()
    return (
        gradio.update(interactive=not running),
        gradio.update(interactive=True),
        gradio.update(interactive=True),
    )


def _apply_target_snapshot_for_processing(target_path: str) -> None:
    import copy

    media_targets.snapshot_current_target_mappings()
    saved = media_targets.get_target_mappings_snapshot(target_path)
    state_manager.set_item('reference_face_dict', copy.deepcopy(saved.get('reference_face_dict') or {}))
    state_manager.set_item('mapping_slot_keys', list(saved.get('mapping_slot_keys') or []))


def _build_step_args_for_target(target_path: str) -> dict:
    state_manager.set_item('target_path', target_path)
    step_args = collect_step_args()
    output_path = get_output_path_auto()
    step_args['output_path'] = output_path
    if is_directory(step_args.get('output_path')):
        step_args['output_path'] = suggest_output_path(output_path, target_path)
    return step_args


def _queue_active_target() -> str:
    sync_active_target_path()
    target_path = state_manager.get_item('target_path')
    if not target_path:
        return 'No active target selected.'
    if queue_panel.enqueue_target(target_path):
        return 'Queued active target.'
    return 'Failed to queue active target.'


def _start_active_target() -> str:
    if job_execution.is_busy():
        return 'A job is already running. Start is disabled.'
    sync_active_target_path()
    target_path = state_manager.get_item('target_path')
    if not target_path:
        return 'No active target selected.'

    _apply_target_snapshot_for_processing(target_path)
    step_args = _build_step_args_for_target(target_path)
    job_id = job_helper.suggest_job_id('ui')
    created = (
        job_manager.create_job(job_id)
        and job_manager.add_step(job_id, step_args)
        and job_manager.submit_job(job_id)
    )
    if not created:
        return 'Failed to create job.'

    register_target_for_job(target_path, job_id, uploaded=is_under_uploads(target_path))

    if job_execution.start_single_job(job_id, process_step):
        return 'Started processing active target (runs in background).'
    return 'Could not start job — queue may already be busy.'


def get_tab_select_outputs() -> list:
    """Outputs for the first Map tab select step (mapping rows refresh in a later .then())."""
    from facefusion.uis.components import media_targets, preview, source_picker

    outputs = []
    if media_targets.MAP_ACTIVE_TARGET_DROPDOWN is not None:
        outputs.append(media_targets.MAP_ACTIVE_TARGET_DROPDOWN)
    outputs.extend(preview.get_target_preview_output_components())
    if source_picker.SOURCE_PICKER_GALLERY is not None:
        outputs.append(source_picker.SOURCE_PICKER_GALLERY)
    return outputs


def on_tab_select():
    """When Map tab opens: sync active target, load preview, trim, and face mapping UI."""
    from facefusion import state_manager
    from facefusion.uis.components import face_selector, media_targets, preview, source_picker
    from facefusion.user_data import sync_active_target_path

    print('[FaceFusion] Map tab refresh started', flush=True)
    media_targets._init_target_state()
    paths = state_manager.get_item('target_paths') or []
    if paths:
        active = state_manager.get_item('active_target_index') or 0
        if active >= len(paths):
            state_manager.set_item('active_target_index', 0)
        sync_active_target_path()
        media_targets._swap_active_target(state_manager.get_item('target_path'))

    updates = []
    if media_targets.MAP_ACTIVE_TARGET_DROPDOWN is not None:
        updates.append(media_targets._target_dropdown_update())
    updates.extend(preview.refresh_target_preview_with_faces())
    if source_picker.SOURCE_PICKER_GALLERY is not None:
        updates.append(source_picker.refresh_gallery_update())
    print('[FaceFusion] Map tab refresh finished', flush=True)
    return tuple(updates)


def register_tab_select(map_tab) -> None:
    if map_tab is not None and hasattr(map_tab, 'select'):
        outputs = get_tab_select_outputs()
        face_sync_outputs = preview.get_map_face_sync_output_components()
        if outputs:
            event = map_tab.select(fn=on_tab_select, outputs=outputs, show_progress='hidden')
            if face_sync_outputs:
                event = event.then(
                    preview.refresh_map_face_sync,
                    outputs=face_sync_outputs,
                    show_progress='hidden',
                )
            mapping_outputs = face_selector.get_mapping_refresh_output_components()
            if mapping_outputs:
                event.then(
                    face_selector.build_mapping_refresh_updates,
                    outputs=mapping_outputs,
                    show_progress='hidden',
                )
            process_outputs = get_process_outputs()
            if process_outputs:
                event.then(
                    _refresh_process_buttons,
                    outputs=process_outputs,
                    show_progress='hidden',
                )


def get_process_outputs() -> list:
    outputs = []
    if MAP_PROCESS_START_BUTTON is not None:
        outputs.append(MAP_PROCESS_START_BUTTON)
    if MAP_PROCESS_QUEUE_BUTTON is not None:
        outputs.append(MAP_PROCESS_QUEUE_BUTTON)
    if MAP_PROCESS_CLEAR_BUTTON is not None:
        outputs.append(MAP_PROCESS_CLEAR_BUTTON)
    return outputs


def listen() -> None:
    preview.listen()
    trim_frame.listen()
    source_picker.listen()
    face_selector.listen()
    if MAP_PROCESS_START_BUTTON is not None and MAP_PROCESS_STATUS is not None:
        MAP_PROCESS_START_BUTTON.click(
            _start_active_target,
            outputs=[MAP_PROCESS_STATUS],
            _js='start_status',
        ).then(
            _refresh_process_buttons,
            outputs=get_process_outputs(),
            show_progress='hidden',
        )
    if MAP_PROCESS_QUEUE_BUTTON is not None and MAP_PROCESS_STATUS is not None:
        MAP_PROCESS_QUEUE_BUTTON.click(
            _queue_active_target,
            outputs=[MAP_PROCESS_STATUS],
        ).then(
            _refresh_process_buttons,
            outputs=get_process_outputs(),
            show_progress='hidden',
        )
    if MAP_PROCESS_CLEAR_BUTTON is not None:
        clear_outputs = []
        detected = face_selector.get_detected_faces_gallery_component()
        if detected is not None:
            clear_outputs.append(detected)
        clear_outputs.extend(face_selector.get_mapping_refresh_output_components())
        if clear_outputs:
            MAP_PROCESS_CLEAR_BUTTON.click(
                face_selector.clear_all_mappings_and_galleries,
                outputs=clear_outputs,
            )


def run(ui: gradio.Blocks) -> None:
    ui.launch(favicon_path='facefusion.ico', inbrowser=state_manager.get_item('open_browser'))
