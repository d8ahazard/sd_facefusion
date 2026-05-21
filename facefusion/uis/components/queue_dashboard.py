import os
from typing import List, Optional, Set, Tuple

import gradio
from gradio import SelectData

from facefusion import process_manager, state_manager
from facefusion.common_helper import get_first
from facefusion.core import process_step
from facefusion.ff_status import FFStatus
from facefusion.filesystem import filter_image_paths, is_image, is_video
from facefusion.jobs import job_execution, job_manager, job_queue_table, job_store
from facefusion.target_registry import cleanup_targets_for_job
from facefusion.uis.components.output import format_status
from facefusion.user_data import ensure_queue_v2_migrated
from facefusion.vision import get_video_frame, normalize_frame_color, read_static_image

JOB_PREVIEW_IMAGE: Optional[gradio.Image] = None
JOB_MAPPING_GALLERY: Optional[gradio.Gallery] = None
JOB_PROGRESS_HTML: Optional[gradio.HTML] = None
JOB_LIVE_PREVIEW: Optional[gradio.Image] = None
QUEUE_POLL_BUTTON: Optional[gradio.Button] = None

ACTIVE_QUEUE_TABLE: Optional[gradio.Dataframe] = None
HISTORY_QUEUE_TABLE: Optional[gradio.Dataframe] = None
ACTIVE_JOB_IDS: Optional[gradio.State] = None
HISTORY_JOB_IDS: Optional[gradio.State] = None
SELECTED_JOB_ID: Optional[gradio.State] = None

RUN_ALL_BUTTON: Optional[gradio.Button] = None
RUN_SELECTED_BUTTON: Optional[gradio.Button] = None
STOP_ALL_BUTTON: Optional[gradio.Button] = None
STOP_SELECTED_BUTTON: Optional[gradio.Button] = None
REMOVE_ALL_BUTTON: Optional[gradio.Button] = None
REMOVE_SELECTED_BUTTON: Optional[gradio.Button] = None

def _current_job_id() -> Optional[str]:
    return job_execution.get_current_job_id()


def _init_jobs() -> bool:
    jobs_path = state_manager.get_item('jobs_path')
    ensure_queue_v2_migrated(jobs_path)
    return job_manager.init_jobs(jobs_path)


def _sync_job_keys() -> None:
    for key in job_store.get_job_keys():
        state_manager.sync_item(key)  # type: ignore[arg-type]


def _button_updates() -> Tuple[gradio.update, ...]:
    running = job_execution.is_busy()
    return (
        gradio.update(interactive=not running),
        gradio.update(interactive=not running),
        gradio.update(interactive=True),
        gradio.update(interactive=True),
        gradio.update(interactive=not running),
        gradio.update(interactive=not running),
    )


def _selected_id_set(selected_job_id: Optional[str] = None, extra: Optional[Set[str]] = None) -> Set[str]:
    selected = set(extra or [])
    if selected_job_id:
        selected.add(selected_job_id)
    return selected


def _table_updates(selected_job_id: Optional[str] = None, selected_ids: Optional[Set[str]] = None) -> Tuple:
    sel = _selected_id_set(selected_job_id, selected_ids)
    _, active_rows, active_ids = job_queue_table.compose_active_rows(_current_job_id(), sel)
    _, history_rows, history_ids = job_queue_table.compose_history_rows(sel)
    return (
        gradio.update(value=active_rows),
        gradio.update(value=history_rows),
        active_ids,
        history_ids,
    )


def _empty_preview_updates() -> Tuple:
    return (
        gradio.update(value=None),
        gradio.update(value=[]),
        gradio.update(value=format_queue_status()),
        gradio.update(value=None, visible=False),
    )


def _build_mapping_gallery(step_args: dict) -> List[Tuple[str, str]]:
    source_frame_dict = step_args.get('source_frame_dict') or {}
    mapping_slot_keys = step_args.get('mapping_slot_keys') or []
    if isinstance(source_frame_dict, dict):
        normalized = {int(k): list(v) for k, v in source_frame_dict.items()}
    else:
        normalized = {}
    items: List[Tuple[str, str]] = []
    for slot_key in mapping_slot_keys:
        slot_index = int(slot_key)
        paths = normalized.get(slot_index, [])
        thumb = get_first(filter_image_paths(paths or []))
        if thumb and os.path.isfile(thumb):
            items.append((thumb, f'Source {slot_index + 1}'))
    return items


def _build_target_preview(step_args: dict):
    target_path = step_args.get('target_path')
    if not target_path:
        return None
    if is_image(target_path):
        return read_static_image(target_path)
    if is_video(target_path):
        frame_number = step_args.get('reference_frame_number') or 0
        frame = get_video_frame(target_path, int(frame_number))
        if frame is not None:
            return normalize_frame_color(frame)
    return None


def _preview_updates_for_job(job_id: Optional[str]) -> Tuple:
    if not job_id:
        return _empty_preview_updates()
    step_args = job_queue_table.get_job_step_args(job_id)
    preview_frame = _build_target_preview(step_args)
    mapping_items = _build_mapping_gallery(step_args)
    live_preview = None
    live_visible = False
    status = FFStatus()
    if status.preview_image and os.path.exists(status.preview_image):
        if _current_job_id() == job_id or job_execution.is_busy():
            live_preview = status.preview_image
            live_visible = True
    return (
        gradio.update(value=preview_frame),
        gradio.update(value=mapping_items),
        gradio.update(value=format_queue_status()),
        gradio.update(value=live_preview, visible=live_visible),
    )


def format_queue_status() -> str:
    return format_status()


def render_preview() -> None:
    global JOB_PREVIEW_IMAGE
    global JOB_MAPPING_GALLERY
    global JOB_PROGRESS_HTML
    global JOB_LIVE_PREVIEW
    global QUEUE_POLL_BUTTON

    if not _init_jobs():
        return

    gradio.Markdown('### Job preview')
    JOB_PREVIEW_IMAGE = gradio.Image(label='Target', interactive=False, height=280)
    gradio.Markdown('**Face mappings**')
    JOB_MAPPING_GALLERY = gradio.Gallery(
        show_label=False,
        columns=8,
        rows=1,
        height=90,
        object_fit='cover',
        allow_preview=False,
        elem_classes=['ff-map-gallery', 'ff-queue-mapping-gallery'],
    )
    JOB_PROGRESS_HTML = gradio.HTML(
        elem_id='ff_queue_status',
        value=format_queue_status(),
    )
    JOB_LIVE_PREVIEW = gradio.Image(
        label='Live frame',
        interactive=False,
        visible=False,
        height=240,
    )
    QUEUE_POLL_BUTTON = gradio.Button(
        value='Poll',
        elem_id='ff_queue_poll',
        visible=False,
    )


def render_tables_and_actions() -> None:
    global ACTIVE_QUEUE_TABLE
    global HISTORY_QUEUE_TABLE
    global ACTIVE_JOB_IDS
    global HISTORY_JOB_IDS
    global SELECTED_JOB_ID
    global RUN_ALL_BUTTON
    global RUN_SELECTED_BUTTON
    global STOP_ALL_BUTTON
    global STOP_SELECTED_BUTTON
    global REMOVE_ALL_BUTTON
    global REMOVE_SELECTED_BUTTON

    if not _init_jobs():
        return

    headers, active_rows, active_ids = job_queue_table.compose_active_rows()
    _, history_rows, history_ids = job_queue_table.compose_history_rows()

    gradio.Markdown('### Active queue')
    ACTIVE_QUEUE_TABLE = gradio.Dataframe(
        headers=headers,
        value=active_rows,
        datatype=['bool', 'str', 'str', 'number', 'str'],
        interactive=True,
        show_label=False,
        elem_classes=['ff-queue-table'],
    )
    ACTIVE_JOB_IDS = gradio.State(active_ids)
    SELECTED_JOB_ID = gradio.State(None)

    gradio.Markdown('### History')
    HISTORY_QUEUE_TABLE = gradio.Dataframe(
        headers=headers,
        value=history_rows,
        datatype=['bool', 'str', 'str', 'number', 'str'],
        interactive=True,
        show_label=False,
        elem_classes=['ff-queue-table'],
    )
    HISTORY_JOB_IDS = gradio.State(history_ids)

    with gradio.Row(elem_classes=['ff-queue-actions']):
        RUN_ALL_BUTTON = gradio.Button('Run all', variant='primary')
        RUN_SELECTED_BUTTON = gradio.Button('Run selected')
    with gradio.Row(elem_classes=['ff-queue-actions']):
        STOP_ALL_BUTTON = gradio.Button('Stop all', variant='stop')
        STOP_SELECTED_BUTTON = gradio.Button('Stop selected', variant='stop')
    with gradio.Row(elem_classes=['ff-queue-actions']):
        REMOVE_ALL_BUTTON = gradio.Button('Remove all', variant='stop')
        REMOVE_SELECTED_BUTTON = gradio.Button('Remove selected', variant='stop')


def _preview_job_id(selected_job_id: Optional[str] = None) -> Optional[str]:
    return _current_job_id() or selected_job_id


def _full_refresh(
    selected_job_id: Optional[str] = None,
    selected_ids: Optional[Set[str]] = None,
) -> Tuple:
    table_updates = _table_updates(selected_job_id, selected_ids)
    preview_updates = _preview_updates_for_job(_preview_job_id(selected_job_id))
    button_updates = _button_updates()
    return table_updates + preview_updates + button_updates + (selected_job_id,)


def _collect_selected_ids(
    active_table,
    history_table,
    active_ids: List[str],
    history_ids: List[str],
    selected_job_id: Optional[str],
) -> Set[str]:
    selected = set(job_queue_table.get_selected_job_ids(active_table, active_ids or []))
    selected.update(job_queue_table.get_selected_job_ids(history_table, history_ids or []))
    if not selected and selected_job_id:
        selected.add(selected_job_id)
    return selected


def _run_all() -> Tuple:
    if job_execution.is_busy():
        return _full_refresh(None)
    FFStatus().start('Running job queue')
    job_execution.start_all_queued(process_step)
    return _full_refresh(None)


def _run_selected(
    active_table,
    history_table,
    active_ids: List[str],
    history_ids: List[str],
    selected_job_id: Optional[str],
) -> Tuple:
    all_selected = _collect_selected_ids(active_table, history_table, active_ids, history_ids, selected_job_id)
    active_selected = [job_id for job_id in all_selected if job_id in (active_ids or [])]
    history_selected = [job_id for job_id in all_selected if job_id in (history_ids or [])]

    if not active_selected and not history_selected:
        return _full_refresh(selected_job_id, all_selected)

    if job_execution.is_busy():
        return _full_refresh(selected_job_id, all_selected)

    FFStatus().start('Running selected jobs')
    failed_ids = set(job_manager.find_job_ids('failed') or [])
    specs = []
    for job_id in active_selected:
        specs.append((job_id, False))
    for job_id in history_selected:
        specs.append((job_id, job_id in failed_ids))
    job_execution.start_jobs_mixed(specs, process_step)
    return _full_refresh(None)


def _stop_all(selected_job_id: Optional[str]) -> Tuple:
    process_manager.stop()
    FFStatus().finish('Stopped')
    return _full_refresh(selected_job_id)


def _stop_selected(
    active_table,
    history_table,
    active_ids: List[str],
    history_ids: List[str],
    selected_job_id: Optional[str],
) -> Tuple:
    active_selected = job_queue_table.get_selected_job_ids(active_table, active_ids or [])
    history_selected = job_queue_table.get_selected_job_ids(history_table, history_ids or [])
    current = _current_job_id()
    should_stop = job_execution.is_busy() and (
        (not active_selected and not history_selected)
        or (current and current in active_selected)
        or (selected_job_id and selected_job_id in active_selected)
    )
    if should_stop:
        process_manager.stop()
        FFStatus().finish('Stopped')
    return _full_refresh(selected_job_id)


def _remove_all() -> Tuple:
    for job_id in job_queue_table.get_all_ui_job_ids():
        cleanup_targets_for_job(job_id, 'deleted')
        job_manager.delete_job(job_id)
    return _full_refresh(None)


def _remove_selected(
    active_table,
    history_table,
    active_ids: List[str],
    history_ids: List[str],
    selected_job_id: Optional[str],
) -> Tuple:
    to_remove = set(job_queue_table.get_selected_job_ids(active_table, active_ids or []))
    to_remove.update(job_queue_table.get_selected_job_ids(history_table, history_ids or []))
    if not to_remove and selected_job_id and job_queue_table.is_ui_job(selected_job_id):
        to_remove.add(selected_job_id)
    for job_id in to_remove:
        cleanup_targets_for_job(job_id, 'deleted')
        job_manager.delete_job(job_id)
    new_selection = None if selected_job_id in to_remove else selected_job_id
    return _full_refresh(new_selection)


def _on_active_select(evt: SelectData, active_ids: List[str]) -> Tuple:
    row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row_index is None or not active_ids or row_index < 0 or row_index >= len(active_ids):
        return _full_refresh(None)
    job_id = active_ids[row_index]
    return _full_refresh(job_id)


def _on_history_select(evt: SelectData, history_ids: List[str]) -> Tuple:
    row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row_index is None or not history_ids or row_index < 0 or row_index >= len(history_ids):
        return _full_refresh(None)
    job_id = history_ids[row_index]
    return _full_refresh(job_id)


def poll_queue_preview(selected_job_id: Optional[str]) -> Tuple:
    preview_job_id = _current_job_id() or selected_job_id
    table_updates = _table_updates(selected_job_id)
    preview_updates = _preview_updates_for_job(preview_job_id)
    button_updates = _button_updates()
    return table_updates + preview_updates + button_updates


def _refresh_on_tab() -> Tuple:
    return _full_refresh(None)


def _poll_button_outputs() -> list:
    return [
        RUN_ALL_BUTTON,
        RUN_SELECTED_BUTTON,
        STOP_ALL_BUTTON,
        STOP_SELECTED_BUTTON,
        REMOVE_ALL_BUTTON,
        REMOVE_SELECTED_BUTTON,
    ]


def listen() -> None:
    if ACTIVE_QUEUE_TABLE is None:
        return

    outputs = [
        ACTIVE_QUEUE_TABLE,
        HISTORY_QUEUE_TABLE,
        ACTIVE_JOB_IDS,
        HISTORY_JOB_IDS,
        JOB_PREVIEW_IMAGE,
        JOB_MAPPING_GALLERY,
        JOB_PROGRESS_HTML,
        JOB_LIVE_PREVIEW,
        RUN_ALL_BUTTON,
        RUN_SELECTED_BUTTON,
        STOP_ALL_BUTTON,
        STOP_SELECTED_BUTTON,
        REMOVE_ALL_BUTTON,
        REMOVE_SELECTED_BUTTON,
        SELECTED_JOB_ID,
    ]

    ACTIVE_QUEUE_TABLE.select(
        _on_active_select,
        inputs=[ACTIVE_JOB_IDS],
        outputs=outputs,
    )
    HISTORY_QUEUE_TABLE.select(
        _on_history_select,
        inputs=[HISTORY_JOB_IDS],
        outputs=outputs,
    )

    poll_outputs = _poll_button_outputs()
    selected_inputs = [
        ACTIVE_QUEUE_TABLE,
        HISTORY_QUEUE_TABLE,
        ACTIVE_JOB_IDS,
        HISTORY_JOB_IDS,
        SELECTED_JOB_ID,
    ]

    RUN_ALL_BUTTON.click(_run_all, outputs=outputs, _js='start_status')
    RUN_SELECTED_BUTTON.click(_button_updates, outputs=poll_outputs, _js='start_status')
    RUN_SELECTED_BUTTON.click(_run_selected, inputs=selected_inputs, outputs=outputs)
    STOP_ALL_BUTTON.click(_stop_all, inputs=[SELECTED_JOB_ID], outputs=outputs, _js='stop_status')
    STOP_SELECTED_BUTTON.click(_button_updates, outputs=poll_outputs, _js='stop_status')
    STOP_SELECTED_BUTTON.click(_stop_selected, inputs=selected_inputs, outputs=outputs)
    REMOVE_ALL_BUTTON.click(_remove_all, outputs=outputs)
    REMOVE_SELECTED_BUTTON.click(
        _remove_selected,
        inputs=[ACTIVE_QUEUE_TABLE, HISTORY_QUEUE_TABLE, ACTIVE_JOB_IDS, HISTORY_JOB_IDS, SELECTED_JOB_ID],
        outputs=outputs,
    )

    if QUEUE_POLL_BUTTON is not None:
        QUEUE_POLL_BUTTON.click(
            poll_queue_preview,
            inputs=[SELECTED_JOB_ID],
            outputs=outputs[:-1],
            show_progress=False,
        )
