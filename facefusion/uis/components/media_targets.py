import copy
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio

from facefusion import state_manager
from facefusion.download import download_video
from facefusion.face_store import clear_reference_faces, clear_static_faces
from facefusion.filesystem import is_directory, is_image, is_url, is_video, list_directory
from facefusion.uis.components.face_selector import clear_selected_faces
from facefusion.uis.components.target import update_from_path
from facefusion.uis.core import register_ui_component
from facefusion.uis.typing import File
from facefusion.target_registry import register_uploaded_target
from facefusion.user_data import save_media_targets, sync_active_target_path

TARGET_LIST: Optional[gradio.Dataframe] = None
REMOVE_TARGET_ON_COMPLETE_CHECKBOX: Optional[gradio.Checkbox] = None
TARGET_FILES_ADD: Optional[gradio.File] = None
TARGET_URL_INPUT: Optional[gradio.Textbox] = None
TARGET_URL_ADD_BUTTON: Optional[gradio.Button] = None
TARGET_FOLDER_INPUT: Optional[gradio.Textbox] = None
TARGET_FOLDER_ADD_BUTTON: Optional[gradio.Button] = None
TARGET_REMOVE_BUTTON: Optional[gradio.Button] = None
ACTIVE_TARGET_DROPDOWN: Optional[gradio.Dropdown] = None
MAP_ACTIVE_TARGET_DROPDOWN: Optional[gradio.Dropdown] = None

# Session-only: per-target face-mapping snapshots. Wiped on process restart.
# Shape: { target_path: {'reference_face_dict': dict, 'mapping_slot_keys': list,
#                        'current_selected_faces_by_slot': dict} }
_target_face_mappings: Dict[str, Dict[str, Any]] = {}
_last_active_target_path: Optional[str] = None


def _init_target_state():
    if state_manager.get_item('target_paths') is None:
        tp = state_manager.get_item('target_path')
        state_manager.init_item('target_paths', [tp] if tp else [])
    if state_manager.get_item('active_target_index') is None:
        state_manager.init_item('active_target_index', 0)
    if state_manager.get_item('remove_target_on_job_completion') is None:
        state_manager.init_item('remove_target_on_job_completion', True)
    if not state_manager.get_item('target_paths'):
        from facefusion.user_data import load_media_targets
        load_media_targets()
    sync_active_target_path()


def _persist_target_state() -> None:
    save_media_targets()


def _list_display() -> List[List[str]]:
    paths = state_manager.get_item('target_paths') or []
    active = state_manager.get_item('active_target_index') or 0
    rows = []
    for i, p in enumerate(paths):
        mark = ' *' if i == active else ''
        rows.append([str(i), os.path.basename(p) + mark, p])
    return rows


def _dropdown_choices() -> List[str]:
    paths = state_manager.get_item('target_paths') or []
    return [f'{i}: {os.path.basename(p)}' for i, p in enumerate(paths)]


def _normalize_dropdown_choice(choice) -> Optional[str]:
    """Gradio 3.x passes a str; Gradio 4/5 may pass a list for Dropdown values."""
    if choice is None:
        return None
    if isinstance(choice, (list, tuple)):
        if not choice:
            return None
        choice = choice[0]
    if not isinstance(choice, str):
        choice = str(choice)
    choice = choice.strip()
    return choice or None


def render() -> None:
    global TARGET_LIST
    global TARGET_FILES_ADD
    global TARGET_URL_INPUT
    global TARGET_URL_ADD_BUTTON
    global TARGET_FOLDER_INPUT
    global TARGET_FOLDER_ADD_BUTTON
    global TARGET_REMOVE_BUTTON
    global ACTIVE_TARGET_DROPDOWN

    _init_target_state()
    choices = _dropdown_choices()
    active_val = choices[state_manager.get_item('active_target_index') or 0] if choices else None

    ACTIVE_TARGET_DROPDOWN = gradio.Dropdown(
        label='Active target (for Map tab)',
        choices=choices,
        value=active_val,
        interactive=True,
        allow_custom_value=False,
    )
    global _last_active_target_path
    _last_active_target_path = state_manager.get_item('target_path')
    TARGET_LIST = gradio.Dataframe(
        headers=['#', 'Name', 'Path'],
        value=_list_display(),
        interactive=False,
    )
    TARGET_FILES_ADD = gradio.File(
        label='Add target files',
        file_count='multiple',
        file_types=['image', 'video'],
    )
    TARGET_URL_INPUT = gradio.Textbox(label='Target URL or path', placeholder='Paste URL or local path')
    TARGET_URL_ADD_BUTTON = gradio.Button('Add URL/path')
    TARGET_FOLDER_INPUT = gradio.Textbox(label='Add folder', placeholder='Directory path')
    TARGET_FOLDER_ADD_BUTTON = gradio.Button('Add folder')
    TARGET_REMOVE_BUTTON = gradio.Button('Remove active target', variant='stop')
    global REMOVE_TARGET_ON_COMPLETE_CHECKBOX
    remove_on_complete = state_manager.get_item('remove_target_on_job_completion')
    if remove_on_complete is None:
        remove_on_complete = True
    REMOVE_TARGET_ON_COMPLETE_CHECKBOX = gradio.Checkbox(
        label='Remove uploaded target when job completes',
        value=bool(remove_on_complete),
    )

    register_ui_component('media_active_target_dropdown', ACTIVE_TARGET_DROPDOWN)


def render_map_dropdown() -> gradio.Dropdown:
    """Second dropdown rendered on the Map tab; kept in sync with the Media one."""
    global MAP_ACTIVE_TARGET_DROPDOWN

    _init_target_state()
    choices = _dropdown_choices()
    active_val = choices[state_manager.get_item('active_target_index') or 0] if choices else None

    MAP_ACTIVE_TARGET_DROPDOWN = gradio.Dropdown(
        label='Active target',
        choices=choices,
        value=active_val,
        interactive=True,
        allow_custom_value=False,
    )
    register_ui_component('map_active_target_dropdown', MAP_ACTIVE_TARGET_DROPDOWN)
    return MAP_ACTIVE_TARGET_DROPDOWN


def _snapshot_target_mappings(target_path: Optional[str]) -> None:
    if not target_path:
        return
    _target_face_mappings[target_path] = {
        'reference_face_dict': copy.deepcopy(state_manager.get_item('reference_face_dict') or {}),
        'mapping_slot_keys': list(state_manager.get_item('mapping_slot_keys') or []),
    }


def _restore_target_mappings(target_path: Optional[str]) -> None:
    """Restore mappings for the new active target; clear if none recorded."""
    from facefusion.uis.components import face_selector

    saved = _target_face_mappings.get(target_path or '') or {}
    state_manager.set_item('reference_face_dict', copy.deepcopy(saved.get('reference_face_dict') or {}))
    state_manager.set_item('mapping_slot_keys', list(saved.get('mapping_slot_keys') or []))
    face_selector.restore_selected_faces_from_state()


def _target_dropdown_update() -> gradio.update:
    choices = _dropdown_choices()
    active = state_manager.get_item('active_target_index') or 0
    active_val = choices[active] if choices and active < len(choices) else None
    return gradio.update(choices=choices, value=active_val)


def _refresh_outputs() -> Tuple:
    """Media-tab handler outputs: list, active dropdown, file add, url."""
    return (
        gradio.update(value=_list_display()),
        _target_dropdown_update(),
        gradio.update(value=None),
        gradio.update(value=''),
    )


def _refresh_outputs_with_map_dropdown() -> Tuple:
    """Same as _refresh_outputs, plus sync the Map tab target dropdown."""
    updates = list(_refresh_outputs())
    if MAP_ACTIVE_TARGET_DROPDOWN is not None:
        updates.append(_target_dropdown_update())
    return tuple(updates)


def get_media_target_handler_outputs() -> List:
    """Output components for Media target add/remove handlers."""
    outputs = [TARGET_LIST, ACTIVE_TARGET_DROPDOWN, TARGET_FILES_ADD, TARGET_URL_INPUT]
    if MAP_ACTIVE_TARGET_DROPDOWN is not None:
        outputs.append(MAP_ACTIVE_TARGET_DROPDOWN)
    return outputs


def build_target_ui_updates() -> Tuple:
    """Workflow refresh: active + map dropdowns, then list/files/url."""
    return (
        _target_dropdown_update(),
        _target_dropdown_update(),
        gradio.update(value=_list_display()),
        gradio.update(value=None),
        gradio.update(value=''),
    )


def _clear_face_state():
    clear_reference_faces()
    clear_static_faces()
    clear_selected_faces()


def _swap_active_target(new_target_path: Optional[str]) -> None:
    """Snapshot mappings for the previous target, then restore for the new one."""
    global _last_active_target_path
    if _last_active_target_path and _last_active_target_path != new_target_path:
        _snapshot_target_mappings(_last_active_target_path)
    _restore_target_mappings(new_target_path)
    clear_reference_faces()
    clear_static_faces()
    _last_active_target_path = new_target_path


def snapshot_current_target_mappings() -> None:
    """Public entry for queue submission: persist the current target's mappings."""
    _snapshot_target_mappings(state_manager.get_item('target_path'))


def get_target_mappings_snapshot(target_path: str) -> Dict[str, Any]:
    return _target_face_mappings.get(target_path, {})


def refresh_map_target_dropdown() -> Tuple:
    """Refresh Map tab dropdown choices from current target_paths (e.g. on tab select)."""
    if MAP_ACTIVE_TARGET_DROPDOWN is None:
        return tuple()
    return (_target_dropdown_update(),)


def add_paths(new_paths: List[str], from_upload: bool = False) -> Tuple:
    paths = list(state_manager.get_item('target_paths') or [])
    for p in new_paths:
        if not p:
            continue
        if from_upload and os.path.isfile(p):
            p = register_uploaded_target(p)
        if p not in paths and (is_image(p) or is_video(p)):
            paths.append(p)
    state_manager.set_item('target_paths', paths)
    if paths and not state_manager.get_item('target_path'):
        state_manager.set_item('active_target_index', 0)
        sync_active_target_path()
        _swap_active_target(state_manager.get_item('target_path'))
    _persist_target_state()
    return _refresh_outputs_with_map_dropdown()


def add_files(files: List[File]) -> Tuple:
    names = [f.name for f in files] if files else []
    return add_paths(names, from_upload=True)


def update_remove_target_on_complete(enabled: bool) -> None:
    state_manager.set_item('remove_target_on_job_completion', bool(enabled))


def add_url_path(url: str) -> Tuple:
    if not url or not url.strip():
        return _refresh_outputs_with_map_dropdown()
    _, file_up = update_from_path(url.strip())
    path = state_manager.get_item('target_path')
    if path:
        return add_paths([path])
    return _refresh_outputs_with_map_dropdown()


def add_folder(folder: str) -> Tuple:
    if not folder or not is_directory(folder):
        return _refresh_outputs_with_map_dropdown()
    new_paths = []
    for name in list_directory(folder):
        full = os.path.join(folder, name)
        if is_image(full) or is_video(full):
            new_paths.append(full)
    return add_paths(new_paths)


def remove_active_target(choice: str) -> Tuple:
    choice = _normalize_dropdown_choice(choice)
    if not choice:
        return _refresh_outputs_with_map_dropdown()
    try:
        index = int(choice.split(':')[0])
    except ValueError:
        return _refresh_outputs_with_map_dropdown()
    return remove_at_index(index)


def remove_at_index(index: int) -> Tuple:
    paths = list(state_manager.get_item('target_paths') or [])
    idx = index
    removed_path = paths[idx] if 0 <= idx < len(paths) else None
    if 0 <= idx < len(paths):
        paths.pop(idx)
    state_manager.set_item('target_paths', paths)
    if removed_path:
        _target_face_mappings.pop(removed_path, None)
    active = state_manager.get_item('active_target_index') or 0
    if active >= len(paths):
        state_manager.set_item('active_target_index', max(0, len(paths) - 1))
    sync_active_target_path()
    _swap_active_target(state_manager.get_item('target_path'))
    _persist_target_state()
    return _refresh_outputs_with_map_dropdown()


def set_active(choice) -> None:
    choice = _normalize_dropdown_choice(choice)
    if not choice:
        return
    try:
        idx = int(choice.split(':')[0])
    except ValueError:
        return
    state_manager.set_item('active_target_index', idx)
    sync_active_target_path()
    _swap_active_target(state_manager.get_item('target_path'))
    _persist_target_state()


def listen() -> None:
    from facefusion.uis.components import face_selector as face_selector_module
    from facefusion.uis.components import preview as preview_module

    target_handler_outputs = get_media_target_handler_outputs()
    preview_outputs = preview_module.get_target_preview_output_components()
    face_sync_outputs = preview_module.get_map_face_sync_output_components()
    mapping_outputs = face_selector_module.get_mapping_refresh_output_components()

    def _then_preview_and_faces(event):
        if preview_outputs:
            event = event.then(
                preview_module.refresh_target_preview,
                outputs=preview_outputs,
                show_progress='hidden',
            )
        if face_sync_outputs:
            event = event.then(
                preview_module.refresh_map_face_sync,
                outputs=face_sync_outputs,
                show_progress='hidden',
            )
        if mapping_outputs:
            event = event.then(
                face_selector_module.build_mapping_refresh_updates,
                outputs=mapping_outputs,
                show_progress='hidden',
            )
        return event

    _then_preview_and_faces(
        TARGET_FILES_ADD.change(add_files, inputs=[TARGET_FILES_ADD], outputs=target_handler_outputs)
    )
    _then_preview_and_faces(
        TARGET_URL_ADD_BUTTON.click(add_url_path, inputs=[TARGET_URL_INPUT], outputs=target_handler_outputs)
    )
    _then_preview_and_faces(
        TARGET_FOLDER_ADD_BUTTON.click(add_folder, inputs=[TARGET_FOLDER_INPUT], outputs=target_handler_outputs)
    )
    _then_preview_and_faces(
        TARGET_REMOVE_BUTTON.click(remove_active_target, inputs=[ACTIVE_TARGET_DROPDOWN], outputs=target_handler_outputs)
    )

    def on_active_change(choice):
        from facefusion.uis.components import source_picker

        set_active(choice)
        updates = list(_refresh_outputs_with_map_dropdown())
        if source_picker.SOURCE_PICKER_GALLERY is not None:
            updates.append(source_picker.refresh_gallery_update())
        return tuple(updates)

    media_outputs = [TARGET_LIST, ACTIVE_TARGET_DROPDOWN, TARGET_FILES_ADD, TARGET_URL_INPUT]
    extra_outputs: List = []
    if MAP_ACTIVE_TARGET_DROPDOWN is not None:
        extra_outputs.append(MAP_ACTIVE_TARGET_DROPDOWN)
    from facefusion.uis.components import source_picker
    if source_picker.SOURCE_PICKER_GALLERY is not None:
        extra_outputs.append(source_picker.SOURCE_PICKER_GALLERY)

    _then_preview_and_faces(
        ACTIVE_TARGET_DROPDOWN.change(
            on_active_change,
            inputs=[ACTIVE_TARGET_DROPDOWN],
            outputs=media_outputs + extra_outputs,
        )
    )

    if MAP_ACTIVE_TARGET_DROPDOWN is not None:
        _then_preview_and_faces(
            MAP_ACTIVE_TARGET_DROPDOWN.change(
                on_active_change,
                inputs=[MAP_ACTIVE_TARGET_DROPDOWN],
                outputs=media_outputs + extra_outputs,
            )
        )

    if REMOVE_TARGET_ON_COMPLETE_CHECKBOX is not None:
        REMOVE_TARGET_ON_COMPLETE_CHECKBOX.change(
            update_remove_target_on_complete,
            inputs=[REMOVE_TARGET_ON_COMPLETE_CHECKBOX],
        )
