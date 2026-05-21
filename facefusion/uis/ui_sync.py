"""Cross-tab UI refresh after state changes (job load, etc.)."""

from typing import List, Tuple

import gradio

from facefusion import state_manager


def get_workflow_refresh_output_components() -> List:
    from facefusion.uis.components import face_selector, media_sources, media_targets, preview, source_picker

    components = []
    if media_targets.ACTIVE_TARGET_DROPDOWN is not None:
        components.append(media_targets.ACTIVE_TARGET_DROPDOWN)
    if media_targets.MAP_ACTIVE_TARGET_DROPDOWN is not None:
        components.append(media_targets.MAP_ACTIVE_TARGET_DROPDOWN)
    if media_targets.TARGET_LIST is not None:
        components.append(media_targets.TARGET_LIST)
    if media_targets.TARGET_FILES_ADD is not None:
        components.append(media_targets.TARGET_FILES_ADD)
    if media_targets.TARGET_URL_INPUT is not None:
        components.append(media_targets.TARGET_URL_INPUT)
    components.extend(media_sources.get_media_slot_output_components())
    for component in preview.get_target_preview_output_components():
        if component is not None and component not in components:
            components.append(component)
    for component in preview.get_map_face_sync_output_components():
        if component is not None and component not in components:
            components.append(component)
    for component in face_selector.get_mapping_refresh_output_components():
        if component is not None and component not in components:
            components.append(component)
    if source_picker.SOURCE_PICKER_GALLERY is not None and source_picker.SOURCE_PICKER_GALLERY not in components:
        components.append(source_picker.SOURCE_PICKER_GALLERY)
    return components


def reload_ui_after_job() -> Tuple:
    """Push state_manager back into all settings + workflow controls after a job finishes."""
    from facefusion.uis.core import reload_all_settings

    return reload_all_settings() + build_workflow_ui_updates()


def get_post_job_reload_outputs() -> List:
    from facefusion.uis.core import get_reload_outputs

    seen = set()
    outputs = []
    for component in get_reload_outputs() + get_workflow_refresh_output_components():
        if component is not None and id(component) not in seen:
            seen.add(id(component))
            outputs.append(component)
    return outputs


def build_workflow_ui_updates() -> Tuple:
    from facefusion.uis.components import media_sources, media_targets, preview

    updates: List = []
    updates.extend(media_targets.build_target_ui_updates())
    updates.extend(media_sources.build_media_source_refresh_updates())
    updates.extend(preview.build_target_preview_updates())
    updates.extend(preview.refresh_map_face_sync())
    return tuple(updates)


def apply_job_step_to_state(step_args: dict) -> None:
    from facefusion.args import apply_args
    from facefusion.face_store import clear_reference_faces, clear_static_faces
    from facefusion.uis.components.face_selector import clear_selected_faces, restore_selected_faces_from_state
    from facefusion.user_data import sync_active_target_path, sync_legacy_source_paths

    apply_args(step_args, True)

    tp = step_args.get('target_path')
    if tp:
        paths = list(state_manager.get_item('target_paths') or [])
        if tp not in paths:
            paths.append(tp)
        state_manager.set_item('target_paths', paths)
        state_manager.set_item('active_target_index', paths.index(tp))

    if step_args.get('target_paths'):
        state_manager.set_item('target_paths', list(step_args['target_paths']))
        if step_args.get('active_target_index') is not None:
            state_manager.set_item('active_target_index', step_args['active_target_index'])

    if step_args.get('mapping_slot_keys') is not None:
        state_manager.set_item('mapping_slot_keys', list(step_args['mapping_slot_keys']))

    sync_active_target_path()
    sync_legacy_source_paths()
    clear_reference_faces()
    clear_static_faces()
    if step_args.get('reference_face_dict') is not None:
        state_manager.set_item('reference_face_dict', step_args['reference_face_dict'])
        restore_selected_faces_from_state()
    else:
        clear_selected_faces()
