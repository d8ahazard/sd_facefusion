import os
from typing import List, Optional, Tuple

import gradio

from facefusion import state_manager
from facefusion.common_helper import get_first
from facefusion.filesystem import filter_image_paths
from facefusion.user_data import (
    delete_source_preset,
    list_source_presets,
    load_source_preset,
    persist_source_files,
    remove_source_asset_dir,
    save_source_preset,
    sync_legacy_source_paths,
)
from facefusion.uis.components import source_slots_store as slots_store
from facefusion.uis.components.source_paths_helper import apply_source_files
from facefusion.uis.core import register_ui_component
from facefusion.uis.typing import File

MAX_SOURCE_SLOTS = slots_store.MAX_SOURCE_SLOTS

SOURCE_SLOT_FILES: List[Optional[gradio.File]] = [None] * MAX_SOURCE_SLOTS
SOURCE_SLOT_ROWS: List[Optional[gradio.Row]] = [None] * MAX_SOURCE_SLOTS
SOURCE_SLOT_NAMES: List[Optional[gradio.Textbox]] = [None] * MAX_SOURCE_SLOTS
SOURCE_SLOT_THUMBNAILS: List[Optional[gradio.Image]] = [None] * MAX_SOURCE_SLOTS
SOURCE_SLOT_SAVE_BUTTONS: List[Optional[gradio.Button]] = [None] * MAX_SOURCE_SLOTS
REMOVE_SLOT_BUTTONS: List[Optional[gradio.Button]] = [None] * MAX_SOURCE_SLOTS
ADD_SOURCE_BUTTON: Optional[gradio.Button] = None


def get_slot_thumbnail(paths: List[str]) -> Optional[str]:
    image_path = get_first(filter_image_paths(paths or []))
    if image_path and os.path.isfile(image_path):
        return image_path
    return None


def _slot_row_updates() -> List[gradio.update]:
    """Redraw every static row from the ordered source_slots array."""
    slots = slots_store.get_source_slots()
    count = len(slots)
    updates: List[gradio.update] = []
    for i in range(MAX_SOURCE_SLOTS):
        visible = i < count
        if visible:
            slot = slots[i]
            name = slot['name']
            paths = slot['paths']
            thumb = get_slot_thumbnail(paths)
        else:
            name = ''
            paths = []
            thumb = None
        updates.append(gradio.update(visible=visible))
        updates.append(gradio.update(value=name, visible=visible))
        updates.append(gradio.update(value=thumb, visible=visible))
        updates.append(gradio.update(value=paths if paths else None, visible=visible))
    return updates


def _all_slot_outputs() -> list:
    out = []
    for i in range(MAX_SOURCE_SLOTS):
        out.extend([
            SOURCE_SLOT_ROWS[i],
            SOURCE_SLOT_NAMES[i],
            SOURCE_SLOT_THUMBNAILS[i],
            SOURCE_SLOT_FILES[i],
        ])
    return out


def get_media_slot_output_components() -> list:
    return _all_slot_outputs()


def build_media_source_ui_updates() -> tuple:
    return tuple(_slot_row_updates())


def build_media_source_refresh_updates() -> tuple:
    """Return values aligned with get_media_source_refresh_outputs() (count must match)."""
    from facefusion.uis.components import face_selector, source_picker

    updates: List = list(_slot_row_updates())
    mapping_outputs = face_selector.get_mapping_refresh_output_components()
    mapping_updates = list(face_selector.build_mapping_refresh_updates())
    if len(mapping_updates) < len(mapping_outputs):
        mapping_updates.extend([gradio.update()] * (len(mapping_outputs) - len(mapping_updates)))
    elif len(mapping_updates) > len(mapping_outputs):
        mapping_updates = mapping_updates[:len(mapping_outputs)]
    updates.extend(mapping_updates)
    if source_picker.SOURCE_PICKER_GALLERY is not None:
        updates.append(source_picker.refresh_gallery_update())
    return tuple(updates)


def render() -> None:
    global ADD_SOURCE_BUTTON

    slots_store.load_slots_from_presets(list_source_presets, load_source_preset)
    slots = slots_store.get_source_slots()

    ADD_SOURCE_BUTTON = gradio.Button('Add Source', variant='primary')

    for i in range(MAX_SOURCE_SLOTS):
        slot = slots[i] if i < len(slots) else None
        visible = slot is not None
        slot_name = slot['name'] if slot else ''
        paths = slot['paths'] if slot else []
        thumb = get_slot_thumbnail(paths) if slot else None
        with gradio.Row(
            visible=visible,
            equal_height=False,
            elem_classes=['ff-src-row'],
        ) as row:
            SOURCE_SLOT_ROWS[i] = row
            with gradio.Column(scale=0, min_width=130, elem_classes=['ff-src-meta']):
                SOURCE_SLOT_NAMES[i] = gradio.Textbox(
                    value=slot_name,
                    show_label=False,
                    lines=1,
                    max_lines=1,
                    elem_classes=['ff-src-name'],
                )
                SOURCE_SLOT_THUMBNAILS[i] = gradio.Image(
                    value=thumb,
                    show_label=False,
                    show_download_button=False,
                    interactive=False,
                    elem_classes=['ff-src-thumb'],
                )
            with gradio.Column(scale=4, min_width=220, elem_classes=['ff-src-files-col']):
                SOURCE_SLOT_FILES[i] = gradio.File(
                    value=paths if paths else None,
                    show_label=False,
                    file_count='multiple',
                    file_types=['audio', 'image'],
                    elem_classes=['ff-src-files'],
                )
            with gradio.Column(scale=0, min_width=92, elem_classes=['ff-src-actions']):
                SOURCE_SLOT_SAVE_BUTTONS[i] = gradio.Button(
                    'Save', size='sm', variant='primary',
                )
                REMOVE_SLOT_BUTTONS[i] = gradio.Button(
                    'Remove slot', size='sm', variant='stop',
                )
            register_ui_component(f'media_source_file_{i}', SOURCE_SLOT_FILES[i])

    register_ui_component('media_add_source_button', ADD_SOURCE_BUTTON)


def get_media_source_refresh_outputs() -> list:
    """All Gradio outputs to refresh when Media sources change (includes Map tab)."""
    from facefusion.uis.components import face_selector, source_picker

    outputs = _all_slot_outputs()
    for component in face_selector.get_mapping_refresh_output_components():
        if component is not None:
            outputs.append(component)
    if source_picker.SOURCE_PICKER_GALLERY is not None:
        outputs.append(source_picker.SOURCE_PICKER_GALLERY)
    return outputs


def add_source_slot():
    slots_store.add_source_at_top()
    sync_legacy_source_paths()
    return build_media_source_refresh_updates()


def remove_source_slot(display_index: int) -> tuple:
    removed_name = slots_store.remove_source_at(display_index)
    if removed_name and removed_name in list_source_presets():
        delete_source_preset(removed_name)
    if removed_name and not slots_store.is_reserved_source_name(removed_name):
        remove_source_asset_dir(removed_name)
    sync_legacy_source_paths()
    return build_media_source_refresh_updates()


def _gr_file_path(file_obj) -> Optional[str]:
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        for key in ('path', 'name', 'orig_name'):
            value = file_obj.get(key)
            if isinstance(value, str) and value and os.path.isabs(value):
                return value
        return file_obj.get('path') or file_obj.get('name')
    for attr in ('path', 'name', 'orig_name'):
        value = getattr(file_obj, attr, None)
        if isinstance(value, str) and value and (os.path.isabs(value) or os.path.exists(value)):
            return value
    for attr in ('path', 'name', 'orig_name'):
        value = getattr(file_obj, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def save_source_slot(display_index: int, name: str, files: List[File]):
    name = (name or '').strip()
    if not name:
        raise gradio.Error('Enter a name before saving.')
    if slots_store.is_reserved_source_name(name):
        raise gradio.Error('Name cannot be "Source 1", "Source 2", or similar.')

    slot = slots_store.get_slot(display_index)
    if not slot:
        raise gradio.Error('Slot not found.')

    previous_name = (slot['name'] or '').strip()
    raw_paths = [p for p in (_gr_file_path(f) for f in (files or [])) if p]
    persistent_paths = persist_source_files(name, raw_paths) if raw_paths else list(slot['paths'])

    apply_source_files(persistent_paths, display_index)
    slots_store.update_slot_name(display_index, name)

    paths_for_preset = slots_store.get_slot(display_index)['paths']
    preset_data = slots_store.slots_for_preset_save(display_index)

    if previous_name and previous_name != name:
        if previous_name in list_source_presets():
            delete_source_preset(previous_name)
        if not slots_store.is_reserved_source_name(previous_name):
            remove_source_asset_dir(previous_name)

    ok = save_source_preset(name, preset_data)
    if not ok:
        raise gradio.Error('Failed to write favorite to disk. Check logs.')

    sync_legacy_source_paths()
    thumb = get_slot_thumbnail(paths_for_preset)
    return (
        gradio.update(value=thumb),
        gradio.update(value=paths_for_preset if paths_for_preset else None),
    )


def on_slot_files_changed(display_index: int, files: List[File]) -> tuple:
    slot = slots_store.get_slot(display_index)
    if slot is None:
        return build_media_source_refresh_updates()
    raw_paths = [p for p in (_gr_file_path(f) for f in (files or [])) if p]
    if set(raw_paths) == set(slot.get('paths') or []):
        return build_media_source_refresh_updates()
    apply_source_files(raw_paths, display_index)
    sync_legacy_source_paths()
    return build_media_source_refresh_updates()


def listen() -> None:
    refresh_outputs = get_media_source_refresh_outputs()
    ADD_SOURCE_BUTTON.click(add_source_slot, outputs=refresh_outputs)
    for i in range(MAX_SOURCE_SLOTS):
        if SOURCE_SLOT_FILES[i]:
            SOURCE_SLOT_FILES[i].change(
                fn=lambda files, idx=i: on_slot_files_changed(idx, files),
                inputs=[SOURCE_SLOT_FILES[i]],
                outputs=refresh_outputs,
            )
        if SOURCE_SLOT_SAVE_BUTTONS[i] and SOURCE_SLOT_NAMES[i] and SOURCE_SLOT_FILES[i]:
            SOURCE_SLOT_SAVE_BUTTONS[i].click(
                fn=lambda name, files, idx=i: save_source_slot(idx, name, files),
                inputs=[SOURCE_SLOT_NAMES[i], SOURCE_SLOT_FILES[i]],
                outputs=[SOURCE_SLOT_THUMBNAILS[i], SOURCE_SLOT_FILES[i]],
            )
        if REMOVE_SLOT_BUTTONS[i]:
            REMOVE_SLOT_BUTTONS[i].click(
                fn=lambda idx=i: remove_source_slot(idx),
                outputs=refresh_outputs,
            )
