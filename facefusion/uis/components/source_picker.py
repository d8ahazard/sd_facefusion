"""Source-picker UI for the Map tab."""

import os
from typing import List, Optional, Tuple

import gradio
from gradio import SelectData

from facefusion.common_helper import get_first
from facefusion.filesystem import filter_image_paths
from facefusion.uis.components import face_selector
from facefusion.uis.components.source_slots_store import get_source_slots
from facefusion.uis.core import register_ui_component

SOURCE_PICKER_GALLERY: Optional[gradio.Gallery] = None
SOURCE_PICKER_ADD_BUTTON: Optional[gradio.Button] = None
SOURCE_PICKER_REMOVE_BUTTON: Optional[gradio.Button] = None

_picker_selected_index: int = -1


def _build_gallery_items() -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for slot in get_source_slots():
        thumb = get_first(filter_image_paths(slot.get('paths') or []))
        if not (thumb and os.path.isfile(thumb)):
            continue
        items.append((thumb, slot['name']))
    return items


def _ordered_slot_keys() -> List[int]:
    """Display indices for sources that have a visible thumbnail (same order as Media list)."""
    keys: List[int] = []
    for index, slot in enumerate(get_source_slots()):
        thumb = get_first(filter_image_paths(slot.get('paths') or []))
        if thumb and os.path.isfile(thumb):
            keys.append(index)
    return keys


def render() -> None:
    global SOURCE_PICKER_GALLERY, SOURCE_PICKER_ADD_BUTTON, SOURCE_PICKER_REMOVE_BUTTON

    with gradio.Column(elem_classes=['ff-src-picker-col']):
        SOURCE_PICKER_GALLERY = gradio.Gallery(
            value=_build_gallery_items(),
            show_label=False,
            columns=12,
            rows=1,
            height=90,
            object_fit='cover',
            allow_preview=False,
            elem_classes=['ff-map-gallery'],
            elem_id='ff_source_picker_gallery',
        )
        with gradio.Row(elem_classes=['ff-map-button-row']):
            SOURCE_PICKER_ADD_BUTTON = gradio.Button(
                'Add', size='sm', variant='primary', elem_classes=['ff-map-add-btn'],
            )
            SOURCE_PICKER_REMOVE_BUTTON = gradio.Button('Remove', size='sm', variant='stop')

    register_ui_component('source_picker_gallery', SOURCE_PICKER_GALLERY)
    register_ui_component('source_picker_add_button', SOURCE_PICKER_ADD_BUTTON)
    register_ui_component('source_picker_remove_button', SOURCE_PICKER_REMOVE_BUTTON)


def refresh_gallery_update() -> gradio.update:
    return gradio.update(value=_build_gallery_items())


def _on_select(event_data: SelectData) -> None:
    global _picker_selected_index
    if isinstance(event_data, SelectData):
        _picker_selected_index = event_data.index


def _on_add():
    keys = _ordered_slot_keys()
    if _picker_selected_index < 0 or _picker_selected_index >= len(keys):
        raise gradio.Error('Select a source thumbnail first.')
    slot_key = keys[_picker_selected_index]
    return face_selector.add_mapping_row_for_source(slot_key)


def _on_remove():
    return face_selector.remove_focused_mapping_row()


def listen() -> None:
    if SOURCE_PICKER_GALLERY is None:
        return
    SOURCE_PICKER_GALLERY.select(_on_select)

    refresh_outputs = face_selector.get_mapping_refresh_output_components()
    if SOURCE_PICKER_ADD_BUTTON is not None:
        SOURCE_PICKER_ADD_BUTTON.click(_on_add, outputs=refresh_outputs)
    if SOURCE_PICKER_REMOVE_BUTTON is not None:
        SOURCE_PICKER_REMOVE_BUTTON.click(_on_remove, outputs=refresh_outputs)
