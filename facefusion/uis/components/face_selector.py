from typing import Dict, List, Optional, Tuple

import gradio
import numpy
from gradio import SelectData

import facefusion.choices
from facefusion import wording, state_manager
from facefusion.face_analyser import get_many_faces
from facefusion.face_selector import sort_and_filter_faces, current_sort_values
from facefusion.face_store import clear_reference_faces, clear_static_faces
from facefusion.filesystem import is_image, is_video
from facefusion.processors.core import get_processors_modules
from facefusion.typing import FaceSelectorMode, VisionFrame, Race, Gender, FaceSelectorOrder, FaceReference
from facefusion.uis.core import get_ui_component, register_ui_component, get_ui_components
from facefusion.uis.typing import Component, ComponentOptions
from facefusion.uis.ui_helper import convert_str_none
from facefusion.vision import get_video_frame, normalize_frame_color, read_static_image, detect_video_fps, \
    count_video_frame_total

# Capacity of mapping rows we statically render on the Map tab.
MAX_SOURCE_SLOTS = 10

FACE_SELECTOR_MODE_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_SELECTOR_ORDER_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_SELECTOR_GENDER_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_SELECTOR_RACE_DROPDOWN: Optional[gradio.Dropdown] = None
FACE_SELECTOR_AGE_RANGE_START_SLIDER: Optional[gradio.Slider] = None
FACE_SELECTOR_AGE_RANGE_END_SLIDER: Optional[gradio.Slider] = None
REFERENCE_FACE_POSITION_GALLERY: Optional[gradio.Gallery] = None
REFERENCE_FACE_DISTANCE_SLIDER: Optional[gradio.Slider] = None
FACE_SELECTOR_GROUP: Optional[gradio.Group] = None

# Per mapping row (display index 0..MAX_SOURCE_SLOTS-1).
SLOT_SELECTION_ROWS: List[Optional[gradio.Row]] = [None] * MAX_SOURCE_SLOTS
SLOT_HEADERS: List[Optional[gradio.Markdown]] = [None] * MAX_SOURCE_SLOTS
SLOT_ADD_BUTTONS: List[Optional[gradio.Button]] = [None] * MAX_SOURCE_SLOTS
SLOT_REMOVE_BUTTONS: List[Optional[gradio.Button]] = [None] * MAX_SOURCE_SLOTS
SLOT_SELECTION_GALLERIES: List[Optional[gradio.Gallery]] = [None] * MAX_SOURCE_SLOTS

# Legacy aliases for code that still references slot 0/1 directly.
REFERENCE_FACES_SELECTION_GALLERY: Optional[gradio.Gallery] = None
REFERENCE_FACES_SELECTION_GALLERY_2: Optional[gradio.Gallery] = None
ADD_REFERENCE_FACE_BUTTON: Optional[gradio.Button] = None
ADD_REFERENCE_FACE_BUTTON_2: Optional[gradio.Button] = None
REMOVE_REFERENCE_FACE_BUTTON: Optional[gradio.Button] = None
REMOVE_REFERENCE_FACE_BUTTON_2: Optional[gradio.Button] = None
REFERENCE_FACE_POSITION_GALLERY_2: Optional[gradio.Gallery] = None  # unused; kept for import compat

# Detected faces cache (target preview frame).
current_reference_faces: list = []
current_reference_frames: list = []

# Per (source slot key) selected-face entries displayed in that row's gallery.
# entry: {'face_data': Face, 'original_face_index': int, 'frame_number': int}
current_selected_faces_by_slot: Dict[int, list] = {}

# Per-row selection inside its mapped-faces gallery.
selected_face_index_by_slot: Dict[int, int] = {}

# Selected detected-face index in the top "detected faces" gallery.
selector_face_index: int = -1

# Which mapping row is "focused" (last interacted with). Used by source picker's Remove.
focused_display_index: int = -1


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def get_mapping_slot_keys() -> List[int]:
    """Source slot indices that currently have a mapping row, in display order."""
    return [int(k) for k in (state_manager.get_item('mapping_slot_keys') or [])]


def set_mapping_slot_keys(keys: List[int]) -> None:
    state_manager.set_item('mapping_slot_keys', [int(k) for k in keys])


def _source_labels() -> Dict[int, str]:
    from facefusion.uis.components.source_slots_store import get_source_slots

    return {i: slot['name'] for i, slot in enumerate(get_source_slots())}


def _slot_display_name(slot_key: int) -> str:
    labels = _source_labels()
    return labels.get(int(slot_key)) or f'Source {int(slot_key) + 1}'


def _selected_faces_for(slot_key: int) -> list:
    return current_selected_faces_by_slot.setdefault(int(slot_key), [])


def _selected_index_for(slot_key: int) -> int:
    return selected_face_index_by_slot.get(int(slot_key), -1)


def _max_mapping_rows() -> int:
    mode = state_manager.get_item('face_selector_mode')
    if mode == 'one':
        return 1
    return MAX_SOURCE_SLOTS


def restore_selected_faces_from_state() -> None:
    """Rebuild module-level selected-face caches from reference_face_dict + current frame.

    Called after a per-target snapshot/restore so the right-column galleries reflect
    the new target's mappings. We can't re-extract face *images* without re-running
    detection on each historical frame, so we leave gallery items empty and rely on
    `extract_gallery_frames` for the active frame plus the FaceReference entries.
    """
    global current_selected_faces_by_slot, selected_face_index_by_slot
    current_selected_faces_by_slot = {}
    selected_face_index_by_slot = {}
    ref_dict = state_manager.get_item('reference_face_dict') or {}
    for slot_key, refs in ref_dict.items():
        slot_key = int(slot_key)
        entries = []
        for ref in refs or []:
            entries.append({
                'face_data': None,
                'original_face_index': ref['face_index'],
                'frame_number': ref['frame_number'],
            })
        current_selected_faces_by_slot[slot_key] = entries


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def resolve_reference_frame_number(target_path: str) -> int:
    """Clamp reference_frame_number to the active target; default to 0."""
    frame_number = state_manager.get_item('reference_frame_number')
    if frame_number is None:
        frame_number = 0
    else:
        frame_number = int(frame_number)
    if is_video(target_path):
        total = count_video_frame_total(target_path)
        if total > 0:
            frame_number = max(0, min(frame_number, total - 1))
    return frame_number


def resolve_preview_frame_number(target_path: str) -> int:
    """Use current reference_frame_number for preview/gallery (no frame scanning)."""
    return resolve_reference_frame_number(target_path)


def render() -> None:
    global FACE_SELECTOR_MODE_DROPDOWN, REFERENCE_FACE_POSITION_GALLERY, FACE_SELECTOR_GROUP
    global REFERENCE_FACES_SELECTION_GALLERY, REFERENCE_FACES_SELECTION_GALLERY_2
    global ADD_REFERENCE_FACE_BUTTON, ADD_REFERENCE_FACE_BUTTON_2
    global REMOVE_REFERENCE_FACE_BUTTON, REMOVE_REFERENCE_FACE_BUTTON_2

    reference_face_gallery_options: ComponentOptions = {
        'label': wording.get('uis.reference_face_gallery'),
        'object_fit': 'cover',
        'columns': 8,
        'allow_preview': False,
        'visible': True,
    }
    target_path = state_manager.get_item('target_path')
    if is_image(target_path):
        reference_frame = read_static_image(target_path)
        if reference_frame is not None:
            reference_face_gallery_options['value'] = extract_gallery_frames(reference_frame)
    elif is_video(target_path):
        frame_number = state_manager.get_item('reference_frame_number')
        if frame_number is None:
            frame_number = 0
        reference_frame = get_video_frame(target_path, int(frame_number))
        if reference_frame is not None:
            reference_face_gallery_options['value'] = extract_gallery_frames(reference_frame)
    if 'value' not in reference_face_gallery_options:
        reference_face_gallery_options['value'] = []

    non_face_processors = ['frame_colorizer', 'frame_enhancer', 'style_transfer']
    show_group = False
    for processor in (state_manager.get_item('processors') or []):
        if processor not in non_face_processors:
            show_group = True
            break

    mapping_keys = get_mapping_slot_keys()

    with gradio.Group(visible=show_group) as FACE_SELECTOR_GROUP:
        FACE_SELECTOR_MODE_DROPDOWN = gradio.Dropdown(
            label=wording.get('uis.face_selector_mode_dropdown'),
            choices=facefusion.choices.face_selector_modes,
            value=state_manager.get_item('face_selector_mode'),
        )
        gradio.Markdown('Detected faces on active target — select a face, then use **Add** on a source row below.')
        REFERENCE_FACE_POSITION_GALLERY = gradio.Gallery(
            **reference_face_gallery_options,
            show_label=False,
            rows=1,
            height=90,
            elem_classes=['ff-map-gallery'],
            elem_id='ff_reference_face_position_gallery',
        )

        for display_index in range(MAX_SOURCE_SLOTS):
            slot_key = mapping_keys[display_index] if display_index < len(mapping_keys) else None
            row_visible = slot_key is not None
            slot_label = _slot_display_name(slot_key) if slot_key is not None else ''
            with gradio.Row(visible=row_visible, elem_classes=['ff-map-mapping-row']) as slot_row:
                SLOT_SELECTION_ROWS[display_index] = slot_row
                with gradio.Column(scale=1, min_width=0, elem_classes=['ff-map-gallery-col']):
                    SLOT_HEADERS[display_index] = gradio.Markdown(
                        value=f'**{slot_label}**' if slot_label else '',
                        elem_classes=['ff-map-slot-name'],
                    )
                    SLOT_SELECTION_GALLERIES[display_index] = gradio.Gallery(
                        label=None,
                        show_label=False,
                        object_fit='cover',
                        columns=12,
                        rows=1,
                        height=90,
                        allow_preview=False,
                        visible=row_visible,
                        elem_classes=['ff-map-gallery'],
                        elem_id=f'ff_reference_faces_selection_gallery_{display_index}',
                    )
                    with gradio.Row(elem_classes=['ff-map-button-row']):
                        SLOT_ADD_BUTTONS[display_index] = gradio.Button(
                            'Add', size='sm', variant='primary', elem_classes=['ff-map-add-btn'],
                        )
                        SLOT_REMOVE_BUTTONS[display_index] = gradio.Button('Remove', size='sm', variant='stop')

    REFERENCE_FACES_SELECTION_GALLERY = SLOT_SELECTION_GALLERIES[0]
    REFERENCE_FACES_SELECTION_GALLERY_2 = SLOT_SELECTION_GALLERIES[1] if MAX_SOURCE_SLOTS > 1 else None
    ADD_REFERENCE_FACE_BUTTON = SLOT_ADD_BUTTONS[0]
    ADD_REFERENCE_FACE_BUTTON_2 = SLOT_ADD_BUTTONS[1] if MAX_SOURCE_SLOTS > 1 else None
    REMOVE_REFERENCE_FACE_BUTTON = SLOT_REMOVE_BUTTONS[0]
    REMOVE_REFERENCE_FACE_BUTTON_2 = SLOT_REMOVE_BUTTONS[1] if MAX_SOURCE_SLOTS > 1 else None

    register_ui_component('face_selector_mode_dropdown', FACE_SELECTOR_MODE_DROPDOWN)
    register_ui_component('reference_face_position_gallery', REFERENCE_FACE_POSITION_GALLERY)
    register_ui_component('reference_faces_selection_gallery', REFERENCE_FACES_SELECTION_GALLERY)
    register_ui_component('reference_faces_selection_gallery_2', REFERENCE_FACES_SELECTION_GALLERY_2)
    register_ui_component('add_reference_face_button', ADD_REFERENCE_FACE_BUTTON)
    register_ui_component('remove_reference_faces_button', REMOVE_REFERENCE_FACE_BUTTON)
    register_ui_component('add_reference_face_button_2', ADD_REFERENCE_FACE_BUTTON_2)
    register_ui_component('remove_reference_faces_button_2', REMOVE_REFERENCE_FACE_BUTTON_2)
    register_ui_component('face_selector_group', FACE_SELECTOR_GROUP)


# ---------------------------------------------------------------------------
# Output-component lists
# ---------------------------------------------------------------------------

def get_detected_faces_gallery_component() -> Optional[Component]:
    return REFERENCE_FACE_POSITION_GALLERY


def get_mapping_gallery_outputs() -> List[Component]:
    """Slot mapping galleries only (detected faces use get_detected_faces_gallery_component)."""
    outputs: List[Component] = []
    for gallery in SLOT_SELECTION_GALLERIES:
        if gallery is not None:
            outputs.append(gallery)
    return outputs


def get_slot_row_output_components() -> List[Component]:
    return [row for row in SLOT_SELECTION_ROWS if row is not None]


def get_slot_header_components() -> List[Component]:
    return [hdr for hdr in SLOT_HEADERS if hdr is not None]


def get_mapping_refresh_output_components() -> List[Component]:
    """Mapping rows/headers/slot galleries (not the detected-faces gallery)."""
    outputs: List[Component] = []
    outputs.extend(get_slot_row_output_components())
    outputs.extend(get_slot_header_components())
    outputs.extend(get_mapping_gallery_outputs())
    return outputs


def get_mode_change_output_components() -> List[Component]:
    outputs: List[Component] = []
    settings_dropdown = get_ui_component('face_selector_mode_settings_dropdown')
    if settings_dropdown is not None:
        outputs.append(settings_dropdown)
    if FACE_SELECTOR_MODE_DROPDOWN is not None and FACE_SELECTOR_MODE_DROPDOWN not in outputs:
        outputs.append(FACE_SELECTOR_MODE_DROPDOWN)
    outputs.extend(get_mapping_refresh_output_components())
    return outputs


# ---------------------------------------------------------------------------
# UI update builders
# ---------------------------------------------------------------------------

def _selected_faces_payload_for(slot_key: int) -> Optional[list]:
    entries = current_selected_faces_by_slot.get(int(slot_key)) or []
    payload = []
    for entry in entries:
        face_data = entry.get('face_data')
        if face_data is None:
            continue
        # face_data here is a numpy crop; reuse extract_gallery_frames format.
        payload.append(face_data)
    return payload or None


def build_slot_row_updates() -> List[gradio.update]:
    keys = get_mapping_slot_keys()
    updates: List[gradio.update] = []
    for display_index in range(MAX_SOURCE_SLOTS):
        visible = display_index < len(keys)
        if SLOT_SELECTION_ROWS[display_index] is None:
            continue
        updates.append(gradio.update(visible=visible))
    return updates


def build_slot_header_updates() -> List[gradio.update]:
    keys = get_mapping_slot_keys()
    updates: List[gradio.update] = []
    for display_index in range(MAX_SOURCE_SLOTS):
        if SLOT_HEADERS[display_index] is None:
            continue
        if display_index < len(keys):
            updates.append(gradio.update(value=f'**{_slot_display_name(keys[display_index])}**'))
        else:
            updates.append(gradio.update(value=''))
    return updates


def build_slot_gallery_updates() -> List[gradio.update]:
    keys = get_mapping_slot_keys()
    updates: List[gradio.update] = []
    for display_index in range(MAX_SOURCE_SLOTS):
        if SLOT_SELECTION_GALLERIES[display_index] is None:
            continue
        if display_index < len(keys):
            slot_key = keys[display_index]
            payload = _selected_faces_payload_for(slot_key)
            updates.append(gradio.update(value=payload, visible=True))
        else:
            updates.append(gradio.update(value=None, visible=False))
    return updates


def build_mapping_refresh_updates() -> Tuple:
    """One gradio.update per component in get_mapping_refresh_output_components()."""
    updates: List[gradio.update] = []
    row_iter = iter(build_slot_row_updates())
    for row in SLOT_SELECTION_ROWS:
        if row is not None:
            updates.append(next(row_iter, gradio.update()))
    header_iter = iter(build_slot_header_updates())
    for header in SLOT_HEADERS:
        if header is not None:
            updates.append(next(header_iter, gradio.update()))
    gallery_iter = iter(build_slot_gallery_updates())
    for gallery in SLOT_SELECTION_GALLERIES:
        if gallery is not None:
            updates.append(next(gallery_iter, gradio.update()))
    return tuple(updates)


def build_mode_and_mapping_updates() -> List[gradio.update]:
    mode = state_manager.get_item('face_selector_mode')
    updates: List[gradio.update] = []
    settings_dropdown = get_ui_component('face_selector_mode_settings_dropdown')
    if settings_dropdown is not None:
        updates.append(gradio.update(value=mode))
    if FACE_SELECTOR_MODE_DROPDOWN is not None:
        updates.append(gradio.update(value=mode))
    updates.extend(list(build_mapping_refresh_updates()))
    return updates


def apply_face_selector_mode(mode: FaceSelectorMode) -> Tuple:
    # In 'one' mode trim mapping rows to a single entry.
    keys = get_mapping_slot_keys()
    if mode == 'one' and len(keys) > 1:
        set_mapping_slot_keys(keys[:1])
    state_manager.set_item('face_selector_mode', mode)
    return tuple(build_mode_and_mapping_updates())


# ---------------------------------------------------------------------------
# Source-picker public API
# ---------------------------------------------------------------------------

def add_mapping_row_for_source(slot_key: int) -> Tuple:
    """Append a mapping row for the given source slot index."""
    global focused_display_index
    keys = get_mapping_slot_keys()
    if slot_key is None or int(slot_key) in keys:
        raise gradio.Error('That source already has a mapping row.')
    if len(keys) >= _max_mapping_rows():
        raise gradio.Error("Mode 'one' allows a single mapping row." if state_manager.get_item('face_selector_mode') == 'one'
                           else 'Mapping row limit reached.')
    keys.append(int(slot_key))
    set_mapping_slot_keys(keys)
    focused_display_index = len(keys) - 1
    return build_mapping_refresh_updates()


def remove_focused_mapping_row() -> Tuple:
    """Remove the mapping row whose face/buttons were last interacted with."""
    global focused_display_index
    keys = get_mapping_slot_keys()
    target_index = focused_display_index
    if target_index < 0 or target_index >= len(keys):
        if not keys:
            raise gradio.Error('No mapping rows to remove.')
        target_index = len(keys) - 1
    slot_key = keys.pop(target_index)
    set_mapping_slot_keys(keys)
    ref_dict = dict(state_manager.get_item('reference_face_dict') or {})
    ref_dict.pop(slot_key, None)
    state_manager.set_item('reference_face_dict', ref_dict)
    current_selected_faces_by_slot.pop(slot_key, None)
    selected_face_index_by_slot.pop(slot_key, None)
    focused_display_index = max(0, target_index - 1) if keys else -1
    return build_mapping_refresh_updates()


def remap_mapping_after_source_removed(removed_index: int) -> None:
    """Drop mappings for removed slot and decrement indices above it."""
    global current_selected_faces_by_slot, selected_face_index_by_slot

    ref_dict = dict(state_manager.get_item('reference_face_dict') or {})
    new_ref: Dict[int, list] = {}
    for key, refs in ref_dict.items():
        key = int(key)
        if key == removed_index:
            continue
        new_ref[key - 1 if key > removed_index else key] = refs
    state_manager.set_item('reference_face_dict', new_ref)

    new_mapping_keys: List[int] = []
    for key in get_mapping_slot_keys():
        key = int(key)
        if key == removed_index:
            continue
        new_mapping_keys.append(key - 1 if key > removed_index else key)
    set_mapping_slot_keys(new_mapping_keys)

    new_selected: Dict[int, list] = {}
    for key, entries in current_selected_faces_by_slot.items():
        key = int(key)
        if key == removed_index:
            continue
        new_selected[key - 1 if key > removed_index else key] = entries
    current_selected_faces_by_slot = new_selected

    new_sel_idx: Dict[int, int] = {}
    for key, idx in selected_face_index_by_slot.items():
        key = int(key)
        if key == removed_index:
            continue
        new_sel_idx[key - 1 if key > removed_index else key] = idx
    selected_face_index_by_slot = new_sel_idx


def remap_source_slot_indices(key_map: Dict[int, int]) -> None:
    """Reindex mapping rows and reference_face_dict when Media source slots shift."""
    global current_selected_faces_by_slot, selected_face_index_by_slot

    new_mapping_keys: List[int] = []
    for key in get_mapping_slot_keys():
        if int(key) in key_map:
            new_mapping_keys.append(int(key_map[int(key)]))
    set_mapping_slot_keys(new_mapping_keys)

    ref_dict = dict(state_manager.get_item('reference_face_dict') or {})
    new_ref_dict: Dict[int, list] = {}
    for old_key, refs in ref_dict.items():
        old_key = int(old_key)
        if old_key in key_map:
            new_ref_dict[int(key_map[old_key])] = refs
    state_manager.set_item('reference_face_dict', new_ref_dict)

    new_selected: Dict[int, list] = {}
    for old_key, entries in current_selected_faces_by_slot.items():
        old_key = int(old_key)
        if old_key in key_map:
            new_selected[int(key_map[old_key])] = entries
    current_selected_faces_by_slot = new_selected

    new_sel_idx: Dict[int, int] = {}
    for old_key, idx in selected_face_index_by_slot.items():
        old_key = int(old_key)
        if old_key in key_map:
            new_sel_idx[int(key_map[old_key])] = idx
    selected_face_index_by_slot = new_sel_idx


def shift_source_slot_indices_up_from(insert_index: int = 0) -> None:
    """After inserting a new empty source at `insert_index`, bump existing indices."""
    sfd = state_manager.get_item('source_frame_dict') or {}
    key_map = {int(k): int(k) + 1 for k in sfd.keys() if int(k) >= insert_index}
    if key_map:
        remap_source_slot_indices(key_map)


def prune_mapping_keys_for_removed_source(slot_key: int) -> None:
    """Drop mapping data for a source slot index that no longer exists."""
    keys = [k for k in get_mapping_slot_keys() if k != int(slot_key)]
    set_mapping_slot_keys(keys)
    ref_dict = dict(state_manager.get_item('reference_face_dict') or {})
    ref_dict.pop(int(slot_key), None)
    state_manager.set_item('reference_face_dict', ref_dict)
    current_selected_faces_by_slot.pop(int(slot_key), None)
    selected_face_index_by_slot.pop(int(slot_key), None)


# ---------------------------------------------------------------------------
# Per-row Add/Remove logic
# ---------------------------------------------------------------------------

def _slot_key_for_display(display_index: int) -> Optional[int]:
    keys = get_mapping_slot_keys()
    if 0 <= display_index < len(keys):
        return keys[display_index]
    return None


def _on_row_add(display_index: int):
    """Assign the currently-selected detected face to this row. Exclusive across rows."""
    global focused_display_index, selector_face_index
    slot_key = _slot_key_for_display(display_index)
    if slot_key is None:
        return _row_outputs_noop()

    if selector_face_index < 0 or selector_face_index >= len(current_reference_faces):
        raise gradio.Error('Select a detected face first.')

    face_index = selector_face_index
    frame_number = state_manager.get_item('reference_frame_number') or 0

    # Remove this (frame, face_index) from any other row first (exclusive assignment).
    ref_dict = dict(state_manager.get_item('reference_face_dict') or {})
    for other_key in list(ref_dict.keys()):
        if int(other_key) == int(slot_key):
            continue
        kept = [r for r in (ref_dict.get(other_key) or [])
                if not (r['frame_number'] == frame_number and r['face_index'] == face_index)]
        ref_dict[int(other_key)] = kept
        bucket = current_selected_faces_by_slot.get(int(other_key)) or []
        current_selected_faces_by_slot[int(other_key)] = [
            e for e in bucket
            if not (e.get('frame_number') == frame_number and e.get('original_face_index') == face_index)
        ]

    own_refs = list(ref_dict.get(int(slot_key)) or [])
    already = any(r['frame_number'] == frame_number and r['face_index'] == face_index for r in own_refs)
    if not already:
        own_refs.append(FaceReference(
            frame_number=frame_number,
            face_index=face_index,
            sorts=current_sort_values(),
        ))
        ref_dict[int(slot_key)] = own_refs
        bucket = current_selected_faces_by_slot.setdefault(int(slot_key), [])
        face_crop = current_reference_frames[face_index] if face_index < len(current_reference_frames) else None
        bucket.append({
            'face_data': face_crop,
            'original_face_index': face_index,
            'frame_number': frame_number,
        })

    state_manager.set_item('reference_face_dict', ref_dict)
    focused_display_index = display_index

    from facefusion.uis.components.preview import update_preview_image
    preview, enable_button, disable_button = update_preview_image(frame_number)
    return _row_outputs(preview, enable_button, disable_button)


def _on_row_remove(display_index: int):
    """Remove a face from this row.

    Priority:
      1. If this row's gallery has a selection, drop that.
      2. Else if the top detected gallery selection points to a face in this row, drop it.
      3. Else no-op.
    """
    global focused_display_index
    slot_key = _slot_key_for_display(display_index)
    if slot_key is None:
        return _row_outputs_noop()

    frame_number = state_manager.get_item('reference_frame_number') or 0
    ref_dict = dict(state_manager.get_item('reference_face_dict') or {})
    own_refs = list(ref_dict.get(int(slot_key)) or [])
    bucket = current_selected_faces_by_slot.setdefault(int(slot_key), [])

    row_sel = selected_face_index_by_slot.get(int(slot_key), -1)
    target_face_index: Optional[int] = None
    target_frame_number: Optional[int] = None

    if 0 <= row_sel < len(bucket):
        target_face_index = bucket[row_sel].get('original_face_index')
        target_frame_number = bucket[row_sel].get('frame_number', frame_number)
    elif selector_face_index >= 0:
        target_face_index = selector_face_index
        target_frame_number = frame_number

    if target_face_index is None:
        return _row_outputs_noop()

    own_refs = [r for r in own_refs
                if not (r['frame_number'] == target_frame_number and r['face_index'] == target_face_index)]
    ref_dict[int(slot_key)] = own_refs
    state_manager.set_item('reference_face_dict', ref_dict)

    current_selected_faces_by_slot[int(slot_key)] = [
        e for e in bucket
        if not (e.get('frame_number') == target_frame_number and e.get('original_face_index') == target_face_index)
    ]
    selected_face_index_by_slot[int(slot_key)] = -1
    focused_display_index = display_index

    from facefusion.uis.components.preview import update_preview_image
    preview, enable_button, disable_button = update_preview_image(frame_number)
    return _row_outputs(preview, enable_button, disable_button)


def _row_outputs(preview, enable_button, disable_button):
    """Build the outputs tuple matching listen() row wiring (all slot galleries + preview)."""
    gallery_updates = build_slot_gallery_updates()
    return tuple(gallery_updates) + (preview, enable_button, disable_button)


def _row_outputs_noop():
    return _row_outputs(gradio.update(), gradio.update(), gradio.update())


def _row_output_components() -> List[Component]:
    outs: List[Component] = []
    for g in SLOT_SELECTION_GALLERIES:
        if g is not None:
            outs.append(g)
    preview_image = get_ui_component('preview_image')
    mask_enable = get_ui_component('mask_enable_button')
    mask_disable = get_ui_component('mask_disable_button')
    for comp in (preview_image, mask_enable, mask_disable):
        if comp is not None:
            outs.append(comp)
    return outs


# ---------------------------------------------------------------------------
# Listen
# ---------------------------------------------------------------------------

def listen() -> None:
    mapping_outputs = get_mapping_gallery_outputs()
    mode_outputs = get_mode_change_output_components()

    if FACE_SELECTOR_MODE_DROPDOWN and mode_outputs:
        FACE_SELECTOR_MODE_DROPDOWN.change(
            apply_face_selector_mode,
            inputs=FACE_SELECTOR_MODE_DROPDOWN,
            outputs=mode_outputs,
        )

    if REFERENCE_FACE_POSITION_GALLERY is not None:
        REFERENCE_FACE_POSITION_GALLERY.select(update_selector_face_index)

    row_outs = _row_output_components()

    for display_index in range(MAX_SOURCE_SLOTS):
        add_btn = SLOT_ADD_BUTTONS[display_index]
        rem_btn = SLOT_REMOVE_BUTTONS[display_index]
        gallery = SLOT_SELECTION_GALLERIES[display_index]
        if not add_btn or not gallery:
            continue

        def _add_fn(idx=display_index):
            return lambda: _on_row_add(idx)

        def _remove_fn(idx=display_index):
            return lambda: _on_row_remove(idx)

        add_btn.click(_add_fn(display_index), outputs=row_outs)
        if rem_btn:
            rem_btn.click(_remove_fn(display_index), outputs=row_outs)

        def _select_fn(idx=display_index):
            def on_select(event_data: SelectData):
                global selector_face_index, focused_display_index
                keys = get_mapping_slot_keys()
                if idx < len(keys):
                    slot_key = int(keys[idx])
                    if isinstance(event_data, SelectData):
                        selected_face_index_by_slot[slot_key] = event_data.index
                        selector_face_index = -1
                        focused_display_index = idx
            return on_select

        gallery.select(_select_fn(display_index))

    preview_frame_slider_component = get_ui_component('preview_frame_slider')
    detected_faces_gallery_outputs: List[Component] = []
    if REFERENCE_FACE_POSITION_GALLERY is not None:
        detected_faces_gallery_outputs.append(REFERENCE_FACE_POSITION_GALLERY)
    mapping_refresh_outputs = get_mapping_refresh_output_components()
    preview_image_outputs = [
        c for c in (
            get_ui_component('preview_image'),
            get_ui_component('mask_enable_button'),
            get_ui_component('mask_disable_button'),
        ) if c is not None
    ]

    def _chain_preview_image(event):
        if preview_frame_slider_component and preview_image_outputs:
            from facefusion.uis.components.preview import update_preview_image
            return event.then(
                update_preview_image,
                inputs=preview_frame_slider_component,
                outputs=preview_image_outputs,
                show_progress='hidden',
            )
        return event

    if preview_frame_slider_component and detected_faces_gallery_outputs:
        _chain_preview_image(
            preview_frame_slider_component.release(
                update_reference_frame_number,
                inputs=preview_frame_slider_component,
                outputs=detected_faces_gallery_outputs,
            )
        )

    preview_slider = preview_frame_slider_component
    frame_nav_outputs = ([preview_slider] if preview_slider else []) + detected_faces_gallery_outputs
    for button_name in (
        'preview_frame_back_button',
        'preview_frame_forward_button',
        'preview_frame_back_five_button',
        'preview_frame_forward_five_button',
    ):
        button = get_ui_component(button_name)
        if button and frame_nav_outputs:
            handler = {
                'preview_frame_back_button': reference_frame_back,
                'preview_frame_forward_button': reference_frame_forward,
                'preview_frame_back_five_button': reference_frame_back_five,
                'preview_frame_forward_five_button': reference_frame_forward_five,
            }[button_name]
            _chain_preview_image(button.click(handler, inputs=preview_slider, outputs=frame_nav_outputs))

    for ui_component in get_ui_components(['target_file', 'target_image', 'target_video']):
        if ui_component is None:
            continue
        for method in ['upload', 'change', 'clear']:
            if not hasattr(ui_component, method):
                continue
            getattr(ui_component, method)(
                update_reference_face_position,
                outputs=[preview_slider] if preview_slider else [],
            )
            if detected_faces_gallery_outputs:
                getattr(ui_component, method)(
                    clear_and_update_reference_position_gallery,
                    outputs=detected_faces_gallery_outputs,
                )

    for ui_component in get_ui_components([
        'face_detector_model_dropdown',
        'face_detector_size_dropdown',
        'face_detector_angles_checkbox_group',
    ]):
        if ui_component and detected_faces_gallery_outputs:
            ui_component.change(
                clear_and_update_reference_position_gallery,
                outputs=detected_faces_gallery_outputs,
            )

    face_detector_score_slider = get_ui_component('face_detector_score_slider')
    if face_detector_score_slider and detected_faces_gallery_outputs:
        face_detector_score_slider.release(
            clear_and_update_reference_position_gallery,
            outputs=detected_faces_gallery_outputs,
        )

    processors_checkbox_group = get_ui_component('processors_checkbox_group')
    if processors_checkbox_group:
        processors_checkbox_group.change(
            toggle_group,
            inputs=processors_checkbox_group,
            outputs=[FACE_SELECTOR_GROUP],
        )


def toggle_group(processors: List[str]) -> gradio.update:
    all_processors = get_processors_modules()
    all_face_processor_names = [p.display_name for p in all_processors if p.is_face_processor]
    for p in (processors or []):
        if p in all_face_processor_names:
            return gradio.update(visible=True)
    return gradio.update(visible=False)


def update_face_selector_mode(face_selector_mode: FaceSelectorMode):
    return apply_face_selector_mode(face_selector_mode)


def update_face_selector_order(face_analyser_order: FaceSelectorOrder) -> Tuple:
    state_manager.set_item('face_selector_order', convert_str_none(face_analyser_order))
    return build_mapping_refresh_updates()


def update_face_selector_gender(face_selector_gender: Gender) -> Tuple:
    state_manager.set_item('face_selector_gender', convert_str_none(face_selector_gender))
    return build_mapping_refresh_updates()


def update_face_selector_race(face_selector_race: Race) -> Tuple:
    state_manager.set_item('face_selector_race', convert_str_none(face_selector_race))
    return build_mapping_refresh_updates()


def update_face_selector_age_range(face_selector_age_range: Tuple[float, float]) -> Tuple:
    start, end = face_selector_age_range
    state_manager.set_item('face_selector_age_start', int(start))
    state_manager.set_item('face_selector_age_end', int(end))
    return build_mapping_refresh_updates()


def clear_selected_faces() -> None:
    global current_selected_faces_by_slot, selected_face_index_by_slot, selector_face_index, focused_display_index
    current_selected_faces_by_slot = {}
    selected_face_index_by_slot = {}
    selector_face_index = -1
    focused_display_index = -1
    state_manager.set_item('reference_face_dict', {})
    state_manager.set_item('mapping_slot_keys', [])


def clear_and_update_reference_face_position(event: gradio.SelectData) -> gradio.Gallery:
    clear_reference_faces()
    clear_static_faces()
    clear_selected_faces()
    update_reference_face_position(event.index)
    return update_reference_position_gallery()


def update_reference_face_position(reference_face_position: int = 0) -> gradio.update:
    state_manager.set_item('reference_face_position', reference_face_position)
    return gradio.update()


def update_reference_face_distance(reference_face_distance: float) -> None:
    state_manager.set_item('reference_face_distance', reference_face_distance)


def update_reference_frame_number(reference_frame_number: int) -> gradio.update:
    """User scrubbed the preview frame slider — refresh detected-face crops only."""
    from facefusion.uis.components.preview import should_ignore_slider_release

    if should_ignore_slider_release():
        return gradio.update()

    state_manager.set_item('reference_frame_number', int(reference_frame_number))
    return update_reference_position_gallery()


def clear_and_update_reference_position_gallery() -> gradio.update:
    clear_reference_faces()
    clear_static_faces()
    clear_selected_faces()
    return update_reference_position_gallery()


def detect_face_gallery_items() -> List[VisionFrame]:
    """Face crops for the active target at reference_frame_number (no frame scan here)."""
    target_path = state_manager.get_item('target_path')
    if not target_path:
        return []

    frame_number = state_manager.get_item('reference_frame_number')
    if frame_number is None:
        frame_number = 0
    else:
        frame_number = int(frame_number)

    if is_image(target_path):
        vision_frame = read_static_image(target_path)
    elif is_video(target_path):
        total = count_video_frame_total(target_path)
        if total > 0:
            frame_number = max(0, min(frame_number, total - 1))
        vision_frame = get_video_frame(target_path, frame_number)
    else:
        return []

    if vision_frame is None:
        return []
    return extract_gallery_frames(vision_frame)


def update_reference_position_gallery() -> gradio.update:
    """Build detected-face crop thumbnails (numpy RGB) for the active reference frame."""
    gallery_frames: List[VisionFrame] = []
    target_path = state_manager.get_item('target_path')
    reference_frame_number = state_manager.get_item('reference_frame_number')
    if reference_frame_number is None:
        reference_frame_number = 0
    if is_image(target_path):
        reference_frame = read_static_image(target_path)
        if reference_frame is not None:
            gallery_frames = extract_gallery_frames(reference_frame)
    elif is_video(target_path):
        reference_frame = get_video_frame(target_path, int(reference_frame_number))
        if reference_frame is not None:
            gallery_frames = extract_gallery_frames(reference_frame)
    frame_number = state_manager.get_item('reference_frame_number')
    target_path = state_manager.get_item('target_path') or ''
    print(
        f'[FaceFusion] Detected faces gallery: {len(gallery_frames)} face(s) '
        f'on frame {frame_number} — {target_path}',
        flush=True,
    )
    if gallery_frames:
        return gradio.update(value=gallery_frames, visible=True)
    return gradio.update(value=None, visible=True)


def refresh_detected_faces_gallery() -> gradio.update:
    return update_reference_position_gallery()


def clear_all_mappings_and_galleries() -> Tuple:
    """Clear detected faces + all mapping rows and their selected galleries."""
    clear_reference_faces()
    clear_static_faces()
    clear_selected_faces()
    detected_update = gradio.update(value=None, visible=True)
    return (detected_update,) + build_mapping_refresh_updates()


def extract_gallery_frames(temp_vision_frame: VisionFrame) -> List[VisionFrame]:
    global current_reference_faces, current_reference_frames
    gallery_vision_frames: list = []
    if temp_vision_frame is None:
        current_reference_faces = []
        current_reference_frames = []
        return gallery_vision_frames
    raw_faces = get_many_faces([temp_vision_frame], is_target_frame=True)
    faces = sort_and_filter_faces(raw_faces, vision_frame=temp_vision_frame, skip_auto_padding=True)
    if raw_faces and not faces:
        print(
            f'[FaceFusion] {len(raw_faces)} face(s) detected but none passed face-selector filters',
            flush=True,
        )
    current_reference_faces = faces
    for face in faces:
        start_x, start_y, end_x, end_y = map(int, face.bounding_box)
        padding_x = int((end_x - start_x) * 0.25)
        padding_y = int((end_y - start_y) * 0.25)
        start_x = max(0, start_x - padding_x)
        start_y = max(0, start_y - padding_y)
        end_x = max(0, end_x + padding_x)
        end_y = max(0, end_y + padding_y)
        crop = temp_vision_frame[start_y:end_y, start_x:end_x]
        crop = normalize_frame_color(crop)
        gallery_vision_frames.append(crop)
    current_reference_frames = gallery_vision_frames
    return gallery_vision_frames


def reference_frame_back(reference_frame_number: int):
    fps = int(detect_video_fps(state_manager.get_item('target_path')))
    return _nav_frame(max(0, reference_frame_number - fps))


def reference_frame_forward(reference_frame_number: int):
    fps = int(detect_video_fps(state_manager.get_item('target_path')))
    total = count_video_frame_total(state_manager.get_item('target_path'))
    return _nav_frame(min(reference_frame_number + fps, total))


def reference_frame_back_five(reference_frame_number: int):
    fps = int(detect_video_fps(state_manager.get_item('target_path')))
    return _nav_frame(max(0, reference_frame_number - 5 * fps))


def reference_frame_forward_five(reference_frame_number: int):
    fps = int(detect_video_fps(state_manager.get_item('target_path')))
    total = count_video_frame_total(state_manager.get_item('target_path'))
    return _nav_frame(min(reference_frame_number + 5 * fps, total))


def _nav_frame(new_frame: int):
    state_manager.set_item('reference_frame_number', new_frame)
    return (
        gradio.update(value=new_frame),
        refresh_detected_faces_gallery(),
    )


def update_selector_face_index(event_data: SelectData) -> None:
    global selector_face_index
    if isinstance(event_data, SelectData):
        selector_face_index = event_data.index


# ---------------------------------------------------------------------------
# Legacy aliases (kept so other modules import without breaking)
# ---------------------------------------------------------------------------

def append_reference_face(*args, **kwargs):  # noqa: ARG001
    """Legacy entry point — superseded by per-row Add. Returns no-ops."""
    return gradio.update(), gradio.update(), gradio.update(), gradio.update()


def delete_reference_face(*args, **kwargs):  # noqa: ARG001
    return gradio.update(), gradio.update(), gradio.update(), gradio.update()


def add_reference_face(*args, **kwargs):
    return append_reference_face(*args, **kwargs)


def remove_reference_face(*args, **kwargs):
    return delete_reference_face(*args, **kwargs)


def add_reference_face_2(*args, **kwargs):
    return append_reference_face(*args, **kwargs)


def remove_reference_face_2(*args, **kwargs):
    return delete_reference_face(*args, **kwargs)


def update_selected_face_index(event_data: SelectData) -> None:
    pass


def update_selected_face_index_2(event_data: SelectData) -> None:
    pass
