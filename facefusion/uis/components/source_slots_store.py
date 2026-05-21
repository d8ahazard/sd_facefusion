"""Ordered source list: single array of {name, paths} objects. UI index == array index."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, TypedDict

from facefusion import state_manager
from facefusion.user_data import sync_legacy_source_paths

MAX_SOURCE_SLOTS = 10
_RESERVED_SOURCE_NAME = re.compile(r'^Source\s+\d+\s*$', re.IGNORECASE)


class SourceSlot(TypedDict):
    name: str
    paths: List[str]


def _empty_slot(name: str) -> SourceSlot:
    return {'name': name, 'paths': []}


def _next_default_name(existing: List[str]) -> str:
    taken = {n for n in existing if n}
    n = 1
    while f'Source {n}' in taken:
        n += 1
    return f'Source {n}'


def _normalize_indexed_dict(raw: Any) -> Dict[int, List[str]]:
    """JSON presets often use string keys ('0', '1'); normalize to int keys."""
    if not isinstance(raw, dict) or not raw:
        return {}
    out: Dict[int, List[str]] = {}
    for key, value in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, list):
            out[index] = [p for p in value if isinstance(p, str) and p]
        elif isinstance(value, str) and value:
            out[index] = [value]
        else:
            out[index] = []
    return out


def _paths_from_frame_dict(raw_sfd: Any) -> List[str]:
    sfd = _normalize_indexed_dict(raw_sfd)
    if not sfd:
        return []
    return list(sfd[min(sfd.keys())])


def _normalize_slot(raw: Any) -> SourceSlot:
    if not isinstance(raw, dict):
        return _empty_slot('Source 1')
    name = str(raw.get('name') or '').strip() or 'Source 1'
    paths = raw.get('paths') or []
    if not isinstance(paths, list):
        paths = [paths] if paths else []
    paths = [p for p in paths if isinstance(p, str) and p]
    return {'name': name, 'paths': paths}


def _migrate_from_legacy() -> List[SourceSlot]:
    sfd_raw = state_manager.get_item('source_frame_dict') or {}
    labels_raw = state_manager.get_item('source_slot_labels') or {}
    if not sfd_raw and not labels_raw:
        return []

    sfd = _normalize_indexed_dict(sfd_raw)
    labels = {int(k): str(v) for k, v in labels_raw.items()}
    slots: List[SourceSlot] = []
    for index in sorted(sfd.keys()):
        slots.append({
            'name': labels.get(index) or f'Source {index + 1}',
            'paths': list(sfd.get(index) or []),
        })
    set_source_slots(slots)
    return slots


def _write_legacy_mirror(slots: List[SourceSlot]) -> None:
    """Keep legacy dict keys 0..n-1 aligned with array order for older code paths."""
    state_manager.set_item(
        'source_frame_dict',
        {i: list(slot['paths']) for i, slot in enumerate(slots)},
    )
    state_manager.set_item(
        'source_slot_labels',
        {i: slot['name'] for i, slot in enumerate(slots)},
    )
    sync_legacy_source_paths()


def get_source_slots() -> List[SourceSlot]:
    raw = state_manager.get_item('source_slots')
    if isinstance(raw, list) and raw:
        return [_normalize_slot(s) for s in raw]
    return _migrate_from_legacy()


def set_source_slots(slots: List[SourceSlot]) -> None:
    normalized = [_normalize_slot(s) for s in slots[:MAX_SOURCE_SLOTS]]
    state_manager.set_item('source_slots', normalized)
    _write_legacy_mirror(normalized)


def slot_count() -> int:
    return len(get_source_slots())


def get_slot(display_index: int) -> Optional[SourceSlot]:
    slots = get_source_slots()
    if 0 <= display_index < len(slots):
        return slots[display_index]
    return None


def update_slot_paths(display_index: int, paths: List[str]) -> None:
    slots = get_source_slots()
    if 0 <= display_index < len(slots):
        slots[display_index]['paths'] = list(paths)
        set_source_slots(slots)


def update_slot_name(display_index: int, name: str) -> None:
    slots = get_source_slots()
    if 0 <= display_index < len(slots):
        slots[display_index]['name'] = name.strip() or slots[display_index]['name']
        set_source_slots(slots)


def load_slots_from_presets(list_names, load_preset) -> None:
    """Build ordered slots from saved favorites (one object per preset)."""
    slots: List[SourceSlot] = []
    for name in sorted(list_names()):
        preset = load_preset(name)
        if not preset:
            continue
        paths: List[str] = []
        if preset.get('source_slots') and isinstance(preset['source_slots'], list):
            first = preset['source_slots'][0]
            if isinstance(first, dict):
                paths = list(first.get('paths') or [])
                name = str(first.get('name') or name)
        else:
            paths = _paths_from_frame_dict(preset.get('source_frame_dict'))
        slots.append({'name': name, 'paths': paths})
    set_source_slots(slots)


def add_source_at_top() -> bool:
    """Insert empty source at index 0; shift mapping indices +1. Returns False if at cap."""
    from facefusion.uis.components.face_selector import remap_source_slot_indices

    slots = get_source_slots()
    if len(slots) >= MAX_SOURCE_SLOTS:
        return False

    key_map = {i: i + 1 for i in range(len(slots))}
    new_name = _next_default_name([s['name'] for s in slots])
    slots = [_empty_slot(new_name)] + slots
    set_source_slots(slots)
    if key_map:
        remap_source_slot_indices(key_map)
    return True


def remove_source_at(display_index: int) -> Optional[str]:
    """Remove slot at display index. Returns removed favorite name if any."""
    from facefusion.uis.components.face_selector import remap_mapping_after_source_removed

    slots = get_source_slots()
    if display_index < 0 or display_index >= len(slots):
        return None

    removed_name = (slots[display_index].get('name') or '').strip()
    slots.pop(display_index)
    set_source_slots(slots)
    remap_mapping_after_source_removed(display_index)
    return removed_name or None


def is_reserved_source_name(name: str) -> bool:
    return bool(_RESERVED_SOURCE_NAME.match((name or '').strip()))


def slots_for_preset_save(display_index: int) -> Dict[str, Any]:
    slot = get_slot(display_index)
    if not slot:
        return {}
    return {
        'source_slots': [deepcopy(slot)],
        'source_frame_dict': {0: list(slot['paths'])},
        'source_slot_labels': {0: slot['name']},
    }
