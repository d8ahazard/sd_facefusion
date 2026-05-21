import os
from typing import List, Tuple

from facefusion import logger, state_manager
from facefusion.common_helper import get_first
from facefusion.filesystem import filter_audio_paths, filter_image_paths, has_audio, has_image, is_audio
from facefusion.processors.classes.style_changer import StyleChanger
from facefusion.temp_helper import get_temp_directory_path
from facefusion.user_data import sync_legacy_source_paths

import gradio


def apply_source_files(file_names: List[str], slot_index: int) -> Tuple[bool, bool, str, str]:
    """
    Update source_slots[slot_index].paths from file paths (display index).
    Returns (has_audio, has_image, audio_path, image_path).
    """
    from facefusion.uis.components.source_slots_store import get_source_slots, set_source_slots

    style_changer = StyleChanger()
    target = state_manager.get_item('style_changer_target')
    slots = get_source_slots()
    if slot_index < 0 or slot_index >= len(slots):
        return False, False, '', ''

    audio_files = []
    for file in file_names:
        if file and os.path.exists(file):
            audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac']
            _, file_extension = os.path.splitext(file.lower())
            if file_extension in audio_extensions:
                audio_files.append(file)
            if is_audio(file) and file not in audio_files:
                audio_files.append(file)

    if 'source' in (target or '') and 'style_changer' in (state_manager.get_item('processors') or []):
        all_image_files = filter_image_paths(file_names)
        for base_file in all_image_files:
            file_base, ext = os.path.splitext(base_file)
            styled_file = os.path.join(get_temp_directory_path(base_file), f'ff_styled{ext}')
            if os.path.exists(styled_file):
                os.remove(styled_file)
            styled_file = style_changer.process_src_image(base_file, styled_file)
            file_names = [styled_file if f == base_file else f for f in file_names]

    has_audio_files = len(audio_files) > 0
    has_image_files = has_image(file_names)

    if file_names:
        if audio_files:
            processed_files = audio_files + [f for f in file_names if f not in audio_files]
            slots[slot_index]['paths'] = processed_files
        else:
            slots[slot_index]['paths'] = file_names
    else:
        slots[slot_index]['paths'] = []
    set_source_slots(slots)
    sync_legacy_source_paths()

    audio_path = get_first(audio_files) if audio_files else None
    image_path = get_first(filter_image_paths(file_names))
    return has_audio_files, has_image_files, audio_path or '', image_path or ''


def get_active_source_slot_count() -> int:
    from facefusion.uis.components.source_slots_store import slot_count

    return slot_count()
