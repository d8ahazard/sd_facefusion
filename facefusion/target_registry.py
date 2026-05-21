import os
import shutil
from typing import Dict, List, Optional, Set

from facefusion import state_manager
from facefusion.filesystem import is_image, is_video, remove_file
from facefusion.temp_helper import get_base_directory_path

# Session-only: target_path -> {uploaded: bool, job_ids: set}
_TARGET_UPLOAD_REGISTRY: Dict[str, Dict] = {}


def get_target_uploads_directory() -> str:
    path = os.path.join(get_base_directory_path(), 'uploads')
    os.makedirs(path, exist_ok=True)
    return path


def is_under_uploads(path: str) -> bool:
    if not path:
        return False
    try:
        uploads_dir = os.path.abspath(get_target_uploads_directory())
        return os.path.commonpath([os.path.abspath(path), uploads_dir]) == uploads_dir
    except ValueError:
        return False


def _get_registry() -> Dict[str, Dict]:
    return _TARGET_UPLOAD_REGISTRY


def register_uploaded_target(path: str) -> str:
    """Copy file into uploads/ if needed; register as uploaded. Returns final path."""
    if not path or not os.path.isfile(path):
        return path
    uploads_dir = get_target_uploads_directory()
    dest = os.path.join(uploads_dir, os.path.basename(path))
    base, ext = os.path.splitext(os.path.basename(path))
    n = 1
    while os.path.exists(dest) and not os.path.samefile(path, dest):
        dest = os.path.join(uploads_dir, f'{base}_{n}{ext}')
        n += 1
    if not is_under_uploads(path):
        shutil.copy2(path, dest)
        path = dest
    entry = _get_registry().setdefault(path, {'uploaded': True, 'job_ids': set()})
    entry['uploaded'] = True
    return path


def register_target_for_job(target_path: str, job_id: str, uploaded: bool = False) -> None:
    if not target_path or not job_id:
        return
    entry = _get_registry().setdefault(target_path, {'uploaded': uploaded, 'job_ids': set()})
    if uploaded:
        entry['uploaded'] = True
    entry['job_ids'].add(job_id)


def get_target_paths_for_job(job_id: str) -> List[str]:
    paths = []
    for path, entry in _get_registry().items():
        if job_id in entry.get('job_ids', set()):
            paths.append(path)
    from facefusion.jobs import job_manager
    if job_manager.validate_job(job_id):
        for step in job_manager.get_steps(job_id) or []:
            tp = step.get('args', {}).get('target_path')
            if tp and tp not in paths:
                paths.append(tp)
    return paths


def _remove_target_from_state(target_path: str) -> None:
    paths = list(state_manager.get_item('target_paths') or [])
    if target_path in paths:
        paths.remove(target_path)
        state_manager.set_item('target_paths', paths)
        active = state_manager.get_item('active_target_index') or 0
        if active >= len(paths):
            state_manager.set_item('active_target_index', max(0, len(paths) - 1))
        from facefusion.user_data import sync_active_target_path
        sync_active_target_path()
    _get_registry().pop(target_path, None)
    try:
        from facefusion.uis.components import media_targets as mt
        mt._target_face_mappings.pop(target_path, None)
    except Exception:
        pass


def _delete_upload_file(target_path: str) -> None:
    if is_under_uploads(target_path) and os.path.isfile(target_path):
        remove_file(target_path)


def _add_target_to_state(target_path: str) -> None:
    paths = list(state_manager.get_item('target_paths') or [])
    if target_path not in paths:
        paths.append(target_path)
        state_manager.set_item('target_paths', paths)
        state_manager.set_item('active_target_index', len(paths) - 1)
        from facefusion.user_data import sync_active_target_path
        sync_active_target_path()


def repair_missing_target_path(target_path: str) -> Optional[str]:
    """
    Restore a missing upload target from the temp-folder copy FaceFusion keeps
  during processing (same basename under temp/, not uploads/).
    """
    if not target_path:
        return None
    if os.path.isfile(target_path):
        return target_path

    basename = os.path.basename(target_path)
    if not basename:
        return None

    fallback = os.path.join(get_base_directory_path(), basename)
    if not os.path.isfile(fallback):
        return None

    if is_under_uploads(target_path):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy2(fallback, target_path)
        register_uploaded_target(target_path)
        _add_target_to_state(target_path)
        return target_path

    return fallback


def cleanup_targets_for_job(job_id: str, reason: str) -> None:
    """
    reason: 'completed' | 'deleted'
    completed: only if remove_target_on_job_completion is True
    deleted: always remove uploaded targets linked to this job when checkbox is off;
             when on, completed path already ran
    """
    remove_on_complete = state_manager.get_item('remove_target_on_job_completion')
    if remove_on_complete is None:
        remove_on_complete = True

    if reason == 'completed' and not remove_on_complete:
        return

    for target_path in get_target_paths_for_job(job_id):
        entry = _get_registry().get(target_path, {})
        if reason == 'completed' and not entry.get('uploaded') and not is_under_uploads(target_path):
            continue
        if reason == 'deleted':
            job_ids = entry.get('job_ids', set())
            job_ids.discard(job_id)
            if job_ids:
                continue
        _delete_upload_file(target_path)
        _remove_target_from_state(target_path)
