import copy
import json
import os
import shutil
from typing import Any, Dict, List, Optional

from facefusion import state_manager
from facefusion.args import apply_args

EXTENSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_LEGACY_USER_DATA_DIR = os.path.join(EXTENSION_ROOT, 'user_data')


def _resolve_user_data_dir() -> str:
    """Resolve the writable user-data dir.

    Auto1111 exposes its base data dir via ``modules.paths_internal.data_path``
    (the same place that hosts ``styles.csv``, ``params.txt``, ``models/``, ...).
    We store our config under ``<data_path>/facefusion``. Falls back to a
    user-home location when not running inside Auto1111.
    """
    base: Optional[str] = None
    try:
        from modules.paths_internal import data_path  # type: ignore
        base = data_path
    except Exception:
        try:
            from modules.paths import data_path  # type: ignore
            base = data_path
        except Exception:
            base = None

    if not base:
        base = os.path.join(os.path.expanduser('~'), '.cache', 'facefusion')

    return os.path.join(base, 'facefusion')


USER_DATA_DIR = _resolve_user_data_dir()
UI_DEFAULTS_PATH = os.path.join(USER_DATA_DIR, 'ui_defaults.json')
SOURCE_PRESETS_PATH = os.path.join(USER_DATA_DIR, 'source_presets.json')
SOURCE_ASSETS_DIR = os.path.join(USER_DATA_DIR, 'source_assets')
MEDIA_TARGETS_PATH = os.path.join(USER_DATA_DIR, 'media_targets.json')
QUEUE_V2_MIGRATED_FLAG = os.path.join(USER_DATA_DIR, '.queue_v2_migrated')


def _migrate_legacy_user_data() -> None:
    """One-time move of any leftover files from extension/user_data/ to the new dir."""
    if not os.path.isdir(_LEGACY_USER_DATA_DIR):
        return
    try:
        os.makedirs(USER_DATA_DIR, exist_ok=True)
        for entry in os.listdir(_LEGACY_USER_DATA_DIR):
            src = os.path.join(_LEGACY_USER_DATA_DIR, entry)
            dst = os.path.join(USER_DATA_DIR, entry)
            if entry == '.gitkeep':
                continue
            if os.path.exists(dst):
                continue
            try:
                shutil.move(src, dst)
            except OSError as exc:
                print(f'[FaceFusion] migrate {src} -> {dst} failed: {exc}')
        try:
            shutil.rmtree(_LEGACY_USER_DATA_DIR)
        except OSError:
            pass
    except OSError as exc:
        print(f'[FaceFusion] legacy user_data migration failed: {exc}')


_migrate_legacy_user_data()

# Persisted via state_manager but not always registered in job_store before extensions load.
SETTINGS_EXTRA_KEYS = (
    'auto_padding_model',
    'auto_padding_confidence',
    'auto_padding_intersection_threshold',
    'auto_padding_mask_areas',
    'face_mask_areas',
    'remove_target_on_job_completion',
    'preview_update_seconds',
)

SETTINGS_EXCLUDE_KEYS = frozenset({
    'source_paths',
    'source_paths_2',
    'source_frame_dict',
    'source_slots',
    'target_path',
    'target_paths',
    'active_target_index',
    'reference_face_dict',
    'output_path',
    'job_id',
    'job_status',
    'step_index',
    'command',
})

# Full session snapshot (includes media/mapping keys excluded from ui_defaults.json).
SESSION_SNAPSHOT_EXTRA_KEYS = (
    'source_paths',
    'source_paths_2',
    'source_frame_dict',
    'source_slots',
    'target_path',
    'target_paths',
    'active_target_index',
    'reference_face_dict',
    'mapping_slot_keys',
    'output_path',
)


def ensure_user_data_dir() -> str:
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    return USER_DATA_DIR


def collect_settings_args() -> Dict[str, Any]:
    from facefusion.jobs import job_store
    from facefusion.processors.core import get_processors_modules

    keys = set(job_store.get_job_keys()) | set(job_store.get_step_keys())

    settings = {}
    for key in keys:
        if key in SETTINGS_EXCLUDE_KEYS or key == 'reference_face_dict_2':
            continue
        value = state_manager.get_item(key)  # type: ignore[arg-type]
        if value is not None:
            settings[key] = value
    for key in SETTINGS_EXTRA_KEYS:
        value = state_manager.get_item(key)  # type: ignore[arg-type]
        settings[key] = value if value is not None else 'None'
    return settings


def read_ui_defaults() -> Dict[str, Any]:
    if not os.path.isfile(UI_DEFAULTS_PATH):
        return {}
    try:
        with open(UI_DEFAULTS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_ui_defaults() -> bool:
    ensure_user_data_dir()
    try:
        with open(UI_DEFAULTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(collect_settings_args(), f, indent=2, default=_json_default)
        return True
    except OSError:
        return False


def load_ui_defaults() -> bool:
    """Apply saved defaults with set_item so they win over globals/ini."""
    data = read_ui_defaults()
    if not data:
        return False
    apply_args(data, True)
    return True


def apply_saved_defaults_over_state() -> bool:
    """Re-apply ui_defaults.json on top of current state (after globals init)."""
    return load_ui_defaults()


def snapshot_session_settings() -> Dict[str, Any]:
    """Capture live UI state before a queued job overwrites it with frozen step args."""
    from facefusion.jobs import job_store

    keys = set(job_store.get_job_keys()) | set(job_store.get_step_keys()) | set(SETTINGS_EXTRA_KEYS)
    keys.update(SESSION_SNAPSHOT_EXTRA_KEYS)
    snapshot: Dict[str, Any] = {}
    for key in keys:
        if key in ('command', 'job_id', 'job_status', 'step_index', 'reference_face_dict_2'):
            continue
        value = state_manager.get_item(key)  # type: ignore[arg-type]
        if value is not None:
            snapshot[key] = copy.deepcopy(value)
        elif key in SESSION_SNAPSHOT_EXTRA_KEYS:
            snapshot[key] = None
    return snapshot


def restore_session_settings(snapshot: Dict[str, Any]) -> None:
    """Restore UI state after job processing so controls match the pre-job session."""
    if not snapshot:
        return
    apply_args(snapshot, True)
    sync_active_target_path()
    sync_legacy_source_paths()
    reference_face_dict = snapshot.get('reference_face_dict')
    if reference_face_dict is not None:
        state_manager.set_item('reference_face_dict', copy.deepcopy(reference_face_dict))
        try:
            from facefusion.uis.components.face_selector import restore_selected_faces_from_state
            restore_selected_faces_from_state()
        except Exception:
            pass
    else:
        try:
            from facefusion.uis.components.face_selector import clear_selected_faces
            clear_selected_faces()
        except Exception:
            pass


def _json_default(obj: Any) -> Any:
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')


def list_source_presets() -> List[str]:
    presets = _read_source_presets()
    return sorted(presets.keys())


def load_source_preset(name: str) -> Optional[Dict[str, Any]]:
    presets = _read_source_presets()
    return presets.get(name)


def save_source_preset(name: str, data: Dict[str, Any]) -> bool:
    if not name or not name.strip():
        return False
    presets = _read_source_presets()
    presets[name.strip()] = data
    return _write_source_presets(presets)


def delete_source_preset(name: str) -> bool:
    presets = _read_source_presets()
    if name in presets:
        del presets[name]
        return _write_source_presets(presets)
    return False


def _read_source_presets() -> Dict[str, Any]:
    ensure_user_data_dir()
    if not os.path.isfile(SOURCE_PRESETS_PATH):
        return {}
    try:
        with open(SOURCE_PRESETS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_source_presets(presets: Dict[str, Any]) -> bool:
    ensure_user_data_dir()
    try:
        with open(SOURCE_PRESETS_PATH, 'w', encoding='utf-8') as f:
            json.dump(presets, f, indent=2, default=_json_default)
        return True
    except (OSError, TypeError) as exc:
        print(f'[FaceFusion] Failed to write source presets: {exc}')
        return False


def _sanitize_dir_name(name: str) -> str:
    safe = ''.join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in name).strip()
    return safe or 'unnamed'


def persist_source_files(favorite_name: str, file_paths: List[str]) -> List[str]:
    """Copy uploaded files into <data_path>/facefusion/source_assets/<name>/ and return new paths."""
    ensure_user_data_dir()
    target_dir = os.path.join(SOURCE_ASSETS_DIR, _sanitize_dir_name(favorite_name))
    os.makedirs(target_dir, exist_ok=True)

    persisted: List[str] = []
    for src in file_paths:
        if not src or not os.path.isfile(src):
            continue
        # If already under target_dir, just keep it.
        try:
            if os.path.commonpath([os.path.abspath(src), target_dir]) == target_dir:
                persisted.append(os.path.abspath(src))
                continue
        except ValueError:
            pass
        dest = os.path.join(target_dir, os.path.basename(src))
        base, ext = os.path.splitext(os.path.basename(src))
        n = 1
        while os.path.exists(dest) and not os.path.samefile(src, dest):
            dest = os.path.join(target_dir, f'{base}_{n}{ext}')
            n += 1
        try:
            if not os.path.exists(dest):
                shutil.copy2(src, dest)
        except OSError as exc:
            print(f'[FaceFusion] Failed to copy source file {src} -> {dest}: {exc}')
            continue
        persisted.append(dest)
    return persisted


def remove_source_asset_dir(favorite_name: str) -> None:
    target_dir = os.path.join(SOURCE_ASSETS_DIR, _sanitize_dir_name(favorite_name))
    if os.path.isdir(target_dir):
        try:
            shutil.rmtree(target_dir)
        except OSError as exc:
            print(f'[FaceFusion] Failed to remove source asset dir {target_dir}: {exc}')


def sync_legacy_source_paths() -> None:
    """Mirror source_frame_dict slots 0/1 to source_paths / source_paths_2."""
    source_frame_dict = state_manager.get_item('source_frame_dict') or {}
    if 0 in source_frame_dict:
        state_manager.set_item('source_paths', source_frame_dict[0])
    else:
        state_manager.clear_item('source_paths')
    if 1 in source_frame_dict:
        state_manager.set_item('source_paths_2', source_frame_dict[1])
    else:
        state_manager.clear_item('source_paths_2')


def _discover_persisted_upload_targets() -> List[str]:
    """List image/video files in the FaceFusion uploads folder (survives restarts)."""
    try:
        from facefusion.target_registry import get_target_uploads_directory
        from facefusion.filesystem import is_image, is_video
    except ImportError:
        return []
    upload_dir = get_target_uploads_directory()
    if not os.path.isdir(upload_dir):
        return []
    discovered: List[str] = []
    for name in sorted(os.listdir(upload_dir)):
        full = os.path.join(upload_dir, name)
        if os.path.isfile(full) and (is_image(full) or is_video(full)):
            discovered.append(os.path.abspath(full))
    return discovered


def read_media_targets() -> Dict[str, Any]:
    if not os.path.isfile(MEDIA_TARGETS_PATH):
        return {}
    try:
        with open(MEDIA_TARGETS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_media_targets() -> bool:
    """Persist target media list and active index (paths must still exist on disk)."""
    ensure_user_data_dir()
    paths = list(state_manager.get_item('target_paths') or [])
    valid_paths = [os.path.abspath(p) for p in paths if p and os.path.isfile(p)]
    active_index = state_manager.get_item('active_target_index') or 0
    if valid_paths and active_index >= len(valid_paths):
        active_index = 0
    payload = {
        'target_paths': valid_paths,
        'active_target_index': active_index,
    }
    try:
        with open(MEDIA_TARGETS_PATH, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        return True
    except OSError as exc:
        print(f'[FaceFusion] Failed to save media targets: {exc}')
        return False


def load_media_targets() -> bool:
    """Restore target_paths from disk; merge any files found in uploads/."""
    data = read_media_targets()
    paths: List[str] = []
    if data:
        for path in data.get('target_paths') or []:
            if path and os.path.isfile(path):
                abs_path = os.path.abspath(path)
                if abs_path not in paths:
                    paths.append(abs_path)
    for path in _discover_persisted_upload_targets():
        if path not in paths:
            paths.append(path)
    if not paths:
        return False
    state_manager.set_item('target_paths', paths)
    active_index = int(data.get('active_target_index', 0)) if data else 0
    if active_index < 0 or active_index >= len(paths):
        active_index = 0
    state_manager.set_item('active_target_index', active_index)
    sync_active_target_path()
    return True


def sync_active_target_path() -> None:
    """Set target_path from target_paths[active_target_index]."""
    target_paths = state_manager.get_item('target_paths') or []
    active_index = state_manager.get_item('active_target_index') or 0
    if target_paths and 0 <= active_index < len(target_paths):
        state_manager.set_item('target_path', target_paths[active_index])
    elif target_paths:
        state_manager.set_item('active_target_index', 0)
        state_manager.set_item('target_path', target_paths[0])
    else:
        state_manager.clear_item('target_path')


def ensure_queue_v2_migrated(jobs_path: str) -> None:
    """One-time wipe of legacy job files when upgrading to the modern Queue tab."""
    if os.path.isfile(QUEUE_V2_MIGRATED_FLAG):
        return
    from facefusion.jobs import job_manager

    try:
        if jobs_path and os.path.isdir(jobs_path):
            job_manager.clear_jobs(jobs_path)
        if jobs_path:
            job_manager.init_jobs(jobs_path)
        os.makedirs(USER_DATA_DIR, exist_ok=True)
        with open(QUEUE_V2_MIGRATED_FLAG, 'w', encoding='utf-8') as flag_file:
            flag_file.write('1\n')
        print('[FaceFusion] Queue v2: cleared legacy job data (one-time migration).', flush=True)
    except OSError as exc:
        print(f'[FaceFusion] Queue v2 migration failed: {exc}', flush=True)
