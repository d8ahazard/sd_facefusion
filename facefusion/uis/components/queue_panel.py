import os
from typing import Optional

from facefusion import state_manager
from facefusion.args import collect_step_args
from facefusion.filesystem import get_output_path_auto, is_directory
from facefusion.jobs import job_helper, job_manager
from facefusion.target_registry import is_under_uploads, register_target_for_job
from facefusion.uis.ui_helper import suggest_output_path
from facefusion.user_data import sync_active_target_path


def _apply_session_mappings_for_target(target_path: str, snapshot_current: bool = True) -> None:
    """Use in-memory per-target mappings when building a queued job step."""
    import copy

    from facefusion.uis.components import media_targets as mt

    if snapshot_current:
        mt.snapshot_current_target_mappings()
    saved = mt.get_target_mappings_snapshot(target_path)
    state_manager.set_item('reference_face_dict', copy.deepcopy(saved.get('reference_face_dict') or {}))
    state_manager.set_item('mapping_slot_keys', list(saved.get('mapping_slot_keys') or []))


def _make_step_args(target_path: str) -> dict:
    sync_active_target_path()
    state_manager.set_item('target_path', target_path)
    step_args = collect_step_args()
    output_path = get_output_path_auto()
    step_args['output_path'] = output_path
    if is_directory(step_args.get('output_path')):
        step_args['output_path'] = suggest_output_path(output_path, target_path)
    return step_args


def _enqueue_target_with_mappings(target_path: str, snapshot_current: bool = True) -> bool:
    if not target_path:
        return False
    jobs_path = state_manager.get_item('jobs_path')
    if not job_manager.init_jobs(jobs_path):
        return False
    _apply_session_mappings_for_target(target_path, snapshot_current=snapshot_current)
    step_args = _make_step_args(target_path)
    job_id = job_helper.suggest_job_id('ui')
    created = (
        job_manager.create_job(job_id)
        and job_manager.add_step(job_id, step_args)
        and job_manager.submit_job(job_id)
    )
    if created:
        register_target_for_job(
            target_path,
            job_id,
            uploaded=is_under_uploads(target_path),
        )
    return created


def enqueue_target(target_path: str) -> bool:
    return _enqueue_target_with_mappings(target_path, snapshot_current=True)
