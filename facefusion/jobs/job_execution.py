import threading
from typing import Callable, List, Optional

from facefusion import process_manager, state_manager
from facefusion.ff_status import FFStatus
from facefusion.jobs import job_manager, job_runner, job_store
from facefusion.target_registry import cleanup_targets_for_job
from facefusion.typing import ProcessStep

_LOCK = threading.Lock()
_WORKER_THREAD: Optional[threading.Thread] = None
_BUSY = False
_CURRENT_JOB_ID: Optional[str] = None


def is_busy() -> bool:
    return _BUSY or process_manager.is_processing()


def get_current_job_id() -> Optional[str]:
    return _CURRENT_JOB_ID


def _sync_job_keys() -> None:
    for key in job_store.get_job_keys():
        state_manager.sync_item(key)  # type: ignore[arg-type]


def _prime_state_for_job_status(job_id: str) -> None:
    """Apply the job step target (and trim) so FFStatus progress totals match the running job."""
    steps = job_manager.get_steps(job_id)
    if not steps:
        return
    step_args = steps[0].get('args') or {}
    if step_args.get('target_path'):
        state_manager.set_item('target_path', step_args['target_path'])
    for key in ('trim_frame_start', 'trim_frame_end', 'processors'):
        if key in step_args and step_args[key] is not None:
            state_manager.set_item(key, step_args[key])


def _run_job_sync(job_id: str, process_step: ProcessStep) -> bool:
    global _CURRENT_JOB_ID
    _CURRENT_JOB_ID = job_id
    try:
        _sync_job_keys()
        ok = job_runner.run_job(job_id, process_step)
        if ok:
            cleanup_targets_for_job(job_id, 'completed')
        return ok
    finally:
        _CURRENT_JOB_ID = None


def _run_one_job(job_id: str, process_step: ProcessStep, retry: bool) -> None:
    if retry:
        job_runner.retry_job(job_id, process_step)
    else:
        _run_job_sync(job_id, process_step)


def _worker_run_jobs(job_specs: List[tuple], process_step: ProcessStep) -> None:
    """job_specs: list of (job_id, retry_bool)"""
    global _BUSY
    status = FFStatus()
    try:
        process_manager.start()
        _sync_job_keys()
        for index, (job_id, retry) in enumerate(job_specs):
            if process_manager.is_stopping():
                break
            _prime_state_for_job_status(job_id)
            if index > 0:
                status.next(f'Running {job_id}')
            else:
                status.start(f'Running {job_id}')
            _run_one_job(job_id, process_step, retry)
        status.finish('Jobs finished.')
    except Exception as exc:
        import traceback
        print(f'[FaceFusion] Job worker error: {exc}', flush=True)
        traceback.print_exc()
        FFStatus().finish(f'Job error: {exc}')
    finally:
        process_manager.end()
        with _LOCK:
            _BUSY = False


def _start_worker(job_specs: List[tuple], process_step: ProcessStep) -> bool:
    global _WORKER_THREAD, _BUSY
    if not job_specs:
        return False
    with _LOCK:
        if _BUSY:
            return False
        _BUSY = True
        _WORKER_THREAD = threading.Thread(
            target=_worker_run_jobs,
            args=(list(job_specs), process_step),
            daemon=True,
            name='facefusion-job-worker',
        )
        _WORKER_THREAD.start()
    return True


def start_all_queued(process_step: ProcessStep) -> bool:
    job_ids = job_manager.find_job_ids('queued') or []
    specs = [(job_id, False) for job_id in job_ids]
    return _start_worker(specs, process_step)


def start_jobs(job_ids: List[str], process_step: ProcessStep, retry_failed: bool = False) -> bool:
    specs = [(job_id, retry_failed) for job_id in job_ids]
    return _start_worker(specs, process_step)


def start_jobs_mixed(job_specs: List[tuple], process_step: ProcessStep) -> bool:
    return _start_worker(job_specs, process_step)


def start_single_job(job_id: str, process_step: ProcessStep) -> bool:
    return _start_worker([(job_id, False)], process_step)
