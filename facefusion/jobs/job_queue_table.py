import os
from datetime import datetime
from typing import List, Optional, Set, Tuple

from facefusion.date_helper import describe_time_ago
from facefusion.jobs import job_manager
from facefusion.typing import TableContents, TableHeaders

UI_JOB_PREFIX = 'ui-'

QUEUE_TABLE_HEADERS: TableHeaders = ['sel', 'target', 'status', 'steps', 'updated']


def is_ui_job(job_id: str) -> bool:
    return bool(job_id) and job_id.startswith(UI_JOB_PREFIX)


def filter_ui_job_ids(job_ids: List[str]) -> List[str]:
    return [job_id for job_id in job_ids if is_ui_job(job_id)]


def get_all_ui_job_ids() -> List[str]:
    job_ids: List[str] = []
    for status in ('drafted', 'queued', 'failed', 'completed'):
        job_ids.extend(filter_ui_job_ids(job_manager.find_job_ids(status) or []))
    return job_ids


def _target_label(job_id: str) -> str:
    steps = job_manager.get_steps(job_id)
    if not steps:
        return ''
    target_path = steps[0].get('args', {}).get('target_path') or ''
    return os.path.basename(target_path) if target_path else ''


def _updated_label(job_id: str) -> Optional[str]:
    job = job_manager.read_job_file(job_id)
    if not job:
        return None
    date_updated = job.get('date_updated') or job.get('date_created')
    if date_updated:
        return describe_time_ago(datetime.fromisoformat(date_updated))
    return None


def _row_for_job(job_id: str, status: str, selected: bool = False) -> List:
    return [
        selected,
        _target_label(job_id),
        status,
        job_manager.count_step_total(job_id),
        _updated_label(job_id),
    ]


def compose_active_rows(
    current_job_id: Optional[str] = None,
    selected_job_ids: Optional[Set[str]] = None,
) -> Tuple[TableHeaders, TableContents, List[str]]:
    rows: TableContents = []
    selected = selected_job_ids or set()
    job_ids: List[str] = filter_ui_job_ids(job_manager.find_job_ids('queued') or [])
    for job_id in job_ids:
        status = 'running' if current_job_id and job_id == current_job_id else 'queued'
        rows.append(_row_for_job(job_id, status, selected=job_id in selected))
    return QUEUE_TABLE_HEADERS, rows, job_ids


def compose_history_rows(
    selected_job_ids: Optional[Set[str]] = None,
) -> Tuple[TableHeaders, TableContents, List[str]]:
    rows: TableContents = []
    job_ids: List[str] = []
    selected = selected_job_ids or set()
    failed_ids = list(reversed(filter_ui_job_ids(job_manager.find_job_ids('failed') or [])))
    completed_ids = list(reversed(filter_ui_job_ids(job_manager.find_job_ids('completed') or [])))
    for job_id in failed_ids:
        job_ids.append(job_id)
        rows.append(_row_for_job(job_id, 'failed', selected=job_id in selected))
    for job_id in completed_ids:
        job_ids.append(job_id)
        rows.append(_row_for_job(job_id, 'completed', selected=job_id in selected))
    return QUEUE_TABLE_HEADERS, rows, job_ids


def _is_sel_checked(value) -> bool:
    if value is True:
        return True
    if isinstance(value, (int, float)) and value == 1:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1', 'yes', 'on', 'x', '✓', '☑')
    return bool(value)


def _table_rows(dataframe_value) -> List:
    if dataframe_value is None:
        return []
    if hasattr(dataframe_value, 'values'):
        return dataframe_value.values.tolist()
    if isinstance(dataframe_value, list):
        return dataframe_value
    return list(dataframe_value)


def get_selected_job_ids(dataframe_value: Optional[List], job_ids: List[str]) -> List[str]:
    rows = _table_rows(dataframe_value)
    if not rows or not job_ids:
        return []
    selected: List[str] = []
    for index, row in enumerate(rows):
        if index >= len(job_ids):
            break
        if not row:
            continue
        if _is_sel_checked(row[0]):
            selected.append(job_ids[index])
    return selected


def get_job_step_args(job_id: str) -> dict:
    steps = job_manager.get_steps(job_id)
    if not steps:
        return {}
    return steps[0].get('args', {}) or {}
