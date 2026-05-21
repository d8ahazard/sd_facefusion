import os
from datetime import datetime
from typing import Optional, Tuple

from facefusion.date_helper import describe_time_ago
from facefusion.jobs import job_manager
from facefusion.typing import JobStatus, TableContents, TableHeaders


def compose_job_list(job_status: JobStatus) -> Tuple[TableHeaders, TableContents]:
    jobs = job_manager.find_jobs(job_status)
    job_headers: TableHeaders = ['job id', 'target', 'steps', 'processors', 'date created', 'date updated', 'job status']
    job_contents: TableContents = []

    for index, job_id in enumerate(jobs):
        if job_manager.validate_job(job_id):
            job = jobs[job_id]
            step_total = job_manager.count_step_total(job_id)
            steps = job_manager.get_steps(job_id)
            target_label = ''
            processors_label = ''
            if steps:
                args = steps[0].get('args', {})
                tp = args.get('target_path') or ''
                target_label = os.path.basename(tp) if tp else ''
                procs = args.get('processors') or []
                processors_label = ', '.join(procs) if isinstance(procs, list) else str(procs)
            date_created = prepare_describe_datetime(job.get('date_created'))
            date_updated = prepare_describe_datetime(job.get('date_updated'))
            job_contents.append(
                [
                    job_id,
                    target_label,
                    step_total,
                    processors_label,
                    date_created,
                    date_updated,
                    job_status
                ])
    return job_headers, job_contents


def prepare_describe_datetime(date_time: Optional[str]) -> Optional[str]:
    if date_time:
        return describe_time_ago(datetime.fromisoformat(date_time))
    return None
