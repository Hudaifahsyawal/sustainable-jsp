from collections import defaultdict


def calculate_cmax(schedule_time):
    """
    Calculate the makespan (Cmax) performance metric for a job shop schedule.

    The makespan (Cmax) is the total time required to complete all jobs, defined as
    the maximum completion time among all operations in the schedule. It represents
    the time when the last operation finishes, which is a fundamental performance
    metric in job shop scheduling problems. Minimizing Cmax is a common optimization
    objective as it directly relates to throughput and resource utilization.
    """
    if schedule_time is not None:
        cmax = max(value['finished_time'] for value in schedule_time.values())
    else:
        cmax = 0
    return cmax


def calculate_flowtime(time_schedule, type="average"):
    """
    Calculate flow time of each job based on time_schedule data.
    """
    job_times = defaultdict(list)

    for (job_id, op_id), time_info in time_schedule.items():
        job_times[job_id].append((op_id, time_info['start_time'], time_info['finished_time']))

    flow_time_per_job = {}

    for job_id, ops in job_times.items():
        ops_sorted = sorted(ops, key=lambda x: x[0])
        start_time_first_op = ops_sorted[0][1]
        finish_time_last_op = ops_sorted[-1][2]
        flow_time = finish_time_last_op - start_time_first_op
        flow_time_per_job[job_id] = flow_time

    max_flow_time = max(flow_time_per_job.values())
    total_flow_time = sum(flow_time_per_job.values())
    count_job = len(flow_time_per_job)
    average_flow_time = total_flow_time / count_job

    if type == "maxValue":
        return round(max_flow_time, 2)
    elif type == "average":
        return round(average_flow_time, 2)
    else:
        return flow_time_per_job


def calculate_cmax_AvgFlowTime(time_schedule, IR=3):
    cmax = calculate_cmax(time_schedule)
    AvgFlowTime = calculate_flowtime(time_schedule)
    classic_performance = IR * cmax + AvgFlowTime
    return round(classic_performance, 2)


def calculate_time_objective(schedule_time, obj_type="cmax+flowtime", IR=3, flowtime_type="average"):
    """
    Calculate time-based objective function for job shop scheduling performance evaluation.

    Computes various performance metrics from a schedule, including makespan (Cmax),
    flow time, or a combined metric. This function supports multiple objective types
    commonly used in job shop scheduling optimization.
    """
    def _calculate_cmax(schedule_time):
        if schedule_time is not None and len(schedule_time) > 0:
            return max(value['finished_time'] for value in schedule_time.values())
        return None

    def _calculate_flowtime(time_schedule, flowtime_type="average"):
        job_times = defaultdict(list)
        for (job_id, op_id), time_info in time_schedule.items():
            job_times[job_id].append((op_id, time_info['start_time'], time_info['finished_time']))

        flow_time_per_job = {}
        for job_id, ops in job_times.items():
            ops_sorted = sorted(ops, key=lambda x: x[0])
            start_time_first_op = ops_sorted[0][1]
            finish_time_last_op = ops_sorted[-1][2]
            flow_time = finish_time_last_op - start_time_first_op
            flow_time_per_job[job_id] = flow_time

        max_flow_time = max(flow_time_per_job.values())
        avg_flow_time = sum(flow_time_per_job.values()) / len(flow_time_per_job)

        if flowtime_type == "maxValue":
            return round(max_flow_time, 2)
        elif flowtime_type == "average":
            return round(avg_flow_time, 2)
        else:
            return flow_time_per_job

    def _calculate_cmax_AvgFlowTime(time_schedule, IR=3, flowtime_type="average"):
        cmax = _calculate_cmax(time_schedule)
        avg_flow = _calculate_flowtime(time_schedule, flowtime_type=flowtime_type)
        return round(IR * cmax + avg_flow, 2)

    if obj_type.lower() == "cmax":
        return _calculate_cmax(schedule_time)

    elif obj_type.lower() == "flowtime":
        return _calculate_flowtime(schedule_time, flowtime_type=flowtime_type)

    elif obj_type.lower() in ["cmax+flowtime", "combined", "classic"]:
        return _calculate_cmax_AvgFlowTime(schedule_time, IR=IR, flowtime_type=flowtime_type)

    else:
        raise ValueError(f"Unknown obj_type '{obj_type}'. Use 'Cmax', 'Flowtime', or 'Cmax+Flowtime'.")


def job_duration(job_data, type="maxValue"):
    """
    Calculate job duration statistics from job shop problem data.

    Computes the total processing duration for each job by summing the durations
    of all operations within that job. The function can return different statistics
    based on the type parameter: maximum duration, average duration, or a detailed
    list of all job durations.
    """
    total_duration_per_job = []
    durasi_list = []

    for i, job in enumerate(job_data, start=1):
        total_duration = sum(op[1] for op in job)
        total_duration_per_job.append((i, total_duration))
        durasi_list.append(total_duration)

    jumlah_job = len(job_data)
    total_durasi = sum(durasi_list)
    average_duration = total_durasi / jumlah_job

    max_duration = max(durasi_list)

    if type == "maxValue":
        return max_duration
    elif type == "average":
        return average_duration
    else:
        return total_duration_per_job
