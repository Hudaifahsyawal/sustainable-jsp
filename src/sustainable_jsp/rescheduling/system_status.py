def get_a2a3_operations(time_schedule, disruption_time):
    """
    Menghitung set operasi a3 (remaining operations) pada saat disruption_time.

    Aturan:
    - finished_time <= disruption_time  -> dianggap sudah selesai (a1)
    - start_time < disruption_time < finished_time -> on-progress (a2)
    - start_time >= disruption_time -> remaining (a3)
    """
    a3 = []
    a2 = []

    for op, tinfo in time_schedule.items():
        st = tinfo['start_time']
        ft = tinfo['finished_time']

        if ft <= disruption_time:
            continue
        elif st < disruption_time < ft:
            a2.append(op)
        else:
            a3.append(op)

    return a2, a3


def get_system_status(time_schedule, disruption_time, solution_optimum, disruption_type=None, new_jobs=None, cancel_jobs=None, speed_factor=0.8):
    """
    Menghitung status sistem pada saat disruption_time.
    """
    cmax = max(tinfo['finished_time'] for tinfo in time_schedule.values())

    a2, a3 = get_a2a3_operations(time_schedule, disruption_time)
    a2a3 = set(a2 + a3)
    n_a2a3 = len(a2a3)
    t_a2a3 = (
        sum(time_schedule[op]['duration'] for op in a3)
        + sum(time_schedule[op]['finished_time'] - disruption_time for op in a2)
    )

    n_all_operations = len(time_schedule)
    t_all_operations = sum(tinfo['duration'] for tinfo in time_schedule.values())

    s1 = disruption_time / cmax
    s2 = n_a2a3 / n_all_operations
    s3 = t_a2a3 / t_all_operations

    machine_utils = {}
    for machine_id in solution_optimum.keys():
        machine_utils[machine_id] = 0

    for (job_id, op_id), tinfo in time_schedule.items():
        st = tinfo['start_time']
        ft = tinfo['finished_time']
        machine_id = tinfo['machine_id']

        if st >= disruption_time:
            dur = tinfo['duration']
            if machine_id in machine_utils:
                machine_utils[machine_id] += dur
            else:
                machine_utils[machine_id] = dur
        elif st < disruption_time < ft:
            dur = ft - disruption_time
            if dur > 0:
                if machine_id in machine_utils:
                    machine_utils[machine_id] += dur
                else:
                    machine_utils[machine_id] = dur

    normalization_factor = cmax - disruption_time
    for machine_id in machine_utils:
        machine_utils[machine_id] = machine_utils[machine_id] / normalization_factor if normalization_factor > 0 else 0
    utilities = list(machine_utils.values())
    if utilities:
        s4 = sum(utilities) / len(utilities)
        s5 = max(utilities)
        s6 = min(utilities)
    else:
        s4 = s5 = s6 = 0

    d1 = d2 = d3 = d4 = d5 = d6 = 0

    if disruption_type == "new_job_arrival":
        n_new_jobs = sum(len(job) for job in new_jobs)
        t_new_jobs = sum(duration * speed_factor for a_job in new_jobs for op in a_job for (_, duration) in op)
        d1 = n_new_jobs / n_a2a3
        d2 = t_new_jobs / (cmax - disruption_time)
        d3 = t_new_jobs / t_a2a3
    elif disruption_type == "cancel_job":
        cancel_operations = [op for op in a3 if op[0] in cancel_jobs]
        n_cancel_operations = len(cancel_operations)
        t_cancel_operations = sum(time_schedule[op]['duration'] for op in cancel_operations)
        d4 = n_cancel_operations / n_a2a3
        d5 = t_cancel_operations / (cmax - disruption_time)
        d6 = t_cancel_operations / t_a2a3

    s_metrics = (f"{s2*100:.2f}% of remaining operations, "
                 f"{s3*100:.2f}% of remaining duration, "
                 f"Avg. machine utility after disruption is {s4:.2f}, "
                 f"with Max utility: {s5:.2f}, Min utility: {s6:.2f}.")

    if disruption_type == "new_job_arrival":
        d_metrics = (f"{d1*100:.2f}% of new operations added over remaining operations, "
                     f"{d2*100:.2f}% of new operations duration over remaining horizon time, "
                     f"{d3*100:.2f}% of new operations duration over duration of all remaining operations.")
        t1_str = (f"New job arrival occurs at time position {s1:.4f}. with current system status: "
                  f"{s_metrics} and new job arrival metrics: {d_metrics}")
    elif disruption_type == "cancel_job":
        d_metrics = (f"{d4*100:.2f}% of operations cancelled over remaining operations, "
                     f"{d5*100:.2f}% of duration reduction over remaining horizon time, "
                     f"{d6*100:.2f}% of duration reduction over duration of all remaining operations.")
        t1_str = (f"Job cancellation occurs at time position {s1:.4f}. with current system status: "
                  f"{s_metrics} and job cancellation metrics: {d_metrics}")
    else:
        t1_str = f"{disruption_type} occurs at time position {s1:.2f}. with current system status: {s_metrics}"

    T1 = t1_str
    return {
        "T2": disruption_type,
        "s1": round(s1, 4), "s2": round(s2, 4), "s3": round(s3, 4),
        "s4": round(s4, 4), "s5": round(s5, 4), "s6": round(s6, 4),
        "d1": round(d1, 4), "d2": round(d2, 4), "d3": round(d3, 4),
        "d4": round(d4, 4), "d5": round(d5, 4), "d6": round(d6, 4),
        "T1": T1
    }
