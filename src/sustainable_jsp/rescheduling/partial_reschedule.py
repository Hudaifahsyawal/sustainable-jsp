from __future__ import annotations

import copy

from sustainable_jsp.core.schedule import (
    generate_feasible_solution1_resch,
    generate_feasible_solution2_resch,
    get_schedule_time_AR_resch,
)
from sustainable_jsp.core.performance_matrix import calculate_time_objective
from sustainable_jsp.core.visualization import create_gantt_chart_final
from sustainable_jsp.algorithms.carbon import (
    calculate_bound,
    update_schedule_time,
    evaluate_individu_CeCmaxZ,
)
from sustainable_jsp.algorithms.workload import (
    AWBO,
    cumulative_workload_resch,
    workload_statistic,
)
from sustainable_jsp.scheduling.sustainable import sustainableJSP_resch
from sustainable_jsp.scheduling.dual_resource import dual_resource_JSP
from sustainable_jsp.rescheduling.helper import (
    initialize_reschedule,
    remove_operations,
    get_machine_start_time,
    get_initial_workload,
)


def reschedule_left_shift(
        cancel_job_id,
        disruption_time: float,
        case1,
        case1_solution_optimum,
        Time_schedule_P2,
        speedlevel_optimum=None,
        fair_operator_assignment=None,
        workload_list=None,
        event_times=None,
        EErate=None,
        carbon_emission_data=None,
        dual_resource: bool = False,
        weight: float = 0.75,
        visualization: bool = True,
        x_start=None,
        x_end=None,
        title=None,
        save: bool = False,
        obj_type: str = "cmax",
        IR: float = 3,
        flowtime_type: str = "average",
        bound=None,
        ) -> dict:
    """
    Reschedule job shop operations using a "left shift" strategy when specific jobs are cancelled.

    Does not re-optimize Phase 1 (machine-operation sequence) or Phase 2 (speed levels). Instead:
    1. Removes cancelled operations from the existing Phase 1 solution.
    2. Uses update_schedule_time() to recalculate start/finish times by shifting remaining
       operations earlier (left shift) to fill gaps left by cancelled operations.
    3. Preserves original speed level assignments in sustainable mode.
    4. Re-optimizes only Phase 3 (operator assignment) using AWBO.
    """
    if isinstance(cancel_job_id, (list, tuple, set)):
        cancel_job_set = set(cancel_job_id)
    else:
        cancel_job_set = {cancel_job_id}

    if not dual_resource:
        if carbon_emission_data is None:
            raise ValueError("carbon_emission_data mandatory if dual_resource=False (Phase 2 evaluation required).")
        if speedlevel_optimum is None:
            raise ValueError("speedlevel_optimum mandatory if dual_resource=False (Phase 2 evaluation requires original speed levels).")

    init = initialize_reschedule(
        case1, case1_solution_optimum, Time_schedule_P2, disruption_time,
        fair_operator_assignment, workload_list, event_times, dual_resource=dual_resource
    )
    a1 = init["a1"]
    a2 = init["a2"]
    a3 = init["a3"]
    machine_start_time = init["machine_start_time"]
    operator_ready_time = init["operator_ready_time"]
    operator_on_progress = init["operator_on_progress"]
    operator_initial_workload = init["operator_initial_workload"]
    a2_info = init["a2_info"]
    init_case1 = init["init_case1"]
    init_case1_solution = init["init_case1_solution"]
    init_time_schedule = init["init_time_schedule"]
    init_fair_operator_assignment = init["init_fair_operator_assignment"]
    init_workload_list = init["init_workload_list"]
    init_event_times = init["init_event_times"]

    active_op = set(a2 + a3)
    cancelled_operations = [op for op in active_op if op[0] in cancel_job_set]

    case1_reschedule = copy.deepcopy(case1)
    case1_reschedule = remove_operations(case1_reschedule, cancelled_operations)

    a2_new = [op for op in a2 if op not in cancelled_operations]
    done_operations = set(a1 + a2_new)
    for (j, o) in done_operations:
        case1_reschedule[j - 1][o - 1] = None

    case1_reschedule_None = copy.deepcopy(case1)
    for (j, o) in cancelled_operations:
        case1_reschedule_None[j - 1][o - 1] = None

    done_operations_None = set(a1 + a2)
    for (j, o) in done_operations_None:
        case1_reschedule_None[j - 1][o - 1] = None

    delete_op = set(a1 + a2 + cancelled_operations)
    RES_solution = copy.deepcopy(case1_solution_optimum)
    for m_id, ops in RES_solution.items():
        RES_solution[m_id] = [op for op in ops if op not in delete_op]

    Time_schedule_P2_active_op_only = {op: info for op, info in Time_schedule_P2.items() if op not in delete_op}
    updated_processing_time = [info['duration'] for info in Time_schedule_P2_active_op_only.values()]

    RES_schedule_time = update_schedule_time(
        case1_reschedule_None,
        RES_solution,
        Time_schedule_P2_active_op_only,
        updated_processing_time,
        reschedule=True,
        machine_start_time=machine_start_time,
        a2=a2_new,
        a2_info=a2_info,
        disruption_time=disruption_time
    )

    schedule_jobs_data = {op: dict(data) for op, data in RES_schedule_time.items() if op not in delete_op}

    if dual_resource:
        time_obj = calculate_time_objective(
            RES_schedule_time,
            obj_type=obj_type,
            IR=IR,
            flowtime_type=flowtime_type
        )
        RES_matrix_performance = {"time_objective": time_obj}

    else:
        op_order = list(Time_schedule_P2.keys())
        op_to_idx = {op: i for i, op in enumerate(op_order)}

        ops_res = []
        for m_id in RES_solution:
            ops_res.extend(RES_solution[m_id])
        ops_res_sorted = sorted(ops_res, key=lambda op: op_to_idx[op])

        RES_speedlevel = [speedlevel_optimum[op_to_idx[op]] for op in ops_res_sorted]

        operation_id = list(schedule_jobs_data.keys())
        original_processing_time = [info["original_duration"] for info in schedule_jobs_data.values()]
        machine_list = [m_id for op in operation_id for m_id, ops in RES_solution.items() if op in ops]

        if bound is None:
            LB_time_obj, UB_time_obj, LB_carbon, UB_carbon = calculate_bound(
                case1_reschedule_None,
                carbon_emission_data,
                RES_solution,
                schedule_jobs_data,
                reschedule=True,
                original_processing_time=original_processing_time,
                machine_start_time=machine_start_time,
                a2=a2_new,
                a2_info=a2_info,
                disruption_time=disruption_time,
                init_time_schedule=init_time_schedule,
                obj_type=obj_type,
                IR=IR,
                flowtime_type=flowtime_type
            )
        else:
            LB_time_obj = bound["LB_time_obj"]
            UB_time_obj = bound["UB_time_obj"]
            LB_carbon = bound["LB_carbon"]
            UB_carbon = bound["UB_carbon"]

        print(f'LB_time_obj:{LB_time_obj}, UB_time_obj: {UB_time_obj}, LB_carbon:{LB_carbon}, UB_carbon: {UB_carbon}')

        updated_time_objective, TCE, z = evaluate_individu_CeCmaxZ(
            RES_speedlevel,
            original_processing_time,
            carbon_emission_data,
            case1_reschedule_None,
            RES_solution,
            schedule_jobs_data,
            machine_list,
            LB_time_obj,
            UB_time_obj,
            LB_carbon,
            UB_carbon,
            weight=weight,
            reschedule=True,
            machine_start_time=machine_start_time,
            a2=a2,
            a2_info=a2_info,
            disruption_time=disruption_time,
            init_time_schedule=init_time_schedule,
            obj_type=obj_type,
            IR=IR,
            flowtime_type=flowtime_type
        )

        RES_matrix_performance = {
            "time_objective": updated_time_objective,
            "TCE": TCE,
            "z": z
        }

    RES_fair_workload = AWBO(
        EErate,
        case1_reschedule_None,
        schedule_jobs_data,
        RES_solution,
        prob=0.95,
        coloni_size=50,
        alpha=10,
        beta=2,
        reschedule=True,
        initial_workload=operator_initial_workload,
        initial_ready_time=operator_ready_time,
        a2=a2_new,
        a2_info=a2_info
    )

    RES_workload_list, RES_event_times, RES_current_workload = cumulative_workload_resch(
        RES_fair_workload,
        EErate,
        schedule_jobs_data,
        case1_reschedule_None,
        visualization=visualization,
        reschedule=True,
        initial_workload=operator_initial_workload,
        a2=a2_new,
        a2_info=a2_info
    )

    operation_to_operator = {task: opr_id for opr_id, ops in RES_fair_workload.items() for task in ops}

    for (j, o), info in RES_schedule_time.items():
        m_id, _ = case1[j - 1][o - 1]
        opr_id = operation_to_operator.get((j, o))
        info["machine_id"] = m_id
        info["operator_id"] = opr_id

    RES_schedule_time = dict(sorted({**init_time_schedule, **RES_schedule_time}.items()))

    a1a2 = a1 + a2
    a1a2_set = set(a1a2)
    for (j, o) in a1a2_set:
        op_key = (j, o)
        op_info = RES_schedule_time.get(op_key) or a2_info.get(op_key)
        machine_id = None
        original_duration = None
        if op_info is not None:
            machine_id = op_info.get('machine_id')
            original_duration = op_info.get('original_duration')
            if original_duration is None:
                original_duration = op_info.get('duration')
        if machine_id is not None and original_duration is not None:
            if 1 <= j <= len(case1_reschedule_None) and 1 <= o <= len(case1_reschedule_None[j - 1]):
                case1_reschedule_None[j - 1][o - 1] = (machine_id, original_duration)

    for m_id, op_list in init_case1_solution.items():
        if m_id in RES_solution:
            RES_solution[m_id] = op_list + RES_solution[m_id]
        else:
            RES_solution[m_id] = op_list

    for opr_id, init_tasks in init_fair_operator_assignment.items():
        if opr_id in RES_fair_workload:
            RES_fair_workload[opr_id] = init_tasks + RES_fair_workload[opr_id]
        else:
            RES_fair_workload[opr_id] = init_tasks

    for opr_id, init_wl in init_workload_list.items():
        if opr_id in RES_workload_list:
            RES_workload_list[opr_id] = init_wl + RES_workload_list[opr_id]
        else:
            RES_workload_list[opr_id] = init_wl

    for opr_id, init_et in init_event_times.items():
        if opr_id in RES_event_times:
            RES_event_times[opr_id] = init_et + RES_event_times[opr_id]
        else:
            RES_event_times[opr_id] = init_et

    if not dual_resource:
        best_speedlevel_all = [RES_schedule_time[op].get("speed_level") for op in RES_schedule_time.keys()]
    else:
        best_speedlevel_all = None

    if visualization:
        create_gantt_chart_final(
            RES_schedule_time,
            RES_solution,
            operator_assignment=RES_fair_workload,
            speedlevel_list=best_speedlevel_all,
            x_start=x_start,
            x_end=x_end,
            title=title,
            save=save
        )

    mean, variance, maximum, std_dev, coef_variation = workload_statistic(RES_current_workload)
    RES_matrix_performance["workload_mean"] = mean
    RES_matrix_performance["workload_variance"] = variance
    RES_matrix_performance["workload_maximum"] = maximum
    RES_matrix_performance["workload_std_dev"] = std_dev
    RES_matrix_performance["workload_coef_variation"] = coef_variation

    return {
        "mode": "dual_resource" if dual_resource else "sustainable",
        "case_reschedule": case1_reschedule_None,
        "solution": RES_solution,
        "speedlevel_optimum": best_speedlevel_all,
        "time_schedule": RES_schedule_time,
        "fair_workload": RES_fair_workload,
        "workload_list": RES_workload_list,
        "event_times": RES_event_times,
        "matrix_performance": RES_matrix_performance
    }


def reschedule_right_shift(
        rework_operation,
        case1,
        case1_solution_optimum,
        Time_schedule_P2,
        speedlevel_optimum=None,
        fair_operator_assignment=None,
        workload_list=None,
        event_times=None,
        EErate=None,
        carbon_emission_data=None,
        dual_resource: bool = False,
        weight: float = 0.75,
        visualization: bool = True,
        x_start=None,
        x_end=None,
        title=None,
        save: bool = False,
        obj_type: str = "cmax+flowtime",
        IR: float = 3,
        flowtime_type: str = "average") -> dict:
    """
    Reschedule job shop operations using a "right shift" strategy when specific operations need to be reworked.

    Does not re-optimize Phase 1 (machine-operation sequence) or Phase 2 (speed levels). Instead:
    1. Reactivates rework operations so they can be rescheduled.
    2. Uses update_schedule_time() to recalculate start/finish times by shifting remaining
       operations later (right shift) to accommodate rework operations.
    3. Preserves original speed level assignments in sustainable mode.
    4. Re-optimizes only Phase 3 (operator assignment) using AWBO.
    """
    if EErate is None:
        raise ValueError("EErate wajib diisi (required for Phase 3 operator assignment).")

    if not dual_resource:
        if carbon_emission_data is None:
            raise ValueError("carbon_emission_data mandatory if dual_resource=False (Phase 2 evaluation required).")
        if speedlevel_optimum is None:
            raise ValueError("speedlevel_optimum mandatory if dual_resource=False (Phase 2 evaluation requires original speed levels).")

    disruption_time = Time_schedule_P2[rework_operation[0]]['finished_time']

    for operation in rework_operation:
        finish_time = Time_schedule_P2[operation]['finished_time']
        if disruption_time != finish_time:
            print(f"disruption time operation {operation} at {disruption_time} missmatch with finished time of other operation at {finish_time}!")
            print('reschedule paused. delete unfinished operation from rework!!')
            return {
                "mode": "dual_resource" if dual_resource else "sustainable",
                "disruption_time": disruption_time,
                "case_reschedule": None,
                "solution": None,
                "speedlevel_optimum": None,
                "time_schedule": None,
                "fair_workload": None,
                "workload_list": None,
                "event_times": None,
                "matrix_performance": None
            }

    init = initialize_reschedule(
        case1, case1_solution_optimum, Time_schedule_P2, disruption_time,
        fair_operator_assignment, workload_list, event_times, dual_resource=dual_resource
    )
    a1 = init["a1"]
    a2 = init["a2"]
    a3 = init["a3"]
    machine_start_time = init["machine_start_time"]
    operator_ready_time = init["operator_ready_time"]
    operator_on_progress = init["operator_on_progress"]
    operator_initial_workload = init["operator_initial_workload"]
    a2_info = init["a2_info"]
    init_case1 = init["init_case1"]
    init_case1_solution = init["init_case1_solution"]
    init_time_schedule = init["init_time_schedule"]
    init_fair_operator_assignment = init["init_fair_operator_assignment"]
    init_workload_list = init["init_workload_list"]
    init_event_times = init["init_event_times"]

    case1_reschedule = copy.deepcopy(case1)

    done_operations = set(a1 + a2)
    for operation in rework_operation:
        if operation in done_operations:
            done_operations.remove(operation)

    for (j, o) in done_operations:
        case1_reschedule[j - 1][o - 1] = None

    RES_solution = copy.deepcopy(case1_solution_optimum)
    for m_id, ops in RES_solution.items():
        RES_solution[m_id] = [op for op in ops if op not in done_operations]

    Time_schedule_P2_active_op_only = {op: info for op, info in Time_schedule_P2.items() if op not in done_operations}
    updated_processing_time = [info['duration'] for info in Time_schedule_P2_active_op_only.values()]

    RES_schedule_time = update_schedule_time(
        case1_reschedule,
        RES_solution,
        Time_schedule_P2_active_op_only,
        updated_processing_time,
        reschedule=True,
        machine_start_time=machine_start_time,
        a2=a2,
        a2_info=a2_info,
        disruption_time=disruption_time
    )

    schedule_jobs_data = {op: dict(data) for op, data in RES_schedule_time.items() if op not in done_operations}

    if dual_resource:
        time_obj = calculate_time_objective(
            RES_schedule_time,
            obj_type=obj_type,
            IR=IR,
            flowtime_type=flowtime_type
        )
        RES_matrix_performance = {"time_objective": time_obj}
        RES_speedlevel = None

    else:
        op_order = list(Time_schedule_P2.keys())
        op_to_idx = {op: i for i, op in enumerate(op_order)}

        ops_res = []
        for m_id in RES_solution:
            ops_res.extend(RES_solution[m_id])
        ops_res_sorted = sorted(ops_res, key=lambda op: op_to_idx[op])

        RES_speedlevel = [speedlevel_optimum[op_to_idx[op]] for op in ops_res_sorted]

        operation_id = list(schedule_jobs_data.keys())
        original_processing_time = [
            case1_reschedule[j - 1][o - 1][1]
            for (j, o) in operation_id
            if case1_reschedule[j - 1][o - 1] is not None
        ]
        machine_list = [m_id for op in operation_id for m_id, ops in RES_solution.items() if op in ops]

        LB_time_obj, UB_time_obj, LB_carbon, UB_carbon = calculate_bound(
            case1_reschedule,
            carbon_emission_data,
            RES_solution,
            schedule_jobs_data,
            reschedule=True,
            original_processing_time=original_processing_time,
            machine_start_time=machine_start_time,
            a2=a2,
            a2_info=a2_info,
            disruption_time=disruption_time,
            init_time_schedule=init_time_schedule,
            obj_type=obj_type,
            IR=IR,
            flowtime_type=flowtime_type
        )

        updated_time_objective, TCE, z = evaluate_individu_CeCmaxZ(
            RES_speedlevel,
            original_processing_time,
            carbon_emission_data,
            case1_reschedule,
            RES_solution,
            schedule_jobs_data,
            machine_list,
            LB_time_obj,
            UB_time_obj,
            LB_carbon,
            UB_carbon,
            weight=weight,
            reschedule=True,
            machine_start_time=machine_start_time,
            a2=a2,
            a2_info=a2_info,
            disruption_time=disruption_time,
            init_time_schedule=init_time_schedule,
            obj_type=obj_type,
            IR=IR,
            flowtime_type=flowtime_type
        )

        RES_matrix_performance = {
            "time_objective": updated_time_objective,
            "TCE": TCE,
            "z": z
        }

    RES_fair_workload = AWBO(
        EErate,
        case1_reschedule,
        schedule_jobs_data,
        RES_solution,
        prob=0.95,
        coloni_size=50,
        alpha=10,
        beta=2,
        reschedule=True,
        initial_workload=operator_initial_workload,
        initial_ready_time=operator_ready_time,
        a2=a2,
        a2_info=a2_info
    )

    RES_workload_list, RES_event_times, RES_current_workload = cumulative_workload_resch(
        RES_fair_workload,
        EErate,
        schedule_jobs_data,
        case1_reschedule,
        visualization=visualization,
        reschedule=True,
        initial_workload=operator_initial_workload,
        a2=a2,
        a2_info=a2_info
    )

    operation_to_operator = {task: opr_id for opr_id, ops in RES_fair_workload.items() for task in ops}

    for (j, o), info in RES_schedule_time.items():
        m_id, _ = case1_reschedule[j - 1][o - 1]
        opr_id = operation_to_operator.get((j, o))
        info["machine_id"] = m_id
        info["operator_id"] = opr_id

    for op in a2:
        RES_schedule_time[op] = dict(a2_info[op])
    RES_schedule_time = dict(sorted(RES_schedule_time.items()))

    if dual_resource:
        best_speedlevel_all = None
    else:
        best_speedlevel_all = [RES_schedule_time[op].get("speed_level") for op in RES_schedule_time.keys()]

    if a2:
        for op in a2:
            j, o = op
            m = RES_schedule_time[op]['machine_id']
            RES_solution[m].insert(0, op)

            opr_id = RES_schedule_time[op]['operator_id']
            RES_fair_workload[opr_id].insert(0, op)

            st = a2_info[op]["start_time"]
            ft = a2_info[op]["finished_time"]
            base = float(operator_initial_workload.get(opr_id, 0.0))

            if opr_id not in RES_event_times:
                RES_event_times[opr_id] = []
            if opr_id not in RES_workload_list:
                RES_workload_list[opr_id] = []

            RES_event_times[opr_id] = [st, ft] + RES_event_times[opr_id]
            RES_workload_list[opr_id] = [base, base] + RES_workload_list[opr_id]

    for (j, o) in a2:
        machine_id = a2_info[(j, o)]['machine_id']
        duration = a2_info[(j, o)]['duration']
        if duration <= 0:
            continue
        if 1 <= j <= len(case1_reschedule) and 1 <= o <= len(case1_reschedule[j - 1]):
            case1_reschedule[j - 1][o - 1] = (machine_id, duration)

    if visualization:
        create_gantt_chart_final(
            RES_schedule_time,
            RES_solution,
            operator_assignment=RES_fair_workload,
            speedlevel_list=best_speedlevel_all,
            x_start=x_start,
            x_end=x_end,
            title=title,
            save=save
        )

    mean, variance, maximum, std_dev, coef_variation = workload_statistic(RES_current_workload)
    RES_matrix_performance["workload_mean"] = mean
    RES_matrix_performance["workload_variance"] = variance
    RES_matrix_performance["workload_maximum"] = maximum
    RES_matrix_performance["workload_std_dev"] = std_dev
    RES_matrix_performance["workload_coef_variation"] = coef_variation

    return {
        "mode": "dual_resource" if dual_resource else "sustainable",
        "disruption_time": disruption_time,
        "case_reschedule": case1_reschedule,
        "solution": RES_solution,
        "speedlevel_optimum": best_speedlevel_all,
        "time_schedule": RES_schedule_time,
        "fair_workload": RES_fair_workload,
        "workload_list": RES_workload_list,
        "event_times": RES_event_times,
        "matrix_performance": RES_matrix_performance
    }


def reschedule_greedy_insertion(
        new_job,
        disruption_time: float,
        case1,
        case1_solution_optimum,
        Time_schedule_P2,
        fair_operator_assignment=None,
        workload_list=None,
        event_times=None,
        EErate=None,
        carbon_emission_data=None,
        dual_resource: bool = False,
        SL=None,
        weight: float = 0.75,
        visualization: bool = True,
        x_start=None,
        x_end=None,
        title=None,
        save: bool = False,
        obj_type: str = "cmax",
        IR: float = 3,
        flowtime_type: str = "average",
        bound=None,
        ) -> dict:
    """
    Reschedule job shop operations using a "greedy insertion" strategy when new jobs arrive.

    Does not re-optimize Phase 1 (machine-operation sequence) or Phase 2 (speed levels). Instead:
    1. Removes finished (A1) and ongoing (A2) operations from the schedule.
    2. For each new job operation, finds the earliest feasible insertion position using greedy approach.
    3. Uses a fixed speed level (SL) for all new operations in sustainable mode.
    4. Re-optimizes only Phase 3 (operator assignment) using AWBO.
    """
    if EErate is None:
        raise ValueError("EErate wajib diisi (required for Phase 3 operator assignment).")

    if not dual_resource and carbon_emission_data is None:
        raise ValueError("carbon_emission_data mandatory if dual_resource=False (Phase 2 evaluation required).")

    init = initialize_reschedule(
        case1, case1_solution_optimum, Time_schedule_P2, disruption_time,
        fair_operator_assignment, workload_list, event_times, dual_resource=dual_resource
    )
    a1 = init["a1"]
    a2 = init["a2"]
    a3 = init["a3"]
    machine_start_time = init["machine_start_time"]
    operator_ready_time = init["operator_ready_time"]
    operator_on_progress = init["operator_on_progress"]
    operator_initial_workload = init["operator_initial_workload"]
    a2_info = init["a2_info"]
    init_case1 = init["init_case1"]
    init_case1_solution = init["init_case1_solution"]
    init_time_schedule = init["init_time_schedule"]
    init_fair_operator_assignment = init["init_fair_operator_assignment"]
    init_workload_list = init["init_workload_list"]
    init_event_times = init["init_event_times"]

    case1_reschedule = copy.deepcopy(case1)

    for (j, o) in a1 + a2:
        case1_reschedule[j - 1][o - 1] = None

    case1_reschedule.extend(new_job)

    delete_op = set(a1 + a2)

    RES_solution = copy.deepcopy(case1_solution_optimum)
    RES_time_schedule = copy.deepcopy(Time_schedule_P2)

    for m_id, ops in RES_solution.items():
        RES_solution[m_id] = [op for op in ops if op not in delete_op]

    next_job_id = len(case1) + 1
    job_ready_time = {}

    for job_offset, job_ops in enumerate(new_job):
        job_id = next_job_id + job_offset
        job_ready_time[job_id] = disruption_time

        for op_idx, (m_id, base_dur) in enumerate(job_ops, start=1):
            new_op = (job_id, op_idx)

            if dual_resource:
                new_dur = round(base_dur, 1)
            else:
                new_dur = round(base_dur * carbon_emission_data[SL][0], 1)

            seq = RES_solution.setdefault(m_id, [])

            for i in range(len(seq) + 1):
                if i == 0:
                    prev_ft = max(disruption_time, machine_start_time.get(m_id, 0.0))
                else:
                    prev_op = seq[i - 1]
                    prev_ft = RES_time_schedule[prev_op]['finished_time']

                earliest_start = max(prev_ft, job_ready_time[job_id])

                if i < len(seq):
                    next_op = seq[i]
                    next_st = RES_time_schedule[next_op]['start_time']

                    if earliest_start + new_dur <= next_st:
                        seq.insert(i, new_op)
                        job_ready_time[job_id] = earliest_start + new_dur
                        RES_time_schedule[new_op] = {
                            'start_time': earliest_start,
                            'duration': new_dur,
                            'finished_time': earliest_start + new_dur,
                            'speed_level': SL if not dual_resource else None,
                            'machine_id': m_id,
                            'original_duration': base_dur,
                        }
                        break
                else:
                    seq.insert(i, new_op)
                    job_ready_time[job_id] = earliest_start + new_dur
                    RES_time_schedule[new_op] = {
                        'start_time': earliest_start,
                        'duration': new_dur,
                        'finished_time': earliest_start + new_dur,
                        'speed_level': SL if not dual_resource else None,
                        'machine_id': m_id,
                        'original_duration': base_dur,
                    }
                    break

    RES_schedule_time = {op: info for op, info in RES_time_schedule.items() if op not in delete_op}

    if dual_resource:
        time_obj = calculate_time_objective(
            RES_schedule_time,
            obj_type=obj_type,
            IR=IR,
            flowtime_type=flowtime_type
        )
        RES_matrix_performance = {"time_objective": time_obj}

    else:
        RES_speedlevel = [data.get('speed_level') for data in RES_schedule_time.values()]

        operation_id = list(RES_schedule_time.keys())

        original_processing_time = [
            case1_reschedule[j - 1][o - 1][1]
            for (j, o) in operation_id
            if case1_reschedule[j - 1][o - 1] is not None
        ]

        machine_list = [m_id for op in operation_id for m_id, ops in RES_solution.items() if op in ops]

        if bound is None:
            LB_time_obj, UB_time_obj, LB_carbon, UB_carbon = calculate_bound(
                case1_reschedule,
                carbon_emission_data,
                RES_solution,
                RES_schedule_time,
                reschedule=True,
                original_processing_time=original_processing_time,
                machine_start_time=machine_start_time,
                a2=a2,
                a2_info=a2_info,
                disruption_time=disruption_time,
                init_time_schedule=init_time_schedule,
                obj_type=obj_type,
                IR=IR,
                flowtime_type=flowtime_type
            )
        else:
            LB_time_obj = bound["LB_time_obj"]
            UB_time_obj = bound["UB_time_obj"]
            LB_carbon = bound["LB_carbon"]
            UB_carbon = bound["UB_carbon"]

        print(f'LB_time_obj:{LB_time_obj}, UB_time_obj: {UB_time_obj}, LB_carbon:{LB_carbon}, UB_carbon: {UB_carbon}')

        updated_time_objective, TCE, z = evaluate_individu_CeCmaxZ(
            RES_speedlevel,
            original_processing_time,
            carbon_emission_data,
            case1_reschedule,
            RES_solution,
            RES_schedule_time,
            machine_list,
            LB_time_obj,
            UB_time_obj,
            LB_carbon,
            UB_carbon,
            weight=weight,
            reschedule=True,
            machine_start_time=machine_start_time,
            a2=a2,
            a2_info=a2_info,
            disruption_time=disruption_time,
            init_time_schedule=init_time_schedule,
            obj_type=obj_type,
            IR=IR,
            flowtime_type=flowtime_type
        )

        RES_matrix_performance = {
            "time_objective": updated_time_objective,
            "TCE": TCE,
            "z": z
        }

    RES_fair_workload = AWBO(
        EErate,
        case1_reschedule,
        RES_schedule_time,
        RES_solution,
        prob=0.95,
        coloni_size=50,
        alpha=10,
        beta=2,
        reschedule=True,
        initial_workload=operator_initial_workload,
        initial_ready_time=operator_ready_time,
        a2=a2,
        a2_info=a2_info
    )

    RES_workload_list, RES_event_times, RES_current_workload = cumulative_workload_resch(
        RES_fair_workload,
        EErate,
        RES_schedule_time,
        case1_reschedule,
        visualization=visualization,
        reschedule=True,
        initial_workload=operator_initial_workload,
        a2=a2,
        a2_info=a2_info
    )

    operation_to_operator = {
        task: opr_id
        for opr_id, ops in RES_fair_workload.items()
        for task in ops
    }

    for (j, o), info in RES_schedule_time.items():
        if case1_reschedule[j - 1][o - 1] is not None:
            m_id, _ = case1_reschedule[j - 1][o - 1]
        else:
            m_id = None
        opr_id = operation_to_operator.get((j, o))
        info["machine_id"] = m_id
        info["operator_id"] = opr_id

        RES_schedule_time = dict(sorted({**init_time_schedule, **RES_schedule_time}.items()))

        a1a2 = a1 + a2
        a1a2_set = set(a1a2)
        for (j, o) in a1a2_set:
            op_key = (j, o)
            op_info = RES_schedule_time.get(op_key) or a2_info.get(op_key)
            machine_id = None
            original_duration = None
            if op_info is not None:
                machine_id = op_info.get('machine_id')
                original_duration = op_info.get('original_duration')
                if original_duration is None:
                    original_duration = op_info.get('duration')
            if machine_id is not None and original_duration is not None:
                if 1 <= j <= len(case1_reschedule) and 1 <= o <= len(case1_reschedule[j - 1]):
                    case1_reschedule[j - 1][o - 1] = (machine_id, original_duration)

    for m_id, op_list in init_case1_solution.items():
        if m_id in RES_solution:
            RES_solution[m_id] = op_list + RES_solution[m_id]
        else:
            RES_solution[m_id] = op_list

    for opr_id, init_tasks in init_fair_operator_assignment.items():
        if opr_id in RES_fair_workload:
            RES_fair_workload[opr_id] = init_tasks + RES_fair_workload[opr_id]
        else:
            RES_fair_workload[opr_id] = init_tasks

    for opr_id, init_wl in init_workload_list.items():
        if opr_id in RES_workload_list:
            RES_workload_list[opr_id] = init_wl + RES_workload_list[opr_id]
        else:
            RES_workload_list[opr_id] = init_wl

    for opr_id, init_et in init_event_times.items():
        if opr_id in RES_event_times:
            RES_event_times[opr_id] = init_et + RES_event_times[opr_id]
        else:
            RES_event_times[opr_id] = init_et

    if dual_resource:
        best_speedlevel_all = None
    else:
        best_speedlevel_all = [RES_schedule_time[op].get('speed_level') for op in RES_schedule_time.keys()]

    if visualization:
        create_gantt_chart_final(
            RES_schedule_time,
            RES_solution,
            operator_assignment=RES_fair_workload,
            speedlevel_list=best_speedlevel_all,
            x_start=x_start,
            x_end=x_end,
            title=title,
            save=save
        )

    mean, variance, maximum, std_dev, coef_variation = workload_statistic(RES_current_workload)
    RES_matrix_performance["workload_mean"] = mean
    RES_matrix_performance["workload_variance"] = variance
    RES_matrix_performance["workload_maximum"] = maximum
    RES_matrix_performance["workload_std_dev"] = std_dev
    RES_matrix_performance["workload_coef_variation"] = coef_variation

    return {
        "mode": "dual_resource" if dual_resource else "sustainable",
        "disruption_time": disruption_time,
        "case_reschedule": case1_reschedule,
        "solution": RES_solution,
        "speedlevel_optimum": best_speedlevel_all,
        "time_schedule": RES_schedule_time,
        "fair_workload": RES_fair_workload,
        "workload_list": RES_workload_list,
        "event_times": RES_event_times,
        "matrix_performance": RES_matrix_performance
    }
