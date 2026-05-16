from __future__ import annotations

import copy

from sustainable_jsp.core.schedule import (
    generate_feasible_solution1_resch,
    generate_feasible_solution2_resch,
    get_schedule_time_AR_resch,
)
from sustainable_jsp.core.performance_matrix import calculate_time_objective
from sustainable_jsp.core.visualization import create_gantt_chart_final
from sustainable_jsp.algorithms.carbon import calculate_bound, update_schedule_time
from sustainable_jsp.algorithms.genetic import (
    crossover1, crossover1P, crossoverNR, crossoverNRP,
    mutate1N, mutate1R,
)
from sustainable_jsp.scheduling.sustainable import sustainableJSP_resch
from sustainable_jsp.scheduling.dual_resource import dual_resource_JSP
from sustainable_jsp.rescheduling.helper import (
    initialize_reschedule,
    classify_operation,
    remove_operations,
    get_machine_start_time,
    get_initial_workload,
    determine_operator_ready_time,
)


def reschedule_new_arrival_job(
        new_job,
        disruption_time,
        case1,
        case1_solution_optimum,
        Time_schedule_P2,
        fair_operator_assignment,
        workload_list,
        event_times,
        EErate,
        carbon_emission_data=None,
        dual_resource=False,
        weight=0.75,
        visualization=True,
        x_start=None,
        x_end=None,
        title=None,
        save=False,
        obj_type="cmax+flowtime",
        workload_obj_type="variance",
        IR=3,
        flowtime_type="average",
        num_iterations1=1500,
        population_size1=75,
        elit_percentage1=0.6,
        num_iterations2=1500,
        population_size2=75,
        elit_percentage2=0.6,
        coloni_size=450,
        alpha=10,
        beta=2):
    """
    Completely Reschedule job shop operations when new jobs arrive at a disruption time.

    Handles rescheduling scenarios where new jobs arrive during the execution of an existing schedule
    using completely reschedule strategy. It: (1) classifies existing operations into finished (A1),
    ongoing (A2), and unprocessed (A3) categories, (2) removes finished and ongoing operations from
    the job data, (3) adds the new arriving jobs, and (4) performs optimization using either
    dual-resource or sustainable scheduling approaches.
    """
    if not dual_resource and carbon_emission_data is None:
        raise ValueError(
            "carbon_emission_data harus diisi jika dual_resource=False "
            "(sustainableJSP_resch digunakan)."
        )

    init = initialize_reschedule(
        case1, case1_solution_optimum, Time_schedule_P2, disruption_time,
        fair_operator_assignment, workload_list, event_times, dual_resource=dual_resource)
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

    print(f"ini adalah hasil dari initial rescheduling code")
    print(f"a2_info = {a2_info}")

    case1_reschedule = copy.deepcopy(case1)

    for (j, o) in a1 + a2:
        case1_reschedule[j - 1][o - 1] = None

    case1_reschedule.extend(new_job)

    print("\njob data after adding new job")
    for i, job in enumerate(case1_reschedule, start=1):
        print(f"Job {i}: {job}")

    if dual_resource:
        out = dual_resource_JSP(
            case1_reschedule,
            EErate,
            mutation_threshold1=0.5,
            num_iterations1=num_iterations1,
            population_size1=population_size1,
            elit_percentage1=elit_percentage1,
            FSG=generate_feasible_solution2_resch,
            crossover=crossoverNR,
            mutate=mutate1N,
            get_schedule_time=get_schedule_time_AR_resch,
            visualization=visualization,
            reschedule=True,
            machine_start_time=machine_start_time,
            prob=0.95,
            coloni_size=coloni_size,
            alpha=alpha,
            beta=beta,
            initial_workload=operator_initial_workload,
            initial_ready_time=operator_ready_time,
            x_start=x_start,
            x_end=x_end,
            a2=a2,
            a2_info=a2_info,
            disruption_time=disruption_time,
            init_time_schedule=init_time_schedule,
            title=title,
            save=save,
            obj_type=obj_type,
            IR=IR,
            flowtime_type=flowtime_type
        )
        return {"mode": "dual_resource", **out}

    else:
        out = sustainableJSP_resch(
            case1_reschedule,
            carbon_emission_data,
            EErate,
            mutation_threshold1=0.5,
            num_iterations1=num_iterations1,
            population_size1=population_size1,
            elit_percentage1=elit_percentage1,
            FSG=generate_feasible_solution2_resch,
            crossover=crossoverNR,
            mutate=mutate1N,
            get_schedule_time=get_schedule_time_AR_resch,
            num_iterations2=num_iterations2,
            population_size2=population_size2,
            elit_percentage2=elit_percentage2,
            mutation_threshold2=0.5,
            weight=weight,
            visualization=visualization,
            reschedule=True,
            machine_start_time=machine_start_time,
            prob=0.95,
            coloni_size=coloni_size,
            alpha=alpha,
            beta=beta,
            initial_workload=operator_initial_workload,
            initial_ready_time=operator_ready_time,
            x_start=x_start,
            x_end=x_end,
            a1=a1,
            a2=a2,
            a2_info=a2_info,
            disruption_time=disruption_time,
            init_time_schedule=init_time_schedule,
            init_case1=init_case1,
            init_case1_solution=init_case1_solution,
            init_fair_operator_assignment=init_fair_operator_assignment,
            init_workload_list=init_workload_list,
            init_event_times=init_event_times,
            title=title,
            save=save,
            obj_type=obj_type,
            workload_obj_type=workload_obj_type,
            IR=IR,
            flowtime_type=flowtime_type
        )
        return {"mode": "sustainable", **out}


def reschedule_rework(
        rework_operation,
        case1,
        case1_solution_optimum,
        Time_schedule_P2,
        fair_operator_assignment,
        workload_list,
        event_times,
        EErate,
        carbon_emission_data=None,
        dual_resource=False,
        weight=0.75,
        visualization=True,
        x_start=None,
        x_end=None,
        title=None,
        save=False,
        obj_type="cmax+flowtime",
        workload_obj_type="variance",
        IR=3,
        flowtime_type="average",
        num_iterations1=1500,
        population_size1=75,
        elit_percentage1=0.6,
        num_iterations2=1500,
        population_size2=75,
        elit_percentage2=0.6,
        coloni_size=450,
        alpha=10,
        beta=2):
    """
    Completely Reschedule job shop operations when certain operations need to be reworked.

    Handles rescheduling scenarios where specific operations need to be redone due to quality issues
    or defects using completely reschedule strategy. It: (1) determines disruption time from the
    rework operations' completion time, (2) validates that all rework operations finish at the same
    time, (3) classifies existing operations into A1, A2, and A3, (4) removes finished and ongoing
    operations EXCEPT the rework operations, and (5) performs optimization.
    """
    disruption_time = Time_schedule_P2[rework_operation[0]]['finished_time']

    for operation in rework_operation:
        finish_time = Time_schedule_P2[operation]['finished_time']
        if disruption_time != finish_time:
            print(f"disruption time operation {operation} at {disruption_time} "
                  f"mismatch with finished time {finish_time}")
            print("reschedule paused. delete unfinished operation from rework!")
            return None, None, None, None, None, None, None, None

    if not dual_resource and carbon_emission_data is None:
        raise ValueError(
            "carbon_emission_data harus diberikan jika dual_resource=False "
            "(sustainableJSP_resch digunakan)."
        )

    init = initialize_reschedule(
        case1, case1_solution_optimum, Time_schedule_P2, disruption_time,
        fair_operator_assignment, workload_list, event_times, dual_resource=dual_resource)
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
        done_operations.discard(operation)

    for (j, o) in done_operations:
        case1_reschedule[j - 1][o - 1] = None

    if dual_resource:
        out = dual_resource_JSP(
            case1_reschedule,
            EErate,
            population_size1=population_size1,
            num_iterations1=num_iterations1,
            mutation_threshold1=0.5,
            elit_percentage1=elit_percentage1,
            FSG=generate_feasible_solution2_resch,
            crossover=crossoverNR,
            mutate=mutate1N,
            get_schedule_time=get_schedule_time_AR_resch,
            visualization=visualization,
            reschedule=True,
            machine_start_time=machine_start_time,
            prob=0.95,
            coloni_size=coloni_size,
            alpha=alpha,
            beta=beta,
            initial_workload=operator_initial_workload,
            initial_ready_time=operator_ready_time,
            x_start=x_start,
            x_end=x_end,
            a2=a2,
            a2_info=a2_info,
            disruption_time=disruption_time,
            init_time_schedule=init_time_schedule,
            title=title,
            save=save,
            obj_type=obj_type,
            IR=IR,
            flowtime_type=flowtime_type
        )
        return {"mode": "dual_resource", **out}

    else:
        out = sustainableJSP_resch(
            case1_reschedule,
            carbon_emission_data,
            EErate,
            population_size1=population_size1,
            num_iterations1=num_iterations1,
            mutation_threshold1=0.5,
            elit_percentage1=elit_percentage1,
            FSG=generate_feasible_solution2_resch,
            crossover=crossoverNR,
            mutate=mutate1N,
            get_schedule_time=get_schedule_time_AR_resch,
            num_iterations2=num_iterations2,
            population_size2=population_size2,
            elit_percentage2=elit_percentage2,
            mutation_threshold2=0.5,
            weight=weight,
            visualization=visualization,
            reschedule=True,
            machine_start_time=machine_start_time,
            prob=0.95,
            coloni_size=coloni_size,
            alpha=alpha,
            beta=beta,
            initial_workload=operator_initial_workload,
            initial_ready_time=operator_ready_time,
            x_start=x_start,
            x_end=x_end,
            a1=a1,
            a2=a2,
            a2_info=a2_info,
            disruption_time=disruption_time,
            init_time_schedule=init_time_schedule,
            init_case1=init_case1,
            init_case1_solution=init_case1_solution,
            init_fair_operator_assignment=init_fair_operator_assignment,
            init_workload_list=init_workload_list,
            init_event_times=init_event_times,
            title=title,
            save=save,
            obj_type=obj_type,
            workload_obj_type=workload_obj_type,
            IR=IR,
            flowtime_type=flowtime_type
        )
        return {"mode": "sustainable", **out}


def reschedule_cancelled_job(
        cancel_job_id,
        disruption_time,
        case1,
        case1_solution_optimum,
        Time_schedule_P2,
        fair_operator_assignment,
        workload_list,
        event_times,
        EErate,
        carbon_emission_data=None,
        dual_resource=False,
        weight=0.75,
        visualization=True,
        x_start=None,
        x_end=None,
        title=None,
        save=False,
        obj_type="cmax",
        IR=3,
        flowtime_type="average",
        num_iterations1=1500,
        population_size1=75,
        elit_percentage1=0.6,
        num_iterations2=1500,
        population_size2=75,
        elit_percentage2=0.6,
        coloni_size=450,
        alpha=10,
        beta=2):
    """
    Completely Reschedule job shop operations when certain jobs are cancelled at a disruption time.

    Handles rescheduling scenarios where specific jobs need to be cancelled during schedule execution
    using completely reschedule strategy. It: (1) normalizes job IDs to cancel into a set,
    (2) classifies existing operations into A1, A2, and A3, (3) identifies cancelled operations
    from active operations whose job_id is in the cancellation set, (4) removes cancelled operations
    and done operations from the job data, and (5) performs optimization.
    """
    if isinstance(cancel_job_id, (list, tuple, set)):
        cancel_job_set = set(cancel_job_id)
    else:
        cancel_job_set = {cancel_job_id}

    init = initialize_reschedule(
        case1, case1_solution_optimum, Time_schedule_P2, disruption_time,
        fair_operator_assignment, workload_list, event_times, dual_resource=dual_resource)
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

    if not dual_resource and carbon_emission_data is None:
        raise ValueError(
            "carbon_emission_data harus diberikan jika dual_resource=False "
            "(sustainableJSP_resch digunakan)."
        )

    case1_reschedule = copy.deepcopy(case1)

    active_op = set(a2 + a3)
    cancelled_operations = [op for op in active_op if op[0] in cancel_job_set]

    case1_reschedule = remove_operations(case1_reschedule, cancelled_operations)

    case1_reschedule_None = copy.deepcopy(case1)
    for (j, o) in cancelled_operations:
        case1_reschedule_None[j - 1][o - 1] = None

    print("cancelled job not be removed but set to be None from original job data")
    for i, job in enumerate(case1_reschedule, start=1):
        print(f"Job {i}: {job}")

    a2_new = [op for op in a2 if op not in cancelled_operations]
    print(f'a2_new: {a2_new}')

    done_operations = set(a1 + a2_new)
    print(f'done_operations: {done_operations}')

    for (j, o) in done_operations:
        case1_reschedule[j - 1][o - 1] = None

    done_operations_None = set(a1 + a2)
    for (j, o) in done_operations_None:
        case1_reschedule_None[j - 1][o - 1] = None

    print(f'case1_reschedule= {case1_reschedule}')
    for i, job in enumerate(case1_reschedule, start=1):
        print(f"Job {i}: {job}")

    print(f'case1_reschedule_None= {case1_reschedule_None}')
    for i, job in enumerate(case1_reschedule_None, start=1):
        print(f"Job {i}: {job}")

    if dual_resource:
        out = dual_resource_JSP(
            case1_reschedule_None,
            EErate,
            population_size1=population_size1,
            num_iterations1=num_iterations1,
            mutation_threshold1=0.5,
            elit_percentage1=elit_percentage1,
            FSG=generate_feasible_solution2_resch,
            crossover=crossoverNR,
            mutate=mutate1N,
            get_schedule_time=get_schedule_time_AR_resch,
            visualization=visualization,
            reschedule=True,
            machine_start_time=machine_start_time,
            prob=0.95,
            coloni_size=coloni_size,
            alpha=alpha,
            beta=beta,
            initial_workload=operator_initial_workload,
            initial_ready_time=operator_ready_time,
            x_start=x_start,
            x_end=x_end,
            a2=a2,
            a2_info=a2_info,
            disruption_time=disruption_time,
            init_time_schedule=init_time_schedule,
            title=title,
            save=save,
            obj_type=obj_type,
            IR=IR,
            flowtime_type=flowtime_type
        )
        return {"mode": "dual_resource", **out}

    else:
        out = sustainableJSP_resch(
            case1_reschedule_None,
            carbon_emission_data,
            EErate,
            population_size1=population_size1,
            num_iterations1=num_iterations1,
            mutation_threshold1=0.5,
            elit_percentage1=elit_percentage1,
            FSG=generate_feasible_solution2_resch,
            crossover=crossoverNR,
            mutate=mutate1N,
            get_schedule_time=get_schedule_time_AR_resch,
            num_iterations2=num_iterations2,
            population_size2=population_size2,
            elit_percentage2=elit_percentage2,
            mutation_threshold2=0.5,
            weight=weight,
            visualization=visualization,
            reschedule=True,
            machine_start_time=machine_start_time,
            prob=0.95,
            coloni_size=coloni_size,
            alpha=alpha,
            beta=beta,
            initial_workload=operator_initial_workload,
            initial_ready_time=operator_ready_time,
            x_start=x_start,
            x_end=x_end,
            a1=a1,
            a2=a2,
            a2_info=a2_info,
            disruption_time=disruption_time,
            init_time_schedule=init_time_schedule,
            init_case1=init_case1,
            init_case1_solution=init_case1_solution,
            init_fair_operator_assignment=init_fair_operator_assignment,
            init_workload_list=init_workload_list,
            init_event_times=init_event_times,
            title=title,
            save=save,
            obj_type=obj_type,
            IR=IR,
            flowtime_type=flowtime_type
        )
        return {"mode": "sustainable", **out}
