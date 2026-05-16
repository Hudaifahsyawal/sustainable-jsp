from __future__ import annotations

import copy

from sustainable_jsp.core.schedule import get_schedule_time_AR_resch
from sustainable_jsp.core.performance_matrix import calculate_time_objective
from sustainable_jsp.algorithms.workload import AWBO
from sustainable_jsp.scheduling.sustainable import sustainableJSP_resch
from sustainable_jsp.scheduling.dual_resource import dual_resource_JSP


def calculate_instability(strain1, strain2):
    """
    Calculate the mismatch count between two sequences.
    A mismatch is counted when elements appear in different order.
    """
    mismatch = 0

    common_elements = [x for x in strain1 if x in strain2]
    aligned_strain2 = [x for x in strain2 if x in strain1]

    for a, b in zip(common_elements, aligned_strain2):
        if a != b:
            mismatch += 1

    return mismatch


def classify_operation(Time_schedule_P2, current_time):
    """
    Classify operations into three possible sets based on status:
    a1 = finished operations (finished_time <= current_time)
    a2 = ongoing operations (start_time < current_time < finished_time)
    a3 = unprocessed operations (start_time >= current_time)
    """
    a1 = []
    a2 = []
    a3 = []

    for operation, times in Time_schedule_P2.items():
        start_time = times['start_time']
        finish_time = times['finished_time']

        if finish_time <= current_time:
            a1.append(operation)
        elif start_time < current_time < finish_time:
            a2.append(operation)
        else:
            a3.append(operation)

    return a1, a2, a3


def remove_operations(case, remove_set):
    """
    Remove operations from case that appear in sets a1 and a2.
    """
    updated_case = []

    for job_index, operations in enumerate(case, start=1):
        filtered_ops = [op for op_index, op in enumerate(operations, start=1) if (job_index, op_index) not in remove_set]
        updated_case.append(filtered_ops)

    return updated_case


def get_machine_start_time(case1, case1_solution_optimum, disruption_time, a2, Time_schedule_P2):
    """
    Determine the machine start time at the time of disruption.
    """
    n_machines = len(case1_solution_optimum)
    machine_start_time = {machine: disruption_time for machine in range(1, n_machines + 1)}
    for operation in a2:
        job_id, operation_id = operation
        machine_id = case1[job_id - 1][operation_id - 1][0]
        finished_time = Time_schedule_P2[operation]["finished_time"]
        machine_start_time[machine_id] = finished_time
    return machine_start_time


def get_initial_workload(workload_list, event_times, disruption_time, operator_on_progress):
    """
    Determine the workload for each operator at a specific disruption time.
    """
    workload_at_disruption = {}

    for operator_id in event_times:
        times = event_times[operator_id]
        workloads = workload_list[operator_id]

        idx = None
        for i, t in enumerate(times):
            if t == disruption_time:
                idx = i
                break
            elif t > disruption_time:
                if operator_id not in operator_on_progress:
                    idx = i - 1
                else:
                    idx = i
                break
        else:
            idx = len(times) - 1

        workload_at_disruption[operator_id] = workloads[idx]

    return workload_at_disruption


def determine_operator_ready_time(fair_operator_assignment, a2, Time_schedule_P2, disruption_time):
    """
    Determine the ready time for each operator after disruption.
    """
    operator_ready_time = {}
    operator_on_progress = []
    for operator_id, tasks in fair_operator_assignment.items():
        matched_op = next((op for op in tasks if op in a2), None)
        if matched_op:
            ready_time = Time_schedule_P2[matched_op]['finished_time']
            operator_on_progress.append(operator_id)
        else:
            ready_time = disruption_time
        operator_ready_time[operator_id] = ready_time

    return operator_ready_time, operator_on_progress


def initialize_reschedule(
    case1: list,
    case1_solution_optimum: list,
    Time_schedule_P2: dict,
    disruption_time: float,
    fair_operator_assignment: dict,
    workload_list: dict,
    event_times: dict,
    dual_resource: bool = False,
) -> dict:
    """
    Initialize rescheduling state by classifying operations and computing resource availability at disruption time.

    Prepares all inputs required for rescheduling algorithms: divides operations into finished (A1),
    ongoing (A2), and unprocessed (A3); computes when each machine and operator becomes available
    after the disruption; and records remaining duration and assignments for ongoing operations.
    """
    a1, a2, a3 = classify_operation(Time_schedule_P2, disruption_time)
    print("-" * 20)
    print(f"\n {'-' * 20}\nRescheduling Initialization\n{'-' * 20}")
    print(f"a1= {a1}")
    print(f"a2= {a2}")
    print(f"a3= {a3}")

    machine_start_time = get_machine_start_time(case1, case1_solution_optimum, disruption_time, a2, Time_schedule_P2)
    print(f"\nmachine_start_time at disruption_time")
    for key, data in machine_start_time.items():
        print(f"machine {key} start_time = {machine_start_time[key]}")

    operator_ready_time, operator_on_progress = determine_operator_ready_time(
        fair_operator_assignment, a2, Time_schedule_P2, disruption_time)
    print(f"\noperator_ready_time at disruption_time")
    for key, data in operator_ready_time.items():
        print(f"operator {key} ready_time = {operator_ready_time[key]}")
    print(f"\noperator_on_progress = {operator_on_progress}")

    operator_initial_workload = get_initial_workload(
        workload_list, event_times, disruption_time, operator_on_progress)
    print("\ninitial workload at disruption time")
    for key, data in operator_initial_workload.items():
        print(f"operator {key} workload = {operator_initial_workload[key]}")

    a2_info = {}

    a2_set = set(a2)
    opA2_to_operator = {
        task: opr_id
        for opr_id, ops in fair_operator_assignment.items()
        for task in ops
        if task in a2_set
    }

    for (j, op) in a2:
        start = Time_schedule_P2[(j, op)]['start_time']
        finish = Time_schedule_P2[(j, op)]['finished_time']
        rem = Time_schedule_P2[(j, op)]['duration']
        original_duration = Time_schedule_P2[(j, op)].get('original_duration', case1[j - 1][op - 1][1])
        m_id, _ = case1[j - 1][op - 1]
        if not dual_resource:
            sl = Time_schedule_P2[(j, op)].get('speed_level', 1)
        else:
            sl = None
        opr_id = opA2_to_operator.get((j, op))
        a2_info[(j, op)] = {
            "start_time": start,
            "duration": rem,
            "finished_time": finish,
            "original_duration": original_duration,
            "machine_id": m_id,
            "speed_level": sl,
            "operator_id": opr_id
        }

    a1a2 = a1 + a2
    a1a2_set = set(a1a2)

    init_case1 = copy.deepcopy(case1)
    for j, job in enumerate(init_case1, start=1):
        init_case1[j - 1] = [
            op for op_index, op in enumerate(job, start=1)
            if (j, op_index) in a1a2_set
        ]

    init_case1_solution = copy.deepcopy(case1_solution_optimum)
    for m_id, ops in init_case1_solution.items():
        init_case1_solution[m_id] = [op for op in ops if op in a1a2_set]

    init_time_schedule = {
        op: dict(Time_schedule_P2[op])
        for op in a1a2
    }

    init_fair_operator_assignment = {
        opr_id: [t for t in tasks if t in a1a2_set]
        for opr_id, tasks in fair_operator_assignment.items()
    }

    init_event_times = {}
    init_workload_list = {}
    for opr_id in event_times:
        n = 2 * len(init_fair_operator_assignment.get(opr_id, []))
        times = event_times[opr_id]
        workloads = workload_list[opr_id]
        init_event_times[opr_id] = times[:n]
        init_workload_list[opr_id] = workloads[:n]

    return {
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "machine_start_time": machine_start_time,
        "operator_ready_time": operator_ready_time,
        "operator_on_progress": operator_on_progress,
        "operator_initial_workload": operator_initial_workload,
        "a2_info": a2_info,
        "init_case1": init_case1,
        "init_case1_solution": init_case1_solution,
        "init_time_schedule": init_time_schedule,
        "init_fair_operator_assignment": init_fair_operator_assignment,
        "init_workload_list": init_workload_list,
        "init_event_times": init_event_times,
    }
