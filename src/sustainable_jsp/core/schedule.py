import math
import random
import copy


# ---------------------------------------------------------------------------
# Feasible solution generators
# ---------------------------------------------------------------------------

def generate_feasible_solution1(jobs_data):
    """
    Generate a feasible solution for a Job Shop Scheduling Problem.

    Creates a machine-based schedule where operations are ordered by their operation
    index (operation ID) within each machine. Operations with the same operation index
    are grouped together, and the sequence of jobs within each group is randomly
    shuffled. This ensures that all first operations across different jobs are scheduled
    before any second operations, maintaining job precedence constraints while
    introducing randomness in job ordering.
    """
    machine_order = {}

    for job_index, job in enumerate(jobs_data, start=1):
        for operation_index, (machine, duration) in enumerate(job, start=1):
            if machine not in machine_order:
                machine_order[machine] = []
            machine_order[machine].append((job_index, operation_index))

    for machine in machine_order:
        grouped = {}
        for job, operation in machine_order[machine]:
            if operation not in grouped:
                grouped[operation] = []
            grouped[operation].append((job, operation))

        for operation in grouped:
            random.shuffle(grouped[operation])

        machine_order[machine] = [pair for operation in sorted(grouped) for pair in grouped[operation]]
    machine_order_sorted = {machine: machine_order[machine] for machine in sorted(machine_order.keys())}
    return machine_order_sorted


def generate_feasible_solution1_resch(jobs_data):
    """
    Generate a feasible solution for rescheduling scenarios
    based on operation-index grouping (solution1 style).

    This function is a rescheduling-aware version of
    generate_feasible_solution1. Operations that are None
    (cancelled or already completed) are skipped safely.

    Args:
        jobs_data (list): Job data where each job is a list of
                          operations (machine, duration) or None.

    Returns:
        dict: Machine-based operation sequence
              {machine: [(job_index, operation_index), ...]}
    """
    machine_order = {}

    for job_index, job in enumerate(jobs_data, start=1):
        for operation_index, operation in enumerate(job, start=1):
            if operation is None:
                continue

            machine, duration = operation

            if machine not in machine_order:
                machine_order[machine] = []

            machine_order[machine].append((job_index, operation_index))

    for machine in machine_order:
        grouped = {}
        for job, operation in machine_order[machine]:
            grouped.setdefault(operation, []).append((job, operation))

        for operation in grouped:
            random.shuffle(grouped[operation])

        machine_order[machine] = [
            pair
            for operation in sorted(grouped)
            for pair in grouped[operation]
        ]

    machine_order_sorted = {
        machine: machine_order[machine]
        for machine in sorted(machine_order)
    }

    return machine_order_sorted


def generate_feasible_solution2(jobs_data):
    """
    Generate a feasible solution for a Job Shop Scheduling Problem using a
    dispatching rule approach.

    Creates a machine-based schedule by iteratively selecting and scheduling
    operations that are ready to be processed. The algorithm uses a dispatching
    list to track which operations are available at each step. Initially, only
    the first operation of each job is available. As operations are scheduled,
    the next operation of the same job becomes available, ensuring job precedence
    constraints are maintained.

    This approach uses an iterative greedy selection process where operations are
    scheduled one at a time as they become available, providing flexibility in
    the solution structure and allowing operations from different jobs to be
    interleaved on machines.
    """
    machine_order = {}

    n_jobs = len(jobs_data)
    n_operations = max([len(job) for job in jobs_data])
    dispatchlist = [[True if operation_index == 0 else False for operation_index in range(n_operations)] for _ in range(n_jobs)]

    def create_operation_list(dispatchlist):
        operation_list = []
        for job_index, job in enumerate(dispatchlist):
            for operation_index, is_possible in enumerate(job):
                if is_possible:
                    operation_list.append((job_index + 1, operation_index + 1))
        return operation_list

    def select_random_operation(operation_list):
        if not operation_list:
            return None
        return random.choice(operation_list)

    while any(any(row) for row in dispatchlist):
        operation_list = create_operation_list(dispatchlist)
        selected_operation = select_random_operation(operation_list)

        job_index, operation_index = selected_operation
        machine_index = jobs_data[job_index - 1][operation_index - 1][0]
        if machine_index not in machine_order:
            machine_order[machine_index] = []
        machine_order[machine_index].append(selected_operation)

        dispatchlist[job_index - 1][operation_index - 1] = False
        if operation_index < len(jobs_data[job_index - 1]):
            dispatchlist[job_index - 1][operation_index] = True
    machine_order_sorted = {machine: machine_order[machine] for machine in sorted(machine_order.keys())}
    return machine_order_sorted


def generate_feasible_solution2_resch(jobs_data):
    """
    Generate a feasible solution for rescheduling scenarios using a dispatching rule approach.

    Creates a machine-based schedule by iteratively selecting and scheduling operations
    that are ready to be processed. This function is specifically designed for
    rescheduling scenarios where some operations may be None (cancelled or already
    completed). None operations are automatically skipped and treated as completed.

    The algorithm uses a dispatching list to track which operations are available at
    each step. Initially, only the first non-None operation of each job is available.
    As operations are scheduled, the next non-None operation of the same job becomes
    available, ensuring job precedence constraints are maintained while skipping
    cancelled operations.
    """
    machine_order = {}
    n_jobs = len(jobs_data)
    n_operations = max(len(job) for job in jobs_data)

    dispatchlist = []
    for job in jobs_data:
        row = [False] * n_operations
        for idx, op in enumerate(job):
            if op is not None:
                row[idx] = True
                break
        dispatchlist.append(row)

    def create_operation_list(dispatchlist):
        ops = []
        for j, row in enumerate(dispatchlist):
            for o, active in enumerate(row):
                if active and jobs_data[j][o] is not None:
                    ops.append((j + 1, o + 1))
        return ops

    while any(any(r) for r in dispatchlist):
        op_list = create_operation_list(dispatchlist)
        if not op_list:
            break
        j, o = random.choice(op_list)
        m, _ = jobs_data[j - 1][o - 1]
        machine_order.setdefault(m, []).append((j, o))

        dispatchlist[j - 1][o - 1] = False
        for k in range(o, len(jobs_data[j - 1])):
            if jobs_data[j - 1][k] is not None:
                dispatchlist[j - 1][k] = True
                break

    return {m: machine_order[m] for m in sorted(machine_order)}


# ---------------------------------------------------------------------------
# Lower bounds
# ---------------------------------------------------------------------------

def calculate_cmax_lower_bound(jobs_data, machine_assignment):
    """
    Calculate a lower bound for the makespan (Cmax) of a job shop schedule.

    Computes a theoretical lower bound by summing the total duration of all operations
    assigned to each machine and taking the maximum. This bound assumes that operations
    on the same machine could be processed in parallel (which is not possible in
    reality), making it a lower bound that the actual makespan cannot be less than.
    """
    machine_durations = {}

    for machine_id, operations in machine_assignment.items():
        total_duration = 0
        for job_id, op_id in operations:
            job_index = job_id - 1
            op_index = op_id - 1
            duration = jobs_data[job_index][op_index][1]
            total_duration += duration
        machine_durations[machine_id] = total_duration

    max_duration = max(machine_durations.values())
    return max_duration


def calculate_cmax_lower_bound_2(TimeSchedule_P2, Solution_P1):
    """
    Calculate a lower bound for the makespan (Cmax) using Phase 2 schedule data.

    Computes a theoretical lower bound by summing the total duration of all operations
    assigned to each machine using durations from TimeSchedule_P2 (which may include
    speed level adjustments) and taking the maximum. This function is similar to
    calculate_cmax_lower_bound(), but uses adjusted durations from Phase 2 scheduling
    instead of raw durations from jobs_data.
    """
    machine_total_duration = {}

    for machine_id, operation_list in Solution_P1.items():
        total_duration = 0
        for job_id, op_id in operation_list:
            duration = TimeSchedule_P2[(job_id, op_id)]['duration']
            total_duration += duration
        machine_total_duration[machine_id] = total_duration

    LB_Cmax = max(machine_total_duration.values())
    return LB_Cmax


# ---------------------------------------------------------------------------
# Schedule decoders
# ---------------------------------------------------------------------------

def get_schedule_time_AR_resch(
        jobs_data,
        machine_order,
        reschedule=False,
        machine_start_time=None,
        a2=None,
        a2_info=None,
        disruption_time=None):
    """
    Decode a solution into a time schedule with adaptive repair for rescheduling scenarios.

    This function generates a time schedule from a machine order solution, specifically
    designed for rescheduling scenarios where some operations may have been cancelled,
    modified, or were in progress at a disruption time. The function includes adaptive
    repair mechanisms to handle infeasible states that may arise during rescheduling.
    """
    if machine_start_time is None:
        machine_start_time = {}

    a2 = a2 or []
    a2_info = a2_info or {}

    if isinstance(machine_order, dict) and len(machine_order) > 0:
        n_machines = max(machine_order.keys())
    else:
        mset = set()
        for job in jobs_data:
            for op in job:
                if op is None:
                    continue
                mset.add(op[0])
        n_machines = max(mset) if mset else 0

    n_jobs = len(jobs_data)
    n_operations = max(len(job) for job in jobs_data) if jobs_data else 0

    if not isinstance(machine_order, dict):
        raise TypeError(f"machine_order must be a dict, got {type(machine_order)}")

    for m in range(1, n_machines + 1):
        if m not in machine_order:
            machine_order[m] = []

    if not reschedule:
        machine_available_time = {m: 0.0 for m in range(1, n_machines + 1)}
    else:
        machine_available_time = copy.deepcopy(machine_start_time)
        for m in range(1, n_machines + 1):
            machine_available_time.setdefault(m, 0.0)

    dispatchlist = []
    for job in jobs_data:
        row = [False] * n_operations
        for idx, op in enumerate(job):
            if op is not None:
                row[idx] = True
                break
        dispatchlist.append(row)

    schedule_time = {}
    for j, job in enumerate(jobs_data, start=1):
        for o, op in enumerate(job, start=1):
            if op is None:
                continue
            m, dur = op
            schedule_time[(j, o)] = {"start_time": math.nan, "duration": dur, "finished_time": math.nan, "original_duration": dur, "machine_id": m}

    start_machine_index = 1
    machine_order_dummy = copy.deepcopy(machine_order)

    visited_machines = set()

    while any(any(row) for row in dispatchlist):
        machine_index = start_machine_index

        if machine_index not in machine_order_dummy:
            machine_order_dummy[machine_index] = []
        if machine_index not in machine_order:
            machine_order[machine_index] = []

        if machine_index in visited_machines:
            sequence_fixed = False

            while not sequence_fixed:
                if not machine_order_dummy.get(machine_index, []):
                    machine_index = (machine_index % n_machines) + 1
                    continue

                if len(machine_order_dummy[machine_index]) == 1:
                    machine_index = (machine_index % n_machines) + 1
                    continue

                for i in range(1, len(machine_order_dummy[machine_index])):
                    job_index, operation_index = machine_order_dummy[machine_index][i]

                    if dispatchlist[job_index - 1][operation_index - 1]:
                        machine_order_dummy[machine_index].insert(0, machine_order_dummy[machine_index].pop(i))

                        front_index = len(machine_order.get(machine_index, [])) - len(machine_order_dummy[machine_index])
                        if front_index < 0:
                            front_index = 0
                        machine_order[machine_index] = machine_order.get(machine_index, [])[:front_index] + machine_order_dummy[machine_index]

                        visited_machines.clear()
                        sequence_fixed = True
                        break

                if not sequence_fixed:
                    machine_index = (machine_index % n_machines) + 1
                else:
                    break

        if not machine_order_dummy.get(machine_index, []):
            start_machine_index = (start_machine_index % n_machines) + 1
            visited_machines.add(machine_index)
            continue

        job_index, operation_index = machine_order_dummy[machine_index][0]

        if dispatchlist[job_index - 1][operation_index - 1]:
            if operation_index == 1:
                prev_finished_time = 0.0
            else:
                pred_idx = operation_index - 1
                pred_key = (job_index, pred_idx)

                pred_is_none = (jobs_data[job_index - 1][pred_idx - 1] is None)
                pred_is_a2 = (pred_key in a2)

                if pred_is_none and not pred_is_a2:
                    prev_finished_time = disruption_time if disruption_time is not None else 0.0
                elif pred_is_none and pred_is_a2:
                    prev_finished_time = a2_info[(job_index, pred_idx)]["finished_time"]
                else:
                    prev_finished_time = schedule_time[(job_index, pred_idx)]["finished_time"]

            start_time = max(machine_available_time.get(machine_index, 0.0), prev_finished_time)

            duration = schedule_time[(job_index, operation_index)]["duration"]
            finished_time = start_time + duration

            schedule_time[(job_index, operation_index)]["start_time"] = start_time
            schedule_time[(job_index, operation_index)]["finished_time"] = finished_time

            machine_available_time[machine_index] = finished_time

            dispatchlist[job_index - 1][operation_index - 1] = False

            if operation_index < len(jobs_data[job_index - 1]):
                dispatchlist[job_index - 1][operation_index] = True

            start_machine_index = (start_machine_index % n_machines) + 1

            machine_order_dummy[machine_index].pop(0)
            visited_machines.clear()

        else:
            start_machine_index = (start_machine_index % n_machines) + 1
            visited_machines.add(machine_index)

    return schedule_time


def get_schedule_time_BL(jobs_data, machine_order, reschedule=False, machine_start_time={}):
    """
    Decode a solution into a time schedule using a baseline approach without adaptive repair.

    This function generates a time schedule from a machine order solution using a
    straightforward scheduling approach. Unlike get_schedule_time_AR_resch(), this function
    does not include adaptive repair mechanisms. If the solution is infeasible
    (operations cannot be scheduled due to precedence violations or circular dependencies),
    the function returns None instead of attempting to repair the solution.
    """
    n_jobs = len(jobs_data)
    n_machines = len(machine_order)
    n_operations = max([len(job) for job in jobs_data])

    dispatchlist = []
    for job in jobs_data:
        if not job:
            dispatchlist.append([False for operation_index in range(n_operations)])
        else:
            dispatchlist.append([True if operation_index == 0 else False for operation_index in range(n_operations)])

    schedule_time = {}
    for job_index, job in enumerate(jobs_data, start=1):
        for operation_index, (machine, duration) in enumerate(job, start=1):
            schedule_time[(job_index, operation_index)] = {
                "start_time": math.nan,
                "duration": duration,
                "finished_time": math.nan
            }

    if not reschedule:
        machine_available_time = {machine: 0 for machine in range(1, n_machines + 1)}
    else:
        machine_available_time = copy.deepcopy(machine_start_time)

    start_machine_index = 1
    machine_order_dummy = copy.deepcopy(machine_order)

    visited_machines = set()

    while any(any(row) for row in dispatchlist):
        machine_index = start_machine_index

        if machine_index in visited_machines:
            print("Infeasible")
            return None

        if not machine_order_dummy[machine_index]:
            start_machine_index = (start_machine_index % n_machines) + 1
            continue

        job_index, operation_index = machine_order_dummy[machine_index][0]

        if dispatchlist[job_index - 1][operation_index - 1]:
            prev_finished_time = (
                schedule_time[(job_index, operation_index - 1)]["finished_time"]
                if operation_index > 1
                else 0
            )
            start_time = max(machine_available_time[machine_index], prev_finished_time)

            duration = schedule_time[(job_index, operation_index)]["duration"]
            finished_time = start_time + duration

            schedule_time[(job_index, operation_index)]["start_time"] = start_time
            schedule_time[(job_index, operation_index)]["finished_time"] = finished_time

            machine_available_time[machine_index] = finished_time

            dispatchlist[job_index - 1][operation_index - 1] = False

            if operation_index < len(jobs_data[job_index - 1]):
                dispatchlist[job_index - 1][operation_index] = True

            start_machine_index = (start_machine_index % n_machines) + 1

            machine_order_dummy[machine_index].pop(0)

            visited_machines.clear()

        else:
            start_machine_index = (start_machine_index % n_machines) + 1
            visited_machines.add(machine_index)

    return schedule_time
