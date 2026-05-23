from __future__ import annotations

import random
import copy

import matplotlib.pyplot as plt
import numpy as np

from sustainable_jsp.physiology.anfis import ANFIS


def classify_operation(Time_schedule_P2, current_time):
    """
    Classify operations into three possible sets based on their status:
    a1 = finish operations
    a2 = ongoing operations
    a3 = unprocessed operations
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


def generate_operator_data(number_operator: int, number_machine: int) -> dict[int, dict[str, int | float | list[int]]]:
    """
    Generate synthetic operator data with physiological characteristics using statistical distributions.

    This function creates a dataset of operator physiological attributes (age, weight, height, BMI,
    resting heart rate) and operating heart rates for each machine. The data is generated using
    normal distributions with predefined means and standard deviations, representing realistic
    operator populations.
    """
    operator_data = {}

    for i in range(1, number_operator + 1):
        age = int(np.random.normal(34.8, 8.8))
        weight = round(np.random.normal(78.4, 10.3), 1)
        height = round(np.random.normal(174, 8), 1)
        bmi = round(weight / ((height / 100) ** 2), 1)
        hr_rest = int(np.random.normal(73.8, 9.4))
        hr_op = [hr_rest + random.randint(0, 35) for _ in range(number_machine)]

        operator_data[i] = {
            'age': age,
            'weight': weight,
            'height': height,
            'bmi': bmi,
            'hr_rest': hr_rest,
            'hr_op': hr_op
        }

    return operator_data


def generate_EErate(num_operator: int, num_machine: int, min_val: float, max_val: float) -> list[list[float]]:
    """
    Generate synthetic Energy Expenditure (EE) rates for each operator-machine combination.
    """
    outer_list = []
    for _ in range(num_operator):
        inner_list = []
        for _ in range(num_machine):
            random_value = round(random.uniform(min_val, max_val), 1)
            inner_list.append(random_value)
        outer_list.append(inner_list)
    return outer_list


def calculate_EErate(dict_operator: dict[int, dict[str, int | float | list[int]]]) -> list[list[float]]:
    """
    Calculate Energy Expenditure (EE) rates for operators using ANFIS model and physiological data.
    """
    EE_list = []
    for opr, data in dict_operator.items():
        ANFISmodel = ANFIS(data["age"], data["weight"], data["height"], data["hr_rest"])
        EE = [round(float(5 * data["weight"] * ANFISmodel.model(hr) / 1000), 1) for hr in data["hr_op"]]
        EE_list.append(EE)
    return EE_list


def workload_statistic(data):
    """
    Calculate mean and variance from a dictionary of numerical values.
    """
    values = list(data.values())
    mean = np.mean(values)
    maximum = np.max(values)
    variance = np.var(values, ddof=0)
    std_dev = np.std(values, ddof=0)
    cv = std_dev / mean if mean != 0 else np.nan

    return round(float(mean), 2), round(float(variance), 2), round(float(maximum), 2), round(float(std_dev), 2), round(float(cv), 2)


def calculate_cv(data):
    """
    Calculate mean, standard deviation, and coefficient of variation (CV)
    from a dictionary of numerical values.
    """
    values = list(data.values())
    mean = np.mean(values)
    std_dev = np.std(values, ddof=0)
    cv = std_dev / mean if mean != 0 else np.nan

    return round(float(mean), 2), round(float(std_dev), 2), round(float(cv), 4)


def calculate_workload(EErate, operator_id, machine_id, duration):
    """
    Calculate workload of an operation done by a specific operator.
    """
    return round(EErate[operator_id - 1][machine_id - 1] * duration, 2)


def calculate_selection_probabilities(data, alpha=10, beta=2):
    """
    Calculate selection probabilities for operators using the given formula.
    """
    import math

    weighted_scores = []
    for operator_id, current_workload, task_workload in data:
        safe_current_workload = max(current_workload, 1e-10)
        safe_task_workload = max(task_workload, 1e-10)
        score = (1 / safe_current_workload) ** alpha * (1 / safe_task_workload) ** beta
        weighted_scores.append((operator_id, score))

    total_score = sum(score for _, score in weighted_scores)
    probabilities = {operator_id: score / total_score for operator_id, score in weighted_scores}

    return probabilities


def select_operator(probabilities):
    """
    Select one operator_id based on selection probabilities.
    """
    operator_ids = list(probabilities.keys())
    probs = list(probabilities.values())
    selected = random.choices(operator_ids, weights=probs, k=1)[0]
    return selected


def distribute_workload(
    EErate: list[list[float]],
    jobs_data: list[list[tuple[int, int] | None]],
    schedule_jobs_phase2: dict[tuple[int, int], dict[str, float]],
    solution_phase1: dict[int, list[tuple[int, int]]],
    prob: float = 0.95,
    reschedule: bool = False,
    initial_workload: dict[int, float] | None = None,
    initial_ready_time: dict[int, float] | None = None
) -> dict[int, list[tuple[int, int]]]:
    """
    Distribute workload fairly among operators using a probabilistic greedy assignment algorithm.
    """
    if initial_workload is None:
        initial_workload = {}
    if initial_ready_time is None:
        initial_ready_time = {}

    sorted_operations = sorted(schedule_jobs_phase2.items(), key=lambda x: (x[1]['start_time'], x[1]['duration']))

    operator_assignmentsList = {}
    operator_last_finish_time = {}
    current_workload = {}

    for i in range(len(solution_phase1)):
        operator_id = i + 1
        operator_assignmentsList[operator_id] = []
        if not reschedule:
            operator_last_finish_time[operator_id] = 0
            current_workload[operator_id] = 0

    if reschedule:
        operator_last_finish_time = initial_ready_time.copy()
        current_workload = initial_workload.copy()

    for (job_id, operation_id), times in sorted_operations:
        start_time, duration = times['start_time'], times['duration']
        machine_id = jobs_data[job_id - 1][operation_id - 1][0]

        available_operators = [operator_id for operator_id, finish_time in operator_last_finish_time.items() if finish_time <= start_time]

        if len(available_operators) == 1:
            chosen_operator = available_operators[0]
        else:
            rand_value = random.random()

            if rand_value < prob:
                options = [
                    (operator_id, current_workload[operator_id], calculate_workload(EErate, operator_id, machine_id, duration))
                    for operator_id in available_operators
                ]
                sorted_options = sorted(options, key=lambda x: (x[1], x[2]))
                chosen_operator = sorted_options[0][0]
            else:
                chosen_operator = random.choice(available_operators)

        operator_assignmentsList[chosen_operator].append((job_id, operation_id))
        operator_last_finish_time[chosen_operator] = times['finished_time']
        current_workload[chosen_operator] += calculate_workload(EErate, chosen_operator, machine_id, duration)

    return operator_assignmentsList


def cumulative_workload(
    operator_operationlist: dict[int, list[tuple[int, int]]],
    EErate: list[list[float]],
    schedule_jobs: dict[tuple[int, int], dict[str, float]],
    jobs_data: list[list[tuple[int, int] | None]],
    visualization: bool = True,
    reschedule: bool = False,
    initial_workload: dict[int, float] | None = None,
    a2_info: dict[tuple[int, int], dict[str, float]] | None = None
) -> tuple[dict[int, list[float]], dict[int, list[float]], dict[int, float]]:
    """
    Calculate cumulative energy expenditure (workload) over time for each operator.
    """
    if initial_workload is None:
        initial_workload = {}

    workload_list = {}
    event_times = {}
    current_workload = {} if not reschedule else copy.deepcopy(initial_workload)

    for operator_id, operation_list in operator_operationlist.items():
        if operator_id not in event_times:
            event_times[operator_id] = []
        for operation in operation_list:
            operation_info = schedule_jobs[operation]
            event_times[operator_id].append(operation_info['start_time'])
            event_times[operator_id].append(operation_info['finished_time'])

        if operator_id not in workload_list:
            workload_list[operator_id] = []
        if operator_id not in current_workload:
            current_workload[operator_id] = 0

        for idx in range(len(event_times[operator_id])):
            if idx % 2 == 0:
                workload_list[operator_id].append(current_workload[operator_id])
            else:
                operation = operation_list[idx // 2]
                machine_id = jobs_data[operation[0] - 1][operation[1] - 1][0]
                start_time = event_times[operator_id][idx - 1]
                finish_time = event_times[operator_id][idx]
                duration = finish_time - start_time
                additional_workload = calculate_workload(EErate, operator_id, machine_id, duration)
                workload_list[operator_id].append(current_workload[operator_id] + additional_workload)
                current_workload[operator_id] += additional_workload

    print("Done.\nCummulative Workload per operator:")
    for operator_id, total in current_workload.items():
        print(f"Operator {operator_id}: {round(total, 2)} kcal")

    if visualization is True:
        plt.figure(figsize=(10, 5))
        for operator_id, values in workload_list.items():
            plt.plot(event_times[operator_id], values, marker='o', label=f'Operator {operator_id}')
        plt.title('Cumulative workload per operator', fontsize=14)
        plt.xlabel('Time horizon (minutes)', fontsize=12)
        plt.ylabel('Total Workload (kcal)', fontsize=12)
        max_time = max(max(times) for times in event_times.values())
        plt.xticks(np.arange(0, max_time + 10, 20))
        plt.legend()
        plt.grid()
        plt.show()

    return workload_list, event_times, current_workload


def baseline_assignment(A2, fairworkload, solutionphase1, RESsolutionphase1):
    """
    Assign resource operations to operators using a baseline heuristic approach.
    """
    assigned_ops = set()
    operator_res_assignment = {op_id: [] for op_id in fairworkload}

    for op in A2:
        operator_id = next((op_id for op_id, ops in fairworkload.items() if op in ops), None)
        print(f"operator id = {operator_id}")
        if operator_id is None:
            continue

        machine_id = next((m_id for m_id, ops in solutionphase1.items() if op in ops), None)
        print(f"machine id = {machine_id}")
        if machine_id is None:
            continue

        for res_op in RESsolutionphase1.get(machine_id, []):
            if res_op not in assigned_ops:
                operator_res_assignment[operator_id].append(res_op)
                assigned_ops.add(res_op)

    unassigned_ops = [op for ops in RESsolutionphase1.values() for op in ops if op not in assigned_ops]
    unassigned_operators = [op_id for op_id, ops in operator_res_assignment.items() if
                            not any(a2 in fairworkload[op_id] for a2 in A2)]

    for op in unassigned_ops:
        if not unassigned_operators:
            target_operator = random.choice(list(operator_res_assignment.keys()))
        else:
            target_operator = random.choice(unassigned_operators)
        operator_res_assignment[target_operator].append(op)
        assigned_ops.add(op)

    return operator_res_assignment


def Greedy_distribute_workload(EErate, jobs_data, schedule_jobs_phase2, solution_phase1, prob=0.95, reschedule=False,
                                initial_workload=None, initial_ready_time=None):
    """
    Distribute workload fairly among operators using a deterministic greedy assignment algorithm.
    """
    if initial_workload is None:
        initial_workload = {}
    if initial_ready_time is None:
        initial_ready_time = {}

    sorted_operations = sorted(schedule_jobs_phase2.items(), key=lambda x: (x[1]['start_time'], x[1]['duration']))

    operator_assignmentsList = {}
    operator_last_finish_time = {}
    current_workload = {}

    for i in range(len(solution_phase1)):
        operator_id = i + 1
        operator_assignmentsList[operator_id] = []
        if not reschedule:
            operator_last_finish_time[operator_id] = 0
            current_workload[operator_id] = 0

    if reschedule:
        operator_last_finish_time = initial_ready_time
        current_workload = initial_workload

    for (job_id, operation_id), times in sorted_operations:
        start_time, duration = times['start_time'], times['duration']
        machine_id = jobs_data[job_id - 1][operation_id - 1][0]

        available_operators = [operator_id for operator_id, finish_time in operator_last_finish_time.items() if
                                finish_time <= start_time]

        if len(available_operators) == 1:
            chosen_operator = available_operators[0]
        else:
            options = [(operator_id, current_workload[operator_id],
                        calculate_workload(EErate, operator_id, machine_id, duration))
                       for operator_id in available_operators]
            sorted_options = sorted(options, key=lambda x: (x[1], x[2]))
            chosen_operator = sorted_options[0][0]

        operator_assignmentsList[chosen_operator].append((job_id, operation_id))
        operator_last_finish_time[chosen_operator] = times['finished_time']
        current_workload[chosen_operator] += calculate_workload(EErate, chosen_operator, machine_id, duration)

    return operator_assignmentsList


def GreedyRandom_distribute_workload(EErate, jobs_data, schedule_jobs_phase2, solution_phase1, prob=0.95,
                                      reschedule=False, initial_workload=None, initial_ready_time=None):
    """
    Distribute workload fairly among operators using a probabilistic greedy assignment algorithm.
    """
    if initial_workload is None:
        initial_workload = {}
    if initial_ready_time is None:
        initial_ready_time = {}

    sorted_operations = sorted(schedule_jobs_phase2.items(), key=lambda x: (x[1]['start_time'], x[1]['duration']))

    operator_assignmentsList = {}
    operator_last_finish_time = {}
    current_workload = {}

    for i in range(len(solution_phase1)):
        operator_id = i + 1
        operator_assignmentsList[operator_id] = []
        if not reschedule:
            operator_last_finish_time[operator_id] = 0
            current_workload[operator_id] = 0

    if reschedule:
        operator_last_finish_time = initial_ready_time
        current_workload = initial_workload

    for (job_id, operation_id), times in sorted_operations:
        start_time, duration = times['start_time'], times['duration']
        machine_id = jobs_data[job_id - 1][operation_id - 1][0]

        available_operators = [operator_id for operator_id, finish_time in operator_last_finish_time.items() if
                                finish_time <= start_time]

        if len(available_operators) == 1:
            chosen_operator = available_operators[0]
        else:
            rand_value = random.random()

            if rand_value < prob:
                options = [
                    (operator_id, current_workload[operator_id],
                     calculate_workload(EErate, operator_id, machine_id, duration))
                    for operator_id in available_operators
                ]
                sorted_options = sorted(options, key=lambda x: (x[1], x[2]))
                chosen_operator = sorted_options[0][0]
            else:
                chosen_operator = random.choice(available_operators)

        operator_assignmentsList[chosen_operator].append((job_id, operation_id))
        operator_last_finish_time[chosen_operator] = times['finished_time']
        current_workload[chosen_operator] += calculate_workload(EErate, chosen_operator, machine_id, duration)

    return operator_assignmentsList


def AWBO(
        EErate,
        jobs_data,
        schedule_jobs_phase2,
        solution_phase1,
        prob=0.95,
        coloni_size=50,
        alpha=10,
        beta=2,
        reschedule=False,
        initial_workload=None,
        initial_ready_time=None,
        a2=None,
        a2_info=None,
        obj_type="variance"):
    """
    Distribute workload among operators using the Ant Work Balance Optimizer (AWBO).

    Each "ant" constructs a complete operator assignment by greedily assigning
    operations to operators using a probabilistic rule based on current workload
    (pheromone) and energy expenditure rate (heuristic). The best assignment
    across ``coloni_size`` ants is returned.

    Parameters
    ----------
    EErate : list of list of float
        Energy expenditure rate table. ``EErate[operator][machine]`` (0-indexed).
    jobs_data : list of list of tuple or None
        Job data. Each job is a list of ``(machine_id, duration)`` tuples.
    schedule_jobs_phase2 : dict
        Time schedule (Phase 2 result). Keys are ``(job_id, op_id)``.
    solution_phase1 : list
        Phase 1 result as returned by :func:`ARGA`: ``[solution_dict, obj, ...]``.
    prob : float, optional
        Probability of using the greedy best-operator selection. Default 0.95.
    coloni_size : int, optional
        Number of ant solutions to generate. Default 50.
    alpha : float, optional
        Influence weight of pheromone (current workload) in selection. Default 10.
    beta : float, optional
        Influence weight of heuristic (EE rate) in selection. Default 2.
    reschedule : bool, optional
        Enable rescheduling mode (uses ``initial_workload`` / ``initial_ready_time``).
        Default ``False``.
    initial_workload : dict[int, float] or None, optional
        Accumulated workload per operator at disruption time. Default ``None``.
    initial_ready_time : dict[int, float] or None, optional
        Earliest availability time per operator at disruption time. Default ``None``.
    a2 : list of tuple or None, optional
        In-progress operations to exclude from new assignment. Default ``None``.
    a2_info : dict or None, optional
        Schedule info for A2 operations. Default ``None``.
    obj_type : str, optional
        Workload objective to minimise. One of ``"variance"``, ``"mean"``,
        ``"maximum"``, ``"std_dev"``, ``"coef_variation"``. Default ``"variance"``.

    Returns
    -------
    dict[int, list]
        Best operator assignment: operator ID → list of ``(job_id, op_id)`` tuples.
    """
    if initial_workload is None:
        initial_workload = {}
    if initial_ready_time is None:
        initial_ready_time = {}

    best_operator_assignmentsList = None
    best_obj_value = float("inf")
    num_ops = len(EErate)

    a2_set = set(a2) if a2 else set()

    schedule_jobs_phase2_copy = {
        k: v for k, v in schedule_jobs_phase2.items()
        if k not in a2_set
    }

    sorted_operations = sorted(schedule_jobs_phase2_copy.items(), key=lambda x: (x[1]['start_time'], x[1]['duration']))

    for ant in range(coloni_size):
        operator_assignmentsList = {i + 1: [] for i in range(num_ops)}

        if reschedule:
            operator_last_finish_time = {i + 1: float(initial_ready_time.get(i + 1, 0.0)) for i in range(num_ops)}
            current_workload = {i + 1: float(initial_workload.get(i + 1, 1.0)) for i in range(num_ops)}
        else:
            operator_last_finish_time = {i + 1: 0.0 for i in range(num_ops)}
            current_workload = {i + 1: 1.0 for i in range(num_ops)}

        for (job_id, operation_id), times in sorted_operations:
            start_time, duration = times['start_time'], times['duration']
            machine_id = jobs_data[job_id - 1][operation_id - 1][0]

            available_operators = [operator_id for operator_id, finish_time in operator_last_finish_time.items() if
                                    finish_time <= start_time]

            if len(available_operators) == 0:
                chosen_operator = min(operator_last_finish_time.items(), key=lambda x: x[1])[0]
            elif len(available_operators) == 1:
                chosen_operator = available_operators[0]
            else:
                options = [(operator_id, current_workload[operator_id],
                            calculate_workload(EErate, operator_id, machine_id, duration))
                           for operator_id in available_operators]
                prob_selection = calculate_selection_probabilities(options, alpha, beta)
                chosen_operator = select_operator(prob_selection)

            operator_assignmentsList[chosen_operator].append((job_id, operation_id))
            operator_last_finish_time[chosen_operator] = times['finished_time']
            current_workload[chosen_operator] += calculate_workload(EErate, chosen_operator, machine_id, duration)

        mean, variance, maximum, std_dev, coef_variation = workload_statistic(current_workload)

        if obj_type == "mean":
            current_obj = mean
        elif obj_type == "variance":
            current_obj = variance
        elif obj_type == "maximum":
            current_obj = maximum
        elif obj_type == "std_dev":
            current_obj = std_dev
        elif obj_type == "coef_variation":
            current_obj = float("inf") if np.isnan(coef_variation) else coef_variation
        else:
            current_obj = variance

        if current_obj < best_obj_value:
            best_obj_value = current_obj
            best_operator_assignmentsList = operator_assignmentsList

    return best_operator_assignmentsList


def static_assignment(solution_phase1, reschedule=False, schedule_jobs_phase2=None, disruption_time=None, RESsolutionphase1=None):
    """
    Assign resource operations to operators using a static assignment strategy.
    """
    if not reschedule:
        operator_res_assignment = solution_phase1
    else:
        A1, A2, A3 = classify_operation(schedule_jobs_phase2, disruption_time)

        assigned_ops = set()
        operator_res_assignment = {op_id: [] for op_id in solution_phase1}

        for op in A2:
            operator_id = next((op_id for op_id, ops in solution_phase1.items() if op in ops), None)
            print(f"operator id = {operator_id}")
            if operator_id is None:
                continue

            machine_id = next((m_id for m_id, ops in solution_phase1.items() if op in ops), None)
            print(f"machine id = {machine_id}")
            if machine_id is None:
                continue

            for res_op in RESsolutionphase1.get(machine_id, []):
                if res_op not in assigned_ops:
                    operator_res_assignment[operator_id].append(res_op)
                    assigned_ops.add(res_op)

        unassigned_ops = [op for ops in RESsolutionphase1.values() for op in ops if op not in assigned_ops]
        unassigned_operators = [op_id for op_id, ops in operator_res_assignment.items() if
                                 not any(a2 in solution_phase1[op_id] for a2 in A2)]

        for op in unassigned_ops:
            if not unassigned_operators:
                target_operator = random.choice(list(operator_res_assignment.keys()))
            else:
                target_operator = random.choice(unassigned_operators)
            operator_res_assignment[target_operator].append(op)
            assigned_ops.add(op)

    return operator_res_assignment


def cumulative_workload_comparison(
    operator_operationlist_dict,
    EErate,
    schedule_jobs,
    jobs_data,
    strategy_names=("AWBO", "Greedy", "Static"),
    save_path="comparison_workload.svg"
):
    """
    Compare multiple workload distribution strategies by visualizing cumulative workload over time.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for i, strategy in enumerate(strategy_names):
        operator_operationlist = operator_operationlist_dict[strategy]

        workload_list = {}
        event_times = {}
        current_workload = {}

        for operator_id, operation_list in operator_operationlist.items():
            if operator_id not in event_times:
                event_times[operator_id] = []
            for operation in operation_list:
                operation_info = schedule_jobs[operation]
                event_times[operator_id].append(operation_info['start_time'])
                event_times[operator_id].append(operation_info['finished_time'])

            if operator_id not in workload_list:
                workload_list[operator_id] = []
            if operator_id not in current_workload:
                current_workload[operator_id] = 0

            for idx in range(len(event_times[operator_id])):
                if idx % 2 == 0:
                    workload_list[operator_id].append(current_workload[operator_id])
                else:
                    operation = operation_list[idx // 2]
                    machine_id = jobs_data[operation[0] - 1][operation[1] - 1][0]
                    start_time = event_times[operator_id][idx - 1]
                    finish_time = event_times[operator_id][idx]
                    duration = finish_time - start_time
                    additional_workload = calculate_workload(EErate, operator_id, machine_id, duration)
                    workload_list[operator_id].append(current_workload[operator_id] + additional_workload)
                    current_workload[operator_id] += additional_workload

        ax = axes[i]
        for operator_id, values in workload_list.items():
            ax.plot(event_times[operator_id], values, marker='o', label=f'Operator {operator_id}')
        ax.set_title(f'{strategy} Assignment', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (minutes)', fontsize=10)
        if i == 0:
            ax.set_ylabel('Workload (kcal)', fontsize=12)
            ax.legend(loc='upper left', fontsize=12)
        ax.grid(True)
        max_time = max(max(times) for times in event_times.values())
        ax.set_xticks(np.arange(0, max_time + 10, 20))

    plt.suptitle('Accumulated workload over the scheduling period', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()


def cumulative_workload_resch(
    operator_operationlist,
    EErate,
    schedule_jobs,
    jobs_data,
    visualization=True,
    reschedule=False,
    initial_workload=None,
    a2=None,
    a2_info=None
):
    """
    Calculate cumulative energy expenditure (workload) over time for each operator, excluding A2 operations.
    """
    if initial_workload is None:
        initial_workload = {}

    a2_set = set(a2) if a2 else set()

    workload_list = {}
    event_times = {}
    current_workload = copy.deepcopy(initial_workload) if reschedule else {}

    for operator_id, operation_list in operator_operationlist.items():
        ops = [op for op in operation_list if op not in a2_set]

        event_times.setdefault(operator_id, [])
        workload_list.setdefault(operator_id, [])
        if operator_id not in current_workload:
            current_workload[operator_id] = 0.0

        for op in ops:
            st = schedule_jobs[op]['start_time']
            ft = schedule_jobs[op]['finished_time']

            event_times[operator_id].append(st)
            workload_list[operator_id].append(current_workload[operator_id])

            j_id, o_id = op
            m_id = jobs_data[j_id - 1][o_id - 1][0]
            dur = ft - st
            inc = calculate_workload(EErate, operator_id, m_id, dur)

            current_workload[operator_id] += inc
            event_times[operator_id].append(ft)
            workload_list[operator_id].append(current_workload[operator_id])

    print("Done.\nCummulative Workload per operator:")
    for operator_id, total in current_workload.items():
        print(f"Operator {operator_id}: {round(total, 2)} kcal")

    if visualization:
        plt.figure(figsize=(10, 5))
        for operator_id, values in workload_list.items():
            plt.plot(event_times[operator_id], values, marker='o', label=f'Operator {operator_id}')
        plt.title('Cumulative workload per operator', fontsize=14)
        plt.xlabel('Time horizon (minutes)', fontsize=12)
        plt.ylabel('Total Workload (kcal)', fontsize=12)
        plt.ylim(bottom=0)
        if event_times:
            max_time = max(max(times) for times in event_times.values() if times)
            plt.xticks(np.arange(0, max_time + 10, 20))
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return workload_list, event_times, current_workload
