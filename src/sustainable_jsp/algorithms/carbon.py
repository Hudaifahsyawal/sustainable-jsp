from __future__ import annotations

import math
import random
import copy

import numpy as np
from scipy.stats import skewnorm
from tqdm import tqdm

from sustainable_jsp.core.performance_matrix import calculate_time_objective


def generate_emission_data(num_machines, min_value=0.01, max_value=0.05):
    """
    Generate carbon emission data for machines at different speed levels.

    Creates a data structure containing carbon emission rates (in grams CO2e per minute)
    for each machine at three different speed levels. The emission data follows a realistic
    pattern where higher speed levels result in higher carbon emissions due to increased
    energy consumption.
    """
    min_value = int(min_value * 100)
    max_value = int(max_value * 100)

    emission_data = {1: [1], 2: [0.8], 3: [0.6]}

    first_row_data = abs(skewnorm.rvs(a=10, loc=5, scale=20, size=num_machines))
    first_row = [round(float(x), 1) for x in first_row_data]

    second_row = [round(value * 1.5, 1) for value in first_row]
    third_row = [round(value * 2.5, 1) for value in first_row]

    emission_data[1].extend(first_row)
    emission_data[2].extend(second_row)
    emission_data[3].extend(third_row)

    return emission_data


def calculate_bound(
    jobs_data,
    carbon_emission_data,
    solution_phase1_jobs_data,
    schedule_jobs_data,
    reschedule=False,
    original_processing_time=None,
    machine_start_time=None,
    a2=None,
    a2_info=None,
    disruption_time=None,
    init_time_schedule=None,
    obj_type="cmax+flowtime",
    IR=3,
    flowtime_type="average"):
    """
    Calculate lower and upper bounds for time objective and total carbon emission (TCE).

    Computes theoretical bounds for the time objective (Cmax, flowtime, or combined) and
    total carbon emission by evaluating the solution at extreme speed level configurations.
    This function is useful for understanding the trade-off space between time performance
    and carbon emissions, and for normalizing objective values in multi-objective optimization.
    """
    if machine_start_time is None:
        machine_start_time = {}

    operation_id = list(schedule_jobs_data.keys())

    if reschedule:
        original_processing_time = original_processing_time
    else:
        original_processing_time = [value['original_duration'] for value in schedule_jobs_data.values()]

    machine_list = [value['machine_id'] for value in schedule_jobs_data.values()]

    speedlevel_list1 = [1 for _ in range(len(machine_list))]
    speedlevel_list2 = [3 for _ in range(len(machine_list))]

    updated_processing_time1 = [round(original_processing_time[i] * carbon_emission_data[speedlevel_list1[i]][0], 1)
                                 for i in range(len(original_processing_time))]
    updated_processing_time2 = [round(original_processing_time[i] * carbon_emission_data[speedlevel_list2[i]][0], 1)
                                 for i in range(len(original_processing_time))]

    updated_schedule1 = update_schedule_time(jobs_data, solution_phase1_jobs_data, schedule_jobs_data, updated_processing_time1, reschedule, machine_start_time, a2, a2_info, disruption_time)
    if reschedule and init_time_schedule is not None:
        updated_schedule1 = dict(sorted({**init_time_schedule, **updated_schedule1}.items(), key=lambda x: x[0]))
    UB_time_obj = round(calculate_time_objective(updated_schedule1, obj_type, IR, flowtime_type), 1)

    updated_schedule2 = update_schedule_time(jobs_data, solution_phase1_jobs_data, schedule_jobs_data, updated_processing_time2, reschedule, machine_start_time, a2, a2_info, disruption_time)
    if reschedule and init_time_schedule is not None:
        updated_schedule2 = dict(sorted({**init_time_schedule, **updated_schedule2}.items(), key=lambda x: x[0]))
    LB_time_obj = round(calculate_time_objective(updated_schedule2, obj_type, IR, flowtime_type), 1)

    if reschedule and init_time_schedule is not None:
        init_ops = sorted(init_time_schedule.keys())
        init_machine_list = [init_time_schedule[op]["machine_id"] for op in init_ops]
        init_duration = [init_time_schedule[op]["duration"] for op in init_ops]
        init_speedlevel = [init_time_schedule[op]["speed_level"] for op in init_ops]
        machine_list_carbon = init_machine_list + machine_list
        updated_processing_time1_carbon = init_duration + updated_processing_time1
        updated_processing_time2_carbon = init_duration + updated_processing_time2
        speedlevel_list1_carbon = init_speedlevel + speedlevel_list1
        speedlevel_list2_carbon = init_speedlevel + speedlevel_list2
        LB_carbon = calculate_total_carbon_emission(machine_list_carbon, updated_processing_time1_carbon, carbon_emission_data, speedlevel_list1_carbon)
        UB_carbon = calculate_total_carbon_emission(machine_list_carbon, updated_processing_time2_carbon, carbon_emission_data, speedlevel_list2_carbon)
    else:
        LB_carbon = calculate_total_carbon_emission(machine_list, updated_processing_time1, carbon_emission_data, speedlevel_list1)
        UB_carbon = calculate_total_carbon_emission(machine_list, updated_processing_time2, carbon_emission_data, speedlevel_list2)

    return LB_time_obj, UB_time_obj, round(LB_carbon, 0), round(UB_carbon, 0)


def calculate_total_carbon_emission(machine_list, updated_processing_time, carbon_emission_data, speedlevel_list):
    """
    Calculate the total carbon emission (TCE) for a schedule with specified speed levels.

    Computes the total carbon dioxide equivalent (CO2e) emissions for all operations in a
    schedule by multiplying each operation's processing time by its corresponding carbon
    emission rate (which depends on the machine and speed level), then summing across all
    operations.
    """
    carbon_values = [carbon_emission_data[speedlevel_list[i]][machine_list[i]] for i in range(len(machine_list))]
    carbon_emission = [round(carbon_values[i] * updated_processing_time[i], 1) for i in range(len(updated_processing_time))]
    Total_carbon_emission = sum(carbon_emission)
    return Total_carbon_emission


def calculate_obj_value(update_time_obj, TCE, LB_time_obj, UB_time_obj, LB_carbon, UB_carbon, weight):
    """
    Calculate normalized weighted multi-objective value combining time objective and carbon emission.

    Computes a scalarized multi-objective value by normalizing both time objective and total
    carbon emission to the [0, 1] range using min-max normalization, then combining them with
    a weighted sum. This allows fair comparison of solutions when objectives have different
    scales and units.
    """
    obj_value = weight * ((update_time_obj - LB_time_obj) / (UB_time_obj - LB_time_obj)) + (1 - weight) * ((TCE - LB_carbon) / (UB_carbon - LB_carbon))
    return obj_value


def update_schedule_time(
        jobs_data: list[list[tuple[int, int] | None]],
        solution_phase1_jobs_data: dict[int, list[tuple[int, int]]],
        schedule_jobs_data: dict[tuple[int, int], dict[str, float]],
        updated_processing_time: list[float],
        reschedule: bool = False,
        machine_start_time: dict[int, float] | None = None,
        a2: list[tuple[int, int]] | None = None,
        a2_info: dict[tuple[int, int], dict[str, float]] | None = None,
        disruption_time: float | None = None) -> dict[tuple[int, int], dict[str, float]]:
    """
    Updates schedule times with adaptive repair when processing times change (due to speed level adjustments).

    This function is typically used in Phase 2 optimization when operation durations are modified
    (e.g., by applying different machine speed levels) while preserving the machine-operation
    sequences determined in Phase 1. It recalculates start and finish times based on the new
    processing times, maintaining job precedence constraints and handling rescheduling scenarios.
    """
    if machine_start_time is None:
        machine_start_time = {}

    a2 = a2 or []
    a2_info = a2_info or {}

    n_jobs = len(jobs_data)
    n_machines = len(solution_phase1_jobs_data)
    n_operations = max(len(job) for job in jobs_data)

    dispatchlist = []
    for job in jobs_data:
        row = [False] * n_operations
        for idx, op in enumerate(job):
            if op is not None:
                row[idx] = True
                break
        dispatchlist.append(row)

    schedule_jobs_data_copy = copy.deepcopy(schedule_jobs_data)
    updated_time_iter = iter(updated_processing_time)
    for key in schedule_jobs_data_copy:
        schedule_jobs_data_copy[key]['duration'] = next(updated_time_iter)
        schedule_jobs_data_copy[key]['start_time'] = math.nan
        schedule_jobs_data_copy[key]['finished_time'] = math.nan

    machine_available_time = {m: 0 for m in range(1, n_machines + 1)} if not reschedule else copy.deepcopy(machine_start_time)

    start_machine_index = 1
    solution_phase1_jobs_data_dummy = copy.deepcopy(solution_phase1_jobs_data)
    visited_machines = set()

    while any(any(row) for row in dispatchlist):
        machine_index = start_machine_index

        if machine_index in visited_machines:
            sequence_fixed = False
            while not sequence_fixed:
                if not solution_phase1_jobs_data_dummy[machine_index]:
                    machine_index = (machine_index % n_machines) + 1
                    continue
                if len(solution_phase1_jobs_data_dummy[machine_index]) == 1:
                    machine_index = (machine_index % n_machines) + 1
                    continue

                for i in range(1, len(solution_phase1_jobs_data_dummy[machine_index])):
                    job_index, operation_index = solution_phase1_jobs_data_dummy[machine_index][i]
                    if dispatchlist[job_index - 1][operation_index - 1]:
                        solution_phase1_jobs_data_dummy[machine_index].insert(
                            0, solution_phase1_jobs_data_dummy[machine_index].pop(i)
                        )
                        front_index = len(solution_phase1_jobs_data[machine_index]) - len(solution_phase1_jobs_data_dummy[machine_index])
                        solution_phase1_jobs_data[machine_index] = (
                            solution_phase1_jobs_data[machine_index][:front_index] +
                            solution_phase1_jobs_data_dummy[machine_index]
                        )
                        visited_machines.clear()
                        sequence_fixed = True
                        break
                if not sequence_fixed:
                    machine_index = (machine_index % n_machines) + 1
                else:
                    break

        if not solution_phase1_jobs_data_dummy[machine_index]:
            start_machine_index = (start_machine_index % n_machines) + 1
            visited_machines.add(machine_index)
            continue

        job_index, operation_index = solution_phase1_jobs_data_dummy[machine_index][0]

        if dispatchlist[job_index - 1][operation_index - 1]:
            if operation_index == 1:
                prev_finished_time = 0
            else:
                pred_idx = operation_index - 1
                pred_key = (job_index, pred_idx)

                pred_is_none = (jobs_data[job_index - 1][pred_idx - 1] is None)
                pred_is_a2 = (pred_key in a2)

                if pred_is_none and not pred_is_a2:
                    prev_finished_time = disruption_time
                elif pred_is_none and pred_is_a2:
                    prev_finished_time = a2_info[(job_index, pred_idx)]["finished_time"]
                else:
                    prev_finished_time = schedule_jobs_data_copy[(job_index, pred_idx)]["finished_time"]

            start_time = max(machine_available_time[machine_index], prev_finished_time)

            duration = schedule_jobs_data_copy[(job_index, operation_index)]["duration"]
            finished_time = start_time + duration

            schedule_jobs_data_copy[(job_index, operation_index)]["start_time"] = round(start_time, 1)
            schedule_jobs_data_copy[(job_index, operation_index)]["finished_time"] = round(finished_time, 1)

            machine_available_time[machine_index] = finished_time

            dispatchlist[job_index - 1][operation_index - 1] = False

            if operation_index < len(jobs_data[job_index - 1]):
                dispatchlist[job_index - 1][operation_index] = True

            start_machine_index = (start_machine_index % n_machines) + 1
            solution_phase1_jobs_data_dummy[machine_index].pop(0)
            visited_machines.clear()

        else:
            start_machine_index = (start_machine_index % n_machines) + 1
            visited_machines.add(machine_index)

    return schedule_jobs_data_copy


def get_initial_population_speedlevels(
        population_size2: int,
        original_processing_time: list[float],
        carbon_emission_data: dict[int, list[float]],
        jobs_data: list[list[tuple[int, int] | None]],
        solution_phase1_jobs_data: dict[int, list[tuple[int, int]]],
        schedule_jobs_data: dict[tuple[int, int], dict[str, float]],
        machine_list: list[int],
        LB_time_obj: float,
        UB_time_obj: float,
        LB_carbon: float,
        UB_carbon: float,
        weight: float,
        reschedule: bool,
        machine_start_time: dict[int, float] | None,
        a2: list[tuple[int, int]] | None,
        a2_info: dict[tuple[int, int], dict[str, float]] | None,
        disruption_time: float | None,
        init_time_schedule: dict[tuple[int, int], dict[str, float]] | None = None,
        obj_type: str = "cmax",
        IR: float = 3,
        flowtime_type: str = "average") -> list[list]:
    """
    Generates an initial population of speed level assignments for Phase 2 genetic algorithm optimization.

    This function creates a diverse set of speed level configurations (each operation assigned a speed level
    from 1 to 3) to kickstart the Phase 2 genetic algorithm optimization. Each individual in the population
    represents a complete speed level assignment for all operations, and is evaluated to determine its
    performance metrics (time objective, total carbon emission, and combined objective value z).
    The population is sorted by the combined objective value (z), with the best solutions first.
    """
    population_speedlevels_pool = []

    for _ in range(population_size2):
        init_list = []
        speedlevel_list = [random.randint(1, 3) for _ in range(len(machine_list))]
        time_obj, TCE, z = evaluate_individu_CeCmaxZ(
            speedlevel_list, original_processing_time, carbon_emission_data, jobs_data,
            solution_phase1_jobs_data, schedule_jobs_data, machine_list, LB_time_obj,
            UB_time_obj, LB_carbon, UB_carbon, weight, reschedule, machine_start_time,
            a2, a2_info, disruption_time, init_time_schedule, obj_type, IR, flowtime_type)
        init_list.append(speedlevel_list)
        init_list.append(time_obj)
        init_list.append(TCE)
        init_list.append(z)
        population_speedlevels_pool.append(init_list)

    sorted_population = sorted(population_speedlevels_pool, key=lambda x: x[-1])
    return sorted_population


def evaluate_individu_CeCmaxZ(
        speedlevel_list: list[int],
        original_processing_time: list[float],
        carbon_emission_data: dict[int, list[float]],
        jobs_data: list[list[tuple[int, int] | None]],
        solution_phase1_jobs_data: dict[int, list[tuple[int, int]]],
        schedule_jobs_data: dict[tuple[int, int], dict[str, float]],
        machine_list: list[int],
        LB_time_obj: float,
        UB_time_obj: float,
        LB_carbon: float,
        UB_carbon: float,
        weight: float,
        reschedule: bool = False,
        machine_start_time: dict[int, float] | None = None,
        a2: list[tuple[int, int]] | None = None,
        a2_info: dict[tuple[int, int], dict[str, float]] | None = None,
        disruption_time: float | None = None,
        init_time_schedule: dict[tuple[int, int], dict[str, float]] | None = None,
        obj_type: str = "cmax",
        IR: float = 3,
        flowtime_type: str = "average") -> tuple[float, float, float]:
    """
    Evaluates a speed level assignment by calculating time objective, total carbon emission, and combined objective value.
    """
    if machine_start_time is None:
        machine_start_time = {}

    updated_processing_time = [round(original_processing_time[j] * carbon_emission_data[speedlevel_list[j]][0], 1) for j in range(len(original_processing_time))]
    updated_schedule_time = update_schedule_time(jobs_data, solution_phase1_jobs_data, schedule_jobs_data, updated_processing_time, reschedule, machine_start_time, a2, a2_info, disruption_time)

    if reschedule and init_time_schedule is not None:
        full_schedule_time = dict(sorted({**init_time_schedule, **updated_schedule_time}.items(), key=lambda x: x[0]))
        updated_time_objective = round(calculate_time_objective(full_schedule_time, obj_type, IR, flowtime_type), 1)
    else:
        updated_time_objective = round(calculate_time_objective(updated_schedule_time, obj_type, IR, flowtime_type), 1)

    if reschedule and init_time_schedule is not None:
        init_ops = sorted(init_time_schedule.keys())
        init_machine_list = [init_time_schedule[op]["machine_id"] for op in init_ops]
        init_duration = [init_time_schedule[op]["duration"] for op in init_ops]
        init_speedlevel = [init_time_schedule[op]["speed_level"] for op in init_ops]
        machine_list_tce = init_machine_list + machine_list
        updated_processing_time_tce = init_duration + updated_processing_time
        speedlevel_list_tce = init_speedlevel + speedlevel_list
        TCE = round(calculate_total_carbon_emission(machine_list_tce, updated_processing_time_tce, carbon_emission_data, speedlevel_list_tce), 1)
    else:
        TCE = round(calculate_total_carbon_emission(machine_list, updated_processing_time, carbon_emission_data, speedlevel_list), 1)

    z = round(calculate_obj_value(updated_time_objective, TCE, LB_time_obj, UB_time_obj, LB_carbon, UB_carbon, weight), 4)
    return updated_time_objective, TCE, z


def speedlevel_parent_selection(
        population_pool: list[list],
        elit_percentage2: float = 0.2) -> tuple[list[int], list[int]]:
    """
    Selects two parents from the population pool using an elitism-based strategy.
    """
    population_size2 = len(population_pool)
    parent1_index: list[int] = list(range(0, math.ceil(population_size2 * elit_percentage2)))
    parent1_index = random.choice(parent1_index)
    parent2_index: list[int] = list(range(0, population_size2))
    parent2_index = random.choice(parent2_index)
    return population_pool[parent1_index][0], population_pool[parent2_index][0]


def speedlevel_crossover(
        parent1: list[int],
        parent2: list[int]) -> tuple[list[int], list[int]]:
    """
    Performs a single-point crossover operation between two parent speed level lists, generating two children.
    """
    crossover_point = len(parent1) // 2

    P1 = copy.deepcopy(parent1)
    P2 = copy.deepcopy(parent2)

    child1 = P1[:crossover_point] + P2[crossover_point:]
    child2 = P2[:crossover_point] + P1[crossover_point:]

    return child1, child2


def speedlevel_mutation(
        child: list[int],
        mutation_threshold2: float) -> list[int]:
    """
    Performs mutation on a child speed level list with probabilistic occurrence and random position selection.
    """
    if random.random() < mutation_threshold2:
        max_mutations = math.ceil(len(child) * 0.2)
        num_mutations = random.randint(1, max_mutations)
        indices_to_mutate = random.sample(range(len(child)), num_mutations)
        for index in indices_to_mutate:
            new_value = random.randint(1, 3)
            child[index] = new_value

    return child


def GANBI(
    jobs_data: list[list[tuple[int, int] | None]],
    schedule_jobs_data_ori: dict[tuple[int, int], dict[str, float]],
    solution_phase1_jobs_data: dict[int, list[tuple[int, int]]],
    carbon_emission_data: dict[int, list[float]],
    num_iterations2: int = 1500,
    population_size2: int = 75,
    elit_percentage2: float = 0.6,
    mutation_threshold2: float = 0.5,
    weight: float = 0.75,
    visualization: bool = False,
    reschedule: bool = False,
    machine_start_time: dict[int, float] | None = None,
    x_start: float | None = None,
    x_end: float | None = None,
    a2: list[tuple[int, int]] | None = None,
    a2_info: dict[tuple[int, int], dict[str, float]] | None = None,
    disruption_time: float | None = None,
    init_time_schedule: dict[tuple[int, int], dict[str, float]] | None = None,
    obj_type: str = "cmax",
    IR: float = 3,
    flowtime_type: str = "average",
    show_progress: bool = False,
    bar_position: int = 0) -> dict:
    """
    Executes a complete genetic algorithm procedure for Phase 2 optimization: carbon-aware speed level assignment.

    This function implements a genetic algorithm to optimize speed levels for all operations in a
    Job Shop Scheduling problem, balancing time objectives (e.g., Cmax, Flowtime) with total carbon
    emissions. It is designed as Phase 2 optimization, following Phase 1 which determines the
    optimal machine-operation sequences.
    """
    if machine_start_time is None:
        machine_start_time = {}

    if a2 is None:
        a2 = set()

    schedule_jobs_data = {op: dict(data) for op, data in schedule_jobs_data_ori.items() if op not in a2}

    operation_id = list(schedule_jobs_data.keys())
    original_processing_time = [value['original_duration'] for value in schedule_jobs_data.values()]
    machine_list = [value['machine_id'] for value in schedule_jobs_data.values()]

    LB_time_obj, UB_time_obj, LB_carbon, UB_carbon = calculate_bound(
        jobs_data, carbon_emission_data, solution_phase1_jobs_data, schedule_jobs_data,
        reschedule, original_processing_time, machine_start_time, a2, a2_info,
        disruption_time, init_time_schedule, obj_type, IR, flowtime_type)

    print(f'LB_time_obj: {LB_time_obj}, UB_time_obj: {UB_time_obj}, LB_carbon: {LB_carbon}, UB_carbon: {UB_carbon} ')

    population = get_initial_population_speedlevels(
        population_size2, original_processing_time, carbon_emission_data,
        jobs_data, solution_phase1_jobs_data, schedule_jobs_data,
        machine_list, LB_time_obj, UB_time_obj, LB_carbon, UB_carbon,
        weight, reschedule, machine_start_time, a2, a2_info, disruption_time,
        init_time_schedule, obj_type, IR, flowtime_type)

    best_speedlevel = copy.deepcopy(population[0])

    pbar = tqdm(range(num_iterations2), desc="GANBI", unit="iter", disable=not show_progress, position=bar_position, leave=True)
    for iteration in pbar:
        child_speedlevel_pool = []

        for _ in range(population_size2 // 2):
            parent1, parent2 = speedlevel_parent_selection(population, elit_percentage2)

            child1, child2 = speedlevel_crossover(parent1, parent2)

            child1 = speedlevel_mutation(child1, mutation_threshold2)
            child2 = speedlevel_mutation(child2, mutation_threshold2)

            child1_list = []
            child1_performance = evaluate_individu_CeCmaxZ(
                child1, original_processing_time, carbon_emission_data, jobs_data,
                solution_phase1_jobs_data, schedule_jobs_data, machine_list,
                LB_time_obj, UB_time_obj, LB_carbon, UB_carbon, weight,
                reschedule, machine_start_time, a2, a2_info, disruption_time,
                init_time_schedule, obj_type, IR, flowtime_type)
            child1_list.extend([child1, child1_performance[0], child1_performance[1], child1_performance[2]])
            child_speedlevel_pool.append(child1_list)

            child2_list = []
            child2_performance = evaluate_individu_CeCmaxZ(
                child2, original_processing_time, carbon_emission_data, jobs_data,
                solution_phase1_jobs_data, schedule_jobs_data, machine_list,
                LB_time_obj, UB_time_obj, LB_carbon, UB_carbon, weight,
                reschedule, machine_start_time, a2, a2_info, disruption_time,
                init_time_schedule, obj_type, IR, flowtime_type)
            child2_list.extend([child2, child2_performance[0], child2_performance[1], child2_performance[2]])
            child_speedlevel_pool.append(child2_list)

        combined_pool = population + child_speedlevel_pool
        sorted_pool = sorted(combined_pool, key=lambda x: x[-1])

        if sorted_pool[0][-1] < best_speedlevel[-1]:
            best_speedlevel = copy.deepcopy(sorted_pool[0])

        population = copy.deepcopy(sorted_pool[:population_size2])
        pbar.set_postfix({"Best Z": f"{best_speedlevel[-1]:.2f}"})

    optimum_processing_time = [round(original_processing_time[i] * carbon_emission_data[best_speedlevel[0][i]][0], 1)
                                for i in range(len(original_processing_time))]

    updated_optimum_scheduling_time = update_schedule_time(
        jobs_data, solution_phase1_jobs_data, schedule_jobs_data,
        optimum_processing_time, reschedule, machine_start_time, a2, a2_info, disruption_time)

    keys = list(updated_optimum_scheduling_time.keys())
    for i, key in enumerate(keys):
        updated_optimum_scheduling_time[key]['speed_level'] = best_speedlevel[0][i]

    final_schedule_time = {}
    for op in schedule_jobs_data_ori.keys():
        if op in a2 and a2_info is not None and op in a2_info:
            final_schedule_time[op] = dict(a2_info[op])
        else:
            final_schedule_time[op] = dict(updated_optimum_scheduling_time[op])

    best_speedlevel_all = []
    for op in schedule_jobs_data_ori.keys():
        sl = final_schedule_time[op].get('speed_level')
        best_speedlevel_all.append(sl)

    matrix_performance = {
        "time_objective": best_speedlevel[1],
        "TCE": best_speedlevel[2],
        "z": best_speedlevel[3]
    }

    if visualization is True:
        print("Visualization disabled in package mode. Use create_gantt_chart_final() directly.")

    pbar.close()
    return {
        "speedlevel_optimum": best_speedlevel_all,
        "final_schedule_time": final_schedule_time,
        "matrix_performance": matrix_performance,
        "bound": {
            "LB_time_obj": LB_time_obj,
            "UB_time_obj": UB_time_obj,
            "LB_carbon": LB_carbon,
            "UB_carbon": UB_carbon
        }
    }


def calculate_SL2(
        jobs_data: list[list[tuple[int, int] | None]],
        carbon_emission_data: dict[int, list[float]],
        solution_phase1_jobs_data: dict[int, list[tuple[int, int]]],
        schedule_jobs_data: dict[tuple[int, int], dict[str, float]],
        weight: float = 0.75,
        reschedule: bool = False,
        machine_start_time: dict[int, float] | None = None,
        a2: list[tuple[int, int]] | None = None,
        a2_info: dict[tuple[int, int], dict[str, float]] | None = None,
        disruption_time: float | None = None,
        init_time_schedule: dict[tuple[int, int], dict[str, float]] | None = None,
        obj_type: str = "cmax",
        IR: float = 3,
        flowtime_type: str = "average") -> tuple[float, float, float, dict]:
    """
    Calculates performance metrics when all operations use speed level 2 (medium speed) as a baseline comparison.
    """
    if machine_start_time is None:
        machine_start_time = {}

    original_processing_time = [value['original_duration'] for value in schedule_jobs_data.values()]
    machine_list = [value['machine_id'] for value in schedule_jobs_data.values()]

    speedlevel_list2 = [2 for _ in range(len(machine_list))]

    LB_time_obj, UB_time_obj, LB_carbon, UB_carbon = calculate_bound(
        jobs_data, carbon_emission_data, solution_phase1_jobs_data, schedule_jobs_data,
        reschedule, original_processing_time, machine_start_time, a2, a2_info,
        disruption_time, init_time_schedule, obj_type, IR, flowtime_type)

    updated_processing_time2 = [round(original_processing_time[i] * carbon_emission_data[speedlevel_list2[i]][0], 1)
                                 for i in range(len(original_processing_time))]

    updated_schedule2 = update_schedule_time(jobs_data, solution_phase1_jobs_data, schedule_jobs_data, updated_processing_time2, reschedule, machine_start_time, a2, a2_info, disruption_time)
    if reschedule and init_time_schedule is not None:
        updated_schedule2 = dict(sorted({**init_time_schedule, **updated_schedule2}.items(), key=lambda x: x[0]))
    time_obj = round(calculate_time_objective(updated_schedule2, obj_type, IR, flowtime_type), 1)

    if reschedule and init_time_schedule is not None:
        init_ops = sorted(init_time_schedule.keys())
        init_machine_list = [init_time_schedule[op]["machine_id"] for op in init_ops]
        init_duration = [init_time_schedule[op]["duration"] for op in init_ops]
        init_speedlevel = [init_time_schedule[op]["speed_level"] for op in init_ops]
        machine_list_carbon = init_machine_list + machine_list
        updated_processing_time2_carbon = init_duration + updated_processing_time2
        speedlevel_list2_carbon = init_speedlevel + speedlevel_list2
        TCE = calculate_total_carbon_emission(machine_list_carbon, updated_processing_time2_carbon, carbon_emission_data, speedlevel_list2_carbon)
    else:
        TCE = calculate_total_carbon_emission(machine_list, updated_processing_time2, carbon_emission_data, speedlevel_list2)

    z = round(calculate_obj_value(time_obj, TCE, LB_time_obj, UB_time_obj, LB_carbon, UB_carbon, weight), 4)
    return time_obj, round(TCE, 0), z, updated_schedule2
