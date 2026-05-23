from __future__ import annotations

import math
import random
import copy

import numpy as np
from tqdm import tqdm

from sustainable_jsp.core.schedule import (
    generate_feasible_solution1,
    generate_feasible_solution2,
    get_schedule_time_AR_resch,
)
from sustainable_jsp.core.performance_matrix import calculate_time_objective


def get_initial_population(job_data, get_schedule_time, population_size1, reschedule, machine_start_time, FSG=generate_feasible_solution1, a2=None, a2_info=None, disruption_time=None, init_time_schedule=None, obj_type="cmax", IR=3, flowtime_type="average"):
    """
    Generate an initial population for genetic algorithm optimization procedures.

    Creates a population of feasible solutions by repeatedly generating solutions using
    a specified feasible solution generator (FSG) and evaluating their performance. Each
    individual in the population consists of a solution and its corresponding objective
    value. The population is sorted by objective value in ascending order (best solutions
    first), which is useful for genetic algorithms that use elitism or rank-based selection.
    """
    population_pool = []

    for _ in range(population_size1):
        init_tuple = []
        solution = FSG(job_data)

        schedule_time = get_schedule_time(job_data, solution, reschedule, machine_start_time, a2, a2_info, disruption_time)
        if schedule_time is not None:
            if reschedule and init_time_schedule is not None:
                merged = {**init_time_schedule, **schedule_time}
                full_schedule_time = dict(sorted(merged.items(), key=lambda x: x[0]))
                time_objective = calculate_time_objective(full_schedule_time, obj_type, IR, flowtime_type)
            else:
                time_objective = calculate_time_objective(schedule_time, obj_type, IR, flowtime_type)
        init_tuple.append(solution)
        init_tuple.append(time_objective)
        population_pool.append(init_tuple)

    sorted_population = sorted(population_pool, key=lambda x: x[1])

    return sorted_population


def parent_selection(population_pool, elit_percentage1):
    """
    Select two parents from the population pool using an elitism-based selection strategy.

    Implements a parent selection mechanism for genetic algorithms that combines elitism
    with random selection. The first parent is selected from the elite portion of the
    population (top individuals based on elit_percentage1), while the second parent is
    selected randomly from the entire population. This approach balances exploitation
    (using good solutions) with exploration (maintaining diversity).
    """
    population_size1 = len(population_pool)
    parent1_index: list[int] = list(range(0, math.ceil(population_size1 * elit_percentage1)))
    parent1_index = random.choice(parent1_index)
    parent2_index: list[int] = list(range(0, population_size1))
    parent2_index = random.choice(parent2_index)

    return population_pool[parent1_index][0], population_pool[parent2_index][0]


def crossover1(parent1, parent2):
    """
    Perform single-point crossover between two parent solutions to generate two children.

    Implements a simple crossover operator for genetic algorithms that exchanges the
    operation sequence of a single randomly selected machine between two parent solutions.
    This creates two offspring solutions that inherit most of their structure from their
    parents while introducing variation through the machine sequence swap.
    """
    keys = list(parent1.keys())
    key_to_swap = random.choice(keys)

    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    child1[key_to_swap], child2[key_to_swap] = child2[key_to_swap], child1[key_to_swap]

    return child1, child2


def crossover1P(parent1, parent2, iteration, total_iterations):
    """
    Perform single-point crossover with probability controlled by a shifted sigmoid function.

    Implements a probabilistic crossover operator that exchanges the operation sequence of a
    single randomly selected machine between two parent solutions. Unlike crossover1(), this
    function uses an adaptive probability mechanism where the crossover probability decreases
    as the algorithm progresses through iterations.
    """
    keys = list(parent1.keys())
    key_to_swap = random.choice(keys)

    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    if random.random() < 1 / (1 + np.exp((iteration - (total_iterations / 2)) / 50)):
        child1[key_to_swap], child2[key_to_swap] = child2[key_to_swap], child1[key_to_swap]

    return child1, child2


def crossoverNR(parent1, parent2):
    """
    Perform multi-point crossover between two parent solutions with random number of swaps.

    Implements a crossover operator that exchanges operation sequences for a random number
    of machines (between 1 and all machines) between two parent solutions. Unlike
    crossover1() which swaps exactly one machine, this function can swap multiple machines,
    creating more diverse offspring solutions.
    """
    keys = list(parent1.keys())
    keys_to_swap = random.sample(keys, random.randint(1, len(keys)))

    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    for key in keys_to_swap:
        child1[key], child2[key] = child2[key], child1[key]

    return child1, child2


def crossoverNRP(parent1, parent2, iteration, total_iterations):
    """
    Perform multi-point crossover with per-machine probability controlled by a shifted sigmoid function.

    Implements a probabilistic multi-point crossover operator that combines the multi-machine
    swapping of crossoverNR() with the adaptive probability mechanism of crossover1P().
    The function randomly selects multiple machines to potentially swap, then applies a
    probability check independently for each selected machine.
    """
    keys = list(parent1.keys())
    keys_to_swap = random.sample(keys, random.randint(1, len(keys)))

    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    for key in keys_to_swap:
        if random.random() < 1 / (1 + np.exp((iteration - (total_iterations / 2)) / 50)):
            child1[key], child2[key] = child2[key], child1[key]

    return child1, child2


def mutate1N(child, mutation_threshold1):
    """
    Perform neighborhood swap mutation on a child solution with per-machine probability.

    Implements a local search mutation operator that swaps adjacent operations within each machine's
    sequence. For each machine, with a given probability, one operation is randomly selected and
    swapped with one of its immediate neighbors (the operation before or after it). This creates
    small, localized changes that maintain most of the solution structure while introducing
    variation.
    """
    mutated_child = copy.deepcopy(child)
    for key, value in mutated_child.items():
        if random.random() < mutation_threshold1 and len(value) > 1:
            idx = random.randint(0, len(value) - 1)

            if idx == 0:
                swap_idx = 1
            elif idx == len(value) - 1:
                swap_idx = len(value) - 2
            else:
                swap_idx = idx + random.choice([-1, 1])

            value[idx], value[swap_idx] = value[swap_idx], value[idx]

    return mutated_child


def mutate1R(child, mutation_threshold1):
    """
    Perform random swap mutation on a child solution with per-machine probability.

    Implements a mutation operator that randomly swaps two operations within each machine's
    sequence. For each machine, with a given probability, two operations are randomly
    selected (without replacement) and swapped. Unlike mutate1N() which swaps adjacent
    operations, this function can swap any two operations regardless of their positions,
    potentially creating larger changes in the solution structure.
    """
    mutated_child = copy.deepcopy(child)
    for key, value in mutated_child.items():
        if random.random() < mutation_threshold1:
            idx1, idx2 = random.sample(range(len(value)), 2)
            value[idx1], value[idx2] = value[idx2], value[idx1]
    return mutated_child


def ARGA(
        job_data,
        population_size1=75,
        num_iterations1=1000,
        mutation_threshold1=0.5,
        elit_percentage1=0.6,
        FSG=generate_feasible_solution2,
        crossover=crossover1,
        mutate=mutate1N,
        get_schedule_time=get_schedule_time_AR_resch,
        reschedule=False,
        machine_start_time=None,
        iteration_report=False,
        a2=None,
        a2_info=None,
        disruption_time=None,
        init_time_schedule=None,
        obj_type="cmax",
        IR=3,
        flowtime_type="average",
        show_progress=False,
        bar_position=0):
    """
    Run the Adaptive Repair Genetic Algorithm (ARGA) for JSP Phase 1 optimization.

    Evolves a population of machine-operation sequences over ``num_iterations1``
    generations using selection, crossover, and mutation operators. Returns the
    best solution (sequence + objective value) found.

    Parameters
    ----------
    job_data : list of list of tuple or None
        Job data. Each job is a list of ``(machine_id, duration)`` tuples.
        ``None`` entries are skipped (finished/cancelled operations).
    population_size1 : int, optional
        Number of individuals in the population. Default 75.
    num_iterations1 : int, optional
        Number of generations. Default 1000.
    mutation_threshold1 : float, optional
        Probability that a mutation is applied to a child. Default 0.5.
    elit_percentage1 : float, optional
        Fraction of top solutions carried over without modification. Default 0.6.
    FSG : callable, optional
        Feasible solution generator used to initialise the population.
        Default ``generate_feasible_solution2``.
    crossover : callable, optional
        Crossover operator. Default ``crossover1``.
    mutate : callable, optional
        Mutation operator. Default ``mutate1N``.
    get_schedule_time : callable, optional
        Function that decodes a solution into a time schedule. Default
        ``get_schedule_time_AR_resch``.
    reschedule : bool, optional
        Enable rescheduling mode (respects ``machine_start_time``). Default ``False``.
    machine_start_time : dict[int, float] or None, optional
        Earliest available time per machine (rescheduling only). Default ``None``.
    obj_type : str, optional
        Objective to minimise: ``"cmax"``, ``"flowtime"``, or ``"cmax+flowtime"``.
        Default ``"cmax"``.
    IR : float, optional
        Impact ratio for combined objective. Default 3.
    flowtime_type : str, optional
        ``"average"`` or ``"total"`` flowtime. Default ``"average"``.
    show_progress : bool, optional
        Show a ``tqdm`` progress bar. Default ``False``.
    bar_position : int, optional
        ``tqdm`` bar position (useful when multiple bars run simultaneously).
        Default 0.

    Returns
    -------
    list
        ``[best_solution_dict, best_objective_value, iteration_best_list]``
        where ``best_solution_dict`` maps machine ID → ordered operation list.
    """
    if machine_start_time is None:
        machine_start_time = {}

    population = get_initial_population(job_data, get_schedule_time, population_size1, reschedule, machine_start_time, FSG, a2, a2_info, disruption_time, init_time_schedule, obj_type, IR, flowtime_type)

    best_solution = copy.deepcopy(population[0])

    iteration_best_solution_list = []

    pbar = tqdm(range(num_iterations1), desc="ARGA", unit="iter", disable=not show_progress, position=bar_position, leave=True)
    for iteration in pbar:
        child_pool = []

        for _ in range(population_size1 // 2):
            parent1, parent2 = parent_selection(population, elit_percentage1)

            if crossover in [crossover1, crossoverNR]:
                child1, child2 = crossover(parent1=parent1, parent2=parent2)
            else:
                child1, child2 = crossover(parent1=parent1, parent2=parent2, iteration=iteration, total_iterations=num_iterations1)

            child1 = mutate(child1, mutation_threshold1)
            child2 = mutate(child2, mutation_threshold1)

            schedule_time1 = get_schedule_time(job_data, child1, reschedule, machine_start_time, a2, a2_info, disruption_time)
            if schedule_time1 is not None:
                if reschedule and init_time_schedule is not None:
                    merged1 = {**init_time_schedule, **schedule_time1}
                    full_schedule_time1 = dict(sorted(merged1.items(), key=lambda x: x[0]))
                    time_objective1 = calculate_time_objective(full_schedule_time1, obj_type, IR, flowtime_type)
                else:
                    time_objective1 = calculate_time_objective(schedule_time1, obj_type, IR, flowtime_type)
                child_pool.append([child1, time_objective1])

            schedule_time2 = get_schedule_time(job_data, child2, reschedule, machine_start_time, a2, a2_info, disruption_time)
            if schedule_time2 is not None:
                if reschedule and init_time_schedule is not None:
                    merged2 = {**init_time_schedule, **schedule_time2}
                    full_schedule_time2 = dict(sorted(merged2.items(), key=lambda x: x[0]))
                    time_objective2 = calculate_time_objective(full_schedule_time2, obj_type, IR, flowtime_type)
                else:
                    time_objective2 = calculate_time_objective(schedule_time2, obj_type, IR, flowtime_type)
                child_pool.append([child2, time_objective2])

        combined_pool = population + child_pool
        sorted_pool = sorted(combined_pool, key=lambda x: x[1])

        if sorted_pool[0][1] < best_solution[1]:
            best_solution = copy.deepcopy(sorted_pool[0])

        population = copy.deepcopy(sorted_pool[:population_size1])

        pbar.set_postfix({f"Best {obj_type}": f"{best_solution[1]:.2f}"})

        iteration_best_solution_list.append(best_solution[1])

    pbar.close()

    if iteration_report is True:
        print(f'best solution in each iteration: {iteration_best_solution_list}')
    return best_solution
