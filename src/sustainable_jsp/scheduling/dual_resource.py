from __future__ import annotations

import copy
import pandas as pd

from sustainable_jsp.core.schedule import (
    generate_feasible_solution1_resch,
    generate_feasible_solution2_resch,
    get_schedule_time_AR_resch,
)
from sustainable_jsp.core.performance_matrix import calculate_time_objective
from sustainable_jsp.core.visualization import create_gantt_chart_final
from sustainable_jsp.algorithms.genetic import (
    ARGA,
    crossover1, crossover1P, crossoverNR, crossoverNRP,
    mutate1N, mutate1R,
)
from sustainable_jsp.algorithms.workload import (
    AWBO,
    cumulative_workload_resch,
    workload_statistic,
)


def complete_machine_keys(solution_phase1, EErate):
    """
    Ensure all machine keys [1..len(EErate)] exist in solution_phase1[0].
    Missing keys will be added with empty list.
    """
    machine_dict = solution_phase1[0]
    num_machines = len(EErate)

    for machine_id in range(1, num_machines + 1):
        machine_dict.setdefault(machine_id, [])

    return solution_phase1


def dual_resource_JSP(
        case1_reschedule: list[list[tuple[int, int] | None]],
        EErate: list[list[float]],
        population_size1: int = 50,
        num_iterations1: int = 1500,
        mutation_threshold1: float = 0.5,
        elit_percentage1: float = 0.6,
        FSG=generate_feasible_solution2_resch,
        crossover=crossoverNR,
        mutate=mutate1N,
        get_schedule_time=get_schedule_time_AR_resch,
        visualization: bool = True,
        reschedule: bool = False,
        machine_start_time: dict[int, float] | None = None,
        prob: float = 0.95,
        coloni_size: int = 450,
        alpha: float = 10,
        beta: float = 2,
        initial_workload: dict[int, float] | None = None,
        initial_ready_time: dict[int, float] | None = None,
        x_start: float | None = None,
        x_end: float | None = None,
        a2: list[tuple[int, int]] | None = None,
        a2_info: dict[tuple[int, int], dict[str, float]] | None = None,
        disruption_time: float | None = None,
        init_time_schedule: dict[tuple[int, int], dict[str, float]] | None = None,
        init_case1: dict[int, list[tuple[int, int]]] | None = None,
        init_case1_solution: dict[int, list[tuple[int, int]]] | None = None,
        init_fair_operator_assignment: dict[int, list[tuple[int, int]]] | None = None,
        init_workload_list: dict[int, list[float]] | None = None,
        init_event_times: dict[int, list[float]] | None = None,
        title: str | None = None,
        save: bool = False,
        obj_type: str = "cmax+flowtime",
        IR: float = 3,
        flowtime_type: str = "average",
        show_progress: bool = False,
        ) -> dict:
    """
    Solve the Dual-Resource Constrained Job Shop Scheduling Problem using two-phase optimization.

    Implements a two-phase pipeline (skipping carbon-aware Phase 2):

    - **Phase 1** — ARGA (genetic algorithm) optimizes machine-operation sequences.
    - **Phase 3** — AWBO (ant colony) assigns operators to balance workload.

    Use this function when carbon emission data is not available or not relevant.
    For the full sustainable three-phase pipeline, use :func:`sustainableJSP_resch`.

    Parameters
    ----------
    case1_reschedule : list of list of tuple or None
        Job data. Each job is a list of ``(machine_id, duration)`` tuples.
        ``None`` entries mark finished or cancelled operations (rescheduling).
    EErate : list of list of float
        Energy expenditure rate table. ``EErate[operator][machine]`` gives the EE
        rate for that operator-machine combination.
    population_size1 : int, optional
        GA population size for Phase 1. Default 50.
    num_iterations1 : int, optional
        Number of GA generations for Phase 1. Default 1500.
    mutation_threshold1 : float, optional
        Mutation probability for Phase 1. Default 0.5.
    elit_percentage1 : float, optional
        Fraction of elite solutions preserved in Phase 1. Default 0.6.
    visualization : bool, optional
        If ``True``, display the Gantt chart after optimization. Default ``True``.
    reschedule : bool, optional
        Set to ``True`` when called from a rescheduling context. Default ``False``.
    machine_start_time : dict[int, float] or None, optional
        Earliest available time per machine (rescheduling only). Default ``None``.
    prob : float, optional
        Ant selection probability parameter for AWBO. Default 0.95.
    coloni_size : int, optional
        Number of ant solutions generated in Phase 3. Default 450.
    alpha : float, optional
        Influence of pheromone in operator selection. Default 10.
    beta : float, optional
        Influence of heuristic in operator selection. Default 2.
    obj_type : str, optional
        Time objective type. One of ``"cmax"``, ``"flowtime"``,
        ``"cmax+flowtime"``. Default ``"cmax+flowtime"``.
    IR : float, optional
        Impact ratio used in combined objective calculation. Default 3.
    flowtime_type : str, optional
        Flowtime aggregation method. One of ``"average"``, ``"total"``. Default ``"average"``.
    show_progress : bool, optional
        Show ``tqdm`` progress bar for Phase 1. Default ``False``.
    title : str or None, optional
        Title for the Gantt chart. Default ``None``.
    save : bool, optional
        Save the Gantt chart to a PDF file. Default ``False``.
    x_start, x_end : float or None, optional
        X-axis limits for the Gantt chart. Default ``None``.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``"case_reschedule"`` : list — final job data
        - ``"solution"`` : dict[int, list] — machine → ordered operation list
        - ``"speedlevel_optimum"`` : ``None`` (no Phase 2)
        - ``"time_schedule"`` : dict — ``(job_id, op_id)`` → schedule info dict
        - ``"fair_workload"`` : dict[int, list] — operator → assigned operations
        - ``"workload_list"`` : dict[int, list] — cumulative workload history per operator
        - ``"event_times"`` : dict[int, list] — event timestamps per operator
        - ``"matrix_performance"`` : dict — ``time_objective``, workload stats

    Examples
    --------
    >>> from sustainable_jsp import dual_resource_JSP
    >>> from sustainable_jsp.algorithms.workload import calculate_EErate
    >>> EErate = calculate_EErate(operator_data)
    >>> result = dual_resource_JSP(jobs_data, EErate,
    ...                            num_iterations1=500, show_progress=True)
    """
    print(f"\n {'-'*20}\nStart optimization phase 1\n{'-'*20}")

    case1_solution_phase1 = ARGA(
        case1_reschedule,
        population_size1,
        num_iterations1,
        mutation_threshold1,
        elit_percentage1,
        FSG,
        crossover,
        mutate,
        get_schedule_time,
        reschedule=reschedule,
        machine_start_time=machine_start_time,
        a2=a2,
        a2_info=a2_info,
        disruption_time=disruption_time,
        init_time_schedule=init_time_schedule,
        obj_type=obj_type,
        IR=IR,
        flowtime_type=flowtime_type,
        show_progress=show_progress)

    time_schedule_phase1 = get_schedule_time(
        case1_reschedule,
        case1_solution_phase1[0],
        reschedule,
        machine_start_time,
        a2,
        a2_info,
        disruption_time)

    matrix_performance = {"time_objective": case1_solution_phase1[1]}

    print(f"\n {'-'*20}\nStart optimization phase 3\n{'-'*20}")

    fair_workload = AWBO(
        EErate,
        case1_reschedule,
        time_schedule_phase1,
        case1_solution_phase1[0],
        prob,
        coloni_size,
        alpha,
        beta,
        reschedule,
        initial_workload,
        initial_ready_time,
        a2,
        a2_info)

    best_workload_list, best_event_times, best_current_workload = cumulative_workload_resch(
        fair_workload,
        EErate,
        time_schedule_phase1,
        case1_reschedule,
        visualization,
        reschedule,
        initial_workload,
        a2,
        a2_info)

    operation_to_operator = {
        task: opr_id
        for opr_id, ops in fair_workload.items()
        for task in ops
    }

    for (j, o), info in time_schedule_phase1.items():
        m_id, _ = case1_reschedule[j - 1][o - 1]
        opr_id = operation_to_operator.get((j, o))
        info["machine_id"] = m_id
        info["operator_id"] = opr_id

    case1_solution_phase1 = complete_machine_keys(case1_solution_phase1, EErate)

    if a2 is not None:
        for op in a2:
            time_schedule_phase1[op] = dict(a2_info[op])
        time_schedule_phase1 = dict(sorted(time_schedule_phase1.items()))

        for op in a2:
            j, o = op
            m = time_schedule_phase1[op]['machine_id']
            if m is not None and m in case1_solution_phase1[0]:
                case1_solution_phase1[0][m].insert(0, op)

            opr_id = time_schedule_phase1[op]['operator_id']
            if opr_id is not None:
                if opr_id not in fair_workload:
                    fair_workload[opr_id] = []
                fair_workload[opr_id].insert(0, op)

                st = a2_info[op]["start_time"]
                ft = a2_info[op]["finished_time"]
                base = float(initial_workload.get(opr_id, 0.0)) if initial_workload else 0.0

                if opr_id not in best_event_times:
                    best_event_times[opr_id] = []
                if opr_id not in best_workload_list:
                    best_workload_list[opr_id] = []

                best_event_times[opr_id] = [st, ft] + best_event_times[opr_id]
                best_workload_list[opr_id] = [base, base] + best_workload_list[opr_id]

        for (j, o) in a2:
            machine_id = a2_info[(j, o)]['machine_id']
            duration = a2_info[(j, o)]['duration']
            if duration <= 0:
                continue
            if 1 <= j <= len(case1_reschedule) and 1 <= o <= len(case1_reschedule[j - 1]):
                case1_reschedule[j - 1][o - 1] = (machine_id, duration)

    if visualization:
        create_gantt_chart_final(
            time_schedule_phase1,
            case1_solution_phase1[0],
            operator_assignment=fair_workload,
            speedlevel_list=None,
            x_start=x_start,
            x_end=x_end,
            title=title,
            save=save)

    print(f"\n3 {'-' * 20}\nOptimization result\n{'-' * 20}")

    print("\noperation sequence in each machine (job_id, operation_id)")
    for key, value in case1_solution_phase1[0].items():
        print(f'Machine {key}: {value}')

    print("\ntime schedule for each operation")
    sorted_schedule = sorted(time_schedule_phase1.items(), key=lambda x: (x[1]['start_time'], x[1]['duration']))
    for value in sorted_schedule:
        print(f'Operation {value}')

    print("\noperation assignment for each operator (job_id, operation_id, machine_id)")
    updated_operator = {}
    for operator_id, task_list in fair_workload.items():
        updated_operator[operator_id] = []
        for job_id, operation_id in task_list:
            if (1 <= job_id <= len(case1_reschedule) and
                    1 <= operation_id <= len(case1_reschedule[job_id - 1]) and
                    case1_reschedule[job_id - 1][operation_id - 1]):
                machine_id = case1_reschedule[job_id - 1][operation_id - 1][0]
                updated_operator[operator_id].append((job_id, operation_id, machine_id))

    print("\noperation assignment for each operator (job_id, operation_id, machine_id)")
    for key, value in updated_operator.items():
        print(f'worker {key}: {value}')

    print("\nperformance of the Ant Work Balance algorithm")
    mean, variance, maximum, std_dev, coef_variation = workload_statistic(best_current_workload)
    print(f'mean: {mean}, \nvariance: {variance}, \nmaximum: {maximum}')

    matrix_performance["workload_mean"] = mean
    matrix_performance["workload_variance"] = variance
    matrix_performance["workload_maximum"] = maximum
    matrix_performance["workload_std_dev"] = std_dev
    matrix_performance["workload_coef_variation"] = coef_variation

    print("matrix_performance:")
    for _k, _v in matrix_performance.items():
        print(f"  '{_k}' : {_v}")

    return {
        "case_reschedule": case1_reschedule,
        "solution": case1_solution_phase1[0],
        "speedlevel_optimum": None,
        "time_schedule": time_schedule_phase1,
        "fair_workload": fair_workload,
        "workload_list": best_workload_list,
        "event_times": best_event_times,
        "matrix_performance": matrix_performance,
    }
