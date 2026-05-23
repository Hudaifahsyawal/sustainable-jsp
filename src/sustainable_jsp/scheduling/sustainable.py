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
from sustainable_jsp.algorithms.carbon import GANBI
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


def dataframe_to_dict(df):
    """
    Convert a pandas DataFrame to a dictionary with row index as keys.
    Make sure the first column name is "ID".
    """
    df.set_index('ID', inplace=True)
    return df.to_dict(orient='index')


def sustainableJSP_resch(
        case1_reschedule: list[list[tuple[int, int] | None]],
        carbon_emission_data: dict[int, list[float]],
        EErate: list[list[float]],
        population_size1: int = 50,
        num_iterations1: int = 1500,
        mutation_threshold1: float = 0.5,
        elit_percentage1: float = 0.6,
        FSG=generate_feasible_solution1_resch,
        crossover=crossoverNR,
        mutate=mutate1N,
        get_schedule_time=get_schedule_time_AR_resch,
        num_iterations2: int = 1500,
        population_size2: int = 75,
        elit_percentage2: float = 0.6,
        mutation_threshold2: float = 0.5,
        weight: float = 0.75,
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
        a1: list[tuple[int, int]] | None = None,
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
        obj_type: str = "cmax",
        workload_obj_type: str = "variance",
        IR: float = 3,
        flowtime_type: str = "average",
        show_progress: bool = False
        ) -> dict:
    """
    Solve the Sustainable Job Shop Scheduling Problem using three-phase optimization.

    Implements a three-phase pipeline:

    - **Phase 1** — ARGA (genetic algorithm) optimizes machine-operation sequences to
      minimize the time objective (Cmax, flowtime, or combined).
    - **Phase 2** — GANBI optimizes speed levels for each operation, balancing the time
      objective against total carbon emissions using a weighted-sum scalarization.
    - **Phase 3** — AWBO (ant colony) assigns operators to operations to balance workload.

    Supports both initial scheduling (``reschedule=False``) and rescheduling scenarios
    (``reschedule=True``) where some operations are already finished or in progress.

    Parameters
    ----------
    case1_reschedule : list of list of tuple or None
        Job data. Each job is a list of ``(machine_id, duration)`` tuples.
        ``None`` entries mark finished or cancelled operations (used in rescheduling).
    carbon_emission_data : dict[int, list[float]]
        Carbon emission rates per machine. Key = machine ID (1-indexed).
        Each value is a list of rates for different speed levels.
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
    num_iterations2 : int, optional
        Number of GA generations for Phase 2. Default 1500.
    population_size2 : int, optional
        GA population size for Phase 2. Default 75.
    elit_percentage2 : float, optional
        Fraction of elite solutions preserved in Phase 2. Default 0.6.
    mutation_threshold2 : float, optional
        Mutation probability for Phase 2. Default 0.5.
    weight : float, optional
        Weight for the time objective in Phase 2 scalarization (``1-weight`` for
        carbon). Range [0, 1]. Default 0.75.
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
        ``"cmax+flowtime"``. Default ``"cmax"``.
    workload_obj_type : str, optional
        Workload objective for Phase 3. One of ``"variance"``, ``"mean"``,
        ``"maximum"``, ``"std_dev"``, ``"coef_variation"``. Default ``"variance"``.
    IR : float, optional
        Impact ratio used in combined objective calculation. Default 3.
    flowtime_type : str, optional
        Flowtime aggregation method. One of ``"average"``, ``"total"``. Default ``"average"``.
    show_progress : bool, optional
        Show ``tqdm`` progress bars for Phase 1 and Phase 2. Default ``False``.
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

        - ``"case_reschedule"`` : list — final job data (with completed ops restored)
        - ``"solution"`` : dict[int, list] — machine → ordered operation list
        - ``"speedlevel_optimum"`` : list — speed level per operation (Phase 2 result)
        - ``"time_schedule"`` : dict — ``(job_id, op_id)`` → schedule info dict
        - ``"fair_workload"`` : dict[int, list] — operator → assigned operations
        - ``"workload_list"`` : dict[int, list] — cumulative workload history per operator
        - ``"event_times"`` : dict[int, list] — event timestamps per operator
        - ``"matrix_performance"`` : dict — ``time_objective``, ``TCE``, workload stats
        - ``"bound"`` : dict — LB/UB for time and carbon objectives

    Examples
    --------
    >>> import json
    >>> from sustainable_jsp import sustainableJSP_resch
    >>> from sustainable_jsp.algorithms.workload import calculate_EErate
    >>> with open("examples/Dataset/Dataset_Jobs/J14D30M5.json") as f:
    ...     jobs_data = [[tuple(op) for op in job] for job in json.load(f)]
    >>> with open("examples/Dataset/Dataset_Carbon/Carbon_data_J14D30M5.json") as f:
    ...     carbon = {int(k): v for k, v in json.load(f).items()}
    >>> with open("examples/Dataset/Dataset_Operator/operators_data_J14D30M5.json") as f:
    ...     EErate = calculate_EErate({int(k): v for k, v in json.load(f).items()})
    >>> result = sustainableJSP_resch(jobs_data, carbon, EErate,
    ...                               num_iterations1=500, num_iterations2=500,
    ...                               show_progress=True)
    >>> print(result["matrix_performance"])
    """
    print(f"\n {'-'*20}\nStart optimization phase 1\n{'-'*20}")
    # print(f"check nilai a2_info: {a2_info}")

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

    print(f"\n {'-'*20}\nStart optimization phase 2\n{'-'*20}")

    result_GANBI = GANBI(
        case1_reschedule,
        time_schedule_phase1,
        case1_solution_phase1[0],
        carbon_emission_data,
        num_iterations2,
        population_size2,
        elit_percentage2,
        mutation_threshold2,
        weight,
        visualization,
        reschedule,
        machine_start_time,
        x_start,
        x_end,
        a2,
        a2_info,
        disruption_time,
        init_time_schedule,
        obj_type,
        IR,
        flowtime_type,
        show_progress=show_progress)

    speedlevel_optimum = result_GANBI["speedlevel_optimum"]
    time_schedule_phase2 = result_GANBI["final_schedule_time"]
    matrix_performance = result_GANBI["matrix_performance"]
    bound = result_GANBI["bound"]

    print(f"\n {'-'*20}\nStart optimization phase 3\n{'-'*20}")

    fair_workload = AWBO(
        EErate,
        case1_reschedule,
        time_schedule_phase2,
        case1_solution_phase1[0],
        prob,
        coloni_size,
        alpha,
        beta,
        reschedule,
        initial_workload,
        initial_ready_time,
        a2,
        a2_info,
        obj_type=workload_obj_type)

    best_workload_list, best_event_times, best_current_workload = cumulative_workload_resch(
        fair_workload,
        EErate,
        time_schedule_phase2,
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

    for (j, o), info in time_schedule_phase2.items():
        m_id, _ = case1_reschedule[j - 1][o - 1]
        opr_id = operation_to_operator.get((j, o))
        info["machine_id"] = m_id
        info["operator_id"] = opr_id

    if reschedule:
        time_schedule_phase2 = dict(sorted({**init_time_schedule, **time_schedule_phase2}.items()))

        a1a2 = a1 + a2
        a1a2_set = set(a1a2)

        for (j, o) in a1a2_set:
            op_key = (j, o)
            if op_key in time_schedule_phase2:
                op_info = time_schedule_phase2[op_key]
                machine_id = op_info.get('machine_id')
                original_duration = op_info.get('original_duration')
                if machine_id is not None and original_duration is not None:
                    case1_reschedule[j - 1][o - 1] = (machine_id, original_duration)

        for m_id, op_list in init_case1_solution.items():
            if m_id in case1_solution_phase1[0]:
                case1_solution_phase1[0][m_id] = op_list + case1_solution_phase1[0][m_id]
            else:
                case1_solution_phase1[0][m_id] = op_list

        for opr_id, init_tasks in init_fair_operator_assignment.items():
            if opr_id in fair_workload:
                fair_workload[opr_id] = init_tasks + fair_workload[opr_id]
            else:
                fair_workload[opr_id] = init_tasks.copy()

        for opr_id, wl in init_workload_list.items():
            if opr_id in best_workload_list:
                best_workload_list[opr_id] = wl + best_workload_list[opr_id]
            else:
                best_workload_list[opr_id] = wl.copy()

        for opr_id, et in init_event_times.items():
            if opr_id in best_event_times:
                best_event_times[opr_id] = et + best_event_times[opr_id]
            else:
                best_event_times[opr_id] = et.copy()

    best_speedlevel_all = []
    for op in time_schedule_phase2.keys():
        sl = time_schedule_phase2[op].get('speed_level')
        best_speedlevel_all.append(sl)

    if visualization is True:
        create_gantt_chart_final(
            time_schedule_phase2,
            case1_solution_phase1[0],
            operator_assignment=fair_workload,
            speedlevel_list=best_speedlevel_all,
            x_start=x_start,
            x_end=x_end,
            title=title,
            save=save)

    print(f"\n3 {'-' * 20}\nOptimization result\n{'-' * 20}")

    _schedule_rows = []
    for (j, o), info in sorted(
        time_schedule_phase2.items(),
        key=lambda x: (x[1]["start_time"], x[1]["duration"]),
    ):
        _orig_dur = info.get("original_duration")
        if _orig_dur is None:
            _cell = case1_reschedule[j - 1][o - 1]
            if _cell:
                _orig_dur = _cell[1]
        _schedule_rows.append(
            {
                "operation id": f"({j},{o})",
                "start time": info.get("start_time"),
                "duration": info.get("duration"),
                "finished time": info.get("finished_time"),
                "original duration": _orig_dur,
                "machine id": info.get("machine_id"),
                "speed level": info.get("speed_level"),
                "operator id": info.get("operator_id"),
            }
        )
    _schedule_df = pd.DataFrame(_schedule_rows)
    print("\nSchedule table (all operations, sorted by start time)")
    with pd.option_context(
        "display.max_rows", None,
        "display.width", None,
        "display.max_columns", None,
    ):
        print(_schedule_df.to_string(index=False))

    print("\nOperation assignment (job_id, operation_id, machine_id)")
    updated_operator = {}
    for operator_id, task_list in fair_workload.items():
        updated_operator[operator_id] = []
        for job_id, operation_id in task_list:
            if (1 <= job_id <= len(case1_reschedule) and
                    1 <= operation_id <= len(case1_reschedule[job_id - 1]) and
                    case1_reschedule[job_id - 1][operation_id - 1]):
                machine_id = case1_reschedule[job_id - 1][operation_id - 1][0]
                updated_operator[operator_id].append((job_id, operation_id, machine_id))
    _worker_series = {
        f"worker {wid}": pd.Series([str(t) for t in tasks], dtype=object)
        for wid, tasks in sorted(updated_operator.items(), key=lambda x: x[0])
    }
    _op_assign_df = pd.DataFrame(_worker_series).fillna("")
    with pd.option_context(
        "display.max_rows", None,
        "display.width", None,
        "display.max_columns", None,
    ):
        print(_op_assign_df.to_string())

    print("\nperformance of the Ant Work Balance algorithm")
    mean, variance, maximum, std_dev, coef_variation = workload_statistic(best_current_workload)
    print(f'mean: {mean}, \nvariance: {variance}, \nmaximum: {maximum}, \nstd_dev: {std_dev}, \ncoef_variation: {coef_variation}')

    matrix_performance["workload_mean"] = mean
    matrix_performance["workload_variance"] = variance
    matrix_performance["workload_maximum"] = maximum
    matrix_performance["workload_std_dev"] = std_dev
    matrix_performance["workload_coef_variation"] = coef_variation

    print("\nPerformance Matrix:")
    for matrix, value in matrix_performance.items():
        print(f"  '{matrix}' : {value}")

    return {
        "case_reschedule": case1_reschedule,
        "solution": case1_solution_phase1[0],
        "speedlevel_optimum": best_speedlevel_all,
        "time_schedule": time_schedule_phase2,
        "fair_workload": fair_workload,
        "workload_list": best_workload_list,
        "event_times": best_event_times,
        "matrix_performance": matrix_performance,
        "bound": bound,
    }
