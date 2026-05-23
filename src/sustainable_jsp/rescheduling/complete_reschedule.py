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
        beta=2,
        show_progress: bool = False):
    """
    Completely reschedule when new jobs arrive at a disruption time.

    Classifies existing operations into:

    - **A1** — finished operations (set to ``None``)
    - **A2** — operations in progress at ``disruption_time``
    - **A3** — unprocessed operations (remain in schedule)

    Then appends the new arriving jobs and re-optimizes using either
    :func:`sustainableJSP_resch` or :func:`dual_resource_JSP`.

    Parameters
    ----------
    new_job : list of list of tuple
        New arriving jobs. Same format as ``case1``:
        each job is a list of ``(machine_id, duration)`` tuples.
    disruption_time : float
        Time at which the new jobs arrive and rescheduling is triggered.
    case1 : list of list of tuple
        Original job data from the initial schedule.
    case1_solution_optimum : dict[int, list]
        Phase 1 solution from the initial schedule (machine → operation list).
    Time_schedule_P2 : dict
        Time schedule from the initial schedule. Keys are ``(job_id, op_id)``.
    fair_operator_assignment : dict[int, list]
        Operator assignment from the initial schedule.
    workload_list : dict[int, list]
        Cumulative workload history per operator from the initial schedule.
    event_times : dict[int, list]
        Event timestamps per operator from the initial schedule.
    EErate : list of list of float
        Energy expenditure rate table (operator × machine).
    carbon_emission_data : dict[int, list[float]] or None, optional
        Required when ``dual_resource=False``. Default ``None``.
    dual_resource : bool, optional
        If ``True``, use :func:`dual_resource_JSP` (no carbon). Default ``False``.
    weight : float, optional
        Time/carbon trade-off weight for Phase 2 (sustainable mode). Default 0.75.
    visualization : bool, optional
        Show Gantt chart after rescheduling. Default ``True``.
    obj_type : str, optional
        Time objective. One of ``"cmax"``, ``"flowtime"``, ``"cmax+flowtime"``.
        Default ``"cmax+flowtime"``.
    workload_obj_type : str, optional
        Workload objective for Phase 3. Default ``"variance"``.
    IR : float, optional
        Impact ratio for combined objective. Default 3.
    flowtime_type : str, optional
        ``"average"`` or ``"total"`` flowtime. Default ``"average"``.
    num_iterations1, num_iterations2 : int, optional
        GA iteration counts for Phase 1 and Phase 2. Default 1500.
    population_size1, population_size2 : int, optional
        GA population sizes. Default 75.
    elit_percentage1, elit_percentage2 : float, optional
        Elite fractions. Default 0.6.
    coloni_size : int, optional
        Ant colony size for Phase 3. Default 450.
    alpha, beta : float, optional
        Ant colony parameters. Default 10 and 2.
    show_progress : bool, optional
        Show ``tqdm`` progress bars. Default ``False``.

    Returns
    -------
    dict
        Contains ``"mode"`` (``"sustainable"`` or ``"dual_resource"``) plus all
        keys from :func:`sustainableJSP_resch` or :func:`dual_resource_JSP`.

    Raises
    ------
    ValueError
        If ``dual_resource=False`` and ``carbon_emission_data`` is ``None``.
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
            flowtime_type=flowtime_type,
            show_progress=show_progress,
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
            flowtime_type=flowtime_type,
            show_progress=show_progress,
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
    Completely reschedule when specific operations must be reworked.

    The disruption time is inferred from the ``finished_time`` of the rework
    operations. All completed/in-progress operations are removed from the
    schedule **except** the rework operations themselves, which are re-inserted
    for re-optimization.

    Parameters
    ----------
    rework_operation : list of tuple
        List of ``(job_id, op_id)`` tuples identifying operations to redo.
        All operations must have the same ``finished_time``; otherwise the
        function returns ``None``.
    case1 : list of list of tuple
        Original job data from the initial schedule.
    case1_solution_optimum : dict[int, list]
        Phase 1 solution from the initial schedule.
    Time_schedule_P2 : dict
        Time schedule from the initial schedule. Keys are ``(job_id, op_id)``.
    fair_operator_assignment : dict[int, list]
        Operator assignment from the initial schedule.
    workload_list : dict[int, list]
        Cumulative workload history per operator.
    event_times : dict[int, list]
        Event timestamps per operator.
    EErate : list of list of float
        Energy expenditure rate table (operator × machine).
    carbon_emission_data : dict[int, list[float]] or None, optional
        Required when ``dual_resource=False``. Default ``None``.
    dual_resource : bool, optional
        Use :func:`dual_resource_JSP` instead of :func:`sustainableJSP_resch`.
        Default ``False``.
    weight : float, optional
        Time/carbon trade-off weight. Default 0.75.
    visualization : bool, optional
        Show Gantt chart after rescheduling. Default ``True``.
    obj_type : str, optional
        Time objective type. Default ``"cmax+flowtime"``.
    workload_obj_type : str, optional
        Workload objective for Phase 3. Default ``"variance"``.
    IR : float, optional
        Impact ratio. Default 3.
    flowtime_type : str, optional
        ``"average"`` or ``"total"`` flowtime. Default ``"average"``.
    num_iterations1, num_iterations2 : int, optional
        GA iteration counts. Default 1500.
    population_size1, population_size2 : int, optional
        GA population sizes. Default 75.
    elit_percentage1, elit_percentage2 : float, optional
        Elite fractions. Default 0.6.
    coloni_size : int, optional
        Ant colony size for Phase 3. Default 450.
    alpha, beta : float, optional
        Ant colony parameters. Default 10 and 2.

    Returns
    -------
    dict or None
        Contains ``"mode"`` plus all keys from the underlying scheduler,
        or ``None`` if rework operations have mismatched finish times.

    Raises
    ------
    ValueError
        If ``dual_resource=False`` and ``carbon_emission_data`` is ``None``.
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
        beta=2,
        workload_obj_type="variance",
        show_progress: bool = False):
    """
    Completely reschedule when one or more jobs are cancelled at a disruption time.

    Operations belonging to the cancelled jobs are removed from the schedule.
    Finished (A1) and in-progress (A2) operations of remaining jobs are set to
    ``None`` to preserve their results. The remaining unprocessed work is
    re-optimized.

    Parameters
    ----------
    cancel_job_id : int or list of int or set of int
        Job ID(s) to cancel (1-indexed). Accepts a single int, list, or set.
    disruption_time : float
        Time at which the cancellation is triggered.
    case1 : list of list of tuple
        Original job data from the initial schedule.
    case1_solution_optimum : dict[int, list]
        Phase 1 solution from the initial schedule.
    Time_schedule_P2 : dict
        Time schedule from the initial schedule. Keys are ``(job_id, op_id)``.
    fair_operator_assignment : dict[int, list]
        Operator assignment from the initial schedule.
    workload_list : dict[int, list]
        Cumulative workload history per operator.
    event_times : dict[int, list]
        Event timestamps per operator.
    EErate : list of list of float
        Energy expenditure rate table (operator × machine).
    carbon_emission_data : dict[int, list[float]] or None, optional
        Required when ``dual_resource=False``. Default ``None``.
    dual_resource : bool, optional
        Use :func:`dual_resource_JSP` instead of :func:`sustainableJSP_resch`.
        Default ``False``.
    weight : float, optional
        Time/carbon trade-off weight. Default 0.75.
    visualization : bool, optional
        Show Gantt chart after rescheduling. Default ``True``.
    obj_type : str, optional
        Time objective type. Default ``"cmax"``.
    workload_obj_type : str, optional
        Workload objective for Phase 3. Default ``"variance"``.
    IR : float, optional
        Impact ratio. Default 3.
    flowtime_type : str, optional
        ``"average"`` or ``"total"`` flowtime. Default ``"average"``.
    num_iterations1, num_iterations2 : int, optional
        GA iteration counts. Default 1500.
    population_size1, population_size2 : int, optional
        GA population sizes. Default 75.
    elit_percentage1, elit_percentage2 : float, optional
        Elite fractions. Default 0.6.
    coloni_size : int, optional
        Ant colony size for Phase 3. Default 450.
    alpha, beta : float, optional
        Ant colony parameters. Default 10 and 2.
    show_progress : bool, optional
        Show ``tqdm`` progress bars. Default ``False``.

    Returns
    -------
    dict
        Contains ``"mode"`` (``"sustainable"`` or ``"dual_resource"``) plus all
        keys from :func:`sustainableJSP_resch` or :func:`dual_resource_JSP`.

    Raises
    ------
    ValueError
        If ``dual_resource=False`` and ``carbon_emission_data`` is ``None``.
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
            flowtime_type=flowtime_type,
            show_progress=show_progress,
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
            workload_obj_type=workload_obj_type,
            IR=IR,
            flowtime_type=flowtime_type,
            show_progress=show_progress,
        )
        return {"mode": "sustainable", **out}
