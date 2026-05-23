import random
import matplotlib.pyplot as plt


def create_gantt_chart_final(
    schedule_time,
    machine_order,
    operator_assignment=None,
    speedlevel_list=None,
    x_start=None,
    x_end=None,
    title=None,
    save=False
):
    """
    Create a Gantt chart visualization of a job shop schedule.

    Renders a horizontal bar chart where each bar represents one operation on its
    assigned machine. Optionally annotates operator assignments (above each bar)
    and speed levels (below each bar).

    Parameters
    ----------
    schedule_time : dict
        Keys are ``(job_id, op_id)`` tuples. Each value is a dict with at least
        ``"start_time"`` and ``"duration"`` entries.
    machine_order : dict[int, list]
        Machine → list of ``(job_id, op_id)`` tuples (operation sequence per machine).
    operator_assignment : dict[int, list] or None, optional
        Operator → list of ``(job_id, op_id)`` tasks. If provided, operator IDs
        are annotated above each bar. Default ``None``.
    speedlevel_list : list or None, optional
        Speed level per operation (same order as ``schedule_time`` keys).
        If provided, speed levels are annotated below each bar. Default ``None``.
    x_start : float or None, optional
        Left x-axis limit. Default ``None`` (auto).
    x_end : float or None, optional
        Right x-axis limit. Default ``None`` (auto).
    title : str or None, optional
        Chart title. Default ``None`` (uses a generic title).
    save : bool, optional
        If ``True``, save the chart as ``{title}.pdf``. Default ``False``.

    Examples
    --------
    >>> from sustainable_jsp import create_gantt_chart_final
    >>> create_gantt_chart_final(
    ...     result["time_schedule"],
    ...     result["solution"],
    ...     operator_assignment=result["fair_workload"],
    ...     speedlevel_list=result["speedlevel_optimum"],
    ...     title="Initial schedule",
    ... )
    """
    if title is None:
        title = 'Gantt chart of scheduling or rescheduling plan'

    total_job = max(j for (j, _) in schedule_time.keys())
    fig, ax = plt.subplots(figsize=(20, 8))

    machine_order_sorted = {m: machine_order[m] for m in sorted(machine_order.keys())}

    random.seed(42)

    job_colors = [
        "#" + ''.join(random.choice('0123456789ABCDEF') for _ in range(6))
        for _ in range(total_job)
    ]

    if operator_assignment is not None:
        random.seed(9)
        operator_colors = [
            "#" + "".join(random.choice("0123456789ABCDEF") for _ in range(6))
            for _ in range(len(operator_assignment))
        ]
    else:
        operator_colors = None

    operation_to_operator = {}
    if operator_assignment is not None:
        for op_id, ops in operator_assignment.items():
            for task in ops:
                operation_to_operator[task] = op_id

    operation_to_speed = {}
    if speedlevel_list is not None:
        op_list = list(schedule_time.keys())
        for i, op in enumerate(op_list):
            if i < len(speedlevel_list):
                operation_to_speed[op] = speedlevel_list[i]

    for machine_id, job_list in machine_order_sorted.items():
        for job_id, operation_index in job_list:
            key = (job_id, operation_index)
            start_time = schedule_time[key]["start_time"]
            duration = schedule_time[key]["duration"]

            job_color = job_colors[job_id - 1]

            op_id = operation_to_operator.get(key, None)
            if operator_colors is not None and op_id is not None and 1 <= op_id <= len(operator_colors):
                op_color = operator_colors[op_id - 1]
            else:
                op_color = 'black'

            speed = operation_to_speed.get(key, None)

            ax.barh(machine_id, duration, left=start_time, color=job_color, edgecolor='black')

            ax.text(start_time + duration / 2, machine_id, f'J{job_id}-{operation_index}',
                    ha='center', va='center', color='black', fontsize=10)

            if op_id is not None:
                ax.text(start_time + duration / 2, machine_id + 0.4, f'O{op_id}',
                        ha='center', va='bottom', color=op_color, fontsize=8, fontweight='bold')

            if speed is not None:
                ax.text(start_time + duration / 2, machine_id - 0.15, f'SL: {speed}',
                        ha='center', va='top', color='black', fontsize=10)

    ax.set_xlabel('Time horizon (minutes)', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.set_yticks(sorted(machine_order_sorted.keys()))
    ax.set_yticklabels([f"Machine {i}" for i in sorted(machine_order_sorted.keys())])
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if x_start is not None or x_end is not None:
        ax.set_xlim(left=x_start, right=x_end)

    if save is True:
        plt.savefig(f"{title}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    rng_state = random.getstate()
    random.setstate(rng_state)
