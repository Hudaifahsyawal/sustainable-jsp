"""
Basic job shop scheduling with ARGA (Phase 1 only).

Loads instance data from examples/Dataset/, runs ARGA, decodes the schedule,
reports Cmax and average flow time, lower bounds, and shows a Gantt chart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Repo root (sustainableJSP5); allow `python examples/...` without pip install -e .
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    try:
        import sustainable_jsp  # noqa: F401
    except ModuleNotFoundError:
        sys.path.insert(0, str(_SRC))

from sustainable_jsp.algorithms.genetic import ARGA
from sustainable_jsp.core.schedule import (
    calculate_cmax_lower_bound,
    generate_feasible_solution2,
    get_schedule_time_AR_resch,
)
from sustainable_jsp.core.performance_matrix import (
    calculate_time_objective,
    job_duration,
)
from sustainable_jsp.core.visualization import create_gantt_chart_final

# Default instance (same naming as Dataset_2draft)
INSTANCE = "J14D30M5"

EXAMPLES_DIR = Path(__file__).resolve().parent
DATASET_DIR = EXAMPLES_DIR / "Dataset"
JOBS_PATH = DATASET_DIR / "Dataset_Jobs" / f"{INSTANCE}.json"

# ARGA parameters
POPULATION_SIZE = 50
NUM_ITERATIONS = 1500
MUTATION_THRESHOLD = 0.5
ELIT_PERCENTAGE = 0.6
RESCHEUDLE = False
OBJ_TYPE = "cmax"
SHOW_PROGRESS = True

def load_jobs_data(jobs_path: Path) -> list:
    with jobs_path.open("r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    jobs_data = [[tuple(operation) for operation in job] for job in loaded_data]
    return jobs_data


def print_schedule_detail(time_schedule: dict) -> None:
    print("\n--- Schedule detail (sorted by start time, then finish time) ---")
    sorted_ops = sorted(
        time_schedule.items(),
        key=lambda item: (item[1]["start_time"], item[1]["finished_time"]),
    )
    for (job_id, op_id), info in sorted_ops:
        print(
            f"Job {job_id}, Op {op_id}: "
            f"start={info['start_time']:.2f}, "
            f"duration={info['duration']:.2f}, "
            f"finish={info['finished_time']:.2f}"
        )


def print_machine_solution(solution: dict) -> None:
    print("\n--- Machine assignment (operation order per machine) ---")
    for machine_id in sorted(solution.keys()):
        ops = solution[machine_id]
        print(f"Machine {machine_id}: {ops}")


def main() -> None:
    if not JOBS_PATH.is_file():
        raise FileNotFoundError(f"Job dataset not found: {JOBS_PATH}")

    jobs_data = load_jobs_data(JOBS_PATH)
    print(f"Loaded jobs data from {JOBS_PATH.name}")
    print(f"Jobs: {len(jobs_data)}, operations per job: {[len(job) for job in jobs_data]}")

    print("\nRunning ARGA...")
    solution_arga = ARGA(
        jobs_data,
        FSG=generate_feasible_solution2,
        population_size1=POPULATION_SIZE,
        num_iterations1=NUM_ITERATIONS,
        mutation_threshold1=MUTATION_THRESHOLD,
        elit_percentage1=ELIT_PERCENTAGE,
        get_schedule_time=get_schedule_time_AR_resch,
        reschedule=RESCHEUDLE,
        obj_type=OBJ_TYPE,
        show_progress=SHOW_PROGRESS,
    )
    machine_solution = solution_arga[0]
    arga_best_cmax_objective = solution_arga[1]

    time_schedule = get_schedule_time_AR_resch(
        jobs_data,
        machine_solution,
        reschedule=False,
    )

    cmax = calculate_time_objective(time_schedule, obj_type="cmax")

    lb_cmax = calculate_cmax_lower_bound(jobs_data, machine_solution)

    print(f"\nARGA best objective (optimized as cmax): {arga_best_cmax_objective:.2f}")
    print(f"Cmax = {cmax:.2f}")
    print(f"Lower Bound of Cmax = {lb_cmax:.2f}")

    print_machine_solution(machine_solution)
    print_schedule_detail(time_schedule)

    create_gantt_chart_final(
        time_schedule,
        machine_solution,
        title=f"ARGA Phase 1 — {INSTANCE}",
    )


if __name__ == "__main__":
    main()
