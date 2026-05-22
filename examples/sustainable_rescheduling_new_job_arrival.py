"""
Complete reschedule for new job arrival (sustainable three-phase pipeline).

Prerequisites:
  1. Run examples/sustainable_scheduling_pipeline.py with SAVE_RESULTS=True for the
     same INSTANCE, producing Dataset/initial_schedule/initial_schedule_{INSTANCE}.pkl
     (or Dataset/Dataset_initial_schedule/initial_schedule_{INSTANCE}.pkl).
  2. Dataset JSON files under examples/Dataset/ for carbon and operators.

Generates new job(s) with generate_problem2, saves them as JSON under
Dataset/Dataset_new_job_arrival/{INSTANCE}_NEW_T{DISRUPTION_TIME}.json
(same list-of-operations format as Dataset_Jobs/), then runs complete reschedule.
"""

from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    try:
        import sustainable_jsp  # noqa: F401
    except ModuleNotFoundError:
        sys.path.insert(0, str(_SRC))

from sustainable_jsp.algorithms.workload import calculate_EErate
from sustainable_jsp.core.problem import generate_problem2
from sustainable_jsp.core.visualization import create_gantt_chart_final
from sustainable_jsp.rescheduling.complete_reschedule import reschedule_new_arrival_job

INSTANCE = "J42D10M9"

EXAMPLES_DIR = Path(__file__).resolve().parent
DATASET_DIR = EXAMPLES_DIR / "Dataset"
CARBON_PATH = DATASET_DIR / "Dataset_Carbon" / f"Carbon_data_{INSTANCE}.json"
OPERATORS_PATH = DATASET_DIR / "Dataset_Operator" / f"operators_data_{INSTANCE}.json"

# Candidate paths for initial schedule pickle
INITIAL_SCHEDULE_CANDIDATES = [
    DATASET_DIR / "initial_schedule" / f"initial_schedule_{INSTANCE}.pkl",
    DATASET_DIR / "Dataset_initial_schedule" / f"initial_schedule_{INSTANCE}.pkl",
]

NEW_JOB_DATASET_DIR = DATASET_DIR / "Dataset_new_job_arrival"

# Disruption scenario
DISRUPTION_TIME = 160
NEW_JOB_SEED = 42

# New job: 1 job, 5 machines (J14D30M5), 4–6 operations, duration 15–45
NEW_JOB_NJOBS = 1
NEW_JOB_NMACHINES = 9 #5 #7 #9
NEW_JOB_NOPERATIONS = (4, 6)
NEW_JOB_DURATION_RANGE = (5, 15) #(15, 45) #(10, 30) #(5, 15)

# reschedule_new_arrival_job / sustainableJSP_resch (complete reschedule)
DUAL_RESOURCE = False
WEIGHT = 0.75
VISUALIZATION = False
OBJ_TYPE = "cmax"
WORKLOAD_OBJ_TYPE = "variance"
IR = 3
FLOWTIME_TYPE = "average"
NUM_ITERATIONS1 = 500
POPULATION_SIZE1 = 75
ELIT_PERCENTAGE1 = 0.6
NUM_ITERATIONS2 = 500
POPULATION_SIZE2 = 75
ELIT_PERCENTAGE2 = 0.6
COLONI_SIZE = 450
ALPHA = 10
BETA = 2
SHOW_PROGRESS = True

SAVE_RESULTS = False
SAVE_DIR: Path | None = None


def load_carbon_emission_data(path: Path) -> dict[int, list[float]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_operator_data(path: Path) -> dict[int, dict]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def jobs_to_dataset_json(jobs_data: list) -> list:
    """Convert internal job list (tuples) to JSON format used in Dataset_Jobs/."""
    return [[[int(m), int(d)] for m, d in job] for job in jobs_data]


def format_jobs_json_compact(jobs_data: list) -> str:
    """Serialize jobs like Dataset_Jobs/: one job per line, compact [machine,duration] pairs."""
    jobs = jobs_to_dataset_json(jobs_data)
    lines = ["["]
    for i, job in enumerate(jobs):
        job_line = json.dumps(job, separators=(",", ":"))
        suffix = "," if i < len(jobs) - 1 else ""
        lines.append(f" {job_line}{suffix}")
    lines.append("]")
    return "\n".join(lines) + "\n"


def new_job_arrival_json_path(instance: str, disruption_time: float) -> Path:
    return NEW_JOB_DATASET_DIR / f"{instance}_NEW_T{disruption_time}.json"


def save_new_job_arrival_json(jobs_data: list, instance: str, disruption_time: float) -> Path:
    NEW_JOB_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    path = new_job_arrival_json_path(instance, disruption_time)
    path.write_text(format_jobs_json_compact(jobs_data), encoding="utf-8")
    return path


def resolve_initial_schedule_path() -> Path:
    for path in INITIAL_SCHEDULE_CANDIDATES:
        if path.is_file():
            return path
    tried = "\n  ".join(str(p) for p in INITIAL_SCHEDULE_CANDIDATES)
    raise FileNotFoundError(
        f"No initial schedule pickle found for {INSTANCE}. Tried:\n  {tried}\n"
        "Run sustainable_scheduling_pipeline.py with SAVE_RESULTS=True first."
    )


def load_initial_schedule(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def print_matrix_performance(matrix_performance: dict) -> None:
    print("matrix_performance:")
    for key, value in matrix_performance.items():
        print(f"  '{key}' : {value}")


def main() -> None:
    if not CARBON_PATH.is_file():
        raise FileNotFoundError(f"Carbon dataset not found: {CARBON_PATH}")
    if not OPERATORS_PATH.is_file():
        raise FileNotFoundError(f"Operator dataset not found: {OPERATORS_PATH}")

    initial_path = resolve_initial_schedule_path()
    print(f"Loaded initial schedule from {initial_path}")

    initial = load_initial_schedule(initial_path)
    case = initial["case_reschedule"]
    solution = initial["solution"]
    time_schedule = initial["time_schedule"]
    fair_workload = initial["fair_workload"]
    workload_list = initial["workload_list"]
    event_times = initial["event_times"]
    speedlevel_optimum = initial.get("speedlevel_optimum")

    carbon_emission_data = load_carbon_emission_data(CARBON_PATH)
    operator_data = load_operator_data(OPERATORS_PATH)
    eerate = calculate_EErate(operator_data)

    random.seed(NEW_JOB_SEED)
    new_job = generate_problem2(
        NEW_JOB_NJOBS,
        NEW_JOB_NMACHINES,
        NEW_JOB_NOPERATIONS,
        NEW_JOB_DURATION_RANGE,
    )
    new_job_json_path = save_new_job_arrival_json(new_job, INSTANCE, DISRUPTION_TIME)
    print(f"\nNew job arrival (disruption_time={DISRUPTION_TIME}):")
    print(f"Saved to {new_job_json_path.resolve()}")
    for i, job in enumerate(new_job, start=len(case) + 1):
        print(f"  Job {i}: {job}")

    # print("\n--- Initial schedule (before disruption) ---")
    # create_gantt_chart_final(
    #     time_schedule,
    #     solution,
    #     operator_assignment=fair_workload,
    #     speedlevel_list=speedlevel_optimum,
    #     x_start=0,
    #     x_end=None,
    #     title=f"Initial schedule — {INSTANCE}",
    #     save=False,
    # )

    print("\n--- Complete reschedule: new job arrival ---")
    cr_output = reschedule_new_arrival_job(
        new_job,
        DISRUPTION_TIME,
        case,
        solution,
        time_schedule,
        fair_workload,
        workload_list,
        event_times,
        eerate,
        carbon_emission_data=carbon_emission_data,
        dual_resource=DUAL_RESOURCE,
        weight=WEIGHT,
        visualization=VISUALIZATION,
        x_start=0,
        x_end=None,
        title=f"Complete reschedule new job — {INSTANCE}",
        save=False,
        obj_type=OBJ_TYPE,
        workload_obj_type=WORKLOAD_OBJ_TYPE,
        IR=IR,
        flowtime_type=FLOWTIME_TYPE,
        num_iterations1=NUM_ITERATIONS1,
        population_size1=POPULATION_SIZE1,
        elit_percentage1=ELIT_PERCENTAGE1,
        num_iterations2=NUM_ITERATIONS2,
        population_size2=POPULATION_SIZE2,
        elit_percentage2=ELIT_PERCENTAGE2,
        coloni_size=COLONI_SIZE,
        alpha=ALPHA,
        beta=BETA,
        show_progress=SHOW_PROGRESS,
    )

    print("\n--------------- Reschedule result ---------------")
    print(f"mode: {cr_output.get('mode')}")
    print_matrix_performance(cr_output.get("matrix_performance", {}))

    if SAVE_RESULTS:
        if SAVE_DIR is None:
            out_dir = DATASET_DIR / "reschedule_results"
            out_dir.mkdir(parents=True, exist_ok=True)
            pkl_path = out_dir / f"complete_new_job_{INSTANCE}.pkl"
        else:
            pkl_path = Path(SAVE_DIR)
            pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with pkl_path.open("wb") as f:
            pickle.dump(cr_output, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nSaved reschedule result to {pkl_path.resolve()}")


if __name__ == "__main__":
    main()
