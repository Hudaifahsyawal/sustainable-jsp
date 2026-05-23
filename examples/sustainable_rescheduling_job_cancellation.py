"""
Complete reschedule for job cancellation (sustainable three-phase pipeline).

Prerequisites:
  1. Run examples/sustainable_scheduling_pipeline.py with SAVE_RESULTS=True for the
     same INSTANCE, producing Dataset/initial_schedule/initial_schedule_{INSTANCE}.pkl
     (or Dataset/Dataset_initial_schedule/initial_schedule_{INSTANCE}.pkl).
  2. Dataset JSON files under examples/Dataset/ for carbon and operators.
  3. Job cancellation scenario JSON under Dataset/Dataset_job_cancellation/
     (default: {INSTANCE}_CANCEL_T{DISRUPTION_TIME}.json — list of job IDs to cancel).
"""

from __future__ import annotations

import json
import pickle
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
from sustainable_jsp.rescheduling.complete_reschedule import reschedule_cancelled_job

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------
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

CANCELLATION_DATASET_DIR = DATASET_DIR / "Dataset_job_cancellation"

# Disruption scenario — DISRUPTION_TIME selects the JSON file by name
DISRUPTION_TIME = 10

# If None, uses Dataset_job_cancellation/{INSTANCE}_CANCEL_T{DISRUPTION_TIME}.json
CANCELLATION_JSON_PATH: Path | None = None

# reschedule_cancelled_job / sustainableJSP_resch parameters
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


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_carbon_emission_data(path: Path) -> dict[int, list[float]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_operator_data(path: Path) -> dict[int, dict]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def job_cancellation_json_path(instance: str, disruption_time: float | int) -> Path:
    t = int(disruption_time) if float(disruption_time) == int(disruption_time) else disruption_time
    return CANCELLATION_DATASET_DIR / f"{instance}_CANCEL_T{t}.json"


def load_job_cancellation_json(path: Path) -> list[int]:
    """Load cancelled job IDs from JSON (list of ints, or {disruption_time, cancel_job_ids})."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return [int(j) for j in raw]
    if isinstance(raw, dict) and "cancel_job_ids" in raw:
        return [int(j) for j in raw["cancel_job_ids"]]
    raise ValueError(
        f"Unexpected format in {path}. Expected a JSON list of job IDs or "
        f'{{"disruption_time": ..., "cancel_job_ids": [...]}}.'
    )


def resolve_job_cancellation_path() -> Path:
    if CANCELLATION_JSON_PATH is not None:
        path = Path(CANCELLATION_JSON_PATH)
        if not path.is_file():
            raise FileNotFoundError(f"Job cancellation JSON not found: {path}")
        return path
    path = job_cancellation_json_path(INSTANCE, DISRUPTION_TIME)
    if not path.is_file():
        raise FileNotFoundError(
            f"Job cancellation JSON not found: {path}\n"
            f"Place file under {CANCELLATION_DATASET_DIR} or set CANCELLATION_JSON_PATH."
        )
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    scenario_path = resolve_job_cancellation_path()
    cancel_ids = load_job_cancellation_json(scenario_path)

    # Validate job IDs
    n_jobs = len(case)
    invalid = [j for j in cancel_ids if not (1 <= j <= n_jobs)]
    if invalid:
        raise ValueError(
            f"CANCEL_JOB_IDS contains invalid job IDs {invalid}. "
            f"Valid range for {INSTANCE}: 1..{n_jobs}."
        )

    # Warn if a job is already fully done before disruption time
    for j in cancel_ids:
        ops = [(j, o + 1) for o in range(len(case[j - 1])) if case[j - 1][o] is not None]
        all_done = all(
            time_schedule.get((jj, oo), {}).get("finished_time", float("inf")) <= DISRUPTION_TIME
            for jj, oo in ops
        )
        if all_done:
            print(
                f"Warning: job {j} has all operations finished before "
                f"disruption_time={DISRUPTION_TIME}. Cancellation will have no effect."
            )

    carbon_emission_data = load_carbon_emission_data(CARBON_PATH)
    operator_data = load_operator_data(OPERATORS_PATH)
    eerate = calculate_EErate(operator_data)

    print(f"\nJob cancellation scenario (disruption_time={DISRUPTION_TIME}):")
    print(f"  Cancelled job IDs: {cancel_ids}")
    print(f"  Loaded from {scenario_path.resolve()}")

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

    print("\n--- Complete reschedule: job cancellation ---")
    cr_output = reschedule_cancelled_job(
        cancel_ids,
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
        title=f"Complete reschedule job cancellation — {INSTANCE}",
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
            pkl_path = out_dir / f"complete_cancel_{INSTANCE}.pkl"
        else:
            pkl_path = Path(SAVE_DIR)
            pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with pkl_path.open("wb") as f:
            pickle.dump(cr_output, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nSaved reschedule result to {pkl_path.resolve()}")


if __name__ == "__main__":
    main()
