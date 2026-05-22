"""
Sustainable job shop scheduling: three-phase pipeline (ARGA → GANBI → AWBO).

Dataset: examples/Dataset/Dataset_Jobs, Dataset_Carbon, Dataset_Operator.

Optional: set SAVE_RESULTS = True to pickle the full return dict to
Dataset/Dataset_initial_schedule/initial_schedule_{INSTANCE}.pkl
(or set SAVE_DIR to a full .pkl path).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

# Repo root; allow running without `pip install -e .`
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    try:
        import sustainable_jsp  # noqa: F401
    except ModuleNotFoundError:
        sys.path.insert(0, str(_SRC))

from sustainable_jsp.algorithms.genetic import crossoverNR, mutate1N
from sustainable_jsp.algorithms.workload import calculate_EErate
from sustainable_jsp.core.schedule import generate_feasible_solution2_resch, get_schedule_time_AR_resch
from sustainable_jsp.scheduling.sustainable import sustainableJSP_resch

# Instance id must match files: J{jobs}_D{ops}_M{machines} pattern in Dataset_2draft naming
INSTANCE = "J42D10M9"

EXAMPLES_DIR = Path(__file__).resolve().parent
DATASET_DIR = EXAMPLES_DIR / "Dataset"
JOBS_PATH = DATASET_DIR / "Dataset_Jobs" / f"{INSTANCE}.json"
CARBON_PATH = DATASET_DIR / "Dataset_Carbon" / f"Carbon_data_{INSTANCE}.json"
OPERATORS_PATH = DATASET_DIR / "Dataset_Operator" / f"operators_data_{INSTANCE}.json"

# After run, write full `out` dict to a single .pkl (for reschedule / reload).
SAVE_RESULTS = True
# Full path to the .pkl file. If None, uses
#   Dataset/Dataset_initial_schedule/initial_schedule_{INSTANCE}.pkl
SAVE_DIR: Path | None = None


def load_jobs_data(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    return [[tuple(op) for op in job] for job in loaded]


def load_carbon_emission_data(path: Path) -> dict[int, list[float]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_operator_data(path: Path) -> dict[int, dict]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def main() -> None:
    for label, path in (
        ("jobs", JOBS_PATH),
        ("carbon", CARBON_PATH),
        ("operators", OPERATORS_PATH),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} dataset not found: {path}")

    case1 = load_jobs_data(JOBS_PATH)
    carbon_emission_data = load_carbon_emission_data(CARBON_PATH)
    operator_data = load_operator_data(OPERATORS_PATH)
    EErate = calculate_EErate(operator_data)

    #parameters for sustainableJSP
    SHOW_PROGRESS = True
    OBJ_TYPE = "cmax"
    IR = 3
    FLOWTIME_TYPE = "average"
    WORKLOAD_OBJ_TYPE = "variance"
    #parameters for phase 1
    POPULATION_SIZE1 = 75
    NUM_ITERATIONS1 = 500
    MUTATION_THRESHOLD1 = 0.5
    ELIT_PERCENTAGE1 = 0.6
    #parameters for phase 2
    POPULATION_SIZE2 = 75
    NUM_ITERATIONS2 = 500
    ELIT_PERCENTAGE2 = 0.6
    MUTATION_THRESHOLD2 = 0.5
    WEIGHT = 0.75
    VISUALIZATION = False
    RESCHEDULE = False
    MACHINE_START_TIME = None
    #parameters for phase 3
    PROB = 0.95
    COLONI_SIZE = 450
    ALPHA = 10
    BETA = 2

    out = sustainableJSP_resch(
        case1,
        carbon_emission_data,
        EErate,
        population_size1=POPULATION_SIZE1,
        num_iterations1=NUM_ITERATIONS1,
        mutation_threshold1=MUTATION_THRESHOLD1,
        elit_percentage1=ELIT_PERCENTAGE1,
        FSG=generate_feasible_solution2_resch,
        crossover=crossoverNR,
        mutate=mutate1N,
        get_schedule_time=get_schedule_time_AR_resch,
        num_iterations2=NUM_ITERATIONS2,
        population_size2=POPULATION_SIZE2,
        elit_percentage2=ELIT_PERCENTAGE2,
        mutation_threshold2=MUTATION_THRESHOLD2,
        weight=WEIGHT,
        visualization=VISUALIZATION,
        reschedule=RESCHEDULE,
        machine_start_time=MACHINE_START_TIME,
        prob=PROB,
        coloni_size=COLONI_SIZE,
        alpha=ALPHA,
        beta=BETA,
        initial_workload=None,
        initial_ready_time=None,
        x_start=None,
        x_end=None,
        a1=None,
        a2=None,
        a2_info=None,
        disruption_time=None,
        init_time_schedule=None,
        init_case1=None,
        init_case1_solution=None,
        init_fair_operator_assignment=None,
        init_workload_list=None,
        init_event_times=None,
        title=None,
        save=False,
        obj_type=OBJ_TYPE,
        workload_obj_type=WORKLOAD_OBJ_TYPE,
        IR=IR,
        flowtime_type=FLOWTIME_TYPE,
        show_progress=SHOW_PROGRESS,
    )

    if SAVE_RESULTS:
        if SAVE_DIR is None:
            _out_dir = DATASET_DIR / f"Dataset_initial_schedule"
            _out_dir.mkdir(parents=True, exist_ok=True)
            pkl_path = _out_dir / f"initial_schedule_{INSTANCE}.pkl"
        else:
            pkl_path = Path(SAVE_DIR)
            pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with pkl_path.open("wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nSaved pipeline result to {pkl_path.resolve()}")


if __name__ == "__main__":
    main()
