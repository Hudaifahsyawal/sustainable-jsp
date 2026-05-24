# sustainable-jsp

A Python package for Sustainable Job Shop Scheduling with multi-objective optimization and dynamic rescheduling.

## Features

- **Core**: Problem generation, schedule decoding, performance metrics, Gantt visualization
- **Algorithms**: ARGA (genetic), GANBI (carbon/time multi-objective), AWBO (operator workload balancing)
- **Scheduling**: Three-phase sustainable pipeline (`sustainableJSP_resch`) and dual-resource variant (`dual_resource_JSP`)
- **Rescheduling**: Complete reschedule for new job arrival, rework, and job cancellation
- **Similarity**: Needleman-Wunsch sequence alignment for schedule comparison
- **Physiology**: ANFIS model for operator energy expenditure estimation

## Installation

For development:

```bash
git clone <repo>
cd sustainableJSP5
pip install -e .
```

## Quick Start

```python
import json
from sustainable_jsp import sustainableJSP_resch, generate_problem2
from sustainable_jsp import create_gantt_chart_final
from sustainable_jsp.algorithms.workload import calculate_EErate

# Load data
with open("examples/Dataset/Dataset_Jobs/J14D30M5.json") as f:
    jobs_data = [[tuple(op) for op in job] for job in json.load(f)]
with open("examples/Dataset/Dataset_Carbon/Carbon_data_J14D30M5.json") as f:
    carbon_emission_data = {int(k): v for k, v in json.load(f).items()}
with open("examples/Dataset/Dataset_Operator/operators_data_J14D30M5.json") as f:
    operator_data = {int(k): v for k, v in json.load(f).items()}

EErate = calculate_EErate(operator_data)

result = sustainableJSP_resch(
    jobs_data,
    carbon_emission_data,
    EErate,
    num_iterations1=500,
    num_iterations2=500,
    show_progress=True,
)

create_gantt_chart_final(
    result["time_schedule"],
    result["solution"],
    operator_assignment=result["fair_workload"],
    speedlevel_list=result["speedlevel_optimum"],
)
```

## Package Structure

```
sustainable_jsp/
├── __init__.py             - Public API exports
├── core/
│   ├── problem.py          - Problem instance generators (generate_problem, generate_problem2, ...)
│   ├── schedule.py         - Schedule decoding, feasible solution generators
│   ├── performance_matrix.py - Objective value calculation (Cmax, flowtime, TCE, workload)
│   └── visualization.py    - Gantt chart (create_gantt_chart_final)
├── algorithms/
│   ├── genetic.py          - ARGA: genetic algorithm for machine-operation sequencing (Phase 1)
│   ├── carbon.py           - GANBI: carbon-aware speed level optimization (Phase 2)
│   └── workload.py         - AWBO: ant colony workload balancing (Phase 3)
├── scheduling/
│   ├── sustainable.py      - sustainableJSP_resch: 3-phase sustainable pipeline
│   └── dual_resource.py    - dual_resource_JSP: 2-phase (no carbon) variant
├── rescheduling/
│   ├── complete_reschedule.py  - reschedule_new_arrival_job, reschedule_rework, reschedule_cancelled_job
│   ├── partial_reschedule.py   - reschedule_left_shift, reschedule_right_shift, reschedule_greedy_insertion
│   ├── helper.py               - initialize_reschedule and internal helpers
│   └── system_status.py        - get_system_status
├── similarity/
│   └── sequence.py         - Needleman-Wunsch schedule alignment
└── physiology/
    └── anfis.py             - ANFIS operator energy estimation
```

## Public API

```python
from sustainable_jsp import (
    # Scheduling
    sustainableJSP_resch,       # 3-phase: ARGA → GANBI → AWBO
    dual_resource_JSP,          # 2-phase: ARGA → AWBO (no carbon)

    # Rescheduling (complete reschedule)
    reschedule_new_arrival_job,
    reschedule_rework,
    reschedule_cancelled_job,

    # Problem generation
    generate_problem,           # basic random instance
    generate_problem2,          # variable operations, custom duration range
    generate_problem3,          # machine-specific duration ranges

    # Visualization
    create_gantt_chart_final,
)
```

## Example Scripts

Ready-to-run scripts under `examples/`:

| Script | Description |
|--------|-------------|
| `sustainable_scheduling_pipeline.py` | Full 3-phase scheduling (ARGA → GANBI → AWBO), saves result to pickle |
| `sustainable_rescheduling_new_job_arrival.py` | Loads initial schedule pickle + new job JSON, runs complete reschedule |
| `sustainable_rescheduling_job_cancellation.py` | Loads initial schedule pickle + cancellation scenario JSON, runs complete reschedule |

### Running an example

1. Run the scheduling pipeline first to generate the initial schedule pickle:
   ```bash
   cd sustainableJSP5
   python examples/sustainable_scheduling_pipeline.py
   ```
   Set `SAVE_RESULTS = True` and choose `INSTANCE` (e.g. `"J14D30M5"`).

2. Run a rescheduling example using the saved pickle and a scenario JSON:
   ```bash
   python examples/sustainable_rescheduling_new_job_arrival.py
   python examples/sustainable_rescheduling_job_cancellation.py
   ```

## Dataset

Pre-built instances under `examples/Dataset/` (9 instances: J14D30, J21D20, J42D10 × M5/M7/M9):

| Folder | Content |
|--------|---------|
| `Dataset_Jobs/` | Job-shop instances (`{INSTANCE}.json`) |
| `Dataset_Carbon/` | Machine carbon intensity data (`Carbon_data_{INSTANCE}.json`) |
| `Dataset_Operator/` | Operator physiological data (`operators_data_{INSTANCE}.json`) |
| `Dataset_new_job_arrival/` | New-job disruption scenarios (`{INSTANCE}_NEW_T{time}.json`) |
| `Dataset_job_cancellation/` | Job-cancellation scenarios (`{INSTANCE}_CANCEL_T{time}.json`) |

## Dependencies

- `numpy`, `pandas`, `matplotlib`, `scipy`, `tqdm`
- Python >= 3.9
