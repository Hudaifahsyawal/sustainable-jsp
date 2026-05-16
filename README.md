# sustainable-jsp

A Python package for Sustainable Job Shop Scheduling with multi-objective optimization and dynamic rescheduling.

## Features

- **Core**: Problem generation, schedule decoding, performance metrics, Gantt visualization
- **Algorithms**: ARGA (genetic), GANBI (carbon/time multi-objective), AWBO (operator workload)
- **Scheduling**: Three-phase sustainable pipeline (`sustainableJSP_resch`) and dual-resource variant
- **Rescheduling**: Complete reschedule (new job, rework, cancellation) and partial reschedule (left-shift, right-shift, greedy insertion)
- **Similarity**: Needleman-Wunsch sequence alignment for schedule comparison
- **Physiology**: ANFIS model for operator energy estimation

## Installation

```bash
pip install sustainable-jsp
```

For development:

```bash
git clone <repo>
cd sustainableJSP5
pip install -e .[dev]
```

## Quick Start

```python
from sustainable_jsp import sustainableJSP_resch, generate_problem

jobs_data = generate_problem(n_jobs=10, n_machines=5)
result = sustainableJSP_resch(jobs_data)
```

## Package Structure

```
sustainable_jsp/
├── core/           - Problem generation, scheduling, performance metrics, visualization
├── algorithms/     - ARGA, GANBI, AWBO optimization algorithms
├── scheduling/     - Main scheduling pipelines
├── rescheduling/   - Dynamic rescheduling strategies
├── similarity/     - Schedule sequence alignment
└── physiology/     - Operator physiological model (ANFIS)
```
