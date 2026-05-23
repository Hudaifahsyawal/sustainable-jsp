Quick Start
===========

Installation
------------

.. code-block:: bash

   git clone <repo>
   cd sustainableJSP5
   pip install -e .

For building the documentation:

.. code-block:: bash

   pip install -r docs/requirements-docs.txt
   cd docs && make html


Initial Scheduling
------------------

Run the full three-phase sustainable pipeline on a benchmark instance:

.. code-block:: python

   import json
   from sustainable_jsp import sustainableJSP_resch, create_gantt_chart_final
   from sustainable_jsp.algorithms.workload import calculate_EErate

   # Load data
   with open("examples/Dataset/Dataset_Jobs/J14D30M5.json") as f:
       jobs_data = [[tuple(op) for op in job] for job in json.load(f)]
   with open("examples/Dataset/Dataset_Carbon/Carbon_data_J14D30M5.json") as f:
       carbon = {int(k): v for k, v in json.load(f).items()}
   with open("examples/Dataset/Dataset_Operator/operators_data_J14D30M5.json") as f:
       EErate = calculate_EErate({int(k): v for k, v in json.load(f).items()})

   # Run scheduling
   result = sustainableJSP_resch(
       jobs_data,
       carbon,
       EErate,
       num_iterations1=500,
       num_iterations2=500,
       show_progress=True,
   )

   print(result["matrix_performance"])

   create_gantt_chart_final(
       result["time_schedule"],
       result["solution"],
       operator_assignment=result["fair_workload"],
       speedlevel_list=result["speedlevel_optimum"],
       title="Initial schedule — J14D30M5",
   )


Saving and Loading a Schedule
------------------------------

.. code-block:: python

   import pickle

   # Save
   with open("initial_schedule_J14D30M5.pkl", "wb") as f:
       pickle.dump(result, f)

   # Load
   with open("initial_schedule_J14D30M5.pkl", "rb") as f:
       initial = pickle.load(f)


Rescheduling: New Job Arrival
------------------------------

.. code-block:: python

   from sustainable_jsp import reschedule_new_arrival_job

   # New job: [(machine_id, duration), ...]
   new_job = [[(1, 38), (2, 18), (4, 36), (3, 43)]]

   cr_output = reschedule_new_arrival_job(
       new_job,
       disruption_time=100,
       case1=initial["case_reschedule"],
       case1_solution_optimum=initial["solution"],
       Time_schedule_P2=initial["time_schedule"],
       fair_operator_assignment=initial["fair_workload"],
       workload_list=initial["workload_list"],
       event_times=initial["event_times"],
       EErate=EErate,
       carbon_emission_data=carbon,
       num_iterations1=500,
       num_iterations2=500,
       show_progress=True,
   )
   print(cr_output["matrix_performance"])


Rescheduling: Job Cancellation
--------------------------------

.. code-block:: python

   from sustainable_jsp import reschedule_cancelled_job

   cr_output = reschedule_cancelled_job(
       cancel_job_id=[7, 9],
       disruption_time=100,
       case1=initial["case_reschedule"],
       case1_solution_optimum=initial["solution"],
       Time_schedule_P2=initial["time_schedule"],
       fair_operator_assignment=initial["fair_workload"],
       workload_list=initial["workload_list"],
       event_times=initial["event_times"],
       EErate=EErate,
       carbon_emission_data=carbon,
       num_iterations1=500,
       num_iterations2=500,
       show_progress=True,
   )
   print(cr_output["matrix_performance"])


Dataset
-------

Pre-built instances are under ``examples/Dataset/``:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Folder
     - Content
   * - ``Dataset_Jobs/``
     - Job-shop instances — ``{INSTANCE}.json``
   * - ``Dataset_Carbon/``
     - Carbon emission rates — ``Carbon_data_{INSTANCE}.json``
   * - ``Dataset_Operator/``
     - Operator physiological data — ``operators_data_{INSTANCE}.json``
   * - ``Dataset_new_job_arrival/``
     - New-job disruption scenarios — ``{INSTANCE}_NEW_T{time}.json``
   * - ``Dataset_job_cancellation/``
     - Job-cancellation scenarios — ``{INSTANCE}_CANCEL_T{time}.json``

Available instances: ``J14D30M5``, ``J14D30M7``, ``J14D30M9``,
``J21D20M5``, ``J21D20M7``, ``J21D20M9``,
``J42D10M5``, ``J42D10M7``, ``J42D10M9``.
