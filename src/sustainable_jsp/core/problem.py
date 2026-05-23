import random


def generate_problem(n_jobs=10, n_machines=5):
    """
    Generate a simple random Job Shop Scheduling Problem instance.

    Each job visits **all** machines exactly once in a random order.
    Processing times are drawn uniformly from ``[2, 7]``.

    Parameters
    ----------
    n_jobs : int, optional
        Number of jobs. Default 10.
    n_machines : int, optional
        Number of machines. Default 5.

    Returns
    -------
    list of list of tuple
        ``jobs_data[job_index]`` is a list of ``(machine_id, duration)`` tuples,
        where ``machine_id`` is 1-indexed.

    Examples
    --------
    >>> from sustainable_jsp import generate_problem
    >>> jobs = generate_problem(n_jobs=5, n_machines=3)
    >>> len(jobs)
    5
    >>> len(jobs[0])   # each job has exactly n_machines operations
    3
    """
    new_jobs_data = []
    for _ in range(n_jobs):
        new_machine_order = list(range(1, n_machines + 1))
        random.shuffle(new_machine_order)
        job = []
        for machine in new_machine_order:
            duration = random.randint(2, 7)
            job.append((machine, duration))
        new_jobs_data.append(job)

    return new_jobs_data


def generate_problem1(njobs, nmachines, noperations):
    """
    Generate a random JSP instance with a **fixed** number of operations per job.

    Each job has exactly ``noperations`` operations. If ``noperations <= nmachines``,
    machines are sampled without replacement; otherwise a no-consecutive-repeat
    selection is used. Processing times are drawn from ``[2, 10]``.

    Parameters
    ----------
    njobs : int
        Number of jobs.
    nmachines : int
        Number of available machines.
    noperations : int
        Fixed number of operations per job.

    Returns
    -------
    list of list of tuple
        ``jobs_data[job_index]`` is a list of ``(machine_id, duration)`` tuples.
    """
    new_jobs_data = []
    for _ in range(njobs):
        job = []
        if noperations <= nmachines:
            selected_machines = random.sample(range(1, nmachines + 1), noperations)
        else:
            selected_machines = []
            available_machines = list(range(1, nmachines + 1))
            selected_machines.append(random.choice(available_machines))
            for _ in range(noperations - 1):
                available_choices = available_machines.copy()
                available_choices.remove(selected_machines[-1])
                selected_machines.append(random.choice(available_choices))
        for machine in selected_machines:
            duration = random.randint(2, 10)
            job.append((machine, duration))
        new_jobs_data.append(job)
    return new_jobs_data


def generate_problem2(njobs, nmachines, noperations, range_duration=(2, 10)):
    """
    Generate a random JSP instance with a **variable** number of operations per job.

    The number of operations for each job is sampled uniformly from
    ``[noperations[0], noperations[1]]``. Processing times are drawn from
    ``range_duration``. This is the recommended generator for rescheduling
    experiments where new jobs arrive with unknown operation counts.

    Parameters
    ----------
    njobs : int
        Number of jobs to generate.
    nmachines : int
        Number of available machines. Must match the instance being rescheduled.
    noperations : tuple of int
        ``(min_operations, max_operations)`` — inclusive range for the random
        number of operations per job.
    range_duration : tuple of int, optional
        ``(min_duration, max_duration)`` for processing time sampling.
        Default ``(2, 10)``.

    Returns
    -------
    list of list of tuple
        ``jobs_data[job_index]`` is a list of ``(machine_id, duration)`` tuples.

    Examples
    --------
    >>> from sustainable_jsp import generate_problem2
    >>> import random; random.seed(42)
    >>> jobs = generate_problem2(1, 5, (4, 6), range_duration=(15, 45))
    >>> jobs  # 1 new job, 4-6 ops, machines 1-5, durations 15-45
    [[(3, 38), (1, 22), (5, 41), (4, 29), (2, 31)]]
    """
    min_operations, max_operations = noperations
    new_jobs_data = []
    for _ in range(njobs):
        job = []
        num_operations = random.randint(min_operations, max_operations)
        if num_operations <= nmachines:
            selected_machines = random.sample(range(1, nmachines + 1), num_operations)
        else:
            selected_machines = []
            available_machines = list(range(1, nmachines + 1))
            selected_machines.append(random.choice(available_machines))
            for _ in range(num_operations - 1):
                available_choices = available_machines.copy()
                available_choices.remove(selected_machines[-1])
                selected_machines.append(random.choice(available_choices))
        for machine in selected_machines:
            duration = random.randint(*range_duration)
            job.append((machine, duration))
        new_jobs_data.append(job)
    return new_jobs_data


def generate_problem3(njobs, nmachines, noperations, list_of_range_duration):
    """
    Generate a random JSP instance with **machine-specific** duration ranges.

    Like :func:`generate_problem2` but each machine ``m`` has its own processing
    time range ``list_of_range_duration[m-1]``, enabling realistic heterogeneous
    manufacturing environments.

    Parameters
    ----------
    njobs : int
        Number of jobs to generate.
    nmachines : int
        Number of available machines.
    noperations : tuple of int
        ``(min_operations, max_operations)`` — range for operation count per job.
    list_of_range_duration : list of tuple of int
        Length must equal ``nmachines``. Each entry is ``(min_dur, max_dur)`` for
        the corresponding machine.

    Returns
    -------
    list of list of tuple
        ``jobs_data[job_index]`` is a list of ``(machine_id, duration)`` tuples.

    Raises
    ------
    AssertionError
        If ``len(list_of_range_duration) != nmachines``.
    """
    assert len(list_of_range_duration) == nmachines, \
        "length of list_of_range_duration must equal with the number of machines"

    min_operations, max_operations = noperations
    new_jobs_data = []

    for _ in range(njobs):
        job = []
        num_operations = random.randint(min_operations, max_operations)

        if num_operations <= nmachines:
            selected_machines = random.sample(range(1, nmachines + 1), num_operations)
        else:
            selected_machines = []
            available_machines = list(range(1, nmachines + 1))
            selected_machines.append(random.choice(available_machines))
            for _ in range(num_operations - 1):
                available_choices = available_machines.copy()
                available_choices.remove(selected_machines[-1])
                selected_machines.append(random.choice(available_choices))

        for machine in selected_machines:
            min_dur, max_dur = list_of_range_duration[machine - 1]
            duration = random.randint(min_dur, max_dur)
            job.append((machine, duration))

        new_jobs_data.append(job)

    return new_jobs_data
