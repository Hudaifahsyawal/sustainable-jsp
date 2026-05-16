import random


def generate_problem(n_jobs=10, n_machines=5):
    """
    Generate a random Job Shop Scheduling Problem (JSP) instance.

    Creates a JSP problem where each job must be processed on all machines in a
    randomly shuffled order. Each operation has a randomly generated processing time.
    This generates a standard JSP problem where each job visits each machine exactly once,
    but the order of machine visits is randomized for each job.
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
    Generate a random Job Shop Scheduling Problem (JSP) instance with a fixed number
    of operations per job.
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
    Generate a random Job Shop Scheduling Problem (JSP) instance with variable
    number of operations per job and customizable duration range.
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
    Generate a random Job Shop Scheduling Problem (JSP) instance with variable
    number of operations per job and machine-specific duration ranges.

    Creates a JSP problem where each job has a random number of operations within
    a specified range. The distinguishing feature of this function is that each
    machine has its own duration range, allowing for realistic modeling of
    heterogeneous manufacturing environments where machines have varying capabilities,
    speeds, or processing characteristics.
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
