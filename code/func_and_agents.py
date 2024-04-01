"""
file: func_and_agents.py
desc: contains objective functions and agents

"""
import pandas as pd
import numpy as np

ta = pd.read_csv('tas.csv')
sections = pd.read_csv('sections.csv')
test1 = pd.read_csv('test1.csv', header=None)
test2 = pd.read_csv('test2.csv', header=None)
test3 = pd.read_csv('test3.csv', header=None)

""" Functions """

# Objective 1: Minimize overallocation of TAs
def overallocation(solution):
    # pull column
    max_assigned = ta['max_assigned']

    # sum number of sections assigned for each TA
    assigned_sections_count = solution.sum(axis=1)

    # calculate how many sections exceed their max assigned value
    overallocation = assigned_sections_count - max_assigned

    # considers only positive overallocations
    overallocation_penalty = overallocation[overallocation > 0].sum()

    return overallocation_penalty


# Objective 2: Minimize time conflicts
def conflicts(solution):
    solution = np.array(solution)
    # pull column
    daytime_column = sections['daytime']

    # initialize set of unique section times/days
    conflicted_tas = set()

    for ta_assignment in solution:
        # initialize set to store day times for current TA
        assigned_daytimes = set()

        for section_idx, assigned in enumerate(ta_assignment):
            if assigned.any():
                # if section is assigned, get index
                section_daytime = daytime_column[section_idx]

                if section_daytime in assigned_daytimes:
                    # if daytime is already assigned for TA, add to conflicted TAs
                    conflicted_tas.add(tuple(ta_assignment))
                    # break loop to not count more than one conflict per TA/not count a conflict more than once
                    break
                else:
                    # if not already assigned, add to assigned times
                    assigned_daytimes.add(section_daytime)

    # sum number of TAs with conflicts
    num_conflicts = len(conflicted_tas)

    return num_conflicts


# Objective 3: Minimize under-support
def undersupport(solution):
    solution = np.array(solution)
    # pull column
    min_assigned = sections['min_ta']

    # iterate through sections and count the number of TAs assigned (1s)
    assigned_tas = map(lambda col: np.sum(solution[:, col]), range(solution.shape[1]))

    # for section in assigned tas column, compare the min_ta to the number assigned, calculate difference;
    # only take positive numbers
    ta_difference = map(lambda idx, assigned_tas: max(min_assigned[idx] - assigned_tas, 0),
                        range(len(min_assigned)), assigned_tas)

    # sum total differences across all sections
    total_penalty = sum(ta_difference)

    return total_penalty


# Objective 4: Minimize unwilling assignments
def unwilling(solution):
    # convert to numpy array with necessary columns
    ta_array = np.array((ta.loc[:, '0':'16']))
    solution = np.array(solution)

    # sum number of times unwilling (U) and assigned (1) occur in same location
    total_unwilling = np.sum((ta_array == 'U') & (solution == 1))

    return total_unwilling


# Objective 5: Minimize unpreferred assignments
def unpreferred(solution):
    # convert to numpy array with necessary columns
    ta_array = np.array((ta.loc[:, '0':'16']))
    solution = np.array(solution)

    # sum number of times W (available but not preferred) appears in same location as assigned (1)
    total_unpreferred = np.sum((ta_array == 'W') & (solution == 1))

    return total_unpreferred


# Mutation agent
def mutate(solution):
    mutated_solution = np.array(solution)  # convert to NumPy array
    ta_index = np.random.randint(len(mutated_solution))
    lab_index = np.random.randint(len(mutated_solution[0]))
    mutated_solution[ta_index, lab_index] = 1 - mutated_solution[ta_index, lab_index]  # toggle assignment
    return mutated_solution

# Crossover agent
def crossover(solution1, solution2):
    crossover_point = np.random.randint(len(solution1))
    new_solution = np.vstack((solution1[:crossover_point], solution2[crossover_point:]))
    return new_solution

def solutions_to_csv(env, filename):
    # identify non-dominated solutions
    nds = env.pop.items()

    # create a list to store the rows of the CSV
    csv_rows = []

    # iterate over non-dominated solutions
    for eval, _ in nds:
        # initialize a dictionary for the row
        row = {
            'groupname': 'group1',
            'overallocation': None,
            'conflicts': None,
            'undersupport': None,
            'unwilling': None,
            'unpreferred': None
        }

        # populate the row with objective scores if available
        for idx, (obj_name, _) in enumerate(env.fitness.items()):
            if idx < len(eval):
                row[obj_name] = eval[idx][1]

        # append the row to the list of rows
        csv_rows.append(row)

    # write to csv
    df = pd.DataFrame(csv_rows)
    df.to_csv(filename, index=False)