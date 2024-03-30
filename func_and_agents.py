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
    max_assigned = ta['max_assigned']
    assigned_sections_count = solution.sum(axis=1)
    overallocation = assigned_sections_count - max_assigned
    # considers only positive overallocations
    overallocation_penalty = overallocation[overallocation > 0].sum()
    return overallocation_penalty


# Objective 2: Minimize time conflicts (not working correctly)
def conflicts(solution):
    section_assignments = np.array(solution)
    num_tas = section_assignments.shape[0]
    num_sections = section_assignments.shape[1]
    tAs_with_conflicts = set()
    num_conflicts = 0

    for i in range(num_sections):  # Iterate over each section
        assigned_tas = np.where(section_assignments[:, i] == 1)[0]
        for j in range(i + 1, num_sections):  # Check conflicts with subsequent sections
            overlapping_tas = np.where(section_assignments[:, j] == 1)[0]
            if assigned_tas.size > 0 and overlapping_tas.size > 0 and len(set(assigned_tas) & set(overlapping_tas)) > 0:
                conflicted_tas = set(assigned_tas) | set(overlapping_tas)
                new_conflicted_tas = conflicted_tas - tAs_with_conflicts
                num_conflicts += len(new_conflicted_tas)
                tAs_with_conflicts.update(new_conflicted_tas)

    return num_tas - len(tAs_with_conflicts)


# Objective 3: Minimize under-support
def undersupport(solution):
    total_penalty = 0
    solution = np.array(solution)
    min_assigned = sections['min_ta']

    assigned_tas = map(lambda col: np.sum(solution[:, col]), range(solution.shape[1]))
    ta_difference = map(lambda idx, assigned_tas: max(min_assigned[idx] - assigned_tas, 0),
                        range(len(min_assigned)), assigned_tas)
    total_penalty = sum(ta_difference)

    return total_penalty


# Objective 4: Minimize unwilling assignments
def unwilling(solution):
    ta_array = np.array((ta.loc[:, '0':'16']))
    solution = np.array(solution)

    total_unwilling = np.sum((ta_array == 'U') & (solution == 1))

    return total_unwilling


# Objective 5: Minimize unpreferred assignments
def unpreferred(solution):
    ta_array = np.array((ta.loc[:, '0':'16']))
    solution = np.array(solution)

    total_unpreferred = np.sum((ta_array == 'W') & (solution == 1))

    return total_unpreferred


# Mutation agent
def mutate(solution):
    mutated_solution = np.array(solution)  # Convert to NumPy array
    ta_index = np.random.randint(len(mutated_solution))
    lab_index = np.random.randint(len(mutated_solution[0]))
    mutated_solution[ta_index, lab_index] = 1 - mutated_solution[ta_index, lab_index]  # Toggle assignment
    return mutated_solution


# Crossover agent
def crossover(solution1, solution2):
    crossover_point = np.random.randint(len(solution1))
    new_solution = np.vstack((solution1[:crossover_point], solution2[crossover_point:]))
    return new_solution


def solutions_to_csv(env, filename):
    # identify non-dominated solutions
    nds = list(env.pop.keys())

    # extract scores from solutions
    objective_scores = []
    for eval, sol in nds:
        groupname = "GroupName"
        overallocation_score = eval[0][1]
        conflicts_score = eval[1][1]
        undersupport_score = eval[2][1]
        unwilling_score = eval[3][1]
        unpreferred_score = eval[4][1]

        # append scores to list
        objective_scores.append({
            'groupname': groupname,
            'overallocation': overallocation_score,
            'conflicts': conflicts_score,
            'undersupport': undersupport_score,
            'unwilling': unwilling_score,
            'unpreferred': unpreferred_score
        })

    # write to csv
    df = pd.DataFrame(objective_scores)
    df.to_csv(filename, index=False)
