"""
file: test.py
desc: tests the objective functions and agents
"""
import numpy as np
import pandas as pd
import pytest

ta = pd.read_csv('tas.csv')
sections = pd.read_csv('sections.csv')
test1 = pd.read_csv('test1.csv', header=None)
test2 = pd.read_csv('test2.csv', header=None)
test3 = pd.read_csv('test3.csv', header=None)

""" Tests """
tests = [test1, test2, test3]
exp_under = [1, 0, 7]
exp_unwilling = [53, 58, 43]
exp_unpreferred = [15, 19, 10]
exp_overallocation = [37, 41, 23]
exp_conflicts = [8, 5, 2]

# Import the overallocation_penalty function from the module
from func_and_agents import overallocation, conflicts, undersupport, unwilling, unpreferred, crossover, mutate


@pytest.mark.parametrize("input_data, expected", zip(tests, exp_overallocation))
def test_overallocation(input_data, expected):
    result = overallocation(input_data)
    assert result == expected


# this one doesn't work
@pytest.mark.parametrize("input_data, expected", zip(tests, exp_conflicts))
def test_conflicts(input_data, expected):
    result = conflicts(input_data)
    assert result == expected


@pytest.mark.parametrize("input_data, expected", zip(tests, exp_under))
def test_undersupport(input_data, expected):
    result = undersupport(input_data)
    assert result == expected


@pytest.mark.parametrize("input_data, expected", zip(tests, exp_unwilling))
def test_unwilling(input_data, expected):
    result = unwilling(input_data)
    assert result == expected


@pytest.mark.parametrize("input_data, expected", zip(tests, exp_unpreferred))
def test_unpreferred(input_data, expected):
    result = unpreferred(input_data)
    assert result == expected


# Test Mutation
def test_mutation():
    # Create a sample solution
    solution = np.zeros((5, 3))

    print("Original Solution:")
    print(solution)

    # Apply mutation
    mutated_solution = mutate(solution)

    print("\nMutated Solution:")
    print(mutated_solution)


# Test Crossover
def test_crossover():
    # Create sample solutions
    solution1 = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]])
    solution2 = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 1]])

    print("Solution 1:")
    print(solution1)
    print("\nSolution 2:")
    print(solution2)

    # Apply crossover
    crossed_solution = crossover(solution1, solution2)

    print("\nCrossed Solution:")
    print(crossed_solution)


if __name__ == "__main__":
    pytest.main()
