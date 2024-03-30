"""
file: main.py

"""

# Main function to run optimizer
import evo
import numpy as np
from func_and_agents import solutions_to_csv, mutate, unwilling, sections, ta


def main():
    # Run optimizer with time limit
    # test_overallocation()
    # create the environment

    solution = np.zeros((len(ta), len(sections)))

    E = evo.Environment()

    # register the fitness functions
    E.add_fitness_criteria("unwilling", unwilling)

    # register the agents
    E.add_agent("mutate", mutate, 1)

    # Adding 1 or more initial solution
    # L = [rnd.randrange(1, 99) for _ in range(30)]
    E.add_solution(solution)

    # Run the evolver
    E.evolve(1000, 100, 100)

    # Print the final result
    print(E)

    solutions_to_csv(E, 'pareto_optimal_solutions.csv')


if __name__ == "__main__":
    main()