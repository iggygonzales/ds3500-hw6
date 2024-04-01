"""
file: main.py
git repo: https://github.com/iggygonzales/ds3500-hw6
"""

# Main function to run optimizer
import evo
import numpy as np
from func_and_agents import solutions_to_csv, mutate, crossover, overallocation, conflicts, undersupport, unpreferred, unwilling, sections, ta

def main():
    # create the environment
    E = evo.Environment()

    # register the fitness functions
    E.add_fitness_criteria("unwilling", unwilling)
    E.add_fitness_criteria("overallocation", overallocation)
    E.add_fitness_criteria("conflicts", conflicts)
    E.add_fitness_criteria("undersupport", undersupport)
    E.add_fitness_criteria("unpreffered", unpreferred)

    # register the agents
    E.add_agent("mutate", mutate, 1)
    E.add_agent("crossover", crossover, 2)

    # Adding 1 or more initial solution
    # L = [rnd.randrange(1, 99) for _ in range(30)]
    solution = np.zeros((len(ta), len(sections)))
    E.add_solution(solution)

    # Run the evolver
    E.evolve(1000, 100, 100)

    # Print the final result
    print(E)
    solutions_to_csv(E, 'pareto_optimal_solutions.csv')

if __name__ == "__main__":
    main()