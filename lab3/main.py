from genetic_algorithm import GeneticAlgorithm

if __name__ == "__main__":
    # TODO Experiment 1...
    ga = GeneticAlgorithm(
        population_size=...,
        mutation_rate=...,
        mutation_strength=...,
        crossover_rate=...,
        num_generations=...,
    )
    best_solutions, best_fitness_values, average_fitness_values = ga.evolve(seed=...)
