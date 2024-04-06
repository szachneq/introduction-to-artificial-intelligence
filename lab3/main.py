from genetic_algorithm import GeneticAlgorithm
from functions import global_minima
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ga = GeneticAlgorithm(
        population_size=10000,
        mutation_rate=0.08,
        mutation_strength=0.08,
        crossover_rate=0.08,
        num_generations=200,
    )
    best_solutions, best_fitness_values, average_fitness_values = ga.evolve(seed=1212309576)
    
    GLOBAL_MINIMA = global_minima()
    print(GLOBAL_MINIMA)
    best_fitness_values_scaled = []
    for value in best_fitness_values:
        best_fitness_values_scaled.append(abs(value - GLOBAL_MINIMA))
    
    average_fitness_values_scaled = []
    for value in average_fitness_values:
        average_fitness_values_scaled.append(abs(value - GLOBAL_MINIMA))
    
    print(best_solutions)
    print(" --- ")
    print(best_fitness_values_scaled)
    print(" --- ")
    print(average_fitness_values_scaled)
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.semilogy(best_fitness_values_scaled, label='Best Fitness')
    plt.semilogy(average_fitness_values_scaled, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value (log scale)')
    plt.title('Evolution of Fitness Values')
    plt.legend()
    plt.show()
