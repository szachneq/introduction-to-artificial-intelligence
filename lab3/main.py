from genetic_algorithm import GeneticAlgorithm
from functions import global_minima
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    ga = GeneticAlgorithm(
        population_size=1000,
        mutation_rate=0.25,
        mutation_strength=1,
        crossover_rate=0.1,
        num_generations=200,
    )
    best_solutions, best_fitness_values, average_fitness_values = ga.evolve(seed=42)
    
    GLOBAL_MINIMA = global_minima()
    best_fitness_values_scaled = np.array(best_fitness_values) - GLOBAL_MINIMA
    # best_fitness_values_scaled = []
    # for value in best_fitness_values:
    #     best_fitness_values_scaled.append(abs(value - GLOBAL_MINIMA))
    
    # average_fitness_values_scaled = average_fitness_values - GLOBAL_MINIMA
    average_fitness_values_scaled = []
    for value in average_fitness_values:
        average_fitness_values_scaled.append(abs(value - GLOBAL_MINIMA))
    
    # print(best_solutions)
    # print(" --- ")
    # print(best_fitness_values_scaled)
    # print(" --- ")
    # print(average_fitness_values_scaled)
    # print(" --- ")
    print("From last generation:")
    print(f"Best fitness value: {best_fitness_values_scaled[-1]:.2E} ({best_fitness_values[-1]:.5E}) for point {best_solutions[-1]}")
    print(f"Average fitness value: {average_fitness_values_scaled[-1]}")
    print("Globally:")
    print(f"Best fitness value: {min(best_fitness_values_scaled):.5E} ({min(best_fitness_values):.5E}) for point {best_solutions[-1]}")
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.semilogy(best_fitness_values_scaled, label='Best Fitness')
    plt.semilogy(average_fitness_values_scaled, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value (log scale)')
    plt.title('Evolution of Fitness Values')
    plt.legend()
    plt.show()
    
    # print(best_fitness_values_scaled)
