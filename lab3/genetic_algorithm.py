import random

import numpy as np
import matplotlib.pyplot as plt

from functions import styblinski_tang_2d


def set_seed(seed: int) -> None:
    # Set fixed random seed to make the results reproducible
    random.seed(seed)
    np.random.seed(seed)


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        mutation_strength: float,
        crossover_rate: float,
        num_generations: int,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations

    def initialize_population(self) -> np.ndarray:
        # Initialize the population and return the result
        return np.random.uniform(-5, 5, (self.population_size, 2))

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        # Evaluate the fitness of the population and return the values for each individual in the population
        x, y = population[:, 0], population[:, 1]
        fitness_values = styblinski_tang_2d(x, y)
        return fitness_values

    def selection(
        self, population: np.ndarray, fitness_values: np.ndarray
    ) -> np.ndarray:
        # Implement selection mechanism and return the selected individuals

        # Rank the individuals in the current population based on their fitness.
        # The best individual gets the highest rank (earlier in the array), and the worst gets the lowest rank (later in the array).
        # In our case, lower is better, so we sort population by fitness score in ascending order
        sorted_indices = np.argsort(fitness_values)
        sorted_population = population[sorted_indices]

        # Assign selection probabilities to each individual based on their rank.
        # Higher-ranked individuals get higher probabilities.
        probabilities = np.linspace(1, 0, self.population_size)  # linear ranking
        #                                   how many options    , how many we pick    , put back    , probability of each array index
        selected_indices = np.random.choice(
            self.population_size,
            self.population_size,
            replace=True,
            p=probabilities / probabilities.sum(),
        )
        selected = sorted_population[selected_indices]

        return selected

    def crossover(self, parents: np.ndarray) -> np.ndarray:
        # Implement the crossover mechanism over the parents and return the offspring
        shuffled_parents = np.random.permutation(parents)
        offspring = np.empty_like(parents)
        
        for i in range(0, shuffled_parents.shape[0], 2):
            if i + 1 < parents.shape[0]:
                x = np.random.rand()
                if x >= self.crossover_rate:
                    offspring[i] = shuffled_parents[i]
                    offspring[i + 1] = shuffled_parents[i + 1]
                else:
                    alpha = np.random.rand()
                    offspring[i] = (
                        alpha * shuffled_parents[i]
                        + (1 - alpha) * shuffled_parents[i + 1]
                    )
                    offspring[i + 1] = (
                        alpha * shuffled_parents[i + 1]
                        + (1 - alpha) * shuffled_parents[i]
                    )
            else:
                offspring[i] = shuffled_parents[i]

        return offspring

    def mutate(self, individuals: np.ndarray) -> np.ndarray:
        # Implement mutation mechanism over the given individuals and return the results
        for individual in individuals:
            if random.random() < self.mutation_rate:
                individual += np.random.normal(
                    0, self.mutation_strength, individual.shape
                )
        return individuals

    def evolve(self, seed: int) -> tuple[list[list[int]], list[float], list[float]]:
        # Run the genetic algorithm and return the lists that contain the best solution for each generation,
        #   the best fitness for each generation and average fitness for each generation
        set_seed(seed)

        population = self.initialize_population()

        best_solutions = []
        best_fitness_values = []
        average_fitness_values = []

        for generation in range(self.num_generations):
            fitness_values = self.evaluate_population(population)
            parents_for_reproduction = self.selection(population, fitness_values)
            offspring = self.crossover(parents_for_reproduction)
            population = self.mutate(offspring)

            # compute fitness of the new generation and save the best solution, best fitness and average fitness
            best_index = np.argmin(fitness_values)
            best_solutions.append(population[best_index])

            best_fitness_values.append(fitness_values[best_index])

            average_fitness_values.append(np.mean(fitness_values))

        return best_solutions, best_fitness_values, average_fitness_values

if __name__ == "__main__":
    ga = GeneticAlgorithm(
        population_size=10,
        mutation_rate=0.1,
        mutation_strength=0.1,
        crossover_rate=0.1,
        num_generations=10,
    )
    best_solutions, best_fitness_values, average_fitness_values = ga.evolve(seed=42)
    
    best_fitness_values_scaled = []
    for value in best_fitness_values:
        best_fitness_values_scaled.append(value + 80)
    average_fitness_values
    
    print(best_solutions)
    print(" --- ")
    print(best_fitness_values)
    print(" --- ")
    print(average_fitness_values)
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.semilogy(best_fitness_values, label='Best Fitness')
    plt.semilogy(average_fitness_values, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value (log scale)')
    plt.title('Evolution of Fitness Values')
    plt.legend()
    plt.show()