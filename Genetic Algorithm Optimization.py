import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# # Implimenting Random Seed  
random.seed(43)

# Load the dataset
file_path = "GSE36961.xlsx"  # Replace with your actual file path
# file_path = "GSE36961_sample.xlsx"  # Replace with your actual file path
df = pd.read_excel(file_path)

# Preprocess the dataset
df.dropna(subset=['ID_REF'], inplace=True)  # Remove rows where 'ID_REF' is missing
df.drop_duplicates(subset='ID_REF', keep='first', inplace=True)  # Remove duplicate 'ID_REF'
df.set_index('ID_REF', inplace=True)  # Set 'ID_REF' as the index

# Normalize the dataset
scaler = MinMaxScaler()  # Create a MinMaxScaler object
normalized_data = scaler.fit_transform(df)  # Fit and transform the data
normalized_df = pd.DataFrame(normalized_data, columns=df.columns, index=df.index)  # Create a DataFrame from normalized data

# Transpose the normalized DataFrame if needed (samples as columns)
df = normalized_df.T

# Split into disease and healthy samples
Disease_matrix = df[df.index.str.contains("HCM")]  # Keeps rows with disease sample
Healthy_matrix = df[df.index.str.contains("control")]  # Keeps rows with healthy sample

# Calculate mean expression values
mean_disease = Disease_matrix.mean(axis=0)  # Mean for disease samples (genes as columns)
mean_healthy = Healthy_matrix.mean(axis=0)  # Mean for healthy samples (genes as columns)

# Calculate absolute mean differences
absolute_mean_difference_df = np.abs(mean_disease - mean_healthy)

# Parameters
population_size = 15    # Number of individuals in the population
num_genes = 10          # Number of genes in each individual
num_generations = 50   # Number of generations to evolve
mutation_rate = 0.1     # Probability of mutation
elitism_count = 1       # Number of top individuals to preserve in each generation

# Initialize random population of individuals (each individual is a unique subset of genes)
def initialize_population(absolute_mean_difference_df, population_size, num_genes):
    random.seed(43)
    population = []
    gene_indices = list(absolute_mean_difference_df.index)  # Get gene indices
    for _ in range(population_size):
        individual = random.sample(gene_indices, num_genes)  # Randomly sample unique genes
        population.append(individual)
    return population

# Fitness function: Sum of absolute mean differences for the selected genes
def calculate_fitness(individual, absolute_mean_difference_df):
    return sum(absolute_mean_difference_df[gene] for gene in individual)

# Roulette Wheel Selection
def roulette_wheel_selection(population, fitness_scores):
    random.seed(43)
    max_fitness = sum(fitness_scores)
    selection_probs = [fitness / max_fitness for fitness in fitness_scores]
    selected_index = np.random.choice(len(population), p=selection_probs)
    return population[selected_index]

# # Crossover (single-point crossover)
# def crossover(parent1, parent2):
#     crossover_point = random.randint(1, num_genes - 1)
#     child1 = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
#     child2 = parent2[:crossover_point] + [gene for gene in parent1 if gene not in parent2[:crossover_point]]
#     return child1, child2
# import random

# Two-point crossover function
def crossover(parent1, parent2):
    random.seed(43)
    # Ensure crossover points are different and within the range
    point1 = random.randint(1, num_genes - 2)
    point2 = random.randint(point1 + 1, num_genes - 1)

    # Create children by slicing and swapping the gene segments between the two parents
    child1 = (parent1[:point1] + 
              parent2[point1:point2] + 
              parent1[point2:])
    
    child2 = (parent2[:point1] + 
              parent1[point1:point2] + 
              parent2[point2:])
    
    return child1, child2

# Mutation: Replace a gene with a new one from the dataset, ensuring no repetition
def mutate(individual, mutation_rate, absolute_mean_difference_df):
    random.seed(43)
    if random.random() < mutation_rate:
        gene_pool = set(absolute_mean_difference_df.index)  # All possible genes from dataset
        current_genes = set(individual)  # Genes already in the individual
        available_genes = list(gene_pool - current_genes)  # Genes not currently in the individual
        
        if available_genes:  # Ensure there are genes available to replace
            new_gene = random.choice(available_genes)  # Select a new gene from the available genes
            idx = random.randint(0, len(individual) - 1)  # Select a random position in the individual
            individual[idx] = new_gene  # Replace the gene at the selected position with the new gene
    return individual

# Genetic Algorithm function
def genetic_algorithm(absolute_mean_difference_df, population_size, num_genes, num_generations, mutation_rate, elitism_count):
    # Initialize population
    population = initialize_population(absolute_mean_difference_df, population_size, num_genes)

    # Track the best fitness score over generations
    best_fitness_per_generation = []

    # Evolve over generations
    for generation in range(num_generations):
        # Calculate fitness for the population
        fitness_scores = [calculate_fitness(individual, absolute_mean_difference_df) for individual in population]
        
        # Sort population by fitness (highest fitness first)
        population_fitness = list(zip(population, fitness_scores))
        population_fitness.sort(key=lambda x: x[1], reverse=True)
        population = [ind for ind, fit in population_fitness]
        fitness_scores = [fit for ind, fit in population_fitness]
        
        # Preserve elite individuals (elitism)
        new_population = population[:elitism_count]
        
        # Generate new offspring
        while len(new_population) < population_size:
            # Selection
            parent1 = roulette_wheel_selection(population, fitness_scores)
            parent2 = roulette_wheel_selection(population, fitness_scores)
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            child1 = mutate(child1, mutation_rate, absolute_mean_difference_df)
            child2 = mutate(child2, mutation_rate, absolute_mean_difference_df)
            
            # Ensure no duplicates in the individual
            if len(set(child1)) == num_genes and len(set(child2)) == num_genes:
                new_population.extend([child1, child2])
        
        # Replace old population with the new one
        population = new_population[:population_size]

        # Track the best fitness score for this generation
        best_fitness = max(fitness_scores)
        best_fitness_per_generation.append(best_fitness)
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

    # Return the best individual from the last generation and fitness tracking
    best_individual = population[0]
    return best_individual, best_fitness_per_generation

# Run the GA and get the best individual and fitness tracking
best_individual, fitness_tracking = genetic_algorithm(
    absolute_mean_difference_df=absolute_mean_difference_df,
    population_size=population_size,
    num_genes=num_genes,
    num_generations=num_generations,
    mutation_rate=mutation_rate,
    elitism_count=elitism_count
)

# Get the gene names of the best individual
print("Best individual (genes):", best_individual)
print(len(best_individual))

# Visualization of Fitness Progression over Generations
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_generations + 1), fitness_tracking)
# plt.title('Genetic Algorithm: Fitness Progression Over Generations')
plt.xlabel('Generation', size = 18)
plt.ylabel('Best Fitness Score', size = 18)
plt.tick_params(axis='both', which='major', labelsize=14)  # Set the font size for the tick labels on both axes
plt.grid(True)

# # Save the figure as a PNG file
# plt.savefig('WGCNA Genetic Algorithm Fitness Score.png', dpi=500)  # Adjust dpi for higher resolution if needed

plt.show()
