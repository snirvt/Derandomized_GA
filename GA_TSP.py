import numpy as np, random, operator
import pandas as pd
import matplotlib.pyplot as plt


# City class for the TSP problem #
class City:
    def __init__(self, index, x, y):
        self.x = x
        self.y = y
        self.index = index

    def distance(self, city):
        x_distance = abs(self.x - city.x)
        y_distance = abs(self.y - city.y)
        distance = np.sqrt((x_distance ** 2) + (y_distance ** 2))
        return distance
    
    def __repr__(self):
        return '(' + str(self.index) + ', (' + str(self.x) + ', ' + str(self.y) + '))'

# Fitness class for evaluating fitness given route #
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def route_distance(self):
        if self.distance == 0:
            path_distance = 0
            for i in range(len(self.route)):
                from_city = self.route[i]
                to_city = None
                if i + 1 < len(self.route):
                    to_city = self.route[i+1]
                else:
                    to_city = self.route[0]
                path_distance += from_city.distance(to_city)
            self.distance = path_distance
        return self.distance

    def route_fitness(self):
        if self.fitness == 0:
            self.fitness = float(self.route_distance())
        return self.fitness

# Route generator #
def create_route(cities_list):
    route = random.sample(cities_list, len(cities_list))
    return route

# Population initialization #
def initial_population(pop_size, cities_list):
    population = []

    for i in range(pop_size):
        population.append(create_route(cities_list))
    return population

# Returns a sorted dictionary of individuals by fitness param
def rank_individuals(population):
    fitness_results = {}
    for i in range(len(population)):
        fitness_results[i] = Fitness(population[i]).route_fitness()
    return sorted(fitness_results.items(), key = operator.itemgetter(1), reverse=False)

# Selection function - cum_sum is cumulative sum
def selection(ranked_pop, elitism_size, tournament_size):
    selection_results = []

    for i in range(elitism_size):
        selection_results.append(ranked_pop[i][0])

    for i in range(tournament_size):
        tournament = [random.choice(ranked_pop) for i in range(tournament_size)]
        best_ind = min(tournament, key=operator.itemgetter(1))
        selection_results.append(best_ind[0])

    return selection_results

# Creating the mating pool
def create_mating_pool(population, selection_results):
    mating_pool = []
    for i in range(len(selection_results)):
        mating_pool.append(population[selection_results[i]])

    return mating_pool

# Order 1 Crossover
def crossover(first_parent, second_parent):
    child = []
    child_fp = []
    child_sp = []

    gene_a = int(random.random() * len(first_parent))
    gene_b = int(random.random() * len(first_parent))

    start_gene = min(gene_a, gene_b)
    end_gene = max(gene_a, gene_b)

    for i in range(start_gene, end_gene):
        child_fp.append(first_parent[i])

    child_sp = [elem for elem in second_parent if elem not in child_fp]

    child = child_fp + child_sp
    return child

# Crossover the population given mating pool and elitism size
def crossover_population(mating_pool, elitism_size):
    children = []
    length = len(mating_pool) - elitism_size
    pool = random.sample(mating_pool, len(mating_pool))

    for i in range(elitism_size):
        children.append(mating_pool[i])

    for i in range(length):
        child = crossover(pool[i], pool[len(mating_pool)-i-1])
        children.append(child)
    return children

# Swap Mutation
def mutate(individual, mutation_p):
    for swapped in range(len(individual)):
        if random.random() < mutation_p:
            swap_with = int(random.random() * len(individual))
            individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
    return individual

# Mutation over the entire population
def mutate_population(population, mutation_p):
    mutated_pop = []
    
    for i in range(len(population)):
        mutated_individual = mutate(population[i], mutation_p)
        mutated_pop.append(mutated_individual)
    return mutated_pop

# Creating the next generation
def next_generation(current_gen, elitism_size, tournament_size, mutation_p):
    ranked_pop = rank_individuals(current_gen)
    selection_results = selection(ranked_pop, elitism_size, tournament_size)
    mating_pool = create_mating_pool(current_gen, selection_results)
    children = crossover_population(mating_pool, elitism_size)
    next_gen = mutate_population(children, mutation_p)
    return next_gen

# Main loop
def genetic_algorithm(population, pop_size, elitism_size, mutation_p, generations):
    pop = initial_population(pop_size, population)
    print(f'Initial distance: ' + str(rank_individuals(pop)[0][1]))

    for i in range(generations):
        pop = next_generation(pop, elitism_size, mutation_p)
    
    print('Final distance: ' + str(rank_individuals(pop)[0][1]))
    best_individual_index = rank_individuals(pop)[0][0]
    best_individual = pop[best_individual_index]
    return best_individual

# Plotting the progress
def genetic_algorithm_plot(population, pop_size, elitism_size, tournament_size, mutation_p, generations):
    pop = initial_population(pop_size, population)
    print(f'Initial distance: ' + str(rank_individuals(pop)[0][1]))
    progress = []
    progress.append(rank_individuals(pop)[0][1])

    for i in range(generations):
        pop = next_generation(pop, elitism_size, tournament_size, mutation_p)
        progress.append(rank_individuals(pop)[0][1])
    
    print('Final distance: ' + str(rank_individuals(pop)[0][1]))
    return rank_individuals(pop)[0][1]
    # plt.plot(progress)
    # plt.ylabel('Distance')
    # plt.xlabel('Generation')
    # plt.show()

# TO IMPLEMENT #
def derandomized_genetic_algorithm_plot(population, pop_size, elitism_size, tournament_size, mutation_p, generations):
    return 0

def both_algorithms(randomized, cities_list):
    if randomized:
        return genetic_algorithm_plot(population=cities_list, pop_size=100, elitism_size=20, tournament_size=10, mutation_p=0.01, generations=100)
    else:
        return derandomized_genetic_algorithm_plot(population=cities_list, pop_size=100, elitism_size=20, tournament_size=10, mutation_p=0.01, generations=100)
        

if __name__ == '__main__':
    cities_list = []
    for i in range(100):
        cities_list.append(City(index = i + 1, x=int(random.random() * 200), y=int(random.random() * 200)))
    ### True is the flag whether to run randomized or not ###
    best_individual = both_algorithms(True, cities_list)