import numpy as np, random, operator
import pandas as pd
import matplotlib.pyplot as plt

from graph_handler import *
from copy import deepcopy

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

    for i in range(len(ranked_pop)-elitism_size):
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



def swapPositions(individual, pos):
    individual[pos[0]], individual[pos[1]] = individual[pos[1]], individual[pos[0]]
    return individual

def one_swap(individual):
    swap_idx = random.sample(range(len(individual)),k=2)
    return swapPositions(individual, swap_idx),swap_idx

# def one_swap_idx(individual,swap_idx):
#     swap_idx = random.sample(range(len(individual)),k=2)
#     return swapPositions(individual, swap_idx),swap_idx

# Mutation over the entire population
def mutate_population(population, mutation_p):
    mutated_pop = []
    mutated_stats = []
    for i in range(len(population)):
        # mutated_individual = mutate(population[i], mutation_p)
        mutation_points_count = np.min([int(np.random.exponential(scale=2, size=1) + 1), 1])
        for _ in range(mutation_points_count):
            prev_fitness = rank_individuals([population[i]])
            mutated_individual, _ = one_swap(population[i])
        post_fitness = rank_individuals([mutated_individual])
        mutated_pop.append(mutated_individual)
        mutated_stats.append(post_fitness[0][1] - prev_fitness[0][1])
    return mutated_pop, mutated_stats

def mutate_population_derandomized(population,normal_edge_embeddings, model, buffer_X,buffer_y, buffer_size, epsilon):
    mutated_pop = []
    improvment_list = []
    idx_list = []
    route_list = []
    mutated_stats = []

    for i in range(len(population)):
        mutation_points_count = np.min([int(np.random.exponential(scale=2, size=1) + 1), 1])
        for j in range(mutation_points_count):
            embedded_ind = get_edge_embedding_individual(normal_edge_embeddings, population[i])
            if epsilon > random.random():
                idx = random.sample(range(len(population[i])),k=2)
            else:    
                idx = np.argsort(model.predict(embedded_ind.reshape(1,49,5)))[0][-2:]
            if j == (mutation_points_count - 1):
                prev_fitness = rank_individuals([population[i]])
                prev_individual = deepcopy(population[i])
            mutated_ind = swapPositions(population[i], idx)

        mutated_pop.append(mutated_ind)
        # mutated_ind, idx = one_swap(deepcopy(population[i]))
        post_fitness = rank_individuals([mutated_ind])

        mutated_stats.append(post_fitness[0][1] - prev_fitness[0][1])

        if post_fitness[0][1] - prev_fitness[0][1] < 0:
            # embedded_ind = get_edge_embedding_individual(normal_edge_embeddings, mutated_ind)
            embedded_ind = get_edge_embedding_individual(normal_edge_embeddings, prev_individual)
            idx_list.append((embedded_ind, idx))
            # route_list.append((get_idividual_individual_route(population[i]),idx))
            improvment_list.append(post_fitness[0][1] - prev_fitness[0][1])
        else:
            embedded_ind = get_edge_embedding_individual(normal_edge_embeddings, mutated_ind) 
            idx_list.append((embedded_ind, idx))
            # route_list.append((get_idividual_individual_route(mutated_ind),idx))
            improvment_list.append(prev_fitness[0][1] - post_fitness[0][1])


    imp_idx = np.argsort(improvment_list)[0:int(len(idx_list)/2)]

    X = np.zeros((len(idx_list),49,5))
    y = np.zeros((len(idx_list),50))

    for i in range(len(idx_list)):
        X[i] = idx_list[i][0]
        y[i][idx_list[i][1]] = 1
        noise_idx = []
        for _ in range(1):
           noise_idx.append(random.choice(range(len(y[i]))))
        y[i][noise_idx] = 1


    i = 0
    # while buffer_size[0] < buffer_X.shape[0] and i < int(len(improvment_list)/2):
    # buffer_X[:] =  deepcopy(X[imp_idx])
    # buffer_y[:] =  deepcopy(y[imp_idx])
    buffer_X[:] =  X[imp_idx]
    buffer_y[:] =  y[imp_idx]
    buffer_size[0] = buffer_X.shape[0]

    # i = 0
    # while buffer_size[0] < buffer_X.shape[0] and i < int(len(improvment_list)/2):
    #     buffer_X[buffer_size[0]] =  deepcopy(X[imp_idx][i])
    #     buffer_y[buffer_size[0]] =  deepcopy(y[imp_idx][i])
    #     buffer_size[0] += 1
    #     i += 1
    # if i == 0:
    #     buffer_idx = random.sample(range(buffer_X.shape[0]),int(len(improvment_list)/2))
    #     buffer_X[buffer_idx] = deepcopy(X[imp_idx])
    #     buffer_y[buffer_idx] = deepcopy(y[imp_idx])

    return mutated_pop, mutated_stats


# Creating the next generation
def next_generation(current_gen, elitism_size, tournament_size, mutation_p):
    ranked_pop = rank_individuals(current_gen)
    selection_results = selection(ranked_pop, elitism_size, tournament_size)
    mating_pool = create_mating_pool(current_gen, selection_results)
    children = crossover_population(mating_pool, elitism_size)

    next_gen,mutated_stats = mutate_population(children, mutation_p)
    return next_gen, mutated_stats

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
    mutated_stats_list = []
    for i in range(generations):
        pop, mutated_stats = next_generation(pop, elitism_size, tournament_size, mutation_p)
        progress.append(rank_individuals(pop)[0][1])
        mutated_stats_list.append(mutated_stats)
    
    print('Final distance: ' + str(min(progress)))
    return progress, mutated_stats_list
    # plt.plot(progress)
    # plt.ylabel('Distance')
    # plt.xlabel('Generation')
    # plt.show()

# TO IMPLEMENT #

def update_model(model, buffer_X, buffer_y, buffer_size):
    # model.fit(buffer_X[0:buffer_size[0],:], buffer_y[0:buffer_size[0]], batch_size=64, epochs=1)
    # print(np.argsort(buffer_y[0])[-2:])
    # print(np.argsort(buffer_y[-1])[-2:])
    # model.fit(buffer_X, buffer_y, batch_size=int(buffer_size[0]/4), epochs=1)
    model.fit(buffer_X, buffer_y, batch_size=buffer_X.shape[0], epochs=1)


def next_generation_derandomized(current_gen, elitism_size, tournament_size, mutation_p, model, normal_edge_embeddings, buffer_X, buffer_y, buffer_size, epsilon):
    ranked_pop = rank_individuals(current_gen)
    selection_results = selection(ranked_pop, elitism_size, tournament_size)
    mating_pool = create_mating_pool(current_gen, selection_results)
    children = crossover_population(mating_pool, elitism_size)
    next_gen,mutated_stats = mutate_population_derandomized(population = children,
        normal_edge_embeddings = normal_edge_embeddings , model = model,
        buffer_X = buffer_X, buffer_y = buffer_y, buffer_size = buffer_size, epsilon = epsilon)
    # if buffer_size[0] == buffer_X.shape[0]:
    update_model(model, buffer_X, buffer_y, buffer_size)
    return next_gen,mutated_stats

from model import get_model

def derandomized_genetic_algorithm_plot(population, pop_size, elitism_size,tournament_size, mutation_p, generations, buffer_size_, epsilon):
    
    pop = initial_population(pop_size, population)
    print(f'Initial distance: ' + str(rank_individuals(pop)[0][1]))
    progress = []
    progress.append(rank_individuals(pop)[0][1])
    mutated_stats_list = []

    model = get_model()
    G = create_city_graph(population)
    node_embeddings, edge_embeddings, graph2vec = embed_graph(G, dimensions = 5, walk_length=10, num_walks=2000, workers=4, window=10, min_count=1, batch_words=4)
    normal_node_embeddings = node_embedding_normalizer(node_embeddings)
    normal_edge_embeddings = edge_embedding_normalizer(edge_embeddings)
    # get_edge_embedding_individual(normal_edge_embeddings, pop[0])
    # get_node_embedding_individual(normal_node_embeddings, pop[0])



    buffer_X = np.zeros((buffer_size_,49,5))
    buffer_y = np.zeros((buffer_size_,50))
    buffer_size = [0]
    for i in range(generations):
        pop, mutated_stats = next_generation_derandomized(current_gen = pop, elitism_size = elitism_size,
         tournament_size = tournament_size, mutation_p = mutation_p, model = model,
          normal_edge_embeddings = normal_edge_embeddings, buffer_X = buffer_X,
           buffer_y = buffer_y, buffer_size = buffer_size, epsilon = epsilon)
        progress.append(rank_individuals(pop)[0][1])
        mutated_stats_list.append(mutated_stats)
    print(f'Finale distance: ' + str(min(progress)))
    return progress, mutated_stats_list
    # plt.plot(progress)
    # plt.ylabel('Distance')
    # plt.xlabel('Generation')
    # plt.show()


def both_algorithms(randomized, cities_list):
    if randomized:
        return genetic_algorithm_plot(population=cities_list, pop_size=1000, elitism_size=2, tournament_size=3, mutation_p=0.01, generations=10)
    else:
        return derandomized_genetic_algorithm_plot(population=cities_list, pop_size=1000, elitism_size=1, tournament_size=3, mutation_p=0.01, generations=15, buffer_size_ = 250, epsilon=0.1)

from excel_handler import writeToExcel, eval_df
def get_statistics(iterations=30):
# iterations=3
    progress_list_rand = []
    mutated_stats_list_list_rand = []

    progress_list_new = []
    mutated_stats_list_list_new = []

    for i in range(iterations):
        cities_list = []
        for i in range(50):
            cities_list.append(City(index = i, x=int(random.random() * 1000), y=int(random.random() * 1000)))
        
        random.seed(None)
        temp_seed = random.randint(0,10**10)
        
        random.seed(temp_seed)
        progress_rand , mutated_stats_list_rand = genetic_algorithm_plot(population=cities_list, pop_size=1000, elitism_size=1, tournament_size=2, mutation_p=0.01, generations=15)
        progress_list_rand.append(progress_rand)
        mutated_stats_list_list_rand.append(mutated_stats_list_rand)
        
        random.seed(temp_seed)
        progress_new , mutated_stats_list_new = derandomized_genetic_algorithm_plot(population=cities_list, pop_size=1000, elitism_size=1, tournament_size=2, mutation_p=0.01, generations=15, buffer_size_ = 500, epsilon=0.5)

        progress_list_new.append(progress_new)
        mutated_stats_list_list_new.append(mutated_stats_list_new)

    writeToExcel('rand.xlsx', 'mutation_stats', pd.DataFrame(mutated_stats_list_list_rand))
    writeToExcel('rand.xlsx', 'progress', pd.DataFrame(progress_list_rand))
    
    writeToExcel('new.xlsx', 'mutation_stats', pd.DataFrame(mutated_stats_list_list_new))
    writeToExcel('new.xlsx', 'progress', pd.DataFrame(progress_list_new))

get_statistics(iterations=30)




# r = eval_df(pd.read_excel('rand.xlsx', sheet_name='mutation_stats', header=None))
# q = pd.read_excel('rand.xlsx', sheet_name='progress', header=None)

# q.values[0]


    
# new_r = eval_df(r)

# type(r.iloc[:,0])
# r.iloc[0,0] = ast.literal_eval(r.iloc[0,0])

# r

# import ast
# len(ast.literal_eval(r.values[0][:]))
# import json

# len(json.loads(r.values[0][0]))




# len(r.values[0].astype(list)[0])
# r.values[0]
# plt.hist(mutated_stats_list_list[0][9])
# plt.show()

# cities_list = []
# for i in range(50):
#     cities_list.append(City(index = i, x=int(random.random() * 1000), y=int(random.random() * 1000)))

# random.seed(5)
# pop1 = initial_population(3, cities_list)
# random.seed(None)
# pop2 = initial_population(3, cities_list)
# pop1 == pop2

# if __name__ == '__main__':
# cities_list = []
# for i in range(50):
#     cities_list.append(City(index = i, x=int(random.random() * 1000), y=int(random.random() * 1000)))

# best_individual = both_algorithms(False, cities_list)
