import random
from GA_TSP import City,initial_population,create_route, mutate

cities_list = []
for i in range(5):
    cities_list.append(City(index = i + 1, x=int(random.random() * 200), y=int(random.random() * 200)))

pop_size=1
pop=initial_population(pop_size, cities_list)

mutation_p = 0.5
mutate(pop[0], mutation_p)

create_route(cities_list)


def get_solution_subgraph(sol):
    pass










