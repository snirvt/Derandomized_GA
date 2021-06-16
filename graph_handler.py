# To import package
import networkx as nx
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
from sklearn.preprocessing import StandardScaler
# from GA_TSP import City,initial_population,create_route, mutate
# import random
import numpy as np

# https://www.geeksforgeeks.org/directed-graphs-multigraphs-and-visualization-in-networkx/
# https://www.geeksforgeeks.org/networkx-python-software-package-study-complex-networks/
# https://github.com/eliorc/node2vec
# http://ise.thss.tsinghua.edu.cn/~wangchaokun/edge2vec/tkdd_embedding_accepted.pdf

def create_city_graph(cities_list):
    G = nx.Graph()
    weighted_edges = []
    for i in range(len(cities_list)):
        for j in range(len(cities_list)):
            if i != j:
                weighted_edges+=[(cities_list[i].index, cities_list[j].index, cities_list[i].distance(cities_list[j]))]
    G.add_weighted_edges_from(weighted_edges)
    return G

def edge_embedding_normalizer(edge_embedding):
    scaler = StandardScaler()
    node1_size, node2_size, embedding_size = edge_embedding.shape
    return scaler.fit_transform(
        edge_embedding.reshape(node1_size*node2_size, embedding_size)).reshape(
            node1_size, node2_size, embedding_size)

def node_embedding_normalizer(node_embedding):
    scaler = StandardScaler()
    return scaler.fit_transform(node_embedding)


def embed_graph(G, dimensions = 3, walk_length=30, num_walks=200, workers=4, window=10, min_count=1, batch_words=4):
    node_embbeding = np.zeros((len(G.nodes()), dimensions))
    edge_embedding = np.zeros((len(G.nodes()),len(G.nodes()), dimensions))
    node2vec = Node2Vec(G, dimensions, walk_length, num_walks, workers)  # Use temp_folder for big graphs
    model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
    for node in G.nodes():
        node_embbeding[node,:] = model.wv.get_vector(str(node))
        for node2 in G.nodes():
            edge_embedding[node,node2] = edges_embs[(str(node),str(node2))]
    # return model.wv.get_normed_vectors()
    return node_embbeding,edge_embedding, model

def get_idividual_individual_route(individual):
    return [ind.index for ind in individual]

def get_edge_embedding_individual(edge_embeddings, individual):
    node_order = get_idividual_individual_route(individual)
    return edge_embeddings[node_order[0:-1],node_order[1:],:]

def get_node_embedding_individual(node_embeddings, individual):
    node_order = get_idividual_individual_route(individual)
    return node_embeddings[node_order,:]



# cities_list = []
# for i in range(5):
#     cities_list.append(City(index = i , x=int(random.random() * 200), y=int(random.random() * 200)))

# G = create_city_graph(cities_list)
# node_embeddings, edge_embeddings, model = embed_graph(G, dimensions = 3, walk_length=30, num_walks=200, workers=4, window=10, min_count=1, batch_words=4)
# normal_node_embeddings = node_embedding_normalizer(node_embeddings)
# normal_edge_embeddings = edge_embedding_normalizer(edge_embeddings)


# pop_size=2
# pop=initial_population(pop_size, cities_list)


# get_idividual_individual_route(pop[0])
# get_edge_embedding_individual(normal_edge_embeddings, pop[0])
# get_node_embedding_individual(normal_node_embeddings, pop[0])



