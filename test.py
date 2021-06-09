import networkx as nx
from node2vec import Node2Vec

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from GA_TSP import City,initial_population,create_route, mutate,Fitness,rank_individuals,initial_population,one_swap,swapPositions
from graph_handler import create_city_graph,embed_graph,get_edge_embedding_individual,node_embedding_normalizer,edge_embedding_normalizer,edge_embedding_normalizer,get_idividual_individual_route
import random


cities_list = []
for i in range(50):
    cities_list.append(City(index = i , x=int(random.random() * 1000), y=int(random.random() * 1000)))
G = create_city_graph(cities_list)

node_embeddings, edge_embeddings, model = embed_graph(G, dimensions = 5, walk_length=10, num_walks=2000, workers=4, window=10, min_count=1, batch_words=4)
normal_edge_embeddings = edge_embedding_normalizer(edge_embeddings)

dif_random = []
for _ in range(10000):
    pop = initial_population(1, cities_list)
    prev_fitness = rank_individuals(pop)
    mutated_ind, idx = one_swap(deepcopy(pop[0]))
    post_fitness = rank_individuals([mutated_ind])
    dif_random.append(post_fitness[0][1] - prev_fitness[0][1])
import matplotlib.pyplot as plt
print(np.mean(dif_random))
print(np.std(dif_random))
plt.hist(dif_random)
plt.show()


import pandas as pd
idx_list = []
route_list = []
improvment_list =[]
data = pd.DataFrame()

for _ in range(50000):
    pop = initial_population(1, cities_list)
    prev_fitness = rank_individuals(pop)
    mutated_ind, idx = one_swap(deepcopy(pop[0]))
    post_fitness = rank_individuals([mutated_ind])
    if post_fitness[0][1] - prev_fitness[0][1] < 0:
        # embedded_ind = get_edge_embedding_individual(normal_edge_embeddings, mutated_ind)
        embedded_ind = get_edge_embedding_individual(normal_edge_embeddings, pop[0])
        idx_list.append((embedded_ind, idx))
        route_list.append((get_idividual_individual_route(pop[0]),idx))
        improvment_list.append(post_fitness[0][1] - prev_fitness[0][1])
        
        # idx_list.append((mutated_ind, idx))

# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/

X = np.zeros((len(idx_list),49,5))
y = np.zeros((len(idx_list),50))

for i in range(len(idx_list)):
    X[i] = idx_list[i][0]
    y[i][idx_list[i][1]] = 1





import numpy as np
from tensorflow import keras
from tensorflow.keras import layers,Sequential
from sklearn.model_selection import train_test_split
import functools
top2_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=2)

# model = Sequential()
# model.add(layers.LSTM(100, input_shape=(49, 5)))
# model.add(layers.Dense(50,activation="sigmoid"))

model = Sequential()
model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(50,activation="sigmoid"))
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])


imp_idx = np.argsort(improvment_list)[0:int(len(improvment_list)/2)]

X_train, X_test, y_train, y_test = train_test_split(X[imp_idx], y[imp_idx], test_size=0.2)

history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))

np.argsort(model.predict(X_test[0:1]))[0][-2:]


dif_deep = []
for _ in range(10000):
    pop = initial_population(1, cities_list)
    prev_fitness = rank_individuals(pop)
    # mutated_ind, idx = one_swap(deepcopy(pop[0]))
    embedded_ind = get_edge_embedding_individual(normal_edge_embeddings, pop[0])
    idx = np.argsort(model.predict(embedded_ind.reshape(1,49,5)))[0][-2:]
    mutated_ind = swapPositions(deepcopy(pop[0]), idx)
    post_fitness = rank_individuals([mutated_ind])
    dif_deep.append(post_fitness[0][1] - prev_fitness[0][1])
import matplotlib.pyplot as plt
print(np.mean(dif_deep))
print(np.std(dif_deep))
plt.hist(dif_deep)
plt.show()


plt.hist(dif_random, alpha=0.5)
plt.hist(dif_deep, alpha=0.5)
plt.show()



print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
max_features = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review

# Input for variable-length sequences of integers
# inputs = keras.Input(shape=(49,3), dtype="int32")
inputs = keras.Input(shape=(None,), dtype="int32")

# Embed each integer in a 128-dimensional vector
# x = layers.Embedding(max_features, 128)(inputs)
x = inputs
# Add 2 bidirectional LSTMs
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
# Add a classifier
outputs = layers.Dense(1, activation="categorical_crossentropy")(x)
model = keras.Model(inputs, outputs)
model.summary()

(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
    num_words=max_features
)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)


model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))



idx_list[0][0] 
get_edge_embedding_individual(normal_edge_embeddings, idx_list[0][0]).shape



get_edge_embedding_individual(normal_edge_embeddings, pop[0])


import matplotlib.pyplot as plt
plt.hist(dif)
plt.show()

get_node_order(pop[0])




one_swap(individual)
def swapPositions(list, pos):
    list[pos[0]], list[pos[1]] = list[pos[1]], list[pos[0]]
    return list


rank_individuals([swapPositions(deepcopy(pop[0]),idx)])


pop[0]
idx = random.sample(range(5),k=2)
swapPositions(pop[0], idx)

# A = np.array([[1, 1,0.5], [2, 1,0.5],[3,1,0.5]])
# G = nx.from_numpy_matrix(A)
# nx.draw(G, cmap = plt.get_cmap('jet'))#, node_color = values)
# plt.show()




# edges = [(1, 2), (1, 6), (2, 3), (2, 4), (2, 6), 
#          (3, 4), (3, 5), (4, 8), (4, 9), (6, 7)]
  
# G.add_edges_from(edges)
# nx.draw_networkx(G, with_label = True)
  
# print("Total number of nodes: ", int(G.number_of_nodes()))
# print("Total number of edges: ", int(G.number_of_edges()))
# print("List of all nodes: ", list(G.nodes()))
# print("List of all edges: ", list(G.edges(data = True)))
# print("Degree for all nodes: ", dict(G.degree()))
  
# print("Total number of self-loops: ", int(G.number_of_selfloops()))
# print("List of all nodes with self-loops: ",
#              list(G.nodes_with_selfloops()))
  
# print("List of all nodes we can go to in a single step from node 2: ",
#                                                  list(G.neighbors(2)))








# # Create a graph
# graph = nx.fast_gnp_random_graph(n=100, p=0.5)

# # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
# node2vec = Node2Vec(graph, dimensions=5, walk_length=30, num_walks=200, workers=4)  # Use temp_folder for big graphs

# # Embed nodes
# model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

# model.wv.word_vec('10')
# # Look for most similar nodes
# model.wv.most_similar('2')  # Output node names are always strings

# # Save embeddings for later use
# # model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# # Save model for later use
# # model.save(EMBEDDING_MODEL_FILENAME)

# # Embed edges using Hadamard method
# from node2vec.edges import HadamardEmbedder

# edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
