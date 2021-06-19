# Derandomized_GA
TODO:

Data Generation:
----------------
1) Create Data set
2) https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html
   https://github.com/ezstoltz/genetic-algorithm/blob/master/genetic_algorithm_TSP.ipynb


Standard GA: RAZ - :white_check_mark:
------------
1) Initialization
2) Selection - Tournament
3) Mutation - flip 2 indices
4) XO
5) Fitness

Embedding: SNIR - :white_check_mark:
----------
1) Turn Individual into a compress version with DeepWalk/Node2Vec.

Neural Network: SNIR - :white_check_mark:
---------------
1) Collect good mutation examples.
2) Train NN, use softmax for prediction which indices to flip
3) Create proof of concept.
4) Apply in the evolution process.

Visualization & Results: AMIR - :white_check_mark:
------------------------
1) Run experiments with different hyper paramters.
2) create nice plots with results
3) search for p-value with permutation testing. 

