'''
The goal of this script is to privide the functionality of a genetic algorithm
    for selecting the architecture of the mlp_nn of sklearn

This module has a simple feed forward network classifier with architecture determined
    by a tupe of default (100,), on a training set of X( m x n) and Y of c classes:
        n -> 100 -> 3
'''

import numpy as np

class ga():

    def __init__(self, population_size=10, chromosome_max_len=10, gene_max=300, gene_min=0):
        self.population_size = population_size
        self.chromosome_len = chromosome_max_len
        self.gene_max = gene_max
        self.gene_min = gene_min
        
    def gen_population(self):
        self.population = np.random.randint(low=self.gene_min,
                high=self.gene_max, size=(self.population_size,self.chromosome_len))

    #TODO: Fitness Raning
    #TODO: Crossover
    #TODO: Mutation
    #TODO: Logging 


