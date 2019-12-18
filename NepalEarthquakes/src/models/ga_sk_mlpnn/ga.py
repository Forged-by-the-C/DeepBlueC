'''
The goal of this script is to privide the functionality of a genetic algorithm
    for selecting the architecture of the mlp_nn of sklearn

This module has a simple feed forward network classifier with architecture determined
    by a tupe of default (100,), on a training set of X( m x n) and Y of c classes:
        n -> 100 -> 3
'''

import pandas as pd
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

    def rank_fitness(self, score_list):
        '''
        input score_list: list, list of fitness socres (float or int) of self.population by row
            len(score_list)=self.population.shape[0]
        '''
        if len(score_list) != self.population.shape[0]:
            raise ValueError("Fitness Score list name same length as population size")
        self.fit_df = pd.DataFrame(self.population)
        self.fit_df['fitness'] = score_list
        self.fit_df.sort_values('fitness', ascending=False, inplace=True)

    #TODO: Crossover
    #TODO: Mutation
    #TODO: Logging 


