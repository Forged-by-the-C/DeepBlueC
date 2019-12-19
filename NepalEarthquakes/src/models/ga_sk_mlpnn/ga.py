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

    def compress_population(self):
        '''
        This function compresses the chromosome to make sure it represents
            a structure the sklearn module can use
        (1,1,0,1) will not work, so it will be compressed to (1,1,1,0)
        Essentially, all 0's must be trailing
        '''
        chromosome_len = self.population.shape[1]
        zero_mask = np.where(self.population==0, 0, 1)
        #The amount of non-zero elements in each row
        #the masking will remove any rows that were all 0's
        row_lens = np.ma.masked_equal(zero_mask.sum(axis=1),0).compressed()
        population_size = len(row_lens)
        if len(row_lens) == 0:
            raise ValueError("Evolution led to an array of 0's")
        #Create the new population array
        new = np.zeros((population_size, chromosome_len), dtype='int')
        #Create a 1-dimensional array of all non-zero elements in population
        non_zero = np.ma.masked_equal(self.population, 0).compressed()
        last_index = 0
        #Iterate through non-zero elements array, grabbing the amount that
        #   belong in each row
        for i in range(population_size):
            new[i][:row_lens[i]] += non_zero[last_index:last_index + row_lens[i]]
            last_index += row_lens[i]
        self.population = new

    #TODO: Crossover
    #TODO: Mutation
    #TODO: Logging 


