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
        '''
        all arguments ints
        '''
        self.population_size = population_size
        self.chromosome_len = chromosome_max_len
        self.gene_max = gene_max
        self.gene_min = gene_min
        
    def gen_population(self):
        self.population = np.random.randint(low=self.gene_min,
                high=self.gene_max, size=(self.population_size,self.chromosome_len))
        self.compress_population()
    
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
        self.population = self.fit_df.drop(columns='fitness').to_numpy()

    def compress_population(self):
        '''
        This function compresses the chromosome to make sure it represents
            a structure the sklearn module can use
        (1,1,0,1) will not work, so it will be compressed to (1,1,1,0)
        Essentially, all 0's must be trailing
        This works by creating a boolean array and sorting it, 
            leaving in place any non-zero ordering
        '''
        self.population = np.take_along_axis(self.population, np.argsort(self.population == 0), 1)

    def raise_int_error(self, int_to_check,  min_, max_, int_name=""):
        '''
        input int_to_check: int
        input min_: minimum value of int, inclusive
        input max_: maximum value of int, inclusive
        input print_str: string to print if ValueError is raised
        '''
        if (int_to_check < min_) or (int_to_check > max_) or (type(int_to_check) != int):
            raise ValueError("{}: {} must be an int between {} and {}".format(int_to_check, int_name, min_, max_))

    def crossover(self, left_chromosome_index, right_chromosome_index):
        '''
        input left_chromosome_index: int, row index in population of left chromosome
        input right_chromosome_index: int, row index in population of right chromosome
        This is the breeding part of the genetic algorithm
        '''
        crossover_index = np.random.randint(0, self.population.shape[1]-1)
        left_mask = np.where(np.arange(self.population.shape[1]) <= crossover_index, 1, 0)
        return self.population[left_chromosome_index]*left_mask + \
                self.population[right_chromosome_index]*(1 - left_mask)

    def mutate(self, genes_to_mutate=1):
        '''
        input genes_to_mutate: int, number of genest to mutate per chromosome, in probability
        '''
        probability=max(1, genes_to_mutate)/float(self.chromosome_len)
        mutate_map = np.where(np.random.uniform(size=self.population.shape) < probability, 1, 0)
        self.population += mutate_map * np.around(np.random.randn(self.population.shape[0], 
                self.population.shape[1])*self.gene_max).astype(int)
        self.population = np.clip(self.population, self.gene_min, self.gene_max)
        
    def breed(self, elitism_ratio=0.2):
        '''
        input elitism_ratio: float, ratio of population to allow to breed for next generation

        Breed new generation with elitisim
        Population should be sorted where top row is the most fit, bottom row least fit
        '''
        new_population = np.zeros((self.population_size,self.chromosome_len), dtype="int")
        num_breeders = max(2, int(round(elitism_ratio*self.population.shape[0])))
        for i in range(1, self.population_size):
            parents = np.random.choice(num_breeders, 2, replace=False)
            child = self.crossover(parents[0], parents[1])
            new_population[i] += child
        #make sure the best chromosome gets passed on to each subsequent population
        old_best = self.population[0]
        self.population = new_population
        self.mutate()
        self.population[0] = old_best
        self.compress_population()

    def to_csv(self, path="population.csv"):
        self.fit_df.to_csv(path, index=False)

    def load_csv(path="population.csv", population_size=None, chromosome_max_len=None, 
            gene_max=None, gene_min=0):
        '''
        input path: str, path to csv
        rest of the arguments are ints
        '''
        fit_df = pd.read_csv(path)
        population = fit_df.drop(columns='fitness').to_numpy()
        if population_size is None:
            population_size = fit_df.shape[0]
        if chromosome_max_len is None:
            chromosome_len = fit_df.shape[1]
        if gene_max is None:
            gene_max = np.amax(population)
        new_ga = ga(population_size, chromosome_max_len, gene_max, gene_min)
        new_ga.fit_df = fit_df
        new_ga.population = population
        return new_ga

    def trim_to_tuple(self, chromosome_number=0):
        '''
        input chromosome_number: int, index of chromosome in population
        output: tuple, without trailing 0's
        '''
        chromosome = self.population[chromosome_number]
        return tuple(np.delete(chromosome, np.argwhere(chromosome==0)))
    
