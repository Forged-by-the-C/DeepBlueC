'''
For the sake of testing this particular algorithm structure of variable lenght 
    up to chromosome_max_len, our fitness function will be 
        a0 - | 0.5*a0 - a1 | - | 0.5*a1 - a2 | - a3 - a4 ...

Where the optimal solution should be
    a0 = cls.gene_max
    a1 = 0.5 * a0
    a2 = 0.5 * a1
    a3 = a4 = .. = 0
'''

import numpy as np
import os
import unittest

from ga import ga

def fitness(tup):
    if len(tup) < 3:
        for i in range(len(tup),3):
            tup = list(tup)
            tup.append(0)
            tup = tuple(tup)
    out = tup[0] - abs(0.5*tup[0] - tup[1]) - abs(0.5*tup[1] - tup[2])
    if len(tup) > 3:
        for i in range(3,len(tup)):
            out -= tup[i]
    return out

class TestGA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.population_size = 6
        cls.chromosome_max_len = 5
        cls.gene_max = 25
        cls.gene_max *= 4 #test_fitness_ranking easier is gene max is multiple of 4
        cls.gene_min = 0
        cls.ga = ga(population_size=cls.population_size,
                chromosome_max_len=cls.chromosome_max_len,
                gene_max=cls.gene_max,
                gene_min=cls.gene_min)

    def test_pop_gen(cls):
        cls.ga.gen_population()
        pop_shape = cls.ga.population.shape
        cls.assertEqual(pop_shape[0], cls.population_size)
        cls.assertEqual(pop_shape[1], cls.chromosome_max_len)
        cls.assertTrue(np.max(cls.ga.population) <= cls.gene_max)
        cls.assertTrue(np.min(cls.ga.population) >= cls.gene_min)

    def test_fitness_func(cls):
        tup = (2,4,5)
        cls.assertEqual((2-3-3), fitness(tup))
        tup = (2,4,5,1)
        cls.assertEqual((2-3-3-1), fitness(tup))
        tup = (2,)
        cls.assertEqual(1, fitness(tup))
        tup = (cls.gene_max, 0.5*cls.gene_max, 0.5*0.5*cls.gene_max)
        cls.assertTrue(fitness(tup) > (cls.gene_max - 2))

    def test_fitness_ranking(cls):
        cls.ga.population = np.array(
                [[2,4,5,0],
                    [2,4,5,1],
                    [2,0,0,0],
                    [cls.gene_max, 0.5*cls.gene_max, 0.5*0.5*cls.gene_max, 0]])
        fit_list = []
        for i in range(cls.ga.population.shape[0]):
            fit_list.append(fitness(tuple(cls.ga.population[i])))
        cls.ga.rank_fitness(fit_list)
        df_fit_list = cls.ga.fit_df['fitness'].tolist()
        fit_list.sort(reverse=True)
        for i in range(len(fit_list)):
            cls.assertEqual(fit_list[i], df_fit_list[i])

    def test_fitness_ranking_error(cls):
        cls.ga.population = np.array(
                [[2,4,5,0],
                    [2,4,5,1],
                    [2,0,0,0],
                    [cls.gene_max, 0.5*cls.gene_max, 0.5*0.5*cls.gene_max, 0]])
        fit_list = [1]
        for i in range(cls.ga.population.shape[0]):
            fit_list.append(fitness(tuple(cls.ga.population[i])))
        cls.assertRaises(ValueError, cls.ga.rank_fitness, fit_list)

    def test_compression(cls):
        cls.ga.population = np.array(
                [[1,1,1,1],
                    [2,0,0,1],
                    [0,0,1,0]])
        compressed_pop = np.array(
                [[1,1,1,1],
                    [2,1,0,0],
                    [1,0,0,0]])
        cls.ga.compress_population()
        np.testing.assert_array_equal(cls.ga.population, compressed_pop)

    def test_crossover(cls):
        cls.ga.population = np.array(
                [[1,1,1,1],
                    [2,1,0,0],
                    [1,0,0,0]])
        crossed = cls.ga.crossover(0,1)
        cls.assertRaises(AssertionError, np.testing.assert_array_equal, 
                crossed, cls.ga.population[0])
        cls.assertRaises(AssertionError, np.testing.assert_array_equal, 
                crossed, cls.ga.population[1])

    def test_raise_int_error(cls):
        cls.ga.raise_int_error(2,1,3,"good")
        cls.assertRaises(ValueError, cls.ga.raise_int_error,2.5,1,3,"not_int")
        cls.assertRaises(ValueError, cls.ga.raise_int_error,0,1,3,"low")
        cls.assertRaises(ValueError, cls.ga.raise_int_error,4,1,3,"high")

    def test_mutataion(cls):
        cls.ga.population = np.zeros((10,10), dtype="int")
        cls.ga.mutate()
        cls.assertTrue(cls.ga.population.sum().sum() > 0)

    def test_breed(cls):
        cls.ga.gen_population()
        old_pop = cls.ga.population
        cls.ga.breed()
        cls.assertEqual(cls.ga.population.shape[0], cls.population_size) 
        cls.assertEqual(cls.ga.population.shape[1], cls.chromosome_max_len) 
        cls.assertRaises(AssertionError, np.testing.assert_array_equal, 
                cls.ga.population, old_pop)

    def test_breed_maintains_best(cls):
        cls.ga.gen_population()
        fit_list = []
        for i in range(cls.ga.population.shape[0]):
            fit_list.append(fitness(tuple(cls.ga.population[i])))
        cls.ga.rank_fitness(fit_list)
        old_best = cls.ga.population[0]
        cls.ga.breed()
        np.testing.assert_array_equal(cls.ga.population[0], old_best)

    def test_algo(cls):
        cls.ga.gen_population()
        num_generations = 10
        fitness_score = -100
        generation_counter = 0
        while (fitness_score<99) and (generation_counter<num_generations): 
            fit_list = []
            for i in range(cls.ga.population.shape[0]):
                fit_list.append(fitness(tuple(cls.ga.population[i])))
            fitness_score = max(fit_list)
            cls.ga.rank_fitness(fit_list)
            #print("Generation {} best chromosome: {}".format(generation_counter, fitness_score))
            cls.ga.breed() 
            generation_counter += 1
        cls.assertTrue(fitness_score>0)

    def test_to_csv(cls):
        cls.ga.population = np.array(
                [[2,4,5,0],
                    [2,4,5,1],
                    [2,0,0,0],
                    [cls.gene_max, 0.5*cls.gene_max, 0.5*0.5*cls.gene_max, 0]])
        fit_list = []
        for i in range(cls.ga.population.shape[0]):
            fit_list.append(fitness(tuple(cls.ga.population[i])))
        cls.ga.rank_fitness(fit_list)
        filename = "test_population.csv"
        cls.ga.to_csv(path=filename)
        cls.assertTrue(os.path.exists(filename)) 
        os.remove(filename)
        cls.assertTrue(~os.path.exists(filename)) 
    
    def test_load_csv(cls):
        cls.ga.population = np.array(
                [[2,4,5,0],
                    [2,4,5,1],
                    [2,0,0,0],
                    [cls.gene_max, 0.5*cls.gene_max, 0.5*0.5*cls.gene_max, 0]])
        fit_list = []
        for i in range(cls.ga.population.shape[0]):
            fit_list.append(fitness(tuple(cls.ga.population[i])))
        cls.ga.rank_fitness(fit_list)
        filename = "test_load_population.csv"
        cls.ga.to_csv(path=filename)
        new_ga = ga.load_csv(filename)
        np.testing.assert_array_equal(cls.ga.population, new_ga.population)
        os.remove(filename)

    def test_trim_to_tuple(cls):
        cls.ga.population = np.array(
                [[2,4,5,0],
                    [2,4,5,1],
                    [2,0,0,0]])
        cls.assertEqual(len(cls.ga.trim_to_tuple(0)), 3)
        cls.assertTrue(type(cls.ga.trim_to_tuple(1)) == tuple)

if __name__ == '__main__':
    unittest.main()
