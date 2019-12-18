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


if __name__ == '__main__':
    unittest.main()
