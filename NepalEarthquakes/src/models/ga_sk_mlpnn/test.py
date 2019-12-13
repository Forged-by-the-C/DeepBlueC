'''
For the sake of testing this particular algorithm structure of variable lenght 
    up to chromosome_max_len, our fitness function will be 
        a1 - || 0.5*a1 - a2|| - || 0.5*a2 - a3|| - a4 - a5 ...

Where the optimal solution should be
    a1 = cls.gene_max
    a2 = 0.5 * a1
    a3 = 0.5 * a2
    a4 = a5 = .. = 0
'''

import numpy as np
import unittest

from ga import ga

class TestGA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.population_size = 6
        cls.chromosome_max_len = 5
        cls.gene_max = 100
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


if __name__ == '__main__':
    unittest.main()
