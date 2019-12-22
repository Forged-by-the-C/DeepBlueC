import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
import time

from src.utils.model_wrapper import model_wrapper
from ga import ga

'''
http://drivendata.co/blog/richters-predictor-benchmark/
'''

NUM_ITERS_PER_TRAIN = 1
POP_TO_CPU_FACTOR = 1
CHROMOSOME_MAX_LEN = 15
GENE_MAX = 300
GENE_MIN = 0

#Silence sklearn convergence warnings
def warn(*args, **kwargs):
        pass
import warnings
warnings.warn = warn

class mlp_ga(ga):

    def fitness(self, chromosome_number=0):
        hidden_layer_sizes = self.trim_to_tuple(chromosome_number)
        clf = MLPClassifier(hidden_layer_sizes, max_iter=NUM_ITERS_PER_TRAIN)
        clf.fit(self.X, self.y)
        score = f1_score(y_true=self.y_val, y_pred=clf.predict(self.X_val), average='micro') 
        print("--val_score: {:.4f} Chromosome: {} ".format(score, hidden_layer_sizes))
        return score 

class mlp(model_wrapper):

    def run_ga(self, load_population=False):
        X, y = self.load_data("train")
        X_val, y_val = self.load_data("val")
        prev_best_fitness = 0
        if load_population:
            print("Loading population from file")
            self.ga = mlp_ga()
            self.ga.load_csv()
            self.ga.population_size = POP_TO_CPU_FACTOR*os.cpu_count()
            self.ga.breed()
            prev_best_fitness = self.ga.fit_df.fitness.max()
            best_params = self.ga.trim_to_tuple(0)
        else:
            self.ga = mlp_ga(population_factor=POP_TO_CPU_FACTOR, 
                    chromosome_max_len=CHROMOSOME_MAX_LEN, gene_max=GENE_MAX, gene_min=GENE_MIN)
            self.ga.gen_population()
        self.ga.X = X
        self.ga.y = y
        self.ga.X_val = X_val
        self.ga.y_val = y_val
        gen_number = 0
        best_gen = 0
        train_start = time.time()
        while True:
            print("{} :: Training {} chromosomes in Gen {}".format(time.strftime("%H:%M:%S", 
                time.gmtime(time.time() - train_start)), self.ga.population_size, gen_number))
            best_fitness = 0
            gen_tic = time.time()
            self.ga.multiprocess_iterate_generation()
            best_fitness = max(self.ga.fit_list)
            self.ga.to_csv()
            if prev_best_fitness >= best_fitness:
                print(" "* 5 + "!"*5 + 
                        " No better genes found since gen {} just finished gen {}".format(
                            best_gen, gen_number))
            else:
                best_gen = gen_number
                prev_best_fitness = best_fitness
                best_params = self.ga.trim_to_tuple(np.asarray(self.ga.fit_list).argmax()) 
            print(" "*5  + 
                "Best Model: {} using {} of {} genes with gene_max {}. Val Score: {:.4f} Time to train gen: {:.1f}".format(
                best_params, len(best_params), self.ga.chromosome_len, self.ga.gene_max, prev_best_fitness, time.time() - gen_tic))
            gen_number += 1

if __name__ == "__main__":
    mod = mlp({"ga":"mlp"})
    mod.run_ga(load_population=False)
    #mod.load_and_score()
    #mod.load_and_predict_submission()
