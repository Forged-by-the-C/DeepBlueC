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

num_iters_per_train = 1
pop_to_cpu_factor = 5

#Silence sklearn convergence warnings
def warn(*args, **kwargs):
        pass
import warnings
warnings.warn = warn

class mlp_ga(ga):

    def fitness(self, chromosome_number=0):
        hidden_layer_sizes = self.trim_to_tuple(chromosome_number)
        clf = MLPClassifier(hidden_layer_sizes, max_iter=num_iters_per_train)
        clf.fit(self.X, self.y)
        return f1_score(y_true=self.y_val, y_pred=clf.predict(self.X_val), average='micro') 

class mlp(model_wrapper):

    def train(self, X,y, n_jobs, hidden_layer_sizes):
        '''
        input X: numpy.ndarray of shape (n_smaples, n_features)
        input y: numpy.ndarray of shape (n_samples, )
        input n_iter: int, number of training iterations if doing a hyper parameter search
        input cv: int, number of cross folds to trian on
        input n_jobs: int, number of processoers to use if doing a hyper parameter seacrch
                            -1 indicates using all processors
        output: trained model
        '''
        clf = MLPClassifier(hidden_layer_sizes, max_iter=num_iters_per_train)
        clf.fit(X, y)
        return clf

    def run_ga(self, load_population=False):
        X, y = self.load_data("train")
        X_val, y_val = self.load_data("val")
        prev_best_fitness = 0
        if load_population:
            print("Loading population from file")
            self.ga = mlp_ga()
            self.ga.load_csv()
            self.ga.population_size = pop_to_cpu_factor*os.cpu_count()
            self.ga.breed()
            prev_best_fitness = self.ga.fit_df.fitness.max()
            best_params = self.ga.trim_to_tuple(0)
        else:
            self.ga = mlp_ga(population_factor=pop_to_cpu_factor, 
                    chromosome_max_len=10, gene_max=300, gene_min=0)
            self.ga.gen_population()
        self.ga.X = X
        self.ga.y = y
        self.ga.X_val = X_val
        self.ga.y_val = y_val
        while True:
            gen_number = 0
            best_gen = 0
            best_fitness = 0
            gen_tic = time.time()
            self.ga.multiprocess_iterate_generation()
            best_fitness = max(self.ga.fit_list)
            self.ga.to_csv()
            if prev_best_fitness == best_fitness:
                print("#"*15 + 
                        "No better genes found since gen {} just finished gen {}".format(
                            best_gen, gen_number) + "#"*15)
            else:
                best_gen = gen_number
                prev_best_fitness = best_fitness
                best_params = self.ga.trim_to_tuple(np.asarray(self.ga.fit_list).argmax()) 
            print("#"*15 +"Best Model: {} Val Score: {:.4f} Time to train gen: {:.1f}".format(
                best_params, best_fitness, time.time() - gen_tic))
            gen_number += 1

if __name__ == "__main__":
    mod = mlp({"ga":"mlp"})
    mod.run_ga(load_population=False)
    #mod.load_and_score()
    #mod.load_and_predict_submission()
