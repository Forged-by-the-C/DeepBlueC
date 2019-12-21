import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
import time

from src.utils.model_wrapper import model_wrapper
from ga import ga

'''
http://drivendata.co/blog/richters-predictor-benchmark/
'''

#TODO: Fix Logging

num_iters_per_train = 1

#Silence sklearn convergence warnings
def warn(*args, **kwargs):
        pass
import warnings
warnings.warn = warn

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
        X,y = self.load_data("train")
        X_val,y_val = self.load_data("val")
        prev_best_fitness = 0
        if load_population:
            print("Loading population from file")
            self.ga = ga.load_csv()
            print(self.ga.population)
            input()
            self.ga.breed()
            prev_best_fitness = self.ga.fit_df.fitness.max()
        else:
            self.ga = ga(population_size=10, chromosome_max_len=10, gene_max=300, gene_min=0)
            self.ga.gen_population()
        while True:
            gen_number = 0
            best_fitness = 0
            fit_list = []
            pop_size = self.ga.population.shape[0]
            gen_tic = time.time()
            for i in range(pop_size):
                tic = time.time()
                hidden_layer_sizes = self.ga.trim_to_tuple(i)
                if prev_best_fitness > 0 and i==0:
                    #Current the algorithm puts the best chromosome from the previous generation first
                    fitness = prev_best_fitness
                else:    
                    model = self.train(X,y,-1,hidden_layer_sizes)
                    train_score = f1_score(y_true=y, y_pred=model.predict(X), average='micro') 
                    fitness = f1_score(y_true=y_val, y_pred=model.predict(X_val), average='micro') 
                    print("*"*5 + " Child {} of {} :: Model: {} Train Score: {:.4f} Val Score: {:4f} Time to train: {:.1f}".format(
                        i, pop_size, hidden_layer_sizes, train_score, fitness, time.time() - tic))
                if fitness > best_fitness:
                    best_params = hidden_layer_sizes
                    best_fitness = fitness
                fit_list.append(fitness)
            self.ga.rank_fitness(fit_list)
            self.ga.to_csv()
            if prev_best_fitness == best_fitness:
                print("#"*15 + "No better genes found since gen {} just finished gen {}".format(best_gen, gen_number) + "#"*15)
            else:
                best_gen = gen_number
                prev_best_fitness = best_fitness
            print("#"*15 +"Best Model: {} Val Score: {:.4f} Time to train gen: {:.1f}".format(
                best_params, best_fitness, time.time() - gen_tic))
            gen_number += 1

if __name__ == "__main__":
    mod = mlp({"ga":"mlp"})
    mod.run_ga(load_population=True)
    #mod.load_and_score()
    #mod.load_and_predict_submission()
