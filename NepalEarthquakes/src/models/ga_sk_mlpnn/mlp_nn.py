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

num_iters_per_train = 1

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
            self.ga = ga.load_csv()
            self.ga.breed()
            prev_best_fitness = self.ga.fit_df.fitness.max()
        else:
            self.ga = ga(population_size=10, chromosome_max_len=10, gene_max=300, gene_min=0)
            self.ga.gen_population()
        while True:
            best_fitness = 0
            fit_list = []
            tic = time.time()
            for i in range(self.ga.population.shape[0]):
                hidden_layer_sizes = self.ga.trim_to_tuple(i)
                if prev_best_fitness > 0 and i==0:
                    #Current the algorithm puts the best chromosome from the previous generation first
                    fitness = prev_best_fitness
                else:    
                    model = self.train(X,y,-1,hidden_layer_sizes)
                    fitness = f1_score(y_true=y, y_pred=model.predict(X), average='micro') 
                print("Model: {} F1 Score: {}".format(hidden_layer_sizes, fitness))
                if fitness > best_fitness:
                    best_model = model
                    best_params = hidden_layer_sizes
                    best_fitness = fitness
                fit_list.append(fitness)
            self.ga.rank_fitness(fit_list)
            self.ga.to_csv()
            #TODO: Fix Logging
            #print("-"*10 +"Logging Generation" + "-"*10)
            val_score = f1_score(y_true=y_val, y_pred=best_model.predict(X_val), average='micro')
            print("Best Model: {} Train Score: {} Val Score: {}".format(best_params, best_fitness, val_score))
            if False:
            #if prev_best_fitness < best_fitness:
                prev_best_fitness = best_fitness
                self.clf = best_model
                self.save_model()
                self.ga.breed()
                self.results_dict["time_to_train_generation"] = time.time() - tic
                self.results_dict["best_params"] = list(best_params)
                self.results_dict["training_score"] = best_fitness 
                self.results_dict["val_score"] = val_score 
                self.log_results()
                #else:
                print("*"*15 + "No better genes found since last generation" + "*"*15)

if __name__ == "__main__":
    mod = mlp({"ga":"mlp"})
    mod.run_ga(load_population=False)
    #mod.load_and_score()
    #mod.load_and_predict_submission()
