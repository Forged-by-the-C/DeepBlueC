import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
import time

from src.utils.model_wrapper import model_wrapper

'''
http://drivendata.co/blog/richters-predictor-benchmark/
'''

HIDDEN_LAYER_SIZES = (400, 153, 198, 43, 146, 50, 28, 332, 302, 364, 45, 394, 90, 106, 56)
ITERS_PER_SAVE = 10 

class mlp(model_wrapper):

    def train(self, X,y, n_iter, cv, n_jobs):
        '''
        input X: numpy.ndarray of shape (n_smaples, n_features)
        input y: numpy.ndarray of shape (n_samples, )
        input n_iter: int, number of training iterations if doing a hyper parameter search
        input cv: int, number of cross folds to trian on
        input n_jobs: int, number of processoers to use if doing a hyper parameter seacrch
                            -1 indicates using all processors
        output: trained model
        '''
        clf = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZES, max_iter=1)
        clf.fit(X, y)
        return clf

    def train_by_iter(self):
        X,y = self.load_data("train")
        X_val,y_val = self.load_data("val")
        self.load_model()
        train_start = time.time()
        total_iterations = 0
        while True:
            for i in range(ITERS_PER_SAVE):
                self.clf.partial_fit(X,y)
                total_iterations += 1
                train_score = f1_score(y_true=y, y_pred=self.clf.predict(X), average='micro') 
                val_score = f1_score(y_true=y_val, y_pred=self.clf.predict(X_val), 
                        average='micro')  
                print("Time Elapsed {} :: Iter: {} Train Score: {:.4f} Val Score {:.4f}".format(
                    time.strftime("%H:%M:%S", time.gmtime(time.time() - train_start)), 
                    total_iterations, train_score, val_score))
            print("++ Starting to save model ++")
            self.save_model()
            print("++ Completed saving model ++")
            self.results_dict["time_to_train"] = time.time() - train_start
            self.results_dict["training_score"] = train_score 
            self.results_dict["val_score"] = val_score 
            self.log_results(print_to_screen=False)

if __name__ == "__main__":
    mod = mlp({"single":"mlp"})
    mod.train_and_score(save_model=True)
    mod.train_by_iter()
    #mod.load_and_score()
    #mod.load_and_predict_submission()
