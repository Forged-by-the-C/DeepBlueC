import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

from src.utils.model_wrapper import model_wrapper

class random_forest(model_wrapper):

    def train(self, X,y):
        '''
        input X: numpy.ndarray of shape (n_smaples, n_features)
        input y: numpy.ndarray of shape (n_samples, )
        output: trained model
        '''
        param_dict = {"md":16, "ne":100}
        clf = RandomForestClassifier(max_depth=param_dict["md"],
                n_estimators=param_dict["ne"],
                random_state=1)
        clf.fit(X, y)
        return clf

    def parameter_search(self):
        ##Params
        n_models = 10
        ne_ar = np.random.randint(10, 100, n_models)
        ##
        X,y = self.load_data("train")
        v_X,v_y = self.load_data("val")
        '''
        ## Delete when running for real
        n_samples = 1000
        X, y = self.trim(X,y, n_samples)
        ##
        '''
        best_score = 0
        best_str = ""
        dict_list = []
        for i in range(n_models):
            param_dict = {"ne":ne_ar[i]}
            clf = RandomForestClassifier(max_depth=16,
                    n_estimators=param_dict["ne"],
                    random_state=1)
            clf.fit(X, y)
            g = clf.predict(X)
            train_score = f1_score(y_true=y, y_pred=g, average='micro')
            g = clf.predict(v_X)
            val_score = f1_score(y_true=v_y, y_pred=g, average='micro')
            param_string = self.gen_param_string(param_dict) 
            print("{} Train: {:.4f} Cross Val {:.4f}".format(param_string,
                train_score, val_score))
            result_dict = param_dict
            result_dict["train_score"] = train_score
            result_dict["val_score"] = val_score
            dict_list.append(result_dict)
            if val_score > best_score:
                best_score = val_score
                self.param_string = param_string
                self.gen_model_file_path()
                self.clf = clf
        self.save_model()
        pd.DataFrame(dict_list).to_csv("results.csv", index=False)

if __name__ == "__main__":
    mod = random_forest({"md":16, "ne":10})
    #mod.parameter_search()
    mod.train_and_score()
    #mod.load_and_score()
    #mod.load_and_predict_submission()
