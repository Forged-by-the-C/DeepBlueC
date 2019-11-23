import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import xgboost as xgb
from src.utils.model_wrapper import model_wrapper

class xg_boosting(model_wrapper):

    def train(self, X,y):
        '''
        input X: numpy.ndarray of shape (n_smaples, n_features)
        input y: numpy.ndarray of shape (n_samples, )
        output: trained model
        '''
        param_dict = {"md":3, "cw": 6, "gamma": 1.5}
        clf = xgb.XGBClassifier(max_depth=param_dict["md"],
                min_child_weight=param_dict["cw"], 
                gamma=param_dict["gamma"],
                random_state=1)
        clf.fit(X, y)
        return clf

    def parameter_search(self):
        ##Params
        n_models = 10
        md_ar = np.random.randint(1, 10, n_models)
        cw_ar = np.random.randint(1, 10, n_models)
        g_ar = np.random.uniform(1,3,n_models)
        ##
        X,y = self.load_data("train")
        v_X,v_y = self.load_data("val")
        ## Delete when running for real
        n_samples = 5000
        X, y = self.trim(X,y, n_samples)
        ##
        best_score = 0
        best_str = ""
        dict_list = []
        for i in range(n_models):
            param_dict = {"md":md_ar[i], "cw": cw_ar[i], "gamma": g_ar[i]}
            clf = xgb.XGBClassifier(max_depth=param_dict["md"],
                    min_child_weight=param_dict["cw"], 
                    gamma=param_dict["gamma"],
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
    mod = xg_boosting({"md":3, "cw": 6, "gamma": 1.5})
    #mod.parameter_search()
    mod.train_and_score()
    #mod.load_and_score()
    #mod.load_and_predict_submission()
