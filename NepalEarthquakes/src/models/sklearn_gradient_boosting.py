from src.utils.model_wrapper import model_wrapper
from sklearn.ensemble import GradientBoostingClassifier

class gradient_boosting(model_wrapper):

    def train(self, X,y):
        '''
        input X: numpy.ndarray of shape (n_smaples, n_features)
        input y: numpy.ndarray of shape (n_samples, )
        output: trained model
        '''
        n_estimators = 10
        max_depth = 3
        clf = GradientBoostingClassifier(n_estimators=n_estimators, 
                max_depth=max_depth, random_state=1)
        clf.fit(X, y)
        return clf

if __name__ == "__main__":
    mod = gradient_boosting(["gb", "10", "3"])
    mod.train_and_score()
    #mod.load_and_score()
    #mod.load_and_predict_submission()
