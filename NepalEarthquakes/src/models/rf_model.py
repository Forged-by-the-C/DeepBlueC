from src.models.model_wrapper import model_wrapper
from sklearn.ensemble import RandomForestClassifier

class random_forrest(model_wrapper):

    def train(self, X,y):
        '''
        input X: numpy.ndarray of shape (n_smaples, n_features)
        input y: numpy.ndarray of shape (n_samples, )
        output: trained model
        '''
        n_estimators = 100
        max_depth = 3
        clf = RandomForestClassifier(n_estimators=n_estimators, 
                max_depth=max_depth, random_state=1)
        clf.fit(X, y)
        return clf

if __name__ == "__main__":
    mod = random_forrest(["rf", "init"])
    mod.train_and_score()
