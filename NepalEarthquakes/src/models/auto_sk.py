import autosklearn.classification
from src.utils.model_wrapper import model_wrapper

class auto_sk(model_wrapper):

    def train(self, X,y):
        '''
        input X: numpy.ndarray of shape (n_smaples, n_features)
        input y: numpy.ndarray of shape (n_samples, )
        output: trained model
        '''
        clf = autosklearn.classification.AutoSklearnClassifier()
        clf.fit(X, y)
        return clf

if __name__ == "__main__":
    mod = auto_sk({"name":"auto_sk"})
    mod.train_and_score()
