import pickle as pkl
from sklearn.svm import SVC
from utils.features import get_text_features
from utils.preprocess import Prepeocessing as prep
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

class language_level_model():

    def __init__(self):
        self.model = Pipeline([
            ('vectorizer', DictVectorizer(sparse=False)),
            ('classifier', SVC(kernel='linear', class_weight="balanced",probability=True))
        ])
    
    def set_model(self, model):
        """Sets the model to be used for predictions."""
        self.model = model

    
    def __vectorize(self,texts,with_pos=False):
        return texts.apply(lambda x: get_text_features(x,with_pos=with_pos))

    def fit(self,X,y):
        """Takes in unprocessed data and trains the model."""
        X_vectorized = self.__vectorize(X,with_pos=False)
        y = prep.encode_labels(y)
        pass

    def predict(self,X_test,y_test):
        pass

    def save(self, path):
        """Save a trained model to a file."""
        pkl.dump(self.model, open(path, 'wb'))

    def load(self, path):
        """Load a pre-trained model from a file."""
        model = pkl.load(open(path, 'rb'))
