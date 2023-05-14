"""
Class for text classification.

Author: Laura Bock Paulsen (202005791)
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

class TextClassification:
    def __init__(self, vec_type='tfidf', clf_type='logistic', **kwargs):
        # check that the input is valid
        if vec_type not in ['tfidf', 'count']:
            raise ValueError('vec_type must be either tfidf or count')
        
        if clf_type not in ['logistic', 'mlp']:
            raise ValueError('clf_type must be either logistic or mlp')
        
        self.vec_type = vec_type
        self.clf_type = clf_type

        # add attributes to the class based on the keyword arguments (if they exist)
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.vec = self._get_vectorizer()
        self.clf = self._get_classifier()

    def _update_params(self, params):
        """
        Updates the default parameters with the ones specified in the class attributes if they exist.
        """
        for key, value in params.items(): # loop over the default parameters
            if hasattr(self, key): # check if the attribute exists in the class
                try: 
                    params[key] = getattr(self, key)
                except AttributeError: # if the key does not exist, do nothing and continue
                    pass

    def _get_vectorizer(self):
        """
        Returns the vectorizer based on the vec_type attribute.
        """
        params = {'ngram_range': (1,2), 
             'lowercase':  True,
             'max_df': 0.95,           
             'min_df': 0.05,
             'max_features': 500}
        
        self._update_params(params)
    
        if self.vec_type == 'tfidf':
            return TfidfVectorizer(**params)
        
        elif self.vec_type == 'count':
            return CountVectorizer(**params)
    
    def _get_classifier(self):
        """
        Returns the classifier based on the clf_type attribute.
        """
        params = {'random_state': 7, 
                  'max_iter': 1000, 
                  'hidden_layer_sizes': (20, 20, 20)}

        self._update_params(params)

        if self.clf_type == 'logistic':
            # remove the hidden layer sizes parameter
            params.pop('hidden_layer_sizes')
            return LogisticRegression(**params)
        
        elif self.clf_type == 'mlp':
            return MLPClassifier(**params)
    

    def train_test_predict(self, X, y):
        """
        Runs the model using train_test_split.

        Parameters
        ----------
        X : pandas.Series
            The text data.
        y : pandas.Series
            The labels.

        Returns
        -------
        y_pred : numpy.ndarray
            The predicted labels.
        y_test : numpy.ndarray
            The true labels.
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

        # vectorize the data
        X_train = self.vec.fit_transform(X_train)
        X_test = self.vec.transform(X_test)

        # fit the classifier
        self.clf.fit(X_train, y_train)

        # predict the labels
        y_pred = self.clf.predict(X_test)

        return y_pred, y_test
    
    @staticmethod
    def get_metrics(y_pred, y):
        return metrics.classification_report(y, y_pred)
    