from collections import defaultdict

import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt 
sns.set(style='darkgrid')

def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])

class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            early_stopping_rule: float = 1e-5,
            plot: bool = False
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        self.early_stopping_rule: float = early_stopping_rule

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        self.loss_derivative2 = lambda y, z: y ** 2 * \
            self.sigmoid(-y * z) * (1 - self.sigmoid(-y * z))


    def fit_new_base_model(self, x, y, predictions):
        # boostrap sample
        rng = np.random.default_rng()
        ind = rng.integers(low = 0, high = x.shape[0], endpoint=False,
             size=int(self.subsample*x.shape[0]))
        # model estimation 
        model = self.base_model_class()
        model.fit(x[ind], y[ind] - predictions[ind])
        y_hat = model.predict(x)
        # gamma
        gamma = self.find_optimal_gamma(y, predictions, y_hat)
        # storing values
        self.gammas.append(gamma)
        self.models.append(model)
        return y_hat

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        for r in range(self.n_estimators):
            train_new_pred = self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions += self.gammas[-1]*self.learning_rate*train_new_pred
            
            valid_new_pred = self.models[-1].predict(x_valid)
            valid_predictions += self.gammas[-1]*self.learning_rate*valid_new_pred
            
            train_loss = self.loss_fn(y_train, train_predictions)
            valid_loss = self.loss_fn(y_valid, valid_predictions)
            
            self.history['train'].append(train_loss)
            self.history['valid'].append(valid_loss)
                    
            if self.early_stopping_rounds is not None and r >= self.early_stopping_rounds:
                if np.all(np.diff(self.history['valid'][-(self.early_stopping_rounds+1):]) > -1e-5):
                        break 
        if self.plot:
            fig = sns.relplot(data=self.history, kind='line')
            fig.set_ylabels("Loss", clear_inner=False)
            fig.set_xlabels("Iteration", clear_inner=False)
        
        return self

    def predict_proba(self, x):
        y_hat = 0
        for gamma, model in zip(self.gammas, self.models):
            y_hat += gamma * model.predict(x)
        positive = self.sigmoid(y_hat)
        negative = 1 - positive
        return np.vstack((negative, positive)).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]
    
    def score(self, x, y):
        return roc_auc_score(y == 1, self.predict_proba(x)[:, 1])

    @property
    def feature_importances_(self, ):
        importances = np.zeros(len(self.models[0].feature_importances_))
        for gamma, model in zip(self.gammas, self.models):
            importances += gamma*model.feature_importances_
        return (importances - np.min(importances))/(np.max(importances) - np.min(importances))
