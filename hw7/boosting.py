from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


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
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []
        self.train_history: list = []
        self.val_history: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        ids = np.random.choice(np.arange(x.shape[0]), size=int(self.subsample * x.shape[0]), replace=False)
        boot_x = x[ids]
        boot_y = y[ids]
        boot_pred = predictions[ids]
        
        model = self.base_model_class(**self.base_model_params)
        s = -self.loss_derivative(boot_y, boot_pred)
        model.fit(boot_x, s)
        gamma = self.find_optimal_gamma(boot_y, boot_pred, model.predict(boot_x))

        self.gammas.append(gamma)
        self.models.append(model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(x_valid.shape[0])
        n = 0
        for i in range(self.n_estimators):
            n = i
            self.fit_new_base_model(x_train, y_train, train_predictions)

            train_predictions += self.models[-1].predict(x_train) * self.gammas[-1] * self.learning_rate
            valid_predictions += self.models[-1].predict(x_valid) * self.gammas[-1] * self.learning_rate
            self.train_history.append(self.loss_fn(y_train, train_predictions))
            self.val_history.append(self.loss_fn(y_valid, valid_predictions))

            if self.early_stopping_rounds is not None:
                if self.early_stopping_rounds > i:
                    if self.val_history[-2] < self.val_history[-1]:
                        break
                
            

        if self.plot:
            fig, axs = plt.subplots(figsize=(16, 5), ncols=2)
            axs[0].plot(np.arange(n + 1), self.train_history)
            axs[0].set_title = 'train loss'
            axs[1].plot(np.arange(n + 1), self.val_history)
            axs[1].set_title = 'valid loss'
            #plt.show()

    def predict_proba(self, x):
        pred = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            pred += model.predict(x) * gamma * self.learning_rate
        pred = self.sigmoid(pred)
        return np.array([1 - pred, pred]).T


    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        pass
