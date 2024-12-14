import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

class FeatureGradientSelector(BaseEstimator, SelectorMixin):
    def __init__(self,
                 n_features=None,
                 penalty=1,
                 learning_rate=1e-1,
                 n_epochs=1,
                 batch_size=1000,
                 preprocess='zscore',
                 verbose=0,
                 device='cpu'):
        self.n_features = n_features
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.verbose = verbose
        self.device = device

    def fit(self, X, y):
        """
        Fit the selector to the data.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        # Preprocess the data (e.g., standardization)
        X = self._preprocess_data(X)

        # Simulate training process (placeholder for actual training logic)
        self.scores_ = np.random.rand(X.shape[1])  # Random scores for illustration

        # Select top features based on scores
        self.selected_features_ = np.argsort(-self.scores_)[:self.n_features]

        return self

    def transform(self, X):
        """
        Transform the data to keep only the selected features.
        """
        check_is_fitted(self, 'selected_features_')

        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_features_]
        else:
            return X[:, self.selected_features_]

    def _preprocess_data(self, X):
        """Apply preprocessing to the data."""
        if self.preprocess == 'zscore':
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            return (X - mean) / (std + 1e-8)
        elif self.preprocess == 'center':
            mean = np.mean(X, axis=0)
            return X - mean
        else:
            return X

    def get_features(self, indices=False):
        """Get the selected feature indices or a mask."""
        check_is_fitted(self, 'selected_features_')

        if indices:
            return self.selected_features_

        mask = np.zeros(self.scores_.shape[0], dtype=bool)
        mask[self.selected_features_] = True
        return mask

    def _get_support_mask(self):
        """
        Required by SelectorMixin. Returns a mask for the selected features.
        """
        check_is_fitted(self, 'selected_features_')
        mask = np.zeros(self.scores_.shape[0], dtype=bool)
        mask[self.selected_features_] = True
        return mask

