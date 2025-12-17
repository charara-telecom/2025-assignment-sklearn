"""Assignment utilities: a KNN estimator and a monthly splitter.

This module implements a simple K-Nearest Neighbors classifier for
classification tasks and a time-based cross-validator that splits data by
consecutive months. It is used for the assignment tests.

The KNearestNeighbors class implements `fit`, `predict`, and `score` and
uses `validate_data` and `check_is_fitted` from scikit-learn for input
validation. The MonthlySplit class yields train/test splits where the
training set is samples from month i and the test set is samples from
month i+1 for each consecutive month pair.

Use `pytest test_sklearn_questions.py` at the repository root to run the
tests. For developer guidance on custom estimators, see:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.multiclass import check_classification_targets


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """K-Nearest Neighbors classifier (majority vote) using Euclidean distance.

    Parameters
    ----------
    n_neighbors : int, default=1
        Number of neighbors to use for prediction.
    """

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the classifier by storing the training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : KNearestNeighbors
            Fitted estimator.
        """
        if int(self.n_neighbors) < 1:
            raise ValueError("n_neighbors must be >= 1.")
        check_classification_targets(y)

        X, y = validate_data(self, X, y, reset=True)
        y = np.asarray(y)

        self.X_ = X
        self.y_ = y
        self.classes_, self.y_encoded_ = np.unique(self.y_, return_inverse=True)

        if int(self.n_neighbors) > self.X_.shape[0]:
            n_samples = self.X_.shape[0]
            raise ValueError(
                "Expected n_neighbors <= n_samples, but "
                f"n_samples = {n_samples}, n_neighbors = "
                f"{int(self.n_neighbors)}."
            )

        return self

    def predict(self, X):
        """Predict class labels for X.

        Parameters
        ----------
        X : ndarray of shape (n_test_samples, n_features)
            Test data.

        Returns
        -------
        y : ndarray of shape (n_test_samples,)
            Predicted labels.
        """
        check_is_fitted(self, attributes=["X_", "y_", "classes_", "y_encoded_"])
        X = validate_data(self, X, reset=False)

        k = int(self.n_neighbors)
        dists = pairwise_distances(X, self.X_, metric="euclidean")

        neigh_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        neigh_codes = self.y_encoded_[neigh_idx]

        if k == 1:
            return self.classes_[neigh_codes.ravel()]
        n_samples = self.X_.shape[0]
        k = int(self.n_neighbors)

        if k > n_samples:
            raise ValueError(
                "Expected n_neighbors <= n_samples, but "
                f"n_samples = {n_samples}, n_neighbors = {k}."
            )

        n_classes = self.classes_.shape[0]
        counts = np.zeros((X.shape[0], n_classes), dtype=int)
        for i in range(X.shape[0]):
            counts[i] = np.bincount(neigh_codes[i], minlength=n_classes)

        winners = counts.argmax(axis=1)
        return self.classes_[winners]

    def score(self, X, y):
        """Return accuracy on (X, y).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to score on.
        y : ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Mean accuracy.
        """
        check_is_fitted(self, attributes=["X_", "y_", "classes_", "y_encoded_"])
        X, y = validate_data(self, X, y, reset=False)
        y = np.asarray(y)

        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


class MonthlySplit(BaseCrossValidator):
    """Cross-validator that splits data by consecutive months.

    For each pair of successive months, yields:
      - train indices = samples in month i
      - test indices  = samples in month i+1

    Parameters
    ----------
    time_col : str, default='index'
        Column name containing datetimes, or 'index' to use the DataFrame index.
    """

    def __init__(self, time_col="index"):  # noqa: D107
        self.time_col = time_col

    def _get_datetime_index(self, X):
        """Extract a DatetimeIndex from X according to `time_col`."""
        if self.time_col == "index":
            if not hasattr(X, "index"):
                raise ValueError("X must have an index when time_col='index'.")
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError("X.index must be a pandas DatetimeIndex.")
            return X.index

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame when time_col != 'index'.")
        if self.time_col not in X.columns:
            raise ValueError(
                "Column '{}' not found in X.".format(self.time_col)
            )
        if not pd.api.types.is_datetime64_any_dtype(X[self.time_col]):
            raise ValueError(
                "Column '{}' must be datetime dtype.".format(self.time_col)
            )

        return pd.DatetimeIndex(X[self.time_col])

    def get_n_splits(self, X, y=None, groups=None):
        """Return number of month-to-next-month splits."""
        dt_index = self._get_datetime_index(X)
        months = dt_index.to_period("M")
        unique_months = months.unique().sort_values()
        return max(int(len(unique_months) - 1), 0)

    def split(self, X, y=None, groups=None):
        """Yield (train_idx, test_idx) for each consecutive month pair."""
        dt_index = self._get_datetime_index(X)
        months = dt_index.to_period("M")
        unique_months = months.unique().sort_values()

        for m_train, m_test in zip(unique_months[:-1], unique_months[1:]):
            idx_train = np.flatnonzero(months == m_train)
            idx_test = np.flatnonzero(months == m_test)
            yield idx_train, idx_test
