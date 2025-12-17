"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:
- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.
Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `validate_data, check_is_fitted` functions
imported in this file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.
Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.
We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.
Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.
Hints
-----
- You can use the function:
from sklearn.metrics.pairwise import pairwise_distances
to compute distances between 2 sets of samples.
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
