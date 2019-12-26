"""
This module contains sklearn wrappers for pymer4
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from pymer4 import Lmer
import inspect
import pandas as pd


class LmerRegressor(BaseEstimator, RegressorMixin):
    """ A regressor that wraps pymer4's lme4 implementation.

    This regressor requires a formula be defined in LME4's style, see pymer4's cheatsheet: http://eshinjolly.com/pymer4/rfx_cheatsheet.html

    Parameters
    ----------
    formula : str 
        Lmer formatted formula string.
    X_cols : list
        List of the names of the X columns.
    predict_rfx: bool, default='False'
        Whether or not the predict method should use random effects in the prediction.
    family: str, default='gausian'
        What family of distributions to use for the link function for the generalized model.
    fit_kwargs: dict, defalut='{}'
        Dictionary of options to pass to lmer fit. See http://eshinjolly.com/pymer4/api.html
    """

    def __init__(
        self, formula, X_cols, predict_rfx=False, family="gaussian", fit_kwargs={}
    ):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def _make_data(self, X=None, y=None, data=None, x_only=False):

        if data is None:
            if x_only:
                if X is None:
                    raise ValueError("If you don't pass data you must pass X")

                # Make a dataframe out of X
                data = pd.DataFrame(X, columns=self.X_cols)
            else:
                if X is None or y is None:
                    raise ValueError("If you don't pass data you must pass X and y")

                # Check that X and y have correct shape
                X, y = check_X_y(X, y)

                # Make a dataframe out of X and y
                data = pd.DataFrame(X, columns=self.X_cols)
                data[self._response_name] = y
        else:
            data = data.copy()
        return data

    def fit(self, X=None, y=None, data=None):
        """ Fit the specified mixed effects model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        data: pandas.DataFrame
            Data can also be passed to fit as a dataframe.

        Returns
        -------
        self : object
            Returns self.
        """

        self._response_name = self.formula.split("~")[0].strip()

        self.data_ = self._make_data(X=X, y=y, data=data)
        self.model = Lmer(self.formula, data=self.data_, family=self.family)
        self.model.fit(summarize=False, verbose=False, **self.fit_kwargs)
        if self.model.warnings is not None:
            if ("converge" in self.model.warnings) | np.any(
                ["converge" in mw for mw in self.model.warnings]
            ):
                self.converged = False
            else:
                self.converged = True
        else:
            self.converged = True
        self.coef_ = self.model.coefs.iloc[1:, 0].values
        self.intercept_ = self.model.coefs.iloc[0, 0]
        return self

    def predict(self, X=None, data=None, **kwargs):
        """ Predict based on the fitted mixed effects model.

        Will use random effects if the estimators predict_rfx attribute is true.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        data: pandas.DataFrame
            Data can also be passed as a dataframe.
        **kwargs:
            Passed through to the pymer4.Lmer.predict method

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns predicted values
        """
        check_is_fitted(self, ["data_", "converged"])

        data = self._make_data(X, data=data, x_only=True)
        try:
            use_rfx = kwargs["use_rfx"]
            kwargs.pop("use_rfx")
        except KeyError:
            use_rfx = self.predict_rfx
        return self.model.predict(data, use_rfx, **kwargs)
