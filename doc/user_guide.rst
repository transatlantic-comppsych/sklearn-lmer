.. title:: User guide : contents

.. _user_guide:

==================================================
User guide: create your own scikit-learn estimator
==================================================

Estimator
---------

The central piece of transformer, regressor, and classifier is
:class:`sklearn.base.BaseEstimator`. All estimators in scikit-learn are derived
from this class. In more details, this base class enables to set and get
parameters of the estimator. It can be imported as::

    >>> from sklearn.base import BaseEstimator

Once imported, you can create a class which inherate from this base class::

    >>> class MyOwnEstimator(BaseEstimator):
    ...     pass


Predictor
---------

Regressor
~~~~~~~~~

Similarly, regressors are scikit-learn estimators which implement a ``predict``
method. The use case is the following:

* at ``fit``, some parameters can be learned from ``X`` and ``y``;
* at ``predict``, predictions will be computed using ``X`` using the parameters
  learned during ``fit``.

In addition, scikit-learn provides a mixin_, i.e.
:class:`sklearn.base.RegressorMixin`, which implements the ``score`` method
which computes the :math:`R^2` score of the predictions.

One can import the mixin as::

    >>> from sklearn.base import RegressorMixin

Therefore, we create a regressor, :class:`MyOwnRegressor` which inherits from
both :class:`sklearn.base.BaseEstimator` and
:class:`sklearn.base.RegressorMixin`. The method ``fit`` gets ``X`` and ``y``
as input and should return ``self``. It should implement the ``predict``
function which should output the predictions of your regressor::

    >>> import numpy as np
    >>> class MyOwnRegressor(BaseEstimator, RegressorMixin):
    ...     def fit(self, X, y):
    ...         return self
    ...     def predict(self, X):
    ...         return np.mean(X, axis=1)

We illustrate that this regressor is working within a scikit-learn pipeline::

    >>> from sklearn.datasets import load_diabetes
    >>> X, y = load_diabetes(return_X_y=True)
    >>> pipe = make_pipeline(MyOwnTransformer(), MyOwnRegressor())
    >>> pipe.fit(X, y)  # doctest: +ELLIPSIS
    Pipeline(...)
    >>> pipe.predict(X)  # doctest: +ELLIPSIS
    array([...])

Since we inherit from the :class:`sklearn.base.RegressorMixin`, we can call
the ``score`` method which will return the :math:`R^2` score::

    >>> pipe.score(X, y)  # doctest: +ELLIPSIS
    -3.9...

