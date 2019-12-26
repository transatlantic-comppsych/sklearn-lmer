.. title:: User guide : contents

.. _user_guide:

========================================================
User guide: running mixed effects regressions in sklearn
========================================================

Linear mixed effects regressions
--------------------------------

Linear mixed effects regressions are great, but if you're here,
you probably already agree. You can find more infomration about
them elsewhere. 

This is a simple package to wrap
the convienience of pymer4's lme4 wrapping in a *mostly* sklearn
compatible regressor class.

Installation
------------

Mixing r and python used to be a bit more fraught, but rpy2 and conda
seem to be working together better these days.
To install first get a conda environment with the dependencies::

   >>> conda create -n sklmer -c conda-forge numpy scipy rpy2 r-lme4 r-lmertest r-lsmeans tzlocal

Then pip install sklearn-lmer::

   >>> pip install sklearn-lmer

Usage
-----
It can be imported as::

    >>> from sklmer import LmerRegressor

Now the *mostly* part of that compatiblity is that init does have two required paramters:
a formula and the names of the columns holding independent variables and grouping variables
(I've called this parameter ``X_cols`` even though it is more than just X). When I use this I've got my data in a dataframe and just pass ``dataframe.columns`` with ``X_cols`` like so ::

    >>> df = pd.read_csv(os.path.join(get_resource_path(),'sample_data.csv'))
    >>> lreg = LmerRegressor('DV ~ IV2 + (IV2|Group)', X_cols=df.columns)

If you want the best compatibility with sklearn it probably makes sense to split
out the dataframe into X, y, and group variables, though since you've defined a formula
it's ok if the y and group columns are in X ::

   >>> X = df.values
   >>> y = df.DV.values
   >>> groups = df.Group.values

Once you've done that, it seems to work fine with other sklearn tools, like ``cross_val_score`` ::

   >>> logo = LeaveOneGroupOut()
   >>> cross_val_score(lreg, X=X, y=y, cv=logo.split(X, groups=groups), scoring='neg_mean_squared_error')

