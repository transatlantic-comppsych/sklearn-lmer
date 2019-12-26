"""
===========================
Basic sklmer example
===========================

This example demonstrates how to use sklmer to fit mixed effects models
and how to predict from those models once they are fit. It also demonstrates
passing control options to LME4 to change the optimizer used.
"""

#%%
# Imports
# -------

import numpy as np
from matplotlib import pyplot as plt
from sklmer import LmerRegressor
import pandas as pd
import os
from pymer4.utils import get_resource_path

#%%
# Load and prepare data
# ---------------------
# split out training and test data
df = pd.read_csv(os.path.join(get_resource_path(), "sample_data.csv"))
df = df.reset_index().rename(columns={'index':'orig_index'})
test = df.groupby('Group').apply(lambda x: x.sample(frac=0.2)).reset_index(drop=True)
train = df.loc[~df.orig_index.isin(test.orig_index), :]

#%%
# Fit and predict with some different estimator options
# -----------------------------------------------------
# First with defaults.

df_estimator = LmerRegressor("DV ~ IV2 + (IV2|Group)", X_cols=df.columns)
df_estimator.fit(data=train)
df_preds = df_estimator.predict(test)

#%%
# Then use random effects information in the prediction.

rfx_preds = df_estimator.predict(test, use_rfx=True)


#%%
# Now add in optimizer tweaks. Note that since we set predict_rfx when creating
# the estimator, we don't need to when we call predict.

nm_estimator = LmerRegressor(
    "DV ~ IV2 + (IV2|Group)",
    X_cols=df.columns,
    predict_rfx=True,
    fit_kwargs={
        "control": "optimizer='Nelder_Mead', optCtrl = list(FtolAbs=1e-8, XtolRel=1e-8)"
    },
)
nm_estimator.fit(data=train)
nm_preds = nm_estimator.predict(test)

#%%
# Finally, we'll plot the results
# -----------------------------------------------------
# Using the random effects in the prediction helps.
# The optimizer changes don't make as big of a differences.

fig, axs = plt.subplots(1, 4, figsize=(12, 3))
ax = axs[0]
_ = ax.plot(test.DV.values, df_preds, "o")
_ = ax.set_title("No rfx prediction, standard optimizer")
_ = ax.set_xlabel("True DV")
_ = ax.set_ylabel("Predicted DV")

ax = axs[1]
_ = ax.plot(test.DV.values, rfx_preds, "o")
_ = ax.set_title("Rfx prediction, standard optimizer")
_ = ax.set_xlabel("True DV")
_ = ax.set_ylabel("Predicted DV")

ax = axs[2]
_ = ax.plot(test.DV.values, nm_preds, "o")
_ = ax.set_title("Rfx prediction, tweaked optimizer")
_ = ax.set_xlabel("True DV")
_ = ax.set_ylabel("Predicted DV")
_ = fig.tight_layout()

ax = axs[3]
_ = ax.plot(
    rfx_preds - test.DV.values, nm_preds - test.DV.values, "o"
)
_ = ax.plot((-45, 45), (-45, 45))
_ = ax.set_title("Optimizer difference")
_ = ax.set_xlabel("DV - Std. Opt Preds")
_ = ax.set_ylabel("DV - NM Opt Preds")
_ = ax.set_xticks(ax.get_yticks())
_ = ax.set_xlim((-45, 45))
_ = ax.set_ylim((-45, 45))
_ = fig.tight_layout()
fig.show()
