"""
===========================
Plotting Lmer Regressor
===========================

An example plot of :class:`sklmer.LmerRegressor`
"""
import numpy as np
from matplotlib import pyplot as plt
from sklmer import LmerRegressor
import pandas as pd
import os
from pymer4.utils import get_resource_path

df = pd.read_csv(os.path.join(get_resource_path(),'sample_data.csv'))

fig, axs = plt.subplots(1, 4, figsize=(12,3))
ax = axs[0]
estimator = LmerRegressor('DV ~ IV2 + (IV2|Group)', X_cols=df.columns)
estimator.fit(data=df)
_ = ax.plot(df.DV.values, estimator.predict(df),  'o')
_ = ax.set_title("No rfx prediction, standard optimizer")
_ = ax.set_xlabel('True DV')
_ = ax.set_ylabel('Predicted DV')

ax = axs[1]
estimator = LmerRegressor('DV ~ IV2 + (IV2|Group)', X_cols=df.columns, predict_rfx=True)
estimator.fit(data=df)
_ = ax.plot(df.DV.values, estimator.predict(df),  'o')
_ = ax.set_title("Rfx prediction, standard optimizer")
_ = ax.set_xlabel('True DV')
_ = ax.set_ylabel('Predicted DV')

ax = axs[2]
nm_estimator = LmerRegressor('DV ~ IV2 + (IV2|Group)', X_cols=df.columns, predict_rfx=True,
    fit_kwargs={
            "control": "optimizer='Nelder_Mead', optCtrl = list(FtolAbs=1e-8, XtolRel=1e-8)"})
nm_estimator.fit(data=df)
_ = ax.plot(df.DV.values, nm_estimator.predict(df),  'o')
_ = ax.set_title("Rfx prediction, tweaked optimizer")
_ = ax.set_xlabel('True DV')
_ = ax.set_ylabel('Predicted DV')
_ = fig.tight_layout()

ax = axs[3]
_ = ax.plot(estimator.predict(df) - df.DV.values, nm_estimator.predict(df) - df.DV.values,  'o')
_ = ax.plot((-60, 60), (-60,60))
_ = ax.set_title("Optimizer difference")
_ = ax.set_xlabel('DV - Std. Opt Preds')
_ = ax.set_ylabel('DV - NM Opt Preds')
_ = ax.set_xticks(ax.get_yticks())
_ = ax.set_xlim((-65,65))
_ = ax.set_ylim((-65,65))
_ = fig.tight_layout()
fig.show()