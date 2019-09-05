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

estimator = LmerRegressor('DV ~ IV2 + (IV2|Group)', X_cols=df.columns)
estimator.fit(data=df)
plt.plot(estimator.predict(df.IV2))
plt.show()
