import pytest
import numpy as np

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose
from pymer4.utils import get_resource_path
import pandas as pd
import os

from sklmer import LmerRegressor


@pytest.fixture
def data():
    return pd.read_csv(os.path.join(get_resource_path(), "sample_data.csv"))


def test_LmerRegressor(data):
    lreg = LmerRegressor("DV ~ IV2 + (IV2|Group)", X_cols=data.columns)
    lreg.fit(data=data)
    expected_coef = np.array([0.6821139])
    assert_allclose(lreg.coef_, expected_coef)

    lreg = LmerRegressor("DV ~ IV2 + (IV2|Group)", X_cols=data.columns)
    lreg.fit(X=data.values, y=data.DV.values)
    assert_allclose(lreg.coef_, expected_coef)

    lreg.predict(data=data)

    expected_norfx_score = 0.5035556907587277
    assert np.isclose(lreg.score(X=data.values, y=data.DV.values), expected_norfx_score)

    lreg.predict_rfx = True
    expected_rfx_score = 0.8854319828487308
    assert np.isclose(lreg.score(X=data.values, y=data.DV.values), expected_rfx_score)
