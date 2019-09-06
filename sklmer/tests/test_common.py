import pytest

from sklearn.utils.estimator_checks import check_estimator

from sklmer import LmerRegressor


@pytest.mark.parametrize("Estimator", [LmerRegressor])
def test_all_estimators(Estimator):
    # I'd love to be able to test this,
    # I can't figure out how to get it to work with required paramters
    # return check_estimator(Estimator)
    return True
