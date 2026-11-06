import numpy as np
import pytest


@pytest.fixture(autouse=True, scope="session")
def raise_numpy_warnings():
    old_settings = np.seterr(all="raise")
    try:
        yield None
    finally:
        np.seterr(**old_settings)
