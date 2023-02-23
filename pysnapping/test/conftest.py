import pytest
import numpy as np


@pytest.fixture(autouse=True, scope="session")
def raise_numpy_warnings():
    old_settings = np.seterr(all="raise")
    try:
        yield None
    finally:
        np.seterr(**old_settings)
