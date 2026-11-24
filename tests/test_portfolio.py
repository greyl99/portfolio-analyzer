import numpy as np
import pandas as pd
from portfolio_v5 import validate_weights
import pytest
def test_invalid_weight_sum():
    weights = pd.Series({
        "AAPL": 0.5,
        "MSFT": 0.5,
        "NVDA": 0.5
    })

    asset_columns = pd.Index(["AAPL", "MSFT", "NVDA"])

    with pytest.raises(ValueError):
        validate_weights(weights, asset_columns)
def test_negative_weights():
    weights = pd.Series({
        "AAPL": 0.5,
        "MSFT": -0.2,
        "NVDA": 0.7
    })

    asset_columns = pd.Index(["AAPL", "MSFT", "NVDA"])

    with pytest.raises(ValueError):
        validate_weights(weights,asset_columns)


def test_duplicate_assets():
    weights = pd.Series(
        [0.4, 0.3, 0.3],
        index=["AAPL", "AAPL", "NVDA"]
    )

    asset_columns = pd.Index(["AAPL", "MSFT", "NVDA"])

    with pytest.raises(ValueError):
        validate_weights(weights,asset_columns)
        
def test_valid_weights():
    weights = pd.Series({
        "NVDA": 0.3,
        "MSFT": 0.3,
        "AAPL": 0.4
    })

    asset_columns = pd.Index(["AAPL", "MSFT", "NVDA"])

    aligned_weights = validate_weights(weights, asset_columns)

    assert np.isclose(aligned_weights.sum(), 1.0)
    assert aligned_weights.index.equals(asset_columns)
    assert np.isclose(aligned_weights["AAPL"], 0.4)