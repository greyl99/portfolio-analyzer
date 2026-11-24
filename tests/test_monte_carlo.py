import numpy as np
from monte_carlo import simulate_prices

def test_same_seed():
    result1=simulate_prices(initial_price=100,
                            mu=0.001,
                            sigma=0.02,
                            days=10,
                            simulations=100,
                            seed=43
                            )
    result2=simulate_prices(initial_price=100,
                            mu=0.001,
                            sigma=0.02,
                            days=10,
                            simulations=100,
                            seed=43
                            )
    assert np.array_equal(result1,result2)

def test_different_seed():
    result1=simulate_prices(initial_price=100,
                            mu=0.001,
                            sigma=0.02,
                            days=10,
                            simulations=100,
                            seed=43
                            )
    result2=simulate_prices(initial_price=100,
                            mu=0.001,
                            sigma=0.02,
                            days=10,
                            simulations=100,
                            seed=123
                            )
    assert not np.array_equal(result1,result2)
    