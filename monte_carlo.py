import numpy as np
import matplotlib.pyplot as plt
def check_price(initial_price):
    if not np.isfinite(initial_price) or initial_price <= 0:
        raise ValueError("Initial price must be finite and positive")
    return initial_price

def check_sigma(sigma):
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Sigma must be finite and non-negative")
    return sigma
def check_days(days):
    if not isinstance(days, (int, np.integer)) or isinstance(days, (bool, np.bool_)) or days <= 0:
        raise ValueError("Days must be a positive integer")
    return days
def check_simulations(simulations):
    if not isinstance(simulations, (int, np.integer)) or isinstance(simulations, (bool, np.bool_)) or simulations <= 0:
            raise ValueError("Simulations must be a positive integer")
    return simulations

def simulate_prices(initial_price,mu,sigma,days,simulations,seed):
    check_days(days)
    check_price(initial_price)
    check_sigma(sigma)
    check_simulations(simulations)
    if not np.isfinite(mu):
        raise ValueError("Mu must be finite")
    rng=np.random.default_rng(seed)
    returns=rng.normal(loc=mu,scale=sigma,size=(days,simulations))
    prices=initial_price*np.cumprod(1+returns,axis=0)
    return prices
def simulate_correlated_returns(mu,cov_matrix,days,simulations,seed):
    check_days(days)
    check_simulations(simulations)
    rng=np.random.default_rng(seed)
    correlated_returns=rng.multivariate_normal(mu,cov_matrix,(days,simulations))
    return correlated_returns
def simulated_portfolio_values(correlated_returns,aligned_weights,initial_capital):
    portfolio_simulated_returns=correlated_returns@aligned_weights
    portfolio_values=initial_capital*np.cumprod(1+portfolio_simulated_returns,axis=0)
    return portfolio_values
    
def calculate_risk(prices,initial_price):
    final_prices=prices[-1,:]
    average_final_price=np.mean(final_prices)
    total_return=(final_prices/initial_price)-1
    average_total_return=np.mean(total_return)
    loss_count=np.sum(final_prices<initial_price)
    loss_probability=loss_count/len(final_prices)
    worst_final_price=np.min(final_prices)
    price_5th=np.percentile(final_prices,5)
    var_95=initial_price-price_5th
    var_95_pct=var_95/initial_price
    result={"Average Final Price":average_final_price,
            "Average Total Return":average_total_return,
            "Loss Probability":loss_probability,
            "Worst Final Price":worst_final_price,
            "95% VaR ($)":var_95,
            "95% VaR (%)":var_95_pct
            }
    return result
def plot_simulation(prices):
    plt.plot(prices, alpha=0.15, linewidth=0.8)
    plt.title("Simulated Stock Prices")
    plt.xlabel("Trading Day")
    plt.ylabel("Stock Price ($)")
    plt.show()

def run_simulation(initial_price, mu, sigma, days, simulations, seed):
    prices = simulate_prices(initial_price, mu, sigma, days, simulations, seed)
    risk = calculate_risk(prices, initial_price)

    for key, value in risk.items():
        print(key, ":", value)

    plot_simulation(prices)
