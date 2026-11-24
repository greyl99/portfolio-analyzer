import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from monte_carlo import simulate_prices,calculate_risk,simulate_correlated_returns,simulated_portfolio_values

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def validate_weights(weights,asset_columns):
    if not isinstance(weights,pd.Series):
        raise ValueError("Error:weights must be Pandas series!")
    if pd.Index(asset_columns).has_duplicates:
        raise ValueError("Duplicate asset columns detected!")
    if weights.index.has_duplicates:
        raise ValueError("Error:weight's index are duplicates!")
    stock_names=set(weights.index)
    data_names = set(asset_columns)
    if stock_names!=data_names:
        raise ValueError("Error:Stock names do not match!")
    
    aligned_weights=weights.reindex(asset_columns)
    if not np.isfinite(aligned_weights).all():
            raise ValueError("weights contains invalid values!")
    if aligned_weights.lt(0).any():
        raise ValueError("Error:weights con not be negtive!")
    if not np.isclose(aligned_weights.sum(),1.0):
        raise ValueError("Error: Portfolio weights must sum to 1!")
    return aligned_weights
def load_prices(filepath):
    df=pd.read_csv(filepath)
    if len(df) < 2:
        raise ValueError("At least two trading dates are required.")
    df["Date"]=pd.to_datetime(df["Date"])
    df=df.set_index("Date")
    df=df.sort_index()
    if df.index.duplicated().any():
        raise ValueError("Error:Duplicate dates detected!")
   
    if not np.isfinite(df.to_numpy()).all():
       raise ValueError("Prices contain NAN or infinite values!")
    
    if(df <= 0).any().any():
       raise ValueError("Error: Invalid stock prices detected!") 
    
    return df
def calculate_returns(prices):
    daily_return=prices.pct_change()
    daily_return=daily_return.dropna()
    cumulative_return=(1+daily_return).cumprod()-1
    total_return=(prices.iloc[-1]/prices.iloc[0])-1
    if not np.allclose(cumulative_return.iloc[-1],total_return):
        raise ValueError("Error:Cumulative returns do not match!")
    return daily_return,cumulative_return,total_return
def calculate_portfolio(daily_return, weights, initial_capital):
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive")
    aligned_weights=validate_weights(weights,daily_return.columns)
    portfolio_daily_return=daily_return@aligned_weights
    portfolio_cumulative_return=(1+portfolio_daily_return).cumprod()-1
    portfolio_growth_factor=portfolio_cumulative_return+1
    portfolio_value=initial_capital*portfolio_growth_factor
    if (portfolio_value <= 0).any():
        raise ValueError("Error: Invalid portfolio value!")
    
    portfolio_total_return = portfolio_value.iloc[-1] / initial_capital - 1
    if  not np.isclose(portfolio_total_return,portfolio_cumulative_return.iloc[-1]):
       raise ValueError("Error:Portfolio total return do not match")
    return portfolio_daily_return, portfolio_cumulative_return, portfolio_value
def save_portfolio_report(portfolio_value, initial_capital, initial_date, output_dir):
    initial_value = pd.Series(
    [initial_capital],
    index=[initial_date])
    portfolio_equity=pd.concat([initial_value,portfolio_value])
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_equity.index, portfolio_equity.values)
    plt.title("Portfolio Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value($)")
    portfolio_report = portfolio_equity.to_frame(name="Portfolio_Value")
    portfolio_report.index.name = "Date"
    portfolio_report.to_csv(output_dir/"portfolio_v5_equity.csv")
    
    plt.savefig(output_dir / "portfolio_v5_equity.png", dpi=150)
    plt.close()
    return portfolio_equity
def estimate_asset_parameters(daily_return):
    mu=daily_return.mean()
    sigma=daily_return.std()
    asset_parameters=pd.DataFrame({"mu":mu,"sigma":sigma})
    return asset_parameters
def estimate_portfolio_parameters(daily_return,weights):
    aligned_weights=validate_weights(weights,daily_return.columns)
    asset_parameters=estimate_asset_parameters(daily_return)
    portfolio_mu=aligned_weights@asset_parameters["mu"]
    cov_matrix=daily_return.cov()
    portfolio_variance=aligned_weights@cov_matrix@aligned_weights
    portfolio_sigma=np.sqrt(portfolio_variance)
    return portfolio_mu,portfolio_sigma
def prepare_multi_asset_parameters(daily_return):
    asset_parameters=estimate_asset_parameters(daily_return)
    mu=asset_parameters["mu"]
    cov_matrix=daily_return.cov()
    return mu,cov_matrix

def calculate_asset_contributions(daily_return,weights):
    weights=validate_weights(weights,daily_return.columns)
    weighted_return=weights*daily_return
    contributions=weighted_return.sum(axis=0)
    return contributions

def main(weights,initial_capital,output_dir,filepath):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)       
    prices = load_prices(filepath)
    daily_return, cumulative_return, total_return = calculate_returns(prices)
    portfolio_daily_return, portfolio_cumulative_return, portfolio_value = calculate_portfolio(daily_return, weights, initial_capital)
    save_portfolio_report(
    portfolio_value,
    initial_capital,
    prices.index[0],
    output_dir
)
    portfolio_mu, portfolio_sigma = estimate_portfolio_parameters(daily_return, weights)
    portfolio_total_return = portfolio_value.iloc[-1] / initial_capital - 1
    print(f"Portfolio Total Return: {portfolio_total_return:.2%}")
    print(f"Final Portfolio Value: ${portfolio_value.iloc[-1]:,.2f}")
    simulation_parameters={"initial_price":initial_capital,
                           "mu":portfolio_mu,
                           "sigma":portfolio_sigma,
                           "days":252,
                           "simulations":10000,
                           "seed":42
                           }
    simulated_prices=simulate_prices(**simulation_parameters)
    expected_shape=(simulation_parameters["days"],
                    simulation_parameters["simulations"]

    )
    if simulated_prices.shape!=expected_shape:
        raise ValueError("Unexpected simulation shape!")
    if not np.isfinite(simulated_prices).all():
        raise ValueError("Simulation contains invalid values!")
    risk_report=calculate_risk(simulated_prices,initial_capital)
    risk_df=pd.DataFrame(risk_report.items(),columns=["Metric","Value"])
    risk_df.to_csv(output_dir/"monte_carlo_risk.csv",index=False)
    mu,cov_matrix=prepare_multi_asset_parameters(daily_return)
    if not mu.index.equals(cov_matrix.index):
        raise ValueError("Asset order mismatch!")
    if not cov_matrix.index.equals(cov_matrix.columns):
        raise ValueError("Covariance matrix labels mismatch!")
    correlated_returns=simulate_correlated_returns(mu,cov_matrix,252,10000,42)

    aligned_weights = validate_weights(
    weights,
    daily_return.columns
).to_numpy()
    portfolio_simulations=simulated_portfolio_values(correlated_returns,aligned_weights,initial_capital)
    expected_shape = (simulation_parameters["days"],
                      simulation_parameters["simulations"]
    )

    if portfolio_simulations.shape != expected_shape:
        raise ValueError("Unexpected portfolio simulation shape!")

    if not np.isfinite(portfolio_simulations).all():
        raise ValueError("Portfolio simulation contains invalid values!")

    if (portfolio_simulations<=0).any():
        raise ValueError("Non-positive portfolio simulation detected!")



    multi_asset_risk=calculate_risk(portfolio_simulations,initial_capital)
    multi_asset_df=pd.DataFrame(
        multi_asset_risk.items(),
        columns=["Metric","Value"])
    multi_asset_df.to_csv(output_dir/"multi_asset_risk.csv",index=False)
    comparison=pd.merge(risk_df,
                        multi_asset_df,
                        on="Metric",
                        suffixes=("_Single","_Multi")
                        )
    comparison["Difference"]=comparison["Value_Multi"]-comparison["Value_Single"]
    
    comparison.to_csv(output_dir/"model_comparison.csv",index=False)
    
if __name__== "__main__":
    main(initial_capital = 10000,
            weights = pd.Series({
                "NVDA": 0.3,
                "MSFT": 0.3,
                "AAPL": 0.4
                })
                ,output_dir=OUTPUT_DIR,
                filepath = BASE_DIR / "data" / "market_prices_1y.csv"
                )
