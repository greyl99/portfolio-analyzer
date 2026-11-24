import yfinance as yf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

tickers = ["AAPL", "MSFT", "NVDA"]

df = yf.download(
    tickers,
    period="1y",
    auto_adjust=True,
    progress=False
)["Close"]

df.index.name = "Date"

print("Data downloaded successfully!")

assert not df.empty
assert df.isna().sum().sum()==0
assert not df.index.has_duplicates
assert (df > 0).all().all()
assert df.index.is_monotonic_increasing
df.to_csv(DATA_DIR / "market_prices_1y.csv")