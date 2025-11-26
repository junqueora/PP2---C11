import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima

df = pd.read_csv("accidents_2017_to_2023_english.csv")
df["date"] = pd.to_datetime(df["inverse_data"])

monthly = df.groupby(pd.Grouper(key="date", freq="ME")).size().rename("accidents")
monthly = monthly.asfreq("ME").fillna(0)

plt.figure(figsize=(12,5))
plt.plot(monthly, linewidth=2, color="blue")
plt.title("Acidentes Mensais (2017–2023)")
plt.xlabel("Data")
plt.ylabel("Acidentes")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

decomp = seasonal_decompose(monthly, model="additive", period=12)
fig = decomp.plot()
fig.set_size_inches(12, 8)
plt.show()

sarimax_model = auto_arima(
    monthly,
    seasonal=True,
    m=12,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

n_periods = 12

future_index = pd.date_range(
    start=monthly.index[-1] + pd.offsets.MonthEnd(1),
    periods=n_periods,
    freq="ME"
)

forecast = sarimax_model.predict(n_periods=n_periods)
forecast = pd.Series(forecast, index=future_index)

plt.figure(figsize=(12,5))
plt.plot(monthly, label="Histórico", linewidth=2, color="blue")
plt.plot(forecast, label="Previsão SARIMAX", linewidth=2, color="red")
plt.title("Previsão Mensal")
plt.xlabel("Data")
plt.ylabel("Acidentes")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()
