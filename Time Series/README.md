**Time Series Models – Study Guide**
**1. Naive Models**
Naive Forecast:
Assumes that the future value is equal to the last observed value.
Formula: ŷₜ₊₁ = yₜ
Best used as a baseline or for slow-changing series.
**2. Moving Average Models**
a. Simple Moving Average (SMA):
Takes the average of the last n observations to smooth out short-term noise.
Formula: SMAₜ = (1/n) ∑ (from i=0 to n-1) yₜ₋ᵢ

b. Weighted Moving Average (WMA):
Recent observations are given more importance by assigning weights.
**3. Exponential Smoothing Models**
a. Simple Exponential Smoothing (SES):
Applies to data with no trend or seasonality.
Formula: ŷₜ₊₁ = αyₜ + (1 - α)ŷₜ

b. Holt’s Linear Trend Model:
Extends SES to account for trend.

c. Holt-Winters Model:
Accounts for both trend and seasonality.
**4. ARIMA (AutoRegressive Integrated Moving Average)**
Combines AR (p), I (d), and MA (q) components.
Used for univariate, stationary time series.
Example: ARIMA(1,1,1) – One autoregressive, one differencing, one moving average term.
**5. SARIMA (Seasonal ARIMA)**
Extends ARIMA to handle seasonality.
Format: SARIMA(p,d,q)(P,D,Q)[m], where m = seasonal period (e.g., 12 for monthly data).
**6. Machine Learning-Based Models**
a. Random Forest / XGBoost:
Work with engineered features such as lags and rolling stats.

b. LSTM (Long Short-Term Memory):
Deep learning model ideal for capturing long-term dependencies in sequential data.
**7. Prophet (by Facebook)**
Designed for business forecasting with trend, seasonality, and holiday effects.
Easy to use, handles outliers and missing data well.
Formula: Forecast = Trend + Seasonality + Holiday Effects
Summary Comparison Table
Model		Trend	Seasonality	Complexity	Suitable For

Naive		❌	❌	Very Low	Baseline

SMA/WMA	❌	❌	Low	Smoothing

SES		❌	❌	Low	No Trend/Seasonality

Holt		✅	❌	Medium	Trend Only

Holt-Winters	✅	✅	Medium	Trend + Seasonality

ARIMA		✅	❌	High	Stationary Series

SARIMA		✅	✅	Higher	Seasonal Series

Prophet		✅	✅	Medium	Business/Marketing

LSTM/XGBoost	✅	✅	Very High	Complex Patterns

---
Time Series Business Case Studies
1. Forecasting Monthly Sales for a Retail Chain
**Industry:** Fashion Retail
**Problem Statement:**
Improve inventory and staffing decisions by forecasting monthly store sales.
**Data Used:**
- Monthly sales for 5 years
- Promotions and holiday data
**Analysis & Modeling Steps:**
- Plotted trend and seasonality
- ADF test → Non-stationary
- Applied differencing
- Chose SARIMA based on evaluation metrics
**Model Chosen:**
SARIMA(1,1,1)(1,1,1)[12]
**Outcome & Impact:**
- Reduced overstock by 18%
- Prevented $120K in lost sales
- Improved staffing efficiency by 22%
**Time Series Principles Applied:**
- Trend
- Seasonality
- Stationarity
- Autocorrelation
2. Predicting Energy Demand for Utility Company
**Industry:** Energy & Utilities
**Problem Statement:**
Forecast hourly electricity demand to optimize grid operations and prevent blackouts.
**Data Used:**
- Hourly usage data
- Weather and temperature data
**Analysis & Modeling Steps:**
- High frequency → seasonality + temperature pattern
- ADF test → stationarity achieved with log + differencing
- Chose SARIMA and LSTM for comparison
**Model Chosen:**
LSTM (performed better due to nonlinear patterns)
**Outcome & Impact:**
- 95% accuracy on peak demand prediction
- $500K saved on emergency load balancing
**Time Series Principles Applied:**
- Seasonality (daily/hourly)
- Noise reduction
- Autocorrelation
3. Airline Passenger Volume Forecasting
**Industry:** Travel & Aviation
**Problem Statement:**
Predict monthly international passenger numbers to manage booking systems and flight scheduling.
**Data Used:**
- Monthly passenger count for 10 years
**Analysis & Modeling Steps:**
- Clear seasonal peak (summer, holidays)
- Trend + strong seasonality
- Used decomposition, then Prophet
**Model Chosen:**
Prophet
**Outcome & Impact:**
- Optimized fleet planning
- Improved flight schedule efficiency by 28%
**Time Series Principles Applied:**
- Trend
- Seasonality
- Decomposition
4. Web Traffic Forecasting for Media Company
**Industry:** Digital Media
**Problem Statement:**
Predict daily website traffic to allocate server resources and advertising slots.
**Data Used:**
- Daily page views, user engagement, marketing calendar
**Analysis & Modeling Steps:**
- Weekend dips, weekday spikes
- ADF test + rolling mean = stationarity issue fixed
- Used ARIMA + Prophet comparison
**Model Chosen:**
Prophet
**Outcome & Impact:**
- Better ad placement timing
- Avoided 3 major server downtimes due to accurate traffic forecasting
**Time Series Principles Applied:**
- Seasonality
- Noise
- Lagged features
5. Forecasting Demand for Ride-Sharing Service
**Industry:** Transportation / Mobility
**Problem Statement:**
Predict hourly ride demand across cities to optimize driver allocation and pricing.
**Data Used:**
- Hourly ride data
- Weather, event, location info
**Analysis & Modeling Steps:**
- Strong autocorrelation at daily and weekly levels
- Used LSTM and XGBoost with lag features
- Feature engineering critical
**Model Chosen:**
XGBoost with time-lagged features
**Outcome & Impact:**
- Increased ride availability by 25%
- Boosted driver earnings by 18% in high demand zones
**Time Series Principles Applied:**
- Autocorrelation
- Lag features
- Trend and cyclic pattern modeling
---

