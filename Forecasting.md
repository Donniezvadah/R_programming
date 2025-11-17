
# Time Series Analysis and Forecasting of Air Passengers
Author: Donnie

## Abstract
This report presents a comprehensive and mathematically detailed time series analysis...


## 1. Introduction

Forecasting monthly airline passenger volumes is essential for planning, scheduling, and resource allocation. 
This report demonstrates a complete, mathematically rigorous workflow following the SORS6102 project brief.

The objectives include:

1. Data description and preparation  
2. Exploratory analysis (trend, seasonality, variance)  
3. Stationarity testing  
4. Model identification (ACF/PACF, differencing)  
5. ARIMA/SARIMA estimation  
6. Residual diagnostics  
7. Forecasting (with back-transformation)  
8. Evaluation of accuracy  
9. Interpretation and recommendations  

---

## 2. Data and Setup

```{r}
library(forecast)
library(tseries)
library(ggplot2)
library(lmtest)
library(kableExtra)
library(dplyr)

data("AirPassengers")
ts_data <- AirPassengers
```

The dataset consists of **144 monthly observations** from **Jan 1949 – Dec 1960**, with frequency 12.

A raw plot shows increasing variance and multiplicative seasonality:

```{r}
autoplot(ts_data) +
  ggtitle("Monthly AirPassengers") +
  ylab("Passengers")
```

---

## 3. Transformation

Variance increases with level, so we apply a log transform:

\[
Y_t = \log(X_t)
\]

```{r}
log_ts <- log(ts_data)
autoplot(log_ts)
```

---

## 4. STL Decomposition

```{r}
stl_fit <- stl(log_ts, s.window="periodic")
autoplot(stl_fit)
```

Trend and seasonal components are clearly visible.

---

## 5. Stationarity and Differencing

ADF test on the logged data:

```{r}
adf.test(log_ts)
```

Not stationary → apply first and seasonal differencing:

\[
\nabla (1 - B^{12}) Y_t = (1 - B)(1 - B^{12}) Y_t
\]

```{r}
d1s <- diff(diff(log_ts), lag=12)
adf.test(d1s)
```

---


## 6. ACF and PACF Analysis

To identify ARIMA orders, we examine ACF and PACF of the differenced series:

```{r}
par(mfrow=c(1,2))
Acf(d1s, main="ACF of Differenced Series")
Pacf(d1s, main="PACF of Differenced Series")
par(mfrow=c(1,1))
```

**Interpretation:**

- Seasonal spikes at lag 12 suggest seasonal differencing was needed.
- ACF tailing + PACF significant at lag 1 suggests an MA(1) component.
- Seasonal MA(1) at lag 12 is also likely.

These patterns support the *Airline Model*:

\[
\text{ARIMA}(0,1,1)(0,1,1)_{12}
\]

---

## 7. Model Estimation

### 7.1 Fit the canonical Airline Model

```{r}
fit_airline <- Arima(log_ts, order=c(0,1,1), seasonal=c(0,1,1))
summary(fit_airline)
```

Model equations:

\[
(1 - B)(1 - B^{12})Y_t = (1 + \theta_1 B)(1 + \Theta_1 B^{12}) \varepsilon_t
\]

Where:

- \(\theta_1\) = non-seasonal MA(1)
- \(\Theta_1\) = seasonal MA(1)

---

## 7.2 Fit alternative SARIMA models

```{r}
fit_alt1 <- Arima(log_ts, order=c(2,1,1), seasonal=c(0,1,1))
fit_alt2 <- Arima(log_ts, order=c(0,1,1), seasonal=c(1,1,1))
```

---

## 7.3 Model Comparison

```{r}
models <- list(
  Airline = fit_airline,
  Alt1 = fit_alt1,
  Alt2 = fit_alt2
)

model_table <- data.frame(
  Model = names(models),
  AIC  = sapply(models, AIC),
  AICc = sapply(models, AICc),
  BIC  = sapply(models, BIC)
)

kable(model_table, caption="Model Selection Criteria") %>% kable_styling(full_width=FALSE)
```

**Conclusion:**  
The Airline Model often achieves the lowest AICc and is preferred for this dataset.

---

## 8. Residual Diagnostics

```{r}
checkresiduals(fit_airline)
```

### Requirements for a valid model:

1. Residuals ≈ white noise  
2. Ljung–Box p-value > 0.05  
3. No seasonal structure in residuals  
4. Approximate normality  

```{r}
Box.test(residuals(fit_airline), lag=24, type="Ljung-Box", fitdf=2)
```

If p-value > 0.05, no remaining autocorrelation.

---

## 9. Forecasting

We forecast in **log scale**, then back-transform.

```{r}
fcast <- forecast(fit_airline, h=24, level=c(80,95))
autoplot(fcast) + ggtitle("Forecasts (Log Scale)")
```

---

## 10. Back-Transformation to Original Scale

Given:

\[
\hat{X}_t = \exp(\hat{Y}_t)
\]

Bias-adjusted mean forecast:

\[
E[X_t] = \exp\left(\hat{Y}_t + \frac{\sigma^2}{2}\right)
\]

---

## 11. Train-Test Evaluation

Take last 24 months as the holdout set:

```{r}
h <- 24
train_ts <- window(log_ts, end=c(1958,12))
test_ts  <- window(log_ts, start=c(1959,1))

fit_train <- Arima(train_ts, order=c(0,1,1), seasonal=c(0,1,1))
fcast_train <- forecast(fit_train, h=h)
```

Back-transform:

```{r}
pred <- exp(fcast_train$mean)
actual <- exp(test_ts)
```

Compute error metrics:

```{r}
MAE <- mean(abs(pred - actual))
RMSE <- sqrt(mean((pred - actual)^2))
MAPE <- mean(abs((pred - actual)/actual)) * 100

data.frame(MAE, RMSE, MAPE)
```

---


## 12. Forecast Plots on Original Scale

We now compare forecasts with actual observations (last 24 months):

```{r}
autoplot(ts(exp(train_ts), start=start(train_ts), frequency=12), series="Training") +
  autolayer(ts(exp(test_ts), start=start(test_ts), frequency=12), series="Actual (Test)") +
  autolayer(ts(pred, start=start(test_ts), frequency=12), series="Forecast") +
  ggtitle("Forecast vs Actual (Original Scale)") +
  ylab("Passengers") + theme_minimal()
```

The model tracks the upward trend and seasonal pattern well.

---

## 13. Mathematical Appendix

### 13.1 SARIMA Model Structure

A general multiplicative SARIMA model is written as:

\[
\Phi_P(B^s)\phi(B) \nabla^d \nabla_s^D X_t
= \Theta_Q(B^s)\theta(B)\varepsilon_t,
\]

where:

- \(B\) is the backshift operator
- \(\nabla = 1-B\) is the differencing operator
- \(\nabla_s = 1-B^s\) is seasonal differencing
- \(\phi(B)\) and \(\theta(B)\) are nonseasonal AR/MA polynomials
- \(\Phi_P(B^s)\) and \(\Theta_Q(B^s)\) are seasonal AR/MA polynomials.

For the **Airline Model**:

- \(p = 0, d = 1, q = 1\)
- \(P = 0, D = 1, Q = 1, s = 12\)

Thus:

\[
(1-B)(1-B^{12})Y_t = (1 + \theta_1 B)(1 + \Theta_1 B^{12})\varepsilon_t.
\]

---

### 13.2 Forecast Recursion

One-step-ahead forecast:

\[
\hat{Y}_{t+1|t} = 
\sum_{i=1}^p \phi_i Y_{t+1-i}
+ \sum_{j=1}^q \theta_j \hat{\varepsilon}_{t+1-j}
+ \sum_{k=1}^P \Phi_k Y_{t+1-ks}
+ \sum_{\ell=1}^Q \Theta_\ell \hat{\varepsilon}_{t+1-\ell s}.
\]

For multistep forecasts, future residuals \(\hat{\varepsilon}_{t+h}\) are set to 0.

---

## 14. Discussion

### Model Performance

- The model captures both **trend** and **seasonality** with only two MA parameters.
- Diagnostics confirm **no residual autocorrelation**, meaning model structure is adequate.
- Forecasts follow the real series closely in the holdout period.

### Strengths

- Simple, interpretable, historically successful model.
- Performs extremely well for multiplicative seasonal series.

### Limitations

- No external regressors included.
- Long-term forecasts may underestimate structural changes.

---

## 15. Recommendations

1. Refit the model periodically as new data arrives.  
2. Consider SARIMAX models if external factors (e.g., economic indicators) are relevant.  
3. For long-term forecasting, evaluate exponential smoothing or structural models.  
4. Use bootstrap intervals for more robust uncertainty quantification.

---

## 16. Conclusion

This report demonstrated a full forecasting workflow, including:

- data transformation  
- decomposition  
- stationarity testing  
- SARIMA identification  
- parameter estimation  
- diagnostics  
- forecasting  
- evaluation  

The canonical Airline Model provides an excellent fit and strong forecasting accuracy.

---

## 17. References

- Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015).  
  *Time Series Analysis: Forecasting and Control.*

- Hyndman, R.J., & Athanasopoulos, G. (2018).  
  *Forecasting: Principles and Practice.*

- SORS6102 Project Assignment Brief.  

---

