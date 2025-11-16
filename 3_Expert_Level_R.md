# Expert Level R

## Table of Contents
1. [Statistical Modeling](#statistical-modeling)
2. [Machine Learning with caret](#machine-learning-with-caret)
3. [Advanced ML Algorithms](#advanced-ml-algorithms)
4. [Model Evaluation and Tuning](#model-evaluation-and-tuning)
5. [Time Series Analysis](#time-series-analysis)
6. [R Markdown and Reproducible Research](#r-markdown)
7. [Performance Optimization](#performance-optimization)
8. [Advanced Topics](#advanced-topics)

---

## Statistical Modeling

### Linear Regression

```r
# Simple linear regression
model <- lm(mpg ~ wt, data = mtcars)
summary(model)

# Multiple regression
model <- lm(mpg ~ wt + hp + cyl, data = mtcars)
summary(model)

# Predictions
new_data <- data.frame(wt = c(2.5, 3.0, 3.5), hp = c(100, 120, 140), cyl = c(4, 6, 6))
predictions <- predict(model, new_data, interval = "confidence")

# Model diagnostics
par(mfrow = c(2, 2))
plot(model)

# Check assumptions
library(car)
vif(model)  # Variance Inflation Factor (multicollinearity)
durbinWatsonTest(model)  # Autocorrelation
ncvTest(model)  # Heteroscedasticity

# Residual analysis
residuals <- resid(model)
shapiro.test(residuals)  # Normality test
```
#### Math and Explanation

We observe a response $y_i$ and $p$ predictors $x_{i1},\dots,x_{ip}$ for $i = 1,\dots,n$. Linear regression assumes the conditional mean of $Y$ is a linear function of the predictors:

$$
y_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip} + \varepsilon_i,
$$

or in matrix form

$$
\mathbf{y} = X\beta + \varepsilon,
$$

where $X$ is the $n \times (p+1)$ design matrix (including a column of ones for the intercept), $\beta$ is the vector of coefficients, and $\varepsilon$ is an error term (often modeled as $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$).

The **ordinary least squares (OLS)** estimator chooses $\hat\beta$ to minimize the sum of squared residuals:

$$
\hat\beta = \arg\min_\beta \sum_{i=1}^n (y_i - \hat y_i)^2
            = \arg\min_\beta \lVert \mathbf{y} - X\beta \rVert_2^2.
$$

When $X^\top X$ is invertible, the minimizer has the closed form

$$
\hat\beta = (X^\top X)^{-1} X^\top \mathbf{y}.
$$

The diagnostic code (residual plots, variance inflation factor, Durbin–Watson test, tests for heteroscedasticity and normality) is checking whether the usual assumptions—linearity, independence, constant variance, and approximately normal residuals—are reasonable for this model.

### Logistic Regression

```r
# Binary classification
data(iris)
iris_binary <- iris[iris$Species != "setosa", ]
iris_binary$Species <- droplevels(iris_binary$Species)

model <- glm(Species ~ Sepal.Length + Sepal.Width, 
             data = iris_binary, 
             family = binomial(link = "logit"))
summary(model)

# Predictions
predictions <- predict(model, type = "response")
predicted_class <- ifelse(predictions > 0.5, "virginica", "versicolor")

# Confusion matrix
table(Predicted = predicted_class, Actual = iris_binary$Species)

# ROC curve and AUC
library(pROC)
roc_obj <- roc(iris_binary$Species, predictions)
plot(roc_obj, main = "ROC Curve")
auc(roc_obj)

# Odds ratios
exp(coef(model))
exp(confint(model))
```
#### Math and Explanation

For a binary response $Y \in \{0,1\}$ and predictors $x$, logistic regression models the conditional probability of the positive class as

$$
P(Y = 1 \mid x) = p(x) = \sigma(\eta) = \frac{1}{1 + e^{-\eta}}, \quad \eta = \beta_0 + \beta^\top x.
$$

Equivalently, the **log-odds** (logit) are linear in the predictors:

$$
\log\frac{p(x)}{1 - p(x)} = \beta_0 + \beta^\top x.
$$

Given data $(x_i, y_i)$, $i = 1,\dots,n$, the likelihood of the parameters $\beta$ is

$$
L(\beta) = \prod_{i=1}^n p(x_i)^{y_i} [1 - p(x_i)]^{1-y_i},
$$

with log-likelihood

$$
\ell(\beta) = \sum_{i=1}^n \big[ y_i \log p(x_i) + (1-y_i) \log(1-p(x_i)) \big].
$$

The fitted coefficients $\hat\beta$ **maximize** $\ell(\beta)$ (or minimize the negative log-likelihood, a cross-entropy loss). There is no closed-form solution, so numerical methods such as iteratively reweighted least squares or gradient-based optimization are used.

Each coefficient $\beta_j$ represents the change in **log-odds** of $Y=1$ for a one-unit increase in $x_j$, holding other predictors fixed. Exponentiating gives an **odds ratio** $e^{\beta_j}$, which is what `exp(coef(model))` and `exp(confint(model))` report. The ROC curve and AUC summarize how well the predicted probabilities separate the two classes over all possible thresholds.

### Generalized Linear Models (GLM)

```r
# Poisson regression (count data)
model_poisson <- glm(count ~ factor1 + factor2, 
                     family = poisson(link = "log"),
                     data = df)

# Negative binomial (overdispersed count data)
library(MASS)
model_nb <- glm.nb(count ~ factor1 + factor2, data = df)

# Gamma regression (continuous positive data)
model_gamma <- glm(response ~ predictor, 
                   family = Gamma(link = "log"),
                   data = df)
```
#### Math and Explanation

Generalized linear models extend linear regression to non-Gaussian response distributions from the **exponential family**. For observation $i$ we write

$$
g(\mu_i) = \eta_i = x_i^\top \beta, \quad \mu_i = \mathbb{E}[Y_i \mid x_i],
$$

where $g(\cdot)$ is a link function and $\mu_i$ is the conditional mean of $Y_i$.

In this code:

- **Poisson regression** assumes $Y_i \sim \text{Poisson}(\lambda_i)$ for count data and uses the log link
  $$\log(\lambda_i) = x_i^\top \beta.$$
- **Negative binomial regression** also models counts but introduces an additional dispersion parameter to handle overdispersion (variance greater than the mean), again with a log link.
- **Gamma regression** assumes $Y_i$ is positive and continuous and often uses a log link
  $$\log(\mu_i) = x_i^\top \beta.$$

The parameters $\beta$ are estimated by **maximum likelihood**, i.e.

$$
\hat\beta = \arg\max_\beta \; \ell(\beta),
$$

where $\ell(\beta)$ is the log-likelihood implied by the chosen distribution and link. In practice, `glm()` uses **iteratively reweighted least squares (IRLS)** to solve this optimization by repeatedly fitting weighted least-squares regressions until convergence.

Diagnostics for GLMs focus on whether the chosen distribution and link adequately capture the mean–variance relationship (e.g. checking residuals, deviance, and overdispersion for Poisson models).

### Mixed Effects Models

```r
library(lme4)

# Linear mixed model
model <- lmer(reaction ~ days + (1 + days | subject), data = sleepstudy)
summary(model)

# Random intercept and slope
model <- lmer(y ~ x + (x | group), data = df)

# Extract random effects
ranef(model)
fixef(model)

# Model comparison
model1 <- lmer(y ~ x + (1 | group), data = df)
model2 <- lmer(y ~ x + (1 + x | group), data = df)
anova(model1, model2)
```
#### Math and Explanation

Mixed effects (multilevel) models include both **fixed effects**, which are shared across all groups, and **random effects**, which capture group-specific deviations.

A simple linear mixed model for observation $i$ in group $j$ can be written as

$$
y_{ij} = \beta_0 + \beta^\top x_{ij} + b_{0j} + b_j^\top z_{ij} + \varepsilon_{ij},
$$

where

- $\beta$ are fixed-effect coefficients,
- $b_j$ are random effects for group $j$ (typically $b_j \sim \mathcal{N}(0, \Sigma_b)$),
- $z_{ij}$ selects which random effects apply to observation $ij$,
- $\varepsilon_{ij} \sim \mathcal{N}(0, \sigma^2)$ are residual errors.

Formulas like `reaction ~ days + (1 + days | subject)` tell `lmer()` to include a fixed effect of `days` plus random intercepts and random slopes by `subject`.

Estimation proceeds by maximizing the **(restricted) marginal likelihood** obtained by integrating out the random effects:

$$
L(\beta, \Sigma_b, \sigma^2) = \prod_j \int f(y_j \mid b_j, \beta, \sigma^2) f(b_j \mid \Sigma_b)\, db_j.
$$

`lme4` uses numerical methods (Laplace approximation, sparse linear algebra) to approximate this likelihood efficiently. The `anova(model1, model2)` call is performing a likelihood-ratio test between nested mixed-effect models to assess whether the more complex random-effect structure significantly improves model fit.

### Survival Analysis

```r
library(survival)

# Kaplan-Meier survival curve
fit <- survfit(Surv(time, status) ~ group, data = lung)
plot(fit, col = c("blue", "red"), 
     xlab = "Time", ylab = "Survival Probability")
legend("topright", legend = c("Group 1", "Group 2"), col = c("blue", "red"), lty = 1)

# Log-rank test
survdiff(Surv(time, status) ~ group, data = lung)

# Cox proportional hazards model
cox_model <- coxph(Surv(time, status) ~ age + sex + ph.ecog, data = lung)
summary(cox_model)

# Hazard ratios
exp(coef(cox_model))
exp(confint(cox_model))
```
#### Math and Explanation

Let $T$ be a time-to-event random variable. Two key functions in survival analysis are:

- **Survival function**: $S(t) = P(T > t)$.
- **Hazard function**: $h(t) = \lim_{\Delta t \to 0} P(t \le T < t + \Delta t \mid T \ge t)/\Delta t$.

The Kaplan–Meier estimator used by `survfit()` estimates $S(t)$ nonparametrically by multiplying conditional survival probabilities at each observed event time.

The **Cox proportional hazards model** assumes that the hazard for subject $i$ with covariates $x_i$ is

$$
h(t \mid x_i) = h_0(t) \exp(\beta^\top x_i),
$$

where $h_0(t)$ is an unspecified baseline hazard and $\beta$ are regression coefficients.

Instead of a full likelihood, Cox regression maximizes the **partial likelihood**

$$
L_p(\beta) = \prod_{i \in \mathcal{D}} \frac{\exp(\beta^\top x_i)}{\sum_{j \in R_i} \exp(\beta^\top x_j)},
$$

where $\mathcal{D}$ is the set of event times and $R_i$ is the risk set at the time of event $i$. Maximizing $L_p(\beta)$ yields estimates $\hat\beta$, and

$$
\exp(\beta_j)
$$

is interpreted as a **hazard ratio**: the multiplicative change in hazard for a one-unit increase in covariate $x_j$.

The log-rank test compares two (or more) survival curves under the null hypothesis that their hazard functions are equal over time.

---

## Machine Learning with caret

### Setup and Data Splitting

```r
library(caret)

# Split data
set.seed(123)
trainIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
train_data <- iris[trainIndex, ]
test_data <- iris[-trainIndex, ]

# Cross-validation setup
train_control <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = "final",
  classProbs = TRUE,
  summaryFunction = multiClassSummary
)
```

### Training Models

```r
# Random Forest
set.seed(123)
rf_model <- train(
  Species ~ .,
  data = train_data,
  method = "rf",
  trControl = train_control,
  tuneLength = 5
)
print(rf_model)

# Support Vector Machine
svm_model <- train(
  Species ~ .,
  data = train_data,
  method = "svmRadial",
  trControl = train_control,
  tuneLength = 5
)

# Gradient Boosting
gbm_model <- train(
  Species ~ .,
  data = train_data,
  method = "gbm",
  trControl = train_control,
  verbose = FALSE
)

# Neural Network
nnet_model <- train(
  Species ~ .,
  data = train_data,
  method = "nnet",
  trControl = train_control,
  trace = FALSE
)
```
#### Math and Explanation

Although `caret::train()` hides the underlying optimization details, each `method` corresponds to a specific model class and loss function:

- **Random Forest (`method = "rf"`)**
  - An ensemble of decision trees $\{T_b(x)\}_{b=1}^B$ grown on bootstrap samples.
  - For classification, the prediction is the **majority vote** of the trees.
  - Each split in a tree greedily chooses the feature and split point that maximizes decrease in node impurity (e.g. Gini index
    $$
    G = \sum_k p_k(1-p_k),
    $$
    where $p_k$ is the class proportion in that node).

- **Support Vector Machine (`method = "svmRadial"`)**
  - Finds a separating surface with maximum margin. In the soft-margin formulation it solves
    $$
    \min_{w,b,\xi} \; \tfrac{1}{2}\lVert w \rVert^2 + C \sum_{i=1}^n \xi_i
    $$
    subject to
    $$
    y_i (w^T \phi(x_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0,
    $$
    where $\phi(\cdot)$ is induced by the RBF kernel and $C$ controls the penalty for misclassification.

- **Gradient Boosting (`method = "gbm"`)**
  - Builds an additive model of weak learners
    $$
    F_M(x) = \sum_{m=1}^M \gamma_m h_m(x),
    $$
    where each new tree $h_m$ is fit to approximate the **negative gradient** of the loss with respect to the current predictions.
  - This is a stagewise functional gradient descent procedure that iteratively reduces training loss.

- **Neural Network (`method = "nnet"`)**
  - Typically a single hidden layer with non-linear activation:
    $$
    h = \sigma(W^{(1)} x + b^{(1)}), \quad \hat y = f(W^{(2)} h + b^{(2)}),
    $$
    where $\sigma$ is an activation (e.g. logistic, ReLU) and $f$ is softmax (classification) or identity (regression).
  - Training minimizes a loss such as cross-entropy using gradient descent and backpropagation.

`caret` handles resampling (cross-validation), hyperparameter tuning, and calling the appropriate underlying algorithms, but the optimization for each model follows these mathematical objectives.

### Predictions and Evaluation

```r
# Make predictions
rf_pred <- predict(rf_model, test_data)
svm_pred <- predict(svm_model, test_data)

# Confusion Matrix
confusionMatrix(rf_pred, test_data$Species)

# Multiple metrics
postResample(rf_pred, test_data$Species)

# Variable importance
varImp(rf_model)
plot(varImp(rf_model), top = 10)
```

### Model Comparison

```r
# Compare multiple models
results <- resamples(list(
  RF = rf_model,
  SVM = svm_model,
  GBM = gbm_model,
  NNET = nnet_model
))

summary(results)
dotplot(results)
bwplot(results)

# Statistical test
diff_results <- diff(results)
summary(diff_results)
```

---

## Advanced ML Algorithms

### XGBoost

```r
library(xgboost)

# Prepare data
train_matrix <- as.matrix(train_data[, -5])
train_label <- as.numeric(train_data$Species) - 1
test_matrix <- as.matrix(test_data[, -5])

# Create DMatrix
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest <- xgb.DMatrix(data = test_matrix)

# Set parameters
params <- list(
  objective = "multi:softmax",
  num_class = 3,
  max_depth = 6,
  eta = 0.3,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain),
  early_stopping_rounds = 10,
  verbose = 0
)

# Predictions
predictions <- predict(xgb_model, dtest)

# Feature importance
importance <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance, top_n = 10)
```
#### Math and Explanation

XGBoost fits an additive ensemble of decision trees:

$$
\hat y_i = \sum_{k=1}^K f_k(x_i), \quad f_k \in \mathcal{F},
$$

where each $f_k$ is a regression tree that assigns a constant score to each leaf region.

For a loss function $l(y_i, \hat y_i)$, it minimizes the **regularized objective**

$$
\mathcal{L} = \sum_{i=1}^n l\big(y_i, \hat y_i\big)
            + \sum_{k=1}^K \Omega(f_k),
$$

with tree complexity penalty

$$
\Omega(f) = \gamma T + \tfrac{1}{2} \lambda \lVert w \rVert^2,
$$

where $T$ is the number of leaves and $w$ are the leaf scores. At each boosting iteration, XGBoost uses a second-order Taylor expansion of the loss around the current predictions to compute optimal leaf weights and select splits, giving a fast approximation to gradient-boosted trees with built-in regularization.

Hyperparameters such as `max_depth`, `eta` (learning rate), `subsample`, and `colsample_bytree` control how complex each tree can be and how strongly the ensemble is regularized.

### Random Forest (ranger)

```r
library(ranger)

# Fast random forest
rf_fast <- ranger(
  Species ~ .,
  data = train_data,
  num.trees = 500,
  importance = "impurity",
  probability = TRUE
)

# Predictions
pred <- predict(rf_fast, test_data)$predictions

# Variable importance
importance(rf_fast)
```
#### Math and Explanation

The `ranger` model is a fast implementation of the same random forest idea used earlier: many decision trees are grown on bootstrap samples of the data, and predictions are aggregated.

If $T_b(x)$ denotes the prediction of tree $b$, then for classification the forest prediction is typically

$$
\hat y = \text{mode}\{T_1(x), T_2(x), \dots, T_B(x)\},
$$

and for regression it is the average. Each tree is built by recursively selecting splits that maximize the reduction in node impurity (e.g. Gini index for classification) while only considering a random subset of predictors at each node.

This combination of **bagging** (bootstrap aggregation) and **random feature selection** reduces variance and decorrelates individual trees, leading to strong predictive performance without overfitting as badly as a single deep tree. The importance scores from `importance(rf_fast)` summarize how much each variable contributes to reducing impurity across the forest.

### Deep Learning with keras

```r
library(keras)

# Prepare data
x_train <- as.matrix(train_data[, -5])
y_train <- to_categorical(as.numeric(train_data$Species) - 1, 3)
x_test <- as.matrix(test_data[, -5])

# Build model
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = ncol(x_train)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 3, activation = "softmax")

# Compile
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(lr = 0.001),
  metrics = c("accuracy")
)

# Train
history <- model %>% fit(
  x_train, y_train,
  epochs = 100,
  batch_size = 16,
  validation_split = 0.2,
  verbose = 0
)

# Plot training history
plot(history)

# Predictions
predictions <- model %>% predict(x_test)
predicted_class <- max.col(predictions) - 1
```
#### Math and Explanation

The `keras` example builds a fully connected **neural network** with two hidden layers. For an input vector $x$, weights $W^{(1)}, W^{(2)}, W^{(3)}$ and biases $b^{(1)}, b^{(2)}, b^{(3)}$, the forward pass is

$$
h^{(1)} = \text{ReLU}(W^{(1)} x + b^{(1)}),\\
h^{(2)} = \text{ReLU}(W^{(2)} h^{(1)} + b^{(2)}),\\
z = W^{(3)} h^{(2)} + b^{(3)},\\
\hat y = \text{softmax}(z),
$$

where ReLU is the rectified linear unit activation and softmax converts logits $z$ into class probabilities.

For multi-class classification with one-hot targets $y_i$, the loss is **categorical cross-entropy**

$$
\mathcal{L} = - \sum_{i=1}^n \sum_{k=1}^K y_{ik} \log \hat y_{ik},
$$

which is minimized using stochastic gradient descent (or a variant like Adam, as used here). Gradients of the loss with respect to all weights and biases are computed efficiently by **backpropagation**.

The dropout layers randomly zero out a fraction of hidden units during training, which acts as a regularizer by preventing co-adaptation of features and approximating an ensemble of many thinned networks.

---

## Model Evaluation and Tuning

### Hyperparameter Tuning

```r
# Grid search
grid <- expand.grid(
  mtry = c(2, 3, 4),
  splitrule = c("gini", "extratrees"),
  min.node.size = c(1, 5, 10)
)

model <- train(
  Species ~ .,
  data = train_data,
  method = "ranger",
  trControl = train_control,
  tuneGrid = grid
)

# Random search
model <- train(
  Species ~ .,
  data = train_data,
  method = "ranger",
  trControl = train_control,
  tuneLength = 20,
  search = "random"
)

# Best parameters
model$bestTune
```

### Cross-Validation Strategies

```r
# K-fold CV
cv_control <- trainControl(method = "cv", number = 10)

# Repeated K-fold
cv_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# Leave-one-out CV
loo_control <- trainControl(method = "LOOCV")

# Time series CV
ts_control <- trainControl(method = "timeslice",
                           initialWindow = 100,
                           horizon = 10,
                           fixedWindow = FALSE)

# Custom CV
folds <- createFolds(train_data$Species, k = 5)
custom_control <- trainControl(method = "cv", index = folds)
```

### Feature Selection

```r
# Recursive feature elimination
control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
results <- rfe(
  train_data[, -5],
  train_data$Species,
  sizes = c(1:4),
  rfeControl = control
)
print(results)
predictors(results)
plot(results, type = c("g", "o"))

# Boruta (all-relevant feature selection)
library(Boruta)
boruta_result <- Boruta(Species ~ ., data = train_data, doTrace = 2)
print(boruta_result)
plot(boruta_result)
getSelectedAttributes(boruta_result, withTentative = FALSE)
```

### Ensemble Methods

```r
# Stacking
library(caretEnsemble)

# Train multiple models
model_list <- caretList(
  Species ~ .,
  data = train_data,
  trControl = train_control,
  methodList = c("rf", "glm", "knn")
)

# Stack models
stack_model <- caretStack(
  model_list,
  method = "glm",
  metric = "Accuracy",
  trControl = trainControl(
    method = "cv",
    number = 10,
    savePredictions = "final",
    classProbs = TRUE
  )
)

# Predictions
stack_pred <- predict(stack_model, test_data)
```

---

## Time Series Analysis

### Basic Time Series

```r
# Create time series
ts_data <- ts(AirPassengers, start = c(1949, 1), frequency = 12)

# Decomposition
decomp <- decompose(ts_data)
plot(decomp)

# STL decomposition
stl_result <- stl(ts_data, s.window = "periodic")
plot(stl_result)
```

### ARIMA Models

```r
library(forecast)

# Automatic ARIMA
fit <- auto.arima(ts_data)
summary(fit)

# Check residuals
checkresiduals(fit)

# Forecast
forecast_result <- forecast(fit, h = 24)
plot(forecast_result)

# Manual ARIMA
fit_manual <- arima(ts_data, order = c(1, 1, 1), seasonal = list(order = c(1, 1, 1)))

# Model diagnostics
tsdiag(fit_manual)
```
#### Math and Explanation

An $\text{ARIMA}(p,d,q)$ model combines **autoregressive (AR)**, **integration (I)**, and **moving-average (MA)** components. Applied to the differenced series

$$
w_t = (1 - B)^d y_t,
$$

where $B$ is the backshift operator ($B y_t = y_{t-1}$), the general form is

$$
\phi(B) w_t = \theta(B) \varepsilon_t,
$$

with AR and MA polynomials

$$
\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p, \qquad
\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q,
$$

and innovations $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$.

The parameters $(\phi_i, \theta_j, d)$ are typically estimated by **maximum likelihood**, which numerically maximizes the probability of the observed time series under the model. `auto.arima()` searches over candidate $(p,d,q)$ values using criteria such as AICc, while `checkresiduals()` and `tsdiag()` help verify that the residuals behave like white noise (no remaining structure).

### Exponential Smoothing

```r
# Simple exponential smoothing
fit_ses <- ses(ts_data, h = 12)

# Holt's method (trend)
fit_holt <- holt(ts_data, h = 12)

# Holt-Winters (trend + seasonality)
fit_hw <- hw(ts_data, seasonal = "multiplicative", h = 24)

# ETS (Error, Trend, Seasonal)
fit_ets <- ets(ts_data)
forecast_ets <- forecast(fit_ets, h = 24)
plot(forecast_ets)
```
#### Math and Explanation

Exponential smoothing methods describe a time series in terms of recursively updated components for **level**, **trend**, and (optionally) **seasonality**.

For example, **simple exponential smoothing** (no trend or seasonality) uses

$$
\ell_t = \alpha y_t + (1-\alpha)\, \ell_{t-1}, \quad
\hat y_{t+1} = \ell_t,
$$

where $\alpha \in (0,1)$ is the smoothing parameter and $\ell_t$ is the smoothed level.

Holt’s method adds a trend component $b_t$ and Holt–Winters adds seasonal components (additive or multiplicative). The `ets()` function chooses among a family of **Error–Trend–Seasonal (ETS)** state-space models and estimates the smoothing parameters by maximizing the likelihood of one-step-ahead forecast errors.

### Prophet (Facebook)

```r
library(prophet)

# Prepare data
df <- data.frame(
  ds = seq(as.Date("2020-01-01"), by = "day", length.out = 365),
  y = rnorm(365, mean = 100, sd = 10)
)

# Fit model
m <- prophet(df)

# Make future dataframe
future <- make_future_dataframe(m, periods = 90)

# Forecast
forecast <- predict(m, future)

# Plot
plot(m, forecast)
prophet_plot_components(m, forecast)
```
#### Math and Explanation

Prophet models a time series as a sum of interpretable components

$$
y(t) = g(t) + s(t) + h(t) + \varepsilon_t,
$$

where

- $g(t)$ is a piecewise linear or logistic **trend** with changepoints,
- $s(t)$ is a **seasonal** component (e.g. yearly, weekly) represented by Fourier series,
- $h(t)$ captures **holiday or event** effects,
- $\varepsilon_t$ is an error term.

Prophet uses a Bayesian/state-space formulation with priors that encourage smooth trends and reasonable seasonal patterns. Fitting `m <- prophet(df)` estimates the parameters of these components (trend slopes, changepoints, seasonal amplitudes, holiday effects), and `prophet_plot_components()` visualizes each component separately so you can interpret how trend, seasonality, and holidays contribute to the observed series.

---

## R Markdown and Reproducible Research

### Basic R Markdown

```markdown
---
title: "My Analysis"
author: "Your Name"
date: "`r Sys.Date()`"
output: html_document
---

## Introduction

This is an R Markdown document.

```r
# chunk: setup, include=FALSE
knitr::opts_chunk$set(echo = TRUE, warning = FALSE, message = FALSE)
library(tidyverse)
```

## Data Analysis

```r
# chunk: load-data
data(mtcars)
summary(mtcars)
```

## Visualization

```r
# chunk: plot, fig.width=8, fig.height=6
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  theme_minimal()
```

## Conclusion

Results show...
```

### Advanced R Markdown Features

```r
# Parameterized reports
---
title: "Parameterized Report"
output: html_document
params:
  year: 2023
  country: "USA"
---

# Inline code
The analysis covers data from `r params$year` in `r params$country`.

# Caching
```r
# chunk: expensive-computation, cache=TRUE
result <- very_long_computation()
```

# Child documents
```r
# chunk: child='analysis-section.Rmd'
```

# Tables with kable
```r
# chunk: r (kable example)
library(knitr)
kable(head(mtcars), caption = "Motor Trend Cars")
```

# Interactive tables
```r
# chunk: r (DT example)
library(DT)
datatable(mtcars, filter = 'top')
```
```

### Output Formats

```r
# PDF output
---
output: pdf_document
---

# Word document
---
output: word_document
---

# Presentation (slides)
---
output: 
  ioslides_presentation:
    widescreen: true
    smaller: true
---

# Dashboard
---
output: flexdashboard::flex_dashboard
---

# Bookdown (books)
---
output: bookdown::gitbook
---
```

---

## Performance Optimization

### Profiling Code

```r
# Profile code execution
library(profvis)

profvis({
  data <- data.frame(x = rnorm(10000), y = rnorm(10000))
  model <- lm(y ~ x, data = data)
  predictions <- predict(model, data)
})

# Benchmark functions
library(microbenchmark)

microbenchmark(
  base = apply(matrix(1:1000000, ncol = 100), 1, sum),
  rowSums = rowSums(matrix(1:1000000, ncol = 100)),
  times = 100
)
```

### Vectorization

```r
# Slow: loop
result <- numeric(10000)
for (i in 1:10000) {
  result[i] <- i^2
}

# Fast: vectorized
result <- (1:10000)^2

# Using apply family
result <- sapply(1:10000, function(x) x^2)

# Even faster: built-in functions
result <- (1:10000)^2
```

### Parallel Processing

```r
library(parallel)

# Detect cores
detectCores()

# Parallel lapply
cl <- makeCluster(4)
result <- parLapply(cl, 1:1000, function(x) x^2)
stopCluster(cl)

# foreach with doParallel
library(doParallel)
registerDoParallel(cores = 4)

result <- foreach(i = 1:1000, .combine = c) %dopar% {
  i^2
}

# Parallel caret training
cl <- makeCluster(4)
registerDoParallel(cl)

model <- train(
  Species ~ .,
  data = iris,
  method = "rf",
  trControl = trainControl(method = "cv", number = 10, allowParallel = TRUE)
)

stopCluster(cl)
```

### data.table for Speed

```r
library(data.table)

# Create data.table
dt <- as.data.table(mtcars)

# Fast aggregation
dt[, .(avg_mpg = mean(mpg)), by = cyl]

# Fast joins
dt1 <- data.table(id = 1:1000000, value = rnorm(1000000))
dt2 <- data.table(id = 1:1000000, category = sample(letters, 1000000, replace = TRUE))
setkey(dt1, id)
setkey(dt2, id)
result <- dt1[dt2]  # Fast join

# Update by reference
dt[, new_col := mpg * 2]
```

---

## Advanced Topics

### Custom Functions and Debugging

```r
# Function with error handling
safe_divide <- function(x, y) {
  tryCatch({
    if (y == 0) stop("Division by zero")
    result <- x / y
    return(result)
  }, error = function(e) {
    message("Error: ", e$message)
    return(NA)
  }, warning = function(w) {
    message("Warning: ", w$message)
  })
}

# Debugging
debug(my_function)  # Enter debug mode
debugonce(my_function)  # Debug once
undebug(my_function)  # Exit debug mode

# Browser
my_function <- function(x) {
  y <- x * 2
  browser()  # Pause execution
  z <- y + 1
  return(z)
}

# Assertions
stopifnot(x > 0)
assertthat::assert_that(is.numeric(x))
```

### S3 and S4 Classes

```r
# S3 class
person <- list(name = "Alice", age = 25)
class(person) <- "Person"

print.Person <- function(x) {
  cat("Person:", x$name, "Age:", x$age, "\n")
}

# S4 class
setClass("Employee",
  slots = list(
    name = "character",
    age = "numeric",
    salary = "numeric"
  )
)

emp <- new("Employee", name = "Bob", age = 30, salary = 50000)

# Methods
setGeneric("getSalary", function(obj) standardGeneric("getSalary"))
setMethod("getSalary", "Employee", function(obj) obj@salary)
```

### Writing Packages

```r
# Create package structure
usethis::create_package("mypackage")

# Add function
usethis::use_r("my_function")

# Add documentation with roxygen2
#' Add Two Numbers
#'
#' @param x A number
#' @param y A number
#' @return The sum of x and y
#' @export
#' @examples
#' add_numbers(2, 3)
add_numbers <- function(x, y) {
  x + y
}

# Generate documentation
devtools::document()

# Add tests
usethis::use_test("my_function")

# Check package
devtools::check()

# Install
devtools::install()
```

---

## Practice Projects

### Project 1: Predictive Modeling Pipeline

```r
library(tidyverse)
library(caret)

# 1. Load and prepare data
data(iris)
set.seed(123)

# 2. Split data
trainIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
train_data <- iris[trainIndex, ]
test_data <- iris[-trainIndex, ]

# 3. Preprocessing
preprocess_params <- preProcess(train_data[, -5], method = c("center", "scale"))
train_processed <- predict(preprocess_params, train_data[, -5])
test_processed <- predict(preprocess_params, test_data[, -5])

# 4. Train multiple models
models <- list(
  rf = train(Species ~ ., data = train_data, method = "rf"),
  svm = train(Species ~ ., data = train_data, method = "svmRadial"),
  gbm = train(Species ~ ., data = train_data, method = "gbm", verbose = FALSE)
)

# 5. Evaluate
results <- lapply(models, function(m) {
  pred <- predict(m, test_data)
  confusionMatrix(pred, test_data$Species)
})

# 6. Select best model
accuracies <- sapply(results, function(r) r$overall["Accuracy"])
best_model <- models[[which.max(accuracies)]]

# 7. Save model
saveRDS(best_model, "best_model.rds")
```

### Project 2: Time Series Forecasting

```r
library(forecast)
library(tidyverse)

# 1. Load data
ts_data <- ts(AirPassengers, start = c(1949, 1), frequency = 12)

# 2. Split train/test
train <- window(ts_data, end = c(1958, 12))
test <- window(ts_data, start = c(1959, 1))

# 3. Fit multiple models
models <- list(
  arima = auto.arima(train),
  ets = ets(train),
  tbats = tbats(train)
)

# 4. Forecast
forecasts <- lapply(models, forecast, h = length(test))

# 5. Evaluate
accuracy_results <- lapply(forecasts, function(f) {
  accuracy(f, test)
})

# 6. Visualize
par(mfrow = c(2, 2))
for (i in seq_along(forecasts)) {
  plot(forecasts[[i]], main = names(forecasts)[i])
  lines(test, col = "red")
}
```

---

## Summary

### Skills Mastered

- ✅ Statistical modeling (linear, logistic, GLM, mixed models)
- ✅ Machine learning with caret and advanced algorithms
- ✅ Model evaluation, tuning, and ensemble methods
- ✅ Time series analysis and forecasting
- ✅ R Markdown for reproducible research
- ✅ Performance optimization and parallel processing
- ✅ Advanced R programming concepts

### Next Steps

1. **Build:** Create end-to-end ML projects
2. **Explore:** Shiny apps (`4_Shiny_Apps_in_R.md`)
3. **Develop:** R packages and developer tools
4. **Publish:** Share analyses on RPubs, GitHub

**Continue to Shiny Apps! 🚀**
