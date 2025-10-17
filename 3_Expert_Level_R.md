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

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, warning = FALSE, message = FALSE)
library(tidyverse)
```

## Data Analysis

```{r load-data}
data(mtcars)
summary(mtcars)
```

## Visualization

```{r plot, fig.width=8, fig.height=6}
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
```{r expensive-computation, cache=TRUE}
result <- very_long_computation()
```

# Child documents
```{r child='analysis-section.Rmd'}
```

# Tables with kable
```{r}
library(knitr)
kable(head(mtcars), caption = "Motor Trend Cars")
```

# Interactive tables
```{r}
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
