install.packages("carData")
install.packages("car", type = "source")
# Core tidyverse packages
install.packages("tidyverse")

# Regression diagnostics (modern replacements for car)
install.packages("performance")
install.packages("lmtest")

# ROC curves & AUC
install.packages("pROC")

# Date & time handling (if needed later)
install.packages("lubridate")

########################################
# PACKAGES
########################################

# Core packages
library(tidyverse)

# Regression diagnostics (instead of car)
library(performance)   # For VIF, heteroscedasticity, DW test
library(lmtest)        # For Breusch–Pagan test
library(pROC)          # For ROC + AUC

########################################
# SIMPLE LINEAR REGRESSION
########################################

model1 <- lm(mpg ~ wt, data = mtcars)
summary(model1)

########################################
# MULTIPLE LINEAR REGRESSION
########################################

model2 <- lm(mpg ~ wt + hp + cyl, data = mtcars)
summary(model2)

########################################
# PREDICTIONS
########################################

new_data <- data.frame(
  wt = c(2.5, 3.0, 3.5),
  hp = c(100, 120, 140),
  cyl = c(4, 6, 6)
)

predictions <- predict(model2, new_data, interval = "confidence")
predictions

########################################
# MODEL DIAGNOSTIC PLOTS
########################################

par(mfrow = c(2, 2))
plot(model2)

########################################
# MODEL ASSUMPTION CHECKS (NO car PACKAGE)
########################################

# 1. Multicollinearity (VIF)
performance::check_collinearity(model2)

# 2. Autocorrelation (Durbin-Watson)
performance::check_autocorrelation(model2)

# 3. Heteroscedasticity
lmtest::bptest(model2)

# 4. Normality of residuals
shapiro.test(resid(model2))

########################################
# LOGISTIC REGRESSION (BINARY)
########################################

data(iris)
iris_binary <- iris[iris$Species != "setosa", ]
iris_binary$Species <- droplevels(iris_binary$Species)

logit_model <- glm(
  Species ~ Sepal.Length + Sepal.Width,
  data = iris_binary,
  family = binomial(link = "logit")
)

summary(logit_model)

########################################
# PREDICTIONS FOR LOGISTIC MODEL
########################################

pred_probs <- predict(logit_model, type = "response")
predicted_class <- ifelse(pred_probs > 0.5, "virginica", "versicolor")

# Confusion matrix
table(Predicted = predicted_class, Actual = iris_binary$Species)

########################################
# ROC CURVE + AUC
########################################

roc_obj <- roc(iris_binary$Species, pred_probs)
plot(roc_obj, main = "ROC Curve")
auc(roc_obj)

########################################
# ODDS RATIOS
########################################

exp(coef(logit_model))

# Confidence intervals
exp(confint(logit_model))




