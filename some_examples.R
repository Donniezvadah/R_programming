##############################################
# Simulating Linear Regression Estimates
# Monte Carlo Simulation in R
# --------------------------------------------
# Goal: Repeatedly simulate data and estimate
#       regression coefficients to check if
#       the estimator is unbiased.
##############################################

# True parameter values
beta0 = 5       # True intercept
beta1 = 4       # True slope

# Sample size
n = 25

# Set seed for reproducibility (same results every run)
set.seed(3635)

# Generate predictor values X ~ N(0,1)
X = rnorm(n, 0, 1)

# Generate one example dataset
Y = beta0 + beta1 * X + rnorm(n, 0, 5)

# Fit linear model
lrm.fit = lm(Y ~ X)

# Display estimated coefficients
lrm.fit$coef


###########################################################
# Monte Carlo Simulation
###########################################################

N = 50000   # Number of repeated simulations

# Empty vectors to store estimated coefficients
int.est = numeric(N)   # For intercept estimates
slp.est = numeric(N)   # For slope estimates

# Loop through simulations
for (i in 1:N) {
  
  # Generate new Y each simulation (same X)
  Y = beta0 + beta1 * X + rnorm(n, 0, 5)
  
  # Fit linear regression
  lrm.fit = lm(Y ~ X)
  
  # Store coefficient estimates
  int.est[i] = lrm.fit$coef[1]   # Estimated intercept
  slp.est[i] = lrm.fit$coef[2]   # Estimated slope
}

# Create a dataframe AFTER the loop (correct location)
df = data.frame(intercept = int.est, slope = slp.est)

# Preview first 6 rows of results
head(df)

# Compute Monte Carlo means of the estimates
apply(df, 2, mean)



