###############################################################
# Simulating and Plotting a Mixture of Two Normal Distributions
# --------------------------------------------------------------
# We create a mixture distribution:
#   With probability e:       X ~ N(0, 1)
#   With probability (1 - e): X ~ N(m, s)
#
# Then we:
# 1. Generate random samples from this mixture.
# 2. Plot a histogram of the simulated distribution.
# 3. Overlay the TRUE mixture density curve for comparison.
###############################################################

# Mixture proportion: probability of selecting the **first** normal
e <- 0.3      # 30% from N(0,1), 70% from N(m, s)

# Number of simulations / sample size
nsim <- 1000

# Parameters of the second normal distribution
m <- 2        # mean of second normal
s <- 1        # sd of second normal

###############################################################
# Step 1: Generate mixture component indicators
###############################################################

# Generate nsim uniform values and check if each < e.
# u = 1 => sample from N(0,1)
# u = 0 => sample from N(m, s)
u <- (runif(nsim) < e)

###############################################################
# Step 2: Generate random samples from both distributions
###############################################################

# First normal distribution: N(0, 1)
z  <- rnorm(nsim)

# Second normal distribution: N(m, s)
z1 <- rnorm(nsim, mean = m, sd = s)

###############################################################
# Step 3: Combine to form the mixture sample
###############################################################

# Mixture sample:
# If u[i] = 1 → take z[i] (the standard normal)
# If u[i] = 0 → take z1[i] (the shifted normal)
x_sample <- u * z + (1 - u) * z1

###############################################################
# Step 4: Plot histogram of simulated mixture
###############################################################

hist(
  x_sample,
  xlab = "x",
  xlim = c(-5, 5),
  freq = FALSE,        # Use density scale, not count scale
  col = "green",
  breaks = 50,
  main = "Mixture of Two Normal Distributions"
)

###############################################################
# Step 5: Define the TRUE mixture density function
###############################################################

mix <- function(x) {
  e  * dnorm(x) +                  # Density of N(0,1)
    (1 - e) * dnorm(x, mean = m, sd = s)   # Density of N(m,s)
}

###############################################################
# Step 6: Overlay the true density curve
###############################################################

# Define x-values for plotting the true curve
xplot <- (-50:50) / 10   # Sequence from -5 to 5 in steps of 0.1

# Add the true mixture density curve on top of the histogram
par(new = TRUE)      # Allow overlay on the same plot

plot(
  xplot,
  mix(xplot),
  xlim = c(-5, 5),
  type = "l",         # Line plot
  yaxt = "n",         # Hide the y-axis to avoid double axis
  ylab = "",          # Remove label so it doesn't overwrite histogram label
  main = ""
)


