# Install and load tidyverse (only need install once)
install.packages("tidyverse")
require(tidyverse)

# ----------------------------------------------------------
# STARWARS DATA EXERCISE
# ----------------------------------------------------------

# Select variables, keep only Humans, remove missing values,
# convert height to meters, calculate BMI, group by gender,
# and compute average BMI
starwars %>% 
  select(gender, mass, height, species) %>% 
  filter(species == "Human") %>% 
  na.omit() %>% 
  mutate(height = height / 100) %>%             # convert cm → m
  mutate(BMI = mass / height^2) %>%             # BMI formula
  group_by(gender) %>% 
  summarise(Average_BMI = mean(BMI))


# ----------------------------------------------------------
# MSLEEP DATA EXERCISES
# ----------------------------------------------------------

# View msleep dataset
View(msleep)

# Filter animals that sleep more than 18 hours
my_data <- msleep %>% 
  select(name, sleep_total) %>% 
  filter(sleep_total > 18)


# Select primates (filter error corrected: order > 20 is invalid)
hello <- msleep %>% 
  select(name, order, bodywt, sleep_total) %>% 
  filter(order == "Primates")
hello

# Filter using OR condition
msleep %>% 
  select(name, order, bodywt, sleep_total) %>% 
  filter(order == "Primates" | order > 20)  # note: order is a category, not numeric


# Filter by specific animal names using OR
msleep %>% 
  select(name, order, bodywt, sleep_total) %>% 
  filter(name == "Cow" | name == "Dog" | name == "Goat")


# More efficient version using %in%
msleep %>% 
  select(name, order, bodywt, sleep_total) %>% 
  filter(name %in% c("Dog", "Goat", "Cow", "Horse"))


# Filter using numeric range: animals with sleep_total between 18 and 20 hours
msleep %>% 
  select(name, order, bodywt, sleep_total) %>% 
  filter(between(sleep_total, 18, 20))


# Filter animals with sleep_total ~ 17 (within ±0.5)
msleep %>% 
  select(name, conservation, order, bodywt, sleep_total) %>% 
  filter(near(sleep_total, 17, tol = 0.5))


# Filter animals with missing conservation values
msleep %>% 
  select(name, order, conservation, bodywt, sleep_total) %>% 
  filter(is.na(conservation))

# Filter animals with conservation values (not missing)
msleep %>% 
  select(name, order, conservation, bodywt, sleep_total) %>% 
  filter(!is.na(conservation))


# Group by diet type (vore), remove missing values, and summarise
msleep %>% 
  drop_na(sleep_rem, vore) %>% 
  group_by(vore) %>% 
  summarise(
    "Average Total Sleep" = mean(sleep_total),
    "Max REM sleep" = max(sleep_rem)
  ) %>% 
  View()


# ----------------------------------------------------------
# EXPLORING YOUR DATA
# ----------------------------------------------------------

library(tidyverse)

?starwars          # Open documentation
dim(starwars)      # Dimensions (rows, columns)
str(starwars)      # Structure of data
glimpse(starwars)  # Compact alternative to str()
attach(starwars)   # Attach variables to search path (use carefully)
names(starwars)    # Column names
length(starwars)   # Number of variables (columns)

# END OF COMMENTED CODE
