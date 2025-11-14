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

length(hair_color)
unique(hair_color)
View(sort(table(hair_color), decreasing = TRUE))
plot(sort(table(eye_color), decreasing = TRUE))


#Functions in R

multiply <- function(x,y) 
{
  return(x*y)
}
multiply(3,5)


# Assigning numbers
a <- 25             # whole number
b <- 4.987          # decimal number

# Checking the type
typeof(a)           # returns "double"
typeof(b)           # returns "double"

# Explicitly creating an integer
c <- 25L            # 'L' makes R treat it specifically as integer
typeof(c)           # returns "integer"

# Character examples
message1 <- "Learning R is fun!"
message2 <- 'Let’s master R together.'

# Confirming data type
typeof(message1)      # returns "character"

# Combining text (concatenation)
paste(message1, message2) # Results in "Learning R is fun! Let’s master R together."


# Logical examples
task_completed <- TRUE
task_failed <- FALSE

typeof(task_completed)  # returns "logical"

# Logical operations
task_completed & task_failed # logical "AND"; returns FALSE 
task_completed | task_failed # logical "OR"; returns TRUE

# Some example variables
x <- 42
name <- "Alex"
is_sunny <- FALSE

# Check data types
is.numeric(x)       # Returns TRUE, because 42 is numeric
is.character(name)  # Returns TRUE, because "Alex" is character text
is.logical(is_sunny)# Returns TRUE, FALSE is logical

# Original variables
x <- 42
name <- "Alex"
is_sunny <- FALSE

# Convert numeric to character
x_as_text <- as.character(x)
x_as_text            # "42"
typeof(x_as_text)    # "character"

name <- "Alex"
name_as_number <- as.numeric(name) # R flags a warning
name_as_number                     # returns NA

number_text <- "56.7"
number_converted <- as.numeric(number_text)
number_converted          # 56.7 (successful conversion)
is.na(number_converted)   # FALSE indicates success, no issues

memory.size() # Checks current memory usage
gc()          # Manually triggers memory cleanup

# Numeric and character combined leads to characters
mixed_vector <- c(12, "data", 50)

typeof(mixed_vector) # returns "character"
mixed_vector         # returns "12", "data", "50"

number_as_text <- "42"
number_converted <- as.numeric(number_as_text)
typeof(number_converted) # returns "double"

my_data <- c(5, NA, 7)
is.na(my_data)  # returns FALSE TRUE FALSE
sum(my_data, na.rm=TRUE) # returns 12 (ignores NA)

empty_var <- NULL
is.null(empty_var)    # TRUE
length(empty_var)     # 0
length(NA)            # 1 (NA counts as a placeholder still occupying space)

undefined_result <- 0/0
is.nan(undefined_result) # TRUE




