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

# Create two variables: glucose_level and glucose_unit
# Use paste() to display them together

glucose_level <- 5.8
glucose_unit <-  "mmol/L"

glucose_result <- paste(glucose_level, glucose_unit )
print(glucose_result )





# ================================================
# RetailTech Basic Data Analysis - SOLUTION
# Module 2 - Lab 1: Variables and Types
# ================================================

# ================================================
# Activity 1: Creating Basic Sales Data
# ================================================

# Practice 1.1: Create sales variables
today_sale <- 150.75
customer_number <- 202

# Practice 1.2: Print your variables
print("Sales Data:")
print(today_sale)      # Should output: 150.75
print(customer_number) # Should output: 202

# Check data types
print("Data Types:")
print(class(today_sale))      # Should output: "numeric"
print(class(customer_number)) # Should output: "numeric"


# ================================================
# Activity 2: Working with Text Data
# ================================================

# Practice 2.1: Create location and payment variables
store_location <- "Downtown"
payment_method <- "Credit"

# Practice 2.2: Print your text variables
print("Store Information:")
print(store_location)  # Should output: "Downtown"
print(payment_method)  # Should output: "Credit"

# Check data types
print("Text Data Types:")
print(class(store_location))  # Should output: "character"
print(class(payment_method))  # Should output: "character"


# ================================================
# Activity 3: Basic Calculations
# ================================================

# Practice 3.1: Create price and calculate discount
base_price <- 75.00
discount_percent <- 0.15  # 15% discount
discount_amount <- base_price * discount_percent
final_price <- base_price - discount_amount

# Practice 3.2: Print all values
print("Price Calculations:")
print(paste("Original Price:", base_price))    # Should output: 75.00
print(paste("Discount Amount:", discount_amount))  # Should output: 11.25
print(paste("Final Price:", final_price))      # Should output: 63.75

# Verify calculations
print("Calculation Check:")
expected_final <- 75.00 * (1 - 0.15)
print(paste("Expected:", expected_final))
print(paste("Calculated:", final_price))
print(paste("Calculation Correct:", 
           abs(expected_final - final_price) < 0.01))



# ================================================
# Activity 4: Type Conversions
# ================================================

# Practice 4.1: Convert values
# Convert order_id (1001) to text and store as order_text
# Convert price ("19.99") to number and store as price_number
order_id <- 1001
order_text <- as.character(order_id)

price_text <- "19.99"
price_number <- as.numeric(price_text)

# Practice 4.2: Print and verify types
# Print each converted value
# Use class() to check the data type
print("Type Conversions:")
print("Order ID Conversion:")
print(order_text)          # Should output: "1001"
print(class(order_text))   # Should output: "character"

print("Price Conversion:")
print(price_number)        # Should output: 19.99
print(class(price_number)) # Should output: "numeric"

# Create a summary list for Q2 FinTech performance
q2_summary <- list(
  users_onboarded = 16500,
  churn_rates = c(0.03, 0.027, 0.025),
  monthly_revenue = c(125000, 138000, 145000),
  goal_met = TRUE
)

# Access key metrics
q2_summary[["monthly_revenue"]]  # View revenue for each month
q2_summary[["goal_met"]]         # See if the team hit the goal


#A Matrix is a 2 dimensional array
sales_matrix <- matrix(
    c(12000, 15000, 17000, 18000, 13000, 19000),
    nrow = 2, 
    ncol = 3
)

i <- matrix(data = dataVec, nrow = nrow, ncol = ncol, byrow = byrow)
imax <- matrix(1:9 , nrow =3)
imax
# This creates a 2-row, 3-column matrix:
# Row 1: 12000, 17000, 13000
# Row 2: 15000, 18000, 19000

# Sales from row 2, column 1
sales_matrix[2,1] # returns 15000



employees <- data.frame(
    name = c("James", "Anita", "Eva"),
    age = c(28, 34, 41),
    department = c("Sales", "Marketing", "R&D"),
    fulltime = c(TRUE, TRUE, FALSE)
)

employees$name # returns "James", "Anita", "Eva"
employees[2, "department"] # "Marketing"


# ================================================
# TechMart Inventory Analysis - SOLUTION
# Lab: Working with Data Structures
# ================================================

# ================================================
# Activity 1: Product Information with Vectors
# ================================================

# Practice 1.1: Create product price vector
prices <- c(899.99, 24.99, 79.99, 299.99, 149.99)
print(prices)
# Expected output: [1] 899.99  24.99  79.99 299.99 149.99

# Practice 1.2: Create product category vector
categories <- c("Electronics", "Accessories", "Accessories", 
                "Electronics", "Accessories")
print(categories)
# Expected output: [1] "Electronics"  "Accessories" "Accessories" ...

# Testing vectors
length(prices)     # Should be 5
length(categories) # Should be 5


# ================================================
# Activity 2: Grouping Data with Lists
# ================================================

# Practice 2.1: Create a product list
mouse_product <- list(
    product_id = 201,
    name = "Premium Mouse",
    price = 24.99,
    available = TRUE
)
print(mouse_product)

# Practice 2.2: Access and modify list elements
print(mouse_product$price)  # Output: 24.99
mouse_product$available <- FALSE
print(mouse_product$available)  # Output: FALSE


# ================================================
# Activity 3: Inventory Tracking with Matrices
# ================================================

# Practice 3.1: Create inventory matrix with labels
store_inventory <- matrix(
    c(4, 10, 10,   # Keyboard
      5, 1, 90,    # Mouse
      56, 94, 45), # Monitor
    nrow = 3,
    byrow = TRUE
)
rownames(store_inventory) <- c("Keyboard", "Mouse", "Monitor")
colnames(store_inventory) <- c("Downtown", "Mall", "Airport")

# Practice 3.2: Print inventory matrix
print(store_inventory)



# ================================================
# Activity 4: Complete Product Database with Data Frames
# ================================================

# Practice 4.1: Create product database
product_db <- data.frame(
    ID = c(101, 102, 103, 104, 105),
    Name = c("Gaming Laptop", "Premium Mouse", "Wireless Keyboard", 
             "Gaming Monitor", "Mechanical Keyboard"),
    Price = prices,
    Category = categories,
    InStock = c(TRUE, TRUE, FALSE, TRUE, TRUE)
)
print(product_db)

# Practice 4.2: Change the variable values of InStock to all be TRUE
product_db$InStock <- TRUE

# ================================================
# Common Issues and Solutions
# ================================================
# Problem: Matrix data not in correct layout
# Solution: Check matrix creation parameters
example_wrong <- matrix(1:6, nrow=2)  # Without byrow=TRUE
example_right <- matrix(1:6, nrow=2, byrow=TRUE)

# Problem: Mixed data types in matrix
# Solution: Convert to consistent type or use data frame
numeric_matrix <- matrix(1:4, nrow=2)  # Works
# mixed_matrix <- matrix(c(1, "a", 2, "b"), nrow=2)  # Would cause issues

print("Solution code execution completed successfully!")



