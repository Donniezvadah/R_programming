# Modules and Functions

## Table of Contents
1. [Writing Robust Functions](#writing-robust-functions)
2. [Function Design Principles](#function-design-principles)
3. [Error Handling](#error-handling)
4. [Functional Programming](#functional-programming)
5. [S3 and S4 Object Systems](#s3-and-s4-object-systems)
6. [Package Structure](#package-structure)
7. [Testing with testthat](#testing-with-testthat)
8. [Debugging Strategies](#debugging-strategies)
9. [Code Organization](#code-organization)
10. [Best Practices](#best-practices)

---

## Writing Robust Functions

### Basic Function Structure

```r
# Simple function
greet <- function(name) {
  paste("Hello,", name)
}

# Function with default arguments
power <- function(x, exponent = 2) {
  x^exponent
}

# Function with multiple returns
calculate_stats <- function(x) {
  list(
    mean = mean(x),
    median = median(x),
    sd = sd(x),
    range = range(x)
  )
}

# Function with validation
safe_divide <- function(x, y) {
  if (!is.numeric(x) || !is.numeric(y)) {
    stop("Both arguments must be numeric")
  }
  if (y == 0) {
    warning("Division by zero, returning Inf")
    return(Inf)
  }
  x / y
}
```

### Documentation with roxygen2

```r
#' Calculate Circle Area
#'
#' This function calculates the area of a circle given its radius.
#'
#' @param radius Numeric value representing the circle's radius (must be positive)
#' @return Numeric value representing the circle's area
#' @export
#' @examples
#' circle_area(5)
#' circle_area(10)
#' @seealso \code{\link{circle_circumference}}
circle_area <- function(radius) {
  if (radius < 0) {
    stop("Radius must be non-negative")
  }
  pi * radius^2
}

#' @describeIn circle_area Calculate circle circumference
#' @export
circle_circumference <- function(radius) {
  if (radius < 0) {
    stop("Radius must be non-negative")
  }
  2 * pi * radius
}
```

### Input Validation

```r
library(assertthat)

# Using assertthat
analyze_data <- function(data, column) {
  assert_that(is.data.frame(data))
  assert_that(is.string(column))
  assert_that(column %in% names(data))
  
  summary(data[[column]])
}

# Using stopifnot
analyze_data2 <- function(data, column) {
  stopifnot(
    is.data.frame(data),
    is.character(column),
    length(column) == 1,
    column %in% names(data)
  )
  
  summary(data[[column]])
}

# Custom validation
validate_age <- function(age) {
  if (!is.numeric(age)) {
    stop("Age must be numeric", call. = FALSE)
  }
  if (age < 0 || age > 150) {
    stop("Age must be between 0 and 150", call. = FALSE)
  }
  TRUE
}
```

---

## Function Design Principles

### Single Responsibility Principle

```r
# Bad: Function does too many things
process_data <- function(file) {
  data <- read.csv(file)
  data <- data[complete.cases(data), ]
  data$new_col <- data$col1 * data$col2
  model <- lm(y ~ x, data = data)
  plot(data$x, data$y)
  return(model)
}

# Good: Separate functions for each task
load_data <- function(file) {
  read.csv(file)
}

clean_data <- function(data) {
  data[complete.cases(data), ]
}

transform_data <- function(data) {
  data$new_col <- data$col1 * data$col2
  data
}

fit_model <- function(data) {
  lm(y ~ x, data = data)
}

plot_data <- function(data) {
  plot(data$x, data$y)
}

# Use pipeline
process_data <- function(file) {
  load_data(file) %>%
    clean_data() %>%
    transform_data() %>%
    fit_model()
}
```

### Pure Functions

```r
# Pure function (no side effects, same input = same output)
calculate_total <- function(prices, tax_rate = 0.1) {
  prices * (1 + tax_rate)
}

# Impure function (has side effects)
counter <- 0
increment_counter <- function() {
  counter <<- counter + 1  # Modifies global state
  counter
}

# Better: Return value instead of modifying global state
increment <- function(x) {
  x + 1
}
```

### Function Composition

```r
# Compose functions
add_one <- function(x) x + 1
multiply_by_two <- function(x) x * 2
square <- function(x) x^2

# Manual composition
result <- square(multiply_by_two(add_one(5)))  # ((5+1)*2)^2 = 144

# Using pipe
library(magrittr)
result <- 5 %>% add_one() %>% multiply_by_two() %>% square()

# Function factory
make_multiplier <- function(n) {
  function(x) x * n
}

multiply_by_three <- make_multiplier(3)
multiply_by_five <- make_multiplier(5)

multiply_by_three(10)  # 30
multiply_by_five(10)   # 50
```

---

## Error Handling

### tryCatch

```r
# Basic tryCatch
safe_log <- function(x) {
  tryCatch(
    {
      log(x)
    },
    error = function(e) {
      message("Error: ", e$message)
      return(NA)
    },
    warning = function(w) {
      message("Warning: ", w$message)
      return(log(x))
    }
  )
}

safe_log(-1)  # Warning, returns NaN
safe_log("a") # Error, returns NA

# Advanced error handling
read_data_safely <- function(file) {
  tryCatch(
    {
      data <- read.csv(file)
      message("Successfully loaded ", nrow(data), " rows")
      return(data)
    },
    error = function(e) {
      stop("Failed to read file: ", e$message, call. = FALSE)
    },
    warning = function(w) {
      message("Warning occurred: ", w$message)
      return(NULL)
    },
    finally = {
      message("Read operation completed")
    }
  )
}
```

### Custom Error Classes

```r
# Define custom error
validation_error <- function(message, field = NULL) {
  err <- list(
    message = message,
    field = field,
    call = sys.call(-1)
  )
  class(err) <- c("validation_error", "error", "condition")
  err
}

# Use custom error
validate_user <- function(user) {
  if (!"name" %in% names(user)) {
    stop(validation_error("Name is required", field = "name"))
  }
  if (!"age" %in% names(user)) {
    stop(validation_error("Age is required", field = "age"))
  }
  if (user$age < 0) {
    stop(validation_error("Age must be positive", field = "age"))
  }
  TRUE
}

# Handle custom error
tryCatch(
  validate_user(list(name = "Alice", age = -5)),
  validation_error = function(e) {
    message("Validation failed in field '", e$field, "': ", e$message)
  }
)
```

### purrr's safely and possibly

```r
library(purrr)

# safely() returns list(result, error)
safe_log <- safely(log)
safe_log(10)   # $result = 2.302585, $error = NULL
safe_log("a")  # $result = NULL, $error = <error>

# Use with map
results <- map(list(10, -1, "a"), safe_log)
results <- transpose(results)
results$result  # List of results
results$error   # List of errors

# possibly() returns default value on error
safe_log2 <- possibly(log, otherwise = NA)
safe_log2(10)  # 2.302585
safe_log2("a") # NA

# quietly() captures messages, warnings, and output
quiet_log <- quietly(log)
quiet_log(-1)
```

---

## Functional Programming

### Higher-Order Functions

```r
# Functions that take functions as arguments
apply_twice <- function(f, x) {
  f(f(x))
}

apply_twice(sqrt, 256)  # sqrt(sqrt(256)) = 4

# Functions that return functions
make_power <- function(n) {
  function(x) x^n
}

square <- make_power(2)
cube <- make_power(3)

square(5)  # 25
cube(5)    # 125
```

### Closures

```r
# Closure: function + environment
make_counter <- function() {
  count <- 0
  
  list(
    increment = function() {
      count <<- count + 1
      count
    },
    decrement = function() {
      count <<- count - 1
      count
    },
    get = function() {
      count
    },
    reset = function() {
      count <<- 0
    }
  )
}

counter <- make_counter()
counter$increment()  # 1
counter$increment()  # 2
counter$get()        # 2
counter$decrement()  # 1
counter$reset()      # 0
```

### Map, Reduce, Filter

```r
library(purrr)

# Map: apply function to each element
numbers <- 1:5
map_dbl(numbers, ~ .x^2)  # c(1, 4, 9, 16, 25)

# Reduce: combine elements
reduce(numbers, `+`)   # 15 (sum)
reduce(numbers, `*`)   # 120 (product)

# Accumulate: cumulative reduce
accumulate(numbers, `+`)  # c(1, 3, 6, 10, 15)

# Filter (keep/discard)
keep(numbers, ~ .x %% 2 == 0)     # c(2, 4)
discard(numbers, ~ .x %% 2 == 0)  # c(1, 3, 5)

# Predicate functions
every(numbers, ~ .x > 0)   # TRUE
some(numbers, ~ .x > 3)    # TRUE
none(numbers, ~ .x > 10)   # TRUE
```

---

## S3 and S4 Object Systems

### S3 Classes (Simple)

```r
# Create S3 object
person <- function(name, age) {
  structure(
    list(name = name, age = age),
    class = "person"
  )
}

# Create instance
alice <- person("Alice", 25)
class(alice)  # "person"

# Generic function
print.person <- function(x, ...) {
  cat("Person:", x$name, "(Age:", x$age, ")\n")
}

print(alice)  # Uses print.person method

# Additional methods
summary.person <- function(object, ...) {
  cat("Name:", object$name, "\n")
  cat("Age:", object$age, "\n")
  cat("Adult:", object$age >= 18, "\n")
}

# Validator
validate_person <- function(x) {
  if (!is.character(x$name) || length(x$name) != 1) {
    stop("name must be a single string")
  }
  if (!is.numeric(x$age) || length(x$age) != 1 || x$age < 0) {
    stop("age must be a single positive number")
  }
  x
}

person <- function(name, age) {
  x <- structure(
    list(name = name, age = age),
    class = "person"
  )
  validate_person(x)
}
```

### S4 Classes (Formal)

```r
# Define S4 class
setClass("Employee",
  slots = c(
    name = "character",
    age = "numeric",
    salary = "numeric",
    department = "character"
  ),
  prototype = list(
    name = NA_character_,
    age = NA_real_,
    salary = NA_real_,
    department = NA_character_
  )
)

# Create instance
emp <- new("Employee",
  name = "Bob",
  age = 30,
  salary = 50000,
  department = "Engineering"
)

# Accessor methods
setGeneric("getName", function(obj) standardGeneric("getName"))
setMethod("getName", "Employee", function(obj) obj@name)

setGeneric("getSalary", function(obj) standardGeneric("getSalary"))
setMethod("getSalary", "Employee", function(obj) obj@salary)

# Show method
setMethod("show", "Employee", function(object) {
  cat("Employee:", object@name, "\n")
  cat("Age:", object@age, "\n")
  cat("Salary: $", object@salary, "\n")
  cat("Department:", object@department, "\n")
})

# Validation
setValidity("Employee", function(object) {
  if (object@age < 18) {
    return("Age must be at least 18")
  }
  if (object@salary < 0) {
    return("Salary must be positive")
  }
  TRUE
})
```

---

## Package Structure

### Basic Package Layout

```r
# Create package
library(usethis)
create_package("mypackage")

# Package structure:
# mypackage/
# ├── R/                  # R code
# │   ├── functions.R
# │   └── utils.R
# ├── man/                # Documentation
# ├── tests/              # Tests
# │   └── testthat/
# ├── data/               # Data files
# ├── vignettes/          # Long-form documentation
# ├── DESCRIPTION         # Package metadata
# ├── NAMESPACE           # Exports
# └── README.md           # README
```

### DESCRIPTION File

```
Package: mypackage
Title: My Awesome Package
Version: 0.1.0
Authors@R: person("Your", "Name", email = "you@example.com",
                  role = c("aut", "cre"))
Description: This package does amazing things with data.
License: MIT + file LICENSE
Encoding: UTF-8
LazyData: true
Roxygen: list(markdown = TRUE)
RoxygenNote: 7.2.0
Imports:
    dplyr (>= 1.0.0),
    ggplot2
Suggests:
    testthat (>= 3.0.0),
    knitr,
    rmarkdown
VignetteBuilder: knitr
```

### Adding Functions

```r
# Create new function file
use_r("my_function")

# In R/my_function.R
#' My Function
#'
#' @param x A numeric vector
#' @return The mean of x
#' @export
my_function <- function(x) {
  mean(x, na.rm = TRUE)
}

# Generate documentation
devtools::document()

# Load package for testing
devtools::load_all()
```

---

## Testing with testthat

### Setup Testing

```r
# Setup testthat
usethis::use_testthat()

# Create test file
usethis::use_test("my_function")
```

### Writing Tests

```r
# tests/testthat/test-my_function.R
library(testthat)

test_that("my_function calculates mean correctly", {
  expect_equal(my_function(c(1, 2, 3)), 2)
  expect_equal(my_function(c(10, 20, 30)), 20)
})

test_that("my_function handles NA values", {
  expect_equal(my_function(c(1, NA, 3)), 2)
  expect_false(is.na(my_function(c(NA, NA, NA))))
})

test_that("my_function validates input", {
  expect_error(my_function("not numeric"))
  expect_error(my_function(NULL))
})

test_that("my_function handles edge cases", {
  expect_equal(my_function(c()), NaN)
  expect_equal(my_function(5), 5)
})
```

### Common Expectations

```r
# Equality
expect_equal(2 + 2, 4)
expect_identical(2L, 2L)
expect_equivalent(c(a = 1), 1)  # Ignores attributes

# Comparison
expect_gt(5, 3)   # Greater than
expect_lt(3, 5)   # Less than
expect_gte(5, 5)  # Greater than or equal
expect_lte(3, 5)  # Less than or equal

# Types
expect_type("hello", "character")
expect_s3_class(lm(y ~ x, data = df), "lm")
expect_s4_class(obj, "Employee")

# Logical
expect_true(2 > 1)
expect_false(2 < 1)

# Errors and warnings
expect_error(log("a"))
expect_warning(log(-1))
expect_message(message("hello"))
expect_silent(2 + 2)

# Matching
expect_match("hello world", "world")
expect_match("test123", "\\d+")  # Regex

# Length
expect_length(1:5, 5)
expect_named(c(a = 1, b = 2), c("a", "b"))
```

### Testing Best Practices

```r
# Arrange-Act-Assert pattern
test_that("calculate_total adds tax correctly", {
  # Arrange
  price <- 100
  tax_rate <- 0.1
  
  # Act
  result <- calculate_total(price, tax_rate)
  
  # Assert
  expect_equal(result, 110)
})

# Test fixtures
test_that("function works with sample data", {
  # Load test data
  test_data <- read.csv("tests/testthat/fixtures/sample_data.csv")
  
  result <- my_analysis(test_data)
  
  expect_equal(nrow(result), 100)
})

# Snapshot testing
test_that("output matches snapshot", {
  expect_snapshot(my_complex_function(data))
})
```

---

## Debugging Strategies

### Basic Debugging Tools

```r
# Print debugging
my_function <- function(x) {
  print(paste("x =", x))  # Debug print
  result <- x * 2
  print(paste("result =", result))  # Debug print
  result
}

# browser() - interactive debugger
my_function <- function(x) {
  y <- x * 2
  browser()  # Execution stops here
  z <- y + 1
  z
}

# Commands in browser:
# n - next line
# s - step into function
# c - continue execution
# Q - quit
# ls() - list variables
```

### Advanced Debugging

```r
# debug() - debug entire function
debug(my_function)
my_function(5)  # Enters debugger
undebug(my_function)

# debugonce() - debug once
debugonce(my_function)

# trace() - add code to function
trace(mean, quote(print(x)))
mean(1:5)  # Prints x before calculating
untrace(mean)

# recover() - debug on error
options(error = recover)
my_buggy_function()  # Opens debugger on error
options(error = NULL)  # Reset
```

### Profiling

```r
# Profile code
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
  optimized = rowSums(matrix(1:1000000, ncol = 100)),
  times = 100
)
```

---

## Code Organization

### Project Structure

```r
# Recommended structure
# project/
# ├── R/                 # R code
# │   ├── 01_load_data.R
# │   ├── 02_clean_data.R
# │   ├── 03_analyze.R
# │   └── utils.R
# ├── data/              # Data files
# │   ├── raw/
# │   └── processed/
# ├── output/            # Results
# │   ├── figures/
# │   └── tables/
# ├── tests/             # Tests
# ├── docs/              # Documentation
# ├── config.R           # Configuration
# ├── main.R             # Main script
# └── README.md
```

### Sourcing Files

```r
# Source all R files in directory
source_files <- list.files("R", pattern = "\\.R$", full.names = TRUE)
invisible(lapply(source_files, source))

# Or use a package
library(here)
source(here("R", "utils.R"))
```

---

## Best Practices

### Code Style

```r
# Use styler for automatic formatting
library(styler)
style_file("R/my_file.R")
style_dir("R/")

# Check style with lintr
library(lintr)
lint("R/my_file.R")
```

### Performance Tips

```r
# Pre-allocate vectors
# Slow
result <- c()
for (i in 1:10000) {
  result <- c(result, i^2)
}

# Fast
result <- numeric(10000)
for (i in 1:10000) {
  result[i] <- i^2
}

# Best: vectorize
result <- (1:10000)^2

# Use appropriate data structures
# Slow: repeated subsetting
for (i in 1:nrow(df)) {
  df[i, "new_col"] <- df[i, "old_col"] * 2
}

# Fast: vectorized
df$new_col <- df$old_col * 2
```

---

## Summary

### Skills Learned

- ✅ Writing robust, documented functions
- ✅ Error handling and validation
- ✅ Functional programming patterns
- ✅ S3 and S4 object systems
- ✅ Package structure and development
- ✅ Comprehensive testing with testthat
- ✅ Debugging and profiling
- ✅ Code organization and style

### Next Steps

1. **Practice:** Write your own R package
2. **Test:** Achieve 80%+ code coverage
3. **Share:** Publish on GitHub or CRAN
4. **Learn:** Continue with `6_R_Developer_Tools.md`

**Continue to Developer Tools! 🚀**
