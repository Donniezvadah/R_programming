# Beginner Level R

## Table of Contents
1. [Introduction to R](#introduction-to-r)
2. [Installing R and RStudio](#installing-r-and-rstudio)
3. [Basic Syntax and Operations](#basic-syntax-and-operations)
4. [Variables and Data Types](#variables-and-data-types)
5. [Vectors](#vectors)
6. [Matrices and Arrays](#matrices-and-arrays)
7. [Data Frames](#data-frames)
8. [Lists](#lists)
9. [Control Structures](#control-structures)
10. [Functions](#functions)
11. [Basic Plotting](#basic-plotting)
12. [Reading and Writing Data](#reading-and-writing-data)
13. [Practice Exercises](#practice-exercises)

---

## Introduction to R

## This is a Live session

### What is R?

R is a powerful programming language specifically designed for:
- Statistical computing and data analysis
- Data visualization and graphics
- Machine learning and predictive modeling
- Reproducible research and reporting

**Key Features:**
- ✅ Free and open-source
- ✅ 18,000+ packages on CRAN
- ✅ Excellent data visualization (ggplot2)
- ✅ Strong statistical capabilities
- ✅ Active global community
- ✅ Industry standard in data science

### Why Learn R?

**Career Opportunities:**
- Data Scientist
- Statistical Analyst
- Bioinformatician
- Financial Analyst
- Research Scientist

**Industries Using R:**
- Tech (Google, Facebook, Microsoft)
- Finance (banks, hedge funds)
- Healthcare and pharmaceuticals
- Academia and research
- Government agencies

---

## Installing R and RStudio

### Step 1: Install R

**Windows:**
```bash
1. Visit https://cran.r-project.org/
2. Click "Download R for Windows" → "base"
3. Download and run the .exe installer
4. Follow installation wizard
```

**macOS:**
```bash
1. Visit https://cran.r-project.org/
2. Click "Download R for macOS"
3. Download the .pkg file for your system
4. Open and install
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install r-base r-base-dev
```

### Step 2: Install RStudio

1. Visit https://posit.co/download/rstudio-desktop/
2. Download RStudio Desktop (Free)
3. Install for your operating system

### Verify Installation

```r
# Check R version
R.version.string
# Output: "R version 4.3.0 (2023-04-21)"

# Check working directory
getwd()

# Set working directory
setwd("/path/to/your/folder")
```

### RStudio Interface

**Four Main Panes:**

1. **Source Editor** (Top-Left)
   - Write and edit R scripts (.R files)
   - Create R Markdown documents
   - View data tables

2. **Console** (Bottom-Left)
   - Execute R commands interactively
   - View output and messages
   - See errors and warnings

3. **Environment/History** (Top-Right)
   - View loaded objects and variables
   - Browse command history
   - Import datasets

4. **Files/Plots/Packages/Help** (Bottom-Right)
   - Navigate project files
   - View generated plots
   - Manage packages
   - Access documentation

**Essential Shortcuts:**

| Action | Windows/Linux | macOS |
|--------|---------------|-------|
| Run line/selection | Ctrl + Enter | Cmd + Enter |
| Assignment `<-` | Alt + - | Option + - |
| Comment code | Ctrl + Shift + C | Cmd + Shift + C |
| Save script | Ctrl + S | Cmd + S |
| Clear console | Ctrl + L | Cmd + L |
| Pipe `%>%` | Ctrl + Shift + M | Cmd + Shift + M |

---

## Basic Syntax and Operations

### R as a Calculator

```r
# Arithmetic operations
5 + 3          # Addition: 8
10 - 4         # Subtraction: 6
6 * 7          # Multiplication: 42
20 / 4         # Division: 5
2^3            # Exponentiation: 8
2**3           # Alternative exponentiation: 8
17 %% 5        # Modulo (remainder): 2
17 %/% 5       # Integer division: 3

# Order of operations (PEMDAS)
2 + 3 * 4      # Result: 14
(2 + 3) * 4    # Result: 20

# Mathematical functions
sqrt(16)       # Square root: 4
abs(-10)       # Absolute value: 10
log(10)        # Natural log: 2.302585
log10(100)     # Base 10 log: 2
exp(1)         # e^1: 2.718282
sin(pi/2)      # Sine: 1
cos(0)         # Cosine: 1
tan(pi/4)      # Tangent: 1
round(3.14159, 2)  # Round: 3.14
ceiling(3.2)   # Round up: 4
floor(3.8)     # Round down: 3
```

### Comments and Documentation

```r
# Single-line comment

# Multiple lines of comments
# Use # at the start of each line

x <- 5  # Inline comment

# Good commenting practice
# Calculate the average of test scores
scores <- c(85, 92, 78, 95, 88)
average_score <- mean(scores)  # Use built-in mean function
```

### Getting Help

```r
# Access function documentation
?mean
help(mean)

# Search for topics
??"linear regression"
help.search("regression")

# Get examples
example(mean)

# View function arguments
args(plot)

# See available methods
methods(plot)

# Find package documentation
help(package = "dplyr")
```

---

## Variables and Data Types

### Creating Variables

```r
# Assignment operator: <- (preferred)
x <- 10
y <- 5
name <- "Alice"

# Alternative: = (works but <- is R convention)
x = 10

# Print variable
print(x)
x  # Auto-print

# Multiple assignments
a <- b <- c <- 0

# Chained operations
result <- (x + y) * 2
```

### Variable Naming Rules

```r
# Valid names
my_var <- 1
myVar <- 2
my.var <- 3
my_var_123 <- 4
.hidden <- 5  # Starts with dot (hidden variable)

# Invalid names (will cause errors)
# 2var <- 1        # Can't start with number
# my-var <- 2      # Hyphens not allowed
# my var <- 3      # Spaces not allowed
# for <- 4         # Reserved word
```

**Best Practices:**

```r
# Use snake_case (recommended for R)
student_count <- 30
average_score <- 85.5
max_temperature <- 98.6

# Be descriptive
total_sales <- 1000000        # Good
ts <- 1000000                 # Bad (unclear)

# Use meaningful names
user_age <- 25                # Good
x <- 25                       # Bad (not descriptive)
```

### Data Types

#### 1. Numeric (Double)

```r
x <- 10.5
class(x)        # "numeric"
typeof(x)       # "double"
is.numeric(x)   # TRUE

# Scientific notation
large_num <- 1.5e6  # 1,500,000
small_num <- 3.2e-4 # 0.00032
```

#### 2. Integer

```r
y <- 5L         # L suffix creates integer
class(y)        # "integer"
typeof(y)       # "integer"
is.integer(y)   # TRUE

# Convert to integer
as.integer(10.7)  # 10 (truncates)
```

#### 3. Character (String)

```r
name <- "John Doe"
class(name)     # "character"

# Single or double quotes
greeting1 <- "Hello"
greeting2 <- 'Hello'

# String operations
paste("Hello", "World")           # "Hello World"
paste0("Hello", "World")          # "HelloWorld" (no space)
toupper("hello")                  # "HELLO"
tolower("HELLO")                  # "hello"
nchar("Hello")                    # 5 (character count)
substr("Hello World", 1, 5)      # "Hello"
```

#### 4. Logical (Boolean)

```r
is_student <- TRUE
is_employed <- FALSE

class(is_student)  # "logical"

# Logical operators
TRUE & FALSE       # AND: FALSE
TRUE | FALSE       # OR: TRUE
!TRUE              # NOT: FALSE

# Comparison operators
5 > 3              # TRUE
5 < 3              # FALSE
5 >= 5             # TRUE
5 == 5             # TRUE (equality)
5 != 3             # TRUE (not equal)
```

#### 5. Complex

```r
z <- 3 + 2i
class(z)           # "complex"
Re(z)              # Real part: 3
Im(z)              # Imaginary part: 2
Mod(z)             # Modulus: 3.605551
```

### Type Conversion

```r
# Convert between types
as.numeric("123")      # 123
as.character(123)      # "123"
as.integer(10.7)       # 10
as.logical(1)          # TRUE
as.logical(0)          # FALSE

# Failed conversion
as.numeric("abc")      # NA (with warning)

# Check for NA
is.na(NA)              # TRUE
```

### Special Values

```r
# NA: Not Available (missing value)
x <- NA
is.na(x)               # TRUE

# NULL: Empty/undefined
y <- NULL
is.null(y)             # TRUE
length(NULL)           # 0

# NaN: Not a Number
z <- 0/0
is.nan(z)              # TRUE

# Inf: Infinity
w <- 1/0
is.infinite(w)         # TRUE

# Differences
length(NA)             # 1
length(NULL)           # 0
NA + 5                 # NA
NULL + 5               # Error
```

---

## Vectors

### Creating Vectors

```r
# Using c() function (combine/concatenate)
numbers <- c(1, 2, 3, 4, 5)
names <- c("Alice", "Bob", "Charlie")
logicals <- c(TRUE, FALSE, TRUE, TRUE)

# Sequences
seq1 <- 1:10                    # 1 2 3 4 5 6 7 8 9 10
seq2 <- 10:1                    # Descending
seq3 <- seq(1, 10, by=2)        # 1 3 5 7 9
seq4 <- seq(0, 1, length.out=5) # 0.00 0.25 0.50 0.75 1.00

# Repeated values
rep1 <- rep(1, times=5)         # 1 1 1 1 1
rep2 <- rep(c(1,2), times=3)    # 1 2 1 2 1 2
rep3 <- rep(c(1,2), each=3)     # 1 1 1 2 2 2
rep4 <- rep(1:3, each=2, times=2) # 1 1 2 2 3 3 1 1 2 2 3 3
```

### Vector Properties

```r
x <- c(10, 20, 30, 40, 50)

length(x)              # 5
class(x)               # "numeric"
typeof(x)              # "double"
is.vector(x)           # TRUE
```

### Vector Arithmetic

```r
x <- c(1, 2, 3, 4, 5)
y <- c(10, 20, 30, 40, 50)

# Element-wise operations
x + y          # 11 22 33 44 55
x - y          # -9 -18 -27 -36 -45
x * y          # 10 40 90 160 250
x / y          # 0.1 0.1 0.1 0.1 0.1
x^2            # 1 4 9 16 25

# Scalar operations
x + 10         # 11 12 13 14 15
x * 2          # 2 4 6 8 10

# Vector recycling (shorter vector repeats)
c(1, 2, 3) + c(10, 20)  # 11 22 13 (with warning)
```

### Indexing and Subsetting

```r
x <- c(10, 20, 30, 40, 50)

# Positive indexing (1-based!)
x[1]           # 10 (first element)
x[3]           # 30
x[c(1, 3, 5)]  # 10 30 50
x[1:3]         # 10 20 30

# Negative indexing (exclude)
x[-1]          # 20 30 40 50
x[-c(1, 3)]    # 20 40 50
x[-(1:3)]      # 40 50

# Logical indexing
x[c(TRUE, FALSE, TRUE, FALSE, TRUE)]  # 10 30 50
x[x > 25]      # 30 40 50
x[x >= 20 & x <= 40]  # 20 30 40

# Named vectors
ages <- c(Alice=25, Bob=30, Charlie=35)
ages["Alice"]                    # 25
ages[c("Alice", "Charlie")]      # 25 35
names(ages)                      # "Alice" "Bob" "Charlie"
```

### Modifying Vectors

```r
x <- c(10, 20, 30, 40, 50)

# Replace elements
x[3] <- 99             # c(10, 20, 99, 40, 50)
x[c(1, 5)] <- 0        # Replace multiple

# Append elements
x <- c(x, 60)          # Add to end
x <- c(0, x)           # Add to beginning
x <- append(x, 100, after=3)  # Insert at position

# Remove elements
x <- x[-1]             # Remove first
x <- x[x != 30]        # Remove all 30s
```

### Vector Functions

```r
x <- c(3, 7, 2, 9, 1, 5)

# Basic statistics
length(x)      # 6
sum(x)         # 27
mean(x)        # 4.5
median(x)      # 4
min(x)         # 1
max(x)         # 9
range(x)       # 1 9
sd(x)          # Standard deviation: 2.88
var(x)         # Variance: 8.3
quantile(x)    # Quartiles

# Sorting
sort(x)                        # 1 2 3 5 7 9
sort(x, decreasing=TRUE)       # 9 7 5 3 2 1
order(x)                       # Indices for sorting
rank(x)                        # Ranks
rev(x)                         # Reverse: 5 1 9 2 7 3

# Unique and duplicates
y <- c(1, 2, 2, 3, 3, 3, 4)
unique(y)                      # 1 2 3 4
duplicated(y)                  # FALSE FALSE TRUE FALSE TRUE TRUE FALSE
table(y)                       # Frequency table

# Cumulative functions
cumsum(c(1,2,3,4))             # 1 3 6 10
cumprod(c(1,2,3,4))            # 1 2 6 24
cummin(c(3,1,4,1,5))          # 3 1 1 1 1
cummax(c(3,1,4,1,5))          # 3 3 4 4 5
```

### Vector Operations

```r
# Set operations
a <- c(1, 2, 3, 4, 5)
b <- c(4, 5, 6, 7, 8)

union(a, b)              # 1 2 3 4 5 6 7 8
intersect(a, b)          # 4 5
setdiff(a, b)            # 1 2 3
setequal(a, b)           # FALSE

# Element-wise comparison
x <- c(1, 2, 3)
y <- c(3, 2, 1)
x == y                   # FALSE TRUE FALSE
x > y                    # FALSE FALSE TRUE
all(x > 0)               # TRUE (all elements)
any(x > 2)               # TRUE (at least one)

# Which indices
x <- c(10, 25, 30, 15, 40)
which(x > 20)            # 2 3 5
which.max(x)             # 5
which.min(x)             # 1
```

---

## Matrices and Arrays

### Creating Matrices

```r
# Using matrix() function
mat1 <- matrix(1:12, nrow=3, ncol=4)
#      [,1] [,2] [,3] [,4]
# [1,]    1    4    7   10
# [2,]    2    5    8   11
# [3,]    3    6    9   12

# Fill by row
mat2 <- matrix(1:12, nrow=3, ncol=4, byrow=TRUE)
#      [,1] [,2] [,3] [,4]
# [1,]    1    2    3    4
# [2,]    5    6    7    8
# [3,]    9   10   11   12

# Combining vectors
mat3 <- rbind(c(1,2,3), c(4,5,6))  # Row bind
mat4 <- cbind(c(1,2,3), c(4,5,6))  # Column bind

# Identity matrix
diag(3)  # 3x3 identity matrix
```

### Matrix Operations

```r
A <- matrix(1:4, nrow=2, ncol=2)
B <- matrix(5:8, nrow=2, ncol=2)

# Element-wise operations
A + B         # Addition
A - B         # Subtraction
A * B         # Element-wise multiplication
A / B         # Element-wise division

# Matrix multiplication
A %*% B       # True matrix multiplication

# Transpose
t(A)          # Transpose

# Matrix functions
det(A)        # Determinant
solve(A)      # Inverse (if invertible)
eigen(A)      # Eigenvalues and eigenvectors
```

### Matrix Indexing

```r
mat <- matrix(1:12, nrow=3, ncol=4)

# Single element
mat[2, 3]     # Row 2, Column 3

# Entire row
mat[2, ]      # All of row 2

# Entire column
mat[, 3]      # All of column 3

# Submatrix
mat[1:2, 2:3]

# Named dimensions
rownames(mat) <- c("A", "B", "C")
colnames(mat) <- c("W", "X", "Y", "Z")
mat["A", "X"]
```

### Matrix Functions

```r
mat <- matrix(1:12, nrow=3, ncol=4)

nrow(mat)     # 3
ncol(mat)     # 4
dim(mat)      # c(3, 4)

rowSums(mat)  # Sum each row
colSums(mat)  # Sum each column
rowMeans(mat) # Mean of each row
colMeans(mat) # Mean of each column

# Apply function
apply(mat, 1, sum)   # Apply to rows (margin=1)
apply(mat, 2, mean)  # Apply to columns (margin=2)
```

---

## Data Frames

### Creating Data Frames

```r
# Using data.frame()
df <- data.frame(
  name = c("Alice", "Bob", "Charlie", "David"),
  age = c(25, 30, 35, 28),
  salary = c(50000, 60000, 75000, 55000),
  employed = c(TRUE, TRUE, FALSE, TRUE),
  stringsAsFactors = FALSE  # Keep strings as character
)

print(df)
View(df)  # Opens in viewer pane
```

### Exploring Data Frames

```r
# Structure and summary
str(df)       # Data structure
summary(df)   # Statistical summary
glimpse(df)   # dplyr alternative (if installed)

# Dimensions
nrow(df)      # Number of rows
ncol(df)      # Number of columns
dim(df)       # Both dimensions

# Column names
names(df)
colnames(df)

# First/last rows
head(df, 3)   # First 3 rows
tail(df, 2)   # Last 2 rows
```

### Accessing Data Frame Elements

```r
# Access columns
df$name           # $ operator (returns vector)
df[["name"]]      # [[ ]] (returns vector)
df[, "name"]      # [ ] (returns vector)
df["name"]        # [ ] (returns data frame)
df[, 1]           # By index

# Multiple columns
df[, c("name", "age")]
df[, 1:2]

# Access rows
df[1, ]           # First row
df[1:2, ]         # First two rows

# Specific cell
df[1, "age"]      # Row 1, age column
df$age[1]         # First element of age
```

### Filtering Data Frames

```r
# Logical subsetting
df[df$age > 28, ]
df[df$employed == TRUE, ]
df[df$salary >= 60000, ]

# Multiple conditions
df[df$age > 25 & df$employed == TRUE, ]
df[df$age < 26 | df$age > 30, ]

# Using subset() function
subset(df, age > 28)
subset(df, age > 25 & employed == TRUE)
subset(df, age > 28, select = c(name, salary))
```

### Modifying Data Frames

```r
# Add new column
df$bonus <- df$salary * 0.1
df$age_group <- ifelse(df$age < 30, "Young", "Senior")

# Modify existing column
df$age <- df$age + 1

# Add new row
new_row <- data.frame(
  name = "Eve",
  age = 32,
  salary = 70000,
  employed = TRUE
)
df <- rbind(df, new_row)

# Remove column
df$bonus <- NULL
df <- df[, -4]  # Remove 4th column

# Remove row
df <- df[-5, ]  # Remove 5th row
```

### Sorting Data Frames

```r
# Sort by one column
df[order(df$age), ]                    # Ascending
df[order(-df$age), ]                   # Descending
df[order(df$age, decreasing=TRUE), ]

# Sort by multiple columns
df[order(df$employed, -df$salary), ]   # employed asc, salary desc
```

---

## Lists

### Creating Lists

```r
# Lists can contain different types
my_list <- list(
  numbers = c(1, 2, 3, 4, 5),
  name = "John",
  matrix = matrix(1:4, nrow=2),
  df = data.frame(x=1:3, y=4:6),
  nested = list(a=1, b=2)
)

print(my_list)
str(my_list)
```

### Accessing List Elements

```r
# Using [[ ]] returns the element
my_list[[1]]           # First element
my_list[["numbers"]]   # By name

# Using $ returns the element
my_list$numbers

# Using [ ] returns a sublist
my_list[1]             # List containing first element
my_list[c(1, 3)]       # List with elements 1 and 3

# Nested access
my_list$nested$a
my_list[["df"]]$x
```

### Modifying Lists

```r
# Add elements
my_list$new_item <- "New value"
my_list[[6]] <- c(10, 20)

# Remove elements
my_list$new_item <- NULL
my_list[[6]] <- NULL

# List functions
length(my_list)        # Number of elements
names(my_list)         # Element names
```

---

## Control Structures

### If-Else Statements

```r
# Basic if
x <- 10
if (x > 5) {
  print("x is greater than 5")
}

# If-else
if (x > 15) {
  print("x is large")
} else {
  print("x is not large")
}

# If-else if-else
score <- 75
if (score >= 90) {
  grade <- "A"
} else if (score >= 80) {
  grade <- "B"
} else if (score >= 70) {
  grade <- "C"
} else if (score >= 60) {
  grade <- "D"
} else {
  grade <- "F"
}

# Vectorized ifelse
scores <- c(95, 82, 78, 65)
grades <- ifelse(scores >= 90, "A",
          ifelse(scores >= 80, "B",
          ifelse(scores >= 70, "C", "F")))
```

### For Loops

```r
# Basic for loop
for (i in 1:5) {
  print(i)
}

# Loop over vector
fruits <- c("apple", "banana", "cherry")
for (fruit in fruits) {
  print(paste("I like", fruit))
}

# Loop with index
for (i in seq_along(fruits)) {
  print(paste(i, ":", fruits[i]))
}

# Nested loops
for (i in 1:3) {
  for (j in 1:3) {
    print(paste(i, "*", j, "=", i*j))
  }
}
```

### While Loops

```r
# Basic while loop
count <- 1
while (count <= 5) {
  print(count)
  count <- count + 1
}

# While with condition
x <- 1
while (x < 100) {
  print(x)
  x <- x * 2
}
```

### Loop Control

```r
# break: exit loop
for (i in 1:10) {
  if (i == 5) break
  print(i)
}

# next: skip to next iteration
for (i in 1:10) {
  if (i %% 2 == 0) next  # Skip even numbers
  print(i)
}
```

---

## Functions

### Creating Functions

```r
# Basic function
greet <- function() {
  print("Hello!")
}
greet()

# Function with parameters
greet_person <- function(name) {
  print(paste("Hello,", name))
}
greet_person("Alice")

# Function with return value
add_numbers <- function(a, b) {
  result <- a + b
  return(result)
}
sum_result <- add_numbers(5, 3)

# Implicit return (last expression)
multiply <- function(a, b) {
  a * b  # Automatically returned
}
```

### Default Parameters

```r
power <- function(base, exponent = 2) {
  base^exponent
}

power(5)      # 25 (uses default)
power(5, 3)   # 125

# Multiple defaults
greet <- function(name = "Guest", time = "day") {
  paste("Good", time, name)
}
greet()                    # "Good day Guest"
greet("Alice")            # "Good day Alice"
greet("Bob", "morning")   # "Good morning Bob"
```

### Multiple Return Values

```r
# Return a list
stats <- function(x) {
  list(
    mean = mean(x),
    median = median(x),
    sd = sd(x),
    range = range(x)
  )
}

result <- stats(c(1, 2, 3, 4, 5))
result$mean    # 3
result$sd      # Standard deviation
```

### Function Documentation

```r
#' Calculate Circle Area
#'
#' @param radius Numeric value for circle radius
#' @return Area of the circle
#' @examples
#' circle_area(5)
circle_area <- function(radius) {
  if (radius < 0) {
    stop("Radius must be positive")
  }
  pi * radius^2
}
```

---

## Basic Plotting

### Scatter Plots

```r
# Basic scatter plot
x <- c(1, 2, 3, 4, 5)
y <- c(2, 4, 5, 4, 6)
plot(x, y)

# Customized scatter plot
plot(x, y,
     main = "My Scatter Plot",
     xlab = "X Axis",
     ylab = "Y Axis",
     col = "blue",
     pch = 19,    # Point type
     cex = 1.5)   # Point size
```

### Line Plots

```r
# Line plot
plot(x, y, type = "l", col = "red", lwd = 2)

# Points and lines
plot(x, y, type = "b", col = "blue")

# Multiple lines
plot(x, y, type = "l", col = "blue")
lines(x, y*1.5, col = "red")
legend("topleft", legend = c("Series 1", "Series 2"),
       col = c("blue", "red"), lty = 1)
```

### Bar Plots

```r
# Simple bar plot
counts <- c(5, 10, 15, 20)
barplot(counts)

# Named bars
names(counts) <- c("A", "B", "C", "D")
barplot(counts,
        main = "Bar Chart",
        xlab = "Category",
        ylab = "Count",
        col = "steelblue")
```

### Histograms

```r
# Generate random data
data <- rnorm(1000, mean = 50, sd = 10)

# Histogram
hist(data,
     main = "Histogram",
     xlab = "Value",
     col = "lightblue",
     breaks = 30)
```

### Box Plots

```r
# Box plot
data1 <- rnorm(100, mean = 50)
data2 <- rnorm(100, mean = 60)
data3 <- rnorm(100, mean = 55)

boxplot(data1, data2, data3,
        names = c("Group A", "Group B", "Group C"),
        main = "Box Plot Comparison",
        col = c("red", "blue", "green"))
```

### Saving Plots

```r
# Save as PDF
pdf("myplot.pdf", width = 8, height = 6)
plot(x, y)
dev.off()

# Save as PNG
png("myplot.png", width = 800, height = 600)
plot(x, y)
dev.off()
```

---

## Reading and Writing Data

### CSV Files

```r
# Read CSV
df <- read.csv("data.csv")
df <- read.csv("data.csv", header = TRUE, stringsAsFactors = FALSE)

# Write CSV
write.csv(df, "output.csv", row.names = FALSE)
```

### Text Files

```r
# Read delimited files
df <- read.table("data.txt", header = TRUE, sep = "\t")

# Write text file
write.table(df, "output.txt", sep = "\t", row.names = FALSE)
```

### Built-in Datasets

```r
# Load built-in datasets
data(mtcars)
head(mtcars)

# See available datasets
data()

# Common datasets
data(iris)
data(airquality)
data(ChickWeight)
```

### RDS Files (R Native Format)

```r
# Save R object
saveRDS(df, "mydata.rds")

# Read R object
df <- readRDS("mydata.rds")

# Save multiple objects
save(df, x, y, file = "myworkspace.RData")

# Load saved objects
load("myworkspace.RData")
```

---

## Practice Exercises

### Exercise 1: Vector Operations

```r
# Create a vector of ages
ages <- c(23, 45, 67, 89, 12, 34, 56, 78)

# Tasks:
# 1. Find mean, median, and standard deviation
# 2. Count how many ages are above 50
# 3. Create a new vector with ages below 40
# 4. Sort ages in descending order

# Solutions:
mean(ages)
median(ages)
sd(ages)
sum(ages > 50)
young <- ages[ages < 40]
sort(ages, decreasing = TRUE)
```

### Exercise 2: Data Frame Manipulation

```r
# Create a data frame of students
students <- data.frame(
  name = c("Alice", "Bob", "Charlie", "Diana", "Eve"),
  math_score = c(85, 92, 78, 95, 88),
  english_score = c(90, 85, 82, 91, 87),
  age = c(20, 21, 19, 22, 20)
)

# Tasks:
# 1. Add a column for average score
# 2. Filter students with average > 85
# 3. Sort by math_score descending
# 4. Find the student with highest total score

# Solutions:
students$avg_score <- (students$math_score + students$english_score) / 2
students[students$avg_score > 85, ]
students[order(-students$math_score), ]
students$total <- students$math_score + students$english_score
students[which.max(students$total), ]
```

### Exercise 3: Functions

```r
# Write a function that:
# 1. Takes a vector of numbers
# 2. Removes outliers (values > 2 SD from mean)
# 3. Returns cleaned vector

remove_outliers <- function(x) {
  mean_x <- mean(x, na.rm = TRUE)
  sd_x <- sd(x, na.rm = TRUE)
  lower_bound <- mean_x - 2 * sd_x
  upper_bound <- mean_x + 2 * sd_x
  x[x >= lower_bound & x <= upper_bound]
}

# Test
data <- c(1, 2, 3, 4, 5, 100, 200)
remove_outliers(data)
```

### Exercise 4: Control Structures

```r
# FizzBuzz: Print numbers 1-100, but:
# - "Fizz" for multiples of 3
# - "Buzz" for multiples of 5
# - "FizzBuzz" for multiples of both

for (i in 1:100) {
  if (i %% 15 == 0) {
    print("FizzBuzz")
  } else if (i %% 3 == 0) {
    print("Fizz")
  } else if (i %% 5 == 0) {
    print("Buzz")
  } else {
    print(i)
  }
}
```

### Exercise 5: Data Analysis Project

```r
# Use the built-in mtcars dataset
data(mtcars)

# Tasks:
# 1. Calculate average MPG by number of cylinders
# 2. Find cars with MPG > 25
# 3. Create a scatter plot of MPG vs Weight
# 4. Identify the most fuel-efficient car

# Solutions:
aggregate(mpg ~ cyl, data = mtcars, FUN = mean)
mtcars[mtcars$mpg > 25, ]
plot(mtcars$wt, mtcars$mpg,
     xlab = "Weight", ylab = "MPG",
     main = "MPG vs Weight")
rownames(mtcars)[which.max(mtcars$mpg)]
```

---

## Summary and Next Steps

### What You've Learned

- ✅ R installation and RStudio interface
- ✅ Basic syntax and data types
- ✅ Vectors, matrices, data frames, and lists
- ✅ Control structures (if/else, loops)
- ✅ Writing functions
- ✅ Basic plotting with base R
- ✅ Reading and writing data files

### Next Steps

1. **Practice Daily:** Code for at least 30 minutes every day
2. **Build Projects:** Apply concepts to real datasets
3. **Move to Intermediate:** Continue with `2_Intermediate_Level_R.md`
4. **Join Communities:** R4DS Slack, Stack Overflow
5. **Read Documentation:** Use `?function` frequently

### Recommended Resources

- **Books:** "R for Data Science" by Hadley Wickham
- **Online:** DataCamp, Coursera R courses
- **Practice:** Kaggle datasets, TidyTuesday challenges

---

**Congratulations!** You've completed the Beginner Level. Move on to [`2_Intermediate_Level_R.md`](2_Intermediate_Level_R.md) to learn data wrangling, visualization, and advanced techniques.

**Keep coding! 🚀**
