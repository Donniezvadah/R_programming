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


gammaaa = matrix(c(0.5, 0.5 , .25 , .75), nrow= 2, ncol = 2, byrow = TRUE)
gammaaa
solve(gammaaa)
eigen(gammaaa)



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
mat



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


# Sort by one column
df[order(df$age), ]                    # Ascending
df[order(-df$age), ]                   # Descending
df[order(df$age, decreasing=TRUE), ]


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


