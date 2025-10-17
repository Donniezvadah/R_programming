# Intermediate Level R

## Table of Contents
1. [Tidyverse Introduction](#tidyverse)
2. [dplyr - Data Manipulation](#dplyr)
3. [ggplot2 - Data Visualization](#ggplot2)
4. [tidyr - Data Reshaping](#tidyr)
5. [purrr - Functional Programming](#purrr)
6. [APIs and File Formats](#apis-and-files)

---

## Tidyverse

### Installation and Loading

```r
install.packages("tidyverse")
library(tidyverse)  # Loads: ggplot2, dplyr, tidyr, readr, purrr, tibble, stringr, forcats
```

### Tibbles vs Data Frames

```r
# Create tibble
tib <- tibble(
  name = c("Alice", "Bob", "Charlie"),
  age = c(25, 30, 35),
  salary = c(50000, 60000, 75000)
)

# Convert
as_tibble(mtcars)
as.data.frame(tib)
```

---

## dplyr - Data Manipulation

### Core Verbs

```r
library(dplyr)

# filter() - Filter rows
mtcars %>% filter(mpg > 25)
mtcars %>% filter(mpg > 20 & cyl == 4)
mtcars %>% filter(cyl %in% c(4, 6))

# select() - Choose columns
mtcars %>% select(mpg, cyl, hp)
mtcars %>% select(starts_with("c"))
mtcars %>% select(where(is.numeric))

# arrange() - Sort rows
mtcars %>% arrange(mpg)
mtcars %>% arrange(desc(mpg))
mtcars %>% arrange(cyl, desc(mpg))

# mutate() - Create/modify columns
mtcars %>% mutate(
  mpg_per_cyl = mpg / cyl,
  efficiency = case_when(
    mpg > 25 ~ "High",
    mpg > 20 ~ "Medium",
    TRUE ~ "Low"
  )
)

# summarise() - Aggregate
mtcars %>% summarise(
  avg_mpg = mean(mpg),
  sd_mpg = sd(mpg),
  n = n()
)

# group_by() - Group operations
mtcars %>%
  group_by(cyl) %>%
  summarise(
    count = n(),
    avg_mpg = mean(mpg),
    avg_hp = mean(hp)
  )
```

### Joins

```r
band_members <- tibble(
  name = c("Mick", "John", "Paul"),
  band = c("Stones", "Beatles", "Beatles")
)

band_instruments <- tibble(
  name = c("John", "Paul", "Keith"),
  plays = c("guitar", "bass", "guitar")
)

# Types of joins
inner_join(band_members, band_instruments, by = "name")
left_join(band_members, band_instruments, by = "name")
right_join(band_members, band_instruments, by = "name")
full_join(band_members, band_instruments, by = "name")
anti_join(band_members, band_instruments, by = "name")
semi_join(band_members, band_instruments, by = "name")
```

### Window Functions

```r
mtcars %>%
  mutate(
    rank = row_number(desc(mpg)),
    mpg_lag = lag(mpg),
    mpg_lead = lead(mpg),
    cumsum_mpg = cumsum(mpg)
  )

# Slicing
slice_head(mtcars, n = 5)
slice_max(mtcars, mpg, n = 5)
slice_sample(mtcars, n = 10)
```

---

## ggplot2 - Data Visualization

### Basic Structure

```r
library(ggplot2)

# ggplot(data, aes(x, y)) + geom_*() + theme_*()

# Scatter plot
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point(size = 3, color = "blue")

# With color by group
ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl))) +
  geom_point(size = 3) +
  geom_smooth(method = "lm", se = TRUE)
```

### Common Geoms

```r
# Line chart
ggplot(economics, aes(x = date, y = unemploy)) +
  geom_line(color = "blue", size = 1)

# Bar chart
ggplot(mtcars, aes(x = factor(cyl))) +
  geom_bar(fill = "steelblue")

# Histogram
ggplot(mtcars, aes(x = mpg)) +
  geom_histogram(bins = 15, fill = "lightblue", color = "black")

# Box plot
ggplot(mtcars, aes(x = factor(cyl), y = mpg)) +
  geom_boxplot(fill = "lightgreen")

# Violin plot
ggplot(mtcars, aes(x = factor(cyl), y = mpg)) +
  geom_violin(fill = "lightblue") +
  geom_boxplot(width = 0.2)
```

### Customization

```r
ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl))) +
  geom_point(size = 3) +
  labs(
    title = "Fuel Efficiency vs Weight",
    subtitle = "Motor Trend Car Road Tests",
    x = "Weight (1000 lbs)",
    y = "Miles per Gallon",
    color = "Cylinders"
  ) +
  scale_color_brewer(palette = "Set1") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    legend.position = "bottom"
  )

# Faceting
ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  facet_wrap(~ cyl)

# Save plot
ggsave("myplot.png", width = 8, height = 6, dpi = 300)
```

---

## tidyr - Data Reshaping

### Pivoting

```r
library(tidyr)

# Wide to long
wide_data <- tibble(
  country = c("USA", "Canada"),
  year_2020 = c(100, 80),
  year_2021 = c(110, 85)
)

long_data <- wide_data %>%
  pivot_longer(
    cols = starts_with("year"),
    names_to = "year",
    names_prefix = "year_",
    values_to = "value"
  )

# Long to wide
wide_data <- long_data %>%
  pivot_wider(
    names_from = year,
    values_from = value
  )
```

### Missing Data

```r
# Drop NAs
df %>% drop_na()
df %>% drop_na(mpg, cyl)

# Replace NAs
df %>% replace_na(list(mpg = 0))

# Fill NAs
df %>% fill(mpg, .direction = "down")
```

### Separate and Unite

```r
# Separate
df <- tibble(date = c("2023-01-15", "2023-02-20"))
df %>% separate(date, into = c("year", "month", "day"), sep = "-")

# Unite
df <- tibble(year = 2023, month = "01", day = "15")
df %>% unite("date", year, month, day, sep = "-")
```

---

## purrr - Functional Programming

### Map Functions

```r
library(purrr)

# map() - returns list
map(1:5, sqrt)

# map_dbl() - returns numeric vector
map_dbl(1:5, sqrt)

# map_chr() - returns character vector
map_chr(1:5, as.character)

# map_lgl() - returns logical vector
map_lgl(1:5, ~ .x > 3)

# map2() - iterate over two vectors
map2_dbl(1:3, 4:6, ~ .x + .y)

# pmap() - iterate over multiple vectors
pmap_dbl(list(1:3, 4:6, 7:9), sum)
```

### Advanced Patterns

```r
# keep() and discard()
keep(1:10, ~ .x %% 2 == 0)  # Keep even
discard(1:10, ~ .x %% 2 == 0)  # Discard even

# reduce() - cumulative operations
reduce(1:5, `+`)  # Sum
reduce(1:5, `*`)  # Product

# map_if() - conditional mapping
df <- tibble(x = 1:3, y = c("a", "b", "c"))
map_if(df, is.numeric, ~ .x * 2)
```

---

## APIs and File Formats

### JSON Data

```r
library(jsonlite)

# Parse JSON string
json_string <- '{"name": "Alice", "age": 25}'
data <- fromJSON(json_string)

# Read JSON file
data <- fromJSON("data.json")

# Convert to JSON
df <- data.frame(name = c("Alice", "Bob"), age = c(25, 30))
json <- toJSON(df, pretty = TRUE)
```

### API Requests

```r
library(httr)

# GET request
response <- GET("https://api.example.com/data")
status_code(response)
content(response, as = "parsed")

# POST request
response <- POST(
  "https://api.example.com/data",
  body = list(name = "Alice", age = 25),
  encode = "json"
)

# With API key
response <- GET(
  "https://api.example.com/data",
  add_headers(Authorization = paste("Bearer", Sys.getenv("API_KEY")))
)
```

### Reading Files

```r
# CSV
df <- read_csv("data.csv")  # readr
df <- read.csv("data.csv")  # base R

# Excel
library(readxl)
df <- read_excel("data.xlsx", sheet = "Sheet1")

# SPSS, Stata, SAS
library(haven)
df <- read_spss("data.sav")
df <- read_stata("data.dta")
df <- read_sas("data.sas7bdat")

# Parquet
library(arrow)
df <- read_parquet("data.parquet")

# Database
library(DBI)
library(RSQLite)
con <- dbConnect(SQLite(), "database.db")
df <- dbReadTable(con, "table_name")
df <- dbGetQuery(con, "SELECT * FROM table_name WHERE age > 25")
dbDisconnect(con)
```

### String Operations (stringr)

```r
library(stringr)

# Basic operations
str_length("Hello")
str_to_upper("hello")
str_to_lower("HELLO")
str_trim("  hello  ")

# Pattern matching
str_detect("Hello World", "World")  # TRUE
str_count("banana", "a")  # 3
str_extract("Phone: 123-456-7890", "\\d{3}-\\d{3}-\\d{4}")

# Manipulation
str_replace("Hello World", "World", "R")
str_replace_all("banana", "a", "o")
str_split("a,b,c", ",")
str_c("Hello", "World", sep = " ")
str_sub("Hello World", 1, 5)  # "Hello"
```

### Date Operations (lubridate)

```r
library(lubridate)

# Parse dates ----
ymd("2023-01-15")
mdy("01/15/2023")
dmy("15-01-2023")
ymd_hms("2023-01-15 14:30:00")

# Extract components
date <- ymd("2023-06-15")
year(date)  # 2023
month(date)  # 6
day(date)   # 15
wday(date, label = TRUE)  # "Thu"

# Date arithmetic
today() + days(7)
today() - months(2)
now() + hours(3)

# Intervals
int <- interval(ymd("2023-01-01"), ymd("2023-12-31"))
time_length(int, "months")
```

---

## Practice Projects

### Project 1: Data Pipeline

```r
# Build a complete data pipeline
library(tidyverse)

# 1. Load and clean data
df <- read_csv("sales_data.csv") %>%
  drop_na(customer_id, amount) %>%
  mutate(
    date = ymd(date),
    month = month(date, label = TRUE),
    year = year(date),
    category = str_to_title(category)
  )

# 2. Aggregate
summary_df <- df %>%
  group_by(year, month, category) %>%
  summarise(
    total_sales = sum(amount),
    avg_sale = mean(amount),
    n_transactions = n(),
    .groups = "drop"
  )

# 3. Visualize
ggplot(summary_df, aes(x = month, y = total_sales, fill = category)) +
  geom_col(position = "dodge") +
  facet_wrap(~ year) +
  theme_minimal() +
  labs(title = "Monthly Sales by Category")
```

### Project 2: API Data Analysis

```r
# Fetch and analyze API data
library(httr)
library(jsonlite)

# 1. Fetch data
response <- GET("https://api.coingecko.com/api/v3/coins/markets",
                query = list(vs_currency = "usd", per_page = 100))

crypto_data <- content(response, as = "parsed") %>%
  map_df(~tibble(
    name = .x$name,
    symbol = .x$symbol,
    price = .x$current_price,
    market_cap = .x$market_cap,
    volume = .x$total_volume
  ))

# 2. Analyze
top_10 <- crypto_data %>%
  arrange(desc(market_cap)) %>%
  slice_head(n = 10)

# 3. Visualize
ggplot(top_10, aes(x = reorder(symbol, market_cap), y = market_cap / 1e9)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Top 10 Cryptocurrencies by Market Cap",
    x = NULL,
    y = "Market Cap (Billions USD)"
  ) +
  theme_minimal()
```

### Project 3: Advanced Visualization

```r
# Create a complex dashboard-style visualization
library(patchwork)  # For combining plots

# Plot 1: Time series
p1 <- ggplot(economics, aes(x = date, y = unemploy / 1000)) +
  geom_line(color = "blue") +
  labs(title = "Unemployment Over Time", y = "Unemployed (thousands)")

# Plot 2: Distribution
p2 <- ggplot(economics, aes(x = pce)) +
  geom_histogram(bins = 30, fill = "coral") +
  labs(title = "Distribution of Personal Consumption")

# Plot 3: Correlation
p3 <- ggplot(economics, aes(x = pce, y = pop)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm") +
  labs(title = "Consumption vs Population")

# Combine plots
(p1 | p2) / p3 + plot_annotation(title = "Economic Indicators Dashboard")
```

---

## Summary

### Key Skills Learned

- ✅ Data manipulation with dplyr (filter, select, mutate, summarise, group_by)
- ✅ Data visualization with ggplot2 (scatter, line, bar, histogram, box plots)
- ✅ Data reshaping with tidyr (pivot_longer, pivot_wider)
- ✅ Functional programming with purrr (map functions)
- ✅ Working with strings (stringr) and dates (lubridate)
- ✅ API integration and JSON handling
- ✅ Reading various file formats

### Next Steps

1. **Practice:** Apply these skills to real datasets (Kaggle, TidyTuesday)
2. **Advance:** Move to `3_Expert_Level_R.md` for statistical modeling and ML
3. **Build:** Create data analysis projects and share on GitHub
4. **Learn:** Explore advanced ggplot2 extensions (plotly, gganimate)

**Continue to Expert Level! 🚀**
