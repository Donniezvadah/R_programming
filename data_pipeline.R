# Load tidyverse: provides dplyr, ggplot2, readr, stringr, lubridate, etc.
library(tidyverse)

# ================================
# 1. LOAD AND CLEAN THE DATA
# ================================
df <- read_csv("sales_data.csv") %>%
  
  # Remove rows where customer_id or amount is missing
  drop_na(customer_id, amount) %>%
  
  # Clean and transform variables
  mutate(
    # Convert date column from text to Date format
    date = ymd(date),
    
    # Extract month as a labelled factor (Jan, Feb, ...)
    month = month(date, label = TRUE),
    
    # Extract year as a numeric value
    year = year(date),
    
    # Capitalize the category name (e.g., "electronics" → "Electronics")
    category = str_to_title(category)
  )

# ================================
# 2. AGGREGATE (GROUP & SUMMARISE)
# ================================
summary_df <- df %>%
  
  # Group by year, month, and product category
  group_by(year, month, category) %>%
  
  summarise(
    # Total sales amount for that group
    total_sales = sum(amount),
    
    # Average sales per transaction
    avg_sale = mean(amount),
    
    # Count number of transactions
    n_transactions = n(),
    
    # Drop grouping structure after summarising
    .groups = "drop"
  )

# ================================
# 3. VISUALIZE  
# ================================
ggplot(summary_df, aes(x = month, y = total_sales, fill = category)) +
  
  # Bar chart with grouped (side-by-side) bars
  geom_col(position = "dodge") +
  
  # Create a separate panel for each year
  facet_wrap(~ year) +
  
  # Use a clean, minimal theme
  theme_minimal() +
  
  # Add title and axis labels
  labs(
    title = "Monthly Sales by Category",
    x = "Month",
    y = "Total Sales",
    fill = "Category"
  )
