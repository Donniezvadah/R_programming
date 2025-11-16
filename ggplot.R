library(ggplot2)

# ggplot(data, aes(x, y)) + geom_*() + theme_*()
# This is the basic structure of a ggplot:
#   - ggplot(): Initialize the plot with a dataset and aesthetic mappings.
#   - aes(): Defines which variables go on x, y, color, size, etc.
#   - geom_*(): Layers that create different types of plots (points, lines, bars).
#   - theme_*(): Optional styling.

ggplot(mtcars, aes(x = wt, y = mpg)) +
  geom_point(size = 3, color = "blue")

ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl))) +
  geom_point(size = 3) +
  geom_smooth(method = "lm", se = TRUE)

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
ggsave("myplot.pdf", width = 8, height = 6, dpi = 300)

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


# Drop NAs
df %>% drop_na()
df %>% drop_na(mpg, cyl)

# Replace NAs
df %>% replace_na(list(mpg = 0))

# Fill NAs
df %>% fill(mpg, .direction = "down")

# Separate
df <- tibble(date = c("2023-01-15", "2023-02-20"))
df %>% separate(date, into = c("year", "month", "day"), sep = "-")

# Unite
df <- tibble(year = 2023, month = "01", day = "15")
df %>% unite("date", year, month, day, sep = "-")


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

# Combine your patchwork dashboard plot
plot1 <- (p1 | p2) / p3 +
  plot_annotation(title = "Economic Indicators Dashboard")

# Save as a 4K-quality PDF
ggsave(
  filename = "economic_dashboard_4k.pdf",
  plot = plot1,
  # device = cairo_pdf,     # best quality for PDF
  width = 12.8,           # inches → equivalent to 3840 px
  height = 7.2,           # inches → equivalent to 2160 px
  dpi = 300               # High resolution
)



