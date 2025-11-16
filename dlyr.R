library(dplyr)

############################################################
# filter() — Select rows based on conditions
############################################################

# Select cars with mpg > 25
mtcars %>% filter(mpg > 25)

# Select cars with mpg > 20 AND exactly 4 cylinders
mtcars %>% filter(mpg > 20 & cyl == 4)

# Select cars whose number of cylinders is 4 OR 6
mtcars %>% filter(cyl %in% c(4, 6))

############################################################
# select() — Choose specific columns
############################################################

# Select only mpg, cyl, and hp columns
mtcars %>% select(mpg, cyl, hp)

# Select all columns whose names start with "c"
mtcars %>% select(starts_with("c"))

# Select only numeric columns using a helper function
mtcars %>% select(where(is.numeric))

############################################################
# arrange() — Sort rows
############################################################

# Sort cars in ascending order of mpg (lowest to highest)
mtcars %>% arrange(mpg)

# Sort cars in descending mpg (highest to lowest)
mtcars %>% arrange(desc(mpg))

# Sort by multiple columns: first by cyl (ascending),
# then by mpg (descending) within each cyl group
mtcars %>% arrange(cyl, desc(mpg))

############################################################
# mutate() — Create new variables or modify existing ones
############################################################

mtcars %>% mutate(
  # Create new column: mpg per cylinder
  mpg_per_cyl = mpg / cyl,
  
  # Create a categorical rating based on mpg
  efficiency = case_when(
    mpg > 25 ~ "High",
    mpg > 20 ~ "Medium",
    TRUE ~ "Low"
  )
)

############################################################
# summarise() — Summarize data (usually after grouping)
############################################################

mtcars %>% summarise(
  avg_mpg = mean(mpg),   # mean miles per gallon
  sd_mpg = sd(mpg),      # standard deviation of mpg
  n = n()                # number of rows in dataset
)

############################################################
# group_by() — Perform summaries within groups
############################################################

mtcars %>%
  group_by(cyl) %>%      # group cars by number of cylinders
  summarise(
    count = n(),         # number of cars in each cyl group
    avg_mpg = mean(mpg), # average mpg for that group
    avg_hp = mean(hp)    # average horsepower for that group
  )


band_members <- tibble(
  name = c("Mick", "John", "Paul"),
  band = c("Stones", "Beatles", "Beatles")
)

band_instruments <- tibble(
  name = c("John", "Paul", "Keith"),
  plays = c("guitar", "bass", "guitar")
)

############################################################
# INNER JOIN
############################################################
# Keeps only rows where the key (name) appears in BOTH tables.
# This shows people who appear in band_members AND band_instruments.
inner_join(band_members, band_instruments, by = "name")

############################################################
# LEFT JOIN
############################################################
# Keeps ALL rows from band_members (left table).
# If a name also appears in band_instruments, it adds the instrument.
# If not, plays = NA.
left_join(band_members, band_instruments, by = "name")

############################################################
# RIGHT JOIN
############################################################
# Keeps ALL rows from band_instruments (right table).
# If a name also appears in band_members, it adds the band.
# If not, band = NA.
right_join(band_members, band_instruments, by = "name")

############################################################
# FULL JOIN
############################################################
# Keeps ALL rows from BOTH tables.
# Missing values are filled with NA.
full_join(band_members, band_instruments, by = "name")

############################################################
# ANTI JOIN
############################################################
# Returns rows from band_members where NO match exists in band_instruments.
# This tells us who does NOT have an instrument listed.
anti_join(band_members, band_instruments, by = "name")

############################################################
# SEMI JOIN
############################################################
# Returns rows from band_members that DO have a match in band_instruments.
# Like a filter()—keeps only names that appear in BOTH, but WITHOUT adding columns.
semi_join(band_members, band_instruments, by = "name")


mtcars %>%
  mutate(
    # Creates a ranking of cars based on mpg, highest mpg = rank 1.
    # desc(mpg) sorts mpg in descending order.
    rank = row_number(desc(mpg)),
    
    # lag(): Shifts mpg DOWN by one row.
    # Shows the previous car’s mpg value.
    mpg_lag = lag(mpg),
    
    # lead(): Shifts mpg UP by one row.
    # Shows the next car’s mpg value.
    mpg_lead = lead(mpg),
    
    # cumsum(): Cumulative sum of mpg.
    # Adds each mpg to all previous ones, building a running total.
    cumsum_mpg = cumsum(mpg)
  )

# slice_head(): Returns the FIRST n rows of a dataset.
slice_head(mtcars, n = 5)


# slice_max(): Returns the top n rows with the HIGHEST value of a variable.
# In this case, top 5 cars with the highest mpg.
slice_max(mtcars, mpg, n = 5)

# slice_sample(): Selects n RANDOM rows from the dataset.
# Good for bootstrapping or random sampling.
slice_sample(mtcars, n = 10)
