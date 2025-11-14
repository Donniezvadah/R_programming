install.packages("tidyverse")

require(tidyverse)


starwars %>% 
  select(gender, mass, height, species) %>% 
  filter(species=="Human") %>% na.omit() %>% 
  mutate(height= height/100) %>% 
  mutate(BMI = mass/height^2 ) %>% 
  group_by(gender) %>%
  summarise(Average_BMI = mean(BMI))


View(msleep)
my_data <- msleep %>% 
  select(name, sleep_total) %>% 
  filter(sleep_total>18)


hello <- msleep %>% 
  select(name,order, bodywt, sleep_total) %>% 
  filter(order == "Primates", order >20)
hello

msleep %>% 
  select(name,order, bodywt, sleep_total) %>% 
  filter(order == "Primates" | order >20)


msleep %>% 
  select(name,order, bodywt, sleep_total) %>% 
  filter(name == "Cow" | name == "Dog" | name == "Goat")

msleep %>% 
  select(name,order, bodywt, sleep_total) %>% 
  msleep %>% 
  select(name,order, bodywt, sleep_total) %>% 
  filter(name %in% c("Dog", "Goat", "Cow", "Horse"))

msleep %>% 
  select(name,order, bodywt, sleep_total) %>% 
  filter(between(sleep_total,18,20))

msleep %>% 
  select(name,conservation,order, bodywt, sleep_total) %>% 
  filter(near(sleep_total,17,tol =.5))

msleep %>% 
  select(name,order,conservation,  bodywt, sleep_total) %>% 
  filter(is.na(conservation))

msleep %>% 
  select(name,order,conservation,  bodywt, sleep_total) %>% 
  filter(!is.na(conservation))


msleep %>% 
  drop_na(sleep_rem, vore) %>% 
  group_by(vore) %>% 
  summarise("Average Total Sleep" = mean(sleep_total), "Max reo sleep"= max(sleep_rem)) %>% 
  view

#Explore your data
library(tidyverse)
?starwars
dim(starwars)
str(starwars)
glimpse(starwars)
attach(starwars)
names(starwars) # Hello world 
length(starwars)
