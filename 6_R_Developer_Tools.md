# R Developer Tools

## Table of Contents
1. [R Package Development](#r-package-development)
2. [Documentation with roxygen2](#documentation-with-roxygen2)
3. [Version Control with Git](#version-control-with-git)
4. [GitHub Workflows](#github-workflows)
5. [Continuous Integration/Deployment](#continuous-integration-deployment)
6. [Code Profiling and Optimization](#code-profiling-and-optimization)
7. [Package Distribution](#package-distribution)
8. [Development Tools](#development-tools)
9. [Code Quality](#code-quality)
10. [Best Practices](#best-practices)

---

## R Package Development

### Creating a Package

```r
# Install development tools
install.packages(c("devtools", "usethis", "roxygen2", "testthat"))

library(usethis)
library(devtools)

# Create new package
create_package("~/mypackage")

# Package structure created:
# mypackage/
# ├── R/
# ├── DESCRIPTION
# ├── NAMESPACE
# └── mypackage.Rproj
```

### DESCRIPTION File

```yaml
Package: mypackage
Type: Package
Title: My Awesome R Package
Version: 0.1.0
Author: Your Name <your.email@example.com>
Maintainer: Your Name <your.email@example.com>
Description: This package provides tools for data analysis and visualization.
    It includes functions for data cleaning, transformation, and statistical modeling.
License: MIT + file LICENSE
Encoding: UTF-8
LazyData: true
Roxygen: list(markdown = TRUE)
RoxygenNote: 7.2.3
Imports:
    dplyr (>= 1.0.0),
    ggplot2 (>= 3.3.0),
    magrittr
Suggests:
    testthat (>= 3.0.0),
    knitr,
    rmarkdown,
    covr
Depends: R (>= 4.0.0)
VignetteBuilder: knitr
URL: https://github.com/username/mypackage
BugReports: https://github.com/username/mypackage/issues
```

### Adding Functions

```r
# Create new R file
use_r("data_processing")

# R/data_processing.R
#' Clean Data
#'
#' Remove missing values and duplicates from a data frame
#'
#' @param data A data frame to clean
#' @param columns Character vector of column names to check for NAs
#' @return A cleaned data frame
#' @export
#' @examples
#' df <- data.frame(x = c(1, 2, NA, 4), y = c("a", "b", "b", "c"))
#' clean_data(df, c("x"))
clean_data <- function(data, columns = NULL) {
  if (!is.data.frame(data)) {
    stop("data must be a data frame")
  }
  
  if (is.null(columns)) {
    columns <- names(data)
  }
  
  # Remove rows with NA in specified columns
  data <- data[complete.cases(data[, columns]), ]
  
  # Remove duplicates
  data <- unique(data)
  
  return(data)
}
```

### Adding Data

```r
# Add internal data (available only to package functions)
use_data_raw("prep_data")

# data-raw/prep_data.R
my_data <- data.frame(
  x = 1:100,
  y = rnorm(100)
)

usethis::use_data(my_data, internal = TRUE)

# Add external data (available to users)
usethis::use_data(my_data, overwrite = TRUE)

# Document data
# R/data.R
#' Sample Dataset
#'
#' A dataset containing sample data for demonstration
#'
#' @format A data frame with 100 rows and 2 variables:
#' \describe{
#'   \item{x}{Numeric values from 1 to 100}
#'   \item{y}{Random normal values}
#' }
#' @source Generated for package demonstration
"my_data"
```

### Package Dependencies

```r
# Add to Imports (required)
use_package("dplyr")
use_package("ggplot2", min_version = "3.3.0")

# Add to Suggests (optional)
use_package("testthat", type = "Suggests")

# Import specific functions
#' @importFrom dplyr filter mutate
#' @importFrom magrittr %>%

# Import entire package
#' @import ggplot2
```

---

## Documentation with roxygen2

### Function Documentation

```r
#' Calculate Summary Statistics
#'
#' This function calculates various summary statistics for a numeric vector.
#' It handles missing values and provides multiple measures of central tendency
#' and dispersion.
#'
#' @param x A numeric vector
#' @param na.rm Logical; if TRUE, missing values are removed before computation
#' @param trim Numeric; fraction (0 to 0.5) of observations to trim from each end
#' @param digits Integer; number of decimal places to round to
#'
#' @return A named list containing:
#'   \item{mean}{Arithmetic mean}
#'   \item{median}{Median value}
#'   \item{sd}{Standard deviation}
#'   \item{min}{Minimum value}
#'   \item{max}{Maximum value}
#'   \item{n}{Number of observations}
#'
#' @export
#' @examples
#' # Basic usage
#' x <- c(1, 2, 3, 4, 5, NA)
#' summary_stats(x, na.rm = TRUE)
#'
#' # With trimming
#' summary_stats(x, na.rm = TRUE, trim = 0.1)
#'
#' @seealso \code{\link{mean}}, \code{\link{median}}, \code{\link{sd}}
#' @family statistical functions
summary_stats <- function(x, na.rm = FALSE, trim = 0, digits = 2) {
  if (!is.numeric(x)) {
    stop("x must be numeric")
  }
  
  result <- list(
    mean = round(mean(x, na.rm = na.rm, trim = trim), digits),
    median = round(median(x, na.rm = na.rm), digits),
    sd = round(sd(x, na.rm = na.rm), digits),
    min = min(x, na.rm = na.rm),
    max = max(x, na.rm = na.rm),
    n = if (na.rm) sum(!is.na(x)) else length(x)
  )
  
  return(result)
}
```

### Building Documentation

```r
# Generate documentation from roxygen comments
devtools::document()

# Check documentation
devtools::check_man()

# Preview help page
?summary_stats
```

### Vignettes

```r
# Create vignette
use_vignette("introduction")

# vignettes/introduction.Rmd
---
title: "Introduction to mypackage"
author: "Your Name"
date: "`r Sys.Date()`"
output: rmarkdown::html_vignette
vignette: >
  %\VignetteIndexEntry{Introduction to mypackage}
  %\VignetteEngine{knitr::rmarkdown}
  %\VignetteEncoding{UTF-8}
---

```{r setup, include = FALSE}
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)
library(mypackage)
```

## Overview

This package provides tools for...

## Installation

```{r eval=FALSE}
devtools::install_github("username/mypackage")
```

## Basic Usage

```{r}
data <- data.frame(x = 1:10, y = rnorm(10))
result <- clean_data(data)
```

# Build vignettes
devtools::build_vignettes()
```

### Package Website with pkgdown

```r
# Setup pkgdown
use_pkgdown()

# Build website
pkgdown::build_site()

# Deploy to GitHub Pages
use_pkgdown_github_pages()
```

---

## Version Control with Git

### Setup Git Repository

```bash
# Initialize git
git init

# Configure user
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Create .gitignore
```

```r
# R approach
use_git()

# Add .gitignore entries
use_git_ignore(c("*.Rproj", ".Rhistory", ".RData"))
```

### Basic Git Workflow

```bash
# Check status
git status

# Stage files
git add R/my_function.R
git add .  # Stage all changes

# Commit
git commit -m "Add data processing function"

# View history
git log --oneline

# Create branch
git checkout -b feature/new-analysis

# Switch branches
git checkout main

# Merge branch
git merge feature/new-analysis

# Delete branch
git branch -d feature/new-analysis
```

### .gitignore for R

```gitignore
# History files
.Rhistory
.Rapp.history

# Session Data files
.RData

# User-specific files
.Ruserdata

# RStudio files
.Rproj.user/
*.Rproj

# OAuth2 token
.httr-oauth

# knitr and R markdown
*.utf8.md
*.knit.md

# Temporary files
*~
*.swp
*.swo

# Output files
*.pdf
*.html

# Data files (if large)
data/raw/*
!data/raw/.gitkeep
```

---

## GitHub Workflows

### Create GitHub Repository

```r
# Create GitHub repo from R
use_github()

# Or manually:
# 1. Create repo on GitHub
# 2. Add remote
git remote add origin https://github.com/username/mypackage.git
git push -u origin main
```

### README

```r
# Create README
use_readme_rmd()

# README.Rmd
---
output: github_document
---

# mypackage

<!-- badges: start -->
[![R-CMD-check](https://github.com/username/mypackage/workflows/R-CMD-check/badge.svg)](https://github.com/username/mypackage/actions)
[![Codecov](https://codecov.io/gh/username/mypackage/branch/main/graph/badge.svg)](https://codecov.io/gh/username/mypackage)
<!-- badges: end -->

The goal of mypackage is to...

## Installation

```{r eval=FALSE}
# Development version
devtools::install_github("username/mypackage")
```

## Example

```{r example}
library(mypackage)
data <- data.frame(x = 1:10, y = rnorm(10))
result <- clean_data(data)
```

# Build README
devtools::build_readme()
```

### LICENSE

```r
# Add MIT license
use_mit_license("Your Name")

# Or GPL-3
use_gpl3_license()

# Or Apache 2.0
use_apache_license()
```

---

## Continuous Integration/Deployment

### GitHub Actions - R CMD Check

```r
# Setup GitHub Actions
use_github_action_check_standard()

# .github/workflows/R-CMD-check.yaml
name: R-CMD-check

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

jobs:
  R-CMD-check:
    runs-on: ${{ matrix.config.os }}

    name: ${{ matrix.config.os }} (${{ matrix.config.r }})

    strategy:
      fail-fast: false
      matrix:
        config:
          - {os: macos-latest,   r: 'release'}
          - {os: windows-latest, r: 'release'}
          - {os: ubuntu-latest,   r: 'devel'}
          - {os: ubuntu-latest,   r: 'release'}
          - {os: ubuntu-latest,   r: 'oldrel-1'}

    steps:
      - uses: actions/checkout@v3

      - uses: r-lib/actions/setup-r@v2
        with:
          r-version: ${{ matrix.config.r }}

      - uses: r-lib/actions/setup-pandoc@v2

      - uses: r-lib/actions/setup-r-dependencies@v2
        with:
          extra-packages: any::rcmdcheck
          needs: check

      - uses: r-lib/actions/check-r-package@v2
```

### Code Coverage

```r
# Setup code coverage
use_github_action("test-coverage")
use_coverage()

# Run locally
library(covr)
package_coverage()

# View report
report()
```

### Automatic pkgdown Deployment

```r
# Setup automatic website deployment
use_github_action("pkgdown")

# Website will be built and deployed on push to main
```

---

## Code Profiling and Optimization

### Profiling with profvis

```r
library(profvis)

# Profile code
profvis({
  data <- data.frame(
    x = rnorm(100000),
    y = rnorm(100000)
  )
  
  # Slow approach
  result <- numeric(nrow(data))
  for (i in 1:nrow(data)) {
    result[i] <- data$x[i] + data$y[i]
  }
  
  # Fast approach
  result2 <- data$x + data$y
})
```

### Benchmarking

```r
library(bench)

# Compare implementations
results <- mark(
  loop = {
    result <- numeric(10000)
    for (i in 1:10000) result[i] <- i^2
  },
  vectorized = {
    result <- (1:10000)^2
  },
  check = FALSE,
  iterations = 100
)

print(results)
plot(results)
```

### Memory Profiling

```r
# Check object size
object.size(large_data)
format(object.size(large_data), units = "MB")

# Profile memory
library(profmem)

p <- profmem({
  x <- numeric(1e6)
  y <- rnorm(1e6)
  z <- x + y
})

print(p)
total(p)
```

### Optimization Strategies

```r
# 1. Vectorization
# Bad
sum_squares_slow <- function(x) {
  total <- 0
  for (i in seq_along(x)) {
    total <- total + x[i]^2
  }
  total
}

# Good
sum_squares_fast <- function(x) {
  sum(x^2)
}

# 2. Pre-allocation
# Bad
grow_vector <- function(n) {
  result <- c()
  for (i in 1:n) {
    result <- c(result, i^2)
  }
  result
}

# Good
preallocate_vector <- function(n) {
  result <- numeric(n)
  for (i in 1:n) {
    result[i] <- i^2
  }
  result
}

# 3. Use efficient data structures
library(data.table)
dt <- as.data.table(large_df)  # Fast operations

# 4. Parallel processing
library(parallel)
cl <- makeCluster(detectCores() - 1)
results <- parLapply(cl, data_list, expensive_function)
stopCluster(cl)
```

---

## Package Distribution

### CRAN Submission

```r
# Check package thoroughly
devtools::check()

# Check on multiple R versions
devtools::check_rhub()
devtools::check_win_devel()
devtools::check_mac_release()

# Build package
devtools::build()

# Submit to CRAN
devtools::submit_cran()
```

### CRAN Release Checklist

```r
# 1. Update version number
use_version()

# 2. Update NEWS.md
use_news_md()

# 3. Run checks
devtools::check()
devtools::check_rhub()
devtools::check_win_devel()

# 4. Update cran-comments.md
use_cran_comments()

# 5. Build package
devtools::build()

# 6. Submit
devtools::submit_cran()
```

### R-universe (Alternative)

```r
# Easy distribution without CRAN submission
# Setup at https://r-universe.dev

# Users install with:
install.packages("mypackage", repos = "https://username.r-universe.dev")
```

---

## Development Tools

### RStudio Addins

```r
# Create addin
use_addin("insert_pipe")

# inst/rstudio/addins.dcf
Name: Insert Pipe
Description: Inserts %>% at cursor
Binding: insert_pipe
Interactive: false

# R/addins.R
insert_pipe <- function() {
  rstudioapi::insertText(" %>% ")
}
```

### Custom RStudio Snippets

```r
# Edit snippets
usethis::edit_rstudio_snippets()

# Add custom snippets:
snippet fun
	${1:name} <- function(${2:args}) {
		${0}
	}

snippet test
	test_that("${1:description}", {
		${0}
	})
```

### Pre-commit Hooks

```r
# Setup pre-commit hooks
library(precommit)
use_precommit()

# .pre-commit-config.yaml
repos:
-   repo: https://github.com/lorenzwalthert/precommit
    rev: v0.3.2
    hooks:
    -   id: style-files
    -   id: roxygenize
    -   id: use-tidy-description
    -   id: spell-check
    -   id: lintr
    -   id: readme-rmd-rendered
    -   id: parsable-R
    -   id: no-browser-statement
```

---

## Code Quality

### Static Code Analysis

```r
# lintr - code style checking
library(lintr)
lint_package()
lint("R/my_function.R")

# Custom linters
my_linters <- linters_with_defaults(
  line_length_linter(120),
  object_name_linter("snake_case")
)

lint_package(linters = my_linters)
```

### Code Formatting

```r
# styler - automatic formatting
library(styler)

# Format file
style_file("R/my_function.R")

# Format package
style_pkg()

# Format with custom style
style_pkg(scope = "line_breaks")
```

### Spell Checking

```r
# Check spelling
library(spelling)
spell_check_package()

# Update wordlist
update_wordlist()

# Add to .aspell file
use_spell_check()
```

---

## Best Practices

### Package Development Workflow

```r
# 1. Write function
use_r("my_function")

# 2. Load all functions
load_all()  # or Ctrl+Shift+L

# 3. Test interactively
my_function(test_data)

# 4. Write tests
use_test("my_function")

# 5. Run tests
test()  # or Ctrl+Shift+T

# 6. Document
document()  # or Ctrl+Shift+D

# 7. Check package
check()  # or Ctrl+Shift+E

# 8. Install
install()
```

### Semantic Versioning

```r
# Version format: MAJOR.MINOR.PATCH
# 1.0.0 -> 1.0.1 (patch: bug fixes)
# 1.0.1 -> 1.1.0 (minor: new features, backward compatible)
# 1.1.0 -> 2.0.0 (major: breaking changes)

# Update version
use_version("patch")
use_version("minor")
use_version("major")
```

### NEWS.md

```markdown
# mypackage 0.2.0

## New features

* Added `new_function()` for advanced analysis (#15)
* Support for tibbles in `clean_data()` (#18)

## Bug fixes

* Fixed issue with NA handling in `summary_stats()` (#12)
* Corrected documentation for `transform_data()` (#14)

## Breaking changes

* Renamed `old_function()` to `new_function()`
* Changed default behavior of `clean_data()`

# mypackage 0.1.0

* Initial CRAN release
```

### Code Review Checklist

```r
# Before committing:
# Functions are documented
# Tests are written and passing
# Code is formatted (styler)
# No linter warnings
# Package checks pass
# NEWS.md is updated
# Version number is bumped
# Examples run correctly
# Vignettes build
# No browser() or debug statements
```

---

## Complete Package Example

### Final Package Structure

```
mypackage/
├── .github/
│   └── workflows/
│       ├── R-CMD-check.yaml
│       ├── test-coverage.yaml
│       └── pkgdown.yaml
├── R/
│   ├── data_processing.R
│   ├── visualization.R
│   ├── utils.R
│   └── mypackage-package.R
├── man/
│   ├── clean_data.Rd
│   └── mypackage-package.Rd
├── tests/
│   ├── testthat.R
│   └── testthat/
│       ├── test-data_processing.R
│       └── test-visualization.R
├── vignettes/
│   └── introduction.Rmd
├── data/
│   └── sample_data.rda
├── data-raw/
│   └── prep_data.R
├── .gitignore
├── .Rbuildignore
├── DESCRIPTION
├── LICENSE
├── LICENSE.md
├── NAMESPACE
├── NEWS.md
├── README.md
├── README.Rmd
├── cran-comments.md
└── mypackage.Rproj
```

---

## Summary

### Skills Mastered

- R package development from scratch
- Comprehensive documentation with roxygen2
- Version control with Git and GitHub
- CI/CD pipelines with GitHub Actions
- Code profiling and optimization
- Package distribution (CRAN, GitHub, R-universe)
- Code quality tools (lintr, styler, spelling)
- Professional development workflow

### Resources

- [R Packages Book](https://r-pkgs.org/) by Hadley Wickham
- [Writing R Extensions](https://cran.r-project.org/doc/manuals/r-release/R-exts.html) (Official)
- [usethis Documentation](https://usethis.r-lib.org/)
- [devtools Documentation](https://devtools.r-lib.org/)
- [GitHub Actions for R](https://github.com/r-lib/actions)

### Next Steps

1. Build: Create your first R package
2. Test: Achieve 100% test coverage
3. Share: Publish on GitHub and CRAN
4. Contribute: Contribute to open-source R packages
5. Teach: Share your knowledge with the R community

---

Congratulations! 

You've completed the entire R Programming course! You now have the skills to:
- Write professional R code
- Build data science projects
- Create interactive applications
- Develop and distribute R packages
- Apply best practices in production environments

Keep coding, keep learning, and contribute to the R community!
