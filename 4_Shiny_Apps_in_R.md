# Shiny Apps in R

## Table of Contents
1. [Introduction to Shiny](#introduction-to-shiny)
2. [Basic App Structure](#basic-app-structure)
3. [UI Components](#ui-components)
4. [Server Logic and Reactivity](#server-logic-and-reactivity)
5. [Layouts and Themes](#layouts-and-themes)
6. [Interactive Outputs](#interactive-outputs)
7. [Advanced Reactivity](#advanced-reactivity)
8. [Shiny Modules](#shiny-modules)
9. [Deployment](#deployment)
10. [Best Practices](#best-practices)

---

## Introduction to Shiny

### What is Shiny?

Shiny is an R package that makes it easy to build interactive web applications directly from R without requiring HTML, CSS, or JavaScript knowledge.

**Key Features:**
- ✅ Reactive programming model
- ✅ Rich UI components
- ✅ Easy deployment
- ✅ Integration with R packages
- ✅ Real-time interactivity

### Installation

```r
# Install Shiny
install.packages("shiny")

# Load library
library(shiny)

# Run example app
runExample("01_hello")
```

---

## Basic App Structure

### Minimal Shiny App

```r
library(shiny)

# Define UI
ui <- fluidPage(
  titlePanel("My First Shiny App"),
  
  sidebarLayout(
    sidebarPanel(
      sliderInput("num", "Choose a number:", 
                  min = 1, max = 100, value = 50)
    ),
    
    mainPanel(
      textOutput("result")
    )
  )
)

# Define server
server <- function(input, output, session) {
  output$result <- renderText({
    paste("You selected:", input$num)
  })
}

# Run app
shinyApp(ui = ui, server = server)
```

### File Structure

**Single-file app (app.R):**
```r
# app.R
library(shiny)

ui <- fluidPage(
  # UI code
)

server <- function(input, output, session) {
  # Server code
}

shinyApp(ui, server)
```

**Two-file app:**
```r
# ui.R
library(shiny)

fluidPage(
  # UI code
)

# server.R
library(shiny)

function(input, output, session) {
  # Server code
}
```

---

## UI Components

### Input Widgets

```r
ui <- fluidPage(
  # Text input
  textInput("name", "Enter your name:", value = ""),
  
  # Numeric input
  numericInput("age", "Enter your age:", value = 25, min = 0, max = 120),
  
  # Slider
  sliderInput("height", "Height (cm):", min = 100, max = 250, value = 170),
  
  # Range slider
  sliderInput("range", "Select range:", min = 0, max = 100, value = c(25, 75)),
  
  # Checkbox
  checkboxInput("subscribe", "Subscribe to newsletter", value = FALSE),
  
  # Checkbox group
  checkboxGroupInput("interests", "Interests:",
                     choices = c("Sports", "Music", "Reading", "Travel")),
  
  # Radio buttons
  radioButtons("gender", "Gender:",
               choices = c("Male", "Female", "Other")),
  
  # Select dropdown
  selectInput("country", "Country:",
              choices = c("USA", "Canada", "UK", "Australia")),
  
  # Multiple select
  selectInput("languages", "Languages:",
              choices = c("R", "Python", "Julia", "SQL"),
              multiple = TRUE),
  
  # Date input
  dateInput("birth_date", "Birth Date:", value = "2000-01-01"),
  
  # Date range
  dateRangeInput("date_range", "Select date range:"),
  
  # File upload
  fileInput("file", "Upload CSV file:", accept = c(".csv")),
  
  # Action button
  actionButton("submit", "Submit", class = "btn-primary"),
  
  # Download button
  downloadButton("download", "Download Data")
)
```

### Output Widgets

```r
ui <- fluidPage(
  # Text output
  textOutput("text"),
  verbatimTextOutput("code"),
  
  # Table output
  tableOutput("table"),
  dataTableOutput("datatable"),
  
  # Plot output
  plotOutput("plot"),
  
  # UI output (dynamic UI)
  uiOutput("dynamic_ui"),
  
  # HTML output
  htmlOutput("html")
)
```

---

## Server Logic and Reactivity

### Basic Reactivity

```r
server <- function(input, output, session) {
  
  # Render text
  output$text <- renderText({
    paste("Hello,", input$name)
  })
  
  # Render table
  output$table <- renderTable({
    head(mtcars, input$rows)
  })
  
  # Render plot
  output$plot <- renderPlot({
    hist(rnorm(input$n), main = "Random Normal Distribution")
  })
  
  # Render data table (interactive)
  output$datatable <- renderDataTable({
    mtcars
  })
}
```

### Reactive Expressions

```r
server <- function(input, output, session) {
  
  # Reactive expression (cached)
  filtered_data <- reactive({
    mtcars[mtcars$cyl == input$cyl, ]
  })
  
  # Use reactive expression
  output$plot <- renderPlot({
    hist(filtered_data()$mpg)
  })
  
  output$summary <- renderPrint({
    summary(filtered_data())
  })
}
```

### Observers and Events

```r
server <- function(input, output, session) {
  
  # Observer (runs automatically)
  observe({
    print(paste("Slider value:", input$num))
  })
  
  # observeEvent (runs on specific event)
  observeEvent(input$submit, {
    showNotification("Form submitted!", type = "message")
  })
  
  # eventReactive (creates reactive value)
  data <- eventReactive(input$load, {
    read.csv(input$file$datapath)
  })
  
  output$table <- renderTable({
    data()
  })
}
```

### Reactive Values

```r
server <- function(input, output, session) {
  
  # Create reactive values
  values <- reactiveValues(
    count = 0,
    data = NULL
  )
  
  # Update reactive values
  observeEvent(input$increment, {
    values$count <- values$count + 1
  })
  
  observeEvent(input$load, {
    values$data <- read.csv(input$file$datapath)
  })
  
  # Use reactive values
  output$count <- renderText({
    values$count
  })
}
```

---

## Layouts and Themes

### Sidebar Layout

```r
ui <- fluidPage(
  titlePanel("My App"),
  
  sidebarLayout(
    sidebarPanel(
      # Inputs
      sliderInput("num", "Number:", 1, 100, 50)
    ),
    
    mainPanel(
      # Outputs
      plotOutput("plot")
    )
  )
)
```

### Tabset Layout

```r
ui <- fluidPage(
  titlePanel("Tabbed Interface"),
  
  tabsetPanel(
    tabPanel("Plot", plotOutput("plot")),
    tabPanel("Summary", verbatimTextOutput("summary")),
    tabPanel("Table", tableOutput("table")),
    tabPanel("About", includeMarkdown("about.md"))
  )
)
```

### Navigation Bar

```r
ui <- navbarPage(
  "My App",
  
  tabPanel("Home",
    h1("Welcome"),
    p("This is the home page.")
  ),
  
  tabPanel("Analysis",
    sidebarLayout(
      sidebarPanel(sliderInput("n", "N:", 1, 100, 50)),
      mainPanel(plotOutput("plot"))
    )
  ),
  
  tabPanel("Data",
    dataTableOutput("table")
  ),
  
  navbarMenu("More",
    tabPanel("Sub-page 1"),
    tabPanel("Sub-page 2")
  )
)
```

### Dashboard Layout

```r
library(shinydashboard)

ui <- dashboardPage(
  dashboardHeader(title = "My Dashboard"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Dashboard", tabName = "dashboard", icon = icon("dashboard")),
      menuItem("Data", tabName = "data", icon = icon("table")),
      menuItem("Settings", tabName = "settings", icon = icon("cog"))
    )
  ),
  
  dashboardBody(
    tabItems(
      tabItem(tabName = "dashboard",
        fluidRow(
          valueBox(123, "Total Sales", icon = icon("dollar"), color = "green"),
          valueBox(456, "Customers", icon = icon("users"), color = "blue"),
          valueBox(789, "Products", icon = icon("box"), color = "yellow")
        ),
        fluidRow(
          box(title = "Plot", plotOutput("plot"), width = 6),
          box(title = "Summary", verbatimTextOutput("summary"), width = 6)
        )
      ),
      
      tabItem(tabName = "data",
        dataTableOutput("table")
      )
    )
  )
)
```

### Themes

```r
library(shinythemes)

ui <- fluidPage(
  theme = shinytheme("flatly"),  # or cerulean, cosmo, darkly, etc.
  
  titlePanel("Themed App"),
  
  # Rest of UI
)

# Custom CSS
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      .title { color: #007bff; }
      .sidebar { background-color: #f8f9fa; }
    "))
  ),
  
  # Rest of UI
)
```

---

## Interactive Outputs

### Plotly (Interactive Plots)

```r
library(plotly)

server <- function(input, output, session) {
  output$plot <- renderPlotly({
    p <- ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl))) +
      geom_point() +
      theme_minimal()
    
    ggplotly(p)
  })
}

ui <- fluidPage(
  plotlyOutput("plot")
)
```

### DT (Interactive Tables)

```r
library(DT)

server <- function(input, output, session) {
  output$table <- renderDataTable({
    datatable(mtcars, 
              filter = 'top',
              options = list(
                pageLength = 10,
                searching = TRUE,
                ordering = TRUE
              ))
  })
}

ui <- fluidPage(
  dataTableOutput("table")
)
```

### Leaflet (Interactive Maps)

```r
library(leaflet)

server <- function(input, output, session) {
  output$map <- renderLeaflet({
    leaflet() %>%
      addTiles() %>%
      addMarkers(lng = -122.4194, lat = 37.7749, 
                 popup = "San Francisco")
  })
}

ui <- fluidPage(
  leafletOutput("map")
)
```

---

## Advanced Reactivity

### Isolate and Debounce

```r
server <- function(input, output, session) {
  
  # Isolate - prevent reactivity
  output$text <- renderText({
    input$button  # Trigger on button click
    isolate({
      paste("Value:", input$slider)  # Don't react to slider changes
    })
  })
  
  # Debounce - delay reactivity
  debounced_value <- debounce(reactive(input$text), 1000)  # 1 second delay
  
  output$result <- renderText({
    debounced_value()
  })
}
```

### Validation

```r
server <- function(input, output, session) {
  
  output$plot <- renderPlot({
    # Validate input
    validate(
      need(input$n > 0, "Please select a positive number"),
      need(input$n <= 1000, "Number too large")
    )
    
    hist(rnorm(input$n))
  })
}
```

### Progress Indicators

```r
server <- function(input, output, session) {
  
  observeEvent(input$run, {
    withProgress(message = 'Processing...', value = 0, {
      for (i in 1:10) {
        incProgress(1/10, detail = paste("Step", i))
        Sys.sleep(0.5)
      }
    })
    
    showNotification("Processing complete!", type = "message")
  })
}
```

### Modal Dialogs

```r
server <- function(input, output, session) {
  
  observeEvent(input$show_modal, {
    showModal(modalDialog(
      title = "Important Message",
      "Are you sure you want to continue?",
      footer = tagList(
        modalButton("Cancel"),
        actionButton("confirm", "Confirm")
      )
    ))
  })
  
  observeEvent(input$confirm, {
    removeModal()
    showNotification("Confirmed!", type = "message")
  })
}
```

---

## Shiny Modules

### Creating a Module

```r
# Module UI function
histogramUI <- function(id) {
  ns <- NS(id)
  
  tagList(
    sliderInput(ns("bins"), "Number of bins:", 1, 50, 30),
    plotOutput(ns("plot"))
  )
}

# Module server function
histogramServer <- function(id, data) {
  moduleServer(id, function(input, output, session) {
    output$plot <- renderPlot({
      hist(data(), breaks = input$bins, main = "Histogram")
    })
  })
}
```

### Using Modules

```r
ui <- fluidPage(
  titlePanel("Modular App"),
  
  tabsetPanel(
    tabPanel("Dataset 1", histogramUI("hist1")),
    tabPanel("Dataset 2", histogramUI("hist2"))
  )
)

server <- function(input, output, session) {
  data1 <- reactive(rnorm(100))
  data2 <- reactive(rnorm(100, mean = 10))
  
  histogramServer("hist1", data1)
  histogramServer("hist2", data2)
}

shinyApp(ui, server)
```

---

## Deployment

### shinyapps.io

```r
# Install rsconnect
install.packages("rsconnect")

# Set up account
library(rsconnect)
setAccountInfo(
  name = "your-account-name",
  token = "your-token",
  secret = "your-secret"
)

# Deploy
rsconnect::deployApp(appDir = "path/to/app")

# Update app
rsconnect::deployApp(appDir = "path/to/app", forceUpdate = TRUE)
```

### Shiny Server (Self-hosted)

```bash
# Install on Ubuntu
sudo apt-get install gdebi-core
wget https://download3.rstudio.org/ubuntu-18.04/x86_64/shiny-server-1.5.18.987-amd64.deb
sudo gdebi shiny-server-1.5.18.987-amd64.deb

# App location
/srv/shiny-server/

# Configuration
/etc/shiny-server/shiny-server.conf
```

### Docker

```dockerfile
# Dockerfile
FROM rocker/shiny:latest

# Install R packages
RUN R -e "install.packages(c('shiny', 'tidyverse', 'plotly'))"

# Copy app files
COPY app.R /srv/shiny-server/

# Expose port
EXPOSE 3838

# Run app
CMD ["/usr/bin/shiny-server"]
```

```bash
# Build and run
docker build -t my-shiny-app .
docker run -p 3838:3838 my-shiny-app
```

---

## Best Practices

### Performance Optimization

```r
# 1. Use reactive expressions for expensive computations
filtered_data <- reactive({
  # Expensive operation
  large_dataset %>% filter(category == input$category)
})

# 2. Cache results
library(memoise)
expensive_function <- memoise(function(x) {
  Sys.sleep(2)  # Simulate expensive operation
  x * 2
})

# 3. Use async for long-running tasks
library(promises)
library(future)
plan(multisession)

output$result <- renderText({
  future({
    # Long-running computation
    slow_function()
  }) %...>% {
    paste("Result:", .)
  }
})

# 4. Limit reactive updates
debounce(reactive(input$text), 1000)
```

### Code Organization

```r
# Project structure
# my_app/
# ├── app.R (or ui.R + server.R)
# ├── R/
# │   ├── modules.R
# │   ├── utils.R
# │   └── data_processing.R
# ├── data/
# │   └── dataset.csv
# ├── www/
# │   ├── styles.css
# │   └── logo.png
# └── README.md

# Source helper functions
source("R/utils.R")
source("R/modules.R")
```

### Security

```r
# 1. Validate user input
server <- function(input, output, session) {
  safe_input <- reactive({
    validate(need(is.numeric(input$value), "Must be numeric"))
    validate(need(input$value > 0, "Must be positive"))
    input$value
  })
}

# 2. Use environment variables for secrets
api_key <- Sys.getenv("API_KEY")

# 3. Sanitize file uploads
observeEvent(input$file, {
  validate(need(tools::file_ext(input$file$name) == "csv", "Only CSV files"))
})

# 4. Set session timeout
options(shiny.maxRequestSize = 30*1024^2)  # 30 MB limit
```

---

## Complete Example App

```r
library(shiny)
library(tidyverse)
library(plotly)
library(DT)

ui <- navbarPage(
  "Data Explorer",
  theme = shinythemes::shinytheme("flatly"),
  
  tabPanel("Data",
    sidebarLayout(
      sidebarPanel(
        fileInput("file", "Upload CSV", accept = ".csv"),
        hr(),
        uiOutput("variable_selector")
      ),
      mainPanel(
        dataTableOutput("data_table")
      )
    )
  ),
  
  tabPanel("Visualize",
    sidebarLayout(
      sidebarPanel(
        selectInput("x_var", "X Variable:", choices = NULL),
        selectInput("y_var", "Y Variable:", choices = NULL),
        selectInput("color_var", "Color by:", choices = NULL)
      ),
      mainPanel(
        plotlyOutput("scatter_plot", height = "600px")
      )
    )
  ),
  
  tabPanel("Summary",
    verbatimTextOutput("summary")
  )
)

server <- function(input, output, session) {
  
  # Load data
  data <- reactive({
    req(input$file)
    read.csv(input$file$datapath)
  })
  
  # Update variable selectors
  observe({
    req(data())
    vars <- names(data())
    numeric_vars <- names(select(data(), where(is.numeric)))
    
    updateSelectInput(session, "x_var", choices = numeric_vars)
    updateSelectInput(session, "y_var", choices = numeric_vars)
    updateSelectInput(session, "color_var", choices = vars)
  })
  
  # Data table
  output$data_table <- renderDataTable({
    req(data())
    datatable(data(), filter = 'top', options = list(pageLength = 25))
  })
  
  # Scatter plot
  output$scatter_plot <- renderPlotly({
    req(data(), input$x_var, input$y_var)
    
    p <- ggplot(data(), aes_string(x = input$x_var, y = input$y_var, 
                                    color = input$color_var)) +
      geom_point(size = 3, alpha = 0.6) +
      theme_minimal() +
      labs(title = paste(input$y_var, "vs", input$x_var))
    
    ggplotly(p)
  })
  
  # Summary
  output$summary <- renderPrint({
    req(data())
    summary(data())
  })
}

shinyApp(ui, server)
```

---

## Summary

### Skills Learned

- ✅ Shiny app structure (UI/Server)
- ✅ Input and output widgets
- ✅ Reactive programming
- ✅ Layouts and themes
- ✅ Interactive visualizations
- ✅ Shiny modules for reusability
- ✅ Deployment strategies
- ✅ Best practices and optimization

### Next Steps

1. **Build:** Create your own interactive dashboard
2. **Explore:** shinydashboard, shinyWidgets, shinyjs packages
3. **Deploy:** Share your app on shinyapps.io
4. **Learn:** Move to `5_Modules_and_Functions.md` for software engineering

### Resources

- [Shiny Gallery](https://shiny.rstudio.com/gallery/)
- [Mastering Shiny Book](https://mastering-shiny.org/)
- [Shiny Cheat Sheet](https://shiny.rstudio.com/articles/cheatsheet.html)

**Continue to Modules & Functions! 🚀**
