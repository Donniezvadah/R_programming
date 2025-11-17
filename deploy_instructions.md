# Deploying Your Shiny App to shinyapps.io

## Step 1: Create shinyapps.io Account
1. Go to [shinyapps.io](https://www.shinyapps.io/)
2. Sign up for a free account (you get 5 free apps with 25 active hours/month)
3. Verify your email

## Step 2: Connect RStudio to shinyapps.io
```r
# In R console, run:
library(rsconnect)

# Set up your account (replace with your details)
rsconnect::setAccountInfo(
  name = "your_username", 
  token = "your_token", 
  secret = "your_secret"
)
```

## Step 3: Deploy Your App
```r
# Navigate to your project directory and run:
rsconnect::deployApp("Forecasting.Rmd")
```

## Alternative: Use RStudio IDE
1. Open `Forecasting.Rmd` in RStudio
2. Click "Publish" button in the top-right
3. Select "shinyapps.io"
4. Follow the authentication steps
5. Click "Publish"

## What You'll Get
- **Live code execution** - Users can run R code in real-time
- **Interactive plots** - Zoom, pan, hover capabilities  
- **Parameter adjustment** - Modify inputs and see results
- **Shareable link** - Anyone can access your interactive app

## Free Plan Limitations
- 5 applications
- 25 active hours per month
- Apps sleep after 15 minutes of inactivity
- Some limitations on resources

## Next Steps
After deployment, you'll get a URL like:
`https://your_username.shinyapps.io/forecasting/`

You can then add this link to your Quarto book for users to access the live version.
