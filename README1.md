# ANA500 Final Project  
## Airline Passenger Satisfaction and Arrival Delay Regression

**Student:** Scyler Ford  
**Course:** ANA500 - Python for Data Science  

## Project Overview

This final project combines ANA500 Micro-Projects 1 through 4 into one complete end-to-end data science project. The analysis uses the Airline Passenger Satisfaction dataset and follows the full data science process from acquisition and preparation through visualization, traditional machine-learning regression, deep-learning regression, reporting, and action recommendations.

The final predictive target is:

**Arrival Delay in Minutes**

## Problem Statement

Airlines collect passenger survey data, service ratings, travel information, and delay information. This project uses that data to describe passenger experience patterns and determine whether arrival delay length can be predicted using flight and passenger-related variables.

## Hypothesis

This project uses two related hypotheses:

1. Passenger satisfaction is expected to be higher when service ratings such as online boarding, seat comfort, inflight entertainment, and onboard service are higher.
2. Departure Delay in Minutes is expected to be the strongest predictor of Arrival Delay in Minutes. A deep neural network may slightly improve prediction compared with traditional regression, but Linear Regression is expected to remain a strong baseline because departure delay and arrival delay are strongly related.

## Dataset

The project starts with the original raw airline dataset:

- Original records: **129,880**
- Original columns: **25**
- Missing Arrival Delay records: **393**
- Cleaned regression records: **129,487**

The 393 missing `Arrival Delay in Minutes` rows were reviewed before deletion. They represented only **0.30%** of the dataset, but they had a higher mean departure delay than records with known arrival delay. They were removed because supervised regression requires a known target value, and this limitation is documented in the project.

## Data Science Process

### Step 1 - Acquire

The original raw airline passenger dataset was loaded and reviewed. The dataset includes passenger demographics, customer type, travel type, flight class, flight distance, service ratings, departure delay, arrival delay, and passenger satisfaction.

### Step 2 - Prepare

The preparation process included:

- Removing identifier fields: `Unnamed: 0` and `id`
- Converting important columns to numeric format
- Creating a missing-target flag using NumPy
- Investigating the 393 missing arrival-delay records
- Creating `satisfaction_binary` for exploratory analysis
- Removing missing target rows for regression modeling
- One-hot encoding categorical predictors
- Scaling numeric predictors
- Splitting the data into training and testing sets

### Step 3 - Analyze

The project includes exploratory analysis and predictive modeling.

Exploratory analysis examined:

- Passenger satisfaction distribution
- Satisfaction by flight class
- Service-rating correlations with satisfaction
- Arrival-delay distribution
- Arrival-delay outliers

Regression models compared:

1. Multiple Linear Regression
2. Linear Support Vector Regression
3. Deep Neural Network Regression

All three models predict the same continuous target, `Arrival Delay in Minutes`, and use consistent evaluation metrics.

### Step 4 - Report

Models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Multiple Linear Regression | 5.341 | 9.980 | 0.9419 |
| LinearSVR | 4.926 | 10.101 | 0.9405 |
| Deep Neural Network | 5.433 | 9.832 | 0.9436 |

The deep neural network had the best RMSE and R², while LinearSVR had the lowest MAE. The improvement from deep learning was small, so Linear Regression remains a strong and explainable baseline.

## Professor Feedback Addressed

The final project addresses previous feedback and grading criteria by:

- Starting from the original raw dataset instead of an already-cleaned file
- Improving the problem statement so it focuses on the data science/business problem
- Investigating the 393 missing arrival-delay records before deletion
- Using both Pandas and NumPy
- Adding deeper EDA and visualizations
- Comparing traditional regression and deep learning on the same target
- Evaluating model performance across delay ranges, including severe delays
- Testing the effect of removing Departure Delay in Minutes to address prediction timing
- Including a complete GitHub repository with notebook, README, data, presentation, and code PDF

## Severe Delay Review

Model error increased as actual delays became more severe. This is important because severe delays may matter more operationally than small routine delays.

For the **120+ minute** delay group:

- Records: **676**
- Mean actual delay: **199.18 minutes**
- Linear Regression MAE: **16.67 minutes**
- LinearSVR MAE: **18.69 minutes**
- Deep Neural Network MAE: **17.44 minutes**

## Departure Delay Timing Review

The strong model results depend heavily on knowing `Departure Delay in Minutes`. When departure delay was removed from the deep-learning model:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| DNN with Departure Delay | 5.433 | 9.832 | 0.9436 |
| DNN without Departure Delay | 20.495 | 41.413 | -0.0006 |

This shows that the model is useful after departure-delay information is available, but weak for early prediction if that variable is not known.

## Final Recommendations

Based on the final project:

1. Use the arrival-delay model after departure-delay information is available.
2. Track severe delays separately because overall averages can hide larger errors.
3. Use Linear Regression as a strong explainable baseline.
4. Add weather, airport congestion, aircraft status, crew status, and route-level variables if earlier prediction is needed.

## Files in This Repository

- `README.md`
- `Scyler_Ford_ANA500_FinalProject.ipynb`
- `airline_raw.csv`
- `airline_cleaned_for_final_project.csv`
- `Scyler_Ford_ANA500_FinalProject.pptx`
- `Scyler_Ford_ANA500_FinalProject.pdf`
- `Scyler_Ford_ANA500_FinalProject_Code.pdf`

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- GitHub

## How to Run

1. Download or clone the repository.
2. Open `Scyler_Ford_ANA500_FinalProject.ipynb` in Jupyter Notebook.
3. Make sure `airline_raw.csv` is in the same folder as the notebook.
4. Run the notebook from top to bottom.
5. The notebook creates `airline_cleaned_for_final_project.csv` during the preparation step.

