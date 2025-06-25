
# California House Price Prediction Model

## Introduction
We are given a dataset which includes several features which may be affecting the prices of houses in different districts.

Our task is to build a ML Model which can predict the house prices, this is clearly a supervised learning task as we already have the median house price for the districts given.

I've tried to follow the best practices for data preparation, visualization, modeling, evaluation, and fine-tuning.

The implementation is fully documented within a Jupyter Notebook, where **each step is explained clearly through comments and markdown cells** to ensure clarity on *what is being done and why*.

##  Dataset Description

The dataset is the **California Housing Dataset**, which contains real-world housing data for districts in California. It includes both numerical and categorical features describing demographic and geographical factors.


###  Features:

| Feature | Description |
|---------|-------------|
| `longitude` | Longitude of the district |
| `latitude` | Latitude of the district |
| `housing_median_age` | Median age of houses |
| `total_rooms` | Total rooms in the district |
| `total_bedrooms` | Total bedrooms in the district |
| `population` | Population in the district |
| `households` | Number of households |
| `median_income` | Median income in the district |
| `ocean_proximity` | Categorical proximity to ocean |
| `median_house_value` | Target variable to predict |

---



## Project Summary

### 1. **Data Exploration & Visualization**
- Inspected structure using `.info()`, `.describe()`, and histograms.
- Created geographical scatterplots to visualize housing prices and population.
- Computed correlation matrix to identify most predictive features (`median_income` showed highest correlation with price).

### 2. **Data Preprocessing**
- Handled missing values with `SimpleImputer`.
- Encoded categorical attributes using `OneHotEncoder`.
- Scaled numeric features using `StandardScaler`.
- Built custom transformers to better normalise data using `FunctionTransformer`.
- Built a **full preprocessing pipeline** using `ColumnTransformer`.

### 3. **Model Training**
- Trained multiple models:
  - **Linear Regression**
  - **Decision Tree**
  - **Random Forest**
- Used cross-validation to assess performance and avoid overfitting.

### 4. **Model Evaluation**
- Compared models using RMSE and cross-validated scores.
- Evaluated final model on a held-out test set.
- Performed residual analysis to identify patterns in prediction errors.

### 5. **Hyperparameter Tuning**
- Used `GridSearchCV` and `RandomizedSearchCV` to optimize hyperparameters (especially for Random Forest).
- Selected best model using validation score.

## Conclusion
The model was able to RMSE of $41422 which is a satisfactory result.

## Acknowledgement
Inspired by the structure and lessons in Hands-On Machine Learning by Aurélien Géron.



