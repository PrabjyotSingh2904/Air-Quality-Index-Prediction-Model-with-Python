
# Air Quality Index (AQI) Prediction Model With Python

This repository contains a Python-based Machine Learning project aimed at predicting Air Quality Index (AQI) using multiple regression models. The dataset used comprises air quality parameters and AQI values recorded across various cities.

## Table of Contents

- [Overview](#overview)
- [Libraries](#libraries)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Machine Learning Models](#machine-learning-models)
- [Model Evaluation Metrics](#model-evaluation-metrics)
- [Results](#results)
- [How to Run the Project](#how-to-run-the-project)
- [License](#license)

## Overview

The goal of this project is to predict the AQI using machine learning techniques, based on several environmental factors. The dataset includes various pollutants such as `PM2.5`, `PM10`, `NO2`, `O3`, `SO2`, and others. We used multiple regression models and compared their performance to choose the best one for AQI prediction.

##  Libraries

  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `warnings`

## Dataset

The dataset(`air quality data.csv`) contains the following columns:
- **Air Quality Parameters**:
  - `PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`
- **Other Features**:
  - `City`: The city where the data was recorded.
  - `Date`: The date the measurements were taken (removed for modeling).
  - `AQI`: The target variable (Air Quality Index).
  - `AQI_Bucket`: Categorized AQI (e.g., Good, Moderate, Unhealthy).



## Data Preprocessing

1. **Handling Missing Values**: Rows with missing values for AQI were dropped. Remaining missing values were replaced using the mean for each column.
2. **Outlier Detection**: Outliers in numerical columns were replaced using the IQR method.
3. **Feature Scaling**: StandardScaler was used to normalize the dataset.

## Exploratory Data Analysis (EDA)

Several analyses were performed:
- Univariate: Histograms for air quality parameters.
- Bivariate: Relationships between various parameters and AQI using boxplots and scatterplots.
- Multivariate: Heatmap for correlations between features.

## Machine Learning Models

We trained the following models to predict AQI:

1. Linear Regression
2. K-Nearest Neighbors (KNN) Regressor
3. Decision Tree Regressor
4. Random Forest Regressor

### Note:
- The data was split into 80% training and 20% testing sets.
- Each model was trained on the training set and predictions were made on both the training and test sets.

## Model Evaluation Metrics

- **Root Mean Squared Error (RMSE)**: Measures the average magnitude of the errors between predicted and actual values.
- **R² Score**: Indicates the proportion of the variance in the target variable (AQI) that is explained by the model.

## Results

The **Random Forest Regressor** achieved the best performance with the lowest RMSE and highest R² score.


## How to Run the Project

1. Clone the repository.
   ```bash
   git clone https://github.com/PrabjyotSingh2904/Air-Quality-Index-Prediction-Model-with-Python.git
   ```

2. Install the required Python libraries
    ```bash
     pip install numpy pandas matplotlib seaborn scikit-learn
    ```

3. Place the dataset(`air quality data.csv`) in the project directory
  
4. Run the notebook or script

## License
This project is open-source and available under the MIT License.
