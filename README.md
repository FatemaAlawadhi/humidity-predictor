# Humidity Predictor

A machine learning project that predicts relative humidity (RH) using various environmental sensor data and seasonal information.

## Overview

This project implements a Gradient Boosting Regressor model to predict humidity levels based on multiple environmental sensors' readings and seasonal data. The model achieves an R² score of approximately 0.98, indicating high prediction accuracy.


## Features

- Advanced feature engineering with over 50 derived features
- Robust data preprocessing and cleaning
- Sophisticated model training with cross-validation
- Detailed performance metrics and feature importance analysis
- Model persistence (save/load functionality)
- Comprehensive error handling and logging

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## Installation

1. Clone the repository:
```
git clone https://github.com/FatemaAlawadhi/humidity-predictor.git
```

2. Navigate to the project directory:
```
cd humidity-predictor
```

3. Install required packages:
```
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```


## Usage
Run the prediction model:
```
python3 src/main.py
```


## Data Format

### Input Data Columns
- ID: Unique identifier
- CO(GT): CO concentration
- PT08.S1(CO): Tin oxide sensor response
- PT08.S2(NMHC): Titania sensor response
- NMHC(GT): Non-metanic hydrocarbons concentration
- NOx(GT): NOx concentration
- PT08.S3(NOx): Tungsten oxide sensor response
- NO2(GT): NO2 concentration
- PT08.S4(NO2): Tungsten oxide sensor response
- PT08.S5(O3): Indium oxide sensor response
- Season: Categorical (Fall, Spring, Summer, Winter)

### Target Variable
- RH: Relative Humidity

## Model Details

### Gradient Boosting Regressor Parameters
- n_estimators: 2000
- learning_rate: 0.008
- max_depth: 10
- min_samples_split: 4
- min_samples_leaf: 3
- subsample: 0.85
- max_features: 0.8
- loss: 'huber'
- alpha: 0.95

### Feature Engineering
- Basic statistical features
- Polynomial features
- Interaction terms
- Ratios and differences
- Log transforms
- Normalized ratios
- Sensor statistics
- Cross-correlations
- Composite indices
- Seasonal interactions

## Performance

The model achieves:
- Model Performance Metrics:
    - Training Metrics:
        - R² Score: 0.9974 (±0.0003)
        - MSE: 6.5052 (±0.7886)
        - MAE: 1.0063 (±0.0316)
    - Validation Metrics:
        - R² Score: 0.9764 (±0.0027)
        - MSE: 58.7642 (±5.0737)
        - MAE: 5.6921 (±0.3282)
- Detailed feature importance analysis
    - Top 10 Most Important Features:
    - feature                               importance
    - PT08.S2_PT08.S5_ratio                 0.142590
    - PT08.S2_PT08.S5_diff                  0.091227
    - PT08.S1_PT08.S2_diff                  0.066338
    - PT08.S2_PT08.S4_diff                  0.065301
    - NOx_to_NO2_ratio                      0.051420
    - PT08.S2_PT08.S4_ratio                 0.047525    
    - PT08.S2_NMHC_to_PT08.S1_CO_Ratio      0.036310
    - PT08.S1_PT08.S2_ratio                 0.025658
    - NO2_normalized                        0.023652
    - PT08.S4_PT08.S5_diff                  0.022099

## Output

The program generates:
- submission.csv: Contains predictions for test data
- correlation_matrix.png: Visualization of feature correlations
- model.joblib: Saved model file

## Error Handling

The project includes comprehensive error handling for:
- Missing data files
- Invalid data formats
- Data preprocessing issues
- Model training failures
- Prediction errors

## AUTHOR

- Fatema Alawadhi

## License
This project was developed for the Zain Challenge in the Machine Learning Olympiad.
