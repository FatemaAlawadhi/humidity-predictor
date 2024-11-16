import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_training import HumidityPredictor
import sys
import os

def explore_data():
    try:
        print("Starting data exploration...")
        
        # Load the datasets
        try:
            train_data = pd.read_csv('./data/TrainData.csv', sep='|')
            test_data = pd.read_csv('./data/TestData.csv', sep='|')
        except FileNotFoundError as e:
            raise Exception(f"Data files not found: {str(e)}")
        except pd.errors.EmptyDataError:
            raise Exception("One or both data files are empty")
        
        # Convert comma decimal separators to periods
        try:
            comma_decimal_columns = ['CO(GT)', 'RH']
            for col in comma_decimal_columns:
                train_data[col] = train_data[col].astype(str).str.replace(',', '.').astype(float)
                if col in test_data.columns:
                    test_data[col] = test_data[col].astype(str).str.replace(',', '.').astype(float)
        except ValueError as e:
            raise Exception(f"Error converting decimal format: {str(e)}")
            
        # Create one-hot encoded columns for Season
        season_encoded = pd.get_dummies(train_data['Season'], prefix='Season')
        
        # Drop original Season column and add encoded columns
        train_data = train_data.drop('Season', axis=1)
        train_data = pd.concat([train_data, season_encoded], axis=1)
        
        # Repeat for test data
        season_encoded_test = pd.get_dummies(test_data['Season'], prefix='Season')
        test_data = test_data.drop('Season', axis=1)
        test_data = pd.concat([test_data, season_encoded_test], axis=1)
        
        # Display basic information
        print("\nTraining Data Shape:", train_data.shape)
        print("Test Data Shape:", test_data.shape)
        print("\nTraining Data Columns:", train_data.columns.tolist())
        
        # Check for missing values
        print("\nMissing Values in Training Data:")
        print(train_data.isnull().sum())
        
        # Create correlation matrix heatmap with correct path
        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'correlation_matrix.png')
        
        # Create and save correlation matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix of Features')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Correlation matrix saved to: {output_path}")
        
        return train_data, test_data
        
    except Exception as e:
        print(f"Error during data exploration: {str(e)}")
        raise

def train_and_predict(train_data, test_data):
    try:
        print("\nStarting model training and prediction...")
        
        if train_data.empty or test_data.empty:
            raise ValueError("Training or test data is empty")
            
        # Clean data and prepare features
        train_data_clean = train_data.dropna()
        if len(train_data_clean) < len(train_data):
            print(f"Warning: Removed {len(train_data) - len(train_data_clean)} rows with missing values")
        
        if len(train_data_clean) == 0:
            raise ValueError("No valid training data after cleaning")
            
        X_train = train_data_clean.drop(['RH'], axis=1)
        y_train = train_data_clean['RH']
        
        # Ensure test data has same columns
        missing_cols = set(X_train.columns) - set(test_data.columns)
        if missing_cols:
            raise ValueError(f"Test data missing columns: {missing_cols}")
            
        test_features = test_data[X_train.columns]
        
        # Train and predict
        predictor = HumidityPredictor()
        predictor.train(X_train, y_train)
        predictions = predictor.predict(test_features)
        
        # Save model with absolute path
        try:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model.joblib')
            predictor.save_model(model_path)
            print(f"Model saved successfully to: {model_path}")
        except Exception as e:
            print(f"Warning: Failed to save model: {str(e)}")
        
        return predictions
        
    except Exception as e:
        print(f"Error during model training and prediction: {str(e)}")
        raise

def create_submission(predictions, test_data):
    try:
        print("\nCreating submission file...")
        
        if len(predictions) != len(test_data):
            raise ValueError("Predictions length doesn't match test data length")
            
        submission = pd.DataFrame({
            'ID': test_data['ID'],
            'RH': predictions
        })
        
        submission.to_csv('./submission.csv', index=False)
        print("Submission file created successfully!")
        
    except Exception as e:
        print(f"Error creating submission file: {str(e)}")
        raise

def main():
    try:
        print("Starting humidity prediction project...")
        
        train_data, test_data = explore_data()
        predictions = train_and_predict(train_data, test_data)
        create_submission(predictions, test_data)
        
        print("\nProject completed successfully!")
        
    except Exception as e:
        print(f"\nProject failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()