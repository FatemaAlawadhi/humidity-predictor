import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import KNNImputer
import joblib
from time import time

class HumidityPredictor:
    def __init__(self):
        try:
            self.scaler = RobustScaler()
            self.imputer = KNNImputer(n_neighbors=5)
            
            # GBM model
            self.model = GradientBoostingRegressor(
                n_estimators=2000,         
                learning_rate=0.008,       
                max_depth=10,               
                min_samples_split=4,       
                min_samples_leaf=3,         
                subsample=0.85,             
                max_features=0.8,          
                loss='huber',               
                alpha=0.95,                 
                validation_fraction=0.1,    
                n_iter_no_change=50,        
                tol=1e-4,                  
                random_state=42,
                warm_start=True          
            )
            
        except Exception as e:
            raise Exception(f"Failed to initialize predictor: {str(e)}")
    
    def engineer_features(self, X):
        try:
            X = X.copy()
            
            # 1. Basic Statistical Features
            sensor_columns = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
            X['sensor_mean'] = X[sensor_columns].mean(axis=1)
            X['sensor_std'] = X[sensor_columns].std(axis=1)
            X['sensor_max'] = X[sensor_columns].max(axis=1)
            X['sensor_min'] = X[sensor_columns].min(axis=1)
            
            # 2. Enhanced Polynomial Features
            X['CO_cubed'] = X['CO(GT)'] ** 3
            X['NOx_cubed'] = X['NOx(GT)'] ** 3
            X['CO_squared'] = X['CO(GT)'] ** 2
            X['NOx_squared'] = X['NOx(GT)'] ** 2
            X['NO2_squared'] = X['NO2(GT)'] ** 2
            
            # 3. Enhanced Interaction Terms
            X['CO_NOx_interaction'] = X['CO(GT)'] * X['NOx(GT)']
            X['CO_NO2_interaction'] = X['CO(GT)'] * X['NO2(GT)']
            X['NO2_O3_interaction'] = X['NO2(GT)'] * X['PT08.S5(O3)']
            X['NOx_O3_interaction'] = X['NOx(GT)'] * X['PT08.S5(O3)']
            X['NOx_NO2_interaction'] = X['NOx(GT)'] * X['NO2(GT)']
            X['CO_O3_interaction'] = X['CO(GT)'] * X['PT08.S5(O3)']
            
            # 4. Enhanced Ratios and Differences
            X['NOx_to_NO2_ratio'] = X['NOx(GT)'] / np.maximum(X['NO2(GT)'], 1e-6)
            X['CO_to_NOx_ratio'] = X['CO(GT)'] / np.maximum(X['NOx(GT)'], 1e-6)
            X['CO_to_NO2_ratio'] = X['CO(GT)'] / np.maximum(X['NO2(GT)'], 1e-6)
            X['O3_to_NOx_ratio'] = X['PT08.S5(O3)'] / np.maximum(X['NOx(GT)'], 1e-6)
            X['NO2_NOx_diff'] = X['NOx(GT)'] - X['NO2(GT)']
            
            # 5. Enhanced Log Transforms
            X['log_CO'] = np.log1p(np.abs(X['CO(GT)']))
            X['log_NOx'] = np.log1p(np.abs(X['NOx(GT)']))
            X['log_NO2'] = np.log1p(np.abs(X['NO2(GT)']))
            X['log_O3'] = np.log1p(np.abs(X['PT08.S5(O3)']))
            
            # 6. Normalized Ratios
            X['CO_normalized'] = X['CO(GT)'] / X['sensor_mean']
            X['NOx_normalized'] = X['NOx(GT)'] / X['sensor_mean']
            X['NO2_normalized'] = X['NO2(GT)'] / X['sensor_mean']
            
            # Add these new features:
            sensor_columns = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)']
            
            # 7. Enhanced Sensor Statistics
            X['sensor_mean'] = X[sensor_columns].mean(axis=1)
            X['sensor_std'] = X[sensor_columns].std(axis=1)
            X['sensor_max'] = X[sensor_columns].max(axis=1)
            X['sensor_min'] = X[sensor_columns].min(axis=1)
            X['sensor_range'] = X['sensor_max'] - X['sensor_min']
            X['sensor_q75'] = X[sensor_columns].quantile(0.75, axis=1)
            X['sensor_q25'] = X[sensor_columns].quantile(0.25, axis=1)
            X['sensor_iqr'] = X['sensor_q75'] - X['sensor_q25']
            
            # 8. Enhanced Ratios
            X['NOx_to_NO2_ratio'] = X['NOx(GT)'] / (X['NO2(GT)'] + 1e-6)
            X['CO_to_NOx_ratio'] = X['CO(GT)'] / (X['NOx(GT)'] + 1e-6)
            X['CO_to_NO2_ratio'] = X['CO(GT)'] / (X['NO2(GT)'] + 1e-6)
            X['O3_to_NOx_ratio'] = X['PT08.S5(O3)'] / (X['NOx(GT)'] + 1e-6)
            
            # 9. Enhanced Differences
            X['NO2_NOx_diff'] = X['NOx(GT)'] - X['NO2(GT)']
            X['CO_NOx_diff'] = X['CO(GT)'] - X['NOx(GT)']
            X['CO_NO2_diff'] = X['CO(GT)'] - X['NO2(GT)']
            
            # 10. Sensor Cross-Correlations
            for i in range(len(sensor_columns)):
                for j in range(i+1, len(sensor_columns)):
                    col1, col2 = sensor_columns[i], sensor_columns[j]
                    name1, name2 = col1.split('(')[0], col2.split('(')[0]
                    X[f'{name1}_{name2}_ratio'] = X[col1] / (X[col2] + 1e-6)
                    X[f'{name1}_{name2}_diff'] = X[col1] - X[col2]
            
            # 11. Enhanced Composite Indices
            X['pollution_index'] = (X['CO(GT)'] + X['NOx(GT)'] + X['NO2(GT)']) / 3
            X['weighted_pollution_index'] = (
                0.4 * X['CO(GT)'] + 
                0.35 * X['NOx(GT)'] + 
                0.25 * X['NO2(GT)']
            )
            
            # 12. Seasonal Interaction Features
            seasonal_columns = ['Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter']
            for season in seasonal_columns:
                X[f'pollution_index_{season}'] = X['pollution_index'] * X[season]
                X[f'weighted_pollution_{season}'] = X['weighted_pollution_index'] * X[season]
            
            # 13. Normalized Features
            X['CO_normalized'] = X['CO(GT)'] / X['sensor_mean']
            X['NOx_normalized'] = X['NOx(GT)'] / X['sensor_mean']
            X['NO2_normalized'] = X['NO2(GT)'] / X['sensor_mean']
            
            # 14. Variance-based Features
            X['pollution_stability'] = X[['CO(GT)', 'NOx(GT)', 'NO2(GT)']].std(axis=1)
            X['sensor_stability'] = X[sensor_columns].std(axis=1) / (X[sensor_columns].mean(axis=1) + 1e-6)
            
            return X
            
        except Exception as e:
            raise Exception(f"Feature engineering failed: {str(e)}")
    
    def preprocess_data(self, X, is_training=True):
        try:
            if X.empty:
                raise ValueError("Empty input data")
                
            # Only fit the imputer during training
            if is_training:
                X_imputed = self.imputer.fit_transform(X)
            else:
                X_imputed = self.imputer.transform(X)
            X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
            
            X_engineered = self.engineer_features(X_imputed)
            
            # Only fit the scaler during training
            if is_training:
                X_scaled = self.scaler.fit_transform(X_engineered)
            else:
                X_scaled = self.scaler.transform(X_engineered)
            X_scaled = pd.DataFrame(X_scaled, columns=X_engineered.columns)
            
            return X_scaled
            
        except Exception as e:
            raise Exception(f"Data preprocessing failed: {str(e)}")
    
    def train(self, X_train, y_train):
        try:
            print("Starting model training...")
            
            # Get processed data with is_training=True
            X_processed = self.preprocess_data(X_train, is_training=True)
            
            # Calculate cross-validation scores
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            print("\nCalculating cross-validation scores...")
            
            scoring = {
                'r2': 'r2',
                'mse': 'neg_mean_squared_error',
                'mae': 'neg_mean_absolute_error',
                'explained_variance': 'explained_variance',
                'max_error': 'max_error'
            }
            
            # Detailed cross-validation
            scores = cross_validate(
                self.model, 
                X_processed, 
                y_train, 
                cv=cv, 
                scoring=scoring,
                n_jobs=-1,
                return_train_score=True  
            )
            
            print("\nModel Performance Metrics:")
            print("\nTraining Metrics:")
            print(f"R² Score: {scores['train_r2'].mean():.4f} (±{scores['train_r2'].std()*2:.4f})")
            print(f"MSE: {-scores['train_mse'].mean():.4f} (±{scores['train_mse'].std()*2:.4f})")
            print(f"MAE: {-scores['train_mae'].mean():.4f} (±{scores['train_mae'].std()*2:.4f})")
            print(f"Explained Variance: {scores['train_explained_variance'].mean():.4f}")
            print(f"Max Error: {-scores['train_max_error'].mean():.4f}")
            
            print("\nValidation Metrics:")
            print(f"R² Score: {scores['test_r2'].mean():.4f} (±{scores['test_r2'].std()*2:.4f})")
            print(f"MSE: {-scores['test_mse'].mean():.4f} (±{scores['test_mse'].std()*2:.4f})")
            print(f"MAE: {-scores['test_mae'].mean():.4f} (±{scores['test_mae'].std()*2:.4f})")
            print(f"Explained Variance: {scores['test_explained_variance'].mean():.4f}")
            print(f"Max Error: {-scores['test_max_error'].mean():.4f}")
            
            # Train the final model
            print("\nTraining final model...")
            start_time = time()
            self.model.fit(X_processed, y_train)
            training_time = time() - start_time
            
            print(f"\nTraining completed in {training_time:.2f} seconds")
            
            # Calculate feature importances
            feature_importance = pd.DataFrame({
                'feature': X_processed.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
            
            return self.model

        except Exception as e:
            raise Exception(f"Model training failed: {str(e)}")
    
    def predict(self, test_data):
        try:
            print(f"Test data shape before preprocessing: {test_data.shape}")
            # Process test data with is_training=False
            X_processed = self.preprocess_data(test_data, is_training=False)
            print(f"Test data shape after preprocessing: {X_processed.shape}")
            
            # Make predictions using GBM model
            predictions = self.model.predict(X_processed)
            print(f"Predictions shape: {predictions.shape}")
            
            # Ensure predictions match the original test data length
            if len(predictions) != len(test_data):
                raise ValueError(f"Prediction length ({len(predictions)}) doesn't match test data length ({len(test_data)}). "
                               f"This might be caused by dropped rows during preprocessing.")
            
            # Return predictions in the expected format
            return predictions
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
    
    def print_feature_importance(self, feature_names):
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nFeature Ranking:")
        for f in range(len(feature_names)):
            print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], 
                                  importances[indices[f]]))
    
    def save_model(self, path):
        joblib.dump((self.model, self.scaler, self.imputer), path)
    
    @classmethod
    def load_model(cls, path):
        predictor = cls()
        predictor.model, predictor.scaler, predictor.imputer = joblib.load(path)
        return predictor
    