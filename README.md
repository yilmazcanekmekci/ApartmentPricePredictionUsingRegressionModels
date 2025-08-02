# ApartmentPricePredictionUsingRegressionModels

# 1. Apartment Price Prediction with Machine Learning

This repository contains a complete machine learning pipeline for predicting apartment prices using structured tabular data. The project combines spatial, temporal, structural, and macroeconomic features to develop a robust regression model. The final model is based on XGBoost, selected for its high predictive power and interpretability.

## 2. Project Overview

Real estate price estimation plays a key role for developers, investors, and individual buyers. This project aims to build an accurate and scalable regression model using advanced preprocessing, feature engineering, and machine learning techniques.

Key points:
- Detailed data preprocessing and EDA
- Feature engineering with domain knowledge
- Multiple regression model comparison with cross-validation
- Hyperparameter tuning using Optuna
- Best-performing model: XGBoost Regressor witl lowest RMSE and highest R²

## 3. Dataset

- Observations: 156,454
- Features: 34 original features, extended with engineered ones
- Target: `price_z` 

Feature types include:
- Numerical: area, age, floor number, price per m²
- Categorical: object type, ownership type, material
- Geospatial: distances to clinics, schools, transport
- Temporal: year built, source date, market volatility

## 4. Exploratory Data Analysis (EDA)

- Missing values identified in critical fields like `cond_class`, `build_mat`, `floor_no`
- Many features exhibited right-skewed distributions
- Correlation matrix used to detect multicollinearity
- Boxplots and scatter plots analyzed categorical and numerical effects on price
- Outliers handled via IQR and Z-score methods

## 5. Data Preprocessing and Feature Engineering

- Temporal features extracted (e.g., month, year from source)
- Missing values filled using median, mode, or placeholder categories
- New features derived:
  - `age`, `floor_ratio`, `rooms_per_m2`, `price_per_m2`
- Categorical variables encoded:
  - One-hot encoding for low-cardinality
  - Frequency encoding for `loc_code`
- Skewed variables transformed using `log1p`
- Highly correlated features removed
- Standard scaling applied (for linear models)

Exported datasets:
- `train_fe.csv`, `test_fe.csv`: preprocessed, unscaled
- `train_fe_scaled.csv`, `test_fe_scaled.csv`: scaled versions

## 6. Modeling

Models Evaluated with 5-Fold Cross Validation

| Model               | RMSE           | MAE            | R²       | Scaled |
|--------------------|----------------|----------------|----------|--------|
| XGBoost Regressor  | 93,259 ± 1,146 | 67,799 ± 486   | 0.9530   | No     |
| LightGBM Regressor | 93,352 ± 1,051 | 68,009 ± 444   | 0.9529   | No     |
| Linear Regression  | 170,105 ± 2,707| 118,581 ± 626  | 0.8436   | Yes    |
| Ridge Regression   | 170,105 ± 2,707| 118,579 ± 626  | 0.8436   | Yes    |
| Lasso Regression   | 170,105 ± 2,707| 118,567 ± 625  | 0.8436   | Yes    |
| SVR (Linear Kernel)| 226,843 ± 4,347| 126,795 ± 1,136| 0.7219   | Yes    |

The best performance was achieved using XGBoost with optimized hyperparameters (via Optuna, 50 trials). It effectively handled non-linearity, multicollinearity, and complex feature interactions.

## 7. Results

### 7.1 Train Set Performance

- XGBoost achieved R² of 0.9681 on the training set.
- Predictions closely aligned with actual values.
- Slight bias observed for high-priced apartments.

<img width="446" height="373" alt="image" src="https://github.com/user-attachments/assets/0d51520d-c186-40dd-9955-571e803f4a14" />

### 7.2 Residual Analysis

- Residuals follow a bell-shaped, symmetric distribution.
- Larger deviations exist at the high-price extremes.

<img width="596" height="350" alt="image" src="https://github.com/user-attachments/assets/6e7a0b5c-023e-454f-9ea7-3894f711bfef" />

### 7.3 Test Set Prediction

- Final predictions made using the trained XGBoost model.
- Results saved in `predicted_prices_using_xgboost.csv`
- Test target values were not available for evaluation.

## 8. Challenges

- High dimensionality with heterogeneous data types
- Missing values in key categorical fields
- Highly skewed variables required transformations
- Multicollinearity handled via feature selection
- Slight prediction drift for high-priced apartments
- Repeated `unit_id` values increased data complexity

## 9. Conclusion

A robust and interpretable machine learning model was developed for apartment price prediction. After extensive preprocessing and model evaluation, XGBoost achieved the best performance. The pipeline is adaptable and scalable for future data and can be extended to include more spatial or economic indicators.

## 10. Reference

Sönmez, N., & Günaydın, H. M. (2024).  
"Using Machine Learning Algorithms for Interpretable Predictions of House Prices and Variable Features".  
8th International Project and Construction Management Conference (IPCMC2024), İstanbul, Türkiye.  
[ResearchGate Link](https://www.researchgate.net/publication/390301420_Using_Machine_Learning_Algorithms_for_Interpretable_Predictions_of_House_Prices_and_Variable_Features)

## Contact
For questions or contributions, please open an issue or contact [yilmazcanekmekci@gmail.com].

