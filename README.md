Project Description:
This project predicts house prices using Decision Tree and Random Forest regression models. 
The dataset is from Kaggle: House Prices - Advanced Regression Techniques (Dataset attached)

_____

The workflow includes:
- Preprocessing numeric and categorical features
- Handling missing values and encoding categorical variables
- Building Decision Tree and Random Forest models
- Evaluating models using R², MAE, MSE
- Visualizing predicted vs actual house prices

_____

Dataset
- Source: Kaggle
- Features: numeric and categorical features related to houses (e.g., LotArea, YearBuilt, Neighborhood, HouseStyle)

_____

Preprocessing
- Numeric features: missing values filled with median
- Categorical features: missing values filled with most frequent value, one-hot encoding
- Combined using ColumnTransformer and Pipeline

_____

Models
1. Decision Tree Regressor
2. Random Forest Regressor

_____

Results

| Model          | R²    | MAE ($)  | MSE ($²)         |
|----------------|-------|----------|-----------------|
| Decision Tree  | 0.71  | 115,436  | 4.3e10          |
| Random Forest  | 0.85  | 72,590   | 2.2e10          |

_____

Plots
Predicted vs Actual Prices attached!

_____

Future Improvements
- Feature engineering (ratios, log transforms)
- Experimenting with XGBoost or LightGBM for better performance
