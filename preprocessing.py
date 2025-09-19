import pandas as pd

# Load the dataset
df = pd.read_csv("house_prices.csv")

# Quick look
print(df.shape)       # rows, columns
print(df.head())      # first rows

X = df.drop("price", axis=1)  # features
Y = df["price"]               # target

#Arranging features based on type
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Categorical pipeline: fill missing values → encode
cat_pipeline = Pipeline(steps=[
    ("imputer", cat_imputer),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_imputer, numeric_features),
        ("cat", cat_pipeline, categorical_features)
    ]
)
