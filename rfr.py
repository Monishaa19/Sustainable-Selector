import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Load your dataset
data = pd.read_csv( r"ecowrapture.csv")

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Check for NaN values
print("Missing values in each column:")
print(data.isnull().sum())

# Handle NaN values (example: dropping rows with NaN)5
data = data.dropna(subset=['cost'])

# Define categorical and numerical features
categorical_features = ['category', 'size', 'materialpreference']
numerical_features = ['volumeinlitres']

# Create preprocessing pipelines
categorical_transformer = OneHotEncoder()
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ])

# Create the regression model for cost, sustainability score, and shelf life
regression_model = RandomForestRegressor()
pipeline_regressor = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', regression_model)
])

# Fit the regression model
pipeline_regressor.fit(data[categorical_features + numerical_features],
                       data[['cost', 'shelflifemonths', 'sustainabilityscore']])

# Create the classification model for material
classification_model = RandomForestClassifier()
pipeline_classifier = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', classification_model)
])

# Fit the classification model
pipeline_classifier.fit(data[categorical_features + numerical_features], data['material'])


import joblib
joblib.dump(pipeline_regressor,'pipeline_regressor.pkl')
joblib.dump(pipeline_classifier,'pipeline_classifier.pkl')
print("model saved")