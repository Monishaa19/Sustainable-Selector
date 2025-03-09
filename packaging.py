from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = Flask(_name_)
CORS(app)

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

# Configure your API key for Google Generative AI
genai.configure(api_key="AIzaSyBCOUSy8DZ5Qzxhqq0gOkRCUJEHNVLjoZM")


def generate_description(material, cost, shelflife, sustainability_score, input_data):
    # Define the prompt for the Gemini model
    prompt = (
        f"Provide a clear and concise explanation of why the material '{material}' should be used based on the following properties:\n"
        f"- Cost: {cost}\n"
        f"- Shelf Life: {shelflife} months\n"
        f"- Sustainability Score: {sustainability_score}\n"
        f"- Input features: {input_data}\n"
        f"Additionally, include the benefits of using this material.\n"
        f"Please also suggest ways to properly dispose of or reuse this packaging material.\n"
        f"Please format the response in a pointwise manner, ensuring each point is on a new line without any special characters or markdown symbols."
    )

    # Generate content using the Gemini model
    model = genai.GenerativeModel("gemini-1.5-flash")  # Use the desired model
    response = model.generate_content(prompt)

    # Clean up the response and format it pointwise
    cleaned_response = response.text.replace("#", "").replace("*", "").strip()

    # Split the response into lines and rejoin with line breaks for clear separation
    pointwise_output = cleaned_response.replace(".", ".\n")  # Add a newline after each sentence
    return pointwise_output


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        # Validate input data
        required_keys = ['category', 'size', 'materialpreference', 'volumeinlitres']
        if not all(key in input_data for key in required_keys):
            return jsonify({"error": "Missing required input data"}), 400

        new_data = pd.DataFrame([input_data])

        # Perform predictions
        regression_predictions = pipeline_regressor.predict(new_data[categorical_features + numerical_features])
        classification_predictions = pipeline_classifier.predict(new_data[categorical_features + numerical_features])

        response_data = {
            "cost": float(regression_predictions[0][0]),  # Convert to float for JSON serialization
            "shelflife": int(regression_predictions[0][1]),  # Convert to int if shelflife is in months
            "sustainability_score": float(regression_predictions[0][2]),  # Ensure it's a float
            "material": classification_predictions[0]  # This should be a string
        }

        # Generate description using the Gemini API
        description = generate_description(
            material=response_data["material"],
            cost=response_data["cost"],
            shelflife=response_data["shelflife"],
            sustainability_score=response_data["sustainability_score"],
            input_data=input_data
        )

        response_data["description"] = description  # Add description to the response

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if _name_ == '_main_':
    app.run(debug=True)