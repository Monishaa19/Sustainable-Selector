from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv  # Import dotenv
import os
load_dotenv()  # Load environment variables
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

app = Flask(__name__)  

model = joblib.load("usage.pkl")  # Load your trained ML model
model2 = joblib.load("rating.pkl")

pipeline_regressor = joblib.load("pipeline_regressor.pkl")  # Load your trained ML model
pipeline_classifier = joblib.load("pipeline_classifier.pkl")

@app.route('/')
def home():
    return render_template('home.html')  # Serve the HTML file

@app.route('/EV')
def ev_selection():
    return render_template('index.html')  # Renders index.html

@app.route('/sustainable')
def sustainable_selection():
    return render_template('index2.html')  # Renders index2.html


@app.route('/predict1', methods=['POST'])
def predict1():
    try:
        data = request.json
        inputs = [float(data[key]) for key in sorted(data)]  # Extract input values
        if len(inputs) != 6:
            return jsonify({'error': 'Expected 6 inputs, got {}'.format(len(inputs))}), 400
        usage = model.predict([inputs])[0]  # Predict using the model
        rating = model2.predict([inputs])[0]  
        
        if(usage=='Yes'):
            prediction='Yes , the material can be used for EV chassis.  '
            prediction+=f"\n The rating of the material is {rating}/5"
        else :
            prediction='No , the material cannot be used for EV chassis  '
            prediction+=f"\n The rating of the material is {rating}/5"
        
        
        print("the prediction is ",prediction)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/predict2', methods=['POST'])
def predict2():


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

    categorical_features = ['category', 'size', 'materialpreference']
    numerical_features = ['volumeinlitres']
    
    
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
    
    
@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run()

