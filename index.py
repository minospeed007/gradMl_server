import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

from flask import Flask, jsonify, request
import joblib
from flask_cors import CORS

# Flask setup
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Print statements for server startup
print('Server running...')

# Path to the dataset
csv_file_path = os.path.join(os.path.dirname(__file__), 'data', 'spam.csv')

# Load and prepare the dataset
def load_dataset():
    try:
        df = pd.read_csv(csv_file_path)
        df.drop_duplicates(keep='first', inplace=True)
        df['Category'] = df['Category'].map({'ham': 1, 'spam': 0})
        return df
    except FileNotFoundError:
        print("File not found. Check the path to the CSV file.")
        return None

# Initial loading of dataset and training models
df = load_dataset()
if df is not None:
    X = df['Message']
    y = df['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Classifier models
    models = {
        'Adaboost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42),
        'LogisticRegression': LogisticRegression(solver='liblinear', penalty='l1'),
        'SVC': SVC(kernel="sigmoid", gamma=1.0),
        'XGB': XGBClassifier(n_estimators=50, random_state=42)
    }

    # Pipelines for each model
    pipelines = {}
    for name, model in models.items():
        pipelines[name] = Pipeline([
            ('vectorizer', TfidfVectorizer()),  # Text vectorization
            ('classifier', model)  # Classifier model
        ])
        # Fit the pipeline on the training data
        pipelines[name].fit(X_train, y_train)
        y_pred = pipelines[name].predict(X_test)

# Evaluate each model's performance
for name, pipeline in pipelines.items():
    try:
        print(f"Testing model: {name}")
        
        # Predict on the test set
        y_pred = pipeline.predict(X_test)
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=0)  
        recall = recall_score(y_test, y_pred, pos_label=1)  
        
        #Print model performance
        # print(f"Model: {name}")
        # print(f"Recall (ham): {recall:.2f}")
        # print(f"Classification Report for {name}:\n", classification_report(y_test, y_pred))
        # print("-" * 50)
        
    except Exception as e:
        print(f"Error testing model {name}: {e}")

# Route for predicting if an email is spam or ham
@app.route('/api/predict', methods=['POST'])
def predict_email():
    print('predict route')

    # Get JSON data from the request
    data = request.json
    email_text = data.get('email')
    
    # Validate email input
    if not email_text:
        return jsonify({"error": "No email text provided."}), 400
    if not isinstance(email_text, str):
        return jsonify({"error": "Invalid email format."}), 400

    # Preprocessing email text (strip, lower)
    email_text = email_text.strip().lower()  # Ensure consistent preprocessing with the model training
    print(f"Processed email text: {email_text}")

    predictions = {}
    for name, pipeline in pipelines.items():
        # Predict the class
        prediction = pipeline.predict([email_text])[0]
        
        # Convert numerical prediction to spam/ham
        predictions[name] = "spam" if prediction == 0 else "ham"
        print(f"Prediction by {name}: {predictions[name]}")

    # Return predictions as JSON
    return jsonify(predictions)

# Route for collecting user feedback and retraining models
@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.json
    email_text = data.get('email')
    is_spam = data.get('is_spam')  # Boolean indicating whether the prediction was spam or ham
    feedback = data.get('feedback')  # 1 for correct prediction, 0 for incorrect prediction

    if feedback is None or email_text is None:
        return jsonify({"error": "Invalid feedback data."}), 400

    # Log the feedback
    print(f'Feedback received: email="{email_text}", prediction_correct={feedback}')

    # Append the feedback to the dataset
    feedback_data = {
        'Message': email_text,
        'Category': 0 if is_spam else 1  # Convert to 0 for spam, 1 for ham
    }

    # Append to CSV file (with deduplication)
    #feedback_df = pd.DataFrame([feedback_data])
    #feedback_df.to_csv(csv_file_path, mode='a', header=False, index=False)

    # Reload the dataset and retrain the models
    try:
        df = load_dataset()  # Reload updated dataset
        if df is not None:
            X = df['Message']
            y = df['Category']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Retrain all models
            for name, pipeline in pipelines.items():
                pipeline.fit(X_train, y_train)

            # Optionally, save the updated models to disk
            joblib.dump(pipelines, 'updated_models.pkl')  # Save updated models for future use

            return jsonify({"message": "Feedback received and model retrained successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main entry point
if __name__ == '__main__':
   port = int(os.environ.get('PORT', 10000))
   app.run(host='0.0.0.0', port=port)
