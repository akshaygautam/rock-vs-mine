import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import jsonify, Flask, send_from_directory

app = Flask(__name__, static_folder="../frontend/build/static", template_folder="../frontend/build")

csvPath = 'd:/code/event-tracker-full-stack-flask/backend/app/resources/RockVsMines.csv'

# Load and prepare data
def load_and_train_model():
    print("Loading and training datamodel")
    sonar_data = pd.read_csv(csvPath, header=None)
    X = sonar_data.drop(columns=60, axis=1)
    Y = sonar_data[60]
    
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
    
    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    
    # Calculate accuracy (for reference)
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print('Accuracy on training data:', training_data_accuracy)
    
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    print('Accuracy on test data:', test_data_accuracy)
    
    return model



model = load_and_train_model()

@app.route('/')
def index():
    return send_from_directory('../frontend/build', 'index.html')

@app.route('/frequency-map', methods=['GET'])
def get_frequency_map():
    frequency_map = generate_frequency_map(model)
    return jsonify(frequency_map)


# Predict function using the trained model
def predict_object(model, input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    
    if prediction[0] == 'R':
        return 'The object is a Rock'
    else:
        return 'The object is a Mine'

def generate_frequency_map(model):
    # Load the CSV data to get feature statistics
    sonar_data = pd.read_csv(csvPath, header=None)
    feature_data = sonar_data.iloc[:, :-1]

    # Get min and max values for each feature
    min_values = feature_data.min()
    max_values = feature_data.max()
    
    frequency_map = {}

    for i in range(0, 101):  # Example range of frequencies
        # Generate synthetic data within the range of the real data
        input_data = np.random.uniform(min_values, max_values, size=feature_data.shape[1])
        
        # Call the predict_object function with the input_data
        prediction = predict_object(model, input_data)
        
        # Map the frequency (or whatever key you are using) to the prediction
        frequency_map[i] = prediction
    
    return frequency_map

app.run(debug=True, port=8080)