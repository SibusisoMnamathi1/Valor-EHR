from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import pyttsx3
import threading
import speech_recognition as sr  # Import speech recognition
import matplotlib.pyplot as plt
import io
import base64
import tensorflow as tf  # Import TensorFlow

app = Flask(__name__)

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Path to CSV file
csv_file = 'patients_data.csv'

# Ensure CSV file exists
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=['timestamp', 'name', 'age', 'medications', 'surgeries', 'health_events'])
    df.to_csv(csv_file, index=False)

# Function to train a KNN model
def train_knn_model():
    data = {
        'age': [25, 30, 35, 40, 45, 50, 55, 60],
        'medications': [1, 2, 2, 3, 3, 4, 4, 5],
        'risk': [0, 0, 1, 1, 1, 2, 2, 2]  # 0: Low, 1: Medium, 2: High
    }
    df = pd.DataFrame(data)
    X = df[['age', 'medications']]
    y = df['risk']
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model

# Train the model once when the application starts
knn_model = train_knn_model()

# Route to render HTML form
@app.route('/')
def patient_form():
    success = request.args.get('success', False)
    return render_template('patient_form.html', success=success)

# Function to handle text-to-speech in a separate thread
def speak(announcement):
    engine.say(announcement)
    engine.runAndWait()
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()  # Make sure the event loop completes before continuing

# Route to handle form submission and save data to CSV
@app.route('/save', methods=['POST'])
def save_data():
    name = request.form['name']
    age = int(request.form['age'])
    medications = int(request.form['medications'])
    surgeries = request.form['surgeries']
    health_events = request.form['health_events']

    timestamp = datetime.now()
    data = pd.DataFrame([[timestamp, name, age, medications, surgeries, health_events]], 
                        columns=['timestamp', 'name', 'age', 'medications', 'surgeries', 'health_events'])
    data.to_csv(csv_file, mode='a', header=False, index=False)

    announcement = f"Patient {name} has been captured successfully."
    threading.Thread(target=speak, args=(announcement,)).start()

    # Sleep for a few seconds to allow the speech to finish before redirecting
    threading.Timer(2.5, lambda: redirect(url_for('display_health_risk', name=name, age=age, health_risk=health_risk))).start()
    
    # Predict health risk using the KNN model
    prediction = knn_model.predict([[age, medications]])
    risk_level = {0: "Low", 1: "Medium", 2: "High"}
    health_risk = risk_level[prediction[0]]

    return redirect(url_for('display_health_risk', name=name, age=age, health_risk=health_risk))



# Route to display health risk
@app.route('/health_risk')
def display_health_risk():
    name = request.args.get('name')
    age = request.args.get('age')
    health_risk = request.args.get('health_risk')
    return render_template('health_risk.html', name=name, age=age, health_risk=health_risk)

# Route to display all patient data with search and pagination
@app.route('/view_patients', methods=['GET', 'POST'])
def view_patients():
    df = pd.read_csv(csv_file)
    name_search = request.form.get('name_search')
    age_search = request.form.get('age_search')
    
    if name_search:
        df = df[df['patient_name'].str.contains(name_search, case=False)]
    if age_search:
        df = df[df['age'] == int(age_search)]

    page = int(request.args.get('page', 1))
    per_page = 10
    total_records = df.shape[0]
    total_pages = (total_records + per_page - 1) // per_page

    start = (page - 1) * per_page
    end = start + per_page
    displayed_patients = df.iloc[start:end]

    return render_template('view_patients.html', patients=displayed_patients, 
                           page=page, total_pages=total_pages, 
                           name_search=name_search, age_search=age_search)

# New route for Speech Recognition
@app.route('/speech_recognition', methods=['GET', 'POST'])
def speech_recognition():
    if request.method == 'POST':
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Please say something:")
            audio = recognizer.listen(source)
            try:
                spoken_text = recognizer.recognize_google(audio)
                print(f"You said: {spoken_text}")
                return render_template('speech_recognition.html', spoken_text=spoken_text)
            except sr.UnknownValueError:
                return "Sorry, I could not understand the audio."
            except sr.RequestError:
                return "Could not request results from the speech recognition service."

    return render_template('speech_recognition.html')

# New route for Time Series Analysis
@app.route('/time_series_analysis')
def time_series_analysis():
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    time_series_data = df.groupby(df['timestamp'].dt.date).size()

    plt.figure(figsize=(10, 5))
    time_series_data.plot(kind='line')
    plt.title('Patient Entries Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Patients')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('time_series_analysis.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
