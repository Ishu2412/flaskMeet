from flask import Flask, render_template, request
import os
import pickle
import numpy as np

# Load the model and vectorizer outside of the route to avoid reloading on each request
model_path = 'C:\\Users\\Hp\\.vscode\\extensions\\python_project2\\model.pkl'
vectorizer_path = 'C:\\Users\\Hp\\.vscode\\extensions\\python_project2\\vectorizer.pkl'

print("Model Path:", model_path)
model = pickle.load(open(model_path, 'rb'))
print("Model Loaded Successfully:", model)

vectorizer = pickle.load(open(vectorizer_path, 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict_classification():
    
        # Get the input text from the form
    text = request.form.get('text')

        # For debugging, print the received form data
    print("Form Data:", request.form)

        # Make sure the model and vectorizer are loaded and input is valid
    if model and vectorizer and isinstance(text, str):
            # Use the vectorizer to transform the text into numeric features
       input_features = vectorizer.transform([text])
            
            # Use the model for prediction
       result = model.predict(input_features)
        
       if result[0] == 1 :
           result = "Spam" 
       else :
           result = "Not Spam"    
       return render_template("index.html",result = result)
        
    


if __name__ == "__main__":
 
    app.run(host="0.0.0.0", port=8080)

