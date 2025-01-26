import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

# Flask app
app = Flask(__name__)

# Load the model
model = pickle.load(open("linearRegression.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the form
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        return render_template(
            "index.html",
            prediction_text=f"Predicted House Price: ${prediction[0]:,.2f}"
        )
    except Exception as e:
        return render_template("index.html", error_text=str(e))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
