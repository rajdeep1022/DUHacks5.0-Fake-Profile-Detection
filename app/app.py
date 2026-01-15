import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)



from flask import Flask, render_template, request
from src.predict import predict_profile

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    prediction, probability = predict_profile(request.form)

    if prediction == 1:
        result = "Fake Profile Detected"
        css_class = "fake"
    else:
        result = "Genuine Profile"
        css_class = "genuine"

    confidence = None
    if probability is not None:
        confidence = round(probability * 100, 2)

    return render_template(
        "index.html",
        prediction=result,
        prediction_class=css_class,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)
