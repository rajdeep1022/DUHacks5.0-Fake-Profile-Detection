# Backend WebAPP Using FLASK
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# ==============================
# LOAD MODEL (PARTNER WILL ADD)
# ==============================
# model = pickle.load(open("model/model.pkl", "rb"))

def extract_features(form):
    username = form["username"]
    fullname = form["fullname"]
    bio = form["bio"]

    # Dataset Feature Engineering
    profile_pic = int(form["profile_pic"])
    nums_length_username = sum(c.isdigit() for c in username) / len(username)
    fullname_words = len(fullname.split())
    nums_length_fullname = (
        sum(c.isdigit() for c in fullname) / len(fullname)
        if len(fullname) > 0 else 0
    )
    name_equals_username = int(username.lower() == fullname.lower())
    description_length = len(bio)
    external_url = int(form["external_url"])
    private = int(form["private"])
    posts = int(form["posts"])
    followers = int(form["followers"])
    follows = int(form["following"])

    return np.array([[
        profile_pic,
        nums_length_username,
        fullname_words,
        nums_length_fullname,
        name_equals_username,
        description_length,
        external_url,
        private,
        posts,
        followers,
        follows
    ]])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = extract_features(request.form)

    # ==============================
    # PREDICTION (PLACEHOLDER)
    # ==============================
    # prediction = model.predict(features)[0]

    # TEMPORARY LOGIC FOR TESTING UI
    prediction = 1 if features[0][9] < 100 else 0

    if prediction == 1:
        result = "Fake Profile Detected"
        css_class = "fake"
    else:
        result = "Genuine Profile"
        css_class = "genuine"

    return render_template(
        "index.html",
        prediction=result,
        prediction_class=css_class
    )

if __name__ == "__main__":
    app.run(debug=True)
