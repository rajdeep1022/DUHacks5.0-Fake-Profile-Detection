import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "trained_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# Load model & scaler once
model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))


def extract_features(form):
    username = form["username"]
    fullname = form["fullname"]
    bio = form.get("bio", "")

    profile_pic = int(form["profile_pic"])
    nums_username_ratio = sum(c.isdigit() for c in username) / len(username)

    fullname_words = len(fullname.split())
    nums_fullname_ratio = (
        sum(c.isdigit() for c in fullname) / len(fullname)
        if len(fullname) > 0 else 0
    )

    name_equals_username = int(username.lower() == fullname.lower())
    bio_length = len(bio)

    external_url = int(form["external_url"])
    is_private = int(form["private"])
    posts = int(form["posts"])
    followers = int(form["followers"])
    following = int(form["following"])

    features = np.array([[
        profile_pic,
        nums_username_ratio,
        fullname_words,
        nums_fullname_ratio,
        name_equals_username,
        bio_length,
        external_url,
        is_private,
        posts,
        followers,
        following
    ]])

    return features


def predict_profile(form):
    features = extract_features(form)

    # Scale features
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    # Optional probability
    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(features_scaled)[0][prediction]

    return prediction, probability
