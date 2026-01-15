import pickle
import numpy as np
import os

# Base directory of project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "trained_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# Load once (important for performance)
model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))

FEATURE_ORDER = [
    "profile_pic",
    "nums_username_ratio",
    "fullname_words",
    "nums_fullname_ratio",
    "name_equals_username",
    "bio_length",
    "external_url",
    "is_private",
    "posts",
    "followers",
    "following"
]

def predict_profile(input_data):
    """
    input_data: list of 11 numeric features in correct order
    return: int (1 = Fake, 0 = Genuine)
    """

    if len(input_data) != 11:
        raise ValueError("Expected 11 input features")

    input_array = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)[0]

    return int(prediction)
