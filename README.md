# 🔍 Fake Instagram Profile Detection using Machine Learning

## 📌 Project Overview

Fake and suspicious social media profiles are widely used for spamming, scams, and spreading misinformation. This project focuses on detecting fake Instagram profiles using Machine Learning by analyzing:

- Profile attributes
- User activity patterns
- Textual and numerical behavior

The system is designed to accurately classify profiles as **Genuine** or **Fake** while minimizing false positives, ensuring that legitimate users are not unfairly flagged.

## 🎯 Problem Statement

Design a machine learning model that analyzes social media profile attributes, user activity patterns, and textual behavior to classify profiles as genuine or fake.

## 🚀 Solution Approach

The project follows a complete end-to-end Machine Learning pipeline:

1. Exploratory Data Analysis (EDA)
2. Feature engineering
3. Model training and evaluation
4. Model serialization using `.pkl`
5. Deployment-ready inference module
6. Web application integration using Flask

The ML logic is fully decoupled from the web layer, making the system modular, scalable, and easy to maintain.

## 🧠 Machine Learning Details

- **Primary Model:** Random Forest Classifier
- **Baseline Model:** Logistic Regression
- **Feature Scaling:** StandardScaler
- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix

Random Forest was selected as the final model due to its higher accuracy and better generalization compared to the baseline model.

## 📊 Model Performance

The following table shows the comparison between baseline and final models:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~95% | ~95% | ~95% | ~95% |
| Random Forest (Final) | ~98% | ~99% | ~98% | ~98% |

### 📐Confusion Matrix



**📌 Observation:**  
Random Forest significantly improves detection performance while reducing false positives, making it more suitable for real-world deployment.

*(Exact values may vary slightly based on random state and dataset split.)*

## 🗂️ Project Folder Structure

```
fake-instagram-profile-detector/
│
├── data/
│   └── Instagram_fake_profile_dataset.csv
│
├── notebooks/
│   └── EDA_and_training.ipynb
│
├── model/
│   ├── trained_model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── __init__.py
│   └── predict.py
│
├── app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
├── requirements.txt
└── README.md
```

## 🧪 Dataset & Feature Description

The dataset contains Instagram profiles labeled as fake or genuine, with features capturing user behavior, profile details, and activity statistics.

### 📊 Feature Explanation

| Column name | Meaning | Why it matters |
|-------------|---------|----------------|
| profile pic | 1 = has profile picture, 0 = no | Fake accounts often skip profile pictures |
| nums/length username | Ratio of numeric characters in username | Fake accounts often use random numbers |
| fullname words | Number of words in full name | Genuine users usually have real names |
| nums/length fullname | Ratio of numeric characters in full name | Bots often use numbers in names |
| name==username | 1 if username equals full name, else 0 | Suspicious if both are identical |
| description length | Length of bio/description | Fake profiles often have empty or short bios |
| external URL | 1 if bio contains external link | Spam profiles usually add links |
| private | 1 = private account, 0 = public | Behavioral difference |
| #posts | Total number of posts | Fake accounts usually post less |
| #followers | Number of followers | Strong authenticity signal |
| #follows | Number of following | Fake accounts follow many users |
| fake | Target (1 = Fake, 0 = Genuine) | Label to be predicted |

### 🧠 Target Variable

- **fake = 1** → Fake Profile
- **fake = 0** → Genuine Profile

The objective is to detect fake profiles while minimizing incorrect classification of genuine users.

## 🧩 ML Inference Module (src/predict.py)

The inference module:

- Loads trained model and scaler
- Applies preprocessing at prediction time
- Exposes a reusable function

```python
from src.predict import predict_profile
prediction = predict_profile(feature_list)
```

**Output:**
- `1` → Fake Profile
- `0` → Genuine Profile

## 🌐 Web Application (Flask)

1. Users enter profile details via a web form
2. Features are extracted and passed to the ML inference module
3. Result is displayed as:
   - 🚨 **Fake Profile Detected**
   - ✅ **Genuine Profile**

The web layer does not contain ML logic.

## 📸 Screenshots

**When It is a Genuine Profile**



**When It is a Fake Profile**



## 🛠️ Installation & Usage

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Suchismita-1022/DUHacks5.0-Fake-Profile-Detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
python app/app.py
```

Open: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

## 🏁 Hackathon Details

- 🎯 **Hackathon:** DUHacks 5.0

This project was developed as part of DUHacks 5.0, focusing on real-world social media security challenges using Machine Learning.

## 🔮 Future Enhancements

- NLP-based bio analysis (TF-IDF / BERT)
- Prediction confidence score
- Cloud deployment
- Multi-platform support

## 🙏 Acknowledgements

This project was developed using several open-source libraries and frameworks. We sincerely acknowledge and thank the contributors and communities behind these tools, which made this project possible:

- `Python` – Core programming language used for data analysis, machine learning, and backend development

- `NumPy` – Efficient numerical computations and array operations

- `Pandas` – Data manipulation, cleaning, and preprocessing

- `Scikit-learn` – Machine learning models, preprocessing utilities, and evaluation metrics

- `Matplotlib` & `Seaborn` – Data visualization during exploratory data analysis

- `Flask` – Lightweight web framework used to build the backend application

- `HTML5`, `CSS3` – Used for building and styling the user interface

- `Jupyter Notebook` – Interactive environment for EDA and model training

We also acknowledge the open-source community for providing extensive documentation, examples, and support that helped in the successful completion of this project.

## 🏆 Conclusion

This project demonstrates a complete ML lifecycle, from data analysis to deployment-ready inference, with a modular architecture suitable for hackathons, academics, and real-world applications.