# 🏠 RoomieMatch AI

<div align="center">
  <p><strong>A machine-learning-powered platform to help you find the perfect roommate based on deep compatibility metrics.</strong></p>
</div>

---

## 📖 Overview

Finding a compatible roommate is more than just matching budgets and locations. **RoomieMatch AI** takes the guesswork out of the process by evaluating a comprehensive set of lifestyle and preference parameters—ranging from cleanliness and noise tolerance to smoking/drinking habits and bill punctuality. 

By employing a **Hybrid Prediction Engine** that combines a robust **Random Forest Regressor** with **Cosine Similarity**, the application is able to score profile compatibility intelligently, highlight potential "dealbreakers," and present results in a beautiful, easy-to-read dashboard.

---

## ✨ Key Features

- **Comprehensive Profile System:** Users fill out detailed surveys capturing their living habits, social preferences, and dealbreakers.
- **Hybrid Matching Engine:** Uses an advanced AI model analyzing over 20 unique features and deltas to pair potential roommates.
- **Dealbreaker Logic:** Enforces severe penalties for absolute mismatches (e.g., conflicting dietary preferences, differing approaches to smoking).
- **Interactive Comparison UI:** Visually compares two profiles side-by-side, distinctly listing key **Green Flags** (perfect matches) and **Red Flags** (areas of conflict).
- **Dynamic Scoring Dashboard:** Displays compatibility scores via CSS-animated circular gauges, breaking matches down into categories like "High Compatibility" or "Incompatible."

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, SQLAlchemy (SQLite)
- **Machine Learning Engine:** Scikit-Learn (Random Forest Regressor, Cosine Similarity), Pandas, NumPy, Joblib
- **Frontend / UI:** HTML5, Vanilla CSS for premium tailored styling, FontAwesome Icons, Vanilla JavaScript

---

## 🚀 Setup & Installation

Follow these steps to get the application running on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/darkacer211/RoomieMatch-AI.git
cd RoomieMatch-AI
```

### 2. Set Up Your Environment
It is highly recommended to use a virtual environment.
```bash
# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (macOS/Linux)
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Initial Model
Before you can match users, you need to generate the synthetic training data and build the Random Forest Model:
```bash
python data_engine.py
```
*This script automatically generates 10,000 synthetic profile pairs, trains the model, and exports `model.pkl` and `model_features.pkl` to your project root.*

### 5. Seed the Database (Optional)
If you want to populate your database with test users (like Ketan, Yash, Krushna, Sam, and Atharwa):
```bash
python seed_users.py
```

### 6. Run the Application
Start the Flask development server:
```bash
python app.py
```
Open a browser and navigate to `http://127.0.0.1:5000/`.

---

## 🧠 How the Matching Algorithm Works

Our `predict_compatibility` function evaluates pairs using a three-step process:

1. **Feature Vectorization:** Both user profiles are converted to numeric vectors capturing 18 independent features.
2. **Delta & Mismatch Calculations:** The algorithm calculates absolute differences (`delta_*`) for scalable traits (like budget or cleanliness) and binary conflict flags (`mismatch_*`) for strict preferences (like smoking or pets).
3. **Random Forest Regression:** These complex feature deltas are fed into the trained Joblib model, which returns an initial score based on thousands of historically weighted scenarios.
4. **Weighted Interpretability:** The system parses the model's `feature_importances_` to explain exactly **why** the score is high or low, dynamically generating green and red flags for the end user.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
