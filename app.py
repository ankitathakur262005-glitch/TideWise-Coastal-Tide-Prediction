from flask import Flask, render_template, request
import os
import math
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ----------------------------
# Project constants (UI choices)
# ----------------------------
TIME_BLOCKS = ["Morning", "Afternoon", "Evening", "Night"]
MOON_PHASES = ["New Moon", "First Quarter", "Full Moon", "Last Quarter"]
TIDE_CLASSES = ["Low", "Normal", "High"]

MODEL_FILE = "tidewise_model.pkl"
SCALER_FILE = "tidewise_scaler.pkl"
FEATURE_FILE = "tidewise_features.pkl"

# ----------------------------
# 1) Dataset (for demo)
# If you don't have real data, we create a starter dataset automatically.
# ----------------------------
def tide_rule(month, time_block, moon_phase, wind_speed, pressure, temperature):
    """
    Rule-based labeling to generate a starter dataset.
    Replace later with real data if you want. Your ML pipeline remains same.
    """
    score = 0

    # Moon: New/Full -> stronger tides
    if moon_phase in ["Full Moon", "New Moon"]:
        score += 2
    else:
        score += 1

    # Wind: stronger wind -> higher tide/waves effect
    if wind_speed >= 22:
        score += 2
    elif wind_speed >= 12:
        score += 1

    # Pressure: lower pressure -> higher sea level/waves
    if pressure <= 1005:
        score += 2
    elif pressure <= 1012:
        score += 1

    # Time block (demo assumption)
    if time_block in ["Evening", "Night"]:
        score += 1

    # Seasonal rough sea (monsoon months)
    if month in [6, 7, 8, 9]:
        score += 1

    # Temperature slight effect
    if temperature >= 32:
        score += 1

    if score >= 7:
        return "High"
    elif score >= 4:
        return "Normal"
    else:
        return "Low"


def generate_dataset(n_rows=1200, seed=42):
    np.random.seed(seed)
    rows = []
    for _ in range(n_rows):
        month = np.random.randint(1, 13)
        time_block = np.random.choice(TIME_BLOCKS)
        moon_phase = np.random.choice(MOON_PHASES)

        wind_speed = np.random.uniform(2, 35)        # km/h
        pressure = np.random.uniform(995, 1025)      # hPa
        temperature = np.random.uniform(18, 38)      # °C

        tide_level = tide_rule(month, time_block, moon_phase, wind_speed, pressure, temperature)
        rows.append([month, time_block, moon_phase, wind_speed, pressure, temperature, tide_level])

    df = pd.DataFrame(rows, columns=[
        "month", "time_block", "moon_phase",
        "wind_speed", "pressure", "temperature",
        "tide_level"
    ])
    return df


# ----------------------------
# 2) Feature engineering (Practical style)
# ----------------------------
def encode_time_block(tb: str) -> int:
    mapping = {name: idx for idx, name in enumerate(TIME_BLOCKS)}
    return mapping.get(tb, 0)

def encode_moon_phase(mp: str) -> int:
    mapping = {name: idx for idx, name in enumerate(MOON_PHASES)}
    return mapping.get(mp, 0)

def month_cyc(month: int):
    month = max(1, min(12, int(month)))
    return math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12)

def preprocess(df: pd.DataFrame):
    """
    Practical flow:
    - Handle missing values
    - Encode categorical
    - Create features
    """
    df = df.dropna().copy()

    df["time_block_enc"] = df["time_block"].apply(encode_time_block)
    df["moon_phase_enc"] = df["moon_phase"].apply(encode_moon_phase)

    m = df["month"].astype(int).clip(1, 12)
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)

    feature_cols = [
        "wind_speed", "pressure", "temperature",
        "time_block_enc", "moon_phase_enc",
        "month_sin", "month_cos"
    ]

    X = df[feature_cols].astype(float)
    y = df["tide_level"].astype(str)
    return X, y, feature_cols


# ----------------------------
# 3) Train models (Practical style)
# LR, KNN, SVM, RF + evaluation
# ----------------------------
def train_and_save():
    df = generate_dataset()  # You can replace with real CSV later if you want

    X, y, feature_cols = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling (important for KNN & SVM)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM (RBF)": SVC(kernel="rbf", probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    best_name, best_model, best_acc = None, None, -1

    # Evaluate each model
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        acc = accuracy_score(y_test, pred)
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    # Confusion matrix (for viva)
    final_pred = best_model.predict(X_test_s)
    cm = confusion_matrix(y_test, final_pred)

    # Save in same folder (minimal structure)
    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(feature_cols, FEATURE_FILE)

    print(f"✅ Best Model: {best_name} | Accuracy: {best_acc:.4f}")
    print("✅ Confusion Matrix:\n", cm)
    return best_model, scaler, feature_cols


def load_or_train():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(FEATURE_FILE):
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        feature_cols = joblib.load(FEATURE_FILE)
        return model, scaler, feature_cols
    return train_and_save()


# Load model once on startup
MODEL, SCALER, FEATURE_COLS = load_or_train()


# ----------------------------
# 4) Advisory module (extra marks)
# ----------------------------
def get_advisory(tide: str):
    if tide == "Low":
        return {
            "title": "Low Tide Advisory",
            "tip": "Generally safer near shore, but shallow waters may affect boats and routes.",
            "do": ["Good time for beach walking", "Check shallow routes if boating"],
            "avoid": ["Avoid rushing into unknown shallow areas"]
        }
    if tide == "Normal":
        return {
            "title": "Normal Tide Advisory",
            "tip": "Typical coastal conditions. Still follow local safety instructions.",
            "do": ["Normal fishing & tourism activities", "Stay updated with local alerts"],
            "avoid": ["Avoid risky rocks/cliffs if waves increase"]
        }
    return {
        "title": "High Tide Advisory",
        "tip": "Stronger waves expected. Higher risk for swimming and small boats near shore.",
        "do": ["Follow coastal safety alerts", "Keep emergency contacts ready"],
        "avoid": ["Avoid swimming", "Avoid small boats without safety checks", "Avoid low-lying shore areas"]
    }


# ----------------------------
# 5) Flask routes
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    advisory = None

    # Default form values
    form_values = {
        "month": "1",
        "time_block": "Morning",
        "moon_phase": "New Moon",
        "wind_speed": "",
        "pressure": "",
        "temperature": ""
    }

    if request.method == "POST":
        # Read user inputs
        month = int(request.form.get("month", 1))
        time_block = request.form.get("time_block", "Morning")
        moon_phase = request.form.get("moon_phase", "New Moon")

        wind_speed = float(request.form.get("wind_speed", 0))
        pressure = float(request.form.get("pressure", 1013))
        temperature = float(request.form.get("temperature", 25))

        # Save back to form (so values remain visible after submit)
        form_values = {
            "month": str(month),
            "time_block": time_block,
            "moon_phase": moon_phase,
            "wind_speed": str(wind_speed),
            "pressure": str(pressure),
            "temperature": str(temperature)
        }

        # Feature engineering (same as training)
        month_sin, month_cos = month_cyc(month)
        time_block_enc = encode_time_block(time_block)
        moon_phase_enc = encode_moon_phase(moon_phase)

        feature_map = {
            "wind_speed": wind_speed,
            "pressure": pressure,
            "temperature": temperature,
            "time_block_enc": time_block_enc,
            "moon_phase_enc": moon_phase_enc,
            "month_sin": month_sin,
            "month_cos": month_cos
        }

        X = np.array([[feature_map[c] for c in FEATURE_COLS]], dtype=float)
        Xs = SCALER.transform(X)

        pred = MODEL.predict(Xs)[0]
        prediction = pred

        # Confidence (if model supports)
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(Xs)[0]
            classes = list(MODEL.classes_)
            confidence = round(float(proba[classes.index(pred)]) * 100, 1)

        advisory = get_advisory(prediction)

    return render_template(
        "index.html",
        time_blocks=TIME_BLOCKS,
        moon_phases=MOON_PHASES,
        prediction=prediction,
        confidence=confidence,
        advisory=advisory,
        form_values=form_values
    )


if __name__ == "__main__":
    app.run(debug=True)
