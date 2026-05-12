# app_fixed.py -- more robust handling for missing form fields & JSON payloads
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
from bson.errors import InvalidId

import os
import uuid
import io
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image

# optional: local LLM interface
try:
    from gpt4all import GPT4All
except Exception:
    GPT4All = None

from fpdf import FPDF

# ====== App Configuration ======
app = Flask(__name__)
# load secret from env if present, otherwise fallback to a development string
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev_secret_change_me")

# ====== MongoDB Configuration ======
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client.get_database("xray_disease_db")
users_collection = db["users"]
reports_collection = db["reports"]

# ====== Upload Folders ======
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
REPORTS_FOLDER = BASE_DIR / "static" / "reports"
MODELS_FOLDER = BASE_DIR / "models"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["REPORTS_FOLDER"] = str(REPORTS_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB limit

# ====== Load Models (robust) ======
model = None
gpt_model = None
disease_labels = ["Normal", "Pneumonia"]

h5_path = MODELS_FOLDER / "xray_final_model.h5"
if h5_path.exists():
    try:
        model = tf.keras.models.load_model(str(h5_path), compile=False)
        print("✅ X-ray Model Loaded Successfully:", h5_path)
    except Exception as e:
        model = None
        print("❌ Error loading X-ray Model:", e)
else:
    print("⚠️ X-ray model not found at", h5_path)

# ====== GPT4All Model Load ======
if GPT4All is not None:
    gguf_path = MODELS_FOLDER / "gpt4all-falcon.Q4_0.gguf"
    if gguf_path.exists():
        try:
            gpt_model = GPT4All(model_name="gpt4all-falcon.Q4_0.gguf", model_path=str(MODELS_FOLDER))
            print("✅ GPT4All Falcon Model Loaded Successfully!")
        except Exception as e:
            gpt_model = None
            print("❌ GPT4All Load Error:", e)
    else:
        print("⚠️ No GPT4All model found in:", MODELS_FOLDER)
else:
    print("⚠️ GPT4All package not installed; skipping chat model.")


# ====== Helpers ======

def _get_post_value(req, key):
    """Safely extract `key` from either form-encoded data or JSON body.
    Returns a stripped string or empty string if missing.
    """
    try:
        if req.is_json:
            data = req.get_json(silent=True) or {}
            val = data.get(key, "")
        else:
            val = req.form.get(key, "")
        if val is None:
            return ""
        return str(val).strip()
    except Exception:
        return ""


def preprocess_xray_image(file_path, target_size=(224, 224)):
    img = image.load_img(file_path, target_size=target_size)  # loads RGB (3 channels)
    arr = image.img_to_array(img)
    arr = arr.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_from_model(img_array):
    if model is None:
        raise RuntimeError("Model not loaded.")
    preds = model.predict(img_array)
    preds = np.array(preds)
    if preds.ndim == 2 and preds.shape[1] == 1:
        p = float(preds[0][0])
        if p > 0.5:
            diagnosis = disease_labels[1] if len(disease_labels) > 1 else "Positive"
            confidence = round(p * 100, 2)
        else:
            diagnosis = disease_labels[0]
            confidence = round((1 - p) * 100, 2)
    elif preds.ndim == 2:
        idx = int(np.argmax(preds[0]))
        diagnosis = disease_labels[idx] if idx < len(disease_labels) else f"Class_{idx}"
        confidence = round(float(preds[0][idx]) * 100, 2)
    else:
        p = float(np.ravel(preds)[0])
        diagnosis = disease_labels[1] if p > 0.5 else disease_labels[0]
        confidence = round(p * 100, 2)
    return diagnosis, confidence

# ====== Routes ======

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = _get_post_value(request, "email").lower()
        password = _get_post_value(request, "password")
        if not email or not password:
            return render_template("login.html", error="Please provide both email and password.")

        user = users_collection.find_one({"email": email})
        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["user_id"]
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid Email or Password")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        # safe extraction that supports both form-encoded and JSON payloads
        name = _get_post_value(request, "name")
        email = _get_post_value(request, "email").lower()
        phone = _get_post_value(request, "phone")
        password = _get_post_value(request, "password")

        # simple validations
        if not name:
            return render_template("signup.html", error="Please enter your name.")
        if not email:
            return render_template("signup.html", error="Please enter your email.")
        if not password:
            return render_template("signup.html", error="Please choose a password.")

        if users_collection.find_one({"email": email}):
            return render_template("signup.html", error="Email already registered!")

        user_id = str(uuid.uuid4())[:8]
        hashed_password = generate_password_hash(password)

        users_collection.insert_one({
            "user_id": user_id,
            "name": name,
            "email": email,
            "phone": phone,
            "password": hashed_password
        })

        return render_template("login.html", success="Signup successful! Please log in.")
    return render_template("signup.html")


@app.route("/predict", methods=["POST"])
def predict():
    # file must be in request.files
    if "xray" not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files["xray"]
    if file.filename == "":
        return jsonify({"error": "No file selected!"}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
    file_path = Path(app.config["UPLOAD_FOLDER"]) / unique_name
    file.save(str(file_path))

    try:
        img_array = preprocess_xray_image(str(file_path), target_size=(224, 224))
        if model is None:
            return render_template("result.html",
                                   diagnosis_text="Model not loaded",
                                   confidence_text=0,
                                   image_url=url_for('static', filename=f"uploads/{unique_name}"),
                                   user_name=_get_post_value(request, "name"),
                                   user_age=_get_post_value(request, "age"),
                                   user_blood=_get_post_value(request, "blood_group"),
                                   user_id=session.get("user_id"),
                                   report_id=None,
                                   precautions=["Unable to generate precautions since the model is not loaded."],
                                   request=request)

        # Run prediction
        diagnosis, confidence_percentage = predict_from_model(img_array)

        # Get patient info
        patient_name = _get_post_value(request, "name")
        patient_age = _get_post_value(request, "age")
        blood_group = _get_post_value(request, "blood_group")

        # Suggested precautions based on diagnosis
        if diagnosis.lower() == "normal":
            precautions = [
                "Your X-ray appears healthy — maintain your current lifestyle!",
                "Continue with balanced nutrition and regular exercise.",
                "No special precautions required at this time. Regular checkups are still beneficial."
            ]
        elif diagnosis.lower() == "pneumonia":
            precautions = [
                "Get plenty of rest and stay hydrated.",
                "Consult your doctor for antibiotic treatment if prescribed.",
                "Avoid cold air, smoke, and dusty environments.",
                "Monitor symptoms such as fever, cough, or difficulty breathing.",
                "Complete your full course of medication if started."
            ]
        else:
            precautions = [
                "Follow your doctor’s advice for further evaluation.",
                "Maintain good hygiene and a balanced diet to support recovery.",
                "Stay in touch with your healthcare provider for updates."
            ]

        # Save report in MongoDB
        report_id = None
        if "user_id" in session:
            report_data = {
                "user_id": session["user_id"],
                "patient_name": patient_name,
                "age": patient_age,
                "blood_group": blood_group,
                "filename": unique_name,
                "file_path": str(file_path),
                "diagnosis": diagnosis,
                "confidence": confidence_percentage,
                "timestamp": datetime.now(),
                "precautions": precautions
            }
            inserted = reports_collection.insert_one(report_data)
            report_id = str(inserted.inserted_id)

        # Render results page
        return render_template("result.html",
                               diagnosis_text=diagnosis,
                               confidence_text=confidence_percentage,
                               image_url=url_for('static', filename=f"uploads/{unique_name}"),
                               user_name=patient_name,
                               user_age=patient_age,
                               user_blood=blood_group,
                               user_id=session.get("user_id"),
                               report_id=report_id,
                               precautions=precautions,
                               request=request)

    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500


@app.route("/chat", methods=["POST"])
def chat_with_ai():
    if gpt_model is None:
        return jsonify({"error": "GPT model not loaded or not available."}), 500

    data = request.get_json(silent=True) or {}
    user_message = (data.get("message", "") or "").strip()
    if not user_message:
        return jsonify({"error": "Empty message!"}), 400

    try:
        try:
            response = gpt_model.generate(user_message, max_tokens=100, temp=0.7)
        except TypeError:
            response = gpt_model.generate(prompt=user_message, max_tokens=100, temp=0.7)
        return jsonify({"response": str(response)})
    except Exception as e:
        return jsonify({"error": f"GPT Error: {str(e)}"}), 500


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = users_collection.find_one({"user_id": session["user_id"]}, {"_id": 0})
    reports = list(reports_collection.find({"user_id": session["user_id"]}).sort("timestamp", -1).limit(3))

    for r in reports:
        r["_id"] = str(r["_id"]) if r.get("_id") else ""
        ts = r.get("timestamp")
        r["timestamp_str"] = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else str(ts)

    return render_template("dashboard.html", user=user, reports=reports)


@app.route("/download_report/<report_id>")
def download_report(report_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    try:
        report = reports_collection.find_one({"_id": ObjectId(report_id)})
        if not report:
            return jsonify({"error": "Report not found!"}), 404

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="X-Ray Disease Detection Report", ln=True, align="C")
        pdf.ln(6)
        pdf.cell(200, 8, txt=f"Name: {report.get('patient_name', 'N/A')}", ln=True)
        pdf.cell(200, 8, txt=f"Age: {report.get('age', 'N/A')}", ln=True)
        pdf.cell(200, 8, txt=f"Blood Group: {report.get('blood_group', 'N/A')}", ln=True)
        pdf.cell(200, 8, txt=f"Disease Detected: {report.get('diagnosis', 'N/A')}", ln=True)
        pdf.cell(200, 8, txt=f"Confidence: {report.get('confidence', 'N/A')}%", ln=True)

        timestamp = report.get("timestamp")
        formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(timestamp, datetime) else "N/A"
        pdf.cell(200, 8, txt=f"Date: {formatted_time}", ln=True)

        img_path = report.get("file_path")
        if img_path and Path(img_path).exists():
            try:
                pdf.ln(6)
                pdf.image(str(img_path), x=10, w=80)
            except Exception:
                pass

        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        return send_file(io.BytesIO(pdf_bytes), as_attachment=True, download_name="xray_disease_report.pdf", mimetype='application/pdf')

    except InvalidId:
        return jsonify({"error": "Invalid Report ID!"}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/all_reports")
def all_reports():
    if "user_id" not in session:
        return redirect(url_for("login"))

    all_reports = list(reports_collection.find({"user_id": session["user_id"]}).sort("timestamp", -1))
    for r in all_reports:
        r["_id"] = str(r["_id"]) if r.get("_id") else ""
        ts = r.get("timestamp")
        r["timestamp_str"] = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else str(ts)

    return render_template("all_reports.html", reports=all_reports)


@app.route("/share/<report_id>")
def share_report(report_id):
    return f"Sharing feature coming soon for report {report_id}"


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=int(os.getenv("PORT", 5000)))
