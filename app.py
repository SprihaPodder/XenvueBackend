# import os
# import uuid
# import pandas as pd
# import numpy as np
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from werkzeug.security import generate_password_hash, check_password_hash
# from flask_sqlalchemy import SQLAlchemy
# from flask_migrate import Migrate

# from cloud import DB_URI

# # Your existing imports for processing
# from data_processing import clean_data, get_data_limitations, detect_biases, trend_data, demographics_data
# from narrative_generation import generate_narrative

# app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": "*"}})

# # -------------------- DB SETUP -------------------- #
# # app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
# app.config['SQLALCHEMY_DATABASE_URI'] = DB_URI


# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)
# migrate = Migrate(app, db)  

# # -------------------- MODELS -------------------- #
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(150), nullable=False)
#     email = db.Column(db.String(150), unique=True, nullable=False)
#     password_hash = db.Column(db.String(256), nullable=False)

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# reports_store = {}

# # -------------------- AUTH ROUTES -------------------- #
# @app.route("/api/signup", methods=["POST"])
# def signup():
#     try:
#         data = request.get_json(force=True)
#         name = data.get("name", "").strip()
#         email = data.get("email", "").strip().lower()
#         password = data.get("password", "")

#         if not name or not email or not password:
#             return jsonify({"error": "Name, email, and password are required"}), 400
#         if len(password) < 6:
#             return jsonify({"error": "Password must be at least 6 characters"}), 400
#         if User.query.filter_by(email=email).first():
#             return jsonify({"error": "Email already registered"}), 400

#         new_user = User(
#             name=name,
#             email=email,
#             password_hash=generate_password_hash(password)
#         )
#         db.session.add(new_user)
#         db.session.commit()
#         return jsonify({"message": "Signup successful"}), 201
#     except Exception as e:
#         return jsonify({"error": "Internal server error", "details": str(e)}), 500

# @app.route("/api/login", methods=["POST"])
# def login():
#     data = request.get_json()
#     email = data.get("email", "").strip().lower()
#     password = data.get("password", "")

#     user = User.query.filter_by(email=email).first()
#     if not user or not check_password_hash(user.password_hash, password):
#         return jsonify({"error": "Invalid email or password"}), 401

#     return jsonify({"message": "Login successful", "name": user.name}), 200

# # -------------------- GENERATE REPORT FROM UPLOAD -------------------- #
# @app.route("/api/generate-report", methods=["POST"])
# def generate_report():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     file = request.files["file"]
#     filename = file.filename
#     if not filename or not filename.lower().endswith((".csv", ".xlsx", ".xls")):
#         return jsonify({"error": "Invalid file type"}), 400

#     filepath = os.path.join(UPLOAD_FOLDER, filename)
#     file.save(filepath)

#     try:
#         if filename.lower().endswith(".csv"):
#             df_raw = pd.read_csv(filepath)
#         else:
#             df_raw = pd.read_excel(filepath)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

#     df = clean_data(df_raw)
#     stakeholder = request.form.get("stakeholder", "policymaker")

#     exec_summary = generate_narrative("Executive Summary", f"{detect_biases(df)}. Trend snapshot: {trend_data(df)}.", stakeholder_type=stakeholder)
#     limitations = get_data_limitations(df)
#     limitations_narr = generate_narrative("Data Limitations", str(limitations), stakeholder_type="researcher")
#     bias_risks = generate_narrative("Bias Risks", detect_biases(df), stakeholder_type="community")
#     community_concerns = generate_narrative("Community Concerns", "Data privacy concerns...", stakeholder_type="community")
#     stat_methods = generate_narrative("Statistical Methods", "Descriptive stats...", stakeholder_type="researcher")
#     trends_json = trend_data(df)
#     demographics_json = demographics_data(df)

#     report_id = str(uuid.uuid4())
#     reports_store[report_id] = {
#         "executive_summary": exec_summary,
#         "limitations": limitations,
#         "limitationsNarrative": limitations_narr,
#         "bias_risks": bias_risks,
#         "community_concerns": community_concerns,
#         "statistical_methods": stat_methods,
#         "trends": trends_json,
#         "demographics": demographics_json,
#         "generatedFor": stakeholder,
#     }
#     return jsonify({"reportId": report_id})

# @app.route("/api/report/<report_id>", methods=["GET"])
# def get_report(report_id):
#     report = reports_store.get(report_id)
#     if not report:
#         return jsonify({"error": "Report not found"}), 404
#     return jsonify(report)

# # -------------------- CATEGORY-SPECIFIC REPORT LOGIC -------------------- #
# def stakeholder_report_payload(stakeholder_type: str):
#     """
#     Generate a live, tailored report for the given stakeholder_type,
#     using actual data processing instead of static text.
#     """
#     files = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER) if f.endswith((".csv",".xlsx",".xls"))]
#     if not files:
#         return {"error": "No data file found on server. Please upload first via /api/generate-report."}

#     latest_file = max(files, key=os.path.getctime)
#     try:
#         if latest_file.lower().endswith(".csv"):
#             df_raw = pd.read_csv(latest_file)
#         else:
#             df_raw = pd.read_excel(latest_file)
#     except Exception as e:
#         return {"error": f"Could not read file: {str(e)}"}

#     df = clean_data(df_raw)

#     exec_summary = generate_narrative("Executive Summary", f"{detect_biases(df)}. Trend snapshot: {trend_data(df)}.", stakeholder_type=stakeholder_type)
#     limitations = get_data_limitations(df)
#     limitations_narr = generate_narrative("Data Limitations", str(limitations), stakeholder_type=stakeholder_type)
#     bias_risks = generate_narrative("Bias Risks", detect_biases(df), stakeholder_type=stakeholder_type)
#     community_concerns = generate_narrative("Community Concerns", "Data privacy, healthcare accessibility, equity.", stakeholder_type=stakeholder_type)
#     stat_methods = generate_narrative("Statistical Methods", "Descriptive stats, regression analysis.", stakeholder_type=stakeholder_type)
#     trends_json = trend_data(df)
#     demographics_json = demographics_data(df)

#     return {
#         "executive_summary": exec_summary,
#         "limitations": limitations,
#         "limitationsNarrative": limitations_narr,
#         "bias_risks": bias_risks,
#         "community_concerns": community_concerns,
#         "statistical_methods": stat_methods,
#         "trends": trends_json,
#         "demographics": demographics_json
#     }

# @app.route("/api/report/policy", methods=["GET"])
# def get_policy_report():
#     return jsonify(stakeholder_report_payload("Policy Manager"))

# @app.route("/api/report/community", methods=["GET"])
# def get_community_report():
#     return jsonify(stakeholder_report_payload("Community Member"))

# @app.route("/api/report/finance", methods=["GET"])
# def get_finance_report():
#     return jsonify(stakeholder_report_payload("Finance Management"))

# @app.route("/api/report/researcher", methods=["GET"])
# def get_researcher_report():
#     return jsonify(stakeholder_report_payload("Researcher"))

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)


import os
import uuid
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from cloud import DB_URI  # Ensure DB_URI is correct in cloud.py

# Your existing imports for processing
from data_processing import clean_data, get_data_limitations, detect_biases, trend_data, demographics_data
from narrative_generation import generate_narrative

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# -------------------- DB SETUP -------------------- #
# If DB_URI isn't set, raise a clear error instead of failing silently
if not DB_URI:
    raise RuntimeError("Database URI not set in cloud.py (DB_URI)")

app.config['SQLALCHEMY_DATABASE_URI'] = DB_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# -------------------- MODELS -------------------- #
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

# Ensure tables exist
with app.app_context():
    db.create_all()

# -------------------- UPLOADS -------------------- #
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
reports_store = {}

# -------------------- AUTH ROUTES -------------------- #
@app.route("/api/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json(force=True)
        name = data.get("name", "").strip()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")

        if not name or not email or not password:
            return jsonify({"error": "Name, email, and password are required"}), 400
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({"error": "Email already registered"}), 400

        new_user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"message": "Signup successful"}), 201
    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@app.route("/api/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({"error": "Invalid email or password"}), 401

        return jsonify({"message": "Login successful", "name": user.name}), 200
    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# -------------------- GENERATE REPORT FROM UPLOAD -------------------- #
@app.route("/api/generate-report", methods=["POST"])
def generate_report():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename
    if not filename or not filename.lower().endswith((".csv", ".xlsx", ".xls")):
        return jsonify({"error": "Invalid file type"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        if filename.lower().endswith(".csv"):
            df_raw = pd.read_csv(filepath)
        else:
            df_raw = pd.read_excel(filepath)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    df = clean_data(df_raw)
    stakeholder = request.form.get("stakeholder", "policymaker")

    exec_summary = generate_narrative("Executive Summary", f"{detect_biases(df)}. Trend snapshot: {trend_data(df)}.", stakeholder_type=stakeholder)
    limitations = get_data_limitations(df)
    limitations_narr = generate_narrative("Data Limitations", str(limitations), stakeholder_type="researcher")
    bias_risks = generate_narrative("Bias Risks", detect_biases(df), stakeholder_type="community")
    community_concerns = generate_narrative("Community Concerns", "Data privacy concerns...", stakeholder_type="community")
    stat_methods = generate_narrative("Statistical Methods", "Descriptive stats...", stakeholder_type="researcher")
    trends_json = trend_data(df)
    demographics_json = demographics_data(df)

    report_id = str(uuid.uuid4())
    reports_store[report_id] = {
        "executive_summary": exec_summary,
        "limitations": limitations,
        "limitationsNarrative": limitations_narr,
        "bias_risks": bias_risks,
        "community_concerns": community_concerns,
        "statistical_methods": stat_methods,
        "trends": trends_json,
        "demographics": demographics_json,
        "generatedFor": stakeholder,
    }
    return jsonify({"reportId": report_id})


@app.route("/api/report/<report_id>", methods=["GET"])
def get_report(report_id):
    report = reports_store.get(report_id)
    if not report:
        return jsonify({"error": "Report not found"}), 404
    return jsonify(report)


# -------------------- CATEGORY-SPECIFIC REPORT LOGIC -------------------- #
def stakeholder_report_payload(stakeholder_type: str):
    """
    Generate a live, tailored report for the given stakeholder_type
    using the latest uploaded dataset.
    """
    files = [os.path.join(UPLOAD_FOLDER, f)
             for f in os.listdir(UPLOAD_FOLDER)
             if f.endswith((".csv", ".xlsx", ".xls"))]
    if not files:
        return {"error": "No data file found on server. Please upload first via /api/generate-report."}

    latest_file = max(files, key=os.path.getctime)
    try:
        if latest_file.lower().endswith(".csv"):
            df_raw = pd.read_csv(latest_file)
        else:
            df_raw = pd.read_excel(latest_file)
    except Exception as e:
        return {"error": f"Could not read file: {str(e)}"}

    df = clean_data(df_raw)

    exec_summary = generate_narrative("Executive Summary", f"{detect_biases(df)}. Trend snapshot: {trend_data(df)}.", stakeholder_type=stakeholder_type)
    limitations = get_data_limitations(df)
    limitations_narr = generate_narrative("Data Limitations", str(limitations), stakeholder_type=stakeholder_type)
    bias_risks = generate_narrative("Bias Risks", detect_biases(df), stakeholder_type=stakeholder_type)
    community_concerns = generate_narrative("Community Concerns", "Data privacy, healthcare accessibility, equity.", stakeholder_type=stakeholder_type)
    stat_methods = generate_narrative("Statistical Methods", "Descriptive stats, regression analysis.", stakeholder_type=stakeholder_type)
    trends_json = trend_data(df)
    demographics_json = demographics_data(df)

    return {
        "executive_summary": exec_summary,
        "limitations": limitations,
        "limitationsNarrative": limitations_narr,
        "bias_risks": bias_risks,
        "community_concerns": community_concerns,
        "statistical_methods": stat_methods,
        "trends": trends_json,
        "demographics": demographics_json
    }

@app.route("/api/report/policy", methods=["GET"])
def get_policy_report():
    return jsonify(stakeholder_report_payload("Policy Manager"))

@app.route("/api/report/community", methods=["GET"])
def get_community_report():
    return jsonify(stakeholder_report_payload("Community Member"))

@app.route("/api/report/finance", methods=["GET"])
def get_finance_report():
    return jsonify(stakeholder_report_payload("Finance Management"))

@app.route("/api/report/researcher", methods=["GET"])
def get_researcher_report():
    return jsonify(stakeholder_report_payload("Researcher"))


# -------------------- MAIN -------------------- #
if __name__ == "__main__":
    # Bind to all interfaces, keep debug for development
    app.run(host="0.0.0.0", port=5000, debug=True)
