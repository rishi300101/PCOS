from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
import os
import json
import sqlite3
from datetime import datetime, timezone

# ✅ ADD THESE TWO LINES
from dotenv import load_dotenv
load_dotenv()
print("Gemini key loaded:", os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

# ✅ Now this will read from .env
app.secret_key = os.getenv("FLASK_SECRET_KEY")

mongo_uri = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://tmoulya:nsrm6457@cluster0.ejxiy.mongodb.net/?appName=Cluster0",
)

mongo_client = MongoClient(mongo_uri)
mongo_db = mongo_client.get_database(os.getenv("MONGODB_DB", "pcos_app"))
users_col = mongo_db["users"]
predictions_col = mongo_db["predictions"]
tracker_col = mongo_db["tracker"]
model = joblib.load("pcos_model.pkl")
scaler = joblib.load("scaler.pkl")


def init_pcod_db():
    """Initialize SQLite database for caching lifestyle plans."""
    conn = sqlite3.connect("pcod.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lifestyle_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            raw_input TEXT NOT NULL,
            diet_plan TEXT,
            exercise_plan TEXT,
            lifestyle_tips TEXT,
            stress_sleep TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# Initialize the database on startup
init_pcod_db()



@app.route("/", methods=["GET"])
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/dashboard", methods=["GET"])
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")


@app.route("/risk", methods=["GET", "POST"])
def risk():
    if "user_id" not in session:
        return redirect(url_for("login"))
    result = None
    message = None
    risk = None

    if request.method == "POST":
        required_fields = [
            "age", "height", "weight", "cycle", "cycle_length",
            "amh", "tsh", "fsh", "lh", "weight_gain", "hair_growth",
            "skin_darkening", "hair_loss", "pimples", "fast_food", "reg_exercise",
        ]
        for field in required_fields:
            if field not in request.form or not str(request.form.get(field, "")).strip():
                form_values = {f: request.form.get(f, "") for f in required_fields}
                return render_template("risk.html", result=None, message=None, risk=None, form_values=form_values)

        try:
            age = float(request.form["age"])
            height = float(request.form["height"])
            weight = float(request.form["weight"])
            cycle = int(request.form["cycle"])
            amh = float(request.form.get("amh") or 0)
            tsh = float(request.form.get("tsh") or 0)
            fsh = float(request.form.get("fsh") or 0)
            lh = float(request.form.get("lh") or 0)
            cycle_length = int(request.form.get("cycle_length") or 5)

            def _to_int_flag(name: str, default: int = 0) -> int:
                raw = request.form.get(name, "")
                if raw in ("0", "1"):
                    return int(raw)
                return default

            weight_gain = _to_int_flag("weight_gain", 0)
            hair_growth = _to_int_flag("hair_growth", 0)
            skin_darkening = _to_int_flag("skin_darkening", 0)
            hair_loss = _to_int_flag("hair_loss", 0)
            pimples = _to_int_flag("pimples", 0)
            fast_food = _to_int_flag("fast_food", 0)
            reg_exercise = _to_int_flag("reg_exercise", 1)

            if age <= 0 or height <= 0 or weight <= 0 or cycle_length < 2 or cycle_length > 12:
                form_values = {f: request.form.get(f, "") for f in required_fields}
                return render_template("risk.html", result=None, message=None, risk=None, form_values=form_values)

            if lh == 0:
                form_values = {f: request.form.get(f, "") for f in required_fields}
                return render_template("risk.html", result=None, message=None, risk=None, form_values=form_values)

        except (ValueError, KeyError):
            form_values = {f: request.form.get(f, "") for f in [
                "age", "height", "weight", "cycle", "cycle_length",
                "amh", "tsh", "fsh", "lh", "weight_gain", "hair_growth",
                "skin_darkening", "hair_loss", "pimples", "fast_food", "reg_exercise",
            ]}
            return render_template("risk.html", result=None, message=None, risk=None, form_values=form_values)

        bmi = weight / ((height / 100) ** 2)
        ratio = fsh / lh

        user_input = {
            'Age (yrs)': age,
            'BMI': bmi,
            'Cycle(R/I)': cycle,
            'Cycle length(days)': cycle_length,
            'AMH(ng/mL)': amh,
            'TSH (mIU/L)': tsh,
            'FSH_LH_Ratio': ratio,
            'Weight gain(Y/N)': weight_gain,
            'hair growth(Y/N)': hair_growth,
            'Skin darkening (Y/N)': skin_darkening,
            'Hair loss(Y/N)': hair_loss,
            'Pimples(Y/N)': pimples,
            'Fast food (Y/N)': fast_food,
            'Reg.Exercise(Y/N)': reg_exercise,
        }

        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[0][1]

        risk = decision_after_prediction(user_input, prob)

        session["last_risk_context"] = {
            "features": user_input,
            "probability": float(prob),
            "risk_level": risk,
            "raw": {
                "age": age,
                "height": height,
                "weight": weight,
                "bmi": round(bmi, 2),
                "amh": amh,
                "tsh": tsh,
                "fsh": fsh,
                "lh": lh,
                "cycle": cycle,
                "cycle_length": cycle_length,
                "weight_gain": weight_gain,
                "hair_growth": hair_growth,
                "skin_darkening": skin_darkening,
                "hair_loss": hair_loss,
                "pimples": pimples,
                "fast_food": fast_food,
                "reg_exercise": reg_exercise,
                "fsh_lh_ratio": round(ratio, 3),
            },
        }

        session["last_form_data"] = {
            "age": request.form.get("age", ""),
            "height": request.form.get("height", ""),
            "weight": request.form.get("weight", ""),
            "amh": request.form.get("amh", ""),
            "tsh": request.form.get("tsh", ""),
            "fsh": request.form.get("fsh", ""),
            "lh": request.form.get("lh", ""),
            "cycle": request.form.get("cycle", ""),
            "cycle_length": request.form.get("cycle_length", ""),
            "weight_gain": request.form.get("weight_gain", ""),
            "hair_growth": request.form.get("hair_growth", ""),
            "skin_darkening": request.form.get("skin_darkening", ""),
            "hair_loss": request.form.get("hair_loss", ""),
            "pimples": request.form.get("pimples", ""),
            "fast_food": request.form.get("fast_food", ""),
            "reg_exercise": request.form.get("reg_exercise", ""),
        }

        if risk == "LOW":
            message = "Low risk detected. You may optionally contribute data."
            data_status = "candidate_low"
        elif risk == "MEDIUM":
            message = "Medium risk detected. Lifestyle changes recommended."
            data_status = "candidate_moderate"
        else:
            message = "High risk detected. Please consult a doctor."
            data_status = "candidate_high"

        save_candidate_data(user_input, prob, risk, data_status)

        from datetime import datetime
        predictions_col.insert_one({
            "user_id": session["user_id"],
            "probability": float(prob),
            "risk_level": risk,
            "timestamp": datetime.utcnow(),
        })

        result = round(prob, 3)

    required_fields = [
        "age", "height", "weight", "amh", "tsh", "fsh", "lh", "cycle",
        "cycle_length", "weight_gain", "hair_growth", "skin_darkening",
        "hair_loss", "pimples", "fast_food", "reg_exercise",
    ]
    if request.method == "POST":
        form_values = {f: request.form.get(f, "") for f in required_fields}
    else:
        form_values = {f: "" for f in required_fields}

    return render_template("risk.html", result=result, message=message, risk=risk, form_values=form_values)


from datetime import datetime
import csv


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        phone = request.form.get("phone", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not phone or not password or not confirm_password:
            return render_template("register.html", error="All fields are required.")
        if password != confirm_password:
            return render_template("register.html", error="Passwords do not match.")
        if users_col.find_one({"email": email}):
            return render_template("register.html", error="An account with this email already exists.")

        hashed_pw = generate_password_hash(password)
        user = {
            "username": username,
            "email": email,
            "phone": phone,
            "password": hashed_pw,
            "created_at": datetime.utcnow(),
        }
        users_col.insert_one(user)
        return redirect(url_for("login", registered="1"))

    return render_template("register.html", error=None)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = users_col.find_one({"email": email})
        if not user or not check_password_hash(user["password"], password):
            return render_template("login.html", error="Invalid email or password.")
        session["user_id"] = str(user["_id"])
        session["email"] = user["email"]
        return redirect(url_for("dashboard"))

    if "user_id" in session:
        return redirect(url_for("dashboard"))

    registered_flag = request.args.get("registered")
    success_msg = "Registration successful. Please log in." if registered_flag else None
    return render_template("login.html", error=None, success=success_msg)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))

    from bson import ObjectId
    user = users_col.find_one({"_id": ObjectId(session["user_id"])})
    if not user:
        session.clear()
        return redirect(url_for("login"))

    error = None
    success = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        phone = request.form.get("phone", "").strip()
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not phone:
            error = "All fields except password are required."
        else:
            if email != user.get("email"):
                existing = users_col.find_one({"email": email})
                if existing:
                    error = "Another account already uses this email."

        if not error and new_password:
            if new_password != confirm_password:
                error = "New password and confirmation do not match."

        if not error:
            update_doc = {"username": username, "email": email, "phone": phone}
            if new_password:
                update_doc["password"] = generate_password_hash(new_password)
            users_col.update_one({"_id": user["_id"]}, {"$set": update_doc})
            user = users_col.find_one({"_id": user["_id"]})
            session["email"] = user.get("email")
            success = "Profile updated successfully."

    return render_template("profile.html", user=user, error=error, success=success)


def save_candidate_data(user_input, prob, risk_level, data_status):
    row = user_input.copy()
    row["probability"] = float(prob)
    row["risk_level"] = str(risk_level)
    row["data_status"] = str(data_status)
    row["timestamp"] = datetime.now().isoformat()

    filename = "data/candidate_data.csv"
    file_exists = False
    try:
        with open(filename, "r", newline="") as _:
            file_exists = True
    except FileNotFoundError:
        pass

    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


@app.route("/trends", methods=["GET"])
def trends():
    if "user_id" not in session:
        return redirect(url_for("login"))

    trend_labels = []
    trend_values = []
    cursor = predictions_col.find({"user_id": session["user_id"]}).sort("timestamp", 1)
    for doc in cursor:
        ts = doc.get("timestamp")
        label = ts.strftime("%Y-%m") if isinstance(ts, datetime) else str(ts)
        trend_labels.append(label)
        trend_values.append(float(doc.get("probability", 0)))

    return render_template(
        "trends.html",
        trend_labels=json.dumps(trend_labels),
        trend_values=json.dumps(trend_values),
    )


@app.route("/assistant", methods=["GET"])
def assistant_page():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("assistant.html")


@app.route("/doctor", methods=["GET"])
def doctor():
    if "user_id" not in session:
        return redirect(url_for("login"))

    ctx = session.get("last_risk_context") or {}
    if (ctx.get("risk_level") or "").upper() != "HIGH":
        return redirect(url_for("risk"))

    probability = float(ctx.get("probability") or 0.0)
    form_data = session.get("last_form_data") or {}
    return render_template("doctor.html", probability=probability, form_data=form_data)


def _build_gemini_lifestyle_prompt(risk, probability, raw):
    """
    Build a detailed, personalised Gemini prompt using all raw user inputs.
    raw is the dict stored in session["last_risk_context"]["raw"].
    """
    age = raw.get("age", "N/A")
    bmi = raw.get("bmi", "N/A")
    weight = raw.get("weight", "N/A")
    height = raw.get("height", "N/A")
    amh = raw.get("amh", "N/A")
    tsh = raw.get("tsh", "N/A")
    fsh = raw.get("fsh", "N/A")
    lh = raw.get("lh", "N/A")
    cycle = raw.get("cycle", "N/A")
    cycle_length = raw.get("cycle_length", "N/A")
    fsh_lh_ratio = raw.get("fsh_lh_ratio", "N/A")
    weight_gain = "Yes" if raw.get("weight_gain") else "No"
    hair_growth = "Yes" if raw.get("hair_growth") else "No"
    skin_darkening = "Yes" if raw.get("skin_darkening") else "No"
    hair_loss = "Yes" if raw.get("hair_loss") else "No"
    pimples = "Yes" if raw.get("pimples") else "No"
    fast_food = "Yes" if raw.get("fast_food") else "No"
    reg_exercise = "Yes" if raw.get("reg_exercise") else "No"

    bmi_category = "Unknown"
    try:
        b = float(bmi)
        if b < 18.5:
            bmi_category = "Underweight"
        elif b < 25:
            bmi_category = "Normal"
        elif b < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"
    except (TypeError, ValueError):
        pass

    cycle_type = "Regular" if str(cycle) == "2" else "Irregular"

    prompt = f"""You are an expert PCOS lifestyle coach and nutritionist.

A user has been assessed for PCOS risk. Based on their complete health profile below, generate a fully personalised 7-day lifestyle plan including diet and exercise. The plan must directly reflect the user's specific health data — do NOT give generic advice.

=== USER HEALTH PROFILE ===
PCOS Risk Level: {risk}
Model Probability: {probability}
Age: {age} years
Weight: {weight} kg | Height: {height} cm | BMI: {bmi} ({bmi_category})
Menstrual Cycle: {cycle_type} | Cycle/Bleeding Length: {cycle_length} days
Hormones: AMH = {amh} ng/mL | TSH = {tsh} mIU/L | FSH = {fsh} mIU/L | LH = {lh} mIU/L | FSH/LH Ratio = {fsh_lh_ratio}
Symptoms: Weight gain = {weight_gain} | Excess hair growth = {hair_growth} | Skin darkening = {skin_darkening} | Hair loss = {hair_loss} | Pimples/Acne = {pimples}
Lifestyle: Eats fast food frequently = {fast_food} | Exercises regularly = {reg_exercise}

=== PERSONALISATION RULES ===
- If BMI >= 30 (Obese): Focus on calorie deficit meals, anti-inflammatory foods, and daily cardio. Be explicit.
- If BMI 25–29.9 (Overweight): Moderate calorie control, reduce refined carbs, mix of cardio and strength.
- If BMI < 25 (Normal/Underweight): Focus on hormone-supportive nutrition, not weight loss.
- If fast food = Yes: Include specific meal swaps. Name the unhealthy foods to avoid and their healthy replacements.
- If fast food = No: Reinforce and optimize their existing good habits with more variety.
- If exercise = No: Start with beginner-friendly, low-impact workouts. Build progressively across the week.
- If exercise = Yes: Suggest intermediate/advanced progressions. Vary the workout types.
- If pimples = Yes: Include anti-acne diet recommendations (low glycemic, zinc-rich foods).
- If hair loss = Yes: Include hair-health nutrients (biotin, iron, omega-3) in meal suggestions.
- If excess hair growth = Yes: Mention spearmint tea and low-androgen diet strategies.
- If weight gain = Yes: Include metabolism-boosting strategies and portion guidance.
- If skin darkening = Yes: Suggest insulin-sensitizing foods (cinnamon, berberine-rich options).
- If irregular cycle: Emphasize seed cycling, anti-inflammatory foods, and consistent sleep.
- If AMH is high (> 4.0): Mention that diet and lifestyle can help manage elevated AMH through anti-inflammatory approaches.
- If TSH is abnormal (< 0.4 or > 4.0): Note thyroid-friendly foods (selenium, iodine) in the diet.
- If FSH/LH ratio is abnormal (> 1.0 or LH/FSH > 2.0): Note hormonal balance tips.
- Risk HIGH: Be more strict and urgent. Emphasize medical consultation alongside the plan.
- Risk MEDIUM: Be proactive and structured. Frame as a corrective plan.
- Risk LOW: Be encouraging and maintenance-focused.

=== OUTPUT FORMAT ===
Generate the response in EXACTLY this format with no extra explanation outside these sections:

DIET_PLAN:
Monday: [Breakfast] | [Lunch] | [Dinner] | [Snacks]
Tuesday: [Breakfast] | [Lunch] | [Dinner] | [Snacks]
Wednesday: [Breakfast] | [Lunch] | [Dinner] | [Snacks]
Thursday: [Breakfast] | [Lunch] | [Dinner] | [Snacks]
Friday: [Breakfast] | [Lunch] | [Dinner] | [Snacks]
Saturday: [Breakfast] | [Lunch] | [Dinner] | [Snacks]
Sunday: [Breakfast] | [Lunch] | [Dinner] | [Snacks]

EXERCISE_PLAN:
Monday: [Exercise name] - [Duration] - [Intensity] - [Why it helps]
Tuesday: [Exercise name] - [Duration] - [Intensity] - [Why it helps]
Wednesday: [Exercise name] - [Duration] - [Intensity] - [Why it helps]
Thursday: [Exercise name] - [Duration] - [Intensity] - [Why it helps]
Friday: [Exercise name] - [Duration] - [Intensity] - [Why it helps]
Saturday: [Exercise name] - [Duration] - [Intensity] - [Why it helps]
Sunday: [Exercise name] - [Duration] - [Intensity] - [Why it helps]

LIFESTYLE_TIPS:
- [Tip specific to user's symptoms/data]
- [Tip specific to user's symptoms/data]
- [Tip specific to user's symptoms/data]
- [Tip specific to user's symptoms/data]
- [Tip specific to user's symptoms/data]

STRESS_AND_SLEEP:
- [Personalised sleep/stress advice based on user data]
- [Personalised sleep/stress advice based on user data]
- [Personalised sleep/stress advice based on user data]

Do not add any text, headings, or explanation outside of these four sections."""

    return prompt


def _parse_gemini_lifestyle_response(text):
    """
    Parse Gemini output into structured dicts/lists.
    Returns (diet_plan, exercise_plan, lifestyle_tips, stress_sleep).
    All are empty if parsing fails.
    """
    diet_plan = {}
    exercise_plan = {}
    lifestyle_tips = []
    stress_sleep = []

    section = None
    days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("DIET_PLAN"):
            section = "diet"
            continue
        elif line.startswith("EXERCISE_PLAN"):
            section = "exercise"
            continue
        elif line.startswith("LIFESTYLE_TIPS"):
            section = "tips"
            continue
        elif line.startswith("STRESS_AND_SLEEP"):
            section = "stress"
            continue

        if section == "diet" and ":" in line:
            day, _, value = line.partition(":")
            day = day.strip()
            if day in days:
                diet_plan[day] = value.strip()

        elif section == "exercise" and ":" in line:
            day, _, value = line.partition(":")
            day = day.strip()
            if day in days:
                exercise_plan[day] = value.strip()

        elif section == "tips" and line.startswith("-"):
            tip = line.lstrip("- ").strip()
            if tip:
                lifestyle_tips.append(tip)

        elif section == "stress" and line.startswith("-"):
            item = line.lstrip("- ").strip()
            if item:
                stress_sleep.append(item)

    return diet_plan, exercise_plan, lifestyle_tips, stress_sleep


def _minimal_fallback_lifestyle():
    """
    Minimal fallback when Gemini API is unavailable or fails.
    Shows a message to configure the API.
    """
    diet_plan = {"Monday-Sunday": "AI recommendations unavailable. Please check Gemini API configuration."}
    exercise_plan = {"Monday-Sunday": "AI recommendations unavailable. Please check Gemini API configuration."}
    lifestyle_tips = ["Gemini API is not configured or unavailable. Please set up your GEMINI_API_KEY environment variable."]
    stress_sleep = ["Contact your healthcare provider for personalized guidance."]

    return diet_plan, exercise_plan, lifestyle_tips, stress_sleep


def get_cached_lifestyle_plan(user_id, risk_level, raw_input_json):
    """Check if a lifestyle plan exists in cache."""
    conn = sqlite3.connect("pcod.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT diet_plan, exercise_plan, lifestyle_tips, stress_sleep FROM lifestyle_plans WHERE user_id = ? AND risk_level = ? AND raw_input = ?",
        (user_id, risk_level, raw_input_json)
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "diet_plan": json.loads(row[0]),
            "exercise_plan": json.loads(row[1]),
            "lifestyle_tips": json.loads(row[2]),
            "stress_sleep": json.loads(row[3])
        }
    return None


def save_lifestyle_plan(user_id, risk_level, raw_input_json, diet_plan, exercise_plan, lifestyle_tips, stress_sleep):
    """Save a lifestyle plan to cache."""
    conn = sqlite3.connect("pcod.db")
    cursor = conn.cursor()
    created_at = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        "INSERT INTO lifestyle_plans (user_id, risk_level, raw_input, diet_plan, exercise_plan, lifestyle_tips, stress_sleep, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            user_id,
            risk_level,
            raw_input_json,
            json.dumps(diet_plan),
            json.dumps(exercise_plan),
            json.dumps(lifestyle_tips),
            json.dumps(stress_sleep),
            created_at
        )
    )
    conn.commit()
    conn.close()


@app.route("/lifestyle", methods=["GET"])
def lifestyle():
    """Personalised lifestyle plan generated by Gemini based on all user inputs."""
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]
    ctx = session.get("last_risk_context") or {}
    risk = (ctx.get("risk_level") or "MEDIUM").upper()
    probability = ctx.get("probability")
    raw = ctx.get("raw") or {}

    # Convert raw input to JSON for comparison
    raw_input_json = json.dumps(raw, sort_keys=True)

    # Check cache first
    cached_plan = get_cached_lifestyle_plan(user_id, risk, raw_input_json)
    if cached_plan:
        return render_template(
            "lifestyle.html",
            risk=risk,
            probability=probability,
            diet_plan=cached_plan["diet_plan"],
            exercise_plan=cached_plan["exercise_plan"],
            lifestyle_tips=cached_plan["lifestyle_tips"],
            stress_sleep=cached_plan["stress_sleep"],
            gemini_used=False,
        )

    api_key = os.getenv("GEMINI_API_KEY", "")

    diet_plan = {}
    exercise_plan = {}
    lifestyle_tips = []
    stress_sleep = []
    gemini_used = False

    if api_key and raw:
        try:
            from google import genai

            client = genai.Client(api_key=api_key)

            prompt = _build_gemini_lifestyle_prompt(risk, probability, raw)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "temperature": 0
                }
            )

            text = response.text if hasattr(response, "text") else ""

            if text:
                diet_plan, exercise_plan, lifestyle_tips, stress_sleep = _parse_gemini_lifestyle_response(text)
                if diet_plan and exercise_plan:
                    gemini_used = True
                    # Save to cache
                    save_lifestyle_plan(
                        user_id,
                        risk,
                        raw_input_json,
                        diet_plan,
                        exercise_plan,
                        lifestyle_tips,
                        stress_sleep
                    )

        except Exception as e:
            print("Gemini Lifestyle Error:", e)
            diet_plan, exercise_plan, lifestyle_tips, stress_sleep = _minimal_fallback_lifestyle()

    else:
        diet_plan, exercise_plan, lifestyle_tips, stress_sleep = _minimal_fallback_lifestyle()

    return render_template(
        "lifestyle.html",
        risk=risk,
        probability=probability,
        diet_plan=diet_plan,
        exercise_plan=exercise_plan,
        lifestyle_tips=lifestyle_tips,
        stress_sleep=stress_sleep,
        gemini_used=gemini_used,
    )


@app.route("/chat", methods=["POST"])
def chat():
    """AI assistant endpoint for lifestyle guidance only."""
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Message is required."}), 400

    ctx = session.get("last_risk_context") or {}
    risk_level = (ctx.get("risk_level") or "UNKNOWN").upper()
    features = ctx.get("features") or {}
    probability = ctx.get("probability")

    feature_lines = [f"- {k}: {v}" for k, v in features.items()]
    feature_summary = "\n".join(feature_lines) if feature_lines else "No structured features available."

    system_instructions = (
        "You are a PCOS lifestyle assistant. "
        "You must NOT provide diagnosis or medication advice. "
        "You can only give general information, lifestyle, diet, exercise, "
        "and habit guidance based on risk levels and user context. "
        "Always include a reminder to consult a doctor for medical decisions."
    )

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return jsonify({"reply": "Gemini API not configured."})

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        prompt = (
            f"{system_instructions}\n\n"
            f"Most recent model prediction context:\n"
            f"- Risk level: {risk_level}\n"
            f"- Model probability of PCOS: {probability if probability is not None else 'N/A'}\n"
            f"- User features:\n{feature_summary}\n\n"
            f"User question: {user_message}"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        reply_text = response.text.strip() if hasattr(response, "text") else ""

        if not reply_text:
            raise ValueError("Empty response from Gemini")

    except Exception as e:
        print("Gemini Chat Error:", e)
        reply_text = (
            "AI service temporarily unavailable. "
            "Please consult a doctor for medical advice."
        )

    return jsonify({"reply": reply_text})

@app.route("/list-models")
def list_models():
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    models = client.models.list()

    model_names = []
    for m in models:
        model_names.append(m.name)

    return jsonify(model_names)
@app.route("/save-tracker", methods=["POST"])
def save_tracker():
    if "user_id" not in session:
        return jsonify({"status": "error"}), 401

    data = request.get_json()

    tracker_entry = {
        "user_id": session["user_id"],
        "date": data.get("date"),
        "cycle_day": data.get("cycle_day"),
        "mood": data.get("mood"),
        "acne": data.get("acne"),
        "hair_loss": data.get("hair_loss"),
        "weight": data.get("weight"),
    }

    tracker_col.insert_one(tracker_entry)

    return jsonify({"status": "success"})
@app.route("/tracker")
def tracker():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("tracker.html")
@app.route("/tracker-history")
def tracker_history():
    if "user_id" not in session:
        return redirect(url_for("login"))

    data = list(tracker_col.find({"user_id": session["user_id"]}).sort("date", 1))

    dates = []
    weights = []

    for entry in data:
        dates.append(entry.get("date"))
        weights.append(float(entry.get("weight", 0)))

    return render_template(
        "tracker_history.html",
        dates=json.dumps(dates),
        weights=json.dumps(weights),
        entries=data
    )

def decision_after_prediction(user_input, prob):
    if prob < 0.35:
        return "LOW"
    elif prob < 0.65:
        return "MEDIUM"
    else:
        return "HIGH"


if __name__ == "__main__":
    app.run(debug=True)
