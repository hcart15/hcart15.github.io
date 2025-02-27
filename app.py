import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Flask
from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask_caching import Cache  # Import caching
import joblib

# Load CSV only once (instead of on every request)
consolidated_data = pd.read_csv("consolidated_data_final_with_composite_boosts.csv")

# Initialize Flask App
app = Flask(__name__, template_folder="templates", static_folder="static")

# Setup Flask-Caching
cache = Cache(app, config={"CACHE_TYPE": "simple"})  # Simple in-memory cache

# Cache static files (CSS, JS, images) for 1 day
@app.route('/static/<path:filename>')
@cache.cached(timeout=86400)  # Cache for 24 hours
def cached_static(filename):
    return send_from_directory("static", filename)

# ---------------------
# Home Route
# ---------------------
@app.route("/")
def index():
    return render_template("index.html")

# ---------------------
# Risk Assessment Tab (Tab 1)
# ---------------------
@app.route("/risk", methods=["GET", "POST"])
def risk():
    property_types = [
        "Bank", "Grocery Store", "Flower Shop", "Gas Station", "Pharmacy",
        "Restaurant", "Retail Store", "Convenience Store", "Shopping Mall",
        "Office Building", "Warehouse", "Factory", "Park", "Parking Lot",
        "Residential House", "Gym", "Library", "Church", "Bar", "Hotel",
        "School", "Medical Clinic"
    ]
    communities = sorted(consolidated_data["Community"].dropna().unique())

    risk_score = None
    consequence = None
    plot_url = None

    if request.method == "POST":
        selected_property = request.form.get("property_type")
        selected_community = request.form.get("community")
        risk_score, consequence = calculate_risk_score(selected_property, selected_community)

        if risk_score is not None and consequence is not None:
            fig, ax = plt.subplots(figsize=(6, 6), facecolor="#f4f4f4")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)

            ax.scatter(risk_score, consequence, color="red", s=100, edgecolors="black", linewidth=1.5, alpha=0.9)
            ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
            ax.axvline(x=50, color="black", linestyle="-", linewidth=1.2, alpha=0.5)
            ax.axhline(y=50, color="black", linestyle="-", linewidth=1.2, alpha=0.5)
            ax.text(risk_score + 2, consequence + 2, f"({risk_score:.2f}, {consequence:.2f})",
                    color="blue", fontsize=12, weight="bold")

            ax.set_xlabel("Likelihood (0 = Low, 100 = High)", fontsize=12, weight="bold")
            ax.set_ylabel("Consequence (0 = Low, 100 = High)", fontsize=12, weight="bold")
            ax.set_title("Risk Assessment", fontsize=14, weight="bold", pad=15)

            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
            plot_url = f"data:image/png;base64,{plot_data}"
            plt.close(fig)

    return render_template("risk.html",
                           property_types=property_types,
                           communities=communities,
                           risk_score=risk_score,
                           consequence=consequence,
                           plot_url=plot_url)

# ---------------------
# CEI Data Tab (Tab 2)
# ---------------------
@app.route("/cei")
def cei():
    # Define only the 5 required columns for display
    cei_display_columns = [
        "Community",
        "total_weighted_CEI_Score",
        "community_belonging_and_safety_domain_score",
        "economic_opportunity_domain_score",
        "adult_litp"
    ]

    # Ensure selected columns exist in the dataset
    existing_columns = [col for col in cei_display_columns if col in consolidated_data.columns]
    df_cei = consolidated_data[existing_columns].copy()

    # Rename columns for better readability
    cei_rename_map = {
        "total_weighted_CEI_Score": "Total Equity Score",
        "community_belonging_and_safety_domain_score": "Community Belonging and Safety Domain Score",
        "economic_opportunity_domain_score": "Economic Opportunity Domain Score",
        "adult_litp": "Low Income Transit Pass (Adult) (%)"
    }
    df_cei.rename(columns=cei_rename_map, inplace=True)

    # Convert data to HTML table
    table_html = df_cei.dropna().to_html(classes="table table-striped", index=False, table_id="data_table")

    return render_template("cei.html", table_data=table_html)

# ---------------------
# Employment Data Tab (Tab 3)
# ---------------------
@app.route("/employment")
def employment():
    emp_columns = [
        "Community", "EMPLOYED", "UNEMPLOYED", "TOTAL_POP_OVER_15_HOUSEHOLD",
        "IN_LABOUR_FORCE", "SELF_EMPLOYED", "NOT_IN_LABOUR_FORCE"
    ]

    existing_columns = [col for col in emp_columns if col in consolidated_data.columns]
    df_emp = consolidated_data[existing_columns].copy()

    table_html = df_emp.dropna().to_html(classes="table table-striped", index=False, table_id="data_table")

    return render_template("employment.html", table_data=table_html)

# ---------------------
# ML Risk Assessment Tab (Tab 4)
# ---------------------
@app.route("/ml", methods=["GET", "POST"])
def ml():
    communities = sorted(consolidated_data["Community"].dropna().unique())
    risk_prediction = None
    selected_community = None  # Track the selected community

    if request.method == "POST":
        selected_community = request.form.get("community_ml")

        try:
            risk_model = joblib.load("risk_model.pkl")
        except Exception as e:
            risk_prediction = f"Error loading model: {e}"
            return render_template("ml.html", communities=communities, selected_community=selected_community, risk_prediction=risk_prediction)

        row = consolidated_data[consolidated_data["Community"] == selected_community]
        if row.empty:
            risk_prediction = "Community data not found."
        else:
            feature_cols = [col for col in row.select_dtypes(include=["number"]).columns if col.lower() != "risk_score"]
            X_new = row[feature_cols].fillna(0)
            if X_new.shape[0] > 1:
                X_new = X_new.mean().to_frame().T
            else:
                X_new = X_new.iloc[[0]]

            if hasattr(risk_model, "feature_names_in_"):
                expected_features = risk_model.feature_names_in_
                X_new = X_new.reindex(columns=expected_features, fill_value=0)

            try:
                pred = risk_model.predict(X_new)[0]
                risk_prediction = f"Predicted ML Risk Score for {selected_community}: {pred:.2f}"
            except Exception as e:
                risk_prediction = f"ML prediction failed: {e}"

    return render_template("ml.html", communities=communities, selected_community=selected_community, risk_prediction=risk_prediction)

# ---------------------
# Risk Score Calculation Function
# ---------------------
def calculate_risk_score(property_type, community):
    AI_PROPERTY_SEVERITY = {
        "Bank": 9, "Grocery Store": 6, "Flower Shop": 2, "Gas Station": 7, "Pharmacy": 8,
        "Restaurant": 5, "Retail Store": 4, "Convenience Store": 7, "Shopping Mall": 7,
        "Office Building": 8, "Warehouse": 7, "Factory": 6, "Park": 3, "Parking Lot": 6,
        "Residential House": 5, "Gym": 3, "Library": 2, "Church": 2, "Bar": 7, "Hotel": 6,
        "School": 4, "Medical Clinic": 6
    }
    COMMON_SENSE_FREQUENCY = {
        "Bank": 3, "Grocery Store": 6, "Flower Shop": 1, "Gas Station": 7, "Pharmacy": 5,
        "Restaurant": 4, "Retail Store": 5, "Convenience Store": 8, "Shopping Mall": 6,
        "Office Building": 3, "Warehouse": 4, "Factory": 3, "Park": 2, "Parking Lot": 6,
        "Residential House": 5, "Gym": 3, "Library": 1, "Church": 1, "Bar": 7, "Hotel": 6,
        "School": 4, "Medical Clinic": 5
    }
    
    base_severity = AI_PROPERTY_SEVERITY.get(property_type, 5)
    base_freq = COMMON_SENSE_FREQUENCY.get(property_type, 5)
    
    subset = consolidated_data[consolidated_data["Community"] == community]
    if subset.empty:
        return 0, 0

    total_crime_weighted = subset["Crime Count"].sum() if "Crime Count" in subset.columns else 0
    property_risk = max(0, min(base_freq + (total_crime_weighted / 50.0), 100))
    consequence = min(base_severity * 10.0, 100)

    return property_risk, consequence

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")

