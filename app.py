import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for Flask
from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import joblib

app = Flask(__name__)

# Load consolidated data (adjust file path if needed)
consolidated_data = pd.read_csv("data/consolidated_data_final_with_composite_boosts copy.csv")

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
        
        # Generate risk plot with axis labels
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axvline(x=50, color="black", linestyle="-", linewidth=1.5)
        ax.axhline(y=50, color="black", linestyle="-", linewidth=1.5)
        ax.grid(True, color="lightgray", linestyle="--", linewidth=0.5)
        ax.scatter(risk_score, consequence, color="red", s=60)
        ax.text(risk_score+1, consequence+1, f"({risk_score:.2f}, {consequence:.2f})", color="blue", fontsize=10)
        ax.set_xlabel("Likelihood (0=low, 100=high)")
        ax.set_ylabel("Consequence (0=low, 100=high)")
        
        buf = BytesIO()
        plt.savefig(buf, format="png")
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

def calculate_risk_score(property_type, community):
    # Example risk calculation logic – integrate your detailed method here.
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
    CRIME_SEVERITY_WEIGHTS = {
        "Theft FROM Vehicle": 2,
        "Break & Enter - Commercial": 7,
        "Break & Enter - Dwelling": 6,
        "Theft OF Vehicle": 4,
        "Violence Other (Non-domestic)": 8,
        "Assault (Non-domestic)": 9,
        "Street Robbery": 7,
        "Commercial Robbery": 9,
        "Break & Enter - Other Premises": 5
    }
    base_severity = AI_PROPERTY_SEVERITY.get(property_type, 5)
    base_freq = COMMON_SENSE_FREQUENCY.get(property_type, 5)
    subset = consolidated_data[consolidated_data["Community"] == community]
    if subset.empty:
        return 0, 0
    total_crime_weighted = subset["Crime Count"].sum() if "Crime Count" in subset.columns else 0
    crime_component = total_crime_weighted / 50.0
    property_risk = base_freq + crime_component
    property_risk = max(0, min(property_risk, 100))
    consequence = min(base_severity * 10.0, 100)
    return property_risk, consequence

# ---------------------
# CEI Data Tab (Tab 2)
# ---------------------
# Define the exact list of CEI columns to display (including "Community")
cei_display_columns = [
    "Community",
    "CTUID",
    "Quadrant",
    "total_weighted_CEI_Score",
    "economic_opportunity_domain_score",
    "accessibility_and_amenities_domain_score",
    "population_health_domain_score",
    "community_belonging_and_safety_domain_score",
    "human_and_social_wellbeing_domain_score",
    "climate_and_environment_domain_score",
    "activity_limitation_relative_severity_perc",
    "adult_litp",
    "average_commute_time",
    "average_transitscore",
    "average_walkscore",
    "children_perc",
    "commute_transit_perc",
    "commute_vehicle_perc",
    "copd_agestd_perc",
    "core_housing_owner_perc",
    "core_housing_renter_perc",
    "diabetes_agestd_perc",
    "educ_no_cert_perc",
    "first_generation_perc",
    "flood_susceptibility",
    "gender_income_ratio",
    "income_inequality_ratio",
    "indigenous_perc",
    "land_surface_temp",
    "lim_at_perc",
    "mental_illness_agestd_perc",
    "movers_1_year_ago",
    "no_official_languages_perc",
    "noise_pollution_score",
    "park_area_perc",
    "part_time_employment_perc",
    "perceived_health_vg_excellent_perc",
    "perceived_mental_vg_excellent_perc",
    "person_crime_rate",
    "physical_disorder_rate",
    "population_in_poverty_perc",
    "property_crime_rate",
    "proximity_childcare",
    "proximity_grocerystores",
    "proximity_healthcare",
    "proximity_library",
    "proximity_neighbourhood_parks",
    "proximity_pharmacies",
    "proximity_primary_education",
    "proximity_secondary_education",
    "recent_immigration_perc",
    "refugee_perc",
    "regular_care_provider_yes_perc",
    "renter_households_perc",
    "second_generation_perc",
    "seniors_65_perc",
    "seniors_in_poverty_perc",
    "seniors_living_alone_perc",
    "sense_of_belonging_perc",
    "single_parent_econ_dependency",
    "single_parent_woman_perc",
    "social_assistance_perc",
    "social_disorder_rate",
    "spatial_access_primary_secondary_education_transit",
    "tree_canopy",
    "tree_density",
    "unemployment_rate_perc",
    "voter_turnout_rate",
    "working_poor_18to64_perc",
    "youth_neet_perc"
]

# Define the rename map (do not include "Community" since we want it unchanged)
cei_rename_map = {
    "CTUID": "Census Tract Unique ID",
    "Quadrant": "City Quadrants",
    "total_weighted_CEI_Score": "Total Equity Score",
    "economic_opportunity_domain_score": "Economic Opportunity Domain Score",
    "accessibility_and_amenities_domain_score": "Accessibility and Amenities Domain Score",
    "population_health_domain_score": "Population Health Domain Score",
    "community_belonging_and_safety_domain_score": "Community Belonging and Safety Domain Score",
    "human_and_social_wellbeing_domain_score": "Human and Social Wellbeing Domain Score",
    "climate_and_environment_domain_score": "Climate and Environment Domain Score",
    "activity_limitation_relative_severity_perc": "Activity Limitation (Relative Severity) (%)",
    "adult_litp": "Low Income Transit Pass (Adult) (%)",
    "average_commute_time": "Commute to Work Time (minutes)",
    "average_transitscore": "Transit Score",
    "average_walkscore": "Walk Score",
    "children_perc": "Children (aged 0 to 14) (%)",
    "commute_transit_perc": "Commute to Work by Transit (%)",
    "commute_vehicle_perc": "Commute to Work by Vehicle (%)",
    "copd_agestd_perc": "COPD Prevalence (%)",
    "core_housing_owner_perc": "Core Housing Need - Owner (%)",
    "core_housing_renter_perc": "Core Housing Need - Renter (%)",
    "diabetes_agestd_perc": "Diabetes Prevalence (%)",
    "educ_no_cert_perc": "No Certificate, Degree, or Diploma (%)",
    "first_generation_perc": "First Generation Immigrants (%)",
    "flood_susceptibility": "Flood Susceptibility Score",
    "gender_income_ratio": "Gender Income Ratio (After-Tax Median Income)",
    "income_inequality_ratio": "Income Inequality Ratio (P90/P10)",
    "indigenous_perc": "Indigenous Identity (%)",
    "land_surface_temp": "Land Surface Temperature",
    "lim_at_perc": "Low-Income Measure (After-Tax) (%)",
    "mental_illness_agestd_perc": "Mental Illness Prevalence (%)",
    "movers_1_year_ago": "Mobility Status (%)",
    "no_official_languages_perc": "No Knowledge of Official Languages (%)",
    "noise_pollution_score": "Noise Pollution Score",
    "park_area_perc": "Share of Park Area (%)",
    "part_time_employment_perc": "Part-Time or Part-Year Employment (%)",
    "perceived_health_vg_excellent_perc": "Perceived Health (%)",
    "perceived_mental_vg_excellent_perc": "Perceived Mental Health (%)",
    "person_crime_rate": "Person Crime Rate",
    "physical_disorder_rate": "Physical Disorder Rate",
    "population_in_poverty_perc": "Population in Poverty (%)",
    "property_crime_rate": "Property Crime Rate",
    "proximity_childcare": "Proximity to Childcare",
    "proximity_grocerystores": "Proximity to Grocery Store",
    "proximity_healthcare": "Proximity to Healthcare",
    "proximity_library": "Proximity to Library",
    "proximity_neighbourhood_parks": "Proximity to Neighbourhood Parks",
    "proximity_pharmacies": "Proximity to Pharmacies",
    "proximity_primary_education": "Proximity to Primary Education",
    "proximity_secondary_education": "Proximity to Secondary Education",
    "recent_immigration_perc": "Recently Immigrated (%)",
    "refugee_perc": "Refugees (%)",
    "regular_care_provider_yes_perc": "Has a Regular Health Care Provider (%)",
    "renter_households_perc": "Renter Households (%)",
    "second_generation_perc": "Second Generation Immigrants (%)",
    "seniors_65_perc": "Seniors (aged 65+) (%)",
    "seniors_in_poverty_perc": "Seniors in Poverty (%)",
    "seniors_living_alone_perc": "Seniors Living Alone (%)",
    "sense_of_belonging_perc": "Sense of Belonging to Local Community (%)",
    "single_parent_econ_dependency": "Economic Dependency Ratio (Single-Parent Families)",
    "single_parent_woman_perc": "Single-Parent (Woman+) Households (%)",
    "social_assistance_perc": "Social Assistance Benefits (%)",
    "social_disorder_rate": "Social Disorder Rate",
    "spatial_access_primary_secondary_education_transit": "Spatial Access to Primary and Secondary Education",
    "tree_canopy": "Tree Canopy (%)",
    "tree_density": "Tree Density",
    "unemployment_rate_perc": "Unemployment Rate (%)",
    "voter_turnout_rate": "Municipal Voter Turnout Rate (%)",
    "working_poor_18to64_perc": "Working Poor, Excluding Students (%)",
    "youth_neet_perc": "Youth Not in Employment, Education, or Training (%)"
}

@app.route("/cei")
def cei():
    # Keep only the columns specified in cei_display_columns that exist in our data
    existing_columns = [col for col in cei_display_columns if col in consolidated_data.columns]
    df_cei = consolidated_data[existing_columns].copy()
    # Rename columns using the rename map (only renames columns present in the map; Community remains unchanged)
    df_cei.rename(columns=cei_rename_map, inplace=True)
    # Convert DataFrame to HTML table with a specified table ID for DataTables initialization
    table_html = df_cei.dropna().to_html(classes="table table-striped", index=False, table_id="data_table")
    return render_template("cei.html", table_data=table_html)

# ---------------------
# Employment Data Tab (Tab 3)
# ---------------------
@app.route("/employment")
def employment():
    emp_columns = ["Community", "EMPLOYED", "UNEMPLOYED", "TOTAL_POP_OVER_15_HOUSEHOLD",
                   "IN_LABOUR_FORCE", "SELF_EMPLOYED", "NOT_IN_LABOUR_FORCE"]
    emp_columns = [col for col in emp_columns if col in consolidated_data.columns]
    table_html = consolidated_data[emp_columns].dropna().to_html(classes="table table-bordered", index=False, table_id="data_table")
    return render_template("employment.html", table_data=table_html)

# ---------------------
# ML Risk Assessment Tab (Tab 4)
# ---------------------
@app.route("/ml", methods=["GET", "POST"])
def ml():
    communities = sorted(consolidated_data["Community"].dropna().unique())
    risk_prediction = None
    if request.method == "POST":
        selected_community = request.form.get("community_ml")
        # Load the pre-trained model (make sure risk_model.pkl exists in your project directory)
        try:
            risk_model = joblib.load("risk_model.pkl copy")
        except Exception as e:
            risk_prediction = f"Error loading model: {e}"
            return render_template("ml.html", communities=communities, risk_prediction=risk_prediction)
        
        # Prepare input: for simplicity, take the row corresponding to the community
        row = consolidated_data[consolidated_data["Community"] == selected_community]
        if row.empty:
            risk_prediction = "Community data not found."
        else:
            # Use all numeric columns (except risk_score if present)
            feature_cols = [col for col in row.select_dtypes(include=["number"]).columns if col.lower() != "risk_score"]
            X_new = row[feature_cols].fillna(0)
            if X_new.shape[0] > 1:
                X_new = X_new.mean().to_frame().T
            else:
                X_new = X_new.iloc[[0]]
            # Reorder columns if the model has a feature_names_in_ attribute
            if hasattr(risk_model, "feature_names_in_"):
                expected_features = risk_model.feature_names_in_
                X_new = X_new.reindex(columns=expected_features, fill_value=0)
            try:
                pred = risk_model.predict(X_new)[0]
                risk_prediction = f"Predicted ML Risk Score for {selected_community}: {pred:.2f}"
            except Exception as e:
                risk_prediction = f"ML prediction failed: {e}"
    return render_template("ml.html", communities=communities, risk_prediction=risk_prediction)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
