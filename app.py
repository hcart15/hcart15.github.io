from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__, template_folder='.')

# Load CSV Data
CSV_PATH = os.path.join(os.path.dirname(__file__), 'consolidated_data_final_with_composite_boosts.csv')
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(f"✅ CSV loaded successfully from {CSV_PATH}")
else:
    print(f"⚠️ CSV not found at: {CSV_PATH}")
    df = pd.DataFrame()

# Mock Data for Dropdowns (replace with actual logic if needed)
property_types = ['Residential', 'Commercial', 'Industrial']
communities = df['Community'].unique().tolist() if not df.empty else ['Community A', 'Community B', 'Community C']

# -------------------------------
# Home Route
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -------------------------------
# Risk Assessment Route
# -------------------------------
@app.route('/risk', methods=['GET', 'POST'])
def risk():
    risk_score = None
    consequence = None
    plot_url = None

    if request.method == 'POST':
        property_type = request.form.get('property_type')
        community = request.form.get('community')

        if property_type and community:
            # Example risk calculation (replace with actual logic)
            risk_score = 75  # Placeholder score
            consequence = "Moderate"
            plot_url = None  # Replace with actual plot generation

    return render_template('risk.html',
                           property_types=property_types,
                           communities=communities,
                           risk_score=risk_score,
                           consequence=consequence,
                           plot_url=plot_url)

# -------------------------------
# CEI Data Route
# -------------------------------
@app.route('/cei')
def cei():
    if df.empty:
        table_data = "<p>No CEI data available.</p>"
    else:
        cei_df = df[['Community', 'CEI Score']]  # Replace with actual columns
        table_data = cei_df.to_html(classes='table table-bordered', index=False)

    return render_template('cei.html', table_data=table_data)

# -------------------------------
# Employment Data Route
# -------------------------------
@app.route('/employment')
def employment():
    if df.empty:
        table_data = "<p>No employment data available.</p>"
    else:
        employment_df = df[['Community', 'Employment Rate']]  # Replace with actual columns
        table_data = employment_df.to_html(classes='table table-bordered', index=False)

    return render_template('employment.html', table_data=table_data)

# -------------------------------
# ML Risk Assessment Route
# -------------------------------
@app.route('/ml', methods=['GET', 'POST'])
def ml():
    risk_prediction = None

    if request.method == 'POST':
        community = request.form.get('community')
        if community:
            # Example ML prediction (replace with actual model)
            risk_prediction = "High" if community == "Community A" else "Low"

    return render_template('ml.html', communities=communities, risk_prediction=risk_prediction)

# -------------------------------
# API Endpoint for Risk Data
# -------------------------------
@app.route('/api/risk_data', methods=['GET'])
def risk_data():
    if not df.empty:
        return jsonify(df.to_dict(orient='records'))
    else:
        return jsonify({"error": "No data available"}), 404

# -------------------------------
# Run Flask App
# -------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
