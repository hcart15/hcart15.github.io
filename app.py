from flask import Flask, render_template, jsonify
import pandas as pd
import os

# Ensure the CSV path is correct
CSV_PATH = os.path.join(os.path.dirname(__file__), 'consolidated_data_final_with_composite_boosts.csv')

# Load the CSV into a DataFrame
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(f"✅ CSV loaded successfully from {CSV_PATH}")
else:
    print(f"⚠️ CSV not found at: {CSV_PATH}")
    df = pd.DataFrame()

# Initialize Flask app
app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Risk Assessment Page
@app.route('/risk')
def risk():
    communities = df['Community'].unique().tolist() if not df.empty else ["Community 1", "Community 2"]
    return render_template('risk.html', communities=communities)

# CEI Page
@app.route('/cei')
def cei():
    return render_template('cei.html')

# Employment Page
@app.route('/employment')
def employment():
    return render_template('employment.html')

# ML Insights Page
@app.route('/ml')
def ml():
    return render_template('ml.html')

# API for Dynamic Data (Optional)
@app.route('/api/risk_data', methods=['GET'])
def risk_data():
    if not df.empty:
        return jsonify(df.to_dict(orient='records'))
    else:
        return jsonify({"error": "No data available"}), 404

# Run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
