from flask import Flask, render_template, jsonify
import pandas as pd
import os

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Data', 'consolidated_data_final_with_composite_boosts.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'risk_model.pkl')

# Load CSV into DataFrame
try:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print("✅ CSV loaded successfully.")
    else:
        print(f"⚠️ CSV not found at: {DATA_PATH}")
        df = pd.DataFrame()
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    df = pd.DataFrame()

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

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
