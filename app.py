import os
import pandas as pd
from flask import Flask, render_template, jsonify

app = Flask(__name__, template_folder='templates', static_folder='static')

# Path to CSV
CSV_FILE = os.path.join(os.path.dirname(__file__), 'consolidated_data_final_with_composite_boosts.csv')

# Load Data
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    print("✅ CSV loaded successfully.")
else:
    print(f"⚠️ CSV not found at: {CSV_FILE}")
    df = pd.DataFrame()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/risk')
def risk():
    communities = df['Community'].unique().tolist() if not df.empty else ["Community 1", "Community 2"]
    return render_template('risk.html', communities=communities)

@app.route('/cei')
def cei():
    return render_template('cei.html')

@app.route('/employment')
def employment():
    return render_template('employment.html')

@app.route('/ml')
def ml():
    return render_template('ml.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
