from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load CSV and Model (example)
DATA_PATH = os.path.join(os.path.dirname(__file__), 'Data', 'consolidated_data_final_with_composite_boosts.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'risk_model.pkl')

# Load data
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    df = pd.DataFrame()

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
    return jsonify(df.to_dict(orient='records'))

# Run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
