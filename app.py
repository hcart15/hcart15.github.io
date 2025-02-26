from flask import Flask, render_template
import pandas as pd
import os

app = Flask(__name__, template_folder='.', static_folder='.')

# Load CSV
CSV_PATH = os.path.join(os.path.dirname(__file__), 'consolidated_data_final_with_composite_boosts.csv')
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:
    print(f"⚠️ CSV not found at: {CSV_PATH}")
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
