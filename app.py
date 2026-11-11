"""Flask demo: single predictions, CSV batch screening, dashboard and local history."""
from pathlib import Path
from datetime import datetime, timezone
import csv, io, json, math, sqlite3
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, Response

BASE=Path(__file__).resolve().parent
ARTIFACT=BASE/'models'/'credit_approval_pipeline.joblib'
METRICS=BASE/'models'/'model_metrics.json'
DB=BASE/'data'/'prediction_history.sqlite3'
DB.parent.mkdir(parents=True, exist_ok=True)
app=Flask(__name__); app.config['MAX_CONTENT_LENGTH']=2*1024*1024
FEATURES=['Debt','BankCustomer','YearsEmployed','PriorDefault','Employed','CreditScore','Income','Industry']
NUMERIC=['Debt','YearsEmployed','CreditScore','Income']; FLAGS=['BankCustomer','PriorDefault','Employed']

def db():
    con=sqlite3.connect(DB); con.row_factory=sqlite3.Row
    con.execute('''CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, label TEXT NOT NULL, probability REAL NOT NULL, model TEXT NOT NULL)''')
    con.commit(); return con

def get_artifact():
    if not ARTIFACT.exists(): raise RuntimeError("Model not found. Run 'python train_model.py' first.")
    return joblib.load(ARTIFACT)

def validate(data):
    if not isinstance(data,dict): raise ValueError('Provide applicant details as a JSON object.')
    missing=[x for x in FEATURES if x not in data]
    if missing: raise ValueError('Missing fields: '+', '.join(missing))
    row={}
    for key in NUMERIC:
        try: v=float(data[key])
        except (TypeError,ValueError): raise ValueError(f'{key} must be numeric.')
        if not math.isfinite(v) or v<0: raise ValueError(f'{key} must be finite and non-negative.')
        row[key]=v
    for key in FLAGS:
        try: v=int(data[key])
        except (TypeError,ValueError): raise ValueError(f'{key} must be 0 or 1.')
        if v not in (0,1): raise ValueError(f'{key} must be 0 or 1.')
        row[key]=v
    row['Industry']=str(data['Industry']).strip()
    if not row['Industry']: raise ValueError('Industry is required.')
    return pd.DataFrame([row],columns=FEATURES)

def predict_frame(frame):
    art=get_artifact(); p=art['pipeline'].predict_proba(frame)[:,1]
    return [{'approved':bool(x>=float(art.get('threshold',.5))),'label':'Approved' if x>=float(art.get('threshold',.5)) else 'Rejected','approval_probability':round(float(x)*100,1),'model':art.get('model_name','Unknown')} for x in p]

def log_prediction(result):
    con=db(); con.execute('INSERT INTO predictions(created_at,label,probability,model) VALUES(?,?,?,?)',(datetime.now(timezone.utc).isoformat(timespec='seconds'),result['label'],result['approval_probability'],result['model'])); con.commit(); con.close()

@app.get('/')
def home(): return render_template('index.html')
@app.get('/dashboard')
def dashboard(): return render_template('dashboard.html')
@app.get('/health')
def health(): return jsonify(status='ok',model_ready=ARTIFACT.exists())
@app.get('/api/metrics')
def metrics():
    if not METRICS.exists(): return jsonify({'error':'Metrics unavailable. Run train_model.py.'}),404
    return jsonify(json.loads(METRICS.read_text(encoding='utf-8')))
@app.get('/api/history')
def history():
    con=db(); rows=con.execute('SELECT id,created_at,label,probability,model FROM predictions ORDER BY id DESC LIMIT 100').fetchall(); con.close()
    return jsonify([dict(r) for r in rows])
@app.post('/predict')
def predict():
    try:
        frame=validate(request.get_json(silent=True)); result=predict_frame(frame)[0]
        result['disclaimer']='Educational estimate only. Not a lending decision.'; log_prediction(result); return jsonify(result)
    except (ValueError,RuntimeError) as e: return jsonify(error=str(e)),400
    except Exception: app.logger.exception('Prediction failed'); return jsonify(error='Prediction failed; check server logs.'),500
@app.post('/batch-predict')
def batch_predict():
    try:
        upload=request.files.get('file')
        if not upload or not upload.filename.lower().endswith('.csv'): raise ValueError('Upload a CSV file.')
        df=pd.read_csv(upload).replace('?',pd.NA)
        missing=[c for c in FEATURES if c not in df.columns]
        if missing: raise ValueError('CSV missing columns: '+', '.join(missing))
        if len(df)>1000: raise ValueError('Maximum batch size is 1,000 rows.')
        clean=[]
        for _,r in df[FEATURES].iterrows(): clean.append(validate(r.to_dict()).iloc[0].to_dict())
        frame=pd.DataFrame(clean,columns=FEATURES); results=predict_frame(frame)
        output=df.copy(); output['Prediction']=[x['label'] for x in results]; output['Approval_Probability_Percent']=[x['approval_probability'] for x in results]
        buf=io.StringIO(); output.to_csv(buf,index=False)
        return Response(buf.getvalue(),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=credit_predictions.csv'})
    except (ValueError,RuntimeError) as e: return jsonify(error=str(e)),400
    except Exception: app.logger.exception('Batch prediction failed'); return jsonify(error='Batch prediction failed; check CSV format.'),500
if __name__=='__main__':
    db()
    app.run(host='0.0.0.0', port=int(__import__('os').environ.get('PORT', 5000)), debug=False)
