# Credit Card Approval Prediction — enhanced academic demo

## Added features
- Compares Logistic Regression, Decision Tree, Random Forest and XGBoost (if installed).
- A reusable scikit-learn preprocessing/model pipeline with missing-value imputation and one-hot encoding.
- Evaluation metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix and classification report.
- Dashboard at `/dashboard` with model comparison, confusion matrix, ROC curve and permutation feature-importance charts.
- Single-applicant prediction at `/` and CSV batch prediction (up to 1,000 rows).
- Local SQLite activity history stores only timestamp, output label, probability and model—not applicant input fields.
- Health endpoint at `/health`; metrics endpoint at `/api/metrics`.
- `Procfile` and `Dockerfile` for container/host deployment preparation.
- IBM Watson Machine Learning deployment checklist in `IBM_WATSON_DEPLOYMENT.md`.

## Dataset and target labels
The supplied dataset's target is interpreted as `Approved=1` and `Approved=0` as rejected. Confirm the meaning against the source dataset documentation before relying on this mapping. The model excludes direct demographic columns, but proxy discrimination can remain.

## Run locally
Use Python 3.10 or 3.11 for broad package compatibility.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000` for predictions and `http://127.0.0.1:5000/dashboard` for charts, metrics and recent activity. Training creates the model under `models/` and charts under `static/generated/`.

## Batch CSV
Upload a CSV containing these exact columns: `Debt`, `BankCustomer`, `YearsEmployed`, `PriorDefault`, `Employed`, `CreditScore`, `Income`, `Industry`. Maximum 1,000 rows. The downloaded file contains two additional columns: `Prediction` and `Approval_Probability_Percent`.

## API
- `GET /health`
- `GET /api/metrics`
- `GET /api/history`
- `POST /predict` with JSON fields listed above
- `POST /batch-predict` with multipart form field `file` containing a CSV

## Cloud deployment
See `IBM_WATSON_DEPLOYMENT.md`. IBM credentials and deployment IDs are intentionally not embedded in this source. Follow the current IBM documentation for supported runtime/software specifications.

## Safety and limitations
Educational prototype only; not a bank-grade underwriting system and must not be used to approve/reject real credit applications. Dataset size and single hold-out split limit reliability. Probabilities are not calibrated or guaranteed to represent real-world risk. Dashboard/history are local-demo features with no authentication; do not expose them publicly without authentication, authorization, secure storage, and privacy review. Before consequential use, require representative data, external validation, calibration, explainability, fairness testing, drift monitoring, legal/compliance review, and meaningful human oversight.


## Deploy publicly on Render

This repository includes `render.yaml` for a Render Web Service. Push the complete project folder to a GitHub repository, then in Render choose **New + → Blueprint** and connect that repository. Render will read `render.yaml`, install dependencies, and start Gunicorn. After deployment, open the `onrender.com` URL shown in the Render dashboard. Check `/health` to confirm the service is running.

Alternative manual setup: create a **Web Service**, choose Python, set build command to `pip install -r requirements.txt`, start command to `gunicorn --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT app:app`, and set `PYTHON_VERSION=3.11.11`.

Before public sharing, add authentication and authorization to `/dashboard`, `/api/history`, `/api/metrics`, and batch prediction; configure persistent storage if prediction history must survive redeploys. Do not use this academic demo for real lending decisions.
