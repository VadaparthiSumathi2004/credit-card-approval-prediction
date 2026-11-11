# IBM Watson Machine Learning deployment guide

The local project now exports a single scikit-learn `Pipeline` in `models/credit_approval_pipeline.joblib`. The Flask app is runnable locally and exposes `/predict`, `/health`, `/api/metrics`, and `/batch-predict`.

## Before cloud deployment

1. Create an IBM Cloud account and provision an IBM Watson Machine Learning service in the region supported by your account.
2. Create or open an IBM watsonx.ai project and associate the Machine Learning service.
3. Record the project ID and create an API key. Keep credentials in IBM Cloud secrets/environment variables; never commit them to Git or put them in browser JavaScript.
4. Check the current Watson Machine Learning documentation for supported scikit-learn and Python software specifications. The local `joblib` artifact must be compatible with the runtime you select.
5. Upload `models/credit_approval_pipeline.joblib` as a model asset using the current Watson Machine Learning Python SDK or the watsonx.ai UI workflow supported by your account. Register the model, select a compatible software specification, and create a deployment.
6. Send a test scoring request using the exact eight input fields documented in `README.md`. Compare its output with local `/predict` before considering the deployment complete.
7. Store the deployment ID and credentials server-side. If the Flask app calls the cloud deployment, make that call from Python—not directly from the browser—and add authentication, request limits, logging controls, and monitoring.

## Important compatibility note

IBM service APIs, SDK versions, regions, and supported software specifications change. This repository intentionally does not hard-code credentials, deployment IDs, or an unverified SDK call. Follow the live IBM documentation for your provisioned service and pin the SDK/runtime versions after a successful test deployment. A local Flask launch is **not** proof that the model has been deployed to IBM Cloud.

## Responsible-use note

This is a classroom prototype using a small historical dataset. Do not use it to make or automate real credit decisions. A production financial workflow requires legal and compliance review, representative data, independent validation, calibration, explainability, privacy/security controls, fairness assessment, and human review.
