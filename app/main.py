from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. Load the trained model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model.joblib"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 2. Define the input schema
# --------------------------------------------------

class CustomerFeatures(BaseModel):

    gender: str

    SeniorCitizen: int = Field(ge=0, le=1)

    Partner: str
    Dependents: str

    tenure: int = Field(ge=0)

    PhoneService: str
    MultipleLines: str
    InternetService: str

    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str

    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    MonthlyCharges: float = Field(ge=0)

    # Missing TotalCharges values are allowed.
    TotalCharges: float | None = Field(
        default=None,
        ge=0
    )


# --------------------------------------------------
# 3. Create the FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Predict customer churn using a trained ML model.",
    version="1.0.0"
)


# --------------------------------------------------
# 4. Define the expected feature order
# --------------------------------------------------

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]


# --------------------------------------------------
# 5. Root endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Telco Customer Churn Prediction API",
        "status": "running"
    }


# --------------------------------------------------
# 6. Health endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


# --------------------------------------------------
# 7. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_churn(customer: CustomerFeatures):

    # Convert the validated request into a dictionary.
    customer_data = customer.model_dump()

    # Create a one-row DataFrame.
    input_df = pd.DataFrame(
        [customer_data],
        columns=FEATURE_COLUMNS
    )

    # Ensure TotalCharges is numeric.
    # A missing value becomes NaN and can be handled
    # by the fitted preprocessing pipeline.
    input_df["TotalCharges"] = pd.to_numeric(
        input_df["TotalCharges"],
        errors="coerce"
    )

    # Generate the predicted class.
    prediction = str(model.predict(input_df)[0])

    # Find the probability column associated with Yes.
    classifier = model.named_steps["classifier"]

    classes = list(classifier.classes_)

    yes_index = classes.index("Yes")

    # Predict the probability of churn.
    probabilities = model.predict_proba(input_df)

    churn_probability = float(
        probabilities[0][yes_index]
    )

    return {
        "prediction": prediction,
        "churn_probability": round(
            churn_probability,
            4
        )
    }