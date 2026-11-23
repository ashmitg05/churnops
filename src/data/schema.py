"""Single source of truth for input columns. Used by Pandera (training) and Pydantic (serving)."""

YES_NO = ["Yes", "No"]
NO_INTERNET = ["Yes", "No", "No internet service"]

CATEGORIES: dict[str, list[str]] = {
    "gender": ["Male", "Female"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": NO_INTERNET,
    "OnlineBackup": NO_INTERNET,
    "DeviceProtection": NO_INTERNET,
    "TechSupport": NO_INTERNET,
    "StreamingTV": NO_INTERNET,
    "StreamingMovies": NO_INTERNET,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
INPUT_COLUMNS = ["customerID", *CATEGORIES, *NUMERIC]
