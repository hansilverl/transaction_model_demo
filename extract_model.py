import os
import pickle
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import fitz  # PyMuPDF

MODEL_DIR = "model_artifacts"

# Load models
models = {
    "after_fee": CatBoostRegressor().load_model(os.path.join(MODEL_DIR, "after_fee_model.cbm")),
    "amount_before": CatBoostRegressor().load_model(os.path.join(MODEL_DIR, "amount_before_model.cbm")),
    "amount_converted": CatBoostRegressor().load_model(os.path.join(MODEL_DIR, "amount_converted_model.cbm")),
    "exchange_rate": CatBoostRegressor().load_model(os.path.join(MODEL_DIR, "exchange_rate_model.cbm")),
    "fee": CatBoostRegressor().load_model(os.path.join(MODEL_DIR, "fee_model.cbm")),
    "from_currency": CatBoostClassifier().load_model(os.path.join(MODEL_DIR, "from_currency_model.cbm")),
    "to_currency": CatBoostClassifier().load_model(os.path.join(MODEL_DIR, "to_currency_model.cbm")),
    "fee_currency": CatBoostClassifier().load_model(os.path.join(MODEL_DIR, "fee_currency_model.cbm")),
}

# Load metadata (e.g., vectorizer)
with open(os.path.join(MODEL_DIR, "metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)

vectorizer: TfidfVectorizer = metadata["vectorizer"]
label_encoders = metadata["label_encoders"]  # dict

def extract_fields_from_pdf(pdf_path):
    # Extract raw text from PDF
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    doc.close()

    tfidf_features = vectorizer.transform([full_text])

    result = {
        "amount_before": float(models["amount_before"].predict(tfidf_features)),
        "from_currency": label_encoders["from_currency"].inverse_transform(
            models["from_currency"].predict(tfidf_features).astype(int)
        )[0],
        "to_currency": label_encoders["to_currency"].inverse_transform(
            models["to_currency"].predict(tfidf_features).astype(int)
        )[0],
        "exchange_rate": float(models["exchange_rate"].predict(tfidf_features)),
        "fee": float(models["fee"].predict(tfidf_features)),
        "fee_currency": label_encoders["fee_currency"].inverse_transform(
            models["fee_currency"].predict(tfidf_features).astype(int)
        )[0],
        "amount_converted": float(models["amount_converted"].predict(tfidf_features)),
        "after_fee": float(models["after_fee"].predict(tfidf_features)),
        "raw_text": full_text[:1000] + "..." if len(full_text) > 1000 else full_text
    }

    return result
