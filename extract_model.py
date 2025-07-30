import os
import pickle
import json
from catboost import CatBoostRegressor, CatBoostClassifier
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

# Load metadata that describes where the encoder pickle files live
with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
    metadata = json.load(f)

# Load each label encoder referenced in the metadata
label_encoders = {}
for field, enc_file in metadata.get("encoders_files", {}).items():
    with open(os.path.join(MODEL_DIR, enc_file), "rb") as ef:
        label_encoders[field] = pickle.load(ef)

# Load TfidfVectorizer used to transform the document text before prediction
with open(os.path.join(MODEL_DIR, metadata["tfidf_file"]), "rb") as f:
    vectorizer = pickle.load(f)

def _decode_label(encoder, value):
    """Decode integer predictions back to their string labels."""
    # Predictions may be floats; round and cast to int for inverse_transform
    idx = int(round(float(value)))
    return encoder.inverse_transform([idx])[0]

def extract_fields_from_pdf(pdf_path):
    # Extract raw text from PDF
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    doc.close()

    # Vectorize the text for the CatBoost models
    features = vectorizer.transform([full_text])

    result = {
        "amount_before": float(models["amount_before"].predict(features)),
        "from_currency": _decode_label(
            label_encoders["from_currency"],
            models["from_currency"].predict(features)[0],
        ),
        "to_currency": _decode_label(
            label_encoders["to_currency"],
            models["to_currency"].predict(features)[0],
        ),
        "exchange_rate": float(models["exchange_rate"].predict(features)),
        "fee": float(models["fee"].predict(features)),
        "fee_currency": _decode_label(
            label_encoders["fee_currency"],
            models["fee_currency"].predict(features)[0],
        ),
        "amount_converted": float(models["amount_converted"].predict(features)),
        "after_fee": float(models["after_fee"].predict(features)),
        "raw_text": full_text[:1000] + "..." if len(full_text) > 1000 else full_text
    }

    return result
