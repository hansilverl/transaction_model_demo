import os
import pickle
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

# Load metadata (e.g., vectorizer)
with open(os.path.join(MODEL_DIR, "metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)

# Models were trained directly on raw text features so no vectorizer is needed
# Older versions of the metadata contained a serialized TfidfVectorizer, but the
# current model artifacts omit it. Predictions are made by passing the document
# text directly to each CatBoost model.
label_encoders = metadata["label_encoders"]  # dict mapping field -> {label_to_int, int_to_label}

def _decode_label(encoder_dict, value):
    """Decode integer predictions back to their string labels."""
    return encoder_dict["int_to_label"].get(int(value), "")

def extract_fields_from_pdf(pdf_path):
    # Extract raw text from PDF
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    doc.close()

    # Pass the raw text directly to the models
    result = {
        "amount_before": float(models["amount_before"].predict([full_text])),
        "from_currency": _decode_label(
            label_encoders["from_currency"],
            models["from_currency"].predict([full_text])[0],
        ),
        "to_currency": _decode_label(
            label_encoders["to_currency"],
            models["to_currency"].predict([full_text])[0],
        ),
        "exchange_rate": float(models["exchange_rate"].predict([full_text])),
        "fee": float(models["fee"].predict([full_text])),
        "fee_currency": _decode_label(
            label_encoders["fee_currency"],
            models["fee_currency"].predict([full_text])[0],
        ),
        "amount_converted": float(models["amount_converted"].predict([full_text])),
        "after_fee": float(models["after_fee"].predict([full_text])),
        "raw_text": full_text[:1000] + "..." if len(full_text) > 1000 else full_text
    }

    return result
