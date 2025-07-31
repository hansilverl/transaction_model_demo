import os
import pickle
import json
import re
from typing import Optional
from datetime import datetime
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


def _parse_number(num_str: str) -> Optional[float]:
    """Convert a numeric string with optional commas into a float."""
    try:
        return float(num_str.replace(",", ""))
    except ValueError:
        return None


def _parse_date(date_str: str) -> Optional[str]:
    """Parse a date like 8/2/2023 into ISO format.

    The PDFs in the dataset are inconsistent about whether the first
    number represents the day or the month. We try interpreting the
    value as day-first first, then fall back to month-first."""
    formats = ["%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _heuristic_extract(text: str) -> dict:
    """Best-effort extraction using regex patterns."""
    result: dict = {}

    # Wire or transfer amounts
    amount_matches = list(
        re.finditer(r"Wire Amount\s*\((?P<cur>[A-Za-z]{3})\):\s*(?P<num>[\d,]+(?:\.\d+)?)",
                    text, re.IGNORECASE)
    )
    if not amount_matches:
        # Alternate Western Union style
        alt = re.search(r"([\d,.]+)\s*(?P<cur>[A-Za-z]{3})\s*Transfer amount", text, re.IGNORECASE)
        if alt:
            amount_matches.append(alt)
        alt_conv = re.search(r"([\d,.]+)\s*(?P<cur>[A-Za-z]{3})\s*Total to receiver", text, re.IGNORECASE)
        if alt_conv:
            amount_matches.append(alt_conv)

    if amount_matches:
        first = amount_matches[0]
        num = first.group("num") if "num" in first.groupdict() else first.group(1)
        cur = first.group("cur").upper()
        result["amount_before"] = _parse_number(num)
        result["from_currency"] = cur
        if len(amount_matches) > 1:
            second = amount_matches[1]
            num2 = second.group("num") if "num" in second.groupdict() else second.group(1)
            cur2 = second.group("cur").upper()
            result["amount_converted"] = _parse_number(num2)
            result["to_currency"] = cur2

    # Exchange rate
    m = re.search(r"([\d,.]+)\s*[A-Za-z]{3}\s*=\s*([\d,.]+)\s*[A-Za-z]{3}", text)
    if m:
        result["exchange_rate"] = _parse_number(m.group(2))
    else:
        m = re.search(r"Exchange Rate[:\s]*([\d.]+)", text, re.IGNORECASE)
        if m:
            result["exchange_rate"] = _parse_number(m.group(1))

    # Wire fee
    m = re.search(r"Wire Fee\s*\((?P<cur>[A-Za-z]{3})\):\s*(?P<num>[\d,]+(?:\.\d+)?)",
                  text, re.IGNORECASE)
    if not m:
        m = re.search(r"([\d,.]+)\s*(?P<cur>[A-Za-z]{3})\s*Transfer fee", text, re.IGNORECASE)
    if m:
        num = m.group("num") if "num" in m.groupdict() else m.group(1)
        result["fee"] = _parse_number(num)
        result["fee_currency"] = m.group("cur").upper()

    # Date
    m = re.search(r"Wire Date[:\s]*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})", text, re.IGNORECASE)
    if m:
        parsed = _parse_date(m.group(1))
        if parsed:
            result["date"] = parsed

    return result

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

    # Use regex heuristics to override predictions when obvious values exist
    heuristics = _heuristic_extract(full_text)
    result.update({k: v for k, v in heuristics.items() if v is not None})

    return result
