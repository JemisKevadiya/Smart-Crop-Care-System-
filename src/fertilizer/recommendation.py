"""Fertilizer and treatment advice for a predicted disease.

    rec = get_recommendation("Tomato___Early_blight")
    rec.status  # "found" | "unknown_disease" | "missing_recommendation" | "data_error"

get_recommendation never raises for bad input or a bad data file; it returns a
Recommendation whose `status` and `message` explain what happened, so the UI
can always show something sensible. Use load_recommendations() directly if
you want the data errors raised instead.

Accepted names: the model's class name ("Tomato___Early_blight"), a readable
form ("Tomato Early blight", "tomato - early blight"), or a Prediction object
from src.disease_detection. A disease name without its crop ("Early blight")
is accepted only when it matches exactly one crop.
"""

from dataclasses import asdict, dataclass, field

from .fertilizer_data import (
    ADVICE_COLUMNS,
    DATA_PATH,
    KEY_COLUMN,
    FertilizerDataError,
    load_recommendations,
    normalize_key,
)

NOT_AVAILABLE = "Not available."
DISCLAIMER = (
    "General guidance only. Product availability, registration, application rates and "
    "pre-harvest intervals differ by country and crop: confirm with a local agricultural "
    "extension officer and always follow the product label.")


@dataclass
class Recommendation:
    status: str
    query: str
    disease: str = ""
    crop: str = ""
    category: str = ""
    fertilizer: str = NOT_AVAILABLE
    treatment: str = NOT_AVAILABLE
    eco_friendly_treatment: str = NOT_AVAILABLE
    chemical_treatment: str = NOT_AVAILABLE
    notes: str = ""
    missing_fields: list = field(default_factory=list)
    message: str = ""
    disclaimer: str = DISCLAIMER

    @property
    def found(self):
        return self.status == "found"

    def to_dict(self):
        return asdict(self)


def _query_text(disease_name):
    """Accept a string or anything with a `class_name` (e.g. a Prediction)."""
    if disease_name is None:
        return ""
    return str(getattr(disease_name, "class_name", disease_name)).strip()


def _find_row(table, query):
    key = normalize_key(query)
    keys = table[KEY_COLUMN].map(normalize_key)
    exact = table[keys == key]
    if len(exact) == 1:
        return exact.iloc[0], None
    # Disease name without crop, e.g. "Late blight": accept only if unambiguous.
    suffix = table[keys.str.endswith(" " + key)] if key else table.iloc[0:0]
    if len(suffix) == 1:
        return suffix.iloc[0], None
    if len(suffix) > 1:
        return None, sorted(suffix[KEY_COLUMN])
    return None, None


def get_recommendation(disease_name, data_path=DATA_PATH):
    """Look up fertilizer and treatment advice for a disease/class name."""
    query = _query_text(disease_name)
    if not query:
        return Recommendation(status="unknown_disease", query=query,
                              message="No disease name was given.")
    try:
        table = load_recommendations(data_path)
    except FertilizerDataError as exc:
        return Recommendation(status="data_error", query=query,
                              message=f"Recommendations are unavailable: {exc}")

    row, ambiguous = _find_row(table, query)
    if row is None:
        if ambiguous:
            message = (f"'{query}' matches several crops ({', '.join(ambiguous)}); "
                       f"include the crop name.")
        else:
            message = f"No recommendation is available for '{query}'; it is not a known disease."
        return Recommendation(status="unknown_disease", query=query, message=message)

    advice = {c: row[c] for c in ADVICE_COLUMNS}
    missing = [c for c, v in advice.items() if not v]
    rec = Recommendation(
        status="found", query=query, disease=row["disease"], crop=row["crop"],
        category=row["category"], notes=row["notes"], missing_fields=missing,
        **{c: (v or NOT_AVAILABLE) for c, v in advice.items()})

    if len(missing) == len(ADVICE_COLUMNS):
        rec.status = "missing_recommendation"
        rec.message = (f"'{row['disease']}' is a known disease, but no fertilizer or treatment "
                       f"advice has been recorded for it yet.")
    elif missing:
        readable = ", ".join(c.replace("_", " ") for c in missing)
        rec.message = f"Some advice is not recorded yet: {readable}."
    elif row["category"].lower() == "healthy":
        rec.message = f"No disease detected on this {row['crop']} leaf; general care advice below."
    else:
        condition = row["disease"].partition("___")[2].replace("_", " ").strip()
        rec.message = f"Recommendation for {row['crop']}: {condition}."
    return rec
