import pandas as pd
import pytest

from src.fertilizer import FertilizerDataError, get_recommendation, load_recommendations
from src.fertilizer.fertilizer_data import REQUIRED_COLUMNS, coverage
from src.preprocessing import splits


def _write(tmp_path, rows, columns=REQUIRED_COLUMNS, name="recs.csv"):
    path = tmp_path / name
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def _row(disease, crop="Tomato", category="Fungal", fertilizer="Balanced NPK.",
         treatment="Remove leaves.", eco="Copper.", chem="Mancozeb.", notes="Note."):
    return [disease, crop, category, fertilizer, treatment, eco, chem, notes]


# --- the real data file -------------------------------------------------------

def test_real_file_covers_every_model_class():
    missing, extra = coverage(splits.load_class_names())
    assert missing == [] and extra == []


def test_every_model_class_has_complete_advice():
    for name in splits.load_class_names():
        rec = get_recommendation(name)
        assert rec.status == "found", name
        assert rec.missing_fields == [], name


@pytest.mark.parametrize(
    "query",
    ["Tomato___Early_blight", "tomato early blight", "  Tomato - Early Blight  "],
)
def test_known_disease_by_class_or_readable_name(query):
    rec = get_recommendation(query)
    assert rec.found and rec.disease == "Tomato___Early_blight"
    assert rec.crop == "Tomato" and rec.category == "Fungal"
    assert "chlorothalonil" in rec.chemical_treatment.lower()
    assert rec.disclaimer


def test_names_with_punctuation_resolve():
    assert get_recommendation("Pepper, bell - Bacterial spot").disease == "Pepper,_bell___Bacterial_spot"
    assert get_recommendation("corn (maize) common rust").disease == "Corn_(maize)___Common_rust_"


def test_healthy_class():
    rec = get_recommendation("Apple___healthy")
    assert rec.found and rec.category == "Healthy"
    assert "No disease detected" in rec.message


def test_disease_without_crop_unique_and_ambiguous():
    assert get_recommendation("Leaf Mold").disease == "Tomato___Leaf_Mold"
    rec = get_recommendation("Late blight")  # potato and tomato
    assert rec.status == "unknown_disease" and "several crops" in rec.message


def test_accepts_prediction_like_object():
    class FakePrediction:
        class_name = "Grape___Black_rot"

    assert get_recommendation(FakePrediction()).disease == "Grape___Black_rot"


# --- unknown / missing ---------------------------------------------------------

@pytest.mark.parametrize("query", ["Banana___Panama_disease", "xyz", "", "   ", None])
def test_unknown_disease(query):
    rec = get_recommendation(query)
    assert rec.status == "unknown_disease" and not rec.found
    assert rec.message
    assert rec.fertilizer == "Not available."


def test_missing_recommendation(tmp_path):
    path = _write(tmp_path, [
        _row("Tomato___Early_blight"),
        _row("Tomato___Target_Spot", fertilizer="", treatment="", eco="", chem=""),
        _row("Tomato___Leaf_Mold", chem=""),
    ])
    empty = get_recommendation("Tomato___Target_Spot", data_path=path)
    assert empty.status == "missing_recommendation"
    assert empty.treatment == "Not available." and "no fertilizer or treatment" in empty.message

    partial = get_recommendation("Tomato___Leaf_Mold", data_path=path)
    assert partial.status == "found" and partial.missing_fields == ["chemical_treatment"]
    assert partial.chemical_treatment == "Not available."
    assert "chemical treatment" in partial.message


# --- invalid data files --------------------------------------------------------

def test_missing_file(tmp_path):
    with pytest.raises(FertilizerDataError, match="not found"):
        load_recommendations(tmp_path / "nope.csv")
    rec = get_recommendation("Tomato___Early_blight", data_path=tmp_path / "nope.csv")
    assert rec.status == "data_error" and "not found" in rec.message


def test_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(FertilizerDataError, match="empty"):
        load_recommendations(path)


def test_malformed_csv(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(",".join(REQUIRED_COLUMNS) + '\n"unterminated,quote\n')
    with pytest.raises(FertilizerDataError, match="not valid CSV"):
        load_recommendations(path)
    assert get_recommendation("x", data_path=path).status == "data_error"


def test_binary_file(tmp_path):
    path = tmp_path / "binary.csv"
    path.write_bytes(bytes(range(256)) * 4)
    with pytest.raises(FertilizerDataError):
        load_recommendations(path)


def test_missing_columns(tmp_path):
    cols = [c for c in REQUIRED_COLUMNS if c not in ("fertilizer", "notes")]
    path = _write(tmp_path, [["Tomato___Early_blight", "Tomato", "Fungal", "a", "b", "c"]], cols)
    with pytest.raises(FertilizerDataError, match="missing required column.*fertilizer.*notes"):
        load_recommendations(path)
    rec = get_recommendation("Tomato___Early_blight", data_path=path)
    assert rec.status == "data_error" and "fertilizer" in rec.message


def test_duplicate_and_blank_keys(tmp_path):
    dupes = _write(tmp_path, [_row("Tomato___Early_blight"), _row("tomato early blight")],
                   name="dupes.csv")
    with pytest.raises(FertilizerDataError, match="duplicate"):
        load_recommendations(dupes)
    blank = _write(tmp_path, [_row("Tomato___Early_blight"), _row("")], name="blank.csv")
    with pytest.raises(FertilizerDataError, match="without a disease name"):
        load_recommendations(blank)


def test_column_names_are_case_and_space_insensitive(tmp_path):
    cols = [f" {c.upper()} " for c in REQUIRED_COLUMNS]
    path = _write(tmp_path, [_row("Tomato___Early_blight")], cols)
    assert get_recommendation("Tomato___Early_blight", data_path=path).found


def test_cache_reloads_when_file_changes(tmp_path):
    path = _write(tmp_path, [_row("Tomato___Early_blight", fertilizer="Old advice.")])
    assert get_recommendation("Tomato___Early_blight", data_path=path).fertilizer == "Old advice."
    _write(tmp_path, [_row("Tomato___Early_blight", fertilizer="New advice, longer text.")])
    assert get_recommendation("Tomato___Early_blight", data_path=path).fertilizer == \
        "New advice, longer text."
