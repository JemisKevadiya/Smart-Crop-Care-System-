"""Load and validate the fertilizer / treatment table.

data/fertilizer/fertilizer_recommendations.csv has one row per model class.
The `disease` column holds the exact class name the classifier outputs
(e.g. "Tomato___Early_blight"), which is the lookup key.
"""

import re
from pathlib import Path

import pandas as pd

from src.preprocessing import config

DATA_PATH = config.ROOT / "data" / "fertilizer" / "fertilizer_recommendations.csv"

KEY_COLUMN = "disease"
REQUIRED_COLUMNS = [
    "disease",
    "crop",
    "category",
    "fertilizer",
    "treatment",
    "eco_friendly_treatment",
    "chemical_treatment",
    "notes",
]
ADVICE_COLUMNS = ["fertilizer", "treatment", "eco_friendly_treatment", "chemical_treatment"]


class FertilizerDataError(RuntimeError):
    """The recommendation file is missing, unreadable or malformed."""


def normalize_key(name):
    """Case/punctuation-insensitive key: 'Corn_(maize)___Common_rust_' -> 'corn maize common rust'."""
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _read_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FertilizerDataError(f"Recommendation file not found: {path}")
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except pd.errors.EmptyDataError:
        raise FertilizerDataError(f"Recommendation file is empty: {path}") from None
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise FertilizerDataError(f"Recommendation file is not valid CSV ({path}): {exc}") from None
    return df


def validate_table(df, source="recommendation table"):
    """Check columns and keys; return a cleaned copy (whitespace stripped)."""
    df = df.rename(columns=lambda c: str(c).strip().lower())
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise FertilizerDataError(f"{source} is missing required column(s): {', '.join(missing)}")
    df = df[REQUIRED_COLUMNS].apply(lambda s: s.astype(str).str.strip())
    if df.empty:
        raise FertilizerDataError(f"{source} has no rows")
    if (df[KEY_COLUMN] == "").any():
        rows = [i + 2 for i in df.index[df[KEY_COLUMN] == ""]]  # +2: header + 1-based
        raise FertilizerDataError(f"{source} has rows without a disease name (lines {rows})")
    keys = df[KEY_COLUMN].map(normalize_key)
    dupes = sorted(df.loc[keys.duplicated(keep=False), KEY_COLUMN])
    if dupes:
        raise FertilizerDataError(f"{source} has duplicate disease entries: {dupes}")
    return df.reset_index(drop=True)


_cache = {}


def load_recommendations(path=DATA_PATH):
    """Load and validate the table. Cached per file until the file changes."""
    path = Path(path)
    try:
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)
    except OSError:
        raise FertilizerDataError(f"Recommendation file not found: {path}") from None
    cached = _cache.get(path)
    if cached and cached[0] == stamp:
        return cached[1]
    table = validate_table(_read_csv(path), source=path.name)
    _cache[path] = (stamp, table)
    return table


def coverage(class_names, path=DATA_PATH):
    """Return (missing, extra): class names without a row, and rows without a class."""
    keys = set(load_recommendations(path)[KEY_COLUMN])
    names = set(class_names)
    return sorted(names - keys), sorted(keys - names)
