"""Check that Smart Crop Care's configuration is complete, without revealing secrets.

    python scripts/check_config.py            # files and settings only (no network)
    python scripts/check_config.py --online   # also tests the API keys and news feeds

Prints OK / WARN / FAIL per item. API keys are reported only as "set" or "not
set" - their values are never printed. Exit code 1 if anything FAILs.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.env import ENV_FILE, get_setting, read_env_file  # noqa: E402

PLACEHOLDER_HINTS = ("your_", "_here", "changeme", "xxx")
results = []


def report(status, item, detail):
    results.append(status)
    print(f"[{status:4}] {item}: {detail}")


def is_placeholder(value):
    return any(h in value.lower() for h in PLACEHOLDER_HINTS)


def check_files():
    env_example = ROOT / ".env.example"
    report("OK" if ENV_FILE.is_file() else "WARN", ".env",
           "found" if ENV_FILE.is_file() else "missing - copy .env.example to .env")
    if env_example.is_file():
        values = read_env_file(env_example)
        real = [k for k, v in values.items() if v and not is_placeholder(v)]
        report("FAIL" if real else "OK", ".env.example",
               f"looks like real values for {real}" if real else "placeholders only")
    else:
        report("WARN", ".env.example", "missing")

    try:
        tracked = subprocess.run(["git", "ls-files", ".env"], cwd=ROOT, capture_output=True,
                                 text=True, timeout=20).stdout.strip()
        ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT,
                                 timeout=20).returncode == 0
        ok = ignored and not tracked
        report("OK" if ok else "FAIL", ".env in git",
               "ignored and not tracked" if ok else "WARNING: .env is tracked or not ignored")
    except (OSError, subprocess.SubprocessError):
        report("WARN", ".env in git", "git not available, not checked")


def check_keys():
    for name, required in (("GROQ_API_KEY", True), ("OPENWEATHER_API_KEY", False)):
        value = get_setting(name)
        if not value:
            report("FAIL" if required else "WARN", name,
                   "not set" + ("" if required else " (current weather will use Open-Meteo)"))
        elif is_placeholder(value):
            report("FAIL", name, "still the placeholder from .env.example")
        else:
            report("OK", name, f"set ({len(value)} characters; value not shown)")


def check_model_and_data():
    from src.disease_detection.model_loader import CLASS_NAMES_PATH, MODEL_PATH, load_class_names
    from src.fertilizer import FertilizerDataError, load_recommendations
    from src.fertilizer.fertilizer_data import DATA_PATH, coverage

    report("OK" if MODEL_PATH.is_file() else "FAIL", "Disease model",
           f"{MODEL_PATH.relative_to(ROOT)} ({MODEL_PATH.stat().st_size / 1e6:.0f} MB)"
           if MODEL_PATH.is_file() else f"missing: {MODEL_PATH.relative_to(ROOT)}")
    try:
        names = load_class_names()
        report("OK", "Class names", f"{len(names)} classes ({CLASS_NAMES_PATH.name})")
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "Class names", str(exc))
        names = []
    try:
        table = load_recommendations()
        report("OK", "Fertilizer/treatment table", f"{len(table)} entries ({DATA_PATH.name})")
        if names:
            missing, _extra = coverage(names)
            report("OK" if not missing else "WARN", "Advice coverage",
                   "every model class has an entry" if not missing else f"missing: {missing}")
    except FertilizerDataError as exc:
        report("FAIL", "Fertilizer/treatment table", str(exc))


def check_settings():
    from src.chatbot import llm
    from src.disease_detection.preprocessing import MAX_FILE_BYTES
    from src.news import get_feeds

    report("OK", "Chatbot models", f"main {llm.get_model_name()}, backup "
           f"{llm.get_fallback_model_name() or 'off'}")
    feeds = get_feeds()
    source = "NEWS_FEEDS in .env" if get_setting("NEWS_FEEDS") else "built-in defaults"
    report("OK", "News feeds", f"{len(feeds)} feeds ({source}); no API key needed")

    limit_mb = None
    config_file = ROOT / ".streamlit" / "config.toml"
    if config_file.is_file():
        for line in config_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("maxUploadSize"):
                limit_mb = int(line.split("=")[1].strip())
    expected = MAX_FILE_BYTES // (1024 * 1024)
    report("OK" if limit_mb == expected else "WARN", "Upload limit",
           f"Streamlit {limit_mb} MB, image check {expected} MB")


def check_online():
    from src.chatbot import AgricultureAssistant, ChatbotError
    from src.news import NewsUnavailableError, get_farmer_news
    from src.weather import get_api_key
    from src.weather.openweather import get_current_weather

    try:
        get_current_weather(21.17, 72.83, get_api_key())
        report("OK", "OpenWeather", "key accepted")
    except Exception as exc:  # noqa: BLE001 - messages never contain the key
        report("WARN", "OpenWeather", str(exc))
    try:
        news = get_farmer_news()
        report("OK", "News feeds", f"{len(news.items)} items"
               + (f"; unreachable: {list(news.errors)}" if news.errors else ""))
    except NewsUnavailableError as exc:
        report("WARN", "News feeds", str(exc))
    try:
        answer = AgricultureAssistant().reply("Reply with the single word OK.")
        report("OK", "Groq", f"answered ({len(answer)} characters)")
    except ChatbotError as exc:
        report("FAIL", "Groq", str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--online", action="store_true", help="also call the external APIs")
    args = parser.parse_args()
    check_files()
    check_keys()
    check_model_and_data()
    check_settings()
    if args.online:
        check_online()
    failed = results.count("FAIL")
    print(f"\n{len(results)} checks: {results.count('OK')} OK, {results.count('WARN')} WARN, "
          f"{failed} FAIL")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
