"""Optional local persisted bundle for the separately deployed API."""
from pathlib import Path
import joblib

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "models" / "bundle.joblib"

def save_bundle(model, background, path=DEFAULT_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "background": background}, path)
    return path

def load_bundle(path=DEFAULT_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No trained model found at {path}. Train and export a model in Streamlit first.")
    return joblib.load(path)
