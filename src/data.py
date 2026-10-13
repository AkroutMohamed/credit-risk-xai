
from __future__ import annotations
import pandas as pd

TARGET = "default"
UCI_ID = 350


def normalize(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    if target is None:
        candidates = ["default_payment_next_month", "default.payment.next.month", "Y", "default"]
        target = next((x for x in candidates if x in df.columns), None)
    if target is None or target not in df.columns:
        raise ValueError("Target column not found; choose a binary target.")
    df = df.rename(columns={target: TARGET})
    df = df.drop(columns=[c for c in ["ID", "id"] if c in df.columns])
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    vals = sorted(df[TARGET].unique().tolist(), key=str)
    if len(vals) != 2:
        raise ValueError("Binary classification requires exactly two target values.")
    if set(vals) == {0, 1}:
        df[TARGET] = df[TARGET].astype(int)
    else:
        
        mapping = {vals[0]: 0, vals[1]: 1}
        df[TARGET] = df[TARGET].map(mapping).astype(int)
    return df


def load_uci() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=UCI_ID)
    x, y = dataset.data.features.copy(), dataset.data.targets.copy()
    target = y.columns[0]
    return normalize(pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1), target)


def audit(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                         "missing_pct": (100 * df.isna().mean()).round(2),
                         "unique": df.nunique(dropna=True)})
