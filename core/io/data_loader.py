import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
import csv

def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)

def get_any(row: pd.Series, keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in row.index and pd.notna(row[k]):
            return float(row[k])
    return None

def save_measurements_csv(path: Path, columns: List[str], measurements: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in measurements:
            w.writerow(row)
