from pathlib import Path
from src_pipeline import generate_ai4i_style_replica

ROOT = Path(__file__).resolve().parent
out = ROOT / "data" / "ai4i2020_offline_replica.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df = generate_ai4i_style_replica()
df.to_csv(out, index=False)
print(f"Saved {len(df):,} rows to {out}")
print(df[["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]].sum())
