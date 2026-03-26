import pandas as pd
from pathlib import Path
from finshape3d import run

# Create dummy csv
csv_file = Path("dummy.csv")
df = pd.DataFrame({
    "id": ["test_fin"],
    "r_BA": [1.0],
    "r_BC5": [0.85],
    "r_BC10": [0.4],
    "r_BC20": [0.4],
    "r_BC30": [1.0],
})
df.to_csv(csv_file, index=False)

outdir = Path("out")

# Run pipeline
run(
    input_path=csv_file,
    outdir=outdir,
    bc30scale=10.0,
    y_slices=5,
    z_slices=3,
    top_focus_power=1.0,
    montage=False
)

print("Pipeline finished successfully.")
