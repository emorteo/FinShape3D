# FinShape3D

FinShape3D is a modular Python library and GUI toolkit for reconstructing dorsal-fin outlines and producing 3D wireframe visualisations from simple morphometric ratio tables. The repository contains the 2D GUI workflow and the core reconstruction library used to compute Bézier control points and 3D wireframes.

Note: the standalone `finshape3d` CLI entrypoint may not be present in this checkout; use the GUI (`finshape2d.py`) or call the library functions directly (see "Programmatic usage" below).

## Highlights
- Modular `core/` package: `math`, `io`, `reconstruction`, `ui`.
- Interactive 2D landmarking GUI: `finshape2d.py`.
- Programmatic reconstruction API in `core.reconstruction.wireframe`.

## Requirements
- Python 3.8 or newer
- Recommended: create a virtual environment

Optional (improves GUI functionality):
- `Pillow` — image transforms, raster export

Install dependencies (recommended):
```bash
# from project root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you prefer to install the package in editable mode:
```bash
pip install -e .
```

## Running the GUI (Quick Start)
1. Activate your virtualenv (see above).
2. Launch the GUI:
```bash
python finshape2d.py
```
3. Workflow (GUI):
	- Load an image.
	- Set landmarks in order: B → A → C30 → C20 → C10 → C5 → D.
	- Adjust Bézier controls if needed, then click `Save Measurement (CSV)`.
	- After saving, click `Generate 3D` to produce 2D/3D outputs in `FinShape3D_outputs/` next to the image file.

## Programmatic usage
If you want to run reconstruction from a script or pipeline, use the library API:

```python
from pathlib import Path
import pandas as pd
from core.reconstruction.wireframe import build_landmarks, build_controls, compute_2d_curves

row = pd.Series(...)  # a row with r_BA,r_BC5,r_BC10,r_BC20,r_BC30 or C*B_over_AB fields
lm, u_AB = build_landmarks(row, BC30_scale=10.0)
ctrl = build_controls(lm, u_AB)
curves2d = compute_2d_curves(ctrl)
# plotting helpers are available in core.ui.plotting_3d
```

## Testing
Run the project's test suite with `pytest`:

```bash
pytest -q
```

Tests use pytest fixtures and create temporary files via `tmp_path`; they do not modify the repository tree.

## Development practices
- Use the provided `.gitignore` to avoid committing virtualenvs, build artifacts and generated outputs.
- Keep long-running or large example images outside the repo (place in a `data/` folder and add to `.gitignore`).

## Troubleshooting
- If `Generate 3D` fails with geometry errors, check the CSV row printed by the GUI error dialog — noisy ratio inputs can violate geometric constraints.
- If image open fails on Linux, ensure `xdg-open` is available or open the output manually.

## Disclaimer
This software is research-oriented and provided "as-is" for academic and development use. There is no warranty of accuracy, fitness for a particular purpose, or non-infringement. Use at your own risk. The authors and contributors are not liable for any damages arising from its use.

## Contributing
Contributions and issues are welcome. Please follow standard GitHub workflows: open an issue to discuss major changes, then submit a pull request with tests and documentation updates.

## License
MIT License.
