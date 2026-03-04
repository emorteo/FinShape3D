[README.md](https://github.com/user-attachments/files/25753492/README.md)
# FinShape2D R-GUI

A lightweight, **Tkinter-based** GUI to measure bottlenose dolphin dorsal fin morphometry from photographs using a landmark workflow (B → A → C30 → C20 → C10 → C5 → D), Bézier curve fitting, Weller-style ratios, and 2D contour export.

> **Status:** active development  
> **Current script:** `FinShape2D_R-GUI_2.0.14.py`

---

## Features

- **Image preparation**: flip, rotate, crop (optional) before morphometry
- **Guided landmark workflow** with on-screen instructions and a zoom window
- **Interactive Bézier control-point fitting** to match fin contour
- **Batch measurement**: multiple images → one CSV file (one row per image)
- **2D contour visualization** with axes (relative units) and export:
  - **SVG** (always)
  - **PNG/JPG/BMP** (requires Pillow)

---

## Requirements

- **Python 3.12+**
- Standard library only for core GUI:
  - `tkinter`, `math`, `csv`, `pathlib`, `dataclasses`
- **Optional (recommended): Pillow**
  - Enables JPEG input, transforms, zoom rendering, and raster contour export

Install Pillow (recommended):

```bash
python -m pip install --upgrade pip
python -m pip install pillow
```

---

## Quick Start

### 1) Get the code

Place the script in a folder, e.g.:

```
FinShape2D/
└─ FinShape2D_R-GUI_2.0.14.py
```

### 2) Run

```bash
python FinShape2D_R-GUI_2.0.14.py
```

On macOS, if `python` points elsewhere, use:

```bash
python3 FinShape2D_R-GUI_2.0.14.py
```

---

## Usage Guide

### Workflow overview

The GUI is designed to be used **in sequence**. Buttons activate as you progress.

1. **Load Image**
2. **Edit image** (optional)
   - **Flip**, **Rotate**, **Crop**
   - Draw a crop rectangle → **Apply Crop** becomes active
3. **Set Landmarks**
   - Click landmarks in this exact order:
     - **B → A → C30 → C20 → C10 → C5 → D**
   - **Tip:** Press **Set Landmarks** again anytime to clear points and restart.
4. **Fit Bézier**
   - Drag orange control points to match the fin contour.
5. **Save Measurement (CSV)**
   - Writes a new row to the session CSV.
   - Locks editing for the current image.
6. **Show 2D**
   - Displays the normalized contour (relative units) and allows contour export.
7. **Measure new image** or **Exit (Save CSV)**

### Output: CSV columns

Each measured image becomes **one row** in the CSV. Columns include:

- `image_name`
- `AB_px` (distance A–B in pixels)
- `area_px2` (contour polygon area in pixel²)
- ratios such as `C5B_over_AB`, `C10B_over_AB`, `C20B_over_AB`, `C30B_over_AB`, `DB_over_AB`
- landmark coordinates relative to **B as (0,0)**:
  - `A_dx`, `A_dy`, `C5_dx`, `C5_dy`, …, `D_dx`, `D_dy`

### Contour export formats

From the 2D contour window:

- **SVG**: always available
- **PNG/JPG/BMP**: available when **Pillow** is installed

---

## Troubleshooting

### “This image format needs Pillow”
Install Pillow:

```bash
python -m pip install pillow
```

### Tkinter not found (rare on macOS)
Use the python.org installer for Python 3.12+ (includes Tk), or ensure your environment includes Tk support.

---

## Project Structure (minimal)

```
.
├─ FinShape2D_R-GUI_2.0.14.py
└─ README.md
```

---

## Contributing

Contributions are welcome—especially improvements to robustness, usability, and export options.

### How to contribute

1. **Fork** the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-change
   ```
3. Make changes with clear, small commits
4. Run the script and test:
   - load/flip/rotate/crop
   - full landmark workflow
   - Bézier fitting
   - CSV save
   - 2D display + export (SVG and at least one raster format if Pillow installed)
5. Open a **Pull Request** with:
   - what changed and why
   - screenshots for GUI changes when helpful
   - any known limitations

### Coding conventions

- Keep dependencies minimal (standard library preferred)
- Maintain cross-platform behavior (macOS/Windows/Linux)
- Prefer clarity over cleverness for geometry steps
- Add comments for any non-obvious math/transform logic

---

## License

Add your preferred license (e.g., MIT, BSD-3, GPL) in a `LICENSE` file. Until then, treat as **all rights reserved** by default.

---

## Acknowledgements

- Bézier curve methods and morphometric ratio conventions inspired by fin-shape workflow practices (including Weller-style ratios).
