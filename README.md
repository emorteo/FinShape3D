# FinShape3D

FinShape3D is a modular Python-based workflow designed to transform 2D dorsal fin photographs into high-fidelity, side-invariant 3D digital twins. By utilizing landmark-constrained Bézier representations and chordwise thickness profiles, it enables precise cross-side matching and morphological analysis for cetacean research.

## Features
- **Modular Architecture**: Clean separation of concerns with `core` modules for math, IO, and reconstruction.
- **Bézier Reconstruction**: Standardized 2D outline generation from discrete landmarks.
- **3D Wireframe Generation**: Extrudes 2D outlines into 3D volumes using biological thickness profiles.
- **Cross-Platform**: Built with Python, utilizing `numpy`, `pandas`, `matplotlib`, and `open3d`.

## Installation

Ensure you have Python 3.8+ installed.

```bash
# Clone the repository
git clone <repository-url>
cd FinShape3D

# Install dependencies
pip install .
```

## Usage

The project provides entry points for both 2D GUI workflows and 3D CLI processing:

- **2D GUI**: `python finshape2d.py`
- **3D CLI**: `python finshape3d.py`

## Project Structure

```
.
├── core/               # Core logic (math, io, reconstruction, ui)
├── docs/               # Documentation (vision, development plan, etc.)
├── tests/              # Unit and integration tests
├── finshape2d.py       # GUI entry point
├── finshape3d.py       # CLI entry point
└── pyproject.toml      # Project configuration
```

## Contributing

Contributions are welcome. Please see the `docs/` folder for development plans and architecture decisions.

## License

MIT License.
