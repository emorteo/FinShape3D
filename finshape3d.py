#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from typing import Optional, List

from core.io.data_loader import read_table
from core.reconstruction.wireframe import build_landmarks, build_controls, compute_2d_curves
from core.ui.plotting_3d import plot_2d, plot_3d_wireframe, make_montage

VERSION = "1.1.9"

def select_input_file_interactive() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        try: root.attributes("-topmost", True)
        except Exception: pass
        file_path = filedialog.askopenfilename(
            title="Select ratios table (CSV/XLS/XLSX)",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
        )
        root.destroy()
        if file_path:
            pth = Path(file_path).expanduser().resolve()
            if pth.exists(): return pth
    except Exception: pass

    while True:
        raw = input("Enter path to ratios CSV/XLS/XLSX (or press Enter to cancel): ").strip().strip('"').strip("'")
        if not raw: raise SystemExit("No input file selected. Exiting.")
        pth = Path(raw).expanduser().resolve()
        if pth.exists(): return pth
        print(f"File not found: {pth}")

def default_outdir_for_input(input_path: Path) -> Path:
    return input_path.parent / f"{input_path.stem}_FinShape3D_outputs_v{VERSION}"

def run(input_path: Path, outdir: Path, bc30scale: float, y_slices: int, z_slices: int, top_focus_power: float, montage: bool) -> None:
    df = read_table(input_path)
    outdir.mkdir(parents=True, exist_ok=True)

    outs2d: List[Path] = []
    outs3d: List[Path] = []

    for i, row in df.iterrows():
        fin_id = str(row.get("id", f"row{i}"))
        lm, u_AB = build_landmarks(row, BC30_scale=bc30scale)
        ctrl = build_controls(lm, u_AB)
        curves2d = compute_2d_curves(ctrl, n=900)

        p2d = outdir / f"FinShape3D_{VERSION}_2D_{fin_id}.png"
        p3d = outdir / f"FinShape3D_{VERSION}_3D_{fin_id}.png"
        plot_2d(curves2d, lm, ctrl, fin_id, p2d, legend_anchor_xy=(-4.0, 4.0))
        plot_3d_wireframe(curves2d, fin_id, p3d, n_y_slices=y_slices, n_z_slices=z_slices, top_focus_power=top_focus_power)

        outs2d.append(p2d)
        outs3d.append(p3d)

    if montage and outs2d:
        make_montage(outs2d, outdir / f"FinShape3D_{VERSION}_montage_2D.png", title=f"FinShape3D {VERSION} — 2D Contours")
    if montage and outs3d:
        make_montage(outs3d, outdir / f"FinShape3D_{VERSION}_montage_3D.png", title=f"FinShape3D {VERSION} — 3D Wireframes")

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Reconstruct 2D/3D dorsal fins from ratio tables.")
    ap.add_argument("--input", type=Path, default=None, help="CSV or XLSX containing ratios.")
    ap.add_argument("--outdir", type=Path, default=None, help="Output directory (default next to input).")
    ap.add_argument("--bc30scale", type=float, default=10.0, help="Set |BC30| to this value (relative units).")
    ap.add_argument("--y_slices", type=int, default=30, help="Horizontal slices for 3D wireframe.")
    ap.add_argument("--z_slices", type=int, default=15, help="Vertical slices for 3D wireframe.")
    ap.add_argument("--top_focus_power", type=float, default=3.0, help="Concentrate y-slices near tip.")
    ap.add_argument("--no_montage", action="store_true", help="Disable montage creation.")

    args, _unknown = ap.parse_known_args(argv)

    input_path = args.input if args.input is not None else select_input_file_interactive()
    outdir = args.outdir if args.outdir is not None else default_outdir_for_input(input_path)

    run(
        input_path=input_path,
        outdir=outdir,
        bc30scale=args.bc30scale,
        y_slices=args.y_slices,
        z_slices=args.z_slices,
        top_focus_power=args.top_focus_power,
        montage=(not args.no_montage),
    )

if __name__ == "__main__":
    main()
