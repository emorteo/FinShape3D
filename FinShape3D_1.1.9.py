#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinShape3D_1.1.9

Fixes vs 1.1.8
--------------
1) Landmark label placement robustness:
   Some fins (e.g., FIN_001_TB, FIN_004_BB) had C5 and/or C20 labels displaced because the
   previous algorithm used a simple centroid ray to decide the "outside" direction. For
   non-convex polygons, the vertex-mean centroid may lie outside the polygon or yield a
   direction that is not locally outward at boundary points.

   This version places LANDMARK labels using a *local outward normal* computed from the
   contour tangent near the landmark, verified by point-in-polygon tests. This makes the
   "outside" placement consistent and near the landmark.

   Control-point labels are still forced INSIDE the contour, but now use a robust interior
   point (guaranteed inside polygon) and point-in-polygon stepping.

2) Still PyCharm / Python 3.12 safe:
   - matplotlib backend forced to Agg.
   - argparse uses parse_known_args() and --input/--outdir are optional.
   - If --input missing, program FIRST asks user to choose CSV/XLS/XLSX (GUI picker if available).

Dependencies (Python 3.12 / macOS / PyCharm):
  pip install numpy pandas matplotlib openpyxl
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


VERSION = "1.1.9"

# Global formatting multipliers
FONT_SCALE = 3.0
LEGEND_SCALE = 2.0

# Landmark/control label scaling (relative to 1.1.7 rule set)
LM_LABEL_MULT = 0.5
CTRL_LABEL_MULT = 0.5

FS_POINT = 10
FS_CTRL = 9
FS_AXLAB = 10
FS_LEGEND = 9

# Headers not scaled
FS_TITLE_2D = 12
FS_TITLE_3D = 11
FS_MONT_TILE = 9
FS_MONT_TITLE = 14


# -----------------------------
# FIRST STEP: ask for input file
# -----------------------------
def select_input_file_interactive() -> Path:
    """Ask the user to select a ratios table (CSV/XLS/XLSX)."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

        file_path = filedialog.askopenfilename(
            title="Select ratios table (CSV/XLS/XLSX)",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if file_path:
            pth = Path(file_path).expanduser().resolve()
            if pth.exists():
                return pth
    except Exception:
        pass

    while True:
        raw = input("Enter path to ratios CSV/XLS/XLSX (or press Enter to cancel): ").strip().strip('"').strip("'")
        if not raw:
            raise SystemExit("No input file selected. Exiting.")
        pth = Path(raw).expanduser().resolve()
        if pth.exists():
            return pth
        print(f"File not found: {pth}")


def default_outdir_for_input(input_path: Path) -> Path:
    return input_path.parent / f"{input_path.stem}_FinShape3D_outputs_v{VERSION}"


# -----------------------------
# Geometry utilities
# -----------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def rot(v: np.ndarray, ang_rad: float) -> np.ndarray:
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=float)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0], float)
    return v / n


def bezier_quad(q0: np.ndarray, q1: np.ndarray, q2: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return ((1 - t) ** 2)[:, None] * q0 + (2 * (1 - t) * t)[:, None] * q1 + (t ** 2)[:, None] * q2


def bezier_cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return ((1 - t) ** 3)[:, None] * p0 + (3 * (1 - t) ** 2 * t)[:, None] * p1 + (3 * (1 - t) * t ** 2)[:, None] * p2 + (t ** 3)[:, None] * p3


def y_on_line_through(B: np.ndarray, A: np.ndarray, x: float) -> float:
    if abs(A[0] - B[0]) < 1e-12:
        return float("inf")
    m = (A[1] - B[1]) / (A[0] - B[0])
    return float(B[1] + m * (x - B[0]))


def polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def x_intersections_with_horizontal(polyline: np.ndarray, yval: float) -> List[float]:
    xs: List[float] = []
    for i in range(len(polyline) - 1):
        x0, y0 = float(polyline[i, 0]), float(polyline[i, 1])
        x1, y1 = float(polyline[i + 1, 0]), float(polyline[i + 1, 1])
        if abs(y1 - y0) < 1e-12:
            continue
        if (y0 - yval) * (y1 - yval) <= 0:
            t = (yval - y0) / (y1 - y0)
            if 0.0 <= t <= 1.0:
                xs.append(x0 + t * (x1 - x0))
    return xs


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _get_any(row: pd.Series, keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in row.index and pd.notna(row[k]):
            return float(row[k])
    return None


# -----------------------------
# Landmarks (normalized)
# -----------------------------
def build_landmarks(row: pd.Series, BC30_scale: float = 10.0) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    B = np.array([0.0, 0.0], float)
    C30 = np.array([-BC30_scale, 0.0], float)

    u_BC30 = (C30 - B) / np.linalg.norm(C30 - B)
    u_AB = rot(u_BC30, math.radians(-30.0))

    r_BA = _get_any(row, ["r_BA", "BA_over_BC30"])
    r_BC5 = _get_any(row, ["r_BC5", "BC5_over_BC30"])
    r_BC10 = _get_any(row, ["r_BC10", "BC10_over_BC30"])
    r_BC20 = _get_any(row, ["r_BC20", "BC20_over_BC30"])
    r_BC30 = _get_any(row, ["r_BC30"])

    if (r_BA is not None) and (r_BC5 is not None) and (r_BC10 is not None) and (r_BC20 is not None):
        if r_BC30 is None:
            r_BC30 = 1.0
        if abs(r_BC30) < 1e-12:
            raise ValueError("r_BC30 is zero/invalid.")
        scale = BC30_scale / r_BC30
        BA = r_BA * scale
        BC5 = r_BC5 * scale
        BC10 = r_BC10 * scale
        BC20 = r_BC20 * scale
    else:
        c30b_over_ab = _get_any(row, ["C30B_over_AB"])
        c5b_over_ab = _get_any(row, ["C5B_over_AB"])
        c10b_over_ab = _get_any(row, ["C10B_over_AB"])
        c20b_over_ab = _get_any(row, ["C20B_over_AB"])
        if None in (c30b_over_ab, c5b_over_ab, c10b_over_ab, c20b_over_ab):
            raise ValueError("Missing required ratios. Provide either r_* columns or C*B_over_AB columns.")
        AB = BC30_scale / c30b_over_ab
        BA = AB
        BC5 = c5b_over_ab * AB
        BC10 = c10b_over_ab * AB
        BC20 = c20b_over_ab * AB

    A = B + BA * u_AB
    C5 = B + BC5 * rot(u_AB, math.radians(5.0))
    C10 = B + BC10 * rot(u_AB, math.radians(10.0))
    C20 = B + BC20 * rot(u_AB, math.radians(20.0))

    return {"B": B, "A": A, "C5": C5, "C10": C10, "C20": C20, "C30": C30}, u_AB


# -----------------------------
# Controls
# -----------------------------
def _te2_control_through_C10_at_half(C5: np.ndarray, C10: np.ndarray, C20: np.ndarray) -> np.ndarray:
    return 2.0 * C10 - 0.5 * (C5 + C20)


def build_controls(lm: Dict[str, np.ndarray], u_AB: np.ndarray) -> Dict[str, np.ndarray]:
    B, A, C5, C10, C20, C30 = lm["B"], lm["A"], lm["C5"], lm["C10"], lm["C20"], lm["C30"]
    BA = float(np.linalg.norm(A - B))
    n_left = rot(u_AB, math.radians(90.0))

    LE_P0 = B
    LE_P1 = B + u_AB * (0.27 * BA) + n_left * (0.08 * BA)
    LE_P2 = A + np.array([0.25 * BA, 0.06 * BA], float)
    if LE_P2[0] <= A[0]:
        raise ValueError("LE_P2 must lie to the right of A.")
    LE_P3 = A

    TE2_Q0 = C5
    TE2_Q2 = C20
    TE2_Q1 = _te2_control_through_C10_at_half(C5, C10, C20)
    if TE2_Q1[1] >= y_on_line_through(B, A, float(TE2_Q1[0])):
        raise ValueError("TE2_Q1 must be below AB.")

    dA = (A - LE_P2)
    dA /= (np.linalg.norm(dA) + 1e-12)
    if dA[1] >= -1e-9:
        raise ValueError("Leading-edge tangent at A not downward; cannot build TE1.")

    dir_C5 = (TE2_Q1 - C5)
    if np.linalg.norm(dir_C5) < 1e-12:
        raise ValueError("Degenerate TE2 tangent at C5.")

    L = 0.5 * float(np.linalg.norm(A - C5))
    alpha = L / float(np.linalg.norm(dir_C5))
    TE1_P2 = C5 - alpha * dir_C5

    kA = min(0.49, max(0.10, 0.18 * BA))
    margin_y = 0.06 * BA
    kA_max_from_y = (TE1_P2[1] + margin_y - A[1]) / dA[1]
    kA = min(kA, kA_max_from_y, 0.49)
    if kA <= 0.0:
        raise ValueError("No feasible TE1_P1 satisfying ordering and |A-TE1_P1|<0.5.")
    TE1_P1 = A + kA * dA

    if not (C5[1] + 1e-6 < TE1_P2[1] < TE1_P1[1] - 1e-6 < A[1] - 1e-6):
        raise ValueError("TE1 ordering violated (C5 < TE1_P2 < TE1_P1 < A).")

    d20 = (C20 - TE2_Q1)
    if np.linalg.norm(d20) < 1e-12:
        raise ValueError("Degenerate TE2 end tangent at C20.")
    d20 /= (np.linalg.norm(d20) + 1e-12)

    seg2030 = float(np.linalg.norm(C20 - C30))
    TE3_P1 = C20 + 0.5 * seg2030 * d20

    if TE3_P1[1] <= C30[1] + 1e-6:
        raise ValueError("TE3_P1 must be above C30.")
    if TE3_P1[1] >= A[1] - 1e-6:
        raise ValueError("TE3_P1 must be below A.")
    x_lo = min(C20[0], C30[0]) - 1e-6
    x_hi = max(C20[0], C30[0]) + 1e-6
    if not (x_lo <= TE3_P1[0] <= x_hi):
        raise ValueError("TE3_P1 must lie within x-span of segment C20–C30.")

    return {
        "LE_P0": LE_P0, "LE_P1": LE_P1, "LE_P2": LE_P2, "LE_P3": LE_P3,
        "TE1_P0": A, "TE1_P1": TE1_P1, "TE1_P2": TE1_P2, "TE1_P3": C5,
        "TE2_Q0": TE2_Q0, "TE2_Q1": TE2_Q1, "TE2_Q2": TE2_Q2,
        "TE3_Q0": C20, "TE3_P1": TE3_P1, "TE3_Q2": C30,
    }


def compute_2d_curves(ctrl: Dict[str, np.ndarray], n: int = 900) -> Dict[str, np.ndarray]:
    t = np.linspace(0.0, 1.0, n)
    leading = bezier_cubic(ctrl["LE_P0"], ctrl["LE_P1"], ctrl["LE_P2"], ctrl["LE_P3"], t)
    te1 = bezier_cubic(ctrl["TE1_P0"], ctrl["TE1_P1"], ctrl["TE1_P2"], ctrl["TE1_P3"], t)
    te2 = bezier_quad(ctrl["TE2_Q0"], ctrl["TE2_Q1"], ctrl["TE2_Q2"], t)
    te3 = bezier_quad(ctrl["TE3_Q0"], ctrl["TE3_P1"], ctrl["TE3_Q2"], t)
    trailing = np.vstack([te1, te2[1:], te3[1:]])
    outline = np.vstack([leading, trailing[1:]])
    return {"leading": leading, "te1": te1, "te2": te2, "te3": te3, "trailing": trailing, "outline": outline}


def fin_area_S(outline: np.ndarray, B: np.ndarray) -> float:
    return polygon_area(np.vstack([outline, B[None, :]]))


# -----------------------------
# 3D thickness model
# -----------------------------
def morteo_SP_T(C: float) -> Tuple[float, float]:
    SP = 0.4357 * C - 1.8803
    T = 0.1191 * C + 0.5164
    return SP, T


def thickness_piecewise(c: np.ndarray, C: float, SP: float, T: float) -> np.ndarray:
    c = np.asarray(c, float)
    out = np.zeros_like(c)

    maskL = c <= SP
    if np.any(maskL) and SP > 1e-9:
        x = c[maskL]
        out[maskL] = (-T * (x ** 2) / (SP ** 2)) + (2 * T * x / SP)

    denom = (C - SP)
    maskR = ~maskL
    if np.any(maskR) and denom > 1e-9:
        x = c[maskR]
        u = (C - x)
        out[maskR] = (-T * (u ** 2) / (denom ** 2)) + (2 * T * u / denom)

    out[out < 0] = 0.0
    return out


def invert_thickness_for_c(target: float, C: float, SP: float, T: float) -> Optional[Tuple[float, float]]:
    if target < 1e-12:
        return 0.0, C
    if target > T - 1e-12:
        return SP, SP

    K = target * (SP ** 2) / max(T, 1e-12)
    disc = SP ** 2 - K
    if disc < 0:
        return None
    cL = SP - math.sqrt(max(0.0, disc))

    denom = (C - SP)
    if denom <= 1e-12:
        return None
    K2 = target * (denom ** 2) / max(T, 1e-12)
    disc2 = denom ** 2 - K2
    if disc2 < 0:
        return None
    u = denom - math.sqrt(max(0.0, disc2))
    cR = C - u
    return cL, cR


def set_equal_3d_aspect(ax, xlim, ylim, zlim) -> None:
    xr = abs(xlim[1] - xlim[0]) or 1.0
    yr = abs(ylim[1] - ylim[0]) or 1.0
    zr = abs(zlim[1] - zlim[0]) or 1.0
    ax.set_box_aspect((xr, yr, zr))


# -----------------------------
# Robust label placement
# -----------------------------
def find_interior_point(poly_path: MplPath, poly: np.ndarray) -> np.ndarray:
    """Return a point guaranteed to be inside polygon (best-effort)."""
    # Try the mean of vertices first
    cand = np.mean(poly, axis=0)
    if poly_path.contains_point((float(cand[0]), float(cand[1]))):
        return cand

    # Try mean of outline (excluding B) if available
    cand2 = np.mean(poly[:-1], axis=0)
    if poly_path.contains_point((float(cand2[0]), float(cand2[1]))):
        return cand2

    # Deterministic pseudo-random sampling inside bbox
    xmin, xmax = float(np.min(poly[:, 0])), float(np.max(poly[:, 0]))
    ymin, ymax = float(np.min(poly[:, 1])), float(np.max(poly[:, 1]))
    rng = np.random.default_rng(12345)
    for _ in range(500):
        x = float(rng.uniform(xmin, xmax))
        y = float(rng.uniform(ymin, ymax))
        if poly_path.contains_point((x, y)):
            return np.array([x, y], float)

    # Fallback: return cand even if outside (should be rare)
    return cand


def nearest_index(polyline: np.ndarray, pt: np.ndarray) -> int:
    d2 = np.sum((polyline - pt[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))


def local_outward_normal(poly_path: MplPath, polyline: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Compute outward unit normal near pt using local polyline tangent and point-in-polygon test."""
    i = nearest_index(polyline, pt)
    n = len(polyline)
    im1 = (i - 1) % n
    ip1 = (i + 1) % n
    tangent = _normalize(polyline[ip1] - polyline[im1])
    normal = _normalize(np.array([-tangent[1], tangent[0]], float))

    eps = 0.05
    test = pt + eps * normal
    if poly_path.contains_point((float(test[0]), float(test[1]))):
        normal = -normal  # flip: we want outward
    return normal


def repel_from_taken(pos: np.ndarray, taken: List[np.ndarray], step: float = 0.20,
                     min_dist: float = 0.55, max_iter: int = 40) -> np.ndarray:
    """Simple label collision avoidance: nudge until far from prior labels."""
    p = pos.copy()
    for _ in range(max_iter):
        ok = True
        for q in taken:
            if float(np.linalg.norm(p - q)) < min_dist:
                ok = False
                # nudge perpendicular-ish
                d = _normalize(p - q)
                perp = np.array([-d[1], d[0]], float)
                p = p + step * perp
                break
        if ok:
            return p
    return p


def place_landmark_label(poly_path: MplPath, polyline: np.ndarray, pt: np.ndarray,
                         taken: List[np.ndarray]) -> np.ndarray:
    n_out = local_outward_normal(poly_path, polyline, pt)
    # start close to point and expand outward until outside
    offset = 0.35
    for _ in range(35):
        cand = pt + offset * n_out
        if not poly_path.contains_point((float(cand[0]), float(cand[1]))):
            cand = repel_from_taken(cand, taken)
            taken.append(cand)
            return cand
        offset += 0.15
    cand = pt + 1.0 * n_out
    cand = repel_from_taken(cand, taken)
    taken.append(cand)
    return cand


def place_control_label(poly_path: MplPath, interior: np.ndarray, pt: np.ndarray,
                        taken: List[np.ndarray]) -> np.ndarray:
    d_in = _normalize(interior - pt)
    offset = 0.15
    for _ in range(60):
        cand = pt + offset * d_in
        if poly_path.contains_point((float(cand[0]), float(cand[1]))):
            cand = repel_from_taken(cand, taken, min_dist=0.50)
            taken.append(cand)
            return cand
        offset += 0.12
    # fallback: try interior itself
    cand = interior.copy()
    cand = repel_from_taken(cand, taken, min_dist=0.50)
    taken.append(cand)
    return cand


# -----------------------------
# Plotting
# -----------------------------
def plot_2d(curves2d: Dict[str, np.ndarray], lm: Dict[str, np.ndarray], ctrl: Dict[str, np.ndarray],
            fin_id: str, out_png: Path, legend_anchor_xy: Tuple[float, float] = (-4.0, 4.0)) -> None:
    B, A, C5, C10, C20, C30 = lm["B"], lm["A"], lm["C5"], lm["C10"], lm["C20"], lm["C30"]
    outline = curves2d["outline"]
    leading = curves2d["leading"]
    te1, te2, te3 = curves2d["te1"], curves2d["te2"], curves2d["te3"]
    S = fin_area_S(outline, B)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    poly = np.vstack([outline, B[None, :]])
    poly_path = MplPath(poly)

    # interior point for control label placement
    interior = find_interior_point(poly_path, poly)

    # shaded fin area
    ax.fill(poly[:, 0], poly[:, 1], alpha=0.18, label=f"S={S:.3f} (rel²)")

    # curves
    ax.plot(leading[:, 0], leading[:, 1], linewidth=2.0, label="Leading edge (cubic)")
    ax.plot(te1[:, 0], te1[:, 1], linewidth=2.0, label="TE1 A→C5 (cubic)")
    ax.plot(te2[:, 0], te2[:, 1], linewidth=2.0, label="TE2 C5→C20 (quadratic)")
    ax.plot(te3[:, 0], te3[:, 1], linewidth=2.0, label="TE3 C20→C30 (quadratic)")

    # rays from B (dotted)
    for pt in (C5, C10, C20, C30):
        ax.plot([B[0], pt[0]], [B[1], pt[1]], linestyle=":", linewidth=1.6, alpha=0.85)

    # control polygons (dotted) to show placement
    def dotted_poly(points: List[np.ndarray]) -> None:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        ax.plot(xs, ys, linestyle=":", linewidth=1.2, alpha=0.75)

    dotted_poly([ctrl["LE_P0"], ctrl["LE_P1"], ctrl["LE_P2"], ctrl["LE_P3"]])
    dotted_poly([ctrl["TE1_P0"], ctrl["TE1_P1"], ctrl["TE1_P2"], ctrl["TE1_P3"]])
    dotted_poly([ctrl["TE2_Q0"], ctrl["TE2_Q1"], ctrl["TE2_Q2"]])
    dotted_poly([ctrl["TE3_Q0"], ctrl["TE3_P1"], ctrl["TE3_Q2"]])

    # label placement with simple collision avoidance
    taken: List[np.ndarray] = []

    # landmarks: OUTSIDE + bold
    lm_fs = int(FS_POINT * FONT_SCALE * LM_LABEL_MULT)
    landmarks = {"B": B, "A": A, "C5": C5, "C10": C10, "C20": C20, "C30": C30}
    for name, pt in landmarks.items():
        ax.scatter([pt[0]], [pt[1]], s=45)
        lab = place_landmark_label(poly_path, outline, pt, taken)
        ax.text(lab[0], lab[1], name, fontsize=lm_fs, fontweight="bold")

    # control points: label INSIDE
    cp_fs = int(FS_CTRL * FONT_SCALE * CTRL_LABEL_MULT)
    ctrl_points = {
        "LE-P1": ctrl["LE_P1"], "LE-P2": ctrl["LE_P2"],
        "TE1-P1": ctrl["TE1_P1"], "TE1-P2": ctrl["TE1_P2"],
        "TE2-Q1": ctrl["TE2_Q1"], "TE3-P1": ctrl["TE3_P1"],
    }
    for name, pt in ctrl_points.items():
        ax.scatter([pt[0]], [pt[1]], s=35, marker="x")
        lab = place_control_label(poly_path, interior, pt, taken)
        ax.text(lab[0], lab[1], name, fontsize=cp_fs)

    # trim axes to nearest integers and include legend anchor
    ax_xmin = min(float(poly[:, 0].min()), legend_anchor_xy[0]) - 1.0
    ax_xmax = max(float(poly[:, 0].max()), legend_anchor_xy[0]) + 1.0
    ax_ymin = min(float(poly[:, 1].min()), 0.0, legend_anchor_xy[1]) - 1.0
    ax_ymax = max(float(poly[:, 1].max()), legend_anchor_xy[1]) + 1.0
    ax.set_xlim(math.floor(ax_xmin), math.ceil(ax_xmax))
    ax.set_ylim(math.floor(ax_ymin), math.ceil(ax_ymax))
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X", fontsize=int(FS_AXLAB * FONT_SCALE))
    ax.set_ylabel("Y (height)", fontsize=int(FS_AXLAB * FONT_SCALE))
    ax.tick_params(labelsize=int(8 * FONT_SCALE))

    ax.set_title(f"{fin_id}: 2D Bézier contour (FinShape3D {VERSION})", fontsize=FS_TITLE_2D)

    ax.legend(
        loc="lower left",
        bbox_to_anchor=legend_anchor_xy,
        bbox_transform=ax.transData,
        framealpha=0.90,
        fontsize=int(FS_LEGEND * LEGEND_SCALE),
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_3d_wireframe(curves2d: Dict[str, np.ndarray], fin_id: str, out_png: Path,
                      n_y_slices: int = 30, n_chord_pts: int = 90, n_z_slices: int = 15,
                      top_focus_power: float = 3.0) -> None:
    leading2d = curves2d["leading"]
    trailing2d = curves2d["trailing"]
    outline2d = curves2d["outline"]

    y_max = float(outline2d[:, 1].max())
    if y_max <= 1e-9:
        raise ValueError(f"{fin_id}: degenerate outline (y_max≈0).")

    t = np.linspace(0.0, 1.0, n_y_slices)
    y_vals = y_max * (1.0 - (1.0 - t) ** top_focus_power)
    y_vals = np.clip(y_vals, 0.0, y_max * 0.9995)

    sections = []
    halfTmax = 0.0
    for y in y_vals:
        xs_le = x_intersections_with_horizontal(leading2d, float(y))
        xs_tr = x_intersections_with_horizontal(trailing2d, float(y))
        if (not xs_le) or (not xs_tr):
            continue
        x_le = max(xs_le)
        x_tr = min(xs_tr)
        C = float(x_le - x_tr)
        if C <= 0.10:
            continue

        SP, Tm = morteo_SP_T(C)
        SP = clamp(SP, 0.05 * C, 0.95 * C)
        Tm = max(Tm, 1e-4)
        halfTmax = max(halfTmax, 0.5 * Tm)
        sections.append((float(y), float(x_le), float(C), float(SP), float(Tm)))

    if len(sections) < 3:
        raise ValueError(f"{fin_id}: insufficient slices for 3D wireframe.")

    z_levels = np.linspace(0.0, halfTmax, n_z_slices)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    wire_color = "blue"

    c_frac = np.linspace(0.0, 1.0, n_chord_pts)
    for (y, x_le, C, SP, Tm) in sections:
        c = c_frac * C
        x = x_le - c
        th = thickness_piecewise(c, C, SP, Tm)
        z_up = 0.5 * th
        z_dn = -0.5 * th
        ax.plot(x, z_up, np.full_like(x, y), linewidth=0.85, color=wire_color)
        ax.plot(x, z_dn, np.full_like(x, y), linewidth=0.85, color=wire_color)

    for zl in z_levels[1:]:
        for sign in (+1.0, -1.0):
            zc = sign * zl
            xs_front, xs_back, ys = [], [], []
            for (y, x_le, C, SP, Tm) in sections:
                ys.append(y)
                target = 2.0 * abs(zc)
                if target > Tm:
                    xs_front.append(np.nan); xs_back.append(np.nan)
                    continue
                inv = invert_thickness_for_c(target, C, SP, Tm)
                if inv is None:
                    xs_front.append(np.nan); xs_back.append(np.nan)
                    continue
                cL, cR = inv
                xs_front.append(x_le - cL)
                xs_back.append(x_le - cR)

            xs_front = np.array(xs_front, float)
            xs_back = np.array(xs_back, float)
            ys = np.array(ys, float)

            for xs in (xs_front, xs_back):
                mask = np.isfinite(xs)
                if mask.sum() < 2:
                    continue
                idx = np.where(mask)[0]
                blocks = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
                for b in blocks:
                    if len(b) >= 2:
                        ax.plot(xs[b], np.full(len(b), zc), ys[b], linewidth=0.8, color=wire_color)

    ax.plot(outline2d[:, 0], np.zeros(len(outline2d)), outline2d[:, 1], linewidth=2.2, color="black")
    ax.scatter([0.0], [0.0], [0.0], s=45, color="black")

    ax.set_xlabel("X", fontsize=int(FS_AXLAB * FONT_SCALE), labelpad=18)
    ax.set_ylabel("Z", fontsize=int(FS_AXLAB * FONT_SCALE), labelpad=24)
    ax.set_zlabel("Y", fontsize=int(FS_AXLAB * FONT_SCALE), labelpad=18)
    ax.tick_params(labelsize=int(8 * FONT_SCALE))

    xmin = float(outline2d[:, 0].min()) - 1.0
    xmax = float(outline2d[:, 0].max()) + 1.0
    xlim = (math.floor(xmin), math.ceil(xmax))

    thickness_lim = max(halfTmax * 1.2, 1.2)
    ylim = (-thickness_lim, thickness_lim)
    zlim = (0.0, y_max * 1.02)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    ax.set_yticks([-1.0, 0.0, 1.0])

    set_equal_3d_aspect(ax, xlim, ylim, zlim)
    ax.view_init(elev=18, azim=-65)
    ax.set_title(f"{fin_id}: 3D wireframe (FinShape3D {VERSION})", fontsize=FS_TITLE_3D)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)


def make_montage(png_paths: List[Path], out_path: Path, title: Optional[str] = None) -> None:
    n = len(png_paths)
    cols = 2
    rows = int(math.ceil(n / cols)) if n > 0 else 1

    imgs = [mpimg.imread(str(p)) for p in png_paths]
    fig, axes = plt.subplots(rows, cols, figsize=(7.0 * cols, 6.2 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for k, (img, p) in enumerate(zip(imgs, png_paths)):
        r = k // cols
        c = k % cols
        axes[r, c].imshow(img)
        axes[r, c].set_title(p.name, fontsize=FS_MONT_TILE)

    if title:
        fig.suptitle(title, fontsize=FS_MONT_TITLE)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Pipeline
# -----------------------------
def run(input_path: Path, outdir: Path, bc30scale: float, y_slices: int, z_slices: int,
        top_focus_power: float, montage: bool) -> None:
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
        plot_3d_wireframe(curves2d, fin_id, p3d, n_y_slices=y_slices, n_z_slices=z_slices,
                          top_focus_power=top_focus_power)

        outs2d.append(p2d)
        outs3d.append(p3d)

    if montage and outs2d:
        make_montage(outs2d, outdir / f"FinShape3D_{VERSION}_montage_2D.png",
                     title=f"FinShape3D {VERSION} — 2D Contours")
    if montage and outs3d:
        make_montage(outs3d, outdir / f"FinShape3D_{VERSION}_montage_3D.png",
                     title=f"FinShape3D {VERSION} — 3D Wireframes")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Reconstruct 2D/3D dorsal fins from ratio tables.")
    ap.add_argument("--input", type=Path, default=None, help="CSV or XLSX containing ratios.")
    ap.add_argument("--outdir", type=Path, default=None, help="Output directory (default next to input).")
    ap.add_argument("--bc30scale", type=float, default=10.0, help="Set |BC30| to this value (relative units).")
    ap.add_argument("--y_slices", type=int, default=30, help="Horizontal slices for 3D wireframe.")
    ap.add_argument("--z_slices", type=int, default=15, help="Vertical slices for 3D wireframe.")
    ap.add_argument("--top_focus_power", type=float, default=3.0, help="Concentrate y-slices near tip.")
    ap.add_argument("--no_montage", action="store_true", help="Disable montage creation.")

    # PyCharm-safe: ignore foreign args
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
