import math
from typing import Tuple, List
import numpy as np

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# Tuple-based geometry (for Tkinter GUI)
def v_add(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])

def v_sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])

def v_mul(a: Tuple[float, float], s: float) -> Tuple[float, float]:
    return (a[0] * s, a[1] * s)

def v_len(a: Tuple[float, float]) -> float:
    return math.hypot(a[0], a[1])

def v_norm(a: Tuple[float, float]) -> Tuple[float, float]:
    n = v_len(a)
    if n < 1e-12:
        return (1.0, 0.0)
    return (a[0] / n, a[1] / n)

def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return v_len(v_sub(a, b))

def shoelace_area(poly: List[Tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5

def rot2(v: Tuple[float, float], ang_rad: float) -> Tuple[float, float]:
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])

def rotate_about_origin(p: Tuple[float, float], ang: float) -> Tuple[float, float]:
    return rot2(p, ang)

# Numpy-based geometry (for 3D generation and matplotlib)
def rot_np(v: np.ndarray, ang_rad: float) -> np.ndarray:
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=float)

def normalize_np(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0], float)
    return v / n

def y_on_line_through_np(B: np.ndarray, A: np.ndarray, x: float) -> float:
    if abs(A[0] - B[0]) < 1e-12:
        return float("inf")
    m = (A[1] - B[1]) / (A[0] - B[0])
    return float(B[1] + m * (x - B[0]))

def polygon_area_np(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def x_intersections_with_horizontal_np(polyline: np.ndarray, yval: float) -> List[float]:
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
