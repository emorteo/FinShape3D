from typing import Tuple, List
import numpy as np

# Tuple-based (for Tkinter GUI)
def bezier_quad(q0: Tuple[float, float], q1: Tuple[float, float], q2: Tuple[float, float], t: float) -> Tuple[float, float]:
    u = 1.0 - t
    return (
        u * u * q0[0] + 2 * u * t * q1[0] + t * t * q2[0],
        u * u * q0[1] + 2 * u * t * q1[1] + t * t * q2[1],
    )

def bezier_cubic(p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], t: float) -> Tuple[float, float]:
    u = 1.0 - t
    return (
        (u ** 3) * p0[0] + 3 * (u ** 2) * t * p1[0] + 3 * u * (t ** 2) * p2[0] + (t ** 3) * p3[0],
        (u ** 3) * p0[1] + 3 * (u ** 2) * t * p1[1] + 3 * u * (t ** 2) * p2[1] + (t ** 3) * p3[1],
    )

def sample_curve(fn, n: int = 280) -> List[Tuple[float, float]]:
    return [fn(i / (n - 1)) for i in range(n)] if n > 1 else [fn(0.0)]


# Numpy-based (for 3D generation and matplotlib)
def bezier_quad_np(q0: np.ndarray, q1: np.ndarray, q2: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return ((1 - t) ** 2)[:, None] * q0 + (2 * (1 - t) * t)[:, None] * q1 + (t ** 2)[:, None] * q2

def bezier_cubic_np(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return ((1 - t) ** 3)[:, None] * p0 + (3 * (1 - t) ** 2 * t)[:, None] * p1 + (3 * (1 - t) * t ** 2)[:, None] * p2 + (t ** 3)[:, None] * p3
