from __future__ import annotations

import numpy as np

from scipy.interpolate import PPoly, CubicHermiteSpline


class PiecewiseQuinticInterpolator(PPoly):
    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 dydx: np.ndarray,
                 d2ydx2: np.ndarray,
                 extrapolate: bool = True) -> None:
        dx = np.diff(x)

        a0, a1, a2 = y[:-1], dydx[:-1], d2ydx2[:-1]
        b0, b1, b2 = y[1:], dydx[1:], d2ydx2[1:]

        c = np.empty((6, len(x) - 1))
        c[0] = (-12 * a0 - 6 * a1 * dx - a2 * dx ** 2 + 12 * b0 - 6 * b1 * dx + b2 * dx ** 2) / (2 * dx ** 5)
        c[1] = (30 * a0 + 16 * a1 * dx + 3 * a2 * dx ** 2 - 30 * b0 + 14 * b1 * dx - 2 * b2 * dx ** 2) / (2 * dx ** 4)
        c[2] = (-20 * a0 - 12 * a1 * dx - 3 * a2 * dx ** 2 + 20 * b0 - 8 * b1 * dx + b2 * dx ** 2) / (2 * dx ** 3)
        c[3] = a2 / 2
        c[4] = a1
        c[5] = a0
        super().__init__(c, x, extrapolate=extrapolate)


class ModifiedAkimaSpline(CubicHermiteSpline):
    def __init__(self, x: np.ndarray | list[float], y: np.ndarray | list[float]) -> None:
        dx = np.diff(x)
        dy = np.diff(y)
        dydx = ModifiedAkimaSpline.modified_akima_slopes(dy / dx)
        super().__init__(x, y, dydx, axis=0, extrapolate=None)

    @staticmethod
    def modified_akima_slopes(edge_slopes: np.ndarray | list[float]) -> np.ndarray:
        n = len(edge_slopes) + 1  # number of grid nodes x
        delta_0 = 2 * edge_slopes[0] - edge_slopes[1]
        delta_m1 = 2 * delta_0 - edge_slopes[0]
        delta_n = 2 * edge_slopes[-1] - edge_slopes[-2]
        delta_n1 = 2 * delta_n - edge_slopes[-1]
        edge_slopes = np.concatenate(([delta_m1, delta_0], edge_slopes, [delta_n, delta_n1]))

        # Weights calculation, modified from Akima's original to reduce overshoot/undershoot
        weights = np.abs(np.diff(edge_slopes)) + np.abs((edge_slopes[:-1] + edge_slopes[1:]) / 2.0)

        weights1 = weights[:n]   # |d(i-1)-d(i-2)|
        weights2 = weights[2:]   # |d(i+1)-d(i)|
        delta1 = edge_slopes[1:n + 1]    # d(i-1)
        delta2 = edge_slopes[2:n + 2]    # d(i)

        weights12 = weights1 + weights2
        indices = np.argwhere(weights12 == 0)
        weights12[indices] = 1
        vertex_slopes = (weights2 / weights12) * delta1 + (weights1 / weights12) * delta2
        vertex_slopes[indices] = 0

        return vertex_slopes
