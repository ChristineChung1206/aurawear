from __future__ import annotations

from typing import List, Tuple
import numpy as np


def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    h = hex_str.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return r, g, b


def _srgb_to_linear(c: float) -> float:
    # c in [0,1]
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _rgb_to_xyz(rgb: Tuple[int, int, int]) -> np.ndarray:
    # D65
    r, g, b = rgb
    r = _srgb_to_linear(r / 255.0)
    g = _srgb_to_linear(g / 255.0)
    b = _srgb_to_linear(b / 255.0)

    # sRGB -> XYZ (D65)
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    return np.array([x, y, z], dtype=np.float32)


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    # Reference white D65
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = xyz[0] / Xn, xyz[1] / Yn, xyz[2] / Zn

    def f(t: float) -> float:
        eps = 216 / 24389
        k = 24389 / 27
        if t > eps:
            return t ** (1/3)
        return (k * t + 16) / 116

    fx, fy, fz = f(float(x)), f(float(y)), f(float(z))
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.array([L, a, b], dtype=np.float32)


def hex_to_lab(hex_str: str) -> np.ndarray:
    rgb = _hex_to_rgb(hex_str)
    xyz = _rgb_to_xyz(rgb)
    lab = _xyz_to_lab(xyz)
    return lab


def deltaE76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    return float(np.linalg.norm(lab1 - lab2))


def color_score_min_deltaE(item_hex_list: List[str], selected_hex_list: List[str], tau: float) -> float:
    """
    Score = exp(-minΔE / tau), where minΔE is min distance between item dominant colors and selected palette colors.
    """
    if not item_hex_list or not selected_hex_list:
        return 0.0

    item_labs = [hex_to_lab(hx) for hx in item_hex_list]
    sel_labs = [hex_to_lab(hx) for hx in selected_hex_list]

    min_de = 1e9
    for il in item_labs:
        for sl in sel_labs:
            de = deltaE76(il, sl)
            if de < min_de:
                min_de = de

    return float(np.exp(-min_de / max(1e-6, tau)))
