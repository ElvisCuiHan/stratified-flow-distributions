from __future__ import annotations

import math
from typing import Any

import numpy as np

from .core import Beta01, GaussianComponent, SFD, TubeComponent, Uniform01


def _pack(x: np.ndarray, y: np.ndarray, z: np.ndarray | float | None, d: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.zeros((x.size, d), dtype=float)
    out[:, 0] = x
    if d >= 2:
        out[:, 1] = y
    if d >= 3:
        out[:, 2] = 0.0 if z is None else np.asarray(z, dtype=float)
    return out


def _line(x0: float, y0: float, x1: float, y1: float, d: int, z_phase: float = 0.0):
    def curve(s):
        s = np.asarray(s, dtype=float)
        x = (1.0 - s) * x0 + s * x1
        y = (1.0 - s) * y0 + s * y1
        z = 0.10 * np.sin(2.0 * np.pi * s + z_phase)
        return _pack(x, y, z, d)

    return curve


def _ellipse(cx: float, cy: float, rx: float, ry: float, d: int, z_phase: float = 0.0):
    def curve(s):
        s = np.asarray(s, dtype=float)
        theta = 2.0 * np.pi * s
        x = cx + rx * np.cos(theta)
        y = cy + ry * np.sin(theta)
        z = 0.10 * np.sin(theta + z_phase)
        return _pack(x, y, z, d)

    return curve


def _arc(cx: float, cy: float, rx: float, ry: float, a0: float, a1: float, d: int, z_phase: float = 0.0):
    def curve(s):
        s = np.asarray(s, dtype=float)
        theta = a0 + (a1 - a0) * s
        x = cx + rx * np.cos(theta)
        y = cy + ry * np.sin(theta)
        z = 0.10 * np.sin(2.0 * np.pi * s + z_phase)
        return _pack(x, y, z, d)

    return curve


def snowflake(
    *,
    d: int = 3,
    sigma_shape: float = 0.035,
    z_sigma: float = 0.012,
    kernel: str = "gaussian",
    kernel_df: float = 5.0,
    random_state: Any = None,
) -> SFD:
    """Snowflake graph-stratified SFD.

    The component weights match the manuscript DGP: 0.58 total main-arm mass,
    0.36 total side-crystal mass, and 0.06 total center mass.
    """

    components = []
    cov_tube = np.diag([sigma_shape**2, sigma_shape**2] + [z_sigma**2] * max(0, d - 2))
    anchors = (0.46, 0.78, 1.08)
    lengths = (0.38, 0.30, 0.22)

    for arm in range(1, 7):
        theta = math.pi / 2.0 + (arm - 1) * math.pi / 3.0
        u = np.array([math.cos(theta), math.sin(theta)])

        def main_curve(s, arm=arm, u=u):
            s = np.asarray(s, dtype=float)
            r = 0.10 + 1.55 * s
            xy = r[:, None] * u[None, :]
            z = 0.30 * np.sin(2.0 * np.pi * (r / 1.65) + arm / 6.0)
            return _pack(xy[:, 0], xy[:, 1], z, d)

        components.append(
            TubeComponent(
                name=f"main_arm_{arm}",
                weight=0.58 / 6.0,
                kind="tube",
                curve=main_curve,
                cov=cov_tube,
                s_dist=Beta01(1.25, 1.45),
                kernel=kernel,
                kernel_df=kernel_df,
                metadata={"arm": arm, "layer": "main_arm"},
            )
        )

        for branchlet, (anchor, length) in enumerate(zip(anchors, lengths), start=1):
            for side in (-1, 1):
                branch_dir = np.array([math.cos(theta + side * math.pi / 3.0), math.sin(theta + side * math.pi / 3.0)])

                def side_curve(s, arm=arm, u=u, branch_dir=branch_dir, anchor=anchor, length=length):
                    s = np.asarray(s, dtype=float)
                    xy = anchor * u[None, :] + (length * s)[:, None] * branch_dir[None, :]
                    t_true = np.minimum(1.0, (anchor + length * s) / 1.65)
                    z = 0.30 * np.sin(2.0 * np.pi * t_true + arm / 6.0)
                    return _pack(xy[:, 0], xy[:, 1], z, d)

                components.append(
                    TubeComponent(
                        name=f"side_arm_{arm}_b{branchlet}_{'plus' if side > 0 else 'minus'}",
                        weight=0.01,
                        kind="tube",
                        curve=side_curve,
                        cov=cov_tube,
                        s_dist=Uniform01(),
                        kernel=kernel,
                        kernel_df=kernel_df,
                        metadata={"arm": arm, "branchlet": branchlet, "side": side, "layer": "side_crystal"},
                    )
                )

        loc = np.zeros(d)
        if d >= 3:
            loc[2] = 0.30 * np.sin(arm / 6.0)
        cov_center = np.diag([(0.12**2 + sigma_shape**2), (0.12**2 + sigma_shape**2)] + [z_sigma**2] * max(0, d - 2))
        components.append(
            GaussianComponent(
                name=f"center_sparkle_{arm}",
                weight=0.01,
                kind="gaussian",
                loc=loc,
                cov=cov_center,
                metadata={"arm": arm, "layer": "center_sparkle"},
            )
        )

    return SFD(components, name="snowflake", random_state=random_state)


def cartoon_face(
    *,
    d: int = 3,
    sigma_curve: float = 0.020,
    sigma_blob: float = 0.030,
    z_sigma: float = 0.012,
    kernel: str = "gaussian",
    kernel_df: float = 5.0,
    random_state: Any = None,
) -> SFD:
    """Doraemon-style cartoon-face SFD made from curve tubes and Gaussian blobs."""

    raw = [
        ("head_outline", "curve", 0.13),
        ("face_outline", "curve", 0.11),
        ("left_eye", "curve", 0.065),
        ("right_eye", "curve", 0.065),
        ("left_pupil", "blob", 0.030),
        ("right_pupil", "blob", 0.030),
        ("nose", "blob", 0.050),
        ("nose_highlight", "blob", 0.012),
        ("nose_line", "curve", 0.040),
        ("mouth_arc", "curve", 0.085),
        ("left_whisker_upper", "curve", 0.035),
        ("left_whisker_mid", "curve", 0.035),
        ("left_whisker_lower", "curve", 0.035),
        ("right_whisker_upper", "curve", 0.035),
        ("right_whisker_mid", "curve", 0.035),
        ("right_whisker_lower", "curve", 0.035),
        ("collar", "curve", 0.080),
        ("bell_outline", "curve", 0.055),
        ("bell_center", "blob", 0.035),
        ("bell_slot", "curve", 0.020),
        ("bell_cut", "curve", 0.018),
    ]
    total = sum(w for _, _, w in raw)
    curve_defs = {
        "head_outline": _ellipse(0, 0.00, 1.72, 1.72, d, 0.1),
        "face_outline": _ellipse(0, -0.23, 1.30, 1.22, d, 0.2),
        "left_eye": _ellipse(-0.36, 0.76, 0.34, 0.47, d, 0.3),
        "right_eye": _ellipse(0.36, 0.76, 0.34, 0.47, d, 0.4),
        "nose_line": _line(0, 0.23, 0, -0.54, d, 0.5),
        "mouth_arc": _arc(0, -0.42, 0.78, 0.58, math.pi, 2 * math.pi, d, 0.6),
        "left_whisker_upper": _line(-0.22, 0.20, -1.40, 0.53, d, 0.7),
        "left_whisker_mid": _line(-0.25, 0.02, -1.50, 0.02, d, 0.8),
        "left_whisker_lower": _line(-0.22, -0.16, -1.35, -0.43, d, 0.9),
        "right_whisker_upper": _line(0.22, 0.20, 1.40, 0.53, d, 1.0),
        "right_whisker_mid": _line(0.25, 0.02, 1.50, 0.02, d, 1.1),
        "right_whisker_lower": _line(0.22, -0.16, 1.35, -0.43, d, 1.2),
        "collar": _line(-1.08, -1.34, 1.08, -1.34, d, 1.3),
        "bell_outline": _ellipse(0, -1.55, 0.25, 0.25, d, 1.4),
        "bell_slot": _line(-0.14, -1.49, 0.14, -1.49, d, 1.5),
        "bell_cut": _line(0, -1.50, 0, -1.70, d, 1.6),
    }
    blob_centers = {
        "left_pupil": (-0.25, 0.72),
        "right_pupil": (0.25, 0.72),
        "nose": (0, 0.34),
        "nose_highlight": (0.065, 0.43),
        "bell_center": (0, -1.55),
    }
    components = []
    curve_cov = np.diag([sigma_curve**2, sigma_curve**2] + [z_sigma**2] * max(0, d - 2))
    blob_cov = np.diag([sigma_blob**2, sigma_blob**2] + [z_sigma**2] * max(0, d - 2))

    for idx, (name, typ, weight) in enumerate(raw, start=1):
        if typ == "curve":
            components.append(
                TubeComponent(
                    name=name,
                    weight=weight / total,
                    kind="tube",
                    curve=curve_defs[name],
                    cov=curve_cov,
                    s_dist=Uniform01(),
                    kernel=kernel,
                    kernel_df=kernel_df,
                    metadata={"layer": "tube"},
                )
            )
        else:
            loc = np.zeros(d)
            loc[0], loc[1] = blob_centers[name]
            if d >= 3:
                loc[2] = 0.06 * np.sin(idx)
            components.append(
                GaussianComponent(
                    name=name,
                    weight=weight / total,
                    kind="gaussian",
                    loc=loc,
                    cov=blob_cov,
                    metadata={"layer": "atom_like_blob"},
                )
            )

    return SFD(components, name="cartoon_face", random_state=random_state)


def _koch_edges(depth: int) -> list[tuple[np.ndarray, np.ndarray]]:
    tri = [
        (np.array([-1.2, -0.7]), np.array([1.2, -0.7])),
        (np.array([1.2, -0.7]), np.array([0.0, 1.35])),
        (np.array([0.0, 1.35]), np.array([-1.2, -0.7])),
    ]
    edges = tri
    rot = np.array([[0.5, -math.sqrt(3) / 2], [math.sqrt(3) / 2, 0.5]])
    for _ in range(depth):
        new_edges = []
        for a, b in edges:
            v = (b - a) / 3.0
            p1 = a + v
            p2 = a + 2.0 * v
            peak = p1 + rot @ v
            new_edges.extend([(a, p1), (p1, peak), (peak, p2), (p2, b)])
        edges = new_edges
    return edges


def koch_prefractal(
    *,
    depth: int = 3,
    d: int = 3,
    sigma: float = 0.018,
    z_sigma: float = 0.010,
    kernel: str = "gaussian",
    kernel_df: float = 5.0,
    random_state: Any = None,
) -> SFD:
    edges = _koch_edges(depth)
    cov = np.diag([sigma**2, sigma**2] + [z_sigma**2] * max(0, d - 2))
    components = []
    for i, (a, b) in enumerate(edges):
        components.append(
            TubeComponent(
                name=f"koch_edge_{i}",
                weight=1.0 / len(edges),
                kind="tube",
                curve=_line(a[0], a[1], b[0], b[1], d, z_phase=i / 7.0),
                cov=cov,
                s_dist=Uniform01(),
                kernel=kernel,
                kernel_df=kernel_df,
                metadata={"edge": i, "depth": depth},
            )
        )
    return SFD(components, name="koch_prefractal", random_state=random_state)


def torus_knot(*, sigma: float = 0.070, kernel: str = "gaussian", kernel_df: float = 5.0, random_state: Any = None) -> SFD:
    def curve(s):
        s = np.asarray(s, dtype=float)
        u = 2.0 * np.pi * s
        x = (2.0 + np.cos(3.0 * u)) * np.cos(2.0 * u)
        y = (2.0 + np.cos(3.0 * u)) * np.sin(2.0 * u)
        z = np.sin(3.0 * u)
        return np.column_stack([x, y, z])

    comp = TubeComponent(
        name="torus_knot",
        weight=1.0,
        kind="tube",
        curve=curve,
        cov=np.eye(3) * sigma**2,
        s_dist=Uniform01(),
        kernel=kernel,
        kernel_df=kernel_df,
    )
    return SFD([comp], name="torus_knot", random_state=random_state)


def maze_graph(
    *,
    d: int = 3,
    sigma_edge: float = 0.032,
    sigma_atom: float = 0.060,
    z_sigma: float = 0.012,
    kernel: str = "gaussian",
    kernel_df: float = 5.0,
    random_state: Any = None,
) -> SFD:
    edges = [
        ((-1.55, -1.20), (1.55, -1.20)),
        ((-1.55, -1.20), (-1.55, 1.20)),
        ((-1.55, 1.20), (1.55, 1.20)),
        ((1.55, 1.20), (1.55, -0.20)),
        ((-1.15, 0.05), (-0.25, 0.05)),
        ((-0.25, 0.05), (-0.25, -0.60)),
        ((-0.25, -0.60), (0.65, -0.60)),
        ((0.65, -0.60), (0.65, 0.50)),
        ((-0.85, -1.20), (-0.85, 0.80)),
        ((-0.85, 0.50), (0.65, 0.50)),
        ((0.15, -1.20), (0.15, -0.90)),
        ((0.15, -0.90), (1.30, -0.90)),
        ((1.30, -0.90), (1.30, -0.20)),
    ]
    junctions = [(-1.15, 0.05), (-0.25, -0.60), (0.65, -0.60), (0.65, 0.50), (1.55, -0.20)]
    pi_atom = 0.18
    cov_edge = np.diag([sigma_edge**2, sigma_edge**2] + [z_sigma**2] * max(0, d - 2))
    cov_atom = np.diag([sigma_atom**2, sigma_atom**2] + [z_sigma**2] * max(0, d - 2))
    components = []
    for i, (a, b) in enumerate(edges):
        components.append(
            TubeComponent(
                name=f"corridor_{i}",
                weight=(1.0 - pi_atom) / len(edges),
                kind="tube",
                curve=_line(a[0], a[1], b[0], b[1], d, z_phase=i / 6.0),
                cov=cov_edge,
                s_dist=Uniform01(),
                kernel=kernel,
                kernel_df=kernel_df,
                metadata={"edge": i, "layer": "corridor"},
            )
        )
    for j, xy in enumerate(junctions):
        loc = np.zeros(d)
        loc[0], loc[1] = xy
        if d >= 3:
            loc[2] = 0.08 * np.sin(j)
        components.append(
            GaussianComponent(
                name=f"junction_{j}",
                weight=pi_atom / len(junctions),
                kind="gaussian",
                loc=loc,
                cov=cov_atom,
                metadata={"junction": j, "layer": "junction"},
            )
        )
    return SFD(components, name="maze_graph", random_state=random_state)


def ecg_like(
    *,
    sigma_x: float = 0.004,
    sigma_y0: float = 0.035,
    sigma_y1: float = 0.055,
    kernel: str = "gaussian",
    kernel_df: float = 5.0,
    random_state: Any = None,
) -> SFD:
    def waveform(s, group):
        shift = 0.018 if group == 1 else 0.0
        amp = 1.12 if group == 1 else 1.0
        return (
            0.35 * np.exp(-((s - (0.18 + shift)) ** 2) / 0.00075)
            - 0.45 * np.exp(-((s - (0.37 + shift)) ** 2) / 0.00015)
            + amp * 1.55 * np.exp(-((s - (0.405 + shift)) ** 2) / 0.00008)
            - 0.55 * np.exp(-((s - (0.43 + shift)) ** 2) / 0.00016)
            + 0.48 * np.exp(-((s - (0.70 + shift)) ** 2) / 0.006)
        )

    comps = []
    for group, sy in [(0, sigma_y0), (1, sigma_y1)]:
        def curve(s, group=group):
            s = np.asarray(s, dtype=float)
            return np.column_stack([s, waveform(s, group)])

        comps.append(
            TubeComponent(
                name=f"A{group}",
                weight=0.5,
                kind="tube",
                curve=curve,
                cov=np.diag([sigma_x**2, sy**2]),
                s_dist=Uniform01(),
                kernel=kernel,
                kernel_df=kernel_df,
                metadata={"A": group},
            )
        )
    return SFD(comps, name="ecg_like", random_state=random_state)
