"""SciPy-style simulation tools for Stratified Flow Distributions.

The public API mirrors the parts of ``scipy.stats`` that are most useful for
simulation work: create a frozen distribution with a factory function, then
call ``rvs``, ``pdf``, ``logpdf``, ``mean``, and ``cov``.
"""

from .core import Beta01, GaussianComponent, SFD, TubeComponent, Uniform01
from .factories import (
    cartoon_face,
    ecg_like,
    koch_prefractal,
    maze_graph,
    snowflake,
    torus_knot,
)

__all__ = [
    "Beta01",
    "GaussianComponent",
    "SFD",
    "TubeComponent",
    "Uniform01",
    "cartoon_face",
    "ecg_like",
    "koch_prefractal",
    "maze_graph",
    "snowflake",
    "torus_knot",
]
