# SPDX-License-Identifier: Apache-2.0
"""Static knowledge about the AhmedML dataset (``neashton/ahmedml`` on HuggingFace).

Reference conditions come from the OpenFOAM case shipped with the dataset
(``openfoam-casesetup.tgz``: ``system/forceCoeffs``, ``constant/transportProperties``,
``constant/turbulenceProperties``, ``system/controlDict``) and the dataset paper
Ashton et al., "AhmedML: High-Fidelity Computational Fluid Dynamics dataset for incompressible,
low-speed bluff body aerodynamics", arXiv:2407.20801 (2024).
"""

from __future__ import annotations

from typing import Any

HF_REPO = "neashton/ahmedml"
HF_REPO_TYPE = "dataset"
PAPER_URL = "https://arxiv.org/abs/2407.20801"
LICENSE = "CC-BY-SA-4.0"

# Per-run files we ingest. ``volume_<i>.vtu`` (5.6 GB per run) is deliberately absent and
# ``FORBIDDEN_PATTERNS`` makes the downloader refuse it even if asked.
RUN_FILES: dict[str, str] = {
    "surface": "run_{i}/boundary_{i}.vtp",
    "stl": "run_{i}/ahmed_{i}.stl",
    "forces": "run_{i}/force_mom_{i}.csv",
    "forces_varref": "run_{i}/force_mom_varref_{i}.csv",
    "geometry": "run_{i}/geo_parameters_{i}.csv",
    "cell_area": "run_{i}/boundary_cell_area_{i}.npy",
}
OPTIONAL_ROLES = frozenset({"forces_varref", "cell_area"})
FORBIDDEN_PATTERNS = ("volume_", ".vtu")

# forceCoeffs function object: rhoInf 1, magUInf 1, lRef 1.04, Aref 0.112032 (constant for the
# ``force_mom_<i>.csv`` files; ``force_mom_varref_<i>.csv`` uses the per-geometry frontal area).
# transportProperties: nu = 3.75e-7 m^2/s so Re_L = U*L/nu = 2.77e6 and Re_H = 7.68e5 with the
# baseline body height H = 0.288 m, matching the paper.
BASELINE_HEIGHT_M = 0.288
CONDITIONS: dict[str, Any] = {
    "u_inf": 1.0,
    "u_inf_units": "m/s",
    "rho": 1.0,
    "rho_units": "kg/m^3",
    "nu": 3.75e-7,
    "nu_units": "m^2/s",
    "reference_area": 0.112032,
    "reference_area_units": "m^2",
    "reference_length": 1.04,
    "reference_length_units": "m",
    "re_L": 1.0 * 1.04 / 3.75e-7,
    "re_H": 1.0 * BASELINE_HEIGHT_M / 3.75e-7,
    "convective_time_units": 80,
    "pressure_kind": "kinematic",  # OpenFOAM incompressible: p is p/rho in m^2/s^2
    "inflow_direction": [1.0, 0.0, 0.0],
    "drag_direction": [1.0, 0.0, 0.0],
    "lift_direction": [0.0, 0.0, 1.0],
    "ground": "moving floor not modelled (stationary ground plane, see paper)",
}

SOLVER: dict[str, Any] = {
    "solver": "OpenFOAM",
    "solver_version": "v2212",
    "application": "pimpleFoam",
    "turbulence_model": "SpalartAllmarasDDES",
    "turbulence_family": "hybrid RANS-LES (DDES, DEShybrid convection scheme, cubeRootVol delta)",
    "physics": "incompressible, Newtonian, isothermal, transient (backward ddt), time-averaged",
    "fidelity": "scale-resolving hybrid RANS-LES (HRLES), ~20M cells, ~80 CTU averaging",
    "fidelity_tier": "hrles",
    "mesh_cells_volume_approx": 20_000_000,
    "citation": PAPER_URL,
}

# Field name -> (units, description). Names are kept exactly as in the VTP files.
FIELDS: dict[str, tuple[str, str]] = {
    "pMean": ("m^2/s^2", "time-averaged kinematic pressure p/rho"),
    "static(p)_coeffMean": ("-", "time-averaged static pressure coefficient Cp"),
    "yPlusMean": ("-", "time-averaged wall y+"),
    "wallShearStressMean": ("m^2/s^2", "time-averaged kinematic wall shear stress vector"),
}
FIELD_ROLES: dict[str, str] = {
    "pressure": "pMean",
    "pressure_coefficient": "static(p)_coeffMean",
    "wall_shear_stress": "wallShearStressMean",
}
GEOMETRY_PARAM_UNITS = "mm"
# The parameter most directly tied to wake topology (Ahmed body drag crisis around 30 deg).
GENERALIZATION_PARAMETER = "slant-angle-degrees"


def run_id(i: int) -> str:
    return f"run_{i}"


def run_files(i: int) -> dict[str, str]:
    """Repo-relative paths of the files for run ``i`` keyed by role."""
    return {role: pattern.format(i=i) for role, pattern in RUN_FILES.items()}


def is_forbidden(repo_path: str) -> bool:
    name = repo_path.rsplit("/", 1)[-1]
    return any(pat in name for pat in FORBIDDEN_PATTERNS)
