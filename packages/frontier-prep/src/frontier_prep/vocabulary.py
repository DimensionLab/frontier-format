# SPDX-License-Identifier: Apache-2.0
"""Canonical physics field vocabulary with a CGNS-style unit model.

Every canonical field carries ``DimensionalExponents`` (mass, length, time, temperature,
angle), a ``DataClass`` (``dimensional`` | ``nondimensional`` | ``normalized_by_dimensional``)
and a canonical SI unit string that ``pint`` understands. Solver names (``pMean``, ``U``,
``wallShearStressMean``, ``Pressure (Pa)``, ``static(p)_coeffMean``) map onto canonical names
through aliases; ambiguous names (``p``: static or kinematic pressure) come back as a
:class:`frontier_prep.op.Judgment` instead of a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

DataClass = str  # "dimensional" | "nondimensional" | "normalized_by_dimensional"


@dataclass(frozen=True)
class Exponents:
    """CGNS ``DimensionalExponents``: mass, length, time, temperature, angle."""

    M: float = 0
    L: float = 0
    T: float = 0
    Th: float = 0
    A: float = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "mass": self.M,
            "length": self.L,
            "time": self.T,
            "temperature": self.Th,
            "angle": self.A,
        }


@dataclass(frozen=True)
class CanonicalField:
    name: str
    unit: str  # canonical SI spelling for pint
    exponents: Exponents
    data_class: DataClass = "dimensional"
    channels: int | None = None  # None = any
    description: str = ""
    aliases: tuple[str, ...] = ()
    plausible: tuple[float, float] | None = None  # |value| range in canonical units (p99)
    related: dict[str, str] = field(
        default_factory=dict
    )  # e.g. {"kinematic": "kinematic_pressure"}


_P = Exponents(M=1, L=-1, T=-2)
_V = Exponents(L=1, T=-1)
_NONE = Exponents()

FIELDS: dict[str, CanonicalField] = {}


def _add(f: CanonicalField) -> None:
    FIELDS[f.name] = f


_add(
    CanonicalField(
        "pressure",
        "Pa",
        _P,
        channels=1,
        description="Static pressure (absolute or gauge; see meta.extra.fields.pressure.gauge)",
        aliases=(
            "p",
            "pMean",
            "Pressure",
            "Pressure (Pa)",
            "pressure_average",
            "static_pressure",
            "p_mean",
            "P",
        ),
        plausible=(0.0, 5e7),
        related={"kinematic": "kinematic_pressure", "coefficient": "pressure_coefficient"},
    )
)
_add(
    CanonicalField(
        "kinematic_pressure",
        "m**2/s**2",
        Exponents(L=2, T=-2),
        channels=1,
        description="Pressure divided by density (OpenFOAM incompressible p)",
        aliases=("p_kinematic", "p_rgh", "pMean_kinematic"),
        plausible=(0.0, 1e5),
        related={"dynamic": "pressure"},
    )
)
_add(
    CanonicalField(
        "pressure_coefficient",
        "",
        _NONE,
        "nondimensional",
        channels=1,
        description="Cp = (p - p_inf) / (0.5 rho U^2)",
        aliases=("Cp", "cp", "static(p)_coeff", "static(p)_coeffMean", "pressure_coeff", "cpMean"),
        plausible=(0.0, 3.0),
    )
)
_add(
    CanonicalField(
        "velocity",
        "m/s",
        _V,
        description="Velocity vector",
        aliases=("U", "UMean", "u", "Velocity", "Velocity (m/s)", "vel", "velocity_average", "V"),
        plausible=(0.0, 5e3),
    )
)
_add(
    CanonicalField(
        "velocity_magnitude",
        "m/s",
        _V,
        channels=1,
        aliases=("Umag", "magU", "|U|", "speed", "UMean_magnitude"),
        plausible=(0.0, 5e3),
    )
)
_add(
    CanonicalField(
        "wall_shear_stress",
        "Pa",
        _P,
        channels=3,
        description="Wall shear stress vector (traction)",
        aliases=(
            "wallShearStress",
            "wallShearStressMean",
            "tau_w",
            "WallShearStress",
            "wall_shear_stress_average",
            "tau_wall",
            "wss",
        ),
        plausible=(0.0, 1e5),
        related={"kinematic": "kinematic_wall_shear_stress"},
    )
)
_add(
    CanonicalField(
        "kinematic_wall_shear_stress",
        "m**2/s**2",
        Exponents(L=2, T=-2),
        channels=3,
        description="Wall shear stress over density (OpenFOAM incompressible wallShearStress)",
        aliases=("wallShearStress_kinematic",),
        plausible=(0.0, 1e3),
        related={"dynamic": "wall_shear_stress"},
    )
)
_add(
    CanonicalField(
        "skin_friction_coefficient",
        "",
        _NONE,
        "nondimensional",
        aliases=("Cf", "cf", "skinFriction", "skin_friction"),
        plausible=(0.0, 0.5),
    )
)
_add(
    CanonicalField(
        "y_plus",
        "",
        _NONE,
        "nondimensional",
        channels=1,
        aliases=("yPlus", "yPlusMean", "y+", "yplus"),
        plausible=(0.0, 1e4),
    )
)
_add(
    CanonicalField(
        "temperature",
        "K",
        Exponents(Th=1),
        channels=1,
        aliases=("T", "Temperature", "Temperature (K)", "TMean", "temp"),
        plausible=(1.0, 5e4),
    )
)
_add(
    CanonicalField(
        "density",
        "kg/m**3",
        Exponents(M=1, L=-3),
        channels=1,
        aliases=("rho", "rhoMean", "Density"),
        plausible=(1e-6, 3e4),
    )
)
_add(
    CanonicalField(
        "turbulent_viscosity",
        "m**2/s",
        Exponents(L=2, T=-1),
        channels=1,
        aliases=("nut", "nutMean", "nu_t", "mut", "turbulentViscosity"),
        plausible=(0.0, 1e2),
    )
)
_add(
    CanonicalField(
        "turbulent_kinetic_energy",
        "m**2/s**2",
        Exponents(L=2, T=-2),
        channels=1,
        aliases=("k", "kMean", "tke"),
        plausible=(0.0, 1e6),
    )
)
_add(
    CanonicalField(
        "turbulent_dissipation_rate",
        "m**2/s**3",
        Exponents(L=2, T=-3),
        channels=1,
        aliases=("epsilon", "epsilonMean"),
        plausible=(0.0, 1e12),
    )
)
_add(
    CanonicalField(
        "specific_dissipation_rate",
        "1/s",
        Exponents(T=-1),
        channels=1,
        aliases=("omega", "omegaMean"),
        plausible=(0.0, 1e12),
    )
)
_add(
    CanonicalField(
        "vorticity",
        "1/s",
        Exponents(T=-1),
        aliases=("vort", "omega_z", "curl_u", "Vorticity"),
        plausible=(0.0, 1e8),
    )
)
_add(
    CanonicalField(
        "heat_flux",
        "W/m**2",
        Exponents(M=1, T=-3),
        aliases=("q", "qdot", "wallHeatFlux", "heatFlux"),
        plausible=(0.0, 1e9),
    )
)
_add(
    CanonicalField(
        "displacement",
        "m",
        Exponents(L=1),
        aliases=("disp", "u_disp", "Displacement", "U1", "U2", "U3", "displacements"),
        plausible=(0.0, 1e2),
    )
)
_add(
    CanonicalField(
        "stress", "Pa", _P, aliases=("sigma", "Stress", "stress_tensor", "S"), plausible=(0.0, 1e10)
    )
)
_add(
    CanonicalField(
        "von_mises_stress",
        "Pa",
        _P,
        channels=1,
        aliases=("stress_vm", "vonMises", "von_mises", "mises", "sigma_vm", "S_mises"),
        plausible=(0.0, 1e10),
    )
)
_add(
    CanonicalField(
        "strain",
        "",
        _NONE,
        "nondimensional",
        aliases=("eps", "Strain", "E", "strain_tensor"),
        plausible=(0.0, 10.0),
    )
)
_add(
    CanonicalField(
        "effective_plastic_strain",
        "",
        _NONE,
        "nondimensional",
        channels=1,
        aliases=("PEEQ", "peeq", "plastic_strain", "eps_p"),
        plausible=(0.0, 10.0),
    )
)
_add(
    CanonicalField(
        "thickness",
        "m",
        Exponents(L=1),
        channels=1,
        aliases=("thick", "shell_thickness"),
        plausible=(0.0, 1.0),
    )
)
_add(
    CanonicalField(
        "mach_number",
        "",
        _NONE,
        "nondimensional",
        channels=1,
        aliases=("Ma", "Mach", "mach", "M"),
        plausible=(0.0, 30.0),
    )
)
_add(
    CanonicalField(
        "phase_fraction",
        "",
        _NONE,
        "nondimensional",
        channels=1,
        aliases=("alpha", "alpha.water", "alpha_water", "vof", "volume_fraction"),
        plausible=(0.0, 1.0),
    )
)
_add(
    CanonicalField(
        "total_pressure",
        "Pa",
        _P,
        channels=1,
        aliases=("p0", "ptot", "totalPressure", "total(p)"),
        plausible=(0.0, 5e7),
    )
)
_add(
    CanonicalField(
        "signed_distance",
        "m",
        Exponents(L=1),
        channels=1,
        aliases=("sdf", "distance", "wallDistance", "yWall"),
        plausible=(0.0, 1e3),
    )
)
_add(
    CanonicalField(
        "electric_potential",
        "V",
        Exponents(M=1, L=2, T=-3),
        channels=1,
        aliases=("phi_e", "potential", "voltage"),
    )
)
_add(CanonicalField("magnetic_field", "T", Exponents(M=1, T=-2), aliases=("B", "Bfield")))
_add(
    CanonicalField(
        "concentration",
        "mol/m**3",
        Exponents(L=-3),
        channels=1,
        aliases=("c", "conc"),
        plausible=(0.0, 1e6),
    )
)

#: alias (lowercased, non-alphanumerics stripped) -> canonical name; ambiguous aliases map
#: to several canonical names and become judgments
_ALIAS_INDEX: dict[str, list[str]] = {}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9+]", "", name.lower())


for _f in FIELDS.values():
    for _alias in (_f.name, *_f.aliases):
        _ALIAS_INDEX.setdefault(_norm(_alias), [])
        if _f.name not in _ALIAS_INDEX[_norm(_alias)]:
            _ALIAS_INDEX[_norm(_alias)].append(_f.name)

#: names whose dynamic/kinematic form depends on the solver: resolved by unit when known
AMBIGUOUS: dict[str, list[str]] = {
    "p": ["pressure", "kinematic_pressure"],
    "pmean": ["pressure", "kinematic_pressure"],
    "wallshearstress": ["wall_shear_stress", "kinematic_wall_shear_stress"],
    "wallshearstressmean": ["wall_shear_stress", "kinematic_wall_shear_stress"],
}

#: unit spellings seen in the wild -> pint spelling
UNIT_ALIASES: dict[str, str] = {
    "-": "",
    "": "",
    "none": "",
    "1": "",
    "dimensionless": "",
    "Pa": "Pa",
    "pa": "Pa",
    "N/m^2": "Pa",
    "N/m2": "Pa",
    "N/m**2": "Pa",
    "MPa": "MPa",
    "kPa": "kPa",
    "bar": "bar",
    "psi": "psi",
    "m^2/s^2": "m**2/s**2",
    "m2/s2": "m**2/s**2",
    "m^2/s": "m**2/s",
    "m2/s": "m**2/s",
    "m/s": "m/s",
    "km/h": "km/h",
    "mm": "mm",
    "cm": "cm",
    "m": "m",
    "in": "inch",
    "inch": "inch",
    "ft": "foot",
    "K": "K",
    "degC": "degC",
    "C": "degC",
    "°C": "degC",
    "kg/m^3": "kg/m**3",
    "kg/m3": "kg/m**3",
    "W/m^2": "W/m**2",
    "W/m2": "W/m**2",
    "1/s": "1/s",
    "s^-1": "1/s",
    "unknown": "unknown",
}


def normalize_unit(u: str | None) -> str:
    if u is None:
        return ""
    u = u.strip()
    return UNIT_ALIASES.get(u, u.replace("^", "**"))


@lru_cache(maxsize=1)
def ureg() -> Any:
    import pint

    reg = pint.UnitRegistry()
    reg.define("@alias kelvin = K")
    return reg


def dimensionality(unit: str) -> Any | None:
    """``pint`` dimensionality of a unit string (``None`` when unknown/unparseable)."""
    u = normalize_unit(unit)
    if u in ("", "unknown"):
        return None if u == "unknown" else ureg().dimensionless.dimensionality
    try:
        return ureg().parse_expression(u).dimensionality
    except Exception:  # noqa: BLE001 - pint raises many types
        return None


def exponents_of(unit: str) -> Exponents | None:
    d = dimensionality(unit)
    if d is None:
        return None
    return Exponents(
        M=float(d.get("[mass]", 0)),
        L=float(d.get("[length]", 0)),
        T=float(d.get("[time]", 0)),
        Th=float(d.get("[temperature]", 0)),
        A=0.0,
    )


@dataclass
class Mapping:
    source: str
    canonical: str | None
    confidence: float
    reason: str
    options: list[str] = field(default_factory=list)
    unit_compatible: bool | None = None


def map_name(name: str, units: str | None = None, channels: int | None = None) -> Mapping:
    """Propose a canonical name for a solver field name. Units (when known) disambiguate the
    dynamic/kinematic forms; a channel count that contradicts the canonical field lowers the
    confidence."""
    key = _norm(name)
    # strip common suffixes/prefixes: Mean, _avg, _average, _t0
    stripped = re.sub(r"(mean|average|avg|_t\d+)$", "", key)
    cands = _ALIAS_INDEX.get(key) or _ALIAS_INDEX.get(stripped) or []
    if key in AMBIGUOUS or stripped in AMBIGUOUS:
        cands = list(AMBIGUOUS.get(key, AMBIGUOUS.get(stripped, [])))
    if not cands:
        return Mapping(name, None, 0.0, "no alias matches", [])
    uexp = exponents_of(units) if units else None
    if uexp is not None and len(cands) > 1:
        compatible = [c for c in cands if FIELDS[c].exponents == uexp]
        if len(compatible) == 1:
            return Mapping(name, compatible[0], 0.95, f"alias + unit {units} match", cands, True)
    if len(cands) > 1:
        return Mapping(
            name,
            None,
            0.5,
            "ambiguous: dynamic or kinematic form; set the unit or decide",
            cands,
            None,
        )
    c = cands[0]
    cf = FIELDS[c]
    conf = 0.9 if key == _norm(c) or key in map(_norm, cf.aliases) else 0.75
    ok: bool | None = None
    if uexp is not None:
        ok = cf.exponents == uexp or cf.data_class == "nondimensional" and uexp == _NONE
        if not ok:
            conf = min(conf, 0.4)
    if channels is not None and cf.channels is not None and channels != cf.channels:
        conf = min(conf, 0.3)
    reason = "alias match" + (
        "" if ok is None else (", unit compatible" if ok else ", UNIT MISMATCH")
    )
    return Mapping(name, c, conf, reason, cands, ok)


def canonical(name: str) -> CanonicalField | None:
    return FIELDS.get(name)


def vocabulary_json() -> list[dict[str, Any]]:
    return [
        {
            "name": f.name,
            "unit": f.unit,
            "exponents": f.exponents.as_dict(),
            "data_class": f.data_class,
            "channels": f.channels,
            "description": f.description,
            "aliases": list(f.aliases),
            "plausible": list(f.plausible) if f.plausible else None,
            "related": dict(f.related),
        }
        for f in FIELDS.values()
    ]
