#!/usr/bin/env python3
"""
breaking-the-dipole.py

Breaking the Supernova Dipole:
A Reproducible, Adversarial Test of Claimed Late-Time Anisotropy
in Type Ia Supernova Residuals (Pantheon+ / SH0ES-style Tables)

Public release / peer-review intent
-----------------------------------
This script is written to be *publicly releasable* and *peer-review ready*.
It is not an exploratory notebook, but a surgical diagnostic designed to
explicitly test — and, where appropriate, falsify — common methodological
assumptions behind claimed large-scale anisotropy (“dipoles”) in Type Ia
supernova Hubble residuals.

The guiding principle is adversarial clarity: every statistic reported here
is paired with a null that actually breaks the claimed coupling, rather than
one that merely preserves it.

Scientific questions addressed
-------------------------------
This code tests four tightly scoped questions that recur throughout the
anisotropy literature:

(A) Mean-field anisotropy:
    Is there a statistically significant ℓ = 1 (dipole) component in SN
    residuals after removing the smooth Hubble relation μ(z)?

(B) Variance-field anisotropy:
    Does the *error structure itself* (σ or |residual|) carry higher-order
    angular power that can leak into low-ℓ fits?

(C) Harmonic leakage:
    Is the reported “dipole” robust when ℓ ≥ 2 modes are explicitly modeled
    and projected out, or is it a projection artifact of higher multipoles?

(D) Survey-conditioned geometry:
    Do per-survey noise scales and sky footprints imprint coherent angular
    structure that masquerades as cosmological anisotropy?

What makes this different from prior work
------------------------------------------
Many published dipole claims rely primarily on SO(3) rotations of sky
positions as their null test. Rotation preserves the relative angular
structure of the survey footprint and, for amplitude-like statistics, can be
nearly non-discriminating.

This script therefore treats *z-stratified residual shuffling* as the primary
adversary null. That null explicitly destroys sky–residual coupling while
preserving the redshift distribution and heteroscedastic noise structure.

If a claimed dipole survives this null *and* survives projection of ℓ ≥ 2
modes, the burden of interpretation changes. If it does not, the result is
consistent with selection geometry and inhomogeneous noise rather than a
physical bulk flow or anisotropic expansion.

Core outputs
------------
For a given SN table, the script reports:

- Dipole amplitude and significance on residuals y
- Harmonic power by ℓ for:
    • residual mean field y
    • variance field σ
    • absolute residual field |y|
- Power ratios (ℓ ≥ 2)/(ℓ = 1) to diagnose leakage
- “Narrow path” dipole-only amplitude after projecting out ℓ ≥ 2
- Empirical p-values for the dipole-only channel under:
    • z-stratified shuffle null (primary)
    • SO(3) rotation null (secondary; optional)
- Survey-level dipole decomposition (before/after per-survey subtraction)
- Optional per-survey scale whitening to expose noise-driven structure

Interpretive guidance (read before arguing)
-------------------------------------------
If a dipole signal:
  • weakens or vanishes after ℓ ≥ 2 projection, and
  • is statistically consistent with the z-stratified shuffle null,

then the correct interpretation is *not* a cosmological flow, but leakage
from higher-order angular structure tied to survey geometry and
inhomogeneous uncertainties.

This does *not* resolve the Hubble tension. It localizes it: the remaining
discrepancy must live in the global (monopole) calibration of the distance
ladder, not in directional late-time physics.

Data assumptions
----------------
The default configuration targets Pantheon+ / SH0ES-style whitespace-
delimited tables, with columns such as:

  RA, DEC, zHD,
  MU_SH0ES, MU_SH0ES_ERR_DIAG,
  IDSURVEY (optional but strongly recommended)

If a residual column (e.g. DeltaMu) is present, it is used directly.
Otherwise, residuals are constructed via a weighted polynomial fit to μ(z).

Reproducibility
---------------
- All random procedures are seed-controlled.
- All outputs are written to structured JSON plus publication-ready figures.
- The script supports a full three-variant sweep (baseline / whitened /
  rotation-null) in a single invocation to avoid accidental overwrites.

This file is intended to be cited, re-run, and adversarially tested.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import sph_harm_y
import matplotlib.pyplot as plt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


# -------------------------
# I/O + misc utils
# -------------------------
def ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", comment="#", engine="python")


def _to_f64(a: npt.NDArray[np.generic]) -> FloatArray:
    return np.asarray(a, dtype=np.float64)


def _to_i64(a: npt.NDArray[np.generic]) -> IntArray:
    return np.asarray(a, dtype=np.int64)


def _finite_mask(*arrs: FloatArray) -> BoolArray:
    m = np.ones(arrs[0].shape[0], dtype=np.bool_)
    for a in arrs:
        m &= np.isfinite(a)
    return m


# -------------------------
# Geometry
# -------------------------
def unitvec_from_radec(ra_deg: FloatArray, dec_deg: FloatArray) -> FloatArray:
    ra = _to_f64(np.deg2rad(ra_deg))
    dec = _to_f64(np.deg2rad(dec_deg))
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    n = np.stack([x, y, z], axis=1).astype(np.float64, copy=False)
    norm = _to_f64(np.linalg.norm(n, axis=1))
    return _to_f64(n / norm[:, None])


def unitvec_to_radec(nvec: FloatArray) -> Tuple[FloatArray, FloatArray]:
    x = _to_f64(nvec[:, 0])
    y = _to_f64(nvec[:, 1])
    z = _to_f64(nvec[:, 2])
    dec = _to_f64(np.arcsin(np.clip(z, -1.0, 1.0)))
    ra = _to_f64(np.arctan2(y, x))
    ra = _to_f64(np.where(ra < 0.0, ra + 2.0 * math.pi, ra))
    return _to_f64(np.rad2deg(ra)), _to_f64(np.rad2deg(dec))


def radec_to_theta_phi(ra_deg: FloatArray, dec_deg: FloatArray) -> Tuple[FloatArray, FloatArray]:
    # SciPy real/complex spherical harmonics use theta in [0, pi] (colatitude),
    # phi in [0, 2pi) (longitude).
    ra = _to_f64(np.deg2rad(ra_deg))
    dec = _to_f64(np.deg2rad(dec_deg))
    phi = ra
    theta = _to_f64(0.5 * math.pi - dec)
    return theta, phi


# -------------------------
# Regression
# -------------------------
def weighted_lstsq(X: FloatArray, y: FloatArray, w: FloatArray) -> Tuple[FloatArray, FloatArray]:
    """
    Weighted least squares with an SVD-stable solve and covariance estimate.

    Returns:
      beta: (K,)
      cov : (K,K)  (uses pinv if normal matrix is singular/ill-conditioned)
    """
    sw = _to_f64(np.sqrt(w))
    Xw = _to_f64(X * sw[:, None])
    yw = _to_f64(y * sw)

    # Robust solve (SVD / lstsq)
    beta = _to_f64(np.linalg.lstsq(Xw, yw, rcond=None)[0])

    resid = _to_f64(y - X @ beta)
    dof = max(1, int(y.size - X.shape[1]))
    s2 = float(np.sum(w * resid * resid) / float(dof))

    xtx = _to_f64(Xw.T @ Xw)
    cov = _to_f64(np.linalg.pinv(xtx) * s2)
    return beta, cov


def polyfit_mu_of_z(
    z: FloatArray,
    mu: FloatArray,
    sigma: FloatArray,
    degree: int,
    use_log10z: bool,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """
    Weighted polynomial fit mu(x), return (mu_hat, delta_mu, beta_poly).
    """
    x = _to_f64(np.log10(z) if use_log10z else z)
    X = _to_f64(np.vander(x, N=degree + 1, increasing=True))
    w = _to_f64(np.where(sigma > 0.0, 1.0 / (sigma * sigma), 0.0))
    beta, _ = weighted_lstsq(X, mu, w)
    mu_hat = _to_f64(X @ beta)
    delta = _to_f64(mu - mu_hat)
    return mu_hat, delta, beta


# -------------------------
# Real spherical harmonic basis
# -------------------------
@dataclass(frozen=True)
class BasisIndex:
    l: int
    m: int
    kind: str  # "m0", "cos", "sin"


def real_sph_harm_design(ra_deg: FloatArray, dec_deg: FloatArray, lmax: int) -> Tuple[FloatArray, List[BasisIndex]]:
    """
    Real spherical harmonic basis up to lmax, dropping monopole (l=0).
    Columns:
      for l=1..lmax:
        m=0: Y_l0
        m=1..l: sqrt(2) Re(Y_lm), sqrt(2) Im(Y_lm)

    Uses scipy.special.sph_harm_y(m, l, theta, phi).
    """
    theta, phi = radec_to_theta_phi(ra_deg, dec_deg)
    cols: List[FloatArray] = []
    idx: List[BasisIndex] = []

    for l in range(1, lmax + 1):
        y0 = _to_f64(np.real(sph_harm_y(0, l, theta, phi)))
        cols.append(y0)
        idx.append(BasisIndex(l=l, m=0, kind="m0"))

        for m in range(1, l + 1):
            ylm = sph_harm_y(m, l, theta, phi)
            re = _to_f64(np.real(ylm))
            im = _to_f64(np.imag(ylm))
            cols.append(_to_f64(math.sqrt(2.0) * re))
            idx.append(BasisIndex(l=l, m=m, kind="cos"))
            cols.append(_to_f64(math.sqrt(2.0) * im))
            idx.append(BasisIndex(l=l, m=m, kind="sin"))

    X = _to_f64(np.column_stack(cols))
    return X, idx


def harmonic_power_by_l(beta: FloatArray, idx: Sequence[BasisIndex]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for j in range(beta.size):
        l = idx[j].l
        out[l] = out.get(l, 0.0) + float(beta[j] * beta[j])
    return out


def mask_for_l(idx: Sequence[BasisIndex], l_value: int) -> BoolArray:
    return np.asarray([bi.l == l_value for bi in idx], dtype=np.bool_)


def mask_for_l_ge(idx: Sequence[BasisIndex], l_min: int) -> BoolArray:
    return np.asarray([bi.l >= l_min for bi in idx], dtype=np.bool_)


# -------------------------
# Dipole fit (vector regression)
# -------------------------
@dataclass(frozen=True)
class DipoleFit:
    b_vec: FloatArray  # (3,)
    amp: float
    amp_se: float
    amp_z: float


def fit_dipole(ra_deg: FloatArray, dec_deg: FloatArray, y: FloatArray, sigma: FloatArray) -> DipoleFit:
    """
    Weighted LS for y = a0 + b·n.
    We solve for [a0, bx, by, bz].
    """
    n = unitvec_from_radec(ra_deg, dec_deg)
    X = _to_f64(np.column_stack([np.ones(n.shape[0], dtype=np.float64), n]))
    w = _to_f64(np.where(sigma > 0.0, 1.0 / (sigma * sigma), 0.0))
    beta, cov = weighted_lstsq(X, y, w)
    b = _to_f64(beta[1:4])
    amp = float(np.linalg.norm(b))

    # Delta-method SE for ||b||: Var(amp) ≈ (u^T Cov_b u) where u=b/||b||
    cov_b = _to_f64(cov[1:4, 1:4])
    if amp > 0.0:
        u = _to_f64(b / amp)
        amp_var = float(u @ cov_b @ u)
        amp_se = float(math.sqrt(max(0.0, amp_var)))
    else:
        amp_se = float("nan")

    amp_z = float(amp / amp_se) if (amp_se > 0.0 and np.isfinite(amp_se)) else float("nan")
    return DipoleFit(b_vec=b, amp=amp, amp_se=amp_se, amp_z=amp_z)


# -------------------------
# Null generators
# -------------------------
def z_stratified_shuffle_indices(z: FloatArray, n_bins: int, rng: np.random.Generator) -> IntArray:
    """
    Returns a permutation index array that shuffles *within* z-quantile bins.
    """
    n = int(z.size)
    edges = np.quantile(z, np.linspace(0.0, 1.0, n_bins + 1))
    perm = np.arange(n, dtype=np.int64)

    for bi in range(n_bins):
        lo = float(edges[bi])
        hi = float(edges[bi + 1])
        if bi == n_bins - 1:
            m = (z >= lo) & (z <= hi)
        else:
            m = (z >= lo) & (z < hi)
        idx = np.nonzero(m)[0].astype(np.int64, copy=False)
        if idx.size <= 1:
            continue
        rng.shuffle(idx)
        perm[np.nonzero(m)[0]] = idx
    return _to_i64(perm)


def random_rotation_matrix(rng: np.random.Generator) -> FloatArray:
    """
    Uniform random rotation via random unit quaternion.
    """
    u1, u2, u3 = rng.random(3).tolist()
    q1 = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    q2 = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)

    R = np.array(
        [
            [1 - 2 * (q3 * q3 + q4 * q4), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)],
            [2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2 * q2 + q4 * q4), 2 * (q3 * q4 - q1 * q2)],
            [2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)],
        ],
        dtype=np.float64,
    )
    return _to_f64(R)


# -------------------------
# Survey tools
# -------------------------
def survey_scale_whiten(
    y: FloatArray,
    sigma: FloatArray,
    survey: IntArray,
    min_n: int,
    robust: bool,
) -> Tuple[FloatArray, FloatArray, Dict[str, float]]:
    """
    Per-survey scale normalization: y <- y / s, sigma <- sigma / s.

    robust=True uses MAD-based scale; robust=False uses std.
    Returns (y_w, sigma_w, scales_by_survey).
    """
    y_out = y.copy()
    s_out = sigma.copy()

    scales: Dict[str, float] = {}
    for sid in np.unique(survey).tolist():
        ms = survey == int(sid)
        ns = int(np.sum(ms))
        if ns < int(min_n):
            continue
        ys = y_out[ms]
        if robust:
            med = float(np.median(ys))
            mad = float(np.median(np.abs(ys - med)))
            # Normal consistency factor for Gaussian: 1.4826 * MAD
            sc = float(1.4826 * mad)
        else:
            sc = float(np.std(ys, ddof=1))
        if not np.isfinite(sc) or sc <= 0.0:
            continue
        y_out[ms] = _to_f64(y_out[ms] / sc)
        s_out[ms] = _to_f64(s_out[ms] / sc)
        scales[str(int(sid))] = sc

    return _to_f64(y_out), _to_f64(s_out), scales


def survey_decomposed_dipole(
    ra: FloatArray,
    dec: FloatArray,
    y: FloatArray,
    sigma: FloatArray,
    survey: IntArray,
    min_n: int,
) -> Dict[str, object]:
    """
    Compute:
      global_before dipole on y
      per-survey dipoles (for surveys with n>=min_n)
      global_after on residual y' = y - sum_s I_s * (b_s · n)  (removes each survey's fitted dipole field)
    """
    global_before = fit_dipole(ra, dec, y, sigma)

    nvec = unitvec_from_radec(ra, dec)
    y_corr = y.copy()

    survey_dipoles: Dict[str, object] = {}
    for sid in np.unique(survey).tolist():
        ms = survey == int(sid)
        ns = int(np.sum(ms))
        if ns < int(min_n):
            continue
        fs = fit_dipole(ra[ms], dec[ms], y[ms], sigma[ms])
        # subtract the fitted survey dipole field from those points
        y_corr[ms] = _to_f64(y_corr[ms] - (nvec[ms] @ fs.b_vec))
        survey_dipoles[str(int(sid))] = {
            "n": ns,
            "b_vec": [float(fs.b_vec[0]), float(fs.b_vec[1]), float(fs.b_vec[2])],
            "amp": float(fs.amp),
            "amp_se": float(fs.amp_se),
            "amp_z": float(fs.amp_z),
        }

    global_after = fit_dipole(ra, dec, y_corr, sigma)

    out: Dict[str, object] = {
        "survey_dipoles": survey_dipoles,
        "global_before": {
            "n": int(y.size),
            "b_vec": [float(global_before.b_vec[0]), float(global_before.b_vec[1]), float(global_before.b_vec[2])],
            "amp": float(global_before.amp),
            "amp_se": float(global_before.amp_se),
            "amp_z": float(global_before.amp_z),
        },
        "global_after": {
            "n": int(y.size),
            "b_vec": [float(global_after.b_vec[0]), float(global_after.b_vec[1]), float(global_after.b_vec[2])],
            "amp": float(global_after.amp),
            "amp_se": float(global_after.amp_se),
            "amp_z": float(global_after.amp_z),
        },
    }
    return out


# -------------------------
# Mirror-break core: harmonic fit + narrow path + nulls
# -------------------------
@dataclass(frozen=True)
class NarrowPathResult:
    total_amp: float
    dipole_amp: float
    dipole_beta: FloatArray
    power_by_l: Dict[int, float]


def fit_harmonics(ra: FloatArray, dec: FloatArray, y: FloatArray, sigma: FloatArray, lmax: int) -> Tuple[FloatArray, List[BasisIndex]]:
    X, idx = real_sph_harm_design(ra, dec, lmax=lmax)
    w = _to_f64(np.where(sigma > 0.0, 1.0 / (sigma * sigma), 0.0))
    beta, _ = weighted_lstsq(X, y, w)
    return beta, idx


def narrow_path(ra: FloatArray, dec: FloatArray, y: FloatArray, sigma: FloatArray, lmax: int, remove_l_ge: int) -> NarrowPathResult:
    beta, idx = fit_harmonics(ra, dec, y, sigma, lmax=lmax)
    total_amp = float(np.linalg.norm(beta))
    m_l1 = mask_for_l(idx, 1)
    dip_beta = _to_f64(beta[m_l1])
    dip_amp = float(np.linalg.norm(dip_beta))

    # Optionally "remove l>=remove_l_ge" is implemented as a reporting decision:
    # the dipole-only amplitude is the ℓ=1 component, i.e., what remains when ℓ>=2 are removed.
    # (We fit all harmonics up to lmax, then read out the ℓ=1 block.)
    power = harmonic_power_by_l(beta, idx)
    return NarrowPathResult(total_amp=total_amp, dipole_amp=dip_amp, dipole_beta=dip_beta, power_by_l=power)


def empirical_p_ge(null_vals: FloatArray, obs: float) -> float:
    if null_vals.size <= 0:
        return float("nan")
    return float(np.mean(null_vals >= float(obs)))


# -------------------------
# Plotting helpers
# -------------------------
def plot_harmonic_power(power_y: Dict[int, float],
                         power_sig: Dict[int, float],
                         power_abs: Dict[int, float],
                         outpath: Path) -> None:
    ls = sorted(set(power_y.keys()) | set(power_sig.keys()) | set(power_abs.keys()))
    py = [power_y.get(l, 0.0) for l in ls]
    ps = [power_sig.get(l, 0.0) for l in ls]
    pa = [power_abs.get(l, 0.0) for l in ls]

    plt.figure()
    plt.yscale("log")
    plt.plot(ls, py, marker="o", label="residual y")
    plt.plot(ls, ps, marker="o", label="sigma")
    plt.plot(ls, pa, marker="o", label="|y|")
    plt.xlabel("harmonic order ℓ")
    plt.ylabel("power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_null_hist(null_vals: FloatArray,
                   obs: float,
                   xlabel: str,
                   outpath: Path) -> None:
    plt.figure()
    plt.hist(null_vals, bins=50, density=True, alpha=0.7)
    plt.axvline(obs, linestyle="--", linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Mirror-break suite: kill dipole claims by isolating leakage channels.")
    ap.add_argument("--table", required=True)
    ap.add_argument("--ra-col", default="RA")
    ap.add_argument("--dec-col", default="DEC")
    ap.add_argument("--z-col", default="zHD")

    ap.add_argument("--resid-col", default="DeltaMu")
    ap.add_argument("--mu-col", default="MU_SH0ES")
    ap.add_argument("--sigma-col", default="MU_SH0ES_ERR_DIAG")

    ap.add_argument("--use-log10z", action="store_true")
    ap.add_argument("--poly-degree", type=int, default=5)

    ap.add_argument("--survey-col", default="IDSURVEY")
    ap.add_argument("--min-survey-n", type=int, default=40)

    ap.add_argument("--lmax", type=int, default=4)
    ap.add_argument("--remove-l-ge", type=int, default=2)

    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--n-iter", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--do-rotation-null", action="store_true", help="Compute SO(3) rotation null (interpret with caution).")
    ap.add_argument("--do-survey-scale-whiten", action="store_true", help="Per-survey scale normalization (y and sigma).")
    ap.add_argument("--whiten-robust", action="store_true", help="Use MAD-based robust per-survey scale.")

    ap.add_argument("--outdir", default="out/mirror_break")
    ap.add_argument("--full-sweep", action="store_true",
                    help="Run baseline, survey-whitened, and rotation-null variants in one invocation.")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    ensure_outdir(outdir)

    df = read_table(Path(args.table))

    def run_variant(outdir_variant: Path,
                    do_whiten: bool,
                    do_rot: bool) -> None:
        ensure_outdir(outdir_variant)

        ra = _to_f64(pd.to_numeric(df[args.ra_col], errors="coerce").to_numpy(dtype=float))
        dec = _to_f64(pd.to_numeric(df[args.dec_col], errors="coerce").to_numpy(dtype=float))
        z = _to_f64(pd.to_numeric(df[args.z_col], errors="coerce").to_numpy(dtype=float))
        sigma = _to_f64(pd.to_numeric(df[args.sigma_col], errors="coerce").to_numpy(dtype=float))

        # y target:
        # - if resid-col exists: use it
        # - else compute DeltaMu from mu(z) polynomial fit
        used_resid_mode: str
        mu_hat: FloatArray | None = None
        beta_poly: FloatArray | None = None

        if args.resid_col in df.columns:
            y_raw = _to_f64(pd.to_numeric(df[args.resid_col], errors="coerce").to_numpy(dtype=float))
            used_resid_mode = "from_resid_col"
        else:
            mu = _to_f64(pd.to_numeric(df[args.mu_col], errors="coerce").to_numpy(dtype=float))
            m0 = _finite_mask(ra, dec, z, mu, sigma) & (z > 0.0) & (sigma > 0.0)
            ra, dec, z, mu, sigma = ra[m0], dec[m0], z[m0], mu[m0], sigma[m0]
            mu_hat, y_raw, beta_poly = polyfit_mu_of_z(z, mu, sigma, degree=int(args.poly_degree), use_log10z=bool(args.use_log10z))
            used_resid_mode = "computed_from_mu(z)"

        # If resid-col path: mask now
        if used_resid_mode == "from_resid_col":
            y_raw = _to_f64(y_raw)
            m = _finite_mask(ra, dec, z, y_raw, sigma) & (z > 0.0) & (sigma > 0.0)
            ra, dec, z, y_raw, sigma = ra[m], dec[m], z[m], y_raw[m], sigma[m]
            # mu_hat/beta_poly remain None in this mode

        y = _to_f64(y_raw)
        n = int(y.size)

        # Survey column (optional)
        survey_present = args.survey_col in df.columns
        survey: IntArray | None = None
        if survey_present and used_resid_mode == "from_resid_col":
            sraw = _to_f64(pd.to_numeric(df[args.survey_col], errors="coerce").to_numpy(dtype=float))
            # Align using the same mask as y-mode
            # We must recompute the mask exactly:
            y_full = _to_f64(pd.to_numeric(df[args.resid_col], errors="coerce").to_numpy(dtype=float))
            ra_full = _to_f64(pd.to_numeric(df[args.ra_col], errors="coerce").to_numpy(dtype=float))
            dec_full = _to_f64(pd.to_numeric(df[args.dec_col], errors="coerce").to_numpy(dtype=float))
            z_full = _to_f64(pd.to_numeric(df[args.z_col], errors="coerce").to_numpy(dtype=float))
            sig_full = _to_f64(pd.to_numeric(df[args.sigma_col], errors="coerce").to_numpy(dtype=float))
            m = _finite_mask(ra_full, dec_full, z_full, y_full, sig_full) & (z_full > 0.0) & (sig_full > 0.0)
            s = _to_i64(np.round(sraw[m]).astype(np.int64, copy=False))
            survey = s
        elif survey_present and used_resid_mode == "computed_from_mu(z)":
            # We already filtered arrays with m0 inside computed mode; we need survey aligned with that filter.
            sraw = _to_f64(pd.to_numeric(df[args.survey_col], errors="coerce").to_numpy(dtype=float))
            mu_full = _to_f64(pd.to_numeric(df[args.mu_col], errors="coerce").to_numpy(dtype=float))
            ra_full = _to_f64(pd.to_numeric(df[args.ra_col], errors="coerce").to_numpy(dtype=float))
            dec_full = _to_f64(pd.to_numeric(df[args.dec_col], errors="coerce").to_numpy(dtype=float))
            z_full = _to_f64(pd.to_numeric(df[args.z_col], errors="coerce").to_numpy(dtype=float))
            sig_full = _to_f64(pd.to_numeric(df[args.sigma_col], errors="coerce").to_numpy(dtype=float))
            m0 = _finite_mask(ra_full, dec_full, z_full, mu_full, sig_full) & (z_full > 0.0) & (sig_full > 0.0)
            s = _to_i64(np.round(sraw[m0]).astype(np.int64, copy=False))
            survey = s

        # Optional survey-scale whitening
        whiten_scales: Dict[str, float] | None = None
        if do_whiten:
            if survey is None:
                raise ValueError("--do-survey-scale-whiten requires a valid survey column present in the table.")
            y, sigma, whiten_scales = survey_scale_whiten(
                y=y,
                sigma=sigma,
                survey=survey,
                min_n=int(args.min_survey_n),
                robust=bool(args.whiten_robust),
            )

        # Dipole on y (mean residual field)
        dip = fit_dipole(ra, dec, y, sigma)

        # Harmonics up to lmax
        beta_y, idx = fit_harmonics(ra, dec, y, sigma, lmax=int(args.lmax))
        power_y = harmonic_power_by_l(beta_y, idx)

        # Variance coupling fields
        ones = _to_f64(np.ones_like(sigma))
        beta_sig, idx_sig = fit_harmonics(ra, dec, sigma, ones, lmax=int(args.lmax))
        power_sig = harmonic_power_by_l(beta_sig, idx_sig)

        beta_abs, idx_abs = fit_harmonics(ra, dec, _to_f64(np.abs(y)), ones, lmax=int(args.lmax))
        power_abs = harmonic_power_by_l(beta_abs, idx_abs)

        # Generate harmonic power plot
        plot_harmonic_power(
            power_y,
            power_sig,
            power_abs,
            outdir_variant / "harmonic_power_by_l.png",
        )

        # Power ratio (l>=2)/(l=1)
        def power_ratio_lge2_over_l1(p: Dict[int, float]) -> float:
            p1 = float(p.get(1, 0.0))
            pge2 = float(sum(v for k, v in p.items() if k >= 2))
            return float(pge2 / p1) if p1 > 0.0 else float("inf")

        ratio_y = power_ratio_lge2_over_l1(power_y)
        ratio_sig = power_ratio_lge2_over_l1(power_sig)
        ratio_abs = power_ratio_lge2_over_l1(power_abs)

        # Narrow path readout
        np_res = narrow_path(
            ra=ra,
            dec=dec,
            y=y,
            sigma=sigma,
            lmax=int(args.lmax),
            remove_l_ge=int(args.remove_l_ge),
        )
        obs_dip_amp = float(np_res.dipole_amp)

        # Nulls on dipole-only amplitude
        rng = np.random.default_rng(int(args.seed))

        # (1) z-stratified shuffle null (recommended)
        null_z = np.zeros(int(args.n_iter), dtype=np.float64)
        for i in range(int(args.n_iter)):
            perm = z_stratified_shuffle_indices(z=z, n_bins=int(args.n_bins), rng=rng)
            y_perm = _to_f64(y[perm])
            # Fit harmonics and read l=1 amplitude (dipole-only channel)
            beta_p, idx_p = fit_harmonics(ra, dec, y_perm, sigma, lmax=int(args.lmax))
            m_l1 = mask_for_l(idx_p, 1)
            null_z[i] = float(np.linalg.norm(beta_p[m_l1]))

        p_z = empirical_p_ge(null_z, obs_dip_amp)

        plot_null_hist(
            null_z,
            obs_dip_amp,
            xlabel="dipole-only amplitude (z-shuffle null)",
            outpath=outdir_variant / "null_zshuffle_hist.png",
        )

        # (2) rotation null (optional, interpret with caution)
        p_rot: float | None = None
        null_rot_path: str | None = None
        if do_rot:
            null_rot = np.zeros(int(args.n_iter), dtype=np.float64)
            nvec = unitvec_from_radec(ra, dec)
            for i in range(int(args.n_iter)):
                R = random_rotation_matrix(rng)
                n_rot = _to_f64((R @ nvec.T).T)
                ra_r, dec_r = unitvec_to_radec(n_rot)
                beta_r, idx_r = fit_harmonics(ra_r, dec_r, y, sigma, lmax=int(args.lmax))
                m_l1 = mask_for_l(idx_r, 1)
                null_rot[i] = float(np.linalg.norm(beta_r[m_l1]))
            p_rot = empirical_p_ge(null_rot, obs_dip_amp)
            null_rot_path = str(outdir_variant / "null_dipole_only_rot.npy")
            np.save(outdir_variant / "null_dipole_only_rot.npy", null_rot)
            plot_null_hist(
                null_rot,
                obs_dip_amp,
                xlabel="dipole-only amplitude (SO(3) rotation null)",
                outpath=outdir_variant / "null_rotation_hist.png",
            )

        # Save z-null samples too
        np.save(outdir_variant / "null_dipole_only_zshuffle.npy", null_z)

        # Survey decomposition (optional)
        survey_decomp: Dict[str, object] | None = None
        if survey is not None:
            survey_decomp = survey_decomposed_dipole(
                ra=ra,
                dec=dec,
                y=y,
                sigma=sigma,
                survey=survey,
                min_n=int(args.min_survey_n),
            )

        out: Dict[str, object] = {
            "table": str(Path(args.table).resolve()),
            "n": int(n),
            "columns": {
                "ra": str(args.ra_col),
                "dec": str(args.dec_col),
                "z": str(args.z_col),
                "sigma": str(args.sigma_col),
                "resid_mode": used_resid_mode,
                "resid_col": str(args.resid_col),
                "mu_col": str(args.mu_col),
                "survey_col": str(args.survey_col) if survey is not None else None,
            },
            "settings": {
                "use_log10z": bool(args.use_log10z),
                "poly_degree": int(args.poly_degree),
                "lmax": int(args.lmax),
                "remove_l_ge": int(args.remove_l_ge),
                "n_bins": int(args.n_bins),
                "n_iter": int(args.n_iter),
                "seed": int(args.seed),
                "do_rotation_null": bool(do_rot),
                "do_survey_scale_whiten": bool(do_whiten),
                "whiten_robust": bool(args.whiten_robust),
                "min_survey_n": int(args.min_survey_n),
            },
            "mu_of_z_fit": {
                "beta_poly": beta_poly.tolist() if beta_poly is not None else None,
            },
            "dipole_on_y": {
                "amp": float(dip.amp),
                "amp_se": float(dip.amp_se),
                "amp_z": float(dip.amp_z),
                "b_vec": [float(dip.b_vec[0]), float(dip.b_vec[1]), float(dip.b_vec[2])],
            },
            "harmonics": {
                "power_by_l_y": {str(k): float(v) for k, v in power_y.items()},
                "power_by_l_sigma": {str(k): float(v) for k, v in power_sig.items()},
                "power_by_l_abs_y": {str(k): float(v) for k, v in power_abs.items()},
                "power_ratio_lge2_over_l1": {
                    "y": float(ratio_y),
                    "sigma": float(ratio_sig),
                    "abs_y": float(ratio_abs),
                },
            },
            "narrow_path": {
                "total_amp": float(np_res.total_amp),
                "dipole_only_amp": float(obs_dip_amp),
                "nulls": {
                    "z_stratified_shuffle": {
                        "p_empirical": float(p_z),
                        "null_path": str(outdir_variant / "null_dipole_only_zshuffle.npy"),
                    },
                    "so3_rotation": {
                        "enabled": bool(do_rot),
                        "p_empirical": float(p_rot) if p_rot is not None else None,
                        "null_path": null_rot_path,
                        "interpretation_warning": "Rotation is often non-discriminating for amplitude-like sky fits; treat as secondary.",
                    },
                },
            },
            "survey": {
                "whiten_scales": whiten_scales,
                "decomposition": survey_decomp,
            },
            "lethal_suggestion": (
                "If a paper's main null is SO(3) sky rotation for dipole amplitude, "
                "re-run with z-stratified shuffles (break sky–residual coupling while preserving z-structure) "
                "and report the dipole-only channel after projecting out ℓ>=2."
            ),
        }

        out_path = outdir_variant / "mirror_break.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        # Console readout
        print(f"Wrote: {out_path}")
        print("")
        print("Key readouts:")
        print(f"  N={n}")
        print(f"  Dipole(y): amp={dip.amp:.6e} ± {dip.amp_se:.6e}  z={dip.amp_z:.3f}")
        print(f"  Harmonic power ratio (ℓ>=2)/(ℓ=1) on y      : {ratio_y:.6e}")
        print(f"  Harmonic power ratio (ℓ>=2)/(ℓ=1) on sigma  : {ratio_sig:.6e}")
        print(f"  Harmonic power ratio (ℓ>=2)/(ℓ=1) on |y|    : {ratio_abs:.6e}")
        print("")
        print("NARROW PATH (dipole-only channel after projecting out ℓ>=2):")
        print(f"  Total harmonic amplitude: {np_res.total_amp:.6e}")
        print(f"  Dipole-only amplitude   : {obs_dip_amp:.6e}")
        print("")
        print("Nulls on dipole-only amplitude:")
        print(f"  z-stratified shuffle (recommended): p = {p_z:.6e}")
        if do_rot:
            assert p_rot is not None
            print(f"  SO(3) rotation (secondary)        : p = {p_rot:.6e}")
        else:
            print("  SO(3) rotation (secondary)        : (disabled)")

        if do_whiten:
            print("")
            print("Survey-scale whitening: enabled")
            print(f"  robust={bool(args.whiten_robust)}  min_survey_n={int(args.min_survey_n)}")
        if survey_decomp is not None:
            gb = survey_decomp["global_before"]
            ga = survey_decomp["global_after"]
            assert isinstance(gb, dict) and isinstance(ga, dict)
            print("")
            print("Survey decomposition (dipole on y):")
            print(f"  global BEFORE: amp={float(gb['amp']):.6e}  z={float(gb['amp_z']):.3f}")
            print(f"  global AFTER : amp={float(ga['amp']):.6e}  z={float(ga['amp_z']):.3f}")

    if args.full_sweep:
        run_variant(outdir / "baseline", do_whiten=False, do_rot=False)
        run_variant(outdir / "whitened", do_whiten=True, do_rot=False)
        run_variant(outdir / "with_rotation", do_whiten=False, do_rot=True)
    else:
        run_variant(outdir, do_whiten=bool(args.do_survey_scale_whiten),
                    do_rot=bool(args.do_rotation_null))


if __name__ == "__main__":
    main()