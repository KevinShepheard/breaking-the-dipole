# Breaking the Dipole

A reproducible, adversarial analysis pipeline demonstrating that reported
Type Ia supernova “dipoles” are not robust cosmological signals, but arise from
higher-multipole leakage, inhomogeneous variance fields, and survey-conditioned
geometry.

This repository contains the exact code used to *kill the dipole* in
Pantheon+ / SH0ES-style SN Ia residual analyses, using nulls that actually test
sky–residual coupling.

The result is negative in the strongest possible sense:
after isolating the dipole-only channel and projecting out ℓ ≥ 2 structure,
the dipole is statistically indistinguishable from noise under adversary nulls.

The Hubble tension survives — but it survives as a **global monopole**, not a
directional flow.


# What this does (in plain terms)

The script:

- Separates mean-field anisotropy from variance-field anisotropy
- Quantifies harmonic power by ℓ instead of assuming “dipole dominance”
- Explicitly removes ℓ ≥ 2 contamination before testing dipole significance
- Tests the *right null*: z-stratified shuffles that break sky–residual coupling
- Treats SO(3) rotation nulls as secondary and documents why
- Diagnoses survey footprint and noise inhomogeneity effects directly

If a dipole survives this pipeline, it deserves serious attention.
In Pantheon+SH0ES, it does not.


# Core script

The analysis lives in a single public-facing script:

- `breaking-the-dipole.py`

It is written to be readable, auditable, and suitable for peer review.
All assumptions are stated in comments.
All outputs are saved as JSON and PNG artifacts.


# Typical usage

Baseline run:

python breaking-the-dipole.py 
–table data/pantheonplus/distance_moduli/Pantheon_SH0ES.dat 
–use-log10z 
–outdir out/mirror_break

Full sweep (recommended):

python breaking-the-dipole.py 
–table data/pantheonplus/distance_moduli/Pantheon_SH0ES.dat 
–use-log10z 
–full-sweep 
–outdir out/mirror_break

This produces three non-overwriting result sets:
- baseline
- survey-whitened
- rotation-null (secondary)


# Key outputs

Each run produces:

- `mirror_break.json` — complete numerical results and metadata
- `harmonic_power_by_l.png` — ℓ-by-ℓ power comparison (y, σ, |y|)
- `null_zshuffle_hist.png` — adversary null distribution
- `null_rotation_hist.png` — rotation null (if enabled)

These artifacts are designed to map one-to-one onto falsified assumptions
in the literature.


# The lethal methodological point

If a claimed dipole disappears under:
- ℓ ≥ 2 projection, and
- z-stratified shuffle nulls,

then it is not evidence for a cosmic flow, anisotropic expansion,
or new late-time physics.

It is evidence for **inhomogeneous noise interacting with survey geometry**.

This repository exists to make that distinction explicit and unavoidable.


# What this does NOT claim

- It does not resolve the Hubble tension
- It does not rule out new physics
- It does not claim ΛCDM is complete

It shows that *dipole-based explanations* of the tension are unsupported
once the analysis is done correctly.


# Reproducibility

- No proprietary data required
- Works directly on Pantheon+ / SH0ES tables
- Deterministic seeds
- All intermediate products saved
- No hidden preprocessing steps


# Citation

If you use this code or its results, please cite:

Kevin Shepheard, *Breaking the Dipole: Harmonic Leakage and Inhomogeneous Noise
in Type Ia Supernova Anisotropy Claims*, Zenodo, YEAR.

10.5281/zenodo.18228684


# License

MIT License

Copyright (c) 2026 Kevin Shepheard

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.