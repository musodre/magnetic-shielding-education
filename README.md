# Magnetic shielding teaching software

Locally prepared version 1.0.0, 3 October 2026. This package implements classical exact magnetostatic fields for a concentric spherical shell and an infinite transverse cylindrical shell, with proposed undergraduate activities. It contains no student-study data. No public release or DOI is asserted.

## Origin and revised repository

This revised resource accompanies manuscript EJP-110956, *Magnetic shielding as a motivational tool in teaching classical electromagnetic theory*, by Murilo S. Marques, Rafael R. Marques, Jonatan J. da Silva and Edward F. de Almeida Júnior (Universidade Federal do Oeste da Bahia).

The original manuscript identified [xedwfe/magneticshielding](https://github.com/xedwfe/magneticshielding) as its code repository. The revised resource is intended for a separate repository named `magnetic-shielding-education`, following the authors' decision to preserve the original repository. This package does not import the original Git commit history or modify that repository. Its URL records provenance; the current remote contents have not been inspected in this preparation.

The revised code and documentation correct the geometry comparison, make B/H conventions and field magnitude scales explicit, provide executable activities and approximation-error checks, and delimit the radial solver's verification. The exact shielding laws are classical. The classroom sequence is proposed and has not been evaluated with students.

## Run

Python 3.9+, NumPy and Matplotlib:

```bash
python -m pip install -r requirements.txt
python generate_revision_figures.py
python classroom_activities.py
python verify_cavity_uniformity.py
```

Run scripts from this directory. Outputs are written beside the scripts and overwritten on rerun. Preserve renamed copies when comparing configurations.

```bash
python classroom_activities.py --mu 100 --k 0.8 --h0 2 --geometry cylinder
```

`mu >= 1`, `0 < k < 1`, `h0 > 0`; all must be finite. Lengths use b=1, a=k. Single-configuration figure, interface CSV and classroom JSON follow these options; sweep figures, CSVs and answer tables use fixed reference parameters. Expected tables and explanations are in the accompanying Supplementary Material II.

## Model and verification

The shell is linear, homogeneous and isotropic, with vacuum cavity and exterior and a uniform static applied field along x. Finite ends, apertures, saturation, hysteresis and conductive time dependence are excluded. The field maps evaluate exact expressions; line count does not represent magnitude or flux.

SF_sphere = 1 + (2/9)(1-k^3)(mu-1)^2/mu.
SF_cylinder = 1 + (1/4)(1-k^2)(mu-1)^2/mu.
The exact geometric crossing is k=(1+sqrt(33))/16, approximately 0.421535. The cylinder has greater SF below it and the sphere above it for positive mu different from 1.

Validation covers interfaces, SF equivalence, geometric ordering, limits and H0 scaling. The radial finite-volume script assumes angular order one and verifies a truncated cylinder problem. Mesh convergence does not remove its separate far-boundary effect (about 15.2% for the default configuration). It is not an unrestricted 2-D simulation or a proof for all modes.

## License and provenance

MIT applies to these scripts and accompanying software documentation; see LICENSE. NumPy and Matplotlib retain their licenses. Manuscript text, supplementary articles and journal class files are excluded from this standalone package and this grant. Author names follow the supplied manuscript title page. Claude assisted with the original code; ChatGPT/Codex assisted with revision, checking and packaging. Authors retain responsibility for the released version. No educational effectiveness has been measured.

CITATION.cff describes this local software package. After a real public release and archival record exist, add their URL and DOI. Do not insert a fabricated identifier.
