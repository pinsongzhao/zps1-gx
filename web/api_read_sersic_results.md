# read_sersic_results -- Parse Galfit / GalfitM Output

Module: `galfitx.read_sersic_results`

Functions for reading and parsing Galfit/GalfitM output FITS files, extracting best-fit Sersic parameters, uncertainties, and chi-squared statistics. Handles both normal (single-component) and bright-object deblending (two-component) fits, as well as missing-output fallback.

---

## derive_primary_chi2

```python
from galfitx.read_sersic_results import derive_primary_chi2
```

```
derive_primary_chi2(
    obj_file: str,
    gal_exe: str,
) -> None
```

Runs a quick Galfit evaluation for the primary source only, in order to extract chi-squared statistics without neighbouring sources contaminating the fit.

### How it works

1. Reads the original Galfit configuration file (`obj_file`).
2. Counts the number of bands and identifies band names from the `A)` / `A1)` header lines.
3. Writes a modified configuration where:
   - The output filename (line `B)`) is changed to a temporary file.
   - Mask files (line `F)`) are rebuilt so that only the primary source is unmasked (all other sources and original masked regions remain masked).
   - All neighbour-component fluxes (lines `P)`) are set to zero.
4. Executes Galfit with the `-o2` flag (fast re-computation, no model image saved).
5. Reads `CHISQ`, `NDOF`, and `CHI2NU` from the `FIT_INFO` extension of the output FITS.
6. Writes these three values to an ASCII file `{obj_file}_primary_fit_info`.
7. Deletes all temporary files (masks, modified config, output FITS).

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `obj_file` | `str` | Path to the original Galfit configuration file (e.g. `./galfit/t.266`). |
| `gal_exe` | `str` | Path to the Galfit executable (`galfitm` or `galfit`). |

### Returns

`None`. The function writes a small ASCII sidecar file and does not return a value.

### Sidecar file format

The file `{obj_file}_primary_fit_info` contains a single line with three whitespace-separated values:

```
{NDOF} {CHISQ} {CHI2NU}
```

---

## read_sersic_results

```python
from galfitx.read_sersic_results import read_sersic_results
```

```
read_sersic_results(
    obj: str,
    nband: int,
    setup: Any,
    bd: Optional[bool] = None,
    final: Optional[Any] = None,
) -> Any
```

Parses a Galfit/GalfitM output FITS file and returns a feedback object whose attributes hold all fit parameters, uncertainties, and statistics. Dynamically creates a class instance depending on whether the fit is a normal single-component run or a bright-object deblending (BD) run.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `obj` | `str` | *required* | Path to the Galfit output FITS file (e.g. `./galfit/t.266_gf.fits`). |
| `nband` | `int` | *required* | Number of photometric bands in the fit. |
| `setup` | `Any` | *required* | A `Setup` instance (from `read_setup()`) containing configuration, especially `setup.galexe` (path to the Galfit executable). |
| `bd` | `Optional[bool]` | `None` | If `True`, interpret as a BD (deblended) fit with diffuse (`_d`) and bulge (`_b`) components (COMP2 and COMP3). If `None` or `False`, interpret as a normal single-component fit (COMP2). |
| `final` | `Optional[Any]` | `None` | Unused, retained for backward compatibility. |

### Returns

A dynamically created object (`feedback`) whose attributes depend on the fit type and whether the output file exists.

---

### Normal fit (`bd=None` or `bd=False`), file exists

First-band scalars:

| Attribute | Type | Description |
|---|---|---|
| `mag_galfit` | `float` | Best-fit magnitude (COMP2, first band). |
| `magerr_galfit` | `float` | Magnitude uncertainty. |
| `re_galfit` | `float` | Effective radius in pixels. |
| `reerr_galfit` | `float` | Effective radius uncertainty. |
| `n_galfit` | `float` | Sersic index. |
| `nerr_galfit` | `float` | Sersic index uncertainty. |
| `q_galfit` | `float` | Axis ratio (b/a). |
| `qerr_galfit` | `float` | Axis ratio uncertainty. |
| `pa_galfit` | `float` | Position angle in degrees. |
| `paerr_galfit` | `float` | PA uncertainty. |
| `x_galfit` | `float` | X centroid position. |
| `xerr_galfit` | `float` | X centroid uncertainty. |
| `y_galfit` | `float` | Y centroid position. |
| `yerr_galfit` | `float` | Y centroid uncertainty. |
| `psf_galfit` | `str` | PSF file used for the first band. |
| `sky_galfit` | `float` | Sky background level (COMP1). |

Per-band arrays (same names with `_band` suffix):

`mag_galfit_band`, `magerr_galfit_band`, `re_galfit_band`, `reerr_galfit_band`, `n_galfit_band`, `nerr_galfit_band`, `q_galfit_band`, `qerr_galfit_band`, `pa_galfit_band`, `paerr_galfit_band`, `x_galfit_band`, `xerr_galfit_band`, `y_galfit_band`, `yerr_galfit_band`, `sky_galfit_band` -- each is a numpy array of length `nband`.

Chebyshev polynomial results (same names with `_cheb` suffix):

`mag_galfit_cheb`, `re_galfit_cheb`, `n_galfit_cheb`, `q_galfit_cheb`, `pa_galfit_cheb`, `x_galfit_cheb`, `y_galfit_cheb` (and their `err` variants) -- arrays of length `nband`.

Fit statistics:

| Attribute | Type | Description |
|---|---|---|
| `chisq_galfit` | `float` | Total chi-squared. |
| `ndof_galfit` | `int` | Number of degrees of freedom. |
| `chi2nu_galfit` | `float` | Reduced chi-squared (chi-squared per DOF). |
| `nfree_galfit` | `int` | Number of free parameters. |
| `nfix_galfit` | `int` | Number of fixed parameters. |
| `niter_galfit` | `int` | Number of iterations. |
| `ngood_galfit_band` | `int` | Number of good (unmasked) pixels across all bands. |
| `nmask_galfit_band` | `int` | Number of masked pixels across all bands. |
| `neigh_galfit` | `int` | Number of neighbour components fitted simultaneously. |
| `flag_galfit` | `int` | Fit status flag: `2` = success, `1` = missing file. |
| `galfit_version` | `str` | Galfit version string. |
| `firstcon_galfit` | `float` | First convergence metric. |
| `lastcon_galfit` | `float` | Last convergence metric. |
| `cputime_setup_galfit` | `float` | CPU time for setup (seconds). |
| `cputime_fit_galfit` | `float` | CPU time for fitting (seconds). |
| `cputime_total_galfit` | `float` | Total CPU time (seconds). |

Chebyshev degree totals:

`x_galfit_deg`, `y_galfit_deg`, `mag_galfit_deg`, `re_galfit_deg`, `n_galfit_deg`, `q_galfit_deg`, `pa_galfit_deg` -- sum of Chebyshev polynomial degrees across bands for each parameter.

Primary-only chi-squared:

| Attribute | Type | Description |
|---|---|---|
| `ndof_galfit_prime` | `int` | DOF from primary-only fit. |
| `chisq_galfit_prime` | `float` | Chi-squared from primary-only fit. |
| `chi2nu_galfit_prime` | `float` | Reduced chi-squared from primary-only fit. |

Metadata:

| Attribute | Type | Description |
|---|---|---|
| `initfile` | `str` | Initialisation file path. |
| `logfile` | `str` | Log file path. |
| `constrnt` | `str` | Constraint file path. |
| `fitsect` | `str` | Fitting section string. |
| `convbox` | `str` | Convolution box size. |
| `psf_galfit_band` | `array` | PSF file path for each band. |

---

### BD fit (`bd=True`), file exists

All scalar and array attributes from the normal fit are present, but with `_d` (diffuse, COMP2) and `_b` (bulge, COMP3) suffixes instead of no suffix. For example:

- `mag_galfit_d`, `re_galfit_d`, `n_galfit_d`, `q_galfit_d`, `pa_galfit_d`, `x_galfit_d`, `y_galfit_d` (and `err` variants)
- `mag_galfit_b`, `re_galfit_b`, `n_galfit_b`, `q_galfit_b`, `pa_galfit_b`, `x_galfit_b`, `y_galfit_b` (and `err` variants)
- Per-band: `mag_galfit_band_d`, `mag_galfit_band_b`, etc.
- Chebyshev: `mag_galfit_cheb_d`, `mag_galfit_cheb_b`, etc.
- Degrees: `mag_galfit_deg_d`, `mag_galfit_deg_b`, etc.
- Statistics use `_bd` suffix: `chisq_galfit_bd`, `ndof_galfit_bd`, `chi2nu_galfit_bd`, `niter_galfit_bd`, `flag_galfit_bd`, etc.
- Primary-only: `ndof_galfit_bd_prime`, `chisq_galfit_bd_prime`, `chi2nu_galfit_bd_prime`.
- Sky: `sky_galfit_bd`, `sky_galfit_band_bd`, `sky_galfit_cheb_bd`.

`neigh_galfit_bd` = total components minus 4 (sky + primary diffuse + primary bulge + one extra).

---

### Missing file fallback

When the output FITS file does not exist, the feedback object is still created with sentinel values:

- Fit parameters: magnitudes set to `-999.0`, radii/Sersic/axis-ratio to `-99.0`, PA/position to `0.0`.
- Errors: `99999.0`.
- Arrays: lists of length `nband` filled with the same sentinel values.
- Statistics: `-99` or `-99.0` as appropriate.
- `flag_galfit` = `1` (normal) or `flag_galfit_bd` = `1` (BD).
- `galfit_version` = `"crash"`.

This allows downstream code to safely access attributes and filter on `flag_galfit != 2` to identify failed fits.
