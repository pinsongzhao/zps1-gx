# eazy_utils -- EAZY Photometric Redshifts

Module: `galfitx.eazy_utils`

Utilities for configuring, running, and analysing photometric redshifts with the EAZY code. Covers configuration-file generation, filter translation, execution, binary-output reading, and interactive plotting.

---

## catalog_filters

```python
from galfitx.eazy_utils import catalog_filters
```

A `dict` mapping standard filter names to their EAZY filter IDs (as strings).

| Key (filter name) | EAZY ID |
|---|---|
| `acs_f435w` | `233` |
| `acs_f606w` | `236` |
| `acs_f814w` | `239` |
| `wfc3_f105w` | `202` |
| `wfc3_f125w` | `203` |
| `wfc3_f140w` | `204` |
| `wfc3_f160w` | `205` |
| `nircam_f090w` | `363` |
| `nircam_f115w` | `364` |
| `nircam_f150w` | `365` |
| `nircam_f200w` | `366` |
| `nircam_f277w` | `375` |
| `nircam_f356w` | `376` |
| `nircam_f410m` | `383` |
| `nircam_f444w` | `377` |
| `miri_f770w` | `396` |
| `miri_f1000w` | `397` |
| `miri_f1500w` | `400` |
| `miri_f1800w` | `401` |
| `csst_nuv` | `425` |
| `csst_u` | `426` |
| `csst_g` | `427` |
| `csst_r` | `428` |
| `csst_i` | `429` |
| `csst_z` | `430` |
| `csst_y` | `431` |

---

## zphot_config

```python
from galfitx.eazy_utils import zphot_config
```

```
zphot_config(
    catfile: str,
    outdir: str,
    temperr: float,
    syserr: float,
    eazypath: str = '/Path/to/eazy',
    template_file: str = 'templates/fsps_full/tweak_fsps_QSF_12_v3.param',
    template_combos: int = 99,
    zmax: float = 12.0,
    prior: int = 0,
    prior_filter: str = '205',
    prior_file: str = 'templates/prior_F160W_TAO.dat',
    zp_offsets: int = 0,
    fixspecz: int = 0,
    not_obs_threshold: float = -90.0,
    configfile: str = './zphot.param',
) -> None
```

Generates a complete EAZY configuration file. The function writes sections covering filter definitions, template sets, input/output file paths, a redshift grid, prior settings, zero-point offsets, and cosmology (`H0=70`, `Omega_m=0.3`, `Omega_Lambda=0.7`).

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `catfile` | `str` | *required* | Path to the input photometric catalog. |
| `outdir` | `str` | *required* | Output directory for EAZY results. |
| `temperr` | `float` | *required* | Template error amplitude (typical range 0.01--0.1). |
| `syserr` | `float` | *required* | Systematic flux error as a fraction of flux (typical range 0.01--0.1). |
| `eazypath` | `str` | `'/Path/to/eazy'` | Root directory of the EAZY installation. |
| `template_file` | `str` | `'templates/fsps_full/tweak_fsps_QSF_12_v3.param'` | Template definition file (relative to EAZY path). |
| `template_combos` | `int` | `99` | Template combination method: `1` = single, `2` = double, `99` = full set. |
| `zmax` | `float` | `12.0` | Maximum redshift in the grid. |
| `prior` | `int` | `0` | Apply magnitude prior (`0` = no, `1` = yes). |
| `prior_filter` | `str` | `'205'` | EAZY filter ID used for prior computation (default is WFC3 F160W). |
| `prior_file` | `str` | `'templates/prior_F160W_TAO.dat'` | Magnitude prior definition file. |
| `zp_offsets` | `int` | `0` | Compute zero-point offsets iteratively (`0` = no, `1` = yes). |
| `fixspecz` | `int` | `0` | Fix redshift to the spectroscopic value from the catalog (`0` = no, `1` = yes). |
| `not_obs_threshold` | `float` | `-90.0` | Flux values below this threshold are treated as non-observed. |
| `configfile` | `str` | `'./zphot.param'` | Output configuration file path. |

The redshift grid uses `Z_MIN=0.01`, `Z_MAX=zmax`, `Z_STEP=0.005`, and `Z_STEP_TYPE=1` (logarithmic steps in `1+z`).

---

## translate_config

```python
from galfitx.eazy_utils import translate_config
```

```
translate_config(
    catfile: str,
    configfile: str = 'zphot.translate',
) -> None
```

Generates an EAZY filter translation file. It reads the column headers of the input catalog, detects columns ending in `_flux`, looks up the corresponding EAZY filter ID in `catalog_filters`, and writes two lines per filter: one mapping `{filter}_flux` to `F{id}` and one mapping `{filter}_fluxerr` to `E{id}`.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `catfile` | `str` | *required* | Path to the photometric catalog (CSV). Column names must follow the convention `{filtername}_flux` / `{filtername}_fluxerr`. |
| `configfile` | `str` | `'zphot.translate'` | Output translation file path. |

---

## run_eazy

```python
from galfitx.eazy_utils import run_eazy
```

```
run_eazy(
    eazypath: str = '/Path/to/eazy',
    configfile: str = 'zphot.param',
    translatefile: str = 'zphot.translate',
    zeropointfile: str = 'zphot.zeropoint',
) -> None
```

Executes the EAZY photometric redshift code. Validates that the executable, configuration file, and translation file all exist before running.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `eazypath` | `str` | `'/Path/to/eazy'` | Root directory of the EAZY installation. The executable is expected at `{eazypath}/src/eazy`. |
| `configfile` | `str` | `'zphot.param'` | Configuration file generated by `zphot_config`. |
| `translatefile` | `str` | `'zphot.translate'` | Filter translation file generated by `translate_config`. |
| `zeropointfile` | `str` | `'zphot.zeropoint'` | Zero-point calibration file (currently not passed to the command-line call). |

---

## show_all_fitting

```python
from galfitx.eazy_utils import show_all_fitting
```

```
show_all_fitting(
    outdir: str,
    gsdir: str,
    output_path: str,
) -> None
```

Generates a full set of EAZY diagnostic figures for every source:

1. A photo-z vs. spec-z comparison plot saved as `{outdir}/photz_specz.png`.
2. Per-galaxy composite figures showing the GalfitS image (top) and the SED + PDF fit (bottom), saved as `{output_path}/{galaxy_id}_fitting.png`.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `outdir` | `str` | EAZY output directory (containing `photz.*` files). |
| `gsdir` | `str` | Directory with GalfitS image files (`obj{id}image_fit.png`). |
| `output_path` | `str` | Directory where per-galaxy figures are written. |

---

## EAzy_analysis

```python
from galfitx.eazy_utils import EAzy_analysis
```

```
EAzy_analysis(
    outputdir: str,
    outputfile: str,
    cache_file: str,
)
```

Reads all EAZY binary outputs and the `.zout` ASCII table into memory, exposing them as attributes for analysis and plotting.

### Constructor parameters

| Parameter | Type | Description |
|---|---|---|
| `outputdir` | `str` | Directory containing the EAZY output files. |
| `outputfile` | `str` | Base name of the output files (no extension), e.g. `'photz'`. |
| `cache_file` | `str` | Template cache file name (with extension), or `None`. |

### Key attributes

| Attribute | Type | Description |
|---|---|---|
| `z_best` | `np.ndarray` | Best photo-z estimate (`z_peak` if available, otherwise `z_a`). |
| `z_spec` | `np.ndarray` | Spectroscopic redshifts from the catalog. |
| `id` | `np.ndarray` | Object IDs. |
| `zgrid` | `np.ndarray` | Redshift grid used by EAZY. |
| `chi2fit` | `np.ndarray` | Chi-squared values, shape `(NOBJ, NZ)`. |
| `fnu` | `np.ndarray` | Observed fluxes, shape `(NOBJ, NFILT)`. |
| `efnu` | `np.ndarray` | Observed flux errors, shape `(NOBJ, NFILT)`. |
| `lc` | `np.ndarray` | Central wavelengths of filters in Angstroms. |
| `coeffs` | `np.ndarray` | Template coefficients, shape `(NOBJ, NTEMP)`. |
| `izbest` | `np.ndarray` | Index of best-fit redshift in `zgrid` for each object. |
| `qz` | `np.ndarray` | Redshift quality factor `q_z` (if available). |
| `u68`, `l68` | `np.ndarray` | Upper/lower 68% confidence limits. |

### Methods

#### `show_photz_compare()`

```
show_photz_compare(
    set1=None, set2=None,
    set1label=None, set2label=None,
    s=10, distin_qz=0.95, distin_chi2=50,
    errorbar=False, zmax=12, deltaz=0.5,
    ax=None,
) -> matplotlib.axes.Axes
```

Scatter plot comparing two redshift sets (default: `z_spec` vs `z_best`). Shows `sigma_NMAD`, outlier fraction, and 5-sigma outlier fraction. Optionally colour-codes by `q_z` and chi-squared, and adds a lower panel showing `dz/(1+z)`.

#### `show_zhist1d()`

```
show_zhist1d(zmax=12, ax=None) -> matplotlib.axes.Axes
```

1D histogram of `z_best`.

#### `show_zhist2d()`

```
show_zhist2d(
    set1=None, set2=None,
    set1label=None, set2label=None,
    zmax=12, ax=None,
) -> matplotlib.axes.Axes
```

2D density histogram (log colour scale) of two redshift sets.

#### `get_sed_fitting()`

```
get_sed_fitting(idx: int)
```

Returns the SED fitting data for one object.

**Returns:** `(lambdaz, itemp_sed, iobs_sed, fobs, efobs)`

| Return value | Shape | Description |
|---|---|---|
| `lambdaz` | `(NTEMPL,)` | Redshifted wavelength array. |
| `itemp_sed` | `(NTEMPL, NTEMP)` | Individual template SEDs scaled by coefficients. |
| `iobs_sed` | `(NFILT,)` | Best-fit model fluxes at filter central wavelengths. |
| `fobs` | `(NFILT,)` | Observed fluxes. |
| `efobs` | `(NFILT,)` | Observed flux errors. |

#### `show_sed_fitting()`

```
show_sed_fitting(
    idx: int,
    xrange=(2000, 50000), yrange=None,
    log_x=True, individual_templates=True,
    ltext=[...], fluxtype='f_lambda',
    ax=None,
) -> matplotlib.axes.Axes
```

Plots the best-fit SED, individual template contributions, and observed fluxes with error bars. Data points are colour-coded by S/N.

#### `get_pdf()`

```
get_pdf() -> None
```

Computes the posterior `p(z)` (with and without the prior) from the chi-squared grid and stores the results in `self.pzout` and `self.noprior_pzout`.

#### `show_pdf()`

```
show_pdf(
    idx: int,
    zrange=(0, 12),
    ax=None,
    specz=0,
) -> matplotlib.axes.Axes
```

Plots the redshift PDF for one object, showing both prior and no-prior curves, with optional spectroscopic redshift line.

#### `show_fitting()`

```
show_fitting(
    idx: int,
    xrange=(2000, 50000), yrange=None,
    log_x=True, individual_templates=True,
    specz=0, ltext=None,
    zrange=(0, 12), fluxtype='f_lambda',
    axs=None,
) -> list[matplotlib.axes.Axes]
```

Combined SED + PDF plot in a two-panel figure.

---

## Binary reading functions

Low-level functions for reading EAZY binary output files. Each returns a dictionary (or pair of dictionaries) with the extracted arrays.

### read_eazy_binary

```
read_eazy_binary(
    outputdir: str = '',
    outputfile: str = 'photz',
    cache_file: str = None,
) -> tuple
```

Reads all binary files at once. Returns a 5-tuple: `(photz_info, tempfilt_out, coeff_out, tempsed_out, pz_out)`.

### read_tempfilt_binary

```
read_tempfilt_binary(
    outputdir: str = '',
    outputfile: str = 'photz',
    cache_file: str = None,
) -> tuple
```

Returns `(photz_info, tempfilt_out)` where:

- `photz_info` = `{'NFILT': int, 'NTEMP': int, 'NZ': int, 'NOBJ': int}`
- `tempfilt_out` = `{'tempfilt': array(NZ, NTEMP, NFILT), 'lc': array(NFILT,), 'zgrid': array(NZ,), 'fnu': array(NOBJ, NFILT), 'efnu': array(NOBJ, NFILT)}`

### read_coeff_binary

```
read_coeff_binary(
    outputdir: str = '',
    outputfile: str = '',
) -> dict
```

Returns `{'coeffs': array(NOBJ, NTEMP), 'izbest': array(NOBJ,), 'tnorm': array(NTEMP,)}`.

### read_tempsed_binary

```
read_tempsed_binary(
    outputdir: str = '',
    outputfile: str = 'photz',
) -> dict
```

Returns `{'tempseds': array(NTEMP, NTEMPL), 'templambda': array(NTEMPL,), 'da': array(NZ,), 'db': array(NZ,)}`.

### read_pz_binary

```
read_pz_binary(
    outputdir: str = '',
    outputfile: str = '',
) -> dict
```

Returns `{'chi2fit': array(NOBJ, NZ), 'kbins': array, 'priorzk': array(NK, NZ), 'kidx': array(NOBJ,), 'NK': int}`. If no prior was applied, `kbins`, `priorzk`, `kidx`, and `NK` are `None`.

---

## zout_analysis

```python
from galfitx.eazy_utils import zout_analysis
```

```
zout_analysis(
    z_spec: np.ndarray,
    z_best: np.ndarray,
) -> tuple
```

Computes photo-z statistics for objects with `z_spec > 0` and `z_best > 0`.

**Returns:** `(dz, f_out, sigma, f_5sigma, compareidx)`

| Return | Type | Description |
|---|---|---|
| `dz` | `np.ndarray` | Normalised redshift difference `(z_best - z_spec) / (1 + z_spec)`. |
| `f_out` | `float` | Outlier fraction (`|dz| >= 0.15`), in percent. |
| `sigma` | `float` | Normalised median absolute deviation (`1.48 * median(|dz - median(dz)|)`). |
| `f_5sigma` | `float` | Fraction of `|dz| >= 5*sigma`, in percent. |
| `compareidx` | `np.ndarray` | Boolean mask of valid comparison objects. |

---

## show_z_compare

```python
from galfitx.eazy_utils import show_z_compare
```

```
show_z_compare(
    set1=None, set2=None,
    set1label=None, set2label=None,
    distin_qz=0.95, distin_chi2=50,
    errorbar=False, zmax=12,
    deltaz=0.5, s=1,
) -> tuple
```

Standalone z-z comparison plot (not a method of `EAzy_analysis`). Returns `(fig, [ax1, ax2])` where `ax1` is the scatter plot and `ax2` is the `dz/(1+z)` residual panel.
