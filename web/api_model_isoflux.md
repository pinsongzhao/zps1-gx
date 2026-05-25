# model_isoflux -- Model Isophotal Photometry

Module: `galfitx.model_isoflux`

This module provides the isophotal photometry pipeline for GALFIT/GalfitS processing. It handles background sky noise analysis, GALFIT model generation, aperture photometry with error propagation, and flux unit conversions.

---

## PhotometryConfig

```python
from galfitx.model_isoflux import PhotometryConfig
```

A `@dataclass` that bundles every parameter needed by the photometry pipeline.

### Image and catalog paths

| Field | Type | Default | Description |
|---|---|---|---|
| `image_list` | `List[str]` | *required* | Paths to science images (one per band). |
| `galaxy_catalog` | `str` | *required* | Path to the source-detection catalog (SExtractor format). |
| `psf_file` | `Optional[str]` | *required* | Path to the PSF FITS file used by GALFIT. |
| `segmentation_map_list` | `Optional[List[str]]` | *required* | Segmentation map for each band; used to compute model isophotal flux fractions. |
| `cutout_dir` | `str` | *required* | Directory containing cutout images (`obj{id}_{band}sci.fits`). |

### Photometry parameters

| Field | Type | Default | Description |
|---|---|---|---|
| `label_list` | `List[str]` | *required* | Band labels (e.g. `["f444w", "f160w"]`). |
| `filter_labels` | `List[str]` | *required* | Filter labels used in GALFITS summary files. |
| `gains` | `List[float]` | *required* | Detector gain for each band (e-/ADU). |
| `exptimes` | `List[float]` | *required* | Exposure time for each band (seconds). |
| `mjsr_list` | `List[float]` | *required* | MegaJansky per steradian conversion factor for each band. |
| `zero_list` | `List[float]` | *required* | Photometric zero-points (AB magnitudes) for each band. |

### Aperture settings

| Field | Type | Default | Description |
|---|---|---|---|
| `apertures_list` | `List[float]` | *required* | Aperture radii in arcsec. |
| `pixel_scales` | `List[float]` | *required* | Pixel scale for each band (arcsec/pixel). |
| `ref_pixel_scale` | `float` | `0.03` | Reference pixel scale (detection image) in arcsec/pixel. |
| `overlap_frac` | `float` | `0.1` | Maximum allowed overlap fraction between background apertures (0 to 1). |

### Matching

| Field | Type | Default | Description |
|---|---|---|---|
| `match_galaxy_catalog` | `Optional[str]` | `None` | Catalog to use for cross-matching. Falls back to `galaxy_catalog` when `None`. |
| `detwht_file` | `Optional[str]` | `None` | Detection weight image; regions with `whtdata==0` are masked. |
| `wht_file_list` | `Optional[List[str]]` | `None` | Per-band weight images used to build Kron masks. |

### Mask settings

| Field | Type | Default | Description |
|---|---|---|---|
| `kron_scale` | `list[float]` | `1.5` | Scale factor applied to the Kron radius when building masks. |
| `external_mask_file_list` | `Optional[List[str]]` | `None` | External mask files used directly instead of generating Kron masks. |

### Spectroscopic redshift

| Field | Type | Default | Description |
|---|---|---|---|
| `specz_cat` | `str` | `None` | Path to spectroscopic redshift catalog (must have `ra`, `dec`, `z_spec` columns). |
| `max_sep` | `float` | `0.1` | Maximum cross-match separation in arcsec. |

### GALFIT settings

| Field | Type | Default | Description |
|---|---|---|---|
| `galfit_path` | `Optional[str]` | `None` | Path to the GALFIT executable. |
| `gsdir` | `Optional[str]` | `None` | Directory containing GALFITS result summaries. |
| `stampfile` | `Optional[str]` | `None` | Stamp coordinate catalog. |
| `try_filterlist` | `List[int]` | `None` | Band indices (0-based) to try when computing model flux fractions; the first successful filter is used. |

### Output settings

| Field | Type | Default | Description |
|---|---|---|---|
| `sky_noise_path` | `str` | `"./sky_noise/"` | Directory for background noise products. |
| `gmoutdir` | `str` | `"./galfitm/"` | Directory for GALFIT model outputs. |
| `outgs_flux_catfile` | `str` | `"./gs_iso_flux.cat"` | Intermediate isophotal flux catalog path. |
| `output_file` | `str` | `"./gsflux_isoerr.cat"` | Final output catalog with errors. |

### Processing options

| Field | Type | Default | Description |
|---|---|---|---|
| `saveconfig` | `bool` | `False` | Keep GALFIT configuration files after model generation. |
| `savemodel` | `bool` | `True` | Keep GALFIT model FITS files after processing. |
| `save_mask` | `bool` | `True` | Save Kron/weight mask images to disk. |
| `save_regions` | `bool` | `True` | Save DS9 region files for background apertures. |
| `plot_histograms` | `bool` | `True` | Generate aperture-flux histogram plots. |

---

## PhotometryPipeline

```python
from galfitx.model_isoflux import PhotometryPipeline, PhotometryConfig

config = PhotometryConfig(...)
pipeline = PhotometryPipeline(config)
```

Orchestrates the full multi-band photometry workflow: background position selection, noise fitting, model flux computation, and error propagation.

### Constructor

```
PhotometryPipeline(config: PhotometryConfig)
```

| Parameter | Type | Description |
|---|---|---|
| `config` | `PhotometryConfig` | Configuration object. |

Initialises internal `BackgroundAnalyzer`, `FluxConverter`, `AperturePhotometry`, and (optionally) `GalfitModelGenerator` instances, and creates output directories.

### Methods

#### `step1_determine_background_positions()`

```
step1_determine_background_positions() -> None
```

Determines sky background aperture positions for every band and every aperture radius. For each band it computes a distance-transform map once, then places non-overlapping circular apertures on unmasked pixels whose distance from any masked region exceeds the aperture radius.

#### `step2_fit_background_noise()`

```
step2_fit_background_noise() -> pandas.DataFrame
```

Fits a single power-law noise model to the measured aperture standard deviations for each band:

```
sigma = sigma_1 * (alpha * r^beta)
```

**Returns:** a DataFrame with columns `BAND`, `SIGMA_1`, `ALPHA`, `BETA`. The result is also saved to `{sky_noise_path}bkg_fit_param.csv`.

#### `_compute_model_fluxes_for_galaxies()`

```
_compute_model_fluxes_for_galaxies(
    galids: np.ndarray,
    filter_list: List[str],
    image_list: List[str],
    gsdir: str,
    outdir: str,
    output_file: str,
) -> pandas.DataFrame
```

Iterates over galaxy IDs, reads the GALFITS summary, and calls `GalfitModelGenerator.get_flux_fraction` to compute the isophotal flux fraction for each source.

| Parameter | Type | Description |
|---|---|---|
| `galids` | `np.ndarray` | Array of integer galaxy IDs. |
| `filter_list` | `List[str]` | Filter names matching the GALFITS summary columns. |
| `image_list` | `List[str]` | Science image paths. |
| `gsdir` | `str` | GALFITS results directory. |
| `outdir` | `str` | Output directory for GALFIT model files. |
| `output_file` | `str` | Path where the resulting catalog CSV is written. |

**Returns:** DataFrame with columns `id`, `flux_frac`, and `m_{filter}` for each filter.

#### `step4_compute_isophotal_errors()`

```
step4_compute_isophotal_errors(
    bg_df: pandas.DataFrame,
    flux_df_path: str,
    gs_flux_err_outfile: str,
) -> pandas.DataFrame
```

Computes total isophotal flux, object noise error, background correlation error, and total error for every source in every band. Cross-matches with a spectroscopic redshift catalog if `specz_cat` is configured.

| Parameter | Type | Description |
|---|---|---|
| `bg_df` | `pandas.DataFrame` | Background fit parameters from `step2_fit_background_noise`. |
| `flux_df_path` | `str` | Path to the intermediate flux catalog from step 3. |
| `gs_flux_err_outfile` | `str` | Path for the final output catalog. |

**Returns:** DataFrame with columns `#id`, `{filter}_flux`, `{filter}_fluxerr_obj`, `{filter}_fluxerr_bkg`, `{filter}_fluxerr`, `ra`, `dec`, `segment_area`, and optionally `z_spec`.

#### `run_all_steps()`

```
run_all_steps() -> None
```

Convenience method that runs all four steps sequentially.

---

## create_kron_mask_numba

```python
from galfitx.model_isoflux import create_kron_mask_numba
```

```
create_kron_mask_numba(
    mask_image: np.ndarray,
    xcentroid: np.ndarray,
    ycentroid: np.ndarray,
    rad: np.ndarray,
    q: np.ndarray,
    theta: np.ndarray,
    xlo: np.ndarray,
    xhi: np.ndarray,
    ylo: np.ndarray,
    yhi: np.ndarray,
) -> np.ndarray
```

Numba JIT-compiled (`nopython=True`, `parallel=True`) function that creates an elliptical Kron mask for multiple sources simultaneously. For each source it iterates over the bounding box, rotates pixel offsets by the position angle, computes the elliptical radius, and sets mask pixels to 1.0 where the radius is within `rad`.

| Parameter | Type | Description |
|---|---|---|
| `mask_image` | `np.ndarray` | 2D output mask array, modified in-place. Existing non-zero values are preserved. |
| `xcentroid` | `np.ndarray` | X centroids of sources (float64). |
| `ycentroid` | `np.ndarray` | Y centroids of sources (float64). |
| `rad` | `np.ndarray` | Kron radii in pixels (float64). |
| `q` | `np.ndarray` | Axis ratios b/a (float64, <= 1). |
| `theta` | `np.ndarray` | Position angles in radians (float64). |
| `xlo` | `np.ndarray` | Left bounding-box x coordinates. |
| `xhi` | `np.ndarray` | Right bounding-box x coordinates. |
| `ylo` | `np.ndarray` | Bottom bounding-box y coordinates. |
| `yhi` | `np.ndarray` | Top bounding-box y coordinates. |

**Returns:** The updated `mask_image` array.

---

## circle_distance_for_overlap

```python
from galfitx.model_isoflux import circle_distance_for_overlap
```

```
circle_distance_for_overlap(frac: float) -> float
```

Solves for the centre-to-centre distance between two equal circles that yields a given overlap area fraction. Uses the geometric equation:

```
arccos(d/2r) - (d/2r) * sqrt(1 - (d/2r)^2) = (frac / 2) * pi
```

| Parameter | Type | Description |
|---|---|---|
| `frac` | `float` | Overlap area as a fraction of one circle's area (0 to 1). |

**Returns:** The normalised distance `d / r` (float).

---

## FluxConverter

```python
from galfitx.model_isoflux import FluxConverter
```

Static utility class for flux unit conversions.

| Method | Signature | Description |
|---|---|---|
| `flux_to_muJy` | `(flux: float, zeropoint: float) -> float` | ADU to micro-Jansky. |
| `mag_to_muJy` | `(mag: float) -> float` | AB magnitude to micro-Jansky. |
| `muJy_to_mag` | `(flux_muJy: float) -> float` | Micro-Jansky to AB magnitude. |
| `muJy_to_ADU` | `(flux_muJy, zeropoint, exptime, gain, mjsr) -> float` | Micro-Jansky to ADU counts. |

---

## corss_match_df

```python
from galfitx.model_isoflux import corss_match_df
```

```
corss_match_df(
    cat1: pd.DataFrame,
    cat2: pd.DataFrame,
    max_sep: float = 0.1,
) -> pd.DataFrame
```

Cross-matches two catalogs by sky position using `astropy.coordinates.match_coordinates_sky`. Adds a `z_spec` column to `cat1` with values from `cat2` where the separation is less than `max_sep` arcsec; unmatched sources receive `z_spec = -99.0`.

---

## GalfitModelGenerator

```python
from galfitx.model_isoflux import GalfitModelGenerator
```

Generates single-band GALFIT models from GALFITS multi-band results and computes isophotal flux fractions.

### Constructor

```
GalfitModelGenerator(galfit_path: str)
```

### Key method: `get_flux_fraction`

```
get_flux_fraction(
    gal_id: int,
    input_image: str,
    psf_file: str,
    filter: str,
    cutout_dir: str,
    band_label: str,
    pixel_scale: float,
    zeropoint: float,
    stamp_file: str = "./stamps",
    catalog_name: str = "./sex/outcat",
    segmentation_map: str = "./sex/outseg.fits",
    gs_dir: str = "./galfits/",
    gmout_dir: str = "./galfitm/",
    saveconfig: bool = True,
    savemodel: bool = True,
) -> Optional[float]
```

Reads the GALFITS summary for a galaxy, writes a single-band GALFIT config, runs GALFIT in model-only mode (`-o1`), and computes the isophotal flux fraction within the segmentation footprint.

**Returns:** the flux fraction (total model flux / isophotal model flux), or `None` if the summary file is missing.

---

## BackgroundAnalyzer

```python
from galfitx.model_isoflux import BackgroundAnalyzer
```

Handles all background noise analysis: Kron mask creation, distance-map computation, aperture placement, and noise-curve fitting.

### Key methods

| Method | Description |
|---|---|
| `create_kron_mask_map(catalog_name, detwht_file, kron_scale, output_path)` | Builds a detection-band Kron mask from the source catalog and weight image. Returns the mask file path. |
| `create_integrated_mask_map(filter_name, catalog_name, detwht_file, wht_file, kron_scale, output_path, save_mask)` | Combines the Kron mask with a per-band weight mask. Returns the 2D mask array. |
| `calculate_distance_map(filter_name, catalog_name, detwht_file, wht_file, kron_scale, external_mask_file, output_path, save_mask)` | Returns a Euclidean distance transform of the complement of the mask (distance to nearest masked pixel). |
| `decide_background_positions_with_distance_map(filter_name, aperture_radius, overlap_frac, distance_map, output_path, save_regions)` | Places non-overlapping background apertures using a pre-computed distance map. Returns a list of `(x, y)` tuples. |
| `compute_background_noise(filter_name, sigma_1, aperture_radii, pixel_scale, output_path, science_file, plot_histograms)` | Measures aperture flux scatter at multiple radii and fits a power-law noise model. Returns `(alpha, beta)`. |

---

## AperturePhotometry

```python
from galfitx.model_isoflux import AperturePhotometry
```

### Key method: `compute_isophotal_errors`

```
compute_isophotal_errors(
    catalog, flux_frac_catfile, band, band_label,
    background_params, gain, exptime, mjsr, zeropoint,
    pixel_scale, ref_pixel_scale=0.03,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

Computes, for every source, the total isophotal flux, object noise error, background correlation error, and total error (quadrature sum).

**Returns:** `(total_flux, object_error, correlation_error, total_error)` as four arrays.

---

## process_multi_band_photometry

```python
from galfitx.model_isoflux import process_multi_band_photometry
```

```
process_multi_band_photometry(**kwargs) -> None
```

Backward-compatible convenience function. Accepts either a `config=PhotometryConfig(...)` keyword argument or individual `PhotometryConfig` fields as keyword arguments, then constructs a pipeline and calls `run_all_steps()`.
