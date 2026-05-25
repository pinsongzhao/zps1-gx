# Source Detection

::: galfitx.source_detection

Module for SExtractor-style source detection on astronomical images, built on top of `photutils`.

---

## `SExtractor`

```python
SExtractor(
    filename,
    catalog_name="default_gx_cat",
    detect_minarea=5,
    detect_thresh=1.5,
    kernel=KERNEL_DEFAULT,
    detect_connectivity=8,
    deblend=False,
    deblend_nthresh=32,
    deblend_mincont=0.005,
    deblend_mode="exponential",
    nproc=1,
    clean=False,
    clean_param=1.0,
    mask=None,
    coverage_mask=None,
    back_type=False,
    back_value=0.0,
    back_size=64,
    back_filtersize=3,
    bkg_estimator=SExtractorBackground(),
    weight_type="NONE",
    weight_name=None,
    checkimage_type=[],
    phot_apertures=None,
    phot_autoparams=[2.5, 1.4],
    phot_petroparams=[2.0, 3.5],
    mag_zeropoint=38.951,
    wcs=None,
    gain=0.0,
    pixel_scale=0.06,
    nnw_sex="./default.nnw",
    fwhm_arcsec=0.16,
    verbose=False,
)
```

Python equivalent of SExtractor source detection on a single FITS image, following the SExtractor methodology. All parameters are named after the SExtractor configuration parameters.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `filename` | `str` | *(required)* | Path to the FITS image file. |
| `catalog_name` | `str` | `"default_gx_cat"` | Output filename for the ASCII catalog. |
| `detect_minarea` | `int` | `5` | Minimum number of connected pixels above threshold for a valid detection. |
| `detect_thresh` | `float` | `1.5` | Detection threshold in units of background RMS. |
| `kernel` | `np.ndarray` or `None` | `KERNEL_DEFAULT` | Convolution kernel applied before detection. Use `None` to skip convolution. The default is the SExtractor 3x3 "all-ground" kernel with FWHM = 2 pixels. |
| `detect_connectivity` | `{4, 8}` | `8` | Pixel connectivity for grouping: 4 = edge-adjacent only; 8 = edge + corner. |
| `deblend` | `bool` | `False` | Whether to deblend overlapping sources. |
| `deblend_nthresh` | `float` | `32` | Number of thresholds used in the multi-thresholding deblending step. |
| `deblend_mincont` | `float` | `0.005` | Minimum contrast ratio for deblending. A child source must have a peak at least this fraction of the parent's peak. |
| `deblend_mode` | `{"exponential", "linear", "log", "asinh"}` | `"exponential"` | Threshold spacing mode for deblending. |
| `nproc` | `int` | `1` | Number of parallel processes for deblending. |
| `clean` | `bool` | `False` | Whether to remove spurious detections using Moffat-profile cleaning. |
| `clean_param` | `float` | `1.0` | Cleaning efficiency parameter (beta in the Moffat profile). |
| `mask` | `np.ndarray` or `None` | `None` | Boolean mask of bad pixels (`True` = masked). |
| `coverage_mask` | `np.ndarray` or `None` | `None` | Boolean coverage mask used during background estimation. |
| `back_type` | `bool` | `False` | If `True`, perform automatic background estimation. If `False`, subtract `back_value` manually. |
| `back_value` | `float` or `np.ndarray` | `0.0` | Scalar or 2D background value used when `back_type=False`. |
| `back_size` | `int` | `64` | Size of the background mesh boxes in pixels. |
| `back_filtersize` | `int` | `3` | Size of the median filter applied to the background mesh. |
| `bkg_estimator` | `BackgroundBase` | `SExtractorBackground()` | Photutils background estimator instance. |
| `weight_type` | `{"NONE", "BACKGROUND", "MAP_RMS", "MAP_WEIGHT", "MAP_VAR"}` | `"NONE"` | Type of weight map. |
| `weight_name` | `str` or `None` | `None` | Path to the weight/error FITS image. Required when `weight_type` is not `"NONE"`. |
| `checkimage_type` | `list[str]` | `[]` | Types of check images to produce (e.g., `"background"`, `"segmentation"`). |
| `phot_apertures` | `float` or `list[float]` or `None` | `None` | Radii in pixels for circular aperture photometry. |
| `phot_autoparams` | `list[float]` | `[2.5, 1.4]` | Parameters `[factor, min_radius]` for Kron AUTO photometry. |
| `phot_petroparams` | `list[float]` | `[2.0, 3.5]` | Parameters `[nsigma, min_radius]` for Petrosian photometry. |
| `mag_zeropoint` | `float` | `38.951` | Magnitude zeropoint. |
| `wcs` | `WCS` or `None` | `None` | WCS object. If `None`, extracted from the FITS header. |
| `gain` | `float` | `0.0` | Gain in electrons/ADU for error propagation. |
| `pixel_scale` | `float` | `0.06` | Pixel scale in arcsec/pixel. |
| `nnw_sex` | `str` | `"./default.nnw"` | Path to the SExtractor neural-network file for star/classification. |
| `fwhm_arcsec` | `float` | `0.16` | Seeing FWHM in arcseconds. |
| `verbose` | `bool` | `False` | Print progress information. |

### Returns

**`tuple[Table, SegmentationImage, SourceCatalog]`**

- **table** (`astropy.table.Table`) -- Source catalog with SExtractor-like measurements (fluxes, magnitudes, shapes, etc.).
- **segment_img** (`photutils.segmentation.SegmentationImage`) -- Segmentation map after detection and optional deblending/cleaning. Each source has a unique positive integer label.
- **cat** (`photutils.segmentation.SourceCatalog`) -- Photutils SourceCatalog object for further queries.

### Example

```python
from galfitx.source_detection import SExtractor

table, segm, cat = SExtractor(
    filename="science.fits",
    catalog_name="my_catalog",
    detect_thresh=1.5,
    deblend=True,
    deblend_nthresh=32,
    clean=True,
    back_type=True,
    back_size=64,
    weight_type="MAP_RMS",
    weight_name="rms.fits",
    mag_zeropoint=25.0,
    pixel_scale=0.03,
    verbose=True,
)
```

---

## `SExtractor_HDR`

```python
SExtractor_HDR(
    filename,
    catalog_name=("coldcat", "hotcat", "outcat"),
    segmap_name=("coldseg.fits", "hotseg.fits", "outseg.fits"),
    path="./sex/",
    kernel=(KERNEL_DEFAULT, KERNEL_DEFAULT),
    detect_minarea=(5, 5),
    detect_thresh=(3, 2),
    detect_connectivity=(8, 8),
    deblend=(False, False),
    deblend_nthresh=(32, 32),
    deblend_mincont=(0.002, 0.005),
    clean=(False, False),
    clean_param=(1.0, 1.0),
    mask=None,
    coverage_mask=None,
    back_type=(False, False),
    back_value=(0.0, 0.0),
    back_size=(128, 32),
    back_filtersize=(3, 3),
    bkg_estimator=SExtractorBackground(),
    bkgrms_estimator=StdBackgroundRMS(),
    weight_type="NONE",
    weight_name=None,
    checkimage_type=[],
    phot_apertures=None,
    phot_autoparams=[2.5, 1.4],
    scale_factor=1.1,
    wcs=None,
    pixel_scale=0.06,
    nnw_sex="default.nnw",
    fwhm_arcsec=0.16,
    gain=0.0,
    verbose=False,
    **kwargs,
)
```

Cold + Hot dual-threshold HDR (High Dynamic Range) source extraction following the method described in Barden et al. (2012). Runs two SExtractor passes: a "cold" detection for large, bright objects and a "hot" detection for small, faint sources, then combines the results using `sexcomb`.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `filename` | `str` | *(required)* | Path to the input FITS image. |
| `catalog_name` | `tuple[str, str, str]` | `("coldcat", "hotcat", "outcat")` | Names for the cold, hot, and combined output catalogs. |
| `segmap_name` | `tuple[str, str, str]` | `("coldseg.fits", "hotseg.fits", "outseg.fits")` | Names for the cold, hot, and combined segmentation FITS files. |
| `path` | `str` | `"./sex/"` | Output directory. **Warning:** removed and recreated on each run. |
| `kernel` | `tuple[np.ndarray, np.ndarray]` | `(KERNEL_DEFAULT, KERNEL_DEFAULT)` | Convolution kernels for the cold and hot passes. |
| `detect_minarea` | `tuple[int, int]` | `(5, 5)` | Minimum pixels for detection (cold, hot). |
| `detect_thresh` | `tuple[float, float]` | `(3, 2)` | Detection threshold in RMS units (cold, hot). |
| `detect_connectivity` | `tuple[{4,8}, {4,8}]` | `(8, 8)` | Pixel connectivity for each pass. |
| `deblend` | `tuple[bool, bool]` | `(False, False)` | Whether to deblend in each pass. |
| `deblend_nthresh` | `tuple[float, float]` | `(32, 32)` | Deblending thresholds for each pass. |
| `deblend_mincont` | `tuple[float, float]` | `(0.002, 0.005)` | Minimum contrast ratio for deblending in each pass. |
| `clean` | `tuple[bool, bool]` | `(False, False)` | Whether to clean spurious detections in each pass. |
| `clean_param` | `tuple[float, float]` | `(1.0, 1.0)` | Cleaning parameter for each pass. |
| `mask` | `np.ndarray` or `None` | `None` | Boolean bad-pixel mask (applied to both passes). |
| `coverage_mask` | `np.ndarray` or `None` | `None` | Boolean coverage mask. |
| `back_type` | `tuple[bool, bool]` | `(False, False)` | Automatic background estimation toggle per pass. |
| `back_value` | `tuple[float, float]` | `(0.0, 0.0)` | Manual background value per pass (when `back_type=False`). |
| `back_size` | `tuple[int, int]` | `(128, 32)` | Background mesh size per pass. |
| `back_filtersize` | `tuple[int, int]` | `(3, 3)` | Background filter size per pass. |
| `bkg_estimator` | `SExtractorBackground` | `SExtractorBackground()` | Background estimator. |
| `bkgrms_estimator` | `StdBackgroundRMS` | `StdBackgroundRMS()` | Background RMS estimator. |
| `weight_type` | `{"NONE", "BACKGROUND", "MAP_RMS", "MAP_WEIGHT", "MAP_VAR"}` | `"NONE"` | Weight map type. |
| `weight_name` | `str` or `None` | `None` | Path to weight map FITS. |
| `checkimage_type` | `list[str]` | `[]` | Check image types. |
| `phot_apertures` | `float` or `list[float]` or `None` | `None` | Aperture radii for photometry. |
| `phot_autoparams` | `list[float]` | `[2.5, 1.4]` | Kron AUTO photometry parameters. |
| `scale_factor` | `float` | `1.1` | Elliptical distance scaling for combining cold/hot catalogs. A hot source is kept only if its scaled distance to every cold source exceeds `scale_factor**2`. |
| `wcs` | `WCS` or `None` | `None` | WCS object. |
| `pixel_scale` | `float` | `0.06` | Pixel scale in arcsec/pixel. |
| `gain` | `float` | `0.0` | Gain in e-/ADU. |
| `verbose` | `bool` | `False` | Print progress information. |
| `**kwargs` | `dict` | -- | Additional keyword arguments passed to both `SExtractor` calls. |

### Returns

**`tuple[Table, SegmentationImage]`**

- **outtab** (`astropy.table.Table`) -- Combined catalog of cold sources and retained hot sources.
- **outsegm** (`photutils.segmentation.SegmentationImage`) -- Combined segmentation map.

### Example

```python
from galfitx.source_detection import SExtractor_HDR

outtab, outsegm = SExtractor_HDR(
    filename="science.fits",
    detect_thresh=(3.0, 1.5),
    back_size=(128, 32),
    deblend=(True, True),
    clean=(True, False),
    weight_type="MAP_RMS",
    weight_name="rms.fits",
    scale_factor=1.1,
    pixel_scale=0.03,
    verbose=True,
)
```

---

## `se_background`

```python
se_background(
    image,
    back_size=64,
    back_filtersize=3,
    bkg_estimator=SExtractorBackground(),
    mask=None,
    coverage_mask=None,
    weight_type="NONE",
    weight_image=None,
    verbose=False,
)
```

Estimate the background and background RMS using the SExtractor mesh method (via `photutils.background.Background2D`).

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | `np.ndarray` | *(required)* | 2D input image data. |
| `back_size` | `int` | `64` | Size of the background mesh boxes in pixels. |
| `back_filtersize` | `int` | `3` | Size of the median filter for background smoothing. |
| `bkg_estimator` | `BackgroundBase` | `SExtractorBackground()` | Photutils background estimator. |
| `mask` | `np.ndarray` or `None` | `None` | Boolean bad-pixel mask. |
| `coverage_mask` | `np.ndarray` or `None` | `None` | Boolean coverage mask. |
| `weight_type` | `{"NONE", "BACKGROUND", "MAP_RMS", "MAP_WEIGHT", "MAP_VAR"}` | `"NONE"` | Determines how the RMS is derived. `"NONE"` uses the median of the low-resolution RMS as a global value. `"MAP_RMS"` uses the weight image directly. `"MAP_WEIGHT"` / `"MAP_VAR"` derive RMS by scaling the weight/variance map to match the background RMS. |
| `weight_image` | `np.ndarray` or `None` | `None` | Weight map array (required for types other than `"NONE"` and `"BACKGROUND"`). |
| `verbose` | `bool` | `False` | Print additional information (e.g., scaling factor). |

### Returns

**`tuple[Background2D, np.ndarray, float]`**

- **bkg** (`photutils.background.Background2D`) -- Full background estimation object.
- **bkg_rms** (`np.ndarray`) -- Background RMS map (same shape as image).
- **global_rms** (`float`) -- Median of the low-resolution background RMS.

---

## `se_clean`

```python
se_clean(
    cat_spur,
    threshold,
    clean_param=1.0,
    detect_minarea=8,
    verbose=False,
)
```

Clean spurious detections from a segmentation image using the Moffat-profile method from SExtractor's `clean.c`. Iteratively determines whether brighter sources "eat" fainter neighbouring sources based on their Moffat amplitude and distance.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `cat_spur` | `SourceCatalog` | *(required)* | Catalog of (potentially spurious) detections. |
| `threshold` | `np.ndarray` | *(required)* | Detection threshold image (same shape as the data). |
| `clean_param` | `float` | `1.0` | Cleaning efficiency (beta in the Moffat profile). |
| `detect_minarea` | `int` | `8` | Minimum pixels for a valid detection (used in `mthresh` computation). |
| `verbose` | `bool` | `False` | Show a progress bar. |

### Returns

**`SegmentationImage`** -- Cleaned segmentation image with spurious sources removed (pixels set to 0).

---

## `se_deblend`

```python
se_deblend(
    data,
    segment_img,
    convolved_data,
    threshold,
    detect_minarea=5,
    deblend_nthresh=32,
    deblend_mincont=0.001,
    detect_connectivity=8,
    mode="exponential",
    nproc=1,
    verbose=False,
)
```

Deblend sources using the Multi-Thresholding + Gather Up method (SExtractor algorithm). Recursively applies decreasing thresholds to split blended sources.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `data` | `np.ndarray` | *(required)* | Background-subtracted image data. |
| `segment_img` | `SegmentationImage` | *(required)* | Input segmentation image (before deblending). |
| `convolved_data` | `np.ndarray` | *(required)* | Convolved version of the image. |
| `threshold` | `np.ndarray` | *(required)* | Detection threshold map. |
| `detect_minarea` | `int` | `5` | Minimum pixels for a detection. Sources smaller than `2 * detect_minarea` are not deblended. |
| `deblend_nthresh` | `int` | `32` | Number of thresholds in the multi-thresholding step. |
| `deblend_mincont` | `float` | `0.001` | Minimum contrast ratio for a child to be considered a separate source. |
| `detect_connectivity` | `{4, 8}` | `8` | Pixel connectivity for grouping. |
| `mode` | `{"exponential", "linear"}` | `"exponential"` | Threshold spacing mode. |
| `nproc` | `int` | `1` | Number of parallel processes. |
| `verbose` | `bool` | `False` | Show a progress bar during serial processing. |

### Returns

**`SegmentationImage`** -- Deblended segmentation image with consecutive labels.

---

## `cat2tab`

```python
cat2tab(
    cat,
    phot_apertures=None,
    mag_zeropoint=38.951,
    kron_fact=2.5,
)
```

Convert a Photutils `SourceCatalog` to an Astropy Table enriched with additional SExtractor-like measurements: elongation, ellipticity, FWHM, Kron radius, flux radius, celestial coordinates, shape parameters (cxx, cxy, cyy), Gini coefficient, segment area, and AUTO magnitude.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `cat` | `SourceCatalog` | *(required)* | Photutils source catalog. |
| `phot_apertures` | `float` or `list[float]` or `None` | `None` | Radius/radii in pixels for circular aperture photometry. If `None`, no aperture photometry is performed. |
| `mag_zeropoint` | `float` | `38.951` | Magnitude zeropoint for converting Kron flux to magnitude. |
| `kron_fact` | `float` | `2.5` | Multiplicative factor for the Kron radius (to match SExtractor convention). |

### Returns

**`astropy.table.Table`** -- Table with all standard and additional measurements, plus optional aperture photometry columns (`aper1_flux`, `aper1_fluxerr`, ...).

---

## `ds9reg`

```python
ds9reg(
    coldtab,
    hottab,
    outtab,
    pixel_scale=0.06,
    path="./sex/",
    fits_image=None,
)
```

Generate DS9 region files for cold, hot, and combined catalogs. Each region file contains elliptical annotations using Kron-aperture sizes. If a FITS image is provided, the WCS rotation angle is extracted and subtracted from the orientation to produce the correct sky position angle.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `coldtab` | `Table` | *(required)* | Cold source table. Must have columns: `label`, `ra`, `dec`, `kron_radius`, `semimajor_sigma`, `semiminor_sigma`, `orientation`. |
| `hottab` | `Table` | *(required)* | Hot source table. Same columns as `coldtab`. |
| `outtab` | `Table` | *(required)* | Combined source table. Same columns as above. |
| `pixel_scale` | `float` | `0.06` | Pixel scale in arcsec/pixel. |
| `path` | `str` | `"./sex/"` | Output directory for region files. |
| `fits_image` | `str` or `None` | `None` | Path to FITS image for WCS rotation correction. |

### Returns

`None` -- Writes three files: `cold.reg`, `hot.reg`, and `outcat.reg`.

---

## `sexcomb`

```python
sexcomb(
    coldtab,
    hottab,
    coldsegm,
    hotsegm,
    scale_factor=1.1,
    chunk_size=1000,
    verbose=False,
)
```

Combine cold and hot catalogs/segmentation maps into a single catalog and segmentation image. Retains hot sources whose scaled elliptical distance to every cold source exceeds `scale_factor**2` (i.e., hot sources that lie outside all cold source ellipses). Uses chunked processing for memory efficiency on large catalogs.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `coldtab` | `Table` | *(required)* | Cold source catalog. Must have columns: `label`, `xcentroid`, `ycentroid`, `cxx`, `cxy`, `cyy`, `kron_radius`. |
| `hottab` | `Table` | *(required)* | Hot source catalog. Same required columns. |
| `coldsegm` | `SegmentationImage` | *(required)* | Cold segmentation map. |
| `hotsegm` | `SegmentationImage` | *(required)* | Hot segmentation map. |
| `scale_factor` | `float` | `1.1` | Elliptical distance scaling threshold. |
| `chunk_size` | `int` | `1000` | Number of hot sources processed per chunk (for memory management). |
| `verbose` | `bool` | `False` | Print progress information. |

### Returns

**`tuple[Table, SegmentationImage]`**

- **outtab** (`Table`) -- Combined catalog (cold sources + retained hot sources).
- **outsegm** (`SegmentationImage`) -- Combined segmentation map.

---

## `se_make_kronmask`

```python
se_make_kronmask(cat, kron_params=(2.5, 1.4))
```

Create a boolean mask covering all Kron apertures for every source in the catalog. Each source's Kron aperture is converted to a pixel mask and combined via logical OR.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `cat` | `SourceCatalog` | *(required)* | Photutils source catalog. |
| `kron_params` | `tuple[float, float]` | `(2.5, 1.4)` | Kron aperture parameters: `(factor, min_radius)`. Matches SExtractor's AUTO settings. |

### Returns

**`np.ndarray` (bool)** -- Boolean mask of the same shape as the original image. `True` where any Kron aperture covers a pixel.
