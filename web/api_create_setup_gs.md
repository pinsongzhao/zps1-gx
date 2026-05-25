# GalfitS Configuration (`galfitx/create_setup_gs.py`)

Module for generating GalfitS configuration (lyric) files for structural fitting of galaxies. Supports pure Image fitting, pure SED fitting, and combined Image+SED fitting modes. Also provides utilities for segmentation reprojection, PSF processing, mask creation, sigma map generation, and unit conversions.

---

## `effective_wave`

A dictionary mapping filter/instrument names (strings) to their effective wavelengths in Angstroms. Covers a wide range of surveys and instruments including GALEX, Pan-STARRS, SDSS, DESI, HSC, DECam, JWST NIRCam/MIRI, HST ACS/WFC3, WISE, Spitzer-IRAC, 2MASS, Keck, Swift, XMM, Herschel, and CSST.

---

## `reproject_segm`

Reproject a segmentation image to match the WCS of a science image using nearest-neighbor interpolation.

```python
reproject_segm(segname, sciname, output=None, type=None)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `segname` | `str` | (required) | Path to the input FITS file containing the segmentation image. |
| `sciname` | `str` | (required) | Path to the science FITS image whose header defines the target WCS. |
| `output` | `str` or `None` | `None` | Path to the output FITS file. If `None`, writes to `sciname` with `.fits` replaced by `_seg.fits`. |
| `type` | `str` or `None` | `None` | If `"mask"`, pixels where the science image is zero or NaN are set to 1 in the output. |

**Returns**

| Type | Description |
|---|---|
| `int` | Always returns 1 (success indicator). |

---

## `process_psf`

Prepare PSF files for each band, cropping them if their dimensions exceed the corresponding science image cutout.

```python
process_psf(id, nband, psf_file_list, image_file_list, label_list, processed_psf_dir)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `id` | `int` | (required) | Object identifier, used in the output filename when cropping is needed. |
| `nband` | `int` | (required) | Number of bands. Must match the length of the list parameters. |
| `psf_file_list` | `list[str]` | (required) | List of paths to the original PSF FITS files, one per band. |
| `image_file_list` | `list[str]` | (required) | List of paths to the science image FITS files, one per band. |
| `label_list` | `list[str]` | (required) | List of band labels (e.g., `"F090W"`, `"F150W"`) used in output filenames. |
| `processed_psf_dir` | `list[str]` | (required) | Directory where cropped PSF files are saved. |

**Returns**

| Type | Description |
|---|---|
| `np.ndarray` of `str` | Array of paths to the PSF files that should be used for each band. For bands where cropping was not needed, the original path is kept. |

---

## `create_mask`

Create a mask for the current (primary) object based on SExtractor catalogs, segmentation data, and coverage maps. Identifies primary, secondary (overlapping), and tertiary (non-overlapping) sources. Returns corner coordinates and a list of co-fitting objects.

```python
create_mask(
    scihdu,
    seg_data,
    cover_data,
    catalog_name,
    paramfile,
    mask_file,
    current,
    scale=1.1,
    offset=4.0,
    limgal=3.0,
    ps=0.03,
)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `scihdu` | `str` | (required) | Path to the science/weight image (used for WCS and bad pixel detection). |
| `seg_data` | `str` | (required) | Path to the segmentation map. |
| `cover_data` | `str` | (required) | Path to the coverage data (pixels with coverage > 0 are masked). |
| `catalog_name` | `str` | (required) | Path to the detection catalog (ASCII). |
| `paramfile` | `str` | (required) | Path to the stamp file defining postage stamp boundaries. |
| `mask_file` | `str` | (required) | Path where the output mask FITS file will be saved. |
| `current` | `int` | (required) | Index of the current (primary) object in the catalog. |
| `scale` | `float` | `1.1` | Scaling factor for the Kron ellipse size. |
| `offset` | `float` | `4.0` | Additional offset in pixels added to the ellipse axes. |
| `limgal` | `float` | `3.0` | Magnitude limit for identifying secondary sources relative to the primary. |
| `ps` | `float` | `0.03` | Pixel scale in arcsec/pixel of the detection image. |

**Returns**

| Type | Description |
|---|---|
| `tuple[list[int], list[int]]` | A tuple of `(corner, objects)`, where `corner` is `[pxlo, pylo]` (pixel coordinates of the stamp origin), and `objects` is a list of catalog indices of co-fitting objects (primary + secondaries). |

---

## `create_sigma`

Create sigma (noise) maps for input science cutouts and prepare science images for GALFIT. Converts images to electrons and computes Poisson + sky noise in quadrature.

```python
create_sigma(
    image_stamp_list="none",
    mask_file_list="none",
    filter_list="none",
    label_list="none",
    expt_list=1,
    gain_list="none",
    mjsr_list="none",
    outpath="./",
)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `image_stamp_list` | `str` or `list[str]` | `"none"` | Path(s) to the input cutout FITS images. |
| `mask_file_list` | `str` or `list[str]` | `"none"` | Path(s) to mask files (unused in current implementation). |
| `filter_list` | `str` or `list[str]` | `"none"` | Instrument/filter identifier(s) for gain and MJy/sr lookup. |
| `label_list` | `str` or `list[str]` | `"none"` | Band label(s) used to construct output filenames. |
| `expt_list` | `float` or `list[float]` | `1` | Exposure time(s) in seconds for each image. |
| `gain_list` | `str` or `list[str]` | `"none"` | Gain value(s) for each image. |
| `mjsr_list` | `str` or `list[str]` | `"none"` | MJy/sr value(s) for each image. |
| `outpath` | `str` | `"./"` | Output directory for generated files. |

**Returns**

`None` -- Writes `*_sci.fits` (cleaned science) and `*_sigma.fits` (noise map) files to disk.

---

## `dist_ellipse`

Compute the elliptical distance from a center for each pixel in a grid. Used internally for constructing elliptical masks.

```python
dist_ellipse(n, xc, yc, ratio, angle, double=False)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `n` | `int` or `tuple[int, int]` or `list[int]` | (required) | Dimensions of the output array. If an integer, creates a square `n x n` array. If a tuple/list `(nx, ny)`, output shape is `(ny, nx)`. |
| `xc` | `float` | (required) | X-coordinate of the ellipse center (0-based). |
| `yc` | `float` | (required) | Y-coordinate of the ellipse center (0-based). |
| `ratio` | `float` | (required) | Stretch factor along the rotated x-axis (axis ratio). |
| `angle` | `float` | (required) | Rotation angle in degrees (counter-clockwise from positive x-axis). |
| `double` | `bool` | `False` | If True, use float64 precision. Otherwise float32. |

**Returns**

| Type | Description |
|---|---|
| `np.ndarray` | 2D array of shape `(ny, nx)` containing the elliptical distance for each pixel. |

---

## `caldelmax`

Compute the maximum RA and Dec offset (in arcseconds) from the image center to the corners of a cutout region.

```python
caldelmax(image_stamp_list, cutsize_arcsec)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `image_stamp_list` | `list[str]` | (required) | List of paths to image FITS files. Only the first element is used. |
| `cutsize_arcsec` | `float` | (required) | Desired cutout square size in arcseconds (full width). |

**Returns**

| Type | Description |
|---|---|
| `tuple[float, float]` | `(maxra_arcsec, maxdec_arcsec)` -- Maximum absolute RA and Dec offsets from center to corners in arcseconds. |

**Notes**

Assumes a pixel scale of 0.03 arcsec/pixel (hard-coded).

---

## `prepare_galfits`

Generate a GalfitS lyric configuration file for pure Image, pure SED, or Image+SED fitting modes. Handles initial guesses from previous fits or the SExtractor catalog, and supports SFH priors from SPS catalog samples.

```python
prepare_galfits(
    lyric_path,
    prior_path,
    cat_file,
    objects,
    det_label,
    sci_list,
    psf_list,
    zero_list,
    pixscl_list,
    label_list,
    filter_list,
    geo_smdir,
    pSed_smdir,
    imgSed_smdir,
    SPS_catalog_path=None,
    sfhs_path=None,
    z_list=None,
    sigma_list=None,
    mask_list=None,
    use_sed=0,
    use_sfh_prior=False,
    convf=False,
    convl=None,
    ebv=None,
)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `lyric_path` | `str` | (required) | Output path for the GalfitS lyric configuration file. |
| `prior_path` | `str` | (required) | Output path for the SFH prior file (written if `use_sfh_prior=True`). |
| `cat_file` | `str` | (required) | Path to the SExtractor detection catalog (ASCII). |
| `objects` | `list[int]` | (required) | List of catalog indices for co-fitting objects (primary + secondaries). |
| `det_label` | `str` | (required) | Label of the detection band in `label_list`. |
| `sci_list` | `list[str]` | (required) | Paths to science image cutouts for each band. |
| `psf_list` | `list[str]` | (required) | Paths to PSF images for each band. |
| `zero_list` | `list[float]` | (required) | Magnitude zeropoints for each band. |
| `pixscl_list` | `list[float]` | (required) | Pixel scales in arcsec/pixel for each band. |
| `label_list` | `list[str]` | (required) | Band labels. |
| `filter_list` | `list[str]` | (required) | Filter identifiers for GalfitS. |
| `geo_smdir` | `str` | (required) | Directory containing pure imaging `.gssummary` files. |
| `pSed_smdir` | `str` | (required) | Directory containing pure SED `.gssummary` files. |
| `imgSed_smdir` | `str` | (required) | Directory containing Image+SED `.gssummary` files. |
| `SPS_catalog_path` | `str` or `None` | `None` | Path to the SPS catalog (required if `use_sfh_prior=True`). |
| `sfhs_path` | `str` or `None` | `None` | Path to the SFH numpy data file (required if `use_sfh_prior=True`). |
| `z_list` | `list[float]` or `None` | `None` | Redshifts for each object. If `None`, reads from catalog `z_peak` column. |
| `sigma_list` | `list[str]` or `None` | `None` | Paths to sigma (noise) maps for each band. Defaults to `"none"`. |
| `mask_list` | `list[str]` or `None` | `None` | Paths to mask files for each band. Defaults to `"none"`. |
| `use_sed` | `int` | `0` | Fitting mode: `0` = pure Image, `1` = Image+SED. |
| `use_sfh_prior` | `bool` | `False` | If True, calculate SFH priors from the SPS catalog. |
| `convf` | `bool` | `False` | If True, use user-provided conversion factors (`convl`). |
| `convl` | `list[float]` or `None` | `None` | User-provided conversion factors for each filter (used if `convf=True`). |
| `ebv` | `float` or `None` | `None` | E(B-V) dust reddening value. If `None`, queried from SFD maps or IRSA. |

**Returns**

`None` -- Writes the lyric configuration file (and optionally the prior file) to disk.

---

## `gal_ebv`

Get E(B-V) dust reddening from the SFD dust maps (via `dustmaps` package) or fall back to IRSA dust query.

```python
gal_ebv(ra, dec)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `ra` | `float` | (required) | Right ascension in degrees. |
| `dec` | `float` | (required) | Declination in degrees. |

**Returns**

| Type | Description |
|---|---|
| `float` | E(B-V) reddening value from the SFD map. |

---

## `gen_pSed_data_lyric`

Generate mock data and lyric file for pure SED fitting. Creates 1x1 FITS "images" from photometry points and writes a GalfitS lyric configuration.

```python
gen_pSed_data_lyric(
    cat_path,
    z_cat_path,
    cutout_dir,
    mock_dir,
    label_list,
    filter_list,
    zp_list,
    ebv=None,
)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `cat_path` | `str` | (required) | Path to the photometry catalog (ASCII). Must contain columns `{filter}_flux` and `{filter}_fluxerr`. |
| `z_cat_path` | `str` | (required) | Path to the redshift catalog (ASCII). Must contain `id` and `z_peak` columns. |
| `cutout_dir` | `str` | (required) | Directory containing science image cutouts (used for WCS coordinates). |
| `mock_dir` | `str` | (required) | Directory where mock FITS images and lyric files are saved. |
| `label_list` | `list[str]` | (required) | Band labels. |
| `filter_list` | `list[str]` | (required) | Filter names matching columns in `cat_path`. |
| `zp_list` | `list[float]` | (required) | Magnitude zeropoints for each filter. |
| `ebv` | `float` or `None` | `None` | E(B-V) value. If `None`, queried from `gal_ebv`. |

**Returns**

`None` -- Writes mock FITS files and lyric configuration files to `mock_dir`.

---

## `cal_sfh_prior`

Calculate star formation history (SFH) priors from an SPS catalog by computing the distribution of sSFR and mass fraction in age bins for galaxies at similar redshifts.

```python
cal_sfh_prior(z, age_list_obj, SPS_catalog_path, sfhs_path, figout_path)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `z` | `float` | (required) | Target redshift. Galaxies within `z +/- 1` are selected. |
| `age_list_obj` | `list[float]` | (required) | Age bin edges in Gyr. |
| `SPS_catalog_path` | `str` | (required) | Path to the SPS catalog (FITS or ASCII). Must contain `id`, `z_50`, `mstar_50`. |
| `sfhs_path` | `str` | (required) | Path to the SFH numpy data file (`.npz` with `sfh`, `agebins_max`, `objid` arrays). |
| `figout_path` | `str` | (required) | Path where the SFH prior diagnostic plot is saved. |

**Returns**

| Type | Description |
|---|---|
| `tuple[np.ndarray, np.ndarray]` | `(logf_cont_median, logf_cont_std)` -- Median and standard deviation of log mass fraction in each age bin. |

---

## `photometry_to_img`

Convert a photometry point to a 1x1 FITS "image" for SED fitting. The flux is converted to luminosity units (1e38 erg/s/A).

```python
photometry_to_img(flux, flux_err, z, outputname, band=None, effectivewave=None, unit="mJy")
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `flux` | `float` | (required) | Flux value. Unit depends on `unit` parameter. |
| `flux_err` | `float` | (required) | Flux uncertainty (same unit as `flux`). |
| `z` | `float` | (required) | Redshift (used for luminosity distance calculation). |
| `outputname` | `str` | (required) | Output FITS file path. |
| `band` | `str` or `None` | `None` | Filter name (looked up in `effective_wave`). |
| `effectivewave` | `float` or `None` | `None` | Effective wavelength in Angstroms. Used if `band` is None. |
| `unit` | `str` | `"mJy"` | Unit of input flux. One of `"mJy"`, `"MagAB"`, `"flambda"`, `"L38"`. |

**Returns**

`None` -- Writes a FITS file with extensions for luminosity, sigma, and weight.

---

## `Fnu_to_Fl`

Convert flux density from millijanskys (mJy) to erg/s/cm^2/A.

```python
Fnu_to_Fl(Fnu, lambd)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `Fnu` | `float` or `np.ndarray` | (required) | Flux density in mJy. |
| `lambd` | `float` or `np.ndarray` | (required) | Wavelength in Angstroms. |

**Returns**

| Type | Description |
|---|---|
| `float` or `np.ndarray` | Flux density in erg/s/cm^2/A. |

---

## `ABmag_to_covf`

Convert AB magnitude to a conversion factor (the inverse of flux density in erg/s/cm^2/A at the given magnitude).

```python
ABmag_to_covf(mzp_AB, wave)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `mzp_AB` | `float` or `np.ndarray` | (required) | AB magnitude zeropoint. |
| `wave` | `float` or `np.ndarray` | (required) | Wavelength in Angstroms. |

**Returns**

| Type | Description |
|---|---|
| `float` or `np.ndarray` | Conversion factor with units of (erg/s/cm^2/A)^-1. Multiply by counts to get physical flux. |
