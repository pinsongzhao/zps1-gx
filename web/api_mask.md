# Mask

::: galfitx.mask

Module for generating binary mask images and GALFITM configuration files for multi-band structural fitting.

---

## `create_mask`

```python
create_mask(
    weight_name,
    seg_name,
    catalog_name,
    paramfile,
    mask_file,
    mask_file_primary,
    current,
    scale,
    offset,
    limgal,
    b,
)
```

Create a binary mask for a postage stamp centred on a primary source. Pixels are classified into four categories based on source overlap and brightness, then combined into a final mask where `0` = good (unmasked) and `1` = masked.

**Pixel classification:**

| Category | Mask Value | Description |
|----------|------------|-------------|
| Primary | 0 | Pixels belonging to the primary source. |
| Secondary | 0 | Pixels of overlapping bright sources that will be co-fitted. |
| Tertiary | 1 | Pixels of other detected sources. |
| Bad | 1 | Zero-weight pixels from the weight image. |

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `weight_name` | `str` | *(required)* | Path to the weight (or science) FITS image. Used for the header and to identify zero-weight pixels. |
| `seg_name` | `str` | *(required)* | Path to the segmentation FITS image (from SExtractor). |
| `catalog_name` | `str` | *(required)* | Path to the SExtractor ASCII catalog. Must contain columns: `label`, `xcentroid`, `ycentroid`, `ellipticity`, `orientation`, `semimajor_sigma`, `kron_radius`, `mag_auto`. |
| `paramfile` | `str` | *(required)* | Stamp file from `create_stamp_file`. Expected columns: `pnum`, `px`, `py`, `pra`, `pdec`, `pxlo`, `pxhi`, `pylo`, `pyhi`, `pixscl`. |
| `mask_file` | `str` | *(required)* | Base name for the output mask FITS file (`.fits` is appended). |
| `mask_file_primary` | `str` | *(required)* | Base name for the primary-only mask FITS file (only written when `b == 1`). |
| `current` | `int` | *(required)* | 0-based index of the primary source in the catalog. |
| `scale` | `float` | *(required)* | Factor multiplying the Kron radius to define the source extent: `radius = scale * semimajor_sigma * kron_radius + offset`. |
| `offset` | `float` | *(required)* | Constant offset in pixels added to the scaled radius. |
| `limgal` | `float` | *(required)* | Magnitude threshold for secondary classification. A source is secondary if `mag_auto < primary_mag + limgal` AND it overlaps the primary. Otherwise it is tertiary. |
| `b` | `int` | *(required)* | Band index. If `b == 1`, an additional primary-only mask is written to `mask_file_primary + ".fits"`. |

### Returns

**`tuple[list[int], list[int]]`**

- **corner** (`list[int]`) -- Two-element list `[pxlo, pylo]` giving the lower-left corner of the postage stamp in the original image coordinates (0-based).
- **objects** (`list[int]`) -- Catalog indices of the primary source and all secondary (co-fitting) sources.

### Example

```python
from galfitx.mask import create_mask

corner, objects = create_mask(
    weight_name="weight.fits",
    seg_name="segmentation.fits",
    catalog_name="catalog.cat",
    paramfile="stamps",
    mask_file="mask_obj42",
    mask_file_primary="mask_primary_obj42",
    current=42,
    scale=3.0,
    offset=20.0,
    limgal=5.0,
    b=1,
)
```

---

## `dist_ellipse`

```python
dist_ellipse(n, xc, yc, ratio, angle, double=False)
```

Compute the elliptical distance from a centre for every pixel in a 2D grid. For each pixel, the Euclidean distance is calculated in a rotated and stretched coordinate system so that an ellipse becomes a circle. The output can be compared to a threshold radius to define elliptical masks.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n` | `int` or `tuple[int, int]` or `list[int, int]` | *(required)* | Grid dimensions. If an integer, creates an `n x n` array. If a tuple/list `(nx, ny)`, the output shape is `(ny, nx)`. |
| `xc` | `float` | *(required)* | X-coordinate of the ellipse centre (0-based, from left edge). |
| `yc` | `float` | *(required)* | Y-coordinate of the ellipse centre (0-based, from top edge). |
| `ratio` | `float` | *(required)* | Stretch factor along the rotated x-axis. Larger values produce more elongated ellipses. Computed as `sqrt((xtemp * ratio)**2 + ytemp**2)`. |
| `angle` | `float` | *(required)* | Rotation angle in degrees (counter-clockwise from positive x-axis). |
| `double` | `bool` | `False` | If `True`, use `float64` precision. If `False`, use `float32`. |

### Returns

**`np.ndarray`** -- 2D array of shape `(ny, nx)` with the elliptical distance at each pixel.

### Raises

`ValueError` -- If `n` is not an integer or a length-2 tuple/list.

### Example

```python
from galfitx.mask import dist_ellipse

# 100x100 distance map centred at (50,50), axis ratio 2, rotated 30 deg
dist = dist_ellipse((100, 100), 50, 50, 2.0, 30.0)
mask = dist <= 50  # True inside an ellipse of "radius" 50
```

---

## `prepare_galfitm`

```python
prepare_galfitm(
    image_stamp_list,
    catalog_name,
    obj_file,
    corner,
    constr_file,
    mask_file,
    mask_primary_file,
    out_file,
    stamp_file,
    sigma_file,
    psf_file,
    sky_file,
    conv_box,
    zero_list,
    wave_list,
    label_list,
    nband,
    conmaxre,
    plate_scl,
    current,
    objects=None,
    setup=None,
    use_cstsky=False,
    cstsky_list=None,
)
```

Generate a GALFITM input configuration file for multi-band fitting of a primary source and its neighbouring (secondary) sources. The function writes two files: the main GALFITM configuration (`obj_file`) and a constraint file (`constr_file`). It uses the SExtractor catalog for initial parameter guesses and can reuse results from previous GALFIT runs if they exist.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image_stamp_list` | `list[str]` | *(required)* | Base names (without extension) of the multi-band stamp FITS images. Actual filenames: `name + ".fits"`. |
| `catalog_name` | `str` | *(required)* | Path to the SExtractor ASCII catalog. Must contain: `label`, `xcentroid`, `ycentroid`, `flux_radius`, `orientation`, `ellipticity`, `mag_auto`. |
| `obj_file` | `str` | *(required)* | Output path for the GALFITM main configuration file. |
| `corner` | `list[int]` | *(required)* | `[pxlo, pylo]` lower-left corner of the stamp (0-based). |
| `constr_file` | `str` | *(required)* | Output path for the constraint file. |
| `mask_file` | `list[str]` | *(required)* | Base names of the mask FITS files for each band. |
| `mask_primary_file` | `str` | *(required)* | Base name of the primary-only mask FITS file. |
| `out_file` | `str` | *(required)* | Base name for the GALFITM output FITS file. |
| `stamp_file` | `str` | *(required)* | Path to the stamp file from `create_stamp_file`. |
| `sigma_file` | `list[str]` | *(required)* | Base names of the sigma/noise FITS files per band. Use `"none"` to skip. |
| `psf_file` | `list[str]` | *(required)* | Base names of the PSF FITS files per band. |
| `sky_file` | `list[str]` | *(required)* | Paths to sky-background text files per band. Each file should contain five numbers: `sky dsky skyrad sky_magobj flag`. |
| `conv_box` | `float` | *(required)* | Convolution box size in pixels (cast to int; same in x and y). |
| `zero_list` | `list[float]` | *(required)* | Magnitude zeropoints for each band. |
| `wave_list` | `list[str]` | *(required)* | Wavelengths (as strings) for each band. Used in GALFITM >= 4.0. |
| `label_list` | `list[str]` | *(required)* | Band labels (e.g., `"F090W"`) used in filenames and output. |
| `nband` | `int` | *(required)* | Number of bands. Must match the length of the per-band lists. |
| `conmaxre` | `float` | *(required)* | Maximum allowed half-light radius (Re) in pixels. |
| `plate_scl` | `float` | *(required)* | Plate scale in arcsec/pixel (same in x and y). |
| `current` | `int` | *(required)* | 0-based index of the primary source in the catalog. |
| `objects` | `list[int]` or `None` | `None` | Catalog indices of all sources (primary + secondaries) to fit. If `None`, no source components are written. |
| `setup` | `Any` | `None` | Configuration object. Required attributes: `outdir`, `version`, `do_restrict`, `restrict_frac_primary`, `conminn`, `conmaxn`, `conminm`, `conmaxm`, `cheb`, `gal_output`. |
| `use_cstsky` | `bool` | `False` | If `True`, use constant sky values from `cstsky_list` instead of reading `sky_file`. |
| `cstsky_list` | `list[float]` or `None` | `None` | Constant sky values per band (used when `use_cstsky=True`). |

### Returns

`None` -- Writes `obj_file` and `constr_file` to disk.

### Notes

- If an existing GALFIT output file is found at `./galfit/output{label}.fits`, its best-fit parameters are reused and the source is included as a static (fixed) component.
- The polynomial degree for each fitted parameter is set to `min(cheb_order + 1, maxdeg)`, where `maxdeg` may be reduced if the primary mask contains too many bad pixels (controlled by `setup.do_restrict` and `setup.restrict_frac_primary`).
- Sources with `mag_auto > 80` are assigned a magnitude of `brightest_mag + 2` to avoid unrealistic values.
- The initial effective radius is estimated from `flux_radius` using the relation `re = 10^(-0.79) * flux_radius^1.87`.
- Parameter bounds are enforced: `Re` in `[0.3, conmaxre]`, `n` in `[conminn, conmaxn]`, `q` in `[0.01, 1.0]`, `PA` in `[-360, 360]`.

### Example

```python
from galfitx.mask import prepare_galfitm

prepare_galfitm(
    image_stamp_list=["obj42_F090W_sci", "obj42_F150W_sci"],
    catalog_name="catalog.cat",
    obj_file="galfit_input/obj42.input",
    corner=[100, 200],
    constr_file="galfit_input/obj42.constr",
    mask_file=["obj42_F090W_mask", "obj42_F150W_mask"],
    mask_primary_file="obj42_mask_primary",
    out_file="obj42_galfit_out",
    stamp_file="stamps",
    sigma_file=["obj42_F090W_sigma", "obj42_F150W_sigma"],
    psf_file=["psf_F090W", "psf_F150W"],
    sky_file=["sky_F090W.txt", "sky_F150W.txt"],
    conv_box=60,
    zero_list=[25.0, 25.5],
    wave_list=["0.90", "1.50"],
    label_list=["F090W", "F150W"],
    nband=2,
    conmaxre=200.0,
    plate_scl=0.03,
    current=42,
    objects=[42, 17, 89],
    setup=setup_object,
)
```
