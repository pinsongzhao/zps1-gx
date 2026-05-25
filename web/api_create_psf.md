# PSF Construction (`galfitx/create_psf.py`)

Module for building point-spread functions (PSFs) using stacked star cutouts and PSFEx. Provides tools for star selection, PSF stacking, PSFEx configuration, and position-dependent PSF reconstruction.

---

## `build_stacked_psf`

Build a stacked PSF from selected stars in a science image using the `psfr` package.

Stars are selected from a SExtractor catalog using the following criteria:
- Elongation < 1.5
- `class_star` > 0.9
- `combined_flags` < 2
- Magnitude within `[mag_bright_limit, mag_faint_limit]`

The selected stars are cut out, masked (using the segmentation map to exclude neighbouring sources), and stacked with oversampled recentering. The final PSF is saved as `psf.fits` in the output directory.

```python
build_stacked_psf(
    sci_image,
    catalog_file,
    seg_file,
    output_dir="./star_cutouts",
    mag_bright_limit=19.0,
    mag_faint_limit=21.0,
    cutout_size=71,
    oversampling=3,
    n_recenter=10,
    num_iteration=20,
    save_fits=True,
    save_figures=True,
    plot_overview=True,
    plot_radial_profile=True,
)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `sci_image` | `str` | (required) | Path to the science FITS image. |
| `catalog_file` | `str` | (required) | Path to the SExtractor output catalogue (ASCII format). |
| `seg_file` | `str` | (required) | Path to the SExtractor segmentation map. |
| `output_dir` | `str` | `"./star_cutouts"` | Directory where cutouts, figures, and the final PSF are saved. |
| `mag_bright_limit` | `float` | `19.0` | Bright magnitude limit for star selection. |
| `mag_faint_limit` | `float` | `21.0` | Faint magnitude limit for star selection. |
| `cutout_size` | `int` | `71` | Size of the square stamp cutout in pixels. |
| `oversampling` | `int` | `3` | Oversampling factor for PSF stacking. |
| `n_recenter` | `int` | `10` | Number of recentering iterations during stacking. |
| `num_iteration` | `int` | `20` | Number of iterations for the stacking algorithm. |
| `save_fits` | `bool` | `True` | If True, save individual star cutouts and the final PSF as FITS files. |
| `save_figures` | `bool` | `True` | If True, save overview and radial profile figures as PNG files. |
| `plot_overview` | `bool` | `True` | If True, generate the overview plot of all star cutouts. |
| `plot_radial_profile` | `bool` | `True` | If True, generate the 1D radial profile plot. |

**Returns**

`None` -- Saves the stacked PSF to `output_dir/psf.fits`.

---

## `build_input_ldac`

Build a PSFEx input LDAC catalog from selected stars. Creates a FITS_LDAC file with VIGNET data and updated header keywords that can be directly used by PSFEx.

```python
build_input_ldac(
    sci_image,
    catalog_file,
    seg_file,
    output_ldac,
    mag_bright_limit=19.0,
    mag_faint_limit=21.0,
    cutout_size=71,
    mask_value=-1e30,
    save_cutouts=True,
    cutouts_dir="./star_cutouts",
    template_file="./output_assoc_temp.cat",
    fwhm_arcsec=0.13,
    mag_zeropoint=28.0,
    ref_pixel_scale=0.074,
)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `sci_image` | `str` | (required) | Path to the science FITS image. |
| `catalog_file` | `str` | (required) | Path to the SExtractor output catalogue (ASCII). |
| `seg_file` | `str` | (required) | Path to the SExtractor segmentation map. |
| `output_ldac` | `str` | (required) | Path for the output PSFEx LDAC catalog. |
| `mag_bright_limit` | `float` | `19.0` | Bright magnitude limit for star selection. |
| `mag_faint_limit` | `float` | `21.0` | Faint magnitude limit for star selection. |
| `cutout_size` | `int` | `71` | Size of the square cutout in pixels. Must match VIGNET dimensions expected in the template. |
| `mask_value` | `float` | `-1e30` | Value used to mask neighbouring sources in the cutout. |
| `save_cutouts` | `bool` | `True` | If True, save individual star cutouts as FITS files. |
| `cutouts_dir` | `str` | `"./star_cutouts"` | Directory where cutout FITS files will be saved. |
| `template_file` | `str` | `"./output_assoc_temp.cat"` | Path to an existing SExtractor ASSOC output (LDAC template). Must have matching column structure. |
| `fwhm_arcsec` | `float` | `0.13` | Approximate FWHM of stars in arcsec, used to update `SEXSFWHM` keyword. |
| `mag_zeropoint` | `float` | `28.0` | Magnitude zeropoint, used to update `SEXMGZPT` keyword. |
| `ref_pixel_scale` | `float` | `0.074` | Pixel scale in arcsec/pixel, used to update `SEXPXSCL` keyword. |

**Returns**

`None` -- Writes the LDAC catalog to `output_ldac`.

**Notes**

Updates the following header keywords in the LDAC IMHEAD: `NAXIS1`, `NAXIS2`, `SEXBKGND`, `SEXBKDEV`, `SEXTHLD`, `SEXATHLD`, `SEXNFIN`, `SEXPXSCL`, `SEXSFWHM`, `SEXMGZPT`.

---

## `psfex_config`

Generate a PSFEx configuration file with sensible defaults. Only the most commonly modified parameters are exposed.

```python
psfex_config(
    output_config="config.psfex",
    *,
    psf_sampling=0.5,
    psf_size=(141, 141),
    sample_autoselect=True,
    sample_fwhmrange="1, 5",
    sample_minsn=10,
    psfvar_degrees=0,
)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `output_config` | `str` | `"config.psfex"` | Path where the configuration file will be written. |
| `psf_sampling` | `float` | `0.5` | Sampling step in pixel units (0.0 = auto). |
| `psf_size` | `tuple[int, int]` | `(141, 141)` | Image size of the PSF model (width, height). |
| `sample_autoselect` | `bool` | `True` | Automatically select the FWHM. |
| `sample_fwhmrange` | `str` | `"1, 5"` | Allowed FWHM range for source selection. |
| `sample_minsn` | `int` | `10` | Minimum signal-to-noise for a source to be used. |
| `psfvar_degrees` | `int` | `0` | Polynomial degree for spatial PSF variation. 0 = constant PSF, 1 = linear, 2 = quadratic. Use with `run_psfex(sample_positions=...)`. |

**Returns**

`None` -- Writes the configuration file to disk.

---

## `interpolate_psf_at_position`

Reconstruct the PSF at a specific position from a PSFEx model using polynomial interpolation. Supports arbitrary polynomial order.

```python
interpolate_psf_at_position(psf_file, x, y, normalize=True)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `psf_file` | `str` | (required) | Path to the PSFEx output file (`.psf`). |
| `x` | `float` | (required) | X coordinate in image pixels (column), **1-indexed** (same as SExtractor's `X_IMAGE`). |
| `y` | `float` | (required) | Y coordinate in image pixels (row), **1-indexed** (same as SExtractor's `Y_IMAGE`). |
| `normalize` | `bool` | `True` | If True, normalize the PSF to unit flux. |

**Returns**

| Type | Description |
|---|---|
| `np.ndarray` | Reconstructed PSF as a 2D numpy array. |

**Raises**

- `FileNotFoundError` -- If `psf_file` does not exist.
- `RuntimeError` -- If the PSFEx file has unexpected structure.
- `ValueError` -- If `POLNAXIS != 2` (only 2D spatial variations are supported).

**Notes**

The interpolation uses polynomial basis terms stored in the PSFEx file. For degree `d`, the basis terms are `[x^nx * y^ny for ny in range(d+1) for nx in range(d+1-ny)]`. For `d=1` this gives `[1, x, y]`, and for `d=2` this gives `[1, x, x^2, y, xy, y^2]`.

The PSF is reconstructed as `PSF(x, y) = sum(P[i] * PSF_MASK[i])` where `P[i]` are polynomial terms evaluated at normalized coordinates `(x - POLZERO1) / POLSCAL1` and `(y - POLZERO2) / POLSCAL2`.

---

## `run_psfex`

Run PSFEx on an LDAC catalog and optionally save the extracted PSF and radial profile plots.

```python
run_psfex(
    catalog,
    config,
    output_dir="./",
    save_psf_fits=True,
    psf_sampling=0.5,
    plot_radial_profile=True,
    sample_positions=None,
)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `catalog` | `str` | (required) | Path to the input FITS_LDAC catalogue (as produced by `build_input_ldac`). |
| `config` | `str` | (required) | Path to the PSFEx configuration file. |
| `output_dir` | `str` | `"./"` | Directory for output files. |
| `save_psf_fits` | `bool` | `True` | If True, save the extracted PSF to a FITS file. For position-dependent PSF (`POLDEG1 > 0`), saves PSFs at the positions specified in `sample_positions`. |
| `psf_sampling` | `float` | `0.5` | PSF sampling step used to compute oversampling factor (`oversampling = 1 / psf_sampling`). |
| `plot_radial_profile` | `bool` | `True` | If True, generate and save the 1D radial profile plot. |
| `sample_positions` | `list[tuple[int, int]]` or `None` | `None` | Positions to sample PSF at when `POLDEG1 > 0`. Each tuple is `(x, y)` in 1-indexed image pixels. If `None` and `POLDEG1 > 0`, no FITS PSF files are saved. For constant PSF (`POLDEG1 == 0`), this parameter is ignored. |

**Returns**

| Type | Description |
|---|---|
| `str` | Full path to the generated PSF model file (`.psf`). Use `interpolate_psf_at_position` to reconstruct PSF at any position. |

**Raises**

- `FileNotFoundError` -- If `catalog` or `config` does not exist, or if the PSF model file is not found after running PSFEx.
- `RuntimeError` -- If PSFEx returns a non-zero exit status.
