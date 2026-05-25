# Utilities (`galfitx/utils.py`)

Utility functions for visualization, model combination, catalog merging, and RGB image creation. These tools support the post-processing and quality-assessment stages of the GalfitX pipeline.

---

## `scaled_kron`

Generate a DS9 region file with scaled Kron ellipses for each source in a catalog. The ellipses are defined using Kron radii, semi-major/minor axes, and orientation angles from a SExtractor output table.

For each source, the ellipse dimensions are computed as:
- `major_aper = (scale * semimajor_sigma * kron_radius + offset) * pixel_scale`
- `minor_aper = (scale * semiminor_sigma * kron_radius + offset * (1 - ellipticity)) * pixel_scale`

The position angle is adjusted by subtracting a constant offset of 49.458529 degrees.

```python
scaled_kron(outtab, scale=1.5, offset=0, pixel_scale=0.06)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `outtab` | `astropy.table.Table` | (required) | Input source catalog. Must contain columns: `label`, `ra`, `dec`, `kron_radius`, `semimajor_sigma`, `semiminor_sigma`, `orientation`, `ellipticity`. |
| `scale` | `float` | `1.5` | Scaling factor applied to the Kron radius and sigma axes. |
| `offset` | `float` | `0` | Additional offset in arcseconds added to the major axis, and scaled by `(1 - ellipticity)` for the minor axis. |
| `pixel_scale` | `float` | `0.06` | Pixel scale in arcsec/pixel, used to convert pixel sizes to arcseconds. |

**Returns**

`None` -- Writes a DS9 region file named `scaled_kron.reg` to the current working directory.

---

## `combine_models`

Combine individual GALFIT model components into a single full-frame model image. Each source's model (from its GALFIT output) is placed at the correct position in the output image.

```python
combine_models(
    original_fits,
    stamp_catalog,
    galfit_dir,
    output_model_file,
    background_value=41,
)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `original_fits` | `str` | (required) | Path to the original FITS image. Its header and data shape are used for the output. |
| `stamp_catalog` | `str` | (required) | Path to an ASCII catalog containing stamp coordinates. Expected columns: ID, (unused), (unused), x1, x2, y1, y2 (zero-based inclusive pixel indices). Must have at least 7 columns. |
| `galfit_dir` | `str` | (required) | Directory containing GALFIT output files named `output<ID>.fits`. |
| `output_model_file` | `str` | (required) | Path where the combined model FITS image will be saved. |
| `background_value` | `float` | `41` | Value used to initialize the model image (baseline for every pixel). |

**Returns**

`None` -- Writes the combined model image to `output_model_file`.

**Notes**

- GALFIT model data is read from extension index 4 (the 5th HDU) of each output file.
- If a component file does not exist, a warning is printed and that source is skipped.
- The output uses the header from `original_fits`.

---

## `plot_oneband_example`

Plot the GALFIT fitting result for a single source in one band. Displays four panels: original data, best-fit model, residual, and mask. Optionally overlays scaled Kron ellipses from the detection catalog.

```python
plot_oneband_example(
    id,
    fits_dir="./galfit/",
    fits_prename="output",
    vmin=-1615.9,
    vmax=13771.5,
    add_scaled_kron=True,
    catalog_name="./sex/outcat",
    scale=1.1,
    offset=4,
)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `id` | `int` | (required) | Source identifier (used to construct filenames). |
| `fits_dir` | `str` | `"./galfit/"` | Directory containing GALFIT output files. |
| `fits_prename` | `str` | `"output"` | Prefix of the GALFIT output FITS file (before the ID). |
| `vmin` | `float` | `-1615.9` | Minimum display value for the image stretch. |
| `vmax` | `float` | `13771.5` | Maximum display value for the image stretch. |
| `add_scaled_kron` | `bool` | `True` | If True, overlay Kron ellipses and source IDs from the detection catalog. |
| `catalog_name` | `str` | `"./sex/outcat"` | Path to the detection catalog (ASCII). Required columns: `label`, `ra`, `dec`, `kron_radius`, `ellipticity`, `semimajor_sigma`, `semiminor_sigma`, `orientation`, `mag_auto`, `flux_radius`. |
| `scale` | `float` | `1.1` | Scaling factor for Kron ellipse size. |
| `offset` | `float` | `4` | Additional offset in pixels added to the ellipse axes. |

**Returns**

`None` -- Displays the plot using `plt.show()`.

**Notes**

- Panel 1 (data): Shows the original cutout with Kron ellipses and source IDs. Title shows `mag`, `re`, and `q` from the catalog.
- Panel 2 (model): Shows the best-fit model. Title shows fitted Sersic parameters (`mag`, `re`, `n`, `q`).
- Panel 3 (residual): Shows data minus model.
- Panel 4 (mask): Shows the binary mask with Kron ellipses overlaid.

---

## `create_rgb_image`

Create an RGB composite image from three single-band FITS files with asinh stretch. Each band is clipped, normalized, stretched, and scaled by a color factor before being combined.

```python
create_rgb_image(
    red_file,
    green_file,
    blue_file,
    vmin=-1615.9,
    vmax=13771.5,
    color_factors=(1.0, 0.8, 1.0),
    a_stretch=0.1,
    show=True,
    dpi=200,
)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `red_file` | `str` | (required) | Path to the FITS file for the red channel. |
| `green_file` | `str` | (required) | Path to the FITS file for the green channel. |
| `blue_file` | `str` | (required) | Path to the FITS file for the blue channel. |
| `vmin` | `float` | `-1615.9` | Minimum value for clipping. |
| `vmax` | `float` | `13771.5` | Maximum value for clipping. |
| `color_factors` | `tuple[float, float, float]` | `(1.0, 0.8, 1.0)` | Scaling factors for the (R, G, B) channels. |
| `a_stretch` | `float` | `0.1` | Asinh stretch parameter. Controls the nonlinearity of the stretch. |
| `show` | `bool` | `True` | If True, display the image using `plt.show()`. |
| `dpi` | `int` | `200` | Resolution in dots per inch for the figure. |

**Returns**

| Type | Description |
|---|---|
| `tuple[np.ndarray, matplotlib.figure.Figure]` | A tuple of `(rgb_image, fig)`, where `rgb_image` is a 3D array of shape `(ny, nx, 3)` and `fig` is the matplotlib figure object. |

---

## `combine_catalogs`

Combine catalogs from the five GalfitX pipeline stages into a single FITS file. The stages are:

1. **Source detection** (SExtractor catalog, used as reference)
2. **Pure imaging fitting** (GalfitS without SED)
3. **Photometric redshift** (EAZy grouped outputs)
4. **Image+SED fitting** (GalfitS with SED)

All catalogs are merged by object identifier (the `label` column). Missing values are filled with `-999.0`.

```python
combine_catalogs(
    filter_list,
    sex_cat,
    gs_dir,
    eazy_dir,
    gssed_dir,
    output_file="combined_catalog.fits",
)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `filter_list` | `list[str]` | (required) | List of filter names, e.g. `["nircam_f444w", ...]`. Used to construct parameter column names. |
| `sex_cat` | `str` | (required) | Path to the source detection catalog (ASCII). Must contain a `label` column as the object identifier. |
| `gs_dir` | `str` | (required) | Directory containing pure imaging `.gssummary` files named `obj<label>.gssummary`. |
| `eazy_dir` | `str` | (required) | Root directory containing EAZy grouped outputs. Subdirectories named by group IDs each contain a `photz.zout` file with `id`, `z_peak`, `z_spec` columns. |
| `gssed_dir` | `str` | (required) | Directory containing Image+SED `.gssummary` files named `obj<label>.gssummary`. |
| `output_file` | `str` | `"combined_catalog.fits"` | Filename for the output combined FITS catalog. |

**Returns**

`None` -- Writes the combined catalog to `output_file`.

**Output Columns**

The combined catalog contains:
- All columns from the source detection catalog
- Pure imaging parameters: `Mag_obj0_<filter>` for each filter, `obj0_Re`, `obj0_n`, `obj0_ang`, `obj0_axrat`
- Photometric redshift: `z_peak`, `z_spec`
- Image+SED physical parameters: `obj0_Z_value`, `obj0_Av_value`, `logM_obj0`, `AVbump_obj0`, `obj0_f_cont_bin1` through `obj0_f_cont_bin5`, `logsfr_obj0`
