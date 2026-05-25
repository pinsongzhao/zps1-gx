# Postage Stamp

::: galfitx.postage_stamp

Module for creating stamp files and cutting postage-stamp sub-images from astronomical FITS images.

---

## `create_stamp_file`

```python
create_stamp_file(
    image_name,
    catalog_name,
    sizefac=2.5,
    outfile="stamps",
    pixel_scale=0.03,
)
```

Generate a stamp file containing bounding-box coordinates for each object in the catalog. The function reads a FITS image to determine its dimensions, then computes a rectangular stamp region for each source based on its size, ellipticity, and orientation. The stamp is centered on the source centroid and its extent is controlled by `sizefac`.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image_name` | `str` | *(required)* | Path to the detection FITS image. Used only to determine image dimensions. |
| `catalog_name` | `str` | *(required)* | Path to the ASCII SExtractor catalog. Must contain columns: `label`, `xcentroid`, `ycentroid`, `ra`, `dec`, `orientation`, `semimajor_sigma`, `kron_radius`, `ellipticity`. |
| `sizefac` | `float` | `2.5` | Scaling factor for the stamp size relative to the object's characteristic radius. The effective radius is `semimajor_sigma * kron_radius`, and the stamp extends `sizefac` times that in each direction. |
| `outfile` | `str` | `"stamps"` | Name of the output text file. |
| `pixel_scale` | `float` | `0.03` | Pixel scale in arcsec/pixel. Written into the output file for reference; not used in computation. |

### Returns

`None` -- Writes the stamp file to disk.

### Output Format

Each line in the output file has the following space-separated columns:

```
ID  xcentroid  ycentroid  ra  dec  xlo  xhi  ylo  yhi  pixel_scale
```

All coordinate values use 1-based indexing. Subtract 1 before indexing into Python arrays.

### Notes

The stamp bounding box is computed as follows:

1. The object's effective radius is `rad = semimajor_sigma * kron_radius`.
2. Two trial extents (x, y) are computed using the source orientation and ellipticity.
3. The longer extent is assigned to the major-axis direction based on the position angle.
4. The final box is centered on `(xcentroid, ycentroid)` and scaled by `sizefac`.

### Example

```python
from galfitx.postage_stamp import create_stamp_file

create_stamp_file(
    image_name="science.fits",
    catalog_name="my_catalog",
    sizefac=2.5,
    outfile="stamps",
    pixel_scale=0.03,
)
```

---

## `cut_stamps`

```python
cut_stamps(
    image_name,
    outdir,
    label,
    stampfile="stamps",
    cut_list=None,
    ps=0.03,
)
```

Cut postage-stamp sub-images from a FITS image using the coordinates defined in a stamp file (produced by `create_stamp_file`). Handles WCS reprojection when the target image has a different pixel scale than the detection image.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image_name` | `str` | *(required)* | Path to the input FITS image to cut stamps from. |
| `outdir` | `str` | *(required)* | Output directory for the cutout stamp FITS files. Created automatically if it does not exist. |
| `label` | `str` | *(required)* | Label string inserted into the output filename (e.g., a band name like `"F090W"`). |
| `stampfile` | `str` | `"stamps"` | Path to the stamp file produced by `create_stamp_file`. |
| `cut_list` | `list[int]` or `None` | `None` | List specifying which sources to cut (`1` = cut, `0` = skip). If `None`, all sources are cut. Must have the same length as the stamp file. |
| `ps` | `float` | `0.03` | Pixel scale of the current input image in arcsec/pixel. Used to convert bounding-box coordinates from the detection pixel scale. |

### Returns

`None` -- Writes FITS stamp files to `outdir`.

### Output Format

Each stamp is saved as:

```
obj{ID}_{label}sci.fits
```

The output FITS header includes the following custom keywords:

| Header Keyword | Description |
|----------------|-------------|
| `OBJ_ID` | Object identifier |
| `RA` | Right ascension in degrees |
| `DEC` | Declination in degrees |
| `ORG_X` | Original X coordinate in the input image |
| `ORG_Y` | Original Y coordinate in the input image |
| `CUT_XMIN` | Cutout X min in the input image (0-based) |
| `CUT_XMAX` | Cutout X max in the input image (0-based) |
| `CUT_YMIN` | Cutout Y min in the input image (0-based) |
| `CUT_YMAX` | Cutout Y max in the input image (0-based) |

### Notes

The pixel-scale conversion works as follows: for each source, the WCS is used to find the source position on the current image (`x0`, `y0`). The stamp boundaries are then rescaled from the detection pixel scale to the current image pixel scale:

```
xlo1 = x0 - (ximg - xlo) * pixel_scale_stamp / ps
xhi1 = (xhi - ximg) * pixel_scale_stamp / ps + x0
```

Sources whose stamps fall entirely outside the image are skipped.

### Example

```python
from galfitx.postage_stamp import cut_stamps

# Cut stamps for all sources in the F090W band
cut_stamps(
    image_name="F090W_sci.fits",
    outdir="./stamps/F090W/",
    label="F090W",
    stampfile="stamps",
    ps=0.03,
)

# Cut stamps for a subset of sources only
cut_stamps(
    image_name="F090W_sci.fits",
    outdir="./stamps/F090W/",
    label="F090W",
    stampfile="stamps",
    cut_list=[1, 0, 1, 0, 0, 1],  # only cut sources 0, 2, 5
    ps=0.03,
)
```
