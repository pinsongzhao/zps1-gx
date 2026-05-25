# Sky Calculation

::: galfitx.sky_calculation

Module for computing local sky backgrounds around astronomical sources using elliptical annular apertures and sigma-clipped statistics.

---

## `create_skymap`

```python
create_skymap(
    weight_name,
    catalog_name,
    skymap_name,
    scale,
    offset,
)
```

Build a sky-map array where each pixel records how many scaled Kron ellipses (from the SExtractor catalog) cover it. The map is used by `getsky` to identify pixels that are safe for local sky estimation.

### Pixel Values

| Value | Meaning |
|-------|---------|
| `-1` | No-flux pixel (weight == 0). |
| `0` | Blank sky (no overlapping ellipse). Safe for sky estimation. |
| `> 0` | Overlap count. Not safe for sky estimation. |

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `weight_name` | `str` | *(required)* | Path to the FITS weight image. Used to identify no-flux pixels and obtain image dimensions and header. |
| `catalog_name` | `str` | *(required)* | Path to the ASCII SExtractor catalog. Must contain columns: `ellipticity`, `xcentroid`, `ycentroid`, `orientation`, `semimajor_sigma`, `kron_radius`. |
| `skymap_name` | `str` | *(required)* | Output filename for the sky-map FITS file. |
| `scale` | `float` | *(required)* | Scaling factor applied to the Kron radius and sigma. The ellipse radius is computed as `scale * semimajor_sigma * kron_radius + offset`. |
| `offset` | `float` | *(required)* | Constant offset added to the ellipse size in pixels. |

### Returns

**`np.ndarray` (2D, int)** -- The sky-map array with the same shape as the weight image.

### Example

```python
from galfitx.sky_calculation import create_skymap

skymap = create_skymap(
    weight_name="weight.fits",
    catalog_name="catalog.cat",
    skymap_name="skymap.fits",
    scale=3.0,
    offset=20.0,
)
```

---

## `getsky`

```python
getsky(
    obj_idx,
    catalog_name,
    image_name,
    skymap_name,
    skyfile,
    dstep=8,
    wstep=8,
    gap=8,
    nslope=5,
    global_sky=43.3,
    global_sigsky=585.72,
)
```

Estimate the local sky background for a given object using a series of elliptical annular apertures centred on the source. For each annulus, pixel values from sky-map-safe regions (value 0) are collected, sigma-clipped, and optionally fitted with a Gaussian histogram to determine the sky value. The search continues outward until the slope of sky vs. radius becomes positive (indicating contamination by nearby sources), and the minimum measured sky is adopted as the final value.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `obj_idx` | `int` | *(required)* | 0-based index of the object in the catalog. |
| `catalog_name` | `str` | *(required)* | Path to the ASCII SExtractor catalog. Must contain: `ellipticity`, `xcentroid`, `ycentroid`, `orientation`, `semimajor_sigma`, `kron_radius`, `mag_auto`. |
| `image_name` | `str` | *(required)* | Path to the science FITS image. |
| `skymap_name` | `str` | *(required)* | Path to the sky-map FITS image (produced by `create_skymap`). Pixels with value 0 are considered safe. |
| `skyfile` | `str` | *(required)* | Output text file for the result. A single line with five values: `sky_value sky_sigma sky_radius object_mag flag`. |
| `dstep` | `int` | `8` | Radial step between successive annuli in pixels. |
| `wstep` | `int` | `8` | Width of each annulus in pixels. |
| `gap` | `int` | `8` | Gap between the Kron radius and the first annulus in pixels. |
| `nslope` | `int` | `5` | Number of successive annuli used to detect a positive slope in sky vs. radius. Once a positive slope is detected `nslope` times, the search stops. |
| `global_sky` | `float` | `43.3` | Fallback sky value used when the measurement fails (e.g., insufficient sky pixels). |
| `global_sigsky` | `float` | `585.72` | Fallback sky uncertainty used when the measurement fails. |

### Returns

`None` -- Writes the result to `skyfile`.

### Output Format

The output file contains a single line with five space-separated values:

```
sky_value  sky_sigma  sky_radius  object_mag  flag
```

### Flag Values

The flag is a bitmask encoding conditions encountered during estimation:

| Bit | Value | Meaning |
|-----|-------|---------|
| 0 | 0 | Normal. |
| 1 | 1 | Kron radius exceeds maximum distance to any sky pixel. |
| 2 | 4 | Outer radius exceeded the maximum distance to a sky pixel. |
| 3 | 8 | Too few valid measurements; fallback to global sky. |
| 4 | 32 | No sky pixels available in the image. |
| 5 | 64 | Gaussian fitting failed; sigma-clipped statistics used instead. |

Multiple bits can be set simultaneously (e.g., `flag = 5` means bits 0 and 2).

### Notes

- The function relies on `dist_ellipse` to compute elliptical distances for each annulus.
- For each annulus with at least 5 sky pixels, a 3-sigma-clipped mean and standard deviation are computed. If enough unique pixel values remain, a Gaussian is fitted to the histogram to determine the peak and its uncertainty.
- The final sky value is the minimum among all measured annuli (to avoid contamination).
- If no sky pixels exist in the image (skymap has no zeros), the function writes a flag-32 result immediately.

### Example

```python
from galfitx.sky_calculation import create_skymap, getsky

# Build the sky map first
skymap = create_skymap(
    weight_name="weight.fits",
    catalog_name="catalog.cat",
    skymap_name="skymap.fits",
    scale=3.0,
    offset=20.0,
)

# Estimate sky for object index 42
getsky(
    obj_idx=42,
    catalog_name="catalog.cat",
    image_name="science.fits",
    skymap_name="skymap.fits",
    skyfile="sky_obj42.txt",
    dstep=8,
    wstep=8,
    gap=8,
    nslope=5,
)
```

---

## `dist_ellipse`

```python
dist_ellipse(n, xc, yc, ratio, angle, double=False)
```

Compute the elliptical distance from a centre for every pixel in a 2D grid. This is the same implementation used in `galfitx.mask`.

For each pixel, the Euclidean distance is calculated in a rotated and stretched coordinate system so that an ellipse becomes a circle. The output can be compared to a threshold radius to define elliptical masks.

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
from galfitx.sky_calculation import dist_ellipse

# 200x200 distance map, centred at (100, 100), axis ratio 1.5, rotated 45 deg
dist = dist_ellipse((200, 200), 100, 100, 1.5, 45.0)
mask = dist <= 80  # True inside an ellipse of "radius" 80
```
