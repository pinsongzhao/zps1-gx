# Demo: PSF Construction

GalfitX includes three demo scripts for generating Point Spread Functions (PSFs), each suited to different observing conditions. A high-quality PSF is essential for accurate structural fitting -- any mismatch between the true PSF and the model PSF will bias measured galaxy sizes, Sersic indices, and other structural parameters.

All three scripts live in `galfitx/` and use the `create_psf` module.

---

## Prerequisites

Before running any PSF demo, you need:

- A SExtractor catalog (`outcat`) and segmentation map (`outseg.fits`) for your science image. Run the source detection step first (see the full pipeline demo).
- The `psfr` package installed (`pip install psfr`) for stacked PSF construction.
- The `psfex` binary installed for PSFEx-based methods.
- A science FITS image with properly calibrated photometry.

---

## Demo 1: Stacked PSF (demo_stackpsf.py)

**File**: `galfitx/demo_stackpsf.py`

This approach selects isolated point sources (stars) from your image, extracts cutouts, and stacks them using the PSFR package to produce a single high-S/N PSF.

### When to use

- Small to medium fields where the PSF is approximately constant across the detector
- When you want a simple, reliable PSF without modeling spatial variation
- When you have enough bright, unsaturated stars in the field (typically 10+)

### Full script walkthrough

```python
import os
from create_psf import build_stacked_psf
```

The only import needed is `build_stacked_psf` from the `create_psf` module.

### Input parameters

```python
sci_image    = "sci_i.fits"
catalog_file = "outcat"
seg_file     = "outseg.fits"
output_dir   = "./stacked_psf_output"
```

Three input files are required:
- **sci_image**: The science FITS image (can be any band)
- **catalog_file**: SExtractor ASCII catalog from the detection step
- **seg_file**: SExtractor segmentation map

### Star selection criteria

```python
mag_bright_limit = 19.0
mag_faint_limit  = 21.0
```

Stars are selected by magnitude range. The bright limit avoids saturated stars (which have corrupted profiles). The faint limit ensures adequate S/N for accurate PSF construction. Adjust these for your image depth -- deeper images can push the faint limit fainter.

Inside `build_stacked_psf()`, the actual selection filters on four catalog columns:

```python
outtab1 = outtab[
    (outtab['elongation'] < 1.5) &      # Round, not elongated
    (outtab['class_star'] > 0.9) &      # High star-galaxy classifier score
    (outtab['combined_flags'] < 2) &    # No bad pixel flags
    (outtab['mag_auto'] > mag_bright_limit) &
    (outtab['mag_auto'] < mag_faint_limit)
]
```

- **elongation < 1.5**: Stars should be round. Galaxies with `elongation > 1.5` are excluded.
- **class_star > 0.9**: SExtractor's neural-network star/galaxy classifier. Values near 1.0 indicate point sources.
- **combined_flags < 2**: Exclude sources with bad pixels, saturation, or truncation near the image edge.

### Cutout parameters

```python
cutout_size = 71  # pixels
```

The square stamp size for each star cutout. This should be large enough to capture the PSF wings (at least 3-5 times the FWHM). For JWST/NIRCam at 0.04"/px with FWHM ~ 0.16" (4 pixels), a 71-pixel cutout captures the wings out to ~ 3 arcsec.

### Stacking parameters

```python
oversampling  = 3
n_recenter    = 10
num_iteration = 20
```

- **oversampling=3**: The output PSF is sampled at 3x the input pixel resolution. For a 0.04"/px input, the output PSF has 0.0133"/px. This oversampled PSF is later downsampled to match each band's pixel scale during fitting.
- **n_recenter=10**: During stacking, each star cutout is recentered (by measuring and subtracting the centroid) up to 10 times to ensure perfect alignment.
- **num_iteration=20**: The stacking algorithm iterates 20 times, progressively improving the PSF model by downweighting outliers.

### Running the stack

```python
build_stacked_psf(
    sci_image=sci_image, catalog_file=catalog_file, seg_file=seg_file,
    output_dir=output_dir,
    mag_bright_limit=mag_bright_limit, mag_faint_limit=mag_faint_limit,
    cutout_size=cutout_size,
    oversampling=oversampling, n_recenter=n_recenter,
    num_iteration=num_iteration,
    save_fits=True, save_figures=True,
    plot_overview=True, plot_radial_profile=True,
)
```

### Output products

| File | Description |
|---|---|
| `star_cutout_*.fits` | Individual star cutouts (one per selected star) |
| `star_cutouts_overview.png` | Grid plot showing all selected star cutouts |
| `radial_profiles.png` | 1D radial profile of the stacked PSF, with measured FWHM |
| `stacked_psf.fits` | The final stacked PSF image |

The measured FWHM is printed to the console and should be compared to the expected value for your instrument/filter. A significant discrepancy may indicate that non-stellar sources were included.

---

## Demo 2: PSFEx Constant PSF (demo_psfex_const_psf.py)

**File**: `galfitx/demo_psfex_const_psf.py`

This approach uses PSFEx (PSF Extractor) to model the PSF as a single, spatially constant function. It fits an analytical basis model to all selected stars simultaneously, producing a clean PSF with well-characterized properties.

### When to use

- Uniform PSF across a small field of view
- When you want a mathematically smooth PSF (not pixel-noisy like a stacked PSF)
- When the field is small enough that optical distortions and PSF variation are negligible

### Three-step workflow

#### Step 1: Build LDAC catalog

```python
from create_psf import build_input_ldac

build_input_ldac(
    sci_image=sci_image, catalog_file=catalog_file, seg_file=seg_file,
    output_ldac=output_ldac,
    mag_bright_limit=mag_bright_limit, mag_faint_limit=mag_faint_limit,
    cutout_size=cutout_size,
    save_cutouts=True,
    cutouts_dir=os.path.join(output_dir, "star_cutouts"),
    template_file=template_file,
    fwhm_arcsec=fwhm_arcsec,
    mag_zeropoint=mag_zeropoint,
    ref_pixel_scale=ref_pixel_scale,
)
```

PSFEx requires input in the LDAC (FITS binary table) catalog format. `build_input_ldac()` converts your SExtractor catalog to this format, extracting star cutouts and populating the required header keywords:

- **template_file**: A reference LDAC catalog that provides the table structure. PSFEx expects specific column names and extensions.
- **fwhm_arcsec=0.13**: The approximate FWHM in arcseconds. PSFEx uses this for initial star selection validation.
- **mag_zeropoint=28.0**: The magnitude zeropoint of the image, needed for S/N calculations.
- **ref_pixel_scale=0.074**: Pixel scale in arcsec/pixel. Used to convert between pixel and angular units.

The star selection criteria (elongation, class_star, flags, magnitude range) are applied inside `build_input_ldac()` using the same logic as the stacked PSF.

#### Step 2: Generate PSFEx configuration

```python
from create_psf import psfex_config

config_file = os.path.join(output_dir, "config.psfex")
psfex_config(
    output_config=config_file,
    psf_sampling=psf_sampling,
    psf_size=psf_size,
    sample_autoselect=True,
    sample_fwhmrange="1, 5",
    sample_minsn=sample_minsn,
    psfvar_degrees=0,  # Constant PSF -- no spatial variation
)
```

Key parameters:

- **psf_sampling=0.5**: The sampling step for the PSF model in pixels. A value of 0.5 means the PSF is modeled at 2x the input pixel resolution. Setting it to 0.0 lets PSFEx auto-determine the sampling.
- **psf_size=(141, 141)**: The output PSF dimensions in pixels. Should be large enough to contain the full PSF including wings.
- **sample_autoselect=True**: Let PSFEx automatically select valid stars from the LDAC catalog.
- **sample_fwhmrange="1, 5"**: Only use stars with measured FWHM between 1 and 5 pixels. This rejects cosmic rays (too narrow) and galaxies (too broad).
- **sample_minsn=10**: Minimum signal-to-noise ratio for star selection.
- **psfvar_degrees=0**: The polynomial degree for spatial variation. **Zero means the PSF is constant across the entire field.** This is the defining parameter for this demo.

#### Step 3: Run PSFEx

```python
from create_psf import run_psfex

psf_file = run_psfex(
    catalog=output_ldac,
    config=config_file,
    output_dir=output_dir,
)
```

`run_psfex()` invokes the PSFEx binary with the specified catalog and configuration. The output is a FITS file containing the PSF model. For a constant PSF (`psfvar_degrees=0`), the file contains a single PSF image.

### Output products

| File | Description |
|---|---|
| `psfex.cat` | LDAC catalog with star cutouts |
| `config.psfex` | PSFEx configuration file |
| `star_cutouts/` | Individual star cutout images |
| PSF FITS file | The final constant PSF model |

The PSF FITS file can be used directly in GalfitX by adding it to `psf_list`.

---

## Demo 3: PSFEx Variable PSF (demo_psfex_var_psf.py)

**File**: `galfitx/demo_psfex_var_psf.py`

This is the most sophisticated PSF construction method. It models the PSF as a spatially varying function, allowing the PSF shape to change across the detector field. This is essential for wide-field images where optical distortions, telescope jitter, and thermal effects cause the PSF to vary.

### When to use

- Large mosaics or wide-field images
- When PSF size or shape visibly varies across the field
- When fitting sources across the full extent of a large image
- For the highest-accuracy structural measurements

### Workflow

Steps 1 and 2 are the same as the constant PSF demo, with one critical difference:

#### Step 2 (modified): Configuration with spatial variation

```python
psfex_config(
    output_config=config_file,
    psf_sampling=psf_sampling,
    psf_size=psf_size,
    sample_autoselect=True,
    sample_fwhmrange="1, 5",
    sample_minsn=sample_minsn,
    psfvar_degrees=1,  # Linear spatial variation
)
```

**`psfvar_degrees=1`** tells PSFEx to model the PSF as a linear function of position across the image. The PSF basis functions are modulated by polynomials in `(x, y)` up to degree 1. For even larger fields, you can use `psfvar_degrees=2` (quadratic variation) at the cost of needing more stars.

#### Step 3: Define sampling positions

```python
sample_positions = [
    (1000, 1000),    # Center
    (100, 100),      # Bottom-left corner
    (1900, 100),     # Bottom-right corner
    (100, 1900),     # Top-left corner
    (1900, 1900),    # Top-right corner
]
```

To evaluate and visualize the spatial variation, the PSF is sampled at five positions: the image center and four corners. These positions are in pixel coordinates of the science image.

#### Step 4: Run PSFEx with sampling

```python
psf_file = run_psfex(
    catalog=output_ldac,
    config=config_file,
    output_dir=output_dir,
    sample_positions=sample_positions,
)
```

The `sample_positions` argument tells `run_psfex()` to extract the PSF at each specified position. For a variable PSF, the output FITS file contains a data cube where each plane is the PSF at a different position.

#### Step 5: Visualize PSF variation

```python
from create_psf import interpolate_psf_at_position

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for idx, (x, y) in enumerate(sample_positions):
    psf = interpolate_psf_at_position(psf_file, x, y, normalize=False)

    # Rebin to regular sampling
    if oversampling > 1:
        psf = oversampled2regular(psf, oversampling)
    psf = psf / np.sum(psf)

    norm = simple_norm(psf, 'asinh')
    axes[idx].imshow(psf, origin='lower', cmap='Greys_r', norm=norm)
    axes[idx].set_title(f"PSF at ({x}, {y})")
```

The visualization creates a 3x3 grid:
- **Top row (5 panels)**: PSF images at the five sampling positions, showing how the PSF shape changes across the field
- **Bottom row (4 panels)**: Difference maps between the center PSF and each corner PSF, highlighting spatial variation

`interpolate_psf_at_position()` reconstructs the PSF at any `(x, y)` position using the polynomial model stored in the PSFEx output file. This function can also be called during fitting to get the exact PSF for any source position.

### Using the variable PSF in GalfitX

The variable PSF output can be used in two ways:

1. **Single position**: Call `interpolate_psf_at_position(psf_file, x, y)` for each source to get a position-specific PSF, then pass these individual PSFs in `psf_list`.

2. **Average PSF**: Use the center position as an approximation for all sources. This is less accurate but simpler.

```python
from create_psf import interpolate_psf_at_position
from astropy.io import fits

# For each source at position (x, y):
psf_at_source = interpolate_psf_at_position(psf_file, x, y, normalize=True)
fits.writeto(f"psf_{source_id}.fits", psf_at_source, overwrite=True)
```

### Output products

| File | Description |
|---|---|
| PSF FITS file | Variable PSF model (data cube at sampled positions) |
| `psf_variation.png` | Visualization of PSFs at 5 positions + difference maps |
| Same as constant PSF demo | LDAC catalog, config, star cutouts |

---

## Choosing the Right Method

| Method | Field size | Stars needed | Accuracy | Complexity |
|---|---|---|---|---|
| Stacked PSF | Small | 10+ | Good | Low |
| PSFEx constant | Small-Medium | 15+ | Very good | Medium |
| PSFEx variable | Large | 30+ | Best | High |

Recommendations:
- For JWST/NIRCam single-detector images: **Stacked PSF** or **PSFEx constant**
- For NIRCam mosaic images: **PSFEx variable** with `psfvar_degrees=1`
- For extremely wide fields (multiple tiles): **PSFEx variable** with `psfvar_degrees=2`

Always check the FWHM of your output PSF against the expected instrument PSF for your filter. The [JWST PSF properties page](https://jwst-docs.stsci.edu/jwst-near-infrared-camera) provides reference values.
