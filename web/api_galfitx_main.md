# Main Pipeline (`galfitx/galfitx_main.py`)

The top-level entry point for the GalfitX multi-band galaxy structural measurement pipeline. Runs the full workflow from source detection through model fitting.

---

## `core_galfitx`

Execute the complete GalfitX pipeline for multi-band imaging measurement. The pipeline performs the following steps in order:

1. **Load configuration** from a TOML config file.
2. **Source detection** using SExtractor HDR (dual-mode cold+hot detection).
3. **Create stamp file** defining postage stamp cutout boundaries for each source.
4. **Cut postage stamps** for all bands at the positions defined in the stamp file.
5. **Create masks** for each source, identifying primary, secondary, and tertiary objects.
6. **Run GalfitS** on each source (sorted by brightness), fitting a Sersic model with co-fitting of overlapping neighbors.

```python
core_galfitx(
    image_name,
    weight_name,
    kernel_name,
    image_list,
    outdir_list,
    filter_list,
    label_list,
    psf_list,
    zero_list,
    config="./csst.toml",
)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `image_name` | `str` | (required) | Path to the detection image (FITS). Used as the reference for source detection and stamp creation. |
| `weight_name` | `str` | (required) | Path to the weight image (FITS). Used for bad pixel identification and masking. |
| `kernel_name` | `str` | (required) | Path to the convolution kernel file (ASCII, two columns). Loaded with `np.loadtxt` and used for SExtractor filtering. |
| `image_list` | `list[str]` | (required) | Paths to multi-band science images (FITS). One path per photometric band. |
| `outdir_list` | `list[str]` | (required) | Output directories for postage stamp cutouts. One directory per band. Existing directories are replaced. |
| `filter_list` | `list[str]` | (required) | Filter identifiers used by GalfitS (e.g., `"nircam_f444w"`). One per band. |
| `label_list` | `list[str]` | (required) | Band labels used for file naming and internal bookkeeping. One per band. |
| `psf_list` | `list[str]` | (required) | Paths to PSF images (FITS). One per band. |
| `zero_list` | `list[float]` | (required) | Magnitude zeropoints for each band. |
| `config` | `str` | `"./csst.toml"` | Path to the TOML configuration file containing pipeline parameters. |

**Returns**

| Type | Description |
|---|---|
| `int` | Returns `1` on successful completion. |

**Configuration File (TOML)**

The `config` file must contain the following keys (example values shown):

```toml
detect_minarea_cold = 5
detect_minarea_hot = 5
detect_thresh_cold = 1.5
detect_thresh_hot = 3.0
deblend = true
deblend_nthresh_cold = 32
deblend_nthresh_hot = 32
deblend_mincont_cold = 0.001
deblend_mincont_hot = 0.001
clean = true
back_type = "AUTO"
back_value = 0.0
back_size_cold = 64
back_size_hot = 32
back_filtersize = 3
BACKGROUND = "BACKGROUND"
scale_factor = 1.0
pixel_scale = 0.03
mag_zeropoint = 28.0
sizefac = 3.0
scale = 1.1
offset = 4.0
limgal = 3.0
b = 0.03
setup = "setup_file.txt"
```

**Output Structure**

The pipeline creates the following directory layout:

```
./sex/                    # SExtractor outputs (outcat, outseg.fits)
./galfits/                # GalfitS working directory
    mask<ID>.fits         # Mask for each source
    obj<ID>               # GalfitS lyric config for each source
    output<ID>.fits       # GalfitS output for each source
<outdir_list[i]>/         # Postage stamp cutouts per band
    <ID>.fits             # Individual stamp cutouts
```

**Processing Order**

Sources are processed in order of ascending `mag_auto` (brightest first). For each source, the pipeline:
1. Creates a mask identifying the primary source and overlapping neighbors.
2. Prepares PSF files (cropping if necessary).
3. Generates a GalfitS lyric configuration file with initial guesses from previous fits or the SExtractor catalog.
4. Runs GalfitS with 8000 optimization steps.

**Notes**

- The weight image is used both for SExtractor detection and for identifying bad pixels (pixels with weight = 0 are masked).
- The segmentation map from SExtractor is used to identify source boundaries and create masks.
- Overlapping sources (secondary objects) are co-fitted simultaneously with the primary source.
- The GalfitS executable path is currently hard-coded as `/home/machao/opt/GalfitS-main/src/galfits/galfitS.py`.
