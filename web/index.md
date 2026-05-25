# GalfitX Documentation

Multi-band galaxy morphology and photometry pipeline built on top of [GalfitS](https://github.com/bsshao/GalfitS), designed for JWST/NIRCam imaging. GalfitX automates source detection, postage stamp cutting, mask generation, PSF construction, multi-band image fitting, isophotal photometry, photometric redshift estimation (EAZy), pure SED fitting, and joint Image+SED fitting.

The pipeline follows a five-stage workflow:

1. **Source Detection** — Cold+Hot dual-threshold SExtractor-style detection
2. **Pure Image Fitting** — Multi-band structural fitting with GalfitS (no SED)
3. **Isophotal Photometry** — Model-based flux and error measurement
4. **Photometric Redshift** — EAZy photo-z estimation from multi-band photometry
5. **Pure SED + Image+SED Fitting** — SED-constrained and joint fitting with SFH priors

## Installation

```bash
pip install -e .
```

Dependencies:

- `numpy`, `astropy`, `photutils`, `scipy`, `matplotlib`, `pandas`
- `reproject`, `psfr`, `numba`, `tqdm`
- GalfitS (external): set `galfitS_path` in demo scripts
- EAZy (optional, for photo-z): set `eazy_path` in demo scripts

## Quick Start

```python
from galfitx import core_galfitx

# Run the full pipeline
core_galfitx(
    image_name="F444W_sci.fits",
    weight_name="F444W_wht.fits",
    kernel_name="gauss_4.0_7x7.conv",
    image_list=["F444W_sci.fits", "F356W_sci.fits", "F277W_sci.fits",
                "F200W_sci.fits", "F150W_sci.fits", "F115W_sci.fits"],
    outdir_list=["./F444W/", "./F356W/", "./F277W/",
                 "./F200W/", "./F150W/", "./F115W/"],
    filter_list=["nircam_f444w", "nircam_f356w", "nircam_f277w",
                 "nircam_f200w", "nircam_f150w", "nircam_f115w"],
    label_list=["F444W", "F356W", "F277W", "F200W", "F150W", "F115W"],
    psf_list=["./psfs/f444w_psf.fits", "./psfs/f356w_psf.fits",
              "./psfs/f277w_psf.fits", "./psfs/f200w_psf.fits",
              "./psfs/f150w_psf.fits", "./psfs/f115w_psf.fits"],
    zero_list=[27.462, 27.462, 27.462, 28.967, 28.967, 28.967],
    config="./csst.toml",
)
```

For the full multi-stage pipeline (including photo-z and SED fitting), see the demo scripts.

## Next Steps

```{toctree}
:maxdepth: 1
:caption: Demos

demo_gxdemo
demo_gxdemo_parallel
demo_psf
```

```{toctree}
:maxdepth: 1
:caption: API Reference

api_galfitx_main
api_source_detection
api_postage_stamp
api_mask
api_sky_calculation
api_create_setup_gs
api_create_psf
api_utils
api_gx_gsutils
api_model_isoflux
api_eazy_utils
api_read_sersic_results
api_read_setup
api_csst_pipe
```
