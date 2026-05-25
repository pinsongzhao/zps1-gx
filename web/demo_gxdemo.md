# Demo: Full Pipeline (gxdemo.py)

This walkthrough covers `galfitx/gxdemo.py` end to end. The script performs multi-band structural analysis on JWST/NIRCam images across six bands -- F444W, F356W, F277W, F200W, F150W, and F115W -- running source detection, image cutouts, pure-image fitting, photometric redshift estimation, pure-SED fitting, and joint image+SED fitting.

Each section below mirrors a logical block in the script. You can follow along in the source file at `galfitx/gxdemo.py`.

---

## Prerequisites

Before running this demo you need:

- A Python environment with `galfitx` installed (plus `astropy`, `numpy`, `photutils`, `reproject`, `psfr`, `eazy`, etc.)
- GalfitS installed and its path set (the variable `galfitS_path` in the script)
- GalfitM (the classic GALFITM binary) available either at a hard-coded public path or on `$PATH`
- EAZY installed and its path set (the variable `eazy_path`)
- SPS templates at the location pointed to by `template_dir`
- Six NIRCam science images and their corresponding weight maps in the working directory

---

## Section 1: Import and Path Setup

```python
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
```

The first line disables JAX/XLA GPU memory preallocation. GalfitS uses JAX internally, and without this flag JAX would grab nearly all GPU memory at import time, leaving none for other workers or subsequent operations.

```python
galfitS_path = "/home/zhongyi/Softwares/GalfitS_fork/src/galfits/galfitS.py"
eazy_path    = '/home/zhongyi/Softwares/eazy-photoz'
```

These two paths point to the GalfitS Python entry point and the EAZY photo-z code. Adjust them to match your system.

```python
template_dir = "/home/zhongyi/Softwares/GalfitX_fork/demo_group/templates"
SPS_catalog_path = os.path.join(template_dir, "UNCOVER_DR4_SPS_catalog.fits")
sfhs_path        = os.path.join(template_dir, "sfhs_SPS_DR4.npz")
```

The SPS (Stellar Population Synthesis) catalog and star-formation history file are used later for SFH priors during image+SED fitting. They contain empirical SFH templates derived from the UNCOVER survey.

The script then locates the GALFITM binary. It first checks a known public path; if that does not exist it falls back to `shutil.which('galfitm')`. This binary is used by the photometry pipeline to generate model images.

```python
from galfitx.source_detection import SExtractor_HDR
from galfitx.postage_stamp import create_stamp_file, cut_stamps
from galfitx.create_setup_gs import prepare_galfits, create_mask, process_psf, reproject_segm, gen_pSed_data_lyric
from galfitx.gx_gsutils import Union_Set
from galfitx.model_isoflux import PhotometryConfig, PhotometryPipeline
from galfitx.eazy_utils import zphot_config, translate_config, run_eazy, show_all_fitting
from galfitx.utils import combine_catalogs
```

All key GalfitX modules are imported at once:
- **source_detection**: SExtractor-compatible source extraction with cold+hot HDR mode
- **postage_stamp**: Stamp file generation and image cutouts
- **create_setup_gs**: GALFITM/GalfitS configuration file generation, mask creation, PSF processing
- **gx_gsutils**: Utility functions including the Union-Find grouping algorithm
- **model_isoflux**: Isophotal flux measurement and error estimation
- **eazy_utils**: EAZY photometric redshift wrapper functions
- **utils**: General utilities including final catalog combination

---

## Section 2: Multi-band Input Configuration

This block defines every piece of data the pipeline needs for six JWST/NIRCam bands.

### Science images

```python
sciname_list = np.array([
    'F444W_sci.fits', 'F356W_sci.fits', 'F277W_sci.fits',
    'F200W_sci.fits', 'F150W_sci.fits', 'F115W_sci.fits'
])
```

Ordered from reddest (F444W, 4.4 micron) to bluest (F115W, 1.15 micron). The ordering does not affect the science, but keeping a consistent order avoids confusion.

### Weight maps

```python
whtname_list = np.array([
    'F444W_wht.fits', 'F356W_wht.fits', 'F277W_wht.fits',
    'F200W_wht.fits', 'F150W_wht.fits', 'F115W_wht.fits'
])
```

One inverse-variance weight map per science image. These are used for source detection (S/N weighting) and noise estimation.

### Band labels and output directories

```python
banddir_list = np.array(['./F444W/', './F356W/', './F277W/',
                         './F200W/', './F150W/', './F115W/'])
label_list   = np.array(['F444W', 'F356W', 'F277W',
                         'F200W', 'F150W', 'F115W'])
```

`label_list` names each band for use in output filenames (e.g. `obj1_F444Wsci.fits`). `banddir_list` is a legacy field for per-band data directories.

### Filter names for GalfitS

```python
filter_list = np.array([
    'nircam_f444w', 'nircam_f356w', 'nircam_f277w',
    'nircam_f200w', 'nircam_f150w', 'nircam_f115w'
])
```

These lowercase strings must exactly match keys in the `effective_wave` dictionary defined in `create_setup_gs.py`. GalfitS uses them to look up the effective wavelength of each filter for SED template evaluation. **Do not change these strings** unless you are adding a new filter.

### PSF files

```python
psf_list = np.array([
    './psfs/f444w_psf.fits', './psfs/f356w_psf.fits', './psfs/f277w_psf.fits',
    './psfs/f200w_psf.fits', './psfs/f150w_psf.fits', './psfs/f115w_psf.fits'
])
```

One PSF image per band. These can be stacked PSFs or PSFEx models (see the PSF Construction demo). They are convolved with model images during fitting.

### Photometric calibration

```python
zero_list        = np.array([27.462, 27.462, 27.462, 28.967, 28.967, 28.967])
expt_list        = np.array([2834.5, 3092.18, 3092.18, 2834.5, 3092.18, 6184.38])
pixel_scale_list = np.array([0.04, 0.04, 0.04, 0.02, 0.02, 0.02])
mjsr_list        = np.array([0.4, 0.42, 0.49, 1.96, 2.33, 3.1])
gain_list        = np.array([1.8, 1.8, 1.8, 1.8, 1.8, 1.8])
```

Key points:
- **Zero points**: The JWST/NIRCam long-wavelength channels (F444W, F356W, F277W) share zeropoint 27.462; the short-wavelength channels (F200W, F150W, F115W) share 28.967.
- **Pixel scales**: Long-wavelength channels are 0.04 arcsec/pixel; short-wavelength channels are 0.02 arcsec/pixel. This difference means segmentation maps and stamps must be reprojected when switching between channels.
- **mjsr_list**: The `PHOTMJSR` header keyword values (MJy/sr per DN/s). Used to convert image units to physical flux.
- **gain_list**: Detector gain in e-/DN, uniform at 1.8 for all NIRCam detectors.

### Derived arrays

```python
covermask_list = np.array([s.replace("sci.fits", "cover_mask.fits") for s in sciname_list])
seglist        = np.array([s.replace(".fits", "_seg.fits") for s in sciname_list])
nband = len(sciname_list)
```

`covermask_list` and `seglist` are derived by string substitution -- coverage masks and segmentation maps are named to match the science images. `nband` is computed rather than hard-coded so the script adapts if you add or remove bands.

---

## Section 3: Detection Setup

```python
detname          = sciname_list[0]   # F444W as detection band
det_label        = label_list[0]
weight_name      = whtname_list[0]
kernel           = np.loadtxt('gauss_4.0_7x7.conv', skiprows=1)
SEx_dir          = "./SEx/"
ref_pixel_scale  = pixel_scale_list[0]   # 0.04 arcsec/pixel
det_segname      = 'outseg.fits'
stampfile        = './stamps.txt'
nnw_sex          = 'default.nnw'
fwhm_arcsec      = 0.16
```

Why F444W as the detection band? It is the reddest (deepest in terms of rest-frame optical for high-z galaxies), has the largest pixels (0.04"/px, so source footprints cover fewer pixels and detection is more robust), and provides a good balance of sensitivity and resolution for identifying sources that will then be measured in all six bands.

- **kernel**: A 7x7 Gaussian convolution kernel (sigma=4 pixels) loaded from a SExtractor-format `.conv` file. This smooths the image before thresholding.
- **fwhm_arcsec**: Expected FWHM of point sources (0.16" is typical for JWST/NIRCam at 4.4 micron). Used by SExtractor for deblending and CLASS_STAR computation.
- **SEx_dir**: All SExtractor intermediate products go here.
- **ref_pixel_scale**: All stamp sizes are defined in the detection band's pixel scale.

---

## Section 4: Source Detection (Cold+Hot HDR)

```python
outtab, outsegm = SExtractor_HDR(
    detname,
    path            = SEx_dir,
    kernel          = (kernel, kernel),
    detect_minarea  = (25, 15),
    detect_thresh   = (3, 1.8),
    deblend         = (True, True),
    deblend_nthresh = (32, 64),
    deblend_mincont = (0.01, 0.001),
    clean           = (True, True),
    back_type       = (True, True),
    back_value      = (0., 0.),
    back_size       = (128, 32),
    back_filtersize = (3, 3),
    weight_type     = 'MAP_WEIGHT',
    weight_name     = weight_name,
    scale_factor    = 0.8,
    pixel_scale     = ref_pixel_scale,
    nnw_sex         = nnw_sex,
    fwhm_arcsec     = fwhm_arcsec,
    mag_zeropoint   = zero_list[0],
    verbose         = True
)
```

`SExtractor_HDR` implements a dual-threshold "High Dynamic Range" detection strategy:

**Cold pass** (first tuple element):
- `detect_thresh=3.0` and `detect_minarea=25` -- a conservative 3-sigma threshold with a minimum of 25 connected pixels. This catches bright, reliable detections.
- `back_size=128` -- estimates the background on 128-pixel meshes, appropriate for capturing large-scale background variations without being biased by bright sources.

**Hot pass** (second tuple element):
- `detect_thresh=1.8` and `detect_minarea=15` -- a more sensitive threshold that detects fainter sources at the cost of more spurious detections.
- `back_size=32` -- smaller background meshes capture local variations better, improving detection in crowded regions.

The two catalogs and segmentation maps are then merged: the cold catalog provides reliable bright sources, and the hot catalog adds fainter sources that the cold pass missed. Overlapping detections are resolved during merging.

After detection, a coverage mask is generated:

```python
det_img = fits.getdata(detname, 0)
coverage_mask = np.isnan(det_img)
```

NaN pixels in the science image are flagged as masked. This mask is written to a FITS file and later reprojected for each band.

Finally, the combined catalog is read:

```python
catalog_name = os.path.join(SEx_dir, 'outcat')
outtab = ascii.read(catalog_name)
```

---

## Section 5: Stamp File and Image Cutouts

### Stamp file

```python
create_stamp_file(detname, catalog_name, sizefac=2.5, outfile=stampfile, pixel_scale=ref_pixel_scale)
```

The stamp file (`stamps.txt`) records, for each detected source, the bounding-box coordinates of a rectangular region on the sky. The region is centered on the source and its size is `sizefac * kron_radius`. With `sizefac=2.5`, each stamp extends to 2.5 times the Kron radius, providing enough sky around each source for sky estimation and fitting.

### Cutting stamps for all bands

```python
for i in range(nband):
    sciname = sciname_list[i]
    label   = label_list[i]
    cut_stamps(sciname, cutout_dir, label=label, stampfile=stampfile, ps=pixel_scale_list[i])
```

`cut_stamps()` reads the stamp file and extracts image cutouts from each full-frame science image. The `ps` parameter tells the function the pixel scale of the image being cut, so it can correctly convert the stamp coordinates (defined in the detection band's pixel scale) to the target band's pixel grid.

For the long-wavelength bands (0.04"/px), the cutout has the same pixel dimensions as the stamp. For short-wavelength bands (0.02"/px), each cutout has twice as many pixels on a side, covering the same sky area at higher resolution.

### Reprojecting segmentation maps

```python
det_seg_fullpath = os.path.join(SEx_dir, det_segname)

for sciname, seg_name, cover_mask_name in zip(sciname_list, seglist, covermask_list):
    reproject_segm(det_seg_fullpath, sciname, output=seg_name)
    reproject_segm(coverage_mask_output, sciname, output=cover_mask_name, type="mask")
```

Because the detection segmentation map is at 0.04"/px but short-wavelength images are at 0.02"/px, `reproject_segm()` uses nearest-neighbor interpolation (via `reproject_interp`) to resample the segmentation map onto each band's WCS. This preserves the integer label values that identify each source. The same is done for coverage masks.

---

## Section 6: Directory Structure and Object Grouping

### Directory layout

```python
gsdir            = './galfits/'
gs_pureImage_dir = os.path.join(gsdir, "pureImage")
gs_pureSed_dir   = os.path.join(gsdir, "pureSed")
gs_image_sed_dir = os.path.join(gsdir, "image_sed")
```

Three subdirectories under `./galfits/` store results from the three fitting stages:
- **pureImage**: Multi-band image fitting with free SED parameters (no physical SED constraints)
- **pureSed**: SED-only fitting using measured fluxes (no spatial information)
- **image_sed**: Joint image+SED fitting using results from the previous two stages as priors

### Masks and objects.json

```python
object_dict = {}
object_path = "./objects.json"

for i in range(nband):
    hdusci     = fits.open(sciname_list[i])[0]
    cover_data = fits.getdata(covermask_list[i])
    segdata    = fits.getdata(seglist[i])

    for idx in sorted_idx:
        hduin = hdusci.copy()
        maskfile = os.path.join(cutout_dir, f"obj{idx+1}_{label}mask.fits")
        corner, objects = create_mask(
            scihdu=hduin, seg_data=segdata, cover_data=cover_data,
            catalog_name=catalog_name, paramfile=stampfile,
            mask_file=maskfile, current=idx, ps=pixel_scale_list[i]
        )
        obj_stored = [int(obj) for obj in objects]
        object_dict[str(idx)] = obj_stored
```

For every source in every band, `create_mask()` produces a binary mask FITS file. The mask marks:
- Pixels belonging to other (non-target) sources as masked (value 1)
- Pixels outside the image coverage as masked
- Pixels with zero weight as masked
- Pixels belonging to the target source or its co-fitting neighbors as unmasked (value 0)

The function also returns `objects`, a list of source indices that overlap with the current source. These are the sources that must be fit simultaneously. This information is stored in `objects.json` for later use.

### Union-Find grouping

```python
with open(object_path, "w") as f:
    json.dump(object_dict, f, indent=4)

with open(object_path, "r") as f:
    object_dict = json.load(f)

groups = Union_Set(object_dict)
```

`Union_Set()` implements a Union-Find (disjoint-set) algorithm. If object A overlaps with object B, and object B overlaps with object C, then A, B, and C are placed in the same group. The result is a list of groups, where each group contains source indices that must be fit together. This ensures that overlapping sources are never fit in isolation, which would bias their parameters.

---

## Section 7: Photometry Configuration

```python
iso_config = PhotometryConfig(
    image_list=sciname_list, galaxy_catalog=catalog_name,
    psf_file=psf_file, segmentation_map_list=seglist,
    cutout_dir=cutout_dir, label_list=label_list,
    filter_labels=filter_list, gains=gain_list,
    exptimes=expt_list, mjsr_list=mjsr_list, zero_list=zero_list,
    apertures_list=apers_list, pixel_scales=pixel_scale_list,
    ref_pixel_scale=ref_pixel_scale, galfit_path=galfit_path,
    stampfile=stampfile, try_filterlist=try_filterlist,
    sky_noise_path=skynoisepath, gmoutdir=gmoutdir,
    save_mask=True, save_regions=True, plot_histograms=True,
    kron_scale=kron_scale, detwht_file=weight_name,
    wht_file_list=whtname_list,
    external_mask_file_list=external_mask_file_list,
)

iso_pipeline = PhotometryPipeline(iso_config)
iso_pipeline.step1_determine_background_positions()
bg_params = iso_pipeline.step2_fit_background_noise()
```

`PhotometryConfig` bundles all the information needed for isophotal flux measurement. Key fields:

- **apers_list**: Aperture radii (in arcsec) for sky noise measurement: `np.array([5,10,15,20,25,30,35,40]) * 0.04`. Noise is measured at multiple apertures to characterize how background scatter scales with aperture size.
- **try_filterlist=[0]**: Only measure noise in the detection band (index 0, F444W).
- **kron_scale=1.5**: Enlarge the Kron ellipse by 50% to define the isophotal aperture.
- **external_mask_file_list**: Optional additional masks (e.g., a global bad-pixel mask for the detection band).

The pipeline has four steps; the first two are run here:
1. **step1_determine_background_positions()**: Identifies sky positions free of sources for background measurement.
2. **step2_fit_background_noise()**: Fits a noise model (sigma vs. aperture size) at each sky position. The resulting `bg_params` DataFrame will be used later for error propagation.

---

## Section 8: Pure Image Fitting Loop

```python
for i_group, group in enumerate(groups):

    sorted_idx_group     = sorted(group, key=lambda x: outtab["mag_auto"][x])
    sorted_idx_group_1idx = [value + 1 for value in sorted_idx_group]
```

For each group of overlapping sources, the members are sorted by brightness (brightest first). Fitting brightest sources first ensures their model parameters are available as initial guesses when fitting fainter overlapping sources.

```python
    for idx in sorted_idx_group:
        objects = object_dict[str(idx)]

        image_stamp_list = []
        mask_file_list   = []
        for i in range(nband):
            label = label_list[i]
            maskfile  = os.path.join(cutout_dir, f"obj{idx+1}_{label}mask.fits")
            image_file = os.path.join(cutout_dir, f"obj{idx+1}_{label}sci.fits")
            image_stamp_list.append(image_file)
            mask_file_list.append(maskfile)

        psf_file_used = process_psf(idx+1, nband, psf_list, image_stamp_list,
                                     label_list, processed_psf_dir)
```

For each source in the group:
1. Gather the stamp images and masks for all bands.
2. `process_psf()` crops each PSF to fit within the stamp dimensions. If a PSF is larger than the stamp in either dimension, a centered odd-sized crop is extracted.

```python
        prepare_galfits(
            lyric_path=lyric_path, prior_path=prior_path,
            cat_file=catalog_name, objects=objects,
            det_label=det_label, sci_list=image_stamp_list,
            psf_list=psf_file_used, zero_list=zero_list,
            pixscl_list=pixel_scale_list, label_list=label_list,
            filter_list=filter_list,
            geo_smdir=gs_pureImage_dir, pSed_smdir=gs_pureSed_dir,
            imgSed_smdir=gs_image_sed_dir,
            mask_list=mask_file_list, use_sed=0, convf=False
        )
```

`prepare_galfits()` generates a `.lyric` configuration file for GalfitS. With `use_sed=0`, the fitting is in "pure image" mode -- structural parameters (position, size, Sersic index, axis ratio, PA) are free, and SED parameters (stellar mass, metallicity, dust, SFH bins) are also free but unconstrained by any physical prior.

```python
        command = (f"CUDA_VISIBLE_DEVICES=7 python {galfitS_path} --config "
                   f"{lyric_path} --workplace {gs_pureImage_dir} "
                   f"--fit_method ES --num_generations 10000")
        os.system(command)
```

GalfitS is invoked with:
- `--fit_method ES`: Evolution Strategy (a gradient-free optimizer well-suited for multi-parameter Sersic fitting)
- `--num_generations 10000`: Maximum number of optimization iterations

---

## Section 9: Isophotal Flux and Errors

```python
    gs_flux_outfile = os.path.join(gmoutdir, f"gs_iso_flux_group{i_group}.cat")
    gs_flux_err_outfile = os.path.join(gmoutdir, f"gs_iso_flux_err_group{i_group}.cat")

    iso_pipeline._compute_model_fluxes_for_galaxies(
        galids=sorted_idx_group_1idx, filter_list=filter_list,
        image_list=sciname_list, gsdir=gs_pureImage_dir,
        outdir=gmoutdir, output_file=gs_flux_outfile
    )
    iso_pipeline.step4_compute_isophotal_errors(
        bg_df=bg_params, flux_df_path=gs_flux_outfile,
        gs_flux_err_outfile=gs_flux_err_outfile
    )
```

After fitting, the GalfitS model images are used to measure isophotal fluxes:
1. **`_compute_model_fluxes_for_galaxies()`**: For each source, it reads the GalfitS model, segments it using the SExtractor segmentation map, and measures the flux within the isophotal aperture for every band. Results are written to an ASCII catalog.
2. **`step4_compute_isophotal_errors()`**: Uses the background noise model (from step 2) to propagate uncertainties. The output catalog has columns like `nircam_f444w_flux`, `nircam_f444w_fluxerr` for each band.

---

## Section 10: EAZY Photo-z

```python
    temperr = 0.03
    syserr  = 0.03
    eazy_out_path = f"./eazy/{i_group}"
    config_file_param = os.path.join(eazy_out_path, "zphot.param")
    os.makedirs(eazy_out_path, exist_ok=True)

    zphot_config(gs_flux_err_outfile, eazy_out_path, temperr, syserr,
                 eazypath=eazy_path, configfile=config_file_param)
    config_file_translate = os.path.join(eazy_out_path, "zphot.translate")
    translate_config(gs_flux_err_outfile, configfile=config_file_translate)
    run_eazy(eazypath=eazy_path, configfile=config_file_param,
             translatefile=config_file_translate)

    figure_output = os.path.join(eazy_out_path, "figures")
    os.makedirs(figure_output, exist_ok=True)
    # show_all_fitting(eazy_out_path, gs_pureImage_dir, output_path=figure_output)
```

Photometric redshifts are estimated using EAZY:
- **`zphot_config()`**: Generates the EAZY parameter file. `temperr=0.03` adds 3% template error (accounting for template imperfections) and `syserr=0.03` adds 3% systematic error to the flux uncertainties. These prevent EAZY from over-fitting noisy data.
- **`translate_config()`**: Creates a translation file mapping catalog column names (e.g. `nircam_f444w_flux`) to EAZY filter IDs.
- **`run_eazy()`**: Executes EAZY, producing `photz.zout` with columns `id`, `z_peak`, `z_spec`, etc.
- **`show_all_fitting()`** (commented out here): Would generate diagnostic plots including photo-z vs. spec-z comparison and best-fit SED figures.

---

## Section 11: Pure SED Fitting

```python
    ebv = None   # will automatically search using package

    z_out_path = os.path.join(eazy_out_path, "photz.zout")

    gen_pSed_data_lyric(
        cat_path=gs_flux_err_outfile, z_cat_path=z_out_path,
        cutout_dir=cutout_dir, mock_dir=gs_pureSed_dir,
        label_list=label_list, filter_list=filter_list,
        zp_list=zero_list, ebv=ebv
    )
```

`gen_pSed_data_lyric()` converts measured fluxes into mock 1x1 pixel "images" -- one per band per source. Each mock image contains a single pixel with value equal to the luminosity (converted from flux using the photo-z and cosmology). A `.lyric` file for pure SED fitting is also written.

```python
    for idx in sorted_idx_group:
        mock_subdir = os.path.join(gs_pureSed_dir, f"{idx+1}")
        lyric_file  = os.path.join(mock_subdir, f"obj{idx+1}_pureSed.lyric")
        workplace   = os.path.join(mock_subdir, "results")

        command = (f"CUDA_VISIBLE_DEVICES=7 python {galfitS_path} --config "
                   f"{lyric_file} --workplace {workplace} "
                   f"--fit_method ES --num_generations 10000")
        os.system(command)
```

GalfitS is run on the mock images. Because the images are 1x1 pixels, the structural parameters (position, size, Sersic index, axis ratio) are meaningless -- only the SED parameters (stellar mass, metallicity, dust extinction, SFH bins) are constrained. This provides a pure SED fit that is later used to initialize the joint image+SED fit.

---

## Section 12: Image+SED Fitting

```python
    z_cat = Table.read(z_out_path, format="ascii")

    for idx in sorted_idx_group:

        objects = object_dict[str(idx)]

        z_list = []
        for obj in objects:
            index = obj + 1
            z_idx = np.where(z_cat["id"] == index)[0][0]
            z_curr = z_cat[z_idx]["z_peak"]
            if z_curr < 0:
                z_curr = 0.001
            z_list.append(z_curr)
```

Photo-z results are read. For each co-fitting object, `z_peak` is extracted. Negative redshifts (which indicate failed fits) are replaced with 0.001 as a floor.

```python
        psf_file_used = process_psf(idx+1, nband, psf_list,
                                     image_stamp_list, label_list,
                                     processed_psf_dir)

        prepare_galfits(
            lyric_path=lyric_path, prior_path=prior_path,
            cat_file=catalog_name, objects=objects,
            det_label=det_label, sci_list=image_stamp_list,
            psf_list=psf_file_used, zero_list=zero_list,
            pixscl_list=pixel_scale_list, label_list=label_list,
            filter_list=filter_list,
            geo_smdir=gs_pureImage_dir, pSed_smdir=gs_pureSed_dir,
            imgSed_smdir=gs_image_sed_dir,
            SPS_catalog_path=SPS_catalog_path, sfhs_path=sfhs_path,
            mask_list=mask_file_list,
            z_list=z_list, use_sed=1, use_sfh_prior=True,
            convf=False, ebv=ebv
        )

        command = (f"CUDA_VISIBLE_DEVICES=7 python {galfitS_path} --config "
                   f"{lyric_path} --workplace {gs_image_sed_dir} "
                   f"--fit_method ES --num_generations 10000 --prior {prior_path}")
        os.system(command)
```

This is the final and most constrained fitting stage:
- `use_sed=1`: SED information is included in the likelihood.
- `use_sfh_prior=True`: SFH priors derived from the empirical SPS catalog are applied. These priors constrain the star-formation history bins to physically plausible ranges based on galaxies at similar redshifts.
- `z_list`: The photo-z is used as a fixed redshift prior (narrow prior around `z_peak`).
- `--prior {prior_path}`: The GalfitS command reads the SFH prior file, which specifies Gaussian priors on SFH bin fractions.

Structural parameters are initialized from the pure-image fit results, and SED parameters from the pure-SED fit. This gives the optimizer an excellent starting point.

---

## Section 13: Combine Results

```python
combine_catalogs(filter_list, catalog_name, gs_pureImage_dir,
                 './eazy', gs_image_sed_dir,
                 output_file='combined_catalog.fits')

print(datetime.datetime.now() - begin_time)
```

`combine_catalogs()` merges results from all stages into a single FITS catalog:
1. Source detection catalog (positions, magnitudes, shapes)
2. Pure-image fitting results (structural parameters for each band)
3. EAZY photometric redshifts (`z_peak`, `z_spec`)
4. Image+SED fitting results (stellar mass, SFR, metallicity, dust, SFH bins)

Missing values are filled with -999.0. The output `combined_catalog.fits` is the final science product.

The total execution time is printed at the end.

---

## Output Summary

After running the full pipeline, you will have:

| Directory | Contents |
|---|---|
| `./SEx/` | SExtractor catalogs and segmentation maps |
| `./cutout/` | Image cutouts and masks for each source and band |
| `./galfits/pureImage/` | GalfitS config files, results, and model images (pure image mode) |
| `./galfits/pureSed/` | Mock images and GalfitS results (pure SED mode) |
| `./galfits/image_sed/` | GalfitS config files, results, and model images (joint mode) |
| `./eazy/` | EAZY configuration, outputs, and photo-z results per group |
| `./galfitm/` | Isophotal flux catalogs and error estimates |
| `combined_catalog.fits` | Final merged catalog with all results |
