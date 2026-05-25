"""
Demo script to run PSFEx for constant PSF modeling.

This script demonstrates the complete workflow:
1. Build LDAC catalog from SExtractor output
2. Generate PSFEx configuration file (PSFVAR_DEGREES=0)
3. Run PSFEx to create constant PSF model

Before running this script, make sure you have:
- Science FITS image (e.g., sci_i.fits)
- SExtractor catalog (e.g., outcat)
- SExtractor segmentation map (e.g., outseg.fits)
- A template LDAC file (e.g., output_assoc_temp.cat)
"""

import os
import sys
from create_psf import build_input_ldac, psfex_config, run_psfex

# ========================================================================
# User-defined parameters - modify these for your data
# ========================================================================
sci_image = "sci_i.fits"           # Path to science image
catalog_file = "outcat"            # SExtractor catalog (ASCII)
seg_file = "outseg.fits"           # SExtractor segmentation map
template_file = "output_assoc_temp.cat"  # Template LDAC file

output_dir = "./psfex_const_output"  # Output directory
output_ldac = "psfex.cat"           # Output LDAC catalog name

# Star selection criteria
mag_bright_limit = 19.0            # Bright magnitude limit
mag_faint_limit = 21.0             # Faint magnitude limit

# Cutout parameters
cutout_size = 71                   # Size of star cutouts (pixels)

# PSFEx configuration parameters
psf_sampling = 0.5                 # Sampling step (0.0 = auto)
psf_size = (141, 141)              # Output PSF size
sample_minsn = 10                  # Minimum S/N for star selection

# Image metadata (adjust for your data)
fwhm_arcsec = 0.13                 # Approximate FWHM in arcsec
mag_zeropoint = 28.0               # Magnitude zero-point
ref_pixel_scale = 0.074            # Pixel scale (arcsec/pixel)

# ========================================================================
# Step 1: Build LDAC catalog for PSFEx
# ========================================================================
print("=" * 60)
print("Step 1: Building LDAC catalog for PSFEx")
print("=" * 60)

build_input_ldac(
    sci_image=sci_image,
    catalog_file=catalog_file,
    seg_file=seg_file,
    output_ldac=output_ldac,
    mag_bright_limit=mag_bright_limit,
    mag_faint_limit=mag_faint_limit,
    cutout_size=cutout_size,
    save_cutouts=True,
    cutouts_dir=os.path.join(output_dir, "star_cutouts"),
    template_file=template_file,
    fwhm_arcsec=fwhm_arcsec,
    mag_zeropoint=mag_zeropoint,
    ref_pixel_scale=ref_pixel_scale,
)

# ========================================================================
# Step 2: Generate PSFEx configuration file
# ========================================================================
print("\n" + "=" * 60)
print("Step 2: Generating PSFEx configuration file (constant PSF)")
print("=" * 60)

config_file = os.path.join(output_dir, "config.psfex")
psfex_config(
    output_config=config_file,
    psf_sampling=psf_sampling,
    psf_size=psf_size,
    sample_autoselect=True,
    sample_fwhmrange="1, 5",
    sample_minsn=sample_minsn,
    psfvar_degrees=0,  # Constant PSF
)

# ========================================================================
# Step 3: Run PSFEx
# ========================================================================
print("\n" + "=" * 60)
print("Step 3: Running PSFEx")
print("=" * 60)

psf_file = run_psfex(
    catalog=output_ldac,
    config=config_file,
    output_dir=output_dir,
)

print("\n" + "=" * 60)
print("PSFEx workflow completed successfully!")
print(f"PSF model saved to: {psf_file}")
print(f"Output directory: {output_dir}")
print("=" * 60)