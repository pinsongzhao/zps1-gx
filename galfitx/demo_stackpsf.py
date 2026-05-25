"""
Demo script to build a stacked PSF using PSFR.

This script demonstrates how to:
1. Select stars from SExtractor catalog
2. Extract star cutouts from science image
3. Build a stacked PSF using the PSFR package

Before running this script, make sure you have:
- Science FITS image (e.g., sci_i.fits)
- SExtractor catalog (e.g., outcat)
- SExtractor segmentation map (e.g., outseg.fits)
"""

import os
import sys
from create_psf import build_stacked_psf

# ========================================================================
# User-defined parameters - modify these for your data
# ========================================================================
sci_image = "sci_i.fits"           # Path to science image
catalog_file = "outcat"            # SExtractor catalog (ASCII)
seg_file = "outseg.fits"           # SExtractor segmentation map

output_dir = "./stacked_psf_output"  # Output directory

# Star selection criteria
mag_bright_limit = 19.0            # Bright magnitude limit
mag_faint_limit = 21.0             # Faint magnitude limit

# Cutout parameters
cutout_size = 71                   # Size of star cutouts (pixels)

# PSF stacking parameters
oversampling = 3                   # Oversampling factor for PSF stacking
n_recenter = 10                    # Number of recentering iterations
num_iteration = 20                 # Number of iterations for stacking algorithm

# Output options
save_fits = True                   # Save individual star cutouts and final PSF
save_figures = True                # Save overview and radial profile plots
plot_overview = True               # Generate overview plot of all star cutouts
plot_radial_profile = True         # Generate 1D radial profile plot

# ========================================================================
# Build stacked PSF
# ========================================================================
print("=" * 60)
print("Building stacked PSF from selected stars")
print("=" * 60)

build_stacked_psf(
    sci_image=sci_image,
    catalog_file=catalog_file,
    seg_file=seg_file,
    output_dir=output_dir,
    mag_bright_limit=mag_bright_limit,
    mag_faint_limit=mag_faint_limit,
    cutout_size=cutout_size,
    oversampling=oversampling,
    n_recenter=n_recenter,
    num_iteration=num_iteration,
    save_fits=save_fits,
    save_figures=save_figures,
    plot_overview=plot_overview,
    plot_radial_profile=plot_radial_profile,
)

print("\n" + "=" * 60)
print(f"Stacked PSF completed!")
print(f"Output saved to: {output_dir}/")
print("=" * 60)
