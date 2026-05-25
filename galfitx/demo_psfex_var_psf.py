"""
Demo script to run PSFEx for position-dependent PSF modeling.

This script demonstrates the complete workflow:
1. Build LDAC catalog from SExtractor output
2. Generate PSFEx configuration file (PSFVAR_DEGREES=1)
3. Run PSFEx to create position-dependent PSF model
4. Sample PSFs at multiple positions

Before running this script, make sure you have:
- Science FITS image (e.g., sci_i.fits)
- SExtractor catalog (e.g., outcat)
- SExtractor segmentation map (e.g., outseg.fits)
- A template LDAC file (e.g., output_assoc_temp.cat)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import simple_norm
from psfr.util import oversampled2regular
from create_psf import build_input_ldac, psfex_config, run_psfex, interpolate_psf_at_position

# ========================================================================
# User-defined parameters - modify these for your data
# ========================================================================
sci_image = "sci_i.fits"           # Path to science image
catalog_file = "outcat"            # SExtractor catalog (ASCII)
seg_file = "outseg.fits"           # SExtractor segmentation map
template_file = "output_assoc_temp.cat"  # Template LDAC file

output_dir = "./psfex_var_output"   # Output directory
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
psfvar_degrees = 1                 # Polynomial degree (0=constant, 1=linear, 2=quadratic)

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
print(f"Step 2: Generating PSFEx configuration file (PSFVAR_DEGREES={psfvar_degrees})")
print("=" * 60)

config_file = os.path.join(output_dir, "config.psfex")
psfex_config(
    output_config=config_file,
    psf_sampling=psf_sampling,
    psf_size=psf_size,
    sample_autoselect=True,
    sample_fwhmrange="1, 5",
    sample_minsn=sample_minsn,
    psfvar_degrees=psfvar_degrees,  # Position-dependent PSF
)

# ========================================================================
# Step 3: Define PSF sampling positions
# ========================================================================
print("\n" + "=" * 60)
print("Step 3: Defining PSF sampling positions")
print("=" * 60)

# Sample positions within 0-2000 range
sample_positions = [
    (1000, 1000),    # Center (displayed first)
    (100, 100),      # Bottom-left corner
    (1900, 100),     # Bottom-right corner
    (100, 1900),     # Top-left corner
    (1900, 1900),    # Top-right corner
]

print(f"Sampling PSF at {len(sample_positions)} positions:")
for i, (x, y) in enumerate(sample_positions, 1):
    print(f"  {i}. ({x}, {y})")

# ========================================================================
# Step 4: Run PSFEx
# ========================================================================
print("\n" + "=" * 60)
print("Step 4: Running PSFEx")
print("=" * 60)

psf_file = run_psfex(
    catalog=output_ldac,
    config=config_file,
    output_dir=output_dir,
    sample_positions=sample_positions,
)

# ========================================================================
# Step 5: Visualize PSF variation
# ========================================================================
print("\n" + "=" * 60)
print("Step 5: Visualizing PSF variation")
print("=" * 60)

# Get oversampling factor from PSF file
with fits.open(psf_file) as hdul:
    actual_sampling = hdul[0].header.get('PSF_SAMP', psf_sampling)
    oversampling = int(1.0 / actual_sampling)

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.ravel()
axes = axes.ravel()

# Plot PSFs at sample positions (rebin + asinh)
for idx, (x, y) in enumerate(sample_positions):
    if idx >= len(axes):
        break

    psf = interpolate_psf_at_position(psf_file, x, y, normalize=False)

    # Rebin to regular sampling
    if oversampling > 1:
        psf = oversampled2regular(psf, oversampling)

    # Normalize for display
    psf = psf / np.sum(psf)

    # Use asinh stretch for display
    norm = simple_norm(psf, 'asinh')
    axes[idx].imshow(psf, origin='lower', cmap='Greys_r', norm=norm)
    axes[idx].set_title(f"PSF at ({x}, {y})")
    axes[idx].axis('off')

# Plot PSF difference (center vs corners, rebin + asinh)
center_psf = interpolate_psf_at_position(psf_file, 1000, 1000, normalize=False)
if oversampling > 1:
    center_psf = oversampled2regular(center_psf, oversampling)
center_psf = center_psf / np.sum(center_psf)

for i, (x, y) in enumerate(sample_positions[1:]):  # Skip center (idx=0)
    corner_psf = interpolate_psf_at_position(psf_file, x, y, normalize=False)
    if oversampling > 1:
        corner_psf = oversampled2regular(corner_psf, oversampling)
    corner_psf = corner_psf / np.sum(corner_psf)

    diff = corner_psf - center_psf

    # Use asinh stretch for difference
    norm = simple_norm(diff, 'asinh')
    axes[i + 5].imshow(diff, origin='lower', cmap='RdBu_r', norm=norm)
    axes[i + 5].set_title(f"Difference: ({x},{y}) - center")
    axes[i + 5].axis('off')

# Hide unused subplots
for idx in range(len(sample_positions) + len(sample_positions[:-1]), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
fig_path = os.path.join(output_dir, "psf_variation.png")
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"PSF variation plot saved to {fig_path}")

print("\n" + "=" * 60)
print("PSFEx position-dependent workflow completed successfully!")
print(f"PSF model saved to: {psf_file}")
print(f"Output directory: {output_dir}")
print("\nTo reconstruct PSF at any position:")
print(f"  psf = interpolate_psf_at_position('{psf_file}', x=100, y=200)")
print("=" * 60)
