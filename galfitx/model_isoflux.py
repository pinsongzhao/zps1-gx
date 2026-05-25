"""
Identifier:     galfitx/model_isoflux.py
Name:           model_isoflux.py
Description:    derive model iso flux and fluxerr
Author:         Chao Ma
Created:        2026-01-19
Modified-History:
    2026-01-19, Chao Ma, created

Photometry module for GALFIT/GALFITS processing.

This module provides functionality for:
- Generating GALFIT models from existing GALFITS results
- Analyzing background sky noise
- Computing aperture photometry with error propagation
- Converting flux units between different systems
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import distance_transform_edt
from astropy.io import fits, ascii
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u
from reproject import reproject_interp
from photutils.aperture import CircularAperture, ApertureStats
from tqdm import tqdm
from typing import Optional, Tuple, List, Union, Callable, Dict
from numba import jit, prange
from scipy.spatial import cKDTree
from dataclasses import dataclass
import traceback
from scipy.optimize import root_scalar

def circle_distance_for_overlap(frac):
    """
    计算两个等圆在给定重叠比例下的圆心距离
    
    Parameters
    ----------
    frac : float
        重叠面积占圆面积的比例 (0 到 1)
    
    Returns
    -------
    float
        圆心距离 d/r
    """
    def equation(x):
        return np.arccos(x) - x * np.sqrt(1 - x**2) - (frac / 2) * np.pi
    
    result = root_scalar(equation, bracket=[0, 1], method='brentq')
    return 2 * result.root

# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class PhotometryConfig:
    """Configuration for photometry pipeline."""
    # Image and catalog paths
    image_list: List[str]
    galaxy_catalog: str
    psf_file: Optional[str]
    segmentation_map_list: Optional[List[str]]  # Added for model flux calculation
    # band_dir_list: List[str]  # Band directories in format like ['./f444w/', './f160w/'] for GALFIT model generation and file naming
    cutout_dir: str

    # Photometry parameters
    label_list: List[str]
    filter_labels: List[str]
    gains: List[float]
    exptimes: List[float]
    mjsr_list: List[float]
    zero_list: List[float]

    # Aperture settings
    apertures_list: List[float]  # in arcsec
    pixel_scales: List[float]
    ref_pixel_scale: float = 0.03
    overlap_frac: float = 0.1  # Maximum allowed overlap fraction between apertures (0 to 1)

    match_galaxy_catalog: Optional[str] = None  # Defaults to galaxy_catalog if not provided
    def __post_init__(self):
        """Set default values for optional parameters."""
        if self.match_galaxy_catalog is None:
            self.match_galaxy_catalog = self.galaxy_catalog


    # If users do not give external mask file, we should difine detwht_file and wht_file_list to make kron mask file
    detwht_file: Optional[str] = None
    wht_file_list: Optional[List[str]] = None  # Effective mask files (whtdata==0 regions to avoid)
    # Kron mask settings
    kron_scale: list[float] = 1.5
    # If users give external_mask_file, this file lists will be directly used as mask image.
    external_mask_file_list: Optional[List[str]] = None

    # Specz settings
    specz_cat: str = None
    max_sep: float = 0.1
    
    # GALFIT settings
    galfit_path: Optional[str] = None
    gsdir: Optional[str] = None
    stampfile: Optional[str] = None
    try_filterlist: List[int] = None

    # Output settings
    sky_noise_path: str = "./sky_noise/"
    gmoutdir: str = "./galfitm/"
    outgs_flux_catfile: str = "./gs_iso_flux.cat"
    output_file: str = "./gsflux_isoerr.cat"

    # Processing options
    saveconfig: bool = False
    savemodel: bool = True
    save_mask: bool = True
    save_regions: bool = True
    plot_histograms: bool = True

    

# =============================================================================
# Numba-Accelerated Utility Functions
# =============================================================================


@jit(nopython=True, cache=True, parallel=True)
def create_kron_mask_numba(
    mask_image: np.ndarray,
    xcentroid: np.ndarray,
    ycentroid: np.ndarray,
    rad: np.ndarray,
    q: np.ndarray,
    theta: np.ndarray,
    xlo: np.ndarray,
    xhi: np.ndarray,
    ylo: np.ndarray,
    yhi: np.ndarray,
) -> np.ndarray:
    """
    Numba-accelerated Kron mask creation with parallel processing.

    Parameters
    ----------
    mask_image : numpy.ndarray
        Output mask array (will be modified in-place)
    xcentroid, ycentroid : numpy.ndarray
        Source centroids
    rad : numpy.ndarray
        Kron radii
    q : numpy.ndarray
        Axis ratios
    theta : numpy.ndarray
        Position angles in radians
    xlo, xhi, ylo, yhi : numpy.ndarray
        Bounding box coordinates

    Returns
    -------
    numpy.ndarray
        Updated mask image
    """
    ny_big, nx_big = mask_image.shape
    n_sources = len(xcentroid)

    for i in prange(n_sources):
        # Skip invalid regions
        if xhi[i] <= xlo[i] or yhi[i] <= ylo[i]:
            continue

        # Get region bounds (already clipped to image boundaries)
        y_lo = int(ylo[i])
        y_hi = int(yhi[i])
        x_lo = int(xlo[i])
        x_hi = int(xhi[i])

        # Additional sanity check: skip if centroid is outside image
        x0 = xcentroid[i]
        y0 = ycentroid[i]
        if x0 < 0 or x0 >= nx_big or y0 < 0 or y0 >= ny_big:
            continue

        # Get source parameters
        r_max = rad[i]
        q_val = q[i]
        pa = theta[i]

        # Precompute rotation
        cos_pa = np.cos(pa)
        sin_pa = np.sin(pa)

        # Loop over pixels in the bounding box
        for y in range(y_lo, y_hi + 1):
            dy = y - y0
            for x in range(x_lo, x_hi + 1):
                dx = x - x0

                # Rotate coordinates
                dx_rot = dx * cos_pa + dy * sin_pa
                dy_rot = -dx * sin_pa + dy * cos_pa

                # Calculate elliptical radius
                r = np.sqrt(dx_rot * dx_rot + (dy_rot / q_val) * (dy_rot / q_val))

                # Update mask if within Kron radius
                if r <= r_max:
                    mask_image[y, x] = 1.0

    return mask_image


# =============================================================================
# Flux Unit Conversion Utilities
# =============================================================================


class FluxConverter:
    """Utility class for flux unit conversions."""

    @staticmethod
    def flux_to_muJy(flux: float, zeropoint: float) -> float:
        """
        Convert ADU flux to microJanskys.

        Parameters
        ----------
        flux : float
            Flux in ADU units
        zeropoint : float
            Photometric zeropoint in magnitudes

        Returns
        -------
        float
            Flux in microJanskys
        """
        return flux * 3.63 * 10 ** (9 - 0.4 * zeropoint)

    @staticmethod
    def mag_to_muJy(mag: float) -> float:
        """
        Convert mag to microJanskys.

        Parameters
        ----------
        mag: float

        Returns
        -------
        float
            Flux in microJanskys
        """
        return 10 ** (mag / -2.5) * 3.63e9

    @staticmethod
    def muJy_to_mag(flux_muJy: float) -> float:
        """
        Convert microJansky flux to AB magnitude.

        Parameters
        ----------
        flux_muJy : float
            Flux in microJanskys

        Returns
        -------
        float
            AB magnitude
        """
        return -2.5 * np.log10(flux_muJy / 3.63e9)

    @staticmethod
    def muJy_to_ADU(flux_muJy: float, zeropoint: float, exptime: float, gain: float, mjsr: float) -> float:
        """
        Convert microJansky flux to ADU counts.

        Parameters
        ----------
        flux_muJy : float
            Flux in microJanskys
        zeropoint : float
            Photometric zeropoint
        exptime : float
            Exposure time in seconds
        gain : float
            Detector gain
        mjsr : float
            Magnitude per steradian conversion factor

        Returns
        -------
        float
            Flux in ADU units
        """
        return flux_muJy / 3.63e9 * 10 ** (zeropoint / 2.5) * exptime * gain * mjsr

# =============================================================================
# Cross-match Catalogs
# =============================================================================

def corss_match_df(
    cat1: pd.DataFrame,
    cat2: pd.DataFrame,
    max_sep: float = 0.1
    ) -> pd.DataFrame:
    """
    Cross-match two catalogs using astropy coordinates.

    This function matches sources from cat1 to cat2 based on their
    RA/Dec coordinates and adds spectroscopic redshift information
    from cat2 to cat1 for matched sources.

    Parameters
    ----------
    cat1 : pandas.DataFrame
        Primary catalog to be matched. Must contain 'ra' and 'dec' columns.
        This catalog will have a 'z_spec' column added with matched redshifts.
    cat2 : pandas.DataFrame
        Reference catalog containing spectroscopic redshifts.
        Must contain 'ra', 'dec', and 'z_spec' columns.
    max_sep : float, optional
        Maximum separation for matching in arcseconds.
        Sources with separation > max_sep will not be matched.
        Default is 0.1 arcsec.

    Returns
    -------
    pandas.DataFrame
        The input cat1 DataFrame with an additional 'z_spec' column.
        - Matched sources: z_spec value from cat2
        - Unmatched sources: z_spec = -99.0

    Notes
    -----
    - Uses astropy.coordinates.match_coordinates_sky for efficient matching
    - Each source in cat1 is matched to the nearest source in cat2
    - Only matches within max_sep are considered valid
    - The 'z_spec' column is initialized to -99.0 (sentinel value)

    Examples
    --------
    >>> import pandas as pd
    >>> # Primary catalog (photometric redshifts)
    >>> cat1 = pd.DataFrame({
    ...     'ra': [150.0, 150.1, 150.2],
    ...     'dec': [2.0, 2.1, 2.2]
    ... })
    >>> # Reference catalog (spectroscopic redshifts)
    >>> cat2 = pd.DataFrame({
    ...     'ra': [150.001, 150.201],
    ...     'dec': [2.001, 2.201],
    ...     'z_spec': [0.5, 0.8]
    ... })
    >>> matched = corss_match_df(cat1, cat2, max_sep=2.0)
    >>> print(matched[['ra', 'dec', 'z_spec']])
    """
    # Extract coordinates from both catalogs
    # Ensure values are numeric arrays (handle case where they might be Quantity objects)
    ra1 = np.asarray(cat1['ra'], dtype=float)
    dec1 = np.asarray(cat1['dec'], dtype=float)

    ra2 = np.asarray(cat2['ra'], dtype=float)
    dec2 = np.asarray(cat2['dec'], dtype=float)
    specz2 = cat2['z_spec']

    # Create SkyCoord objects for matching
    coords1 = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)
    coords2 = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)

    # Perform coordinate matching
    # idx: indices of closest matches in coords2
    # d2d: angular separation to the closest match
    idx, d2d, _ = match_coordinates_sky(coords1, coords2)

    # Apply separation constraint
    max_sep_units = max_sep * u.arcsec
    sep_constraint = d2d < max_sep_units

    # Initialize z_spec column with sentinel value
    cat1['z_spec'] = -99.

    # Assign redshifts for matched sources
    cat1.loc[sep_constraint, 'z_spec'] = specz2.iloc[idx[sep_constraint]].values

    return cat1

# =============================================================================
# GALFIT Model Generation
# =============================================================================


class GalfitModelGenerator:
    """Generate GALFIT models from GALFITS results."""

    def __init__(self, galfit_path: str):
        """
        Initialize the model generator.

        Parameters
        ----------
        galfit_path : str
            Path to the GALFIT executable
        """
        self.galfit_path = galfit_path

    def get_flux_fraction(
        self,
        gal_id: int,
        input_image: str,
        psf_file: str,
        filter: str,
        # band_dir: str,
        cutout_dir: str,
        band_label:str,
        pixel_scale: float,
        zeropoint: float,
        stamp_file: str = "./stamps",
        catalog_name: str = "./sex/outcat",
        segmentation_map: str = "./sex/outseg.fits",
        gs_dir: str = "./galfits/",
        gmout_dir: str = "./galfitm/",
        saveconfig: bool = True,
        savemodel: bool = True,
    ) -> Optional[float]:
        """
        Calculate the flux fraction from a GALFIT model.

        Parameters
        ----------
        gal_id : int
            Galaxy ID number
        input_image : str
            Path to input image
        psf_file : str
            Path to PSF file
        band : str
            Filter band name
        filter_dir : str
            Directory containing filter images
        pixel_scale : float
            Pixel scale in arcsec/pixel
        zeropoint : float
            Photometric zeropoint
        stamp_file : str
            Path to stamp catalog
        catalog_name : str
            Path to SExtractor catalog
        segmentation_map : str
            Path to segmentation map FITS file
        gs_dir : str
            GALFITS results directory
        gmout_dir : str
            Output directory for models
        save_config : bool
            Whether to save GALFIT config file
        save_model : bool
            Whether to save model FITS file

        Returns
        -------
        float or None
            Flux fraction, or None if file not found
        """
        # Create output directory
        os.makedirs(gmout_dir, exist_ok=True)

        # Read stamp coordinates
        stamps = ascii.read(stamp_file)
        stamp_idx = np.where(stamps.columns[0] == gal_id)[0][0]

        x = stamps.columns[1][stamp_idx]   ## 1 based 
        y = stamps.columns[2][stamp_idx]
        xlo = stamps.columns[5][stamp_idx]-1
        xhi = stamps.columns[6][stamp_idx]-1
        ylo = stamps.columns[7][stamp_idx]-1
        yhi = stamps.columns[8][stamp_idx]-1

        refdata = fits.getdata(input_image)
        ny_full, nx_full = refdata.shape
        

        # xdim = nx_full
        # ydim = ny_full

        # xlo = max(xlo, 0)   ### xlo, ylo already changed to 0-based before.
        # ylo = max(ylo, 0)
        xlo = max(xlo, 0)
        xhi = min(xhi, nx_full -1)
        ylo = max(ylo, 0)
        yhi = min(yhi, ny_full -1)
        
        xdim = xhi - xlo + 1
        ydim = yhi - ylo + 1

        # Read GALFITS results
        summary_file = os.path.join(gs_dir,f"obj{gal_id}.gssummary")
        if not os.path.exists(summary_file):
            print(f"{summary_file} does not exist.")
            return None

        result = ascii.read(summary_file)

        # Extract Sersic parameters
        mag = result["best_value"][result["pname"] == f"Mag_obj0_{filter}"][0]
        re = result["best_value"][result["pname"] == "obj0_Re"][0] / pixel_scale
        n = result["best_value"][result["pname"] == "obj0_n"][0]
        q = result["best_value"][result["pname"] == "obj0_axrat"][0]
        pa = result["best_value"][result["pname"] == "obj0_ang"][0]

        # Compute position angle correction to align with North
        pa = pa + self._correct_position_angle(cutout_dir, band_label, gal_id) + 90
        pa = (pa+360)%360
        

        # Generate and run GALFIT config
        config_file = f"{gmout_dir}galfit_input{gal_id}"

        self._write_galfit_config(
            config_file,
            f"{gmout_dir}model{gal_id}.fits",
            psf_file,
            xdim,
            ydim,
            x,
            y,
            xlo,
            ylo,
            zeropoint,
            pixel_scale,
            mag,
            re,
            n,
            q,
            pa,
        )

        # Run GALFIT
        if not os.path.exists(f"{gmout_dir}model{gal_id}.fits"):
            os.system(f"{self.galfit_path} -o1 {config_file}")

        # Cleanup temporary files
        if not saveconfig:
            os.remove(config_file)
        if not savemodel:
            os.remove(f"{gmout_dir}model{gal_id}.fits")

        # Calculate flux fraction from model
        return self._calculate_isofluxfrac(gal_id, gmout_dir, segmentation_map, zeropoint, mag, xlo, xhi, ylo, yhi)

    def _correct_position_angle(self, cutout_dir: str, band_label:str, gal_id: int) -> float:
        """Correct position angle to align with North."""
        # Extract directory name from path for file prefix
        # cutfile = f"{band_dir}{gal_id}.fits"
        cutfile = os.path.join(cutout_dir, f"obj{gal_id}_{band_label}sci.fits")
        header = fits.getheader(cutfile)
        wcs = WCS(header)

        xshape = header["NAXIS1"]
        yshape = header["NAXIS2"]

        ra,dec = wcs.all_pix2world((xshape-1)/2, (yshape-1)/2, 0)
            
        srcPstXY = wcs.all_world2pix([ra], [dec], 1)
        srcXp = srcPstXY[0][0]
        srcYp = srcPstXY[1][0]
        srcPstXY = wcs.all_world2pix([ra + 1.0 / 60], [dec], 1)
        srcPstXY = wcs.all_world2pix([ra], [dec + 1.0 / 60], 1)
        srcXpdec = srcPstXY[0][0]
        srcYpdec = srcPstXY[1][0]

        dx, dy = srcXpdec - srcXp, srcYpdec - srcYp
        pa_north = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
        delta_ang = pa_north

        return delta_ang

    def _write_galfit_config(
        self,
        config_file: str,
        output_file: str,
        psf_file: str,
        xdim: int,
        ydim: int,
        x: float,
        y: float,
        xlo: int,
        ylo: int,
        zeropoint: float,
        pixel_scale: float,
        mag: float,
        re: float,
        n: float,
        q: float,
        pa: float,
    ) -> None:
        """Write GALFIT configuration file."""
        config_lines = [
            f"A) none      # Input data image (FITS file)",
            f"B) {output_file}     # Output model image",
            "C) none                # Sigma image name",
            f"D) {psf_file}         # Input PSF",
            "E) 1                   # PSF fine sampling factor",
            "F) none            # Bad pixel mask",
            "G) none                # Constraint file",
            f"H) 1 {xdim} 1 {ydim}    # Region to fit",
            f"I) 200 200  # Convolution box",
            f"J) {zeropoint}        # Magnitude zeropoint",
            f"K) {pixel_scale} {pixel_scale}      # Plate scale",
            "O) regular             # Display type",
            "P) 1                   # Choose: 0=optimize, 1=model",
            "",
            "0) sersic             # Object type",
            #f"1) {round(x+1-xlo,2)} {round(y+1-ylo,2)} 1 1      # Position x, y",
            f"1) {round(x-xlo,2)} {round(y-ylo,2)} 1 1      # Position x, y",
            f"3) {mag:.3f} 1         # Integrated magnitude",
            f"4) {re:.3f} 1         # R_e [pix]",
            f"5) {n:.3f} 1         # Sersic index n",
            f"9) {q:.3f} 1         # Axis ratio (b/a)",
            f"10) {pa:.3f} 1     # Position angle [deg]",
            f"Z) 0                  # Output option",
        ]

        with open(config_file, "w") as f:
            f.write("\n".join(config_lines))

    def _calculate_isofluxfrac(
        self,
        gal_id: int,
        output_dir: str,
        segmentation_map: str,
        zeropoint: float,
        mag: float,
        xlo: int,
        xhi: int,
        ylo: int,
        yhi: int,
    ) -> float:
        """Calculate isophotal flux from model."""
        # model = fits.getdata(f"{output_dir}model{gal_id}.fits", ext=0)
        hdul = fits.open(f"{output_dir}model{gal_id}.fits")
        for hdu in hdul:
            if hdu.header.get('EXTNAME') == 'MODEL': # 假设模型扩展的EXTNAME是'MODEL'
                model_hdu = hdu
                model = hdu.data
        segmap = fits.getdata(segmentation_map, ext=0)
        segcut = segmap[ylo: yhi + 1, xlo: xhi + 1]

        isoflux_bunit = np.sum(model[segcut == gal_id])
        isomag = -2.5 * np.log10(isoflux_bunit) + zeropoint

        return 10 ** ((mag - isomag) / 2.5)


# =============================================================================
# Background Sky Noise Analysis
# =============================================================================


class BackgroundAnalyzer:
    """Analyze background sky noise for aperture photometry."""

    def create_kron_mask_map(
        self,
        catalog_name: str,
        detwht_file: str,
        kron_scale: float = 1.0,
        output_path: str = "./sky_noise/"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute integrated mask and distance map for a band.

        This is called once per band, not per aperture, to avoid redundant computation.

        Parameters
        ----------
        catalog_name : str
            Path to SExtractor catalog
        detwht_file : str
            Path to detection weight image
        kron_scale : float
            Scale factor for Kron radius
        output_path : str
            Path to save mask files

        Returns
        -------
        str
            Path to save mask files
        """
        # Check if integrated mask already exists (Kron + wht combined)
        mask_file = f"{output_path}detect_kron_scale{kron_scale}_mask.fits"
        if os.path.exists(mask_file):
            return mask_file
        else:
            # Need to generate the integrated mask
            # Read science image to get dimensions
            detwht_image = fits.getdata(detwht_file)
            ny_big, nx_big = detwht_image.shape

            # Read catalog to get source parameters
            cat = ascii.read(catalog_name)

            # Create mask image using Kron radius for each source
            mask_image = np.zeros((ny_big, nx_big), dtype=float)
            mask_image[~(detwht_image > 0)] = 1.
            # Calculate all parameters at once (vectorized)
            theta = cat["orientation"] * np.pi / 180  # convert degrees to radians
            rad = kron_scale * cat["semimajor_sigma"] * cat["kron_radius"]
            q = 1.0 - cat["ellipticity"]  # Axis ratio (b/a), should be ≤ 1

            # Vectorized calculation of bounding boxes
            # For rotated ellipse, compute extent in each direction
            xfac = rad * (np.abs(np.cos(theta)) + q * np.abs(np.sin(theta)))
            yfac = rad * (np.abs(np.sin(theta)) + q * np.abs(np.cos(theta)))
            xfac = np.maximum(xfac, 10)
            yfac = np.maximum(yfac, 10)

            xcentroid = cat["xcentroid"]
            ycentroid = cat["ycentroid"]

            xlo = np.maximum(np.round(xcentroid - xfac).astype(int), 0)
            xhi = np.minimum(np.round(xcentroid + xfac).astype(int), nx_big - 1)
            ylo = np.maximum(np.round(ycentroid - yfac).astype(int), 0)
            yhi = np.minimum(np.round(ycentroid + yfac).astype(int), ny_big - 1)

            # Use Numba-accelerated mask creation
            mask_image = create_kron_mask_numba(
                mask_image,
                xcentroid.astype(np.float64),
                ycentroid.astype(np.float64),
                rad.astype(np.float64),
                q.astype(np.float64),
                theta.astype(np.float64),
                xlo.astype(np.float64),
                xhi.astype(np.float64),
                ylo.astype(np.float64),
                yhi.astype(np.float64),
            )

            # Save integrated Kron+wht mask if requested
            # Get header from science image
            header = fits.getheader(detwht_file)
            fits.writeto(mask_file, mask_image.astype(np.uint8), header=header, overwrite=True)
            print(f"  Saved Kron mask to {mask_file}")

        return mask_file
    
    def create_integrated_mask_map(
        self,
        filter_name: str,
        catalog_name: str,
        detwht_file: str,
        wht_file: str,
        kron_scale: float = 1.0,
        output_path: str = "./sky_noise/",
        save_mask: bool = False,
    ) -> Tuple[np.ndarray]:
        """
        Compute integrated mask and distance map for a band.

        This is called once per band, not per aperture, to avoid redundant computation.

        Parameters
        ----------
        filter_name : str
            Filter name for file naming (e.g., 'f444w', 'f160w')
        catalog_name : str
            Path to SExtractor catalog
        detwht_image : str
            Path to detection weight image FITS file
        wht_file : str, optional
            Path to weight image FITS file
        kron_scale : float
            Scale factor for Kron radius
        output_path : str
            Path to save mask files
        save_mask : bool
            Whether to save the mask file

        Returns
        -------
        tuple
            mask_image arrays
        """
        
        # Check if integrated mask already exists (Kron + wht combined)
        mask_file = f"{output_path}{filter_name}_kron_scale{kron_scale}_integrated_mask.fits"
        if save_mask and os.path.exists(mask_file):
            print(f"  Loading existing integrated mask: {mask_file}")
            mask_image = fits.getdata(mask_file).astype(float)
        else:
            kron_mask_file = f"{output_path}detect_kron_scale{kron_scale}_mask.fits"
            if not os.path.exists(kron_mask_file):
                # Need to generate the Kron mask
                self.create_kron_mask_map(
                    catalog_name,
                    detwht_file,
                    kron_scale,
                    output_path
                )

            print(f"  Loading existing Kron mask: {kron_mask_file}")
            kron_mask_hdu = fits.open(kron_mask_file)

            # Read effective mask image to get dimensions
            wht_header = fits.getheader(wht_file)
            wht_image = fits.getdata(wht_file)
            ny_eff, nx_eff = wht_image.shape

            # reproject the Kron mask to the science image
            mask_image, _ = reproject_interp(kron_mask_hdu, wht_header, order="nearest-neighbor")
            # Need to generate the integrated mask
            
            # Apply wht BEFORE saving (so saved mask includes both Kron and wht)
            print(f"  Applying wht: {wht_file}")
            # Mask out regions where wht == 0 OR nan (invalid weight regions)
            invalid_pixels = (wht_image == 0) | (np.isnan(wht_image))
            mask_image[:ny_eff, :nx_eff][invalid_pixels] = 1

            # Save integrated Kron+wht mask if requested
            if save_mask:
                fits.writeto(mask_file, mask_image.astype(np.uint8), header=wht_header, overwrite=True)
                print(f"  Saved integrated mask (Kron + wht) to {mask_file}")


        return mask_image

    def calculate_distance_map(
        self,
        filter_name: str,
        catalog_name: str,
        detwht_file: str,
        wht_file: str,
        kron_scale: float = 1.0,
        external_mask_file: Optional[str] = None,
        output_path: str = "./sky_noise/",
        save_mask: bool = False,
    ):
        if external_mask_file is None:
            mask_image = self.create_integrated_mask_map(
                filter_name,
                catalog_name,
                detwht_file,
                wht_file,
                kron_scale,
                output_path,
                save_mask
            )
        else:
            mask_image = fits.getdata(external_mask_file)

        # Compute distance transform (only ONCE per band)
        
        print(f"  Computing distance transform...")

        # Add boundary constraint: mask image edges to prevent apertures from going out of bounds
        # This ensures that distance to boundary is considered when sampling aperture positions
        
        mask_with_boundary = mask_image.copy()
        mask_with_boundary[0, :] = 1  # Top edge
        mask_with_boundary[-1, :] = 1  # Bottom edge
        mask_with_boundary[:, 0] = 1  # Left edge
        mask_with_boundary[:, -1] = 1  # Right edge

        distance_map = distance_transform_edt(1.0 - mask_with_boundary)
        print(f"    Distance map computed: shape={distance_map.shape}")
        return distance_map

    @staticmethod
    def check_overlapping_apertures(positions: np.ndarray, min_distance: float) -> np.ndarray:
       """
       Remove overlapping apertures from a set of positions.

       Parameters
       ----------
       positions : numpy.ndarray
           Array of (y, x) positions
       min_distance : float
           Minimum distance between apertures (e.g., 2*radius for no overlap)

       Returns
       -------
       numpy.ndarray
           Filtered positions without overlaps
       """
        

       num_apertures = positions.shape[0]
       distances = pdist(positions, "euclidean")

       # Convert to full distance matrix and check overlaps
        
       dist_matrix = squareform(distances)  # Shape: (num_apertures, num_apertures)

       # Check for overlaps (distance < 2*radius, excluding self-comparison)
       overlap_matrix = (dist_matrix < min_distance) & (dist_matrix > 0)

       # Count overlaps per aperture
       sums = np.sum(overlap_matrix, axis=1)

       # Greedily remove apertures with most overlaps until no overlaps remain
       keep = np.ones(num_apertures, dtype=bool)
       while np.any(sums > 0):
           i = np.argmax(sums)
           keep[i] = False
           # Update overlap matrix: remove row i and column i
           overlap_matrix[i, :] = False
           overlap_matrix[:, i] = False
           sums = np.sum(overlap_matrix, axis=1)

       print(f'  Removed {num_apertures - np.sum(keep)} overlapping apertures, kept {np.sum(keep)}')
       return positions[keep]

    def decide_background_positions_with_distance_map(
        self,
        filter_name: str,
        aperture_radius: float,
        overlap_frac: float,
        distance_map: np.ndarray,
        output_path: str = "./sky_noise/",
        save_regions: bool = False,
    ) -> List[Tuple[float, float]]:
        """
        Decide background aperture positions using pre-computed distance map.

        This is much faster than decide_background_positions as it reuses the
        distance map computed for the band.

        Parameters
        ----------
        filter_name : str
            Filter name for file naming (e.g., 'f444w', 'f160w')
        aperture_radius : float
            Aperture radius in pixels
        distance_map : numpy.ndarray
            Pre-computed distance transform of mask
        output_path : str
            Path to save region files
        save_regions : bool
            Whether to save DS9 region file

        Returns
        -------
        list of tuple
            List of (x, y) positions for background apertures
        """
        
        # Check if region file already exists
        if save_regions:
            reg_file = f'{output_path}{filter_name}_bkg_aper{aperture_radius:.2f}.reg'
            if os.path.exists(reg_file):
                return []

        # Find pixels where distance >= aperture_radius (using pre-computed distance map)
        valid_pixels = np.array(np.where(distance_map > aperture_radius)).T

        if len(valid_pixels) == 0:
            print(f"  Warning: No valid pixels found for aperture radius {aperture_radius:.2f}")
            return []

        # Sample random positions (adaptive number based on available area)
        num_valid = len(valid_pixels)
        # Aim for ~500-5000 final apertures, sample more to account for overlaps
        num_samples = min(5000, num_valid)
        
        
        if num_valid <= num_samples*4*aperture_radius**2:  # If area is small, sample all and rely on overlap removal
            
            tree = cKDTree(valid_pixels)
            order = np.arange(num_valid)
            removed = np.zeros(num_valid, dtype=bool)
            selected_indices = []
            if overlap_frac == 0:
                min_distance = 2*aperture_radius
            else:
                min_distance = aperture_radius*circle_distance_for_overlap(frac=overlap_frac)
            
            for idx in order:
                if removed[idx]:
                    continue
                selected_indices.append(idx)
                # Find neighbors within min_distance and mark them as removed
                neighbors = tree.query_ball_point(valid_pixels[idx], min_distance)
                # remove all neighbors from consideration
                removed[neighbors] = True
            
            non_overlapping = valid_pixels[selected_indices]
            if len(non_overlapping) > num_samples:
                rng = np.random.default_rng(123456789)
                rand_indices = rng.integers(low=0, high=len(non_overlapping), size=num_samples)
                non_overlapping = non_overlapping[rand_indices]

        else:
            rng = np.random.default_rng(123456789)
            rand_indices = rng.integers(low=0, high=num_valid, size=num_samples)
            positions = valid_pixels[rand_indices]

            # Remove overlapping apertures
            print(f"    Before overlap check: {len(positions)} positions")
            min_distance = aperture_radius*circle_distance_for_overlap(frac=overlap_frac)
            non_overlapping = self.check_overlapping_apertures(positions, min_distance)
            
            print(f"  Aperture radius {aperture_radius:.2f}px: {len(non_overlapping)} non-overlapping positions")

        if save_regions:
            self._save_region_file(filter_name, aperture_radius, non_overlapping, output_path)

        return list(zip(non_overlapping[:, 1], non_overlapping[:, 0]))

    def _save_region_file(self, band: str, aperture_radius: float, positions: np.ndarray, output_path: str) -> None:
        """Save DS9 region file for background apertures."""
        filename = f"{output_path}{band}_bkg_aper{aperture_radius:.2f}.reg"
        with open(filename, "w") as f:
            f.write("# Region file format: DS9 version 4.1\n")
            f.write('global color=blue dashlist=8 3 width=1 font="helvetical 10 normal roman" ')
            f.write("select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
            f.write("image\n")
            for pos in positions:
                x, y = pos[1], pos[0]
                ds9_x = x + 1  # Convert to 1-based coordinates for DS9
                ds9_y = y + 1
                f.write(f"circle({ds9_x:.4f},{ds9_y:.4f},{aperture_radius:.1f})\n")

    def compute_background_noise(
        self,
        filter_name: str,
        sigma_1: float,
        aperture_radii: np.ndarray,
        pixel_scale: float,
        output_path: str,
        science_file: str,
        plot_histograms: bool = True,
    ) -> Tuple[float, float, float, float]:
        """
        Compute background noise as a function of aperture radius using single power law model.

        Model: sigma = sigma_1 * (alpha * r^beta)

        Parameters
        ----------
        filter_name : str
            Filter name for file naming and plot labels (e.g., 'f444w', 'f160w')
        sigma_1: float
            Sigma of background 
        aperture_radii : numpy.ndarray
            Array of aperture radii in arcsec
        pixel_scale : float
            Pixel scale in arcsec/pixel
        output_path : str
            Path to save plots
        science_file : str
            Path to science image FITS file
        kron_scale : float
            Kron scale factor used to generate mask (default: 1.0)
        plot_histograms : bool
            Whether to create histogram plots

        Returns
        -------
        tuple
            (alpha, beta) for sigma = sigma_1 * (alpha * r^beta)
        """
        science_data = fits.getdata(science_file)
        aperture_radii_px = (aperture_radii / pixel_scale).astype(float)
        aperture_radii_px = aperture_radii_px[aperture_radii_px > 0]

        std_list = []
        min_radius = np.min(aperture_radii_px)
        max_radius = np.max(aperture_radii_px)

        for radius in aperture_radii_px:
            reg_file = f"{output_path}{filter_name}_bkg_aper{radius:.2f}.reg"
            text = pd.read_csv(reg_file, header=None, skiprows=3, delimiter=",")
            bkg_positions = list(zip(text[0].str[7:].astype(float)-1, text[1]-1))  # Convert to 0-based coordinates

            aperture_fluxes = ApertureStats(science_data, CircularAperture(bkg_positions, r=radius)).sum
            aperture_fluxes = aperture_fluxes[aperture_fluxes != 0]

            sigma = 1.4826 * np.median(np.abs(aperture_fluxes - np.median(aperture_fluxes)))

            std_list.append(sigma)

            # Plot histograms for min/max apertures
            if plot_histograms:
                if radius == min_radius or radius == max_radius:
                    self._plot_aperture_histograms(
                        filter_name, radius, aperture_radii_px, aperture_fluxes, output_path
                    )

        # Fit single power law noise model: sigma = sigma_1 * (alpha * r^beta)
        model, p0, bounds = self._get_single_model_config(sigma_1)
        fit_result = self._fit_noise_model(aperture_radii_px, std_list, model, p0, bounds)

        # Save noise curve plot
        self._plot_noise_curve(
            filter_name,
            aperture_radii_px,
            std_list,
            fit_result['params'],
            pixel_scale,
            output_path,
            model
        )

        # Return fitted parameters (alpha, beta)
        return tuple(fit_result['params'])

    def _plot_aperture_histograms(
        self, filter_name: str, radius: int, all_radii: np.ndarray, fluxes: np.ndarray, output_path: str
    ) -> None:
        """Plot aperture flux histograms for min and max radii."""
        min_radius = np.min(all_radii)
        max_radius = np.max(all_radii)

        flux_median = np.median(fluxes)
        flux_mean, _, flux_std = sigma_clipped_stats(fluxes - flux_median, sigma=5)
        valid_mask = (fluxes > flux_mean - flux_std * 5) & (fluxes < flux_mean + flux_std * 5)

        plt.figure(figsize=(8, 6))

        if (radius == min_radius) or radius == max_radius:
            plt.hist(
                fluxes[valid_mask] / (np.pi * min_radius**2),
                bins=100,
                label=f"r = {round(radius, 2)} pix",
                color="b",
                histtype="step",
                density=True,
            )
            plt.axvline(np.median(fluxes[valid_mask] / (np.pi * min_radius**2)), ls="--", color="b")
        
        plt.xlabel(f"{filter_name} aperture flux density std")
        plt.legend()
        plt.savefig(f"{output_path}{filter_name}_bkg_aper{radius}fluxhist.png", dpi=100)
        plt.close()

    def _plot_noise_curve(
        self,
        filter_name: str,
        radii: np.ndarray,
        stds: List[float],
        params: np.ndarray,
        pixel_scale: float,
        output_path: str,
        model: Callable,
    ) -> None:
        """
        Plot background noise vs aperture diameter.

        Parameters
        ----------
        filter_name : str
            Filter name for labeling
        radii : np.ndarray
            Aperture radii in pixels
        stds : List[float]
            Measured noise values
        params : np.ndarray
            Fitted model parameters
        pixel_scale : float
            Pixel scale in arcsec/pixel
        output_path : str
            Path to save plot
        model : Callable
            Model function with signature f(x, *params)
        """
        min_r = np.min(radii)
        max_r = np.max(radii)
        x = np.linspace(min_r, max_r, 100)
        y = model(x, *params)

        plt.figure(figsize=(8, 6))
        plt.plot(x * 2 * pixel_scale, y, c="k", ls="--", label="Fit")
        plt.scatter(radii * 2 * pixel_scale, stds, marker="^", label="Data")

        plt.xlabel("Aperture diameter [arcsec]")
        plt.ylabel(f"{filter_name} $\\sigma$")

        # Display parameters for single power law model: sigma = sigma_1 * (alpha * r^beta)
        alpha, beta = params
        param_text = (
            r"$\sigma_1 \cdot (\alpha \cdot r^{\beta})$" + "\n" +
            r"$\alpha$ = " + f"{alpha:.4f}" + "\n" +
            r"$\beta$ = " + f"{beta:.4f}"
        )

        plt.text(
            0.1,
            0.8,
            param_text,
            fontsize=13,
            transform=plt.gca().transAxes,
        )
        plt.legend()
        plt.savefig(f"{output_path}{filter_name}_bkg_aperstdcurve.png", dpi=100)
        plt.close()

    @staticmethod
    def _noise_model(r: np.ndarray, sigma_1: float, alpha: float, beta: float) -> np.ndarray:
        """
        Single power law noise model: sigma = sigma_1 * (alpha * r^beta).

        Parameters
        ----------
        r : np.ndarray
            Aperture radii
        sigma_1 : float
            Background noise at 1 pixel (computed from background pixels)
        alpha : float
            Amplitude of power law component
        beta : float
            Power index of component (typically 1 < beta < 2)

        Returns
        -------
        np.ndarray
            Predicted noise values

        Examples
        --------
        >>> noise = _noise_model(r, 0.01, 1.0, 1.8)
        >>> # sigma = 0.01 * (1.0 * r^1.8)
        """
        return sigma_1 * (alpha * r**beta)

    def _get_single_model_config(self, sigma_1: float):
        """
        Get single power law model: sigma = sigma_1 * (alpha * r^beta).

        Parameters
        ----------
        sigma_1 : float
            Background noise at 1 pixel

        Returns
        -------
        tuple
            (model_function, initial_guess, bounds)
            - model_function: f(r, alpha, beta) = sigma_1 * (alpha * r^beta)
            - initial_guess: None (let curve_fit decide)
            - bounds: ((alpha_min, beta_min), (alpha_max, beta_max))
        """
        return (
            lambda r, alpha, beta: self._noise_model(r, sigma_1, alpha, beta),
            None,
            ((1e-10, 1.0), (np.inf, 2.0))
        )

    def _fit_noise_model(
        self,
        x: np.ndarray,
        y: np.ndarray,
        model: Optional[Callable] = None,
        p0: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[Tuple, Tuple]] = None,
    ) -> Dict:
        """
        Fit a noise model to data.

        Parameters
        ----------
        x : np.ndarray
            Independent variable (aperture radii)
        y : np.ndarray
            Dependent variable (noise values)
        model : Callable, optional
            Model function to fit. If None, uses simple power law.
        p0 : np.ndarray, optional
            Initial parameter guesses
        bounds : tuple of tuples, optional
            Parameter bounds ((lower...), (upper...))

        Returns
        -------
        dict
            Dictionary containing:
            - 'params': Optimal parameters
            - 'covariance': Covariance matrix
            - 'std_errors': Standard errors
            - 'y_fit': Fitted y values
            - 'residuals': Residuals
            - 'rmse': Root mean square error
            - 'r_squared': R-squared value
        """
        if model is None:
            model = self._noise_model
            if p0 is None:
                p0 = [1.0, 1.5]

        try:
            popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=10000)

            # Calculate statistics
            perr = np.sqrt(np.diag(pcov))
            y_fit = model(x, *popt)
            residuals = y - y_fit
            rmse = np.sqrt(np.mean(residuals**2))

            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return {
                'params': popt,
                'covariance': pcov,
                'std_errors': perr,
                'y_fit': y_fit,
                'residuals': residuals,
                'rmse': rmse,
                'r_squared': r_squared
            }

        except Exception as e:
            raise RuntimeError(f"Noise model fitting failed: {e}")


# =============================================================================
# Aperture Photometry with Error Propagation
# =============================================================================


class AperturePhotometry:
    """Perform aperture photometry with error propagation."""

    def __init__(self, converter: FluxConverter):
        """
        Initialize photometry calculator.

        Parameters
        ----------
        converter : FluxConverter
            Flux unit converter instance
        """
        self.converter = converter

    def compute_isophotal_errors(
        self,
        catalog: pd.DataFrame,
        flux_frac_catfile: pd.DataFrame,
        band: str,
        band_label: str,
        background_params: pd.DataFrame,
        gain: float,
        exptime: float,
        mjsr: float,
        zeropoint: float,
        pixel_scale: float,
        ref_pixel_scale: float = 0.03,
    ) -> Tuple[np.ndarray, ...]:
        """
        Compute isophotal flux errors for all sources.

        Parameters
        ----------
        catalog : pandas.DataFrame
            Source catalog with segment_area
        flux_catalog : pandas.DataFrame
            GALFITS flux catalog
        band : str
            Filter band name
        band_label : str
            Band label for flux column
        background_params : pandas.DataFrame
            Background noise parameters (ALPHA, BETA per band)
        gain : float
            Detector gain
        exptime : float
            Exposure time
        mjsr : float
            Magnitude per steradian factor
        zeropoint : float
            Photometric zeropoint
        pixel_scale : float
            Pixel scale in arcsec/pixel
        ref_pixel_scale : float
            Reference pixel scale for detection image

        Returns
        -------
        tuple of numpy.ndarray
            (total_flux, object_error, correlation_error, total_error)
        """
        # Get background parameters for this band
        bg_row = background_params[background_params["BAND"] == band]
        if len(bg_row) == 0:
            raise ValueError(f"No background parameters found for band {band}")

        # Read all 3 parameters: (sigma_1, alpha, beta)
        sigma_1, alpha, beta = bg_row[["SIGMA_1", "ALPHA", "BETA"]].values[0]
        bg_params = (sigma_1, alpha, beta)

        source_ids = catalog["label"].values
        ref_ids = flux_frac_catfile["id"].values
        ref_fluxfracs = flux_frac_catfile["flux_frac"].values
        ref_mages = flux_frac_catfile[f"m_{band_label}"].values

        num_sources = len(ref_ids)

        total_flux = np.full(num_sources, -99.0)
        object_errors = np.full(num_sources, -99.0)
        correlation_errors = np.full(num_sources, -99.0)
        total_errors = np.full(num_sources, -99.0)

        # Compute isophotal radii
        # iso_radii = np.sqrt(catalog["segment_area"].values / np.pi)
        source_ref_idx = np.array([np.where(source_ids == rid)[0][0] for rid in ref_ids])
        iso_radii_cut = np.sqrt(catalog["segment_area"][source_ref_idx].values / np.pi)
        for i in tqdm(range(num_sources), desc=band.ljust(10) + " Photometry", ncols=100):
            ref_mag = ref_mages[i]
            if ref_mag <= -99.0:
                continue
            ref_flux = self.converter.mag_to_muJy(ref_mag) * ref_fluxfracs[i]

            # Convert reference flux to physical units
            cir_radius = iso_radii_cut[i] / (pixel_scale / ref_pixel_scale)
            flux_adu = self.converter.muJy_to_ADU(ref_flux, zeropoint, exptime, gain, mjsr)

            # Object error (Poisson noise)
            object_errors[i] = self.converter.flux_to_muJy(np.sqrt(flux_adu) / (exptime * gain * mjsr), zeropoint)

            # Correlation error (background) - automatically uses simple or single model based on params
            correlation_errors[i] = self.converter.flux_to_muJy(
                BackgroundAnalyzer._noise_model(cir_radius, *bg_params), zeropoint
            )

            # Total flux and error
            total_flux[i] = ref_flux
            total_errors[i] = np.sqrt(object_errors[i] ** 2 + correlation_errors[i] ** 2)

        return total_flux, object_errors, correlation_errors, total_errors


# =============================================================================
# Convenience Function (Backward Compatible - Direct Function Call)
# =============================================================================


def process_multi_band_photometry(**kwargs) -> None:
    """
    Process multi-band photometry using configuration parameters.

    This is a convenience function that creates a PhotometryConfig object
    and runs the pipeline. Can be called with either a PhotometryConfig object
    or individual configuration parameters.

    Parameters
    ----------
    **kwargs : dict
        Either:
        - config: PhotometryConfig object
        - Individual PhotometryConfig parameters (bands, image_list, etc.)

    Examples
    --------
    # Using config object
    process_multi_band_photometry(config=my_config)

    # Using individual parameters
    process_multi_band_photometry(
        band_dir_list=["./f444w/", "./f160w/"],
        image_list=["path1.fits", "path2.fits"],
        ...
    )
    """
    if "config" in kwargs:
        config = kwargs["config"]
    else:
        config = PhotometryConfig(**kwargs)

    pipeline = PhotometryPipeline(config)
    pipeline.run_all_steps()


# =============================================================================
# Photometry Pipeline (Refactored)
# =============================================================================


class PhotometryPipeline:
    """
    Pipeline for multi-band photometry processing with error propagation.

    This class orchestrates the full pipeline:
    1. Determine background aperture positions
    2. Fit background noise models
    3. Compute model isophotal fluxes
    4. Compute isophotal fluxes with errors
    """

    def __init__(self, config: PhotometryConfig):
        """
        Initialize the pipeline with configuration.

        Parameters
        ----------
        config : PhotometryConfig
            Configuration object containing all parameters
        """
        self.config = config
        self.analyzer = BackgroundAnalyzer()
        self.converter = FluxConverter()
        self.photometry = AperturePhotometry(self.converter)
        self.generator = None

        if config.galfit_path is not None:
            self.generator = GalfitModelGenerator(config.galfit_path)

        # Create output directories
        os.makedirs(config.sky_noise_path, exist_ok=True)
        os.makedirs(config.gmoutdir, exist_ok=True)

    def run_all_steps(self) -> None:
        """Run all pipeline steps sequentially."""
        self.step1_determine_background_positions()
        bg_params = self.step2_fit_background_noise()
        self.step3_compute_model_fluxes()
        self.step4_compute_isophotal_errors(bg_params, flux_df_path=self.config.outgs_flux_catfile, gs_flux_err_outfile=self.config.output_file)

    def step1_determine_background_positions(self) -> None:
        """
        Step 1: Determine background aperture positions for each band.

        Uses Kron radius masking with distance transform for optimal performance
        with multiple apertures.
        """
        print("Step 1: Determining background aperture positions...")
        for i, (label, filter_label) in enumerate(zip(self.config.label_list, self.config.filter_labels)):
            print(f"Processing {filter_label} ...")
            start_time = time.time()

            # Get wht file for this band if available
            wht_file = None
            if self.config.wht_file_list is not None and i < len(self.config.wht_file_list):
                wht_file = self.config.wht_file_list[i]
                if wht_file and not os.path.exists(wht_file):
                    print(f"  Warning: wht file not found: {wht_file}")
                    wht_file = None

            external_mask_file = None
            if self.config.external_mask_file_list is not None and i < len(self.config.external_mask_file_list):
                external_mask_file = self.config.external_mask_file_list[i]
                if external_mask_file and not os.path.exists(external_mask_file):
                    print(f"  Warning: external mask file not found: {external_mask_file}")
                    external_mask_file = None
            
            # Pre-compute mask and distance map ONCE per band (not per aperture)
            distance_map = self.analyzer.calculate_distance_map(
                filter_label,  # Use filter_name for file naming
                self.config.galaxy_catalog,
                self.config.detwht_file,
                wht_file=wht_file,
                kron_scale=self.config.kron_scale,
                output_path=self.config.sky_noise_path,
                external_mask_file=external_mask_file,
                save_mask=self.config.save_mask,
            )
            
            # Now use the same mask/distance map for all apertures
            for aperture in self.config.apertures_list:
                aperture_px = float(aperture / self.config.pixel_scales[i])
                self.analyzer.decide_background_positions_with_distance_map(
                    filter_label,  # Use filter_name for file naming
                    aperture_px,
                    overlap_frac=self.config.overlap_frac,
                    distance_map=distance_map,
                    output_path=self.config.sky_noise_path,
                    save_regions=self.config.save_regions,
                )

            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.1f} seconds")

    def step2_fit_background_noise(self) -> pd.DataFrame:
        """
        Step 2: Fit background noise models for each band.

        Uses single power law model: sigma = sigma_1 * (alpha * r^beta)

        Returns
        -------
        pandas.DataFrame
            Background fit parameters (SIGMA_1, ALPHA, BETA) for each band
        """
        print("\nStep 2: Fitting background noise models...")
        background_results = []

        for i, filter_label in enumerate(self.config.filter_labels):
            print(f"Processing {filter_label} ...")
            start_time = time.time()

            # Also compute sigma_1 for saving
            if self.config.external_mask_file_list is not None:
                external_mask_file = self.config.external_mask_file_list[i]
                if external_mask_file is None:
                    mask_file = f"{self.config.sky_noise_path}{filter_label}_kron_scale{self.config.kron_scale}_integrated_mask.fits"
                else:
                    mask_file = external_mask_file
            mask_data = fits.getdata(mask_file)
            science_data = fits.getdata(self.config.image_list[i])
            background_pixels = science_data[mask_data == 0]
            sigma_1 = 1.4826 * np.median(np.abs(background_pixels - np.median(background_pixels)))

            alpha, beta = self.analyzer.compute_background_noise(
                filter_label,
                sigma_1,
                np.array(self.config.apertures_list),
                self.config.pixel_scales[i],
                self.config.sky_noise_path,
                self.config.image_list[i],
                plot_histograms=self.config.plot_histograms,
            )

            
            # Store all parameters
            background_results.append([filter_label, sigma_1, alpha, beta])
            print(f"  sigma_1={sigma_1:.6f}, alpha={alpha:.6f}, beta={beta:.6f}")

            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.1f} seconds")

        # Save and return background parameters
        bg_df = pd.DataFrame(background_results, columns=["BAND", "SIGMA_1", "ALPHA", "BETA"])

        bg_file = f"{self.config.sky_noise_path}bkg_fit_param.csv"
        bg_df.to_csv(bg_file, index=False)
        print(f"\nBackground parameters saved to {bg_file}")

        return bg_df

    def step3_compute_model_fluxes(self,) -> pd.DataFrame:
        """
        Step 3: Compute model isophotal fluxes using GALFIT.
        curr_idx: int
            the idx of current source, which is 0-indexed.

        Returns
        -------
        pandas.DataFrame
            Model flux catalog with isoflux fractions
        """
        if self.generator is None:
            raise ValueError("galfit_path must be provided in config to compute model fluxes")

        print("\nStep 3: Computing model isophotal fluxes with GALFIT...")

        # Read galaxy catalog
        cat_df = pd.read_csv(self.config.match_galaxy_catalog, sep=r"\s+")
        
        model_flux_df = self._compute_model_fluxes_for_galaxies(
            cat_df["label"].values,
            self.config.filter_labels,
            # self.config.band_dir_list,
            self.config.image_list,
            self.config.gsdir,
            self.config.gmoutdir,
            self.config.outgs_flux_catfile,
        )

        print(f"  Model flux catalog computed with {len(model_flux_df)} galaxies")
        return model_flux_df

    def _compute_model_fluxes_for_galaxies(
        self,
        galids: np.ndarray,
        filter_list: List[str],
        # band_dir_list: List[str],
        image_list: List[str],
        gsdir: str,
        outdir: str,
        output_file: str,
    ) -> pd.DataFrame:
        """Compute model fluxes for all galaxies."""
        df = pd.DataFrame()

        for galid in tqdm(galids, desc="Computing model fluxes"):
            flux_list = {"id": galid}
            
            # Read GALFITS results
            summary_file = f"{gsdir}/obj{galid}.gssummary"
            if not os.path.exists(summary_file):
                print(f"{summary_file} does not exist.")
                return None

            result = ascii.read(summary_file)

            # Try each filter until one works
            for band_idx in self.config.try_filterlist:
                filter_name = filter_list[band_idx]
                # band_dir = band_dir_list[band_idx]

                if f"Mag_obj0_{filter_name}" in result["pname"]:
                    flux_frac = self.generator.get_flux_fraction(
                        gal_id=galid,
                        # input_image=image_list[band_idx],
                        input_image=image_list[0],   ## ref img
                        psf_file=self.config.psf_file,
                        filter=filter_name,
                        # band_dir=band_dir,
                        cutout_dir = self.config.cutout_dir,
                        band_label = self.config.label_list[band_idx],
                        pixel_scale=self.config.pixel_scales[band_idx],
                        zeropoint=self.config.zero_list[band_idx],
                        stamp_file=self.config.stampfile,
                        catalog_name=self.config.match_galaxy_catalog,
                        segmentation_map=self.config.segmentation_map_list[band_idx],
                        gs_dir=gsdir,
                        gmout_dir=outdir,
                        saveconfig=self.config.saveconfig,
                        savemodel=self.config.savemodel,
                    )
                    flux_list[f"flux_frac"] = flux_frac
                
                    break

            

            for filter_name in filter_list:
                if f"Mag_obj0_{filter_name}" not in result["pname"]:
                    flux_list[f"m_{filter_name}"] = -99.0
                else:
                    mag = result["best_value"][result["pname"] == f"Mag_obj0_{filter_name}"][0]
                    flux_list[f"m_{filter_name}"] = mag

            if len(flux_list) > 1:
                df = pd.concat([df, pd.DataFrame([flux_list])], ignore_index=True)

        df.to_csv(output_file, index=False)
        return df

    def step4_compute_isophotal_errors(
        self,
        bg_df: pd.DataFrame,
        flux_df_path: str,
        gs_flux_err_outfile,
    ) -> pd.DataFrame:
        """
        Step 4: Compute isophotal fluxes with error propagation and cross-match with spec-z catalog.

        This is the final step in the photometry pipeline. It computes the total isophotal
        fluxes for all sources in all filters, propagates errors from object noise and
        background variations, adds source metadata (RA, Dec, segment area), and
        cross-matches the catalog with a spectroscopic redshift catalog.

        Parameters
        ----------
        bg_df : pandas.DataFrame
            Background fit parameters from Step 2 (step2_estimate_background).
            Must contain background model parameters for each filter used
            in the error computation.

        Returns
        -------
        pandas.DataFrame
            Final catalog with multi-band photometry and errors. Contains:
            - #id: Source label/ID from detection catalog
            - {filter}_flux: Total isophotal flux in each filter (µJy)
            - {filter}_fluxerr_obj: Object noise error component
            - {filter}_fluxerr_bkg: Background error component
            - {filter}_fluxerr: Total flux error (quadrature sum)
            - ra, dec: Source coordinates (degrees)
            - segment_area: Segmentation area (pixels²)
            - z_spec: Spectroscopic redshift from cross-match (-99.0 if unmatched)

        Notes
        -----
        Processing Steps:
        1. Read source catalog and ISO flux catalog from previous steps
        2. Loop through each filter and compute:
           - Total isophotal flux from GALFIT models
           - Object noise error (source+sky Poisson noise)
           - Background error (from background fit variations)
           - Total error = sqrt(obj_err² + bkg_err²)
        3. Add source metadata (coordinates, segment area)
        4. Cross-match with spectroscopic redshift catalog
        5. Save final catalog to config.output_file

        Error Propagation:
        - Object error: Poisson noise from source and sky background
        - Background error: Uncertainty from background model fitting
        - Total error: Quadrature sum of both components

        The cross-match uses astropy.coordinates.match_coordinates_sky to find
        the nearest spectroscopic redshift within max_sep for each source.

        Examples
        --------
        >>> # Assume pipeline is configured and steps 1-3 are complete
        >>> bg_params = pipeline.step2_estimate_background()
        >>> final_catalog = pipeline.step4_compute_isophotal_errors(
        ...     bg_df=bg_params
        ... )
        >>> print(final_catalog.columns)
        Index(['#id', 'f444w_flux', 'f444w_fluxerr', ..., 'ra', 'dec', 'z_spec'])
        """
        print("\nStep 4: Computing isophotal fluxes with errors...")

        # Read catalogs
        cat_df = pd.read_csv(self.config.match_galaxy_catalog, sep=r"\s+")
        source_ids = cat_df["label"]
        flux_df = pd.read_csv(flux_df_path)
        ref_ids = flux_df["id"]
        output_df = pd.DataFrame({"#id": flux_df["id"]})
        # Process each filter
        for i, filter_label in enumerate(self.config.filter_labels):
            print(f"Processing {filter_label}...")
            start_time = time.time()

            # Compute isophotal fluxes with full error propagation
            total_flux, obj_err, corr_err, tot_err = self.photometry.compute_isophotal_errors(
                cat_df,
                flux_df,
                filter_label,  # Use filter_label for band parameter (to match background_params)
                filter_label,
                bg_df,
                self.config.gains[i],
                self.config.exptimes[i],
                self.config.mjsr_list[i],
                self.config.zero_list[i],
                self.config.pixel_scales[i],
                self.config.ref_pixel_scale,
            )

            # Store results for this filter
            output_df[f"{filter_label}_flux"] = total_flux
            output_df[f"{filter_label}_fluxerr_obj"] = obj_err
            output_df[f"{filter_label}_fluxerr_bkg"] = corr_err
            output_df[f"{filter_label}_fluxerr"] = tot_err

            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.1f} seconds")

        # Add source metadata (coordinates and segment area)
        source_ref_idx = np.array([np.where(source_ids == rid)[0][0] for rid in ref_ids])
        output_df["ra"] = cat_df["ra"][source_ref_idx].values
        output_df["dec"] = cat_df["dec"][source_ref_idx].values
        output_df["segment_area"] = cat_df["segment_area"][source_ref_idx].values
        if self.config.specz_cat is not None:

            # Cross-match with spectroscopic redshift catalog
            print(f"\nCross-matching with spec-z catalog: {self.config.specz_cat}")
            specz_df = pd.read_csv(self.config.specz_cat)
            output_df = corss_match_df(output_df, specz_df, max_sep=self.config.max_sep)

        # Save final catalog
        output_df.to_csv(gs_flux_err_outfile, index=False)
        print(f"\nFinal catalog saved to {gs_flux_err_outfile}")

        return output_df

