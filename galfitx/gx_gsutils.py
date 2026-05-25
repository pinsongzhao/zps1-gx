"""
Identifier:     galfitx/gx_gsutils.py
Name:           gx_gsutils.py
Description:    some functions for preparing galfits
Author:         Chao Ma
Created:        2026-01-19
Modified-History:
    2026-01-19, Chao Ma, created
"""

import numpy as np
from astropy.io import fits
from typing import Union, List
from typing import List, Optional, Union, Literal, Any, Tuple
import os 
# copied from galfits.
effective_wave = {
    # flux is Flambda
    # counts_rate/flux for galex
    "galex_fuv": 1538.6,
    "galex_nuv": 2315.7,
    # counts_rate/flux for panstarrs
    "panstarrs_g": 4810,
    "panstarrs_r": 6170,
    "panstarrs_i": 7520,
    "panstarrs_z": 8660,
    "panstarrs_y": 9620,
    # nanomaggies/flux for SDSS
    "sloan_u": 3543,
    "sloan_g": 4770,
    "sloan_r": 6231,
    "sloan_i": 7625,
    "sloan_z": 9134,
    # nanomaggies/flux for DESI
    "desi_s_g": 4829,
    "desi_s_r": 6436,
    "desi_s_z": 9178,
    "desi_n_g": 4776,
    "desi_n_r": 6412,
    "desi_n_z": 9264,
    "desi_i": 7840,
    # CGS survey
    "cgs_u": 3551.05,
    "cgs_b": 4369.53,
    "cgs_v": 5467.57,
    "cgs_r": 6357.35,
    "cgs_i": 7828.65,
    # 2MASS use dN
    "j": 12350,
    "h": 16608,
    "ks": 21590,
    # ALLWISE image, counts rate units
    "wise_ch1": 34000,
    "wise_ch2": 46000,
    "wise_ch3": 120000,
    "wise_ch4": 220000,
    # Subaru-HSC
    "hsc_g": 4816.12,
    "hsc_r": 6234.11,
    "hsc_i": 7740.58,
    "hsc_z": 9125.20,
    "hsc_y": 9779.93,
    # DECam
    "decam_u": 3550,
    "decam_g": 4730,
    "decam_r": 6420,
    "decam_i": 7840,
    "decam_z": 9260,
    "decam_y": 10090,
    # Subaru FOCAS
    "focas_U": 3600.0,
    "focas_B": 4400.0,
    "focas_V": 5500.0,
    "focas_R": 6600.0,
    "focas_I": 8050.0,
    # JWST Wen's image
    "nircam_f070w": 7043.721411041347,
    "nircam_f090w": 8984.98,
    "nircam_f115w": 11542.61,
    "nircam_f140m": 14054.613516500445,
    "nircam_f150w": 15007.44,
    "nircam_f162m": 16271.463297896555,
    "nircam_f182m": 18452.27687805039,
    "nircam_f200w": 19886.48,
    "nircam_f210m": 20956.41149309253,
    "nircam_f250m": 25032.646167918316,
    "nircam_f277w": 27617.40,
    "nircam_f300m": 29961.4686458197,
    "nircam_f335m": 33623.44734205056,
    "nircam_f356w": 35683.62,
    "nircam_f360m": 36233.126420440145,
    "nircam_f410m": 40822.38,
    "nircam_f444w": 44043.15,
    "miri_f770w": 76604.7,
    "miri_f1000w": 99620.625,
    "miri_f1280w": 128313.970,
    "miri_f1500w": 150760.77,
    "miri_f1800w": 179922.53,
    "miri_f2100w": 208425.266,
    # HST
    "acs_f435w": 4317.61,
    "acs_f606w": 5809.26,
    "acs_f814w": 7973.39,
    "wfc3_f438w": 4327.16,
    "wfc3_f475w": 4777.38,
    "wfc3_f547m": 5471.32,
    "wfc3_f555w": 5305.85,
    "wfc3_f606w": 5921.97,
    "wfc3_f814w": 8024.22,
    "wfc3_f105w": 10430.83,
    "wfc3_f110w": 11534.52,
    "wfc3_f125w": 12363.55,
    "wfc3_f140w": 13734.66,
    "wfc3_f160w": 15278.47,
    # swift UVOT
    "swift_u": 3465.00,
    "swift_b": 4392.00,
    "swift_v": 5468.00,
    "swift_w1": 2600.00,
    "swift_m2": 2246.00,
    "swift_w2": 1928.00,
    # XMM-OM
    "xmmom_u": 3440.00,
    "xmmom_b": 4500.00,
    "xmmom_v": 5500.00,
    "xmmom_m2": 2310.00,
    "xmmom_w1": 2910.00,
    "xmmom_w2": 2120.00,
    # Spitzer-IRAC
    "ch1": 36000.0,
    "ch2": 45000.0,
    "ch3": 58000.0,
    "ch4": 80000.0,
    # Keck-LRIS
    "Un_lris_B": 3450.0,
    "B_lris": 4370.0,
    "g_lris": 4731.0,
    "V_lris": 5437.0,
    "B_lris_B": 4377.0,
    "V_lris_B": 5473.0,
    "R_lris": 6417.0,
    "I_lris": 7599.0,
    "Rs_lris": 6809.0,
    # Keck-MOSFIRE
    "mosfire_Y": 10480.0,
    "mosfire_J": 12530.0,
    "mosfire_H": 16370.0,
    "mosfire_Ks": 21470.0,
    # CSST
    "csst_nuv": 2867.7,
    "csst_u": 3601.1,
    "csst_g": 4754.5,
    "csst_r": 6199.8,
    "csst_i": 7653.2,
    "csst_z": 9600.6,
    "csst_y": 10051.0
}

def Fnu_to_Fl(Fnu: Union[float, np.ndarray], lambd: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert flux density from mJy to erg/s/cm²/A.

    Parameters
    ----------
    Fnu : float or np.ndarray
        Flux density in millijanskys (mJy). 1 mJy = 10⁻²⁶ erg/s/cm²/Hz.
    lambd : float or np.ndarray
        Wavelength in Angstroms (Å).

    Returns
    -------
    float or np.ndarray
        Flux density in erg/s/cm²/Å.

    """
    c = 2.9979246e5  # speed of light in km/s
    mJy = 1e-26  # 1 mJy = 10⁻²⁶ erg/s/cm²/Hz
    a = Fnu * mJy
    Fl = a * (c * 1e13) / (lambd**2)
    return Fl


def ABmag_to_covf(mzp_AB: Union[float, np.ndarray], wave: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert AB magnitude to a conversion factor (1 / flux density in erg/s/cm²/Å).

    The conversion factor is defined as the inverse of the flux density (in
    erg/s/cm²/Å) corresponding to the given AB magnitude. This is often used
    to transform counts or other instrumental units to physical flux units.

    Steps:
    1. Convert AB magnitude to flux density in mJy:
    flux_mJy = 10^(-0.4 * mzp_AB) * 3631 * 1000 (3631 Jy is the AB reference flux, multiplied by 1000 to get mJy)
    2. Convert mJy to erg/s/cm²/Å using `Fnu_to_Fl`.
    3. Return the reciprocal of that flux density.

    Parameters
    ----------
    mzp_AB : float or np.ndarray
        AB magnitude(s) (dimensionless). Can be a scalar or an array.
    wave : float or np.ndarray
        Wavelength(s) in Angstroms (Å). Should be broadcastable with `mzp_AB`.

    Returns
    -------
    float or np.ndarray
        Conversion factor(s) with units of (erg/s/cm²/Å)⁻¹. The output has the
        same shape as the broadcasted inputs.
    """

    # Convert AB magnitude to flux density in millijanskys (mJy)
    # 1 Jy = 10⁻²³ erg/s/cm²/Hz, so 1 mJy = 10⁻²⁶ erg/s/cm²/Hz
    flux_mJy = 10 ** (-0.4 * (mzp_AB)) * 3631 * 1000

    # Convert mJy to erg/s/cm²/Å using the previously defined function
    flux_fl_fuv = Fnu_to_Fl(flux_mJy, wave)

    return 1 / flux_fl_fuv


def calconvfactor(
    zpab_list: List[float],  # zeropoint of all input filters
    wave_list: List[float],  # in A
    image_stamp_list: Union[str, List[str]] = "none",
    filter_list: Union[str, List[str]] = "none",
) -> np.ndarray:
    """
    Calculate conversion factors to transform instrumental counts to physical flux
    (in erg/s/cm²/Å) for a set of filters.

    The conversion factor for each filter is obtained via `ABmag_to_covf`, which
    returns 1 / (flux density in erg/s/cm²/Å) for a given AB magnitude zeropoint
    and wavelength. This factor can be multiplied by counts to yield flux.

    Parameters
    ----------
    zpab_list : list of float
        AB magnitude zeropoints for each filter.
    wave_list : list of float
        Central wavelengths [Å] corresponding to each filter.
    image_stamp_list : str or list of str, optional
        Paths to image stamp files (currently unused in the active code).
    filter_list : str or list of str, optional
        galfits label

    Returns
    -------
    np.ndarray
        1D array of conversion factors (units: (erg/s/cm²/Å)⁻¹) for each filter.

    Notes
    -----
    The active part of the function simply iterates over `zpab_list` and `wave_list`,
    calling `ABmag_to_covf` for each pair. The commented‑out section shows a more
    elaborate version that would read image headers to derive zeropoints or
    PHOTFLAM values for specific instruments; it is kept as a reference for
    future extension.
    """

    confl = []

    for nn in range(len(zpab_list)):
        a = ABmag_to_covf(zpab_list[nn], wave_list[nn])
        confl.append(a)

    # for sci,key in zip(image_stamp_list,filter_list):
    #     if ('nircam' in key) or ('miri' in key):
    #         img = fits.open(sci)
    #         header = img[0].header
    #         ZP_GALFIT = 2.5*np.log10(3631/(header['PIXAR_SR']*1e6))
    #         magab = ZP_GALFIT
    #         a = ABmag_to_covf(magab,effective_wave[key])
    #         confl.append(a)
    #     elif ('acs' in key) or ('wfc3' in key):
    #         img = fits.open(sci)
    #         header = img[0].header
    #         a = header['PHOTFLAM']
    #         confl.append(1/a)
    #     else:## for other filters, how to derive conversion factor?

    #         a = ABmag_to_covf(magab,effective_wave[key])
    #         confl.append(a)

    # confl.append(0)

    confl = np.array(confl)
    return confl

def Union_Set(objects: List):
    '''
    construct the groups from objects.

    Return:
    groups: List[List]

    '''
    parent = {}

    def find(x):
        '''
        find the parent point of x.
        '''
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x,y):
        '''
        union the tree of x and y, if they haven't in the same tree.
        '''
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    ## inital the forest
    for key, values in objects.items():
        for v in values:
            union(key, v)

    group_dict = {}

    for key in objects:

        root = find(key)
        group_dict.setdefault(root, set()).add(int(key))

        for v in objects[key]:
            group_dict.setdefault(root, set()).add(int(v))

    groups = [g for g in group_dict.values()]
    groups = sorted(groups, key = lambda x:(len(x), x), reverse=True)

    return groups


def stackimg(imglist: List[str],        # List of science image paths
             siglist: List[str],        # List of sigma/noise image paths
             whtlist: List[str],        # List of weight map paths
             mode: Literal["whtm", "snr"] = "whtm",
             savepath: List[str] = ['./detection.fits', 'detection_wht.fits'],
             saveresult: bool = True) -> List[Optional[np.ndarray]]:
    """
    Accelerated & robust multi-band image stacking.
    - whtm: Inverse-variance (noise-equalized) weighted stack (standard for astronomy)
    - snr : Stack of SNR images (simple average)
    
    Returns: [stacked_science, stacked_weight (or None)]
    """
    # --------------------------
    # Speed: Read template once
    # --------------------------
    template_data = fits.getdata(imglist[0])
    img_shape = template_data.shape
    dtype = np.float32  # Use float32 to save memory & speed up
    
    # --------------------------
    # Initialize arrays (fast)
    # --------------------------
    if mode == "whtm":
        sumsci = np.zeros(img_shape, dtype=dtype)
        sumwht = np.zeros(img_shape, dtype=dtype)
    else:  # snr
        sumsnr = np.zeros(img_shape, dtype=dtype)
    
    # Get header once
    hdr = fits.getheader(imglist[0])

    # --------------------------
    # Fast loop with no repeated I/O
    # --------------------------
    n_bands = len(imglist)
    for jj in range(n_bands):
        # Read data quickly
        sci = fits.getdata(imglist[jj]).astype(dtype)
        
        # Shape check (critical for bug prevention)
        if sci.shape != img_shape:
            raise ValueError(f"Shape mismatch: {imglist[jj]}")

        if mode == "whtm":
            wht = fits.getdata(whtlist[jj]).astype(dtype)
            wht = np.nan_to_num(wht, nan=0.0, posinf=0.0, neginf=0.0)  # Clean bad pixels
            sumsci += sci * wht
            sumwht += wht

        else:  # SNR mode
            sig = fits.getdata(siglist[jj]).astype(dtype)
            with np.errstate(divide='ignore', invalid='ignore'):
                snr = sci / sig
            snr = np.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)
            sumsnr += snr

    # --------------------------
    # Compute final stack
    # --------------------------
    if mode == "whtm":
        # Noise-equalized stack (core formula)
        with np.errstate(divide='ignore', invalid='ignore'):
            stack = sumsci / sumwht
        
        # Replace invalid pixels (0 weight) with 0
        stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
        whtstack = sumwht  # CORRECT total weight (your old /N was wrong!)

    else:  # SNR mode
        stack = sumsnr / np.sqrt(n_bands)
        stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
        whtstack = None     ### in this mode, the det image is already weighted by noise image. 

    # --------------------------
    # Save output
    # --------------------------
    if saveresult:
        fits.writeto(savepath[0], stack, hdr, overwrite=True)
        if mode == "whtm" and whtstack is not None:
            fits.writeto(savepath[1], whtstack, hdr, overwrite=True)

    return [stack, whtstack]               
               
