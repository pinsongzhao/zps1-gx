"""
Identifier:     galfitx/create_setup_gs.py
Name:           create_setup_gs.py
Description:    generate galfits config file
Author:         Chao Ma
Created:        2026-01-19
Modified-History:
    2026-01-19, Chao Ma, created
"""
import os
import numpy as np
import string
from astropy.io import fits, ascii
import tqdm
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from galfitx.read_sersic_results import read_sersic_results
import random
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.cosmology import FlatLambdaCDM
from reproject import reproject_interp
from typing import List, Tuple, Union, Optional, Any
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from dustmaps.sfd import SFDQuery
from astroquery.ipac.irsa.irsa_dust import IrsaDust
import matplotlib.pyplot as plt
try:
    sfd = SFDQuery()
    USE_DUSTMAPS = True
except (ImportError, FileNotFoundError, OSError):
    sfd = None
    USE_DUSTMAPS = False
    print("Warning: dustmaps not available, will use astroquery as fallback")


cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

effective_wave ={
    # flux is Flambda
    # counts_rate/flux for galex
    'galex_fuv' : 1538.6, 
    'galex_nuv' : 2315.7,
    # counts_rate/flux for panstarrs 
    'panstarrs_g' : 4810,  
    'panstarrs_r' : 6170,
    'panstarrs_i' : 7520,
    'panstarrs_z' : 8660,
    'panstarrs_y' : 9620,
    # nanomaggies/flux for SDSS
    'sloan_u' : 3543,
    'sloan_g' : 4770,
    'sloan_r' : 6231, 
    'sloan_i' : 7625,
    'sloan_z' : 9134,
    # nanomaggies/flux for DESI
    'desi_s_g' : 4829,
    'desi_s_r' : 6436,
    'desi_s_z' : 9178,
    'desi_n_g' : 4776,
    'desi_n_r' : 6412,
    'desi_n_z' : 9264,
    'desi_i': 7840,
    # CGS survey
    'cgs_u' : 3551.05,
    'cgs_b' : 4369.53,
    'cgs_v' : 5467.57,
    'cgs_r' : 6357.35,
    'cgs_i' : 7828.65,
    # 2MASS use dN  
    'j' : 12350,
    'h' : 16608, 
    'ks' : 21590,
    # ALLWISE image, counts rate units
    'wise_ch1' : 34000,
    'wise_ch2' : 46000,
    'wise_ch3' : 120000,
    'wise_ch4' : 220000,
    # Subaru-HSC
    'hsc_g' : 4816.12,
    'hsc_r' : 6234.11,
    'hsc_i' : 7740.58,
    'hsc_z' : 9125.20,
    'hsc_y' : 9779.93,
    # DECam
    'decam_u' : 3550,
    'decam_g' : 4730,
    'decam_r' : 6420,
    'decam_i' : 7840,
    'decam_z' : 9260,
    'decam_y' : 10090,
    # Subaru FOCAS
    'focas_U' : 3600.,
    'focas_B' : 4400.,
    'focas_V' : 5500.,
    'focas_R' : 6600.,
    'focas_I' : 8050.,
    # JWST 
    'nircam_f070w': 7043.721411041347,
    'nircam_f090w': 8984.98,
    'nircam_f115w': 11542.61,  
    'nircam_f140m': 14054.613516500445,
    'nircam_f150w': 15007.44,
    'nircam_f162m': 16271.463297896555,
    'nircam_f182m': 18452.27687805039,
    'nircam_f200w': 19886.48,
    'nircam_f210m': 20956.41149309253,
    'nircam_f250m': 25032.646167918316,
    'nircam_f277w': 27617.40,
    'nircam_f300m': 29961.4686458197,
    'nircam_f335m': 33623.44734205056,
    'nircam_f356w': 35683.62,
    'nircam_f360m': 36233.126420440145,
    'nircam_f410m': 40822.38,
    'nircam_f444w': 44043.15,
    'miri_f770w': 76604.7,
    'miri_f1000w': 99620.625,
    'miri_f1500w': 150760.77,
    'miri_f1800w': 179922.53,
    # HST
    'acs_f435w':4317.61,
    'acs_f606w':5809.26,
    'acs_f814w':7973.39,
    'wfc3_f225w': 2380.02,
    'wfc3_f336w':3358.18,
    'wfc3_f438w':4327.16,
    'wfc3_f475w':4777.38,
    'wfc3_f547m':5471.32,
    'wfc3_f555w':5305.85,
    'wfc3_f606w':5921.97,
    'wfc3_f814w':8024.22,
    'wfc3_f105w':10430.83,
    'wfc3_f110w':11534.52,
    'wfc3_f125w':12363.55,
    'wfc3_f140w':13734.66,
    'wfc3_f160w':15278.47,
    # swift UVOT
    'swift_u':3465.00,
    'swift_b':4392.00,
    'swift_v':5468.00,
    'swift_w1':2600.00,
    'swift_m2':2246.00,
    'swift_w2':1928.00,
    # XMM-OM
    'xmmom_u':3440.00,
    'xmmom_b':4500.00,
    'xmmom_v':5500.00,
    'xmmom_m2':2310.00,
    'xmmom_w1':2910.00,
    'xmmom_w2':2120.00,
    # Spitzer-IRAC
    'ch1':36000.,
    'ch2':45000.,
    'ch3':58000.,
    'ch4':80000.,
    # Keck-LRIS
    'Un_lris_B':3450.,
    'B_lris':4370.,
    'g_lris':4731.,
    'V_lris':5437.,
    'B_lris_B':4377.,
    'V_lris_B':5473.,
    'R_lris':6417.,
    'I_lris':7599.,
    'Rs_lris':6809.,
    # Keck-MOSFIRE
    'mosfire_Y':10480.,
    'mosfire_J':12530.,
    'mosfire_H':16370.,
    'mosfire_Ks':21470.,
    # Herschel
    'herschel_psw': 2516960.,
    'herschel_pmw': 3524879.,
    'herschel_plw': 5116700.,
    # CSST
    'csst_nuv': 2867.7, 
    'csst_u': 3601.1,
    'csst_g': 4754.5, 
    'csst_r': 6199.8,
    'csst_i': 7653.2, 
    'csst_z': 9600.6, 
    'csst_y': 10051.0,
}

def gal_ebv(ra, dec):
    coord = SkyCoord(ra, dec, unit='deg')
    if USE_DUSTMAPS and sfd is not None:
        ebv = sfd(coord)
        return float(ebv)
    else:
        result = IrsaDust.get_query_table(coord)
        ebv = result['ext SandF mean'][0]  # SFD E(B-V) 值
    return float(ebv)


def reproject_segm(segname: str, sciname: str, output: Optional[str] = None, type:str = None) -> int:
    """
    Reproject a segmentation image to match the world coordinate system of a science image.

    The function uses `reproject_interp` with nearest‑neighbor interpolation to preserve
    the integer label values of the segmentation map. After reprojection, any pixel where
    the science image data is zero is set to zero in the output segmentation (useful for
    masking regions with no coverage). The resulting image can be saved to a user‑specified
    file or to a default name (`<sciname>_seg.fits`).

    Parameters
    ----------
    segname : str
        Path to the input FITS file containing the segmentation image.
    sciname : str
        Path to the science FITS image whose header defines the target WCS.
    output : str, optional
        Path to the output FITS file. If None, the output is written to
        `sciname.replace('.fits', '_seg.fits')`. Default is None.

    Returns
    -------
    int
        Always returns 1 to indicate successful completion (convention used in some scripts).

    Notes
    -----
    - The reprojection is performed with `order='nearest-neighbor'` to avoid introducing
      fractional values in the segmentation map.
    - Pixels where the science image has value 0 are forced to 0 in the output segmentation,
      effectively masking those regions.
    """

    hduseg = fits.open(segname)[0]
    hdusci = fits.open(sciname)[0]
    outseg, _ = reproject_interp(hduseg, hdusci.header, order="nearest-neighbor")

    if type == "mask":
        outseg[(hdusci.data == 0) | (np.isnan(hdusci.data))] = 1

    if output is not None:
        fits.writeto(output, outseg, hdusci.header, overwrite=True)
    else:
        fits.writeto(sciname.replace(".fits", "_seg.fits"), outseg, hdusci.header, overwrite=True)

    return 1


def process_psf(
    id: int,
    nband: int,
    psf_file_list: List[str],
    image_file_list: List[str],
    label_list: List[str],
    processed_psf_dir: list, 

) -> np.ndarray:
    """
    Prepare PSF files for each band, cropping them if necessary to fit the image size.

    For each band, this function checks whether the PSF image is larger than the
    corresponding science image. If the PSF dimensions exceed the image dimensions,
    a central square crop of odd size (≤ min(image side)) is applied and the cropped
    PSF is saved to a new file. The path(s) of the (possibly cropped) PSF(s) are
    returned in an array.

    Parameters
    ----------
    id : int
        Object identifier, used in the output filename when cropping is needed.
    nband : int
        Number of bands (should match the length of the three input lists).
    psf_file_list : list of str
        List of paths to the original PSF FITS files, one per band.
    image_file_list : list of str
        List of paths to the science image FITS files, one per band.
    label_list : list of str
        List of band labels (e.g., 'F090W', 'F150W') used in the output filename.
    use_sed : int, optional
        If 0, cropped PSFs are saved under "./galfits/"; if non‑zero, under
        "./galfits_sed/". Default is 0.

    Returns
    -------
    psf_file_used : `~numpy.ndarray` of str
        Array of paths to the PSF files that should be used for each band.
        For bands where cropping was not needed, the original path is kept;
        otherwise, the path points to the newly created cropped PSF.

    Notes
    -----
    - The cropping logic ensures that the cropped PSF has odd dimensions (to preserve
      a central pixel) and that its size does not exceed the smallest side of the
      corresponding science image.
    - The output directory ("./galfits/" or "./galfits_sed/") must exist; otherwise,
      `fits.writeto` will raise an error.
    """

    psf_file_used = []
    for b in range(nband):

        header = fits.open(image_file_list[b])[0].header
        xmax = header["NAXIS1"]
        ymax = header["NAXIS2"]

        # add condition check to ensure convolution works
        psf_size = fits.open(psf_file_list[b])[0].header["NAXIS1"]

        if (min(xmax, ymax) > psf_size) or (max(xmax, ymax) < psf_size):
            psf_file_used.append(psf_file_list[b])
        else:
            # i.e.,mixed dimensions, crop the psf size to below min(xmax,ymax)
            min_side = min(xmax, ymax)
            # Calculate largest odd size <= min_side
            new_size = min_side if min_side % 2 == 1 else min_side - 1
            start = (psf_size - new_size) // 2
            end = start + new_size
            psf = fits.open(psf_file_list[b])[0].data
            psf_used = psf[start:end, start:end]
            psf_path = os.path.join(processed_psf_dir,f"obj{id}_{label_list[b]}_psf.fits")
            fits.writeto(psf_path, psf_used, overwrite=True)
            psf_file_used.append(psf_path)
    psf_file_used = np.array(psf_file_used)

    return psf_file_used



def create_mask(
    scihdu: str,
    seg_data: str,
    cover_data: str, 
    catalog_name: str,
    paramfile: str,
    mask_file: str,
    current: int,
    scale: float = 1.1,
    offset: float = 4.0,
    limgal: float = 3.0,
    ps: float = 0.03,
) -> Tuple[int, int, List[int]]:
    """
    Create a mask for the current object (primary) based on SExtractor catalogs.

    The mask is constructed for a postage stamp cutout defined by a stamp file
    (`paramfile`). It identifies pixels belonging to:
    - the primary source,
    - secondary sources (those overlapping with the primary),
    - tertiary sources (non‑overlapping),
    - bad pixels (from weight image) and
    - other segmentation regions.
    The final mask is a binary image (0 = good, 1 = masked) saved to a FITS file.
    """

    
    wcs = WCS(scihdu)
    wht = scihdu.data
    coverage_mask = cover_data
    header = scihdu.header
    ny_full, nx_full = wht.shape

    seg = seg_data
    cat = ascii.read(catalog_name)
    paramdata = ascii.read(
        paramfile,
        names=["pnum", "px", "py", "pra", "pdec", "pxlo", "pxhi", "pylo", "pyhi", "ps"],
    )

    idx = np.where(paramdata["pnum"] == cat[current]["label"])
    if idx[0].size == 0:
        raise ValueError(f"no postage stamp definition found for current object{cat[current]['label']}")

    idx = int(idx[0][0])  # this is for multiple version of numpy
    pnum = paramdata["pnum"][idx]
    px = float(paramdata["px"][idx])
    py = float(paramdata["py"][idx])
    pra = float(paramdata["pra"][idx])
    pdec = float(paramdata["pdec"][idx])
    xlo = float(paramdata["pxlo"][idx])
    xhi = float(paramdata["pxhi"][idx])
    ylo = float(paramdata["pylo"][idx])
    yhi = float(paramdata["pyhi"][idx])
    pscale = float(paramdata["ps"][idx])

    # x0, y0 are in unit of image
    x0, y0 = wcs.all_world2pix(pra, pdec, 1)

    xlo1 = int(round(x0 - (px - xlo) * pscale / ps)) - 1  # convert from original pixel scale.
    xhi1 = int(round((xhi - px) * pscale / ps + x0)) - 1  # convert from original pixel scale.
    ylo1 = int(round(y0 - (py - ylo) * pscale / ps)) - 1  # convert from original pixel scale.
    yhi1 = int(round((yhi - py) * pscale / ps + y0)) - 1  # convert from original pixel scale.

    pxlo = max(xlo1, 0)
    pxhi = min(xhi1, nx_full - 1)
    pylo = max(ylo1, 0)
    pyhi = min(yhi1, ny_full - 1)

    # pixel, pscale is for the detection band, ps can vary from different filter.
    rad = (scale * cat["semimajor_sigma"] * cat["kron_radius"] + offset) * pscale / ps

    mask = wht[pylo: pyhi + 1, pxlo: pxhi + 1]
    segm = seg[pylo: pyhi + 1, pxlo: pxhi + 1]
    coverm = coverage_mask[pylo: pyhi + 1, pxlo: pxhi + 1]
    badpix = np.where(mask == 0)
    mask *= 0
    ny, nx = mask.shape

    # objects for the GALFIT start file (primary source must be included)
    objects = [current]
    ntab = len(cat["label"])

    # need modfiy this for images with different coordinate systems
    theta = cat["orientation"] * np.pi / 180  # convert degrees to radian

    # calculate the radius array for the primary source
    arr_current = dist_ellipse(
        [pxhi - pxlo + 1, pyhi - pylo + 1],
        x0 + 1 - pxlo,
        y0 + 1 - pylo,
        1.0 / (1.0 - cat[current]["ellipticity"]),
        theta[current] * 180 / np.pi - 90,
    )

    # Precompute pixel coordinates for all catalog objects once, then restrict
    # the expensive per-object processing to sources whose centers fall within
    # a conservatively expanded region around the current postage stamp.
    ra_all = np.asarray(cat["ra"])
    dec_all = np.asarray(cat["dec"])
    xi_all, yi_all = wcs.all_world2pix(ra_all, dec_all, 1)

    # Conservative padding: any object farther away than this by center cannot
    # contribute pixels to the stamp through the later ellipse-box construction.
    # Use the largest source radius plus the enforced minimum half-box size.
    search_pad = max(float(np.max(rad)), 10.0)

    candidate_mask = (
        (xi_all >= (pxlo - search_pad)) &
        (xi_all <= (pxhi + search_pad)) &
        (yi_all >= (pylo - search_pad)) &
        (yi_all <= (pyhi + search_pad))
    )
    candidate_indices = np.where(candidate_mask)[0]
    # print(current,len(candidate_indices))
    # loop over nearby sources in the catalog.
    for i in candidate_indices:
        if i == current:
            continue

        xi = xi_all[i]
        yi = yi_all[i]

        # calculate the angle between the loop sources and the [current] object
        dx = x0 + 1 - xi
        dy = y0 + 1 - yi
        if dx == 0:
            angle = np.pi / 2.0
        else:
            angle = np.arctan(dy / dx)

        # calculate the angle pointing from the loop source to the [current] object
        # i.e,from the perspective of the _i_th source, how much you must rotate its own axis to point toward the current object.
        angle1 = angle - theta[i]

        # calculate the angle pointing from the [current] object to the loop source
        # from the perspective of the current object, how much you must rotate its image‐frame axis to point back at the loop source.
        angle2 = angle - theta[current]

        # calculate the extent of the loop source towards the [current] object
        r1 = np.sqrt(
            1.0
            / (np.sin(angle1) ** 2 / (rad[i] * (1 - cat[i]["ellipticity"])) ** 2 + np.cos(angle1) ** 2 / rad[i] ** 2)
        )

        # calculate the extent of the [current] source towards the loop object
        r2 = np.sqrt(
            1.0
            / (
                np.sin(angle2) ** 2 / (rad[current] * (1 - cat[current]["ellipticity"])) ** 2
                + np.cos(angle2) ** 2 / rad[current] ** 2
            )
        )

        # calculate the distance between the two objects
        d = np.sqrt(dx**2 + dy**2)

        xfac = rad[i] * (abs(np.sin(theta[i])) + (1 - cat[i]["ellipticity"]) * abs(np.cos(theta[i])))
        yfac = rad[i] * (abs(np.cos(theta[i])) + (1 - cat[i]["ellipticity"]) * abs(np.sin(theta[i])))
        # the size along two axises after rotation.

        xfac = max(xfac, 10)
        yfac = max(yfac, 10)

        major = max([xfac, yfac])
        minor = min([xfac, yfac])

        ang = theta[i] * 180 / np.pi  # convert radian to degrees
        ang %= 360  # contraint to [0,360)
        if ang > 180:
            ang -= 180
        if ang > 90:
            ang -= 180
        if abs(ang) < 45:
            xfac = major
            yfac = minor
        else:
            xfac = minor
            yfac = major

        xlo = min(max(round(xi - xfac), 0), nx_full - 1)
        xhi = min(max(round(xi + xfac), 0), nx_full - 1)
        ylo = min(max(round(yi - yfac), 0), ny_full - 1)
        yhi = min(max(round(yi + yfac), 0), ny_full - 1)
        # xlo,ylo,xhi,yhi are on the system of the input image, but not
        # on the postage stamp.
        xlo = min(max(xlo - pxlo, 0), nx - 1)
        xhi = min(max(xhi - pxlo, 0), nx - 1)
        ylo = min(max(ylo - pylo, 0), ny - 1)
        yhi = min(max(yhi - pylo, 0), ny - 1)

        if np.sum([xhi - xlo + 1, yhi - ylo + 1]) > 2:
            arr = dist_ellipse(
                [xhi - xlo + 1, yhi - ylo + 1],
                xi - pxlo - xlo,
                yi - pylo - ylo,
                1.0 / (1.0 - cat[i]["ellipticity"]),
                theta[i] * 180 / np.pi - 90,
            )
        else:
            arr = 1e30

        small_mask = mask[ylo: yhi + 1, xlo: xhi + 1]
        faintlim = limgal  # has omit the star

        if (r1 + r2 > d) and (cat[i]["mag_auto"] < (cat[current]["mag_auto"] + faintlim)):
            # loop source has overlap with current --> secondary
            idx = np.where((arr <= rad[i]) & ((small_mask % 2) == 0))
            if idx[0].size > 0:
                small_mask[idx] += 1
            objects.append(i)
        else:
            # loop source has NO overlap with current --> tertiary
            plus = 2
            idx = np.where((arr <= rad[i]) & (small_mask < 2))
            if idx[0].size > 0:
                small_mask[idx] += plus

        if (r1 + r2 > d) and (cat[i]["mag_auto"] >= (cat[current]["mag_auto"] + faintlim)):
            # loop source has overlap with current --> secondary
            idx = np.where(
                (arr <= rad[i]) & ((small_mask % 2) == 0) & (arr_current[ylo: yhi + 1, xlo: xhi + 1] <= rad[current])
            )
            if idx[0].size > 0:
                small_mask[idx] = 0

        mask[ylo: yhi + 1, xlo: xhi + 1] = small_mask

    # set secondaries to 0
    idx = np.where((mask == 1) | (mask == 3))
    if idx[0].size > 0:
        mask[idx] = 0

    # update the header
    stamp_header = header.copy()  # avoid mutating the original
    stamp_header["CRPIX1"] -= pxlo  # Shift reference pixel
    stamp_header["CRPIX2"] -= pylo

    # set rest to 1
    mask = np.minimum(mask, 1)

    # pixels without weight in the wht image will be masked
    if badpix[0].size > 0:
        mask[badpix] = 1

    # any pixel in the segmentation map not from the primary or secondary
    # will be masked
    mask1 = np.minimum(segm, 1)
    for i in range(len(objects)):
        idx = np.where((segm == cat[objects[i]]["label"]) & (segm > 0))
        if idx[0].size > 0:
            mask1[idx] = 0

    mask = np.minimum(mask + mask1, 1)
    mask[coverm > 0] = 1
    fits.writeto(mask_file, mask, header=stamp_header, overwrite=True)

    corner = [pxlo, pylo]
    return corner, objects




def create_sigma(
    image_stamp_list: Union[str, List[str]] = "none",
    mask_file_list: Union[str, List[str]] = "none",
    filter_list: Union[str, List[str]] = "none",
    label_list: Union[str, List[str]] = "none",
    expt_list: Union[float, List[float]] = 1,
    gain_list: Union[str, List[str]] = "none",
    mjsr_list: Union[str, List[str]] = "none",
    outpath: str = "./",
) -> None:
    """
    Create sigma (noise) maps for input science cutouts and prepare science images for GALFIT.

    For each input science image (cutout), this function:
        1. Converts the image data to electrons using gain, MJy/sr, and exposure time.
        2. Replaces NaN pixels with 0 and saves the cleaned image as `*_sci.fits`.
        3. Computes a sigma map as the quadrature sum of Poisson noise (from the electron
           image) and the sky background standard deviation (estimated from the original image).
        4. Writes the sigma map to `*_sigma.fits`.

    The gain and MJy/sr values are looked up in hard‑coded dictionaries based on the
    instrument/filter name provided in `filter_list`.

    Parameters
    ----------
    image_stamp_list : str or list of str, optional
        Path(s) to the input cutout FITS images. If a single string, it is treated as
        a list of one element. Default "none".
    mask_file_list : str or list of str, optional
        Path(s) to mask files (unused in current implementation). Default "none".
    filter_list : str or list of str, optional
        Instrument/filter identifier(s) used to look up gain and MJy/sr in the internal
        dictionaries. Must match keys in `gainl` and `mjsrl`. Default "none".
    label_list : str or list of str, optional
        Band label(s) used to construct output filenames. Default "none".
    expt_list : float or list of float, optional
        Exposure time(s) in seconds for each image. If a single float, it is applied to
        all images. Default 1.
    gain : str, optional
        Unused parameter (kept for compatibility). Default "none".
    mjsr : str, optional
        Unused parameter (kept for compatibility). Default "none".
    outpath : str, optional
        Output directory for the generated files. Default "./".

    Returns
    -------
    None
        The function writes FITS files to disk and does not return any value.

    Notes
    -----
    - The internal dictionaries `gainl` and `mjsrl` contain values for various HST and JWST
      instruments/filters. They are defined at module level.
    - The effective gain used for electron conversion is: `gain_eff = gain / mjsr * expt`.
    - Sigma maps are computed as `sqrt( (sqrt(electrons)/gain_eff)^2 + sky_std^2 )`,
      with a lower bound of 1e20 where the result would be zero.
    """


    for sci, key, band, expt, gain, mjsr in zip(image_stamp_list, filter_list, label_list, expt_list, gain_list, mjsr_list):

        gain_eff = gain / mjsr * expt

        basename = os.path.basename(sci)
        img = fits.open(sci)[0].data
        imgheader = fits.open(sci)[0].header

        argnan = np.where(np.isnan(img))
        img[argnan] = 0
        fits.writeto(outpath + basename.replace(".fits", "_" + band + "sci.fits"), img, imgheader, overwrite=True)

        image_data_electrons = np.abs(img) * gain_eff  # convert to electrons
        sky_mean, sky_median, sky_std = sigma_clipped_stats(img, sigma=3.0, maxiters=5)
        sigma_poisson = np.sqrt(image_data_electrons) / gain_eff
        sigmap = np.sqrt(sigma_poisson**2 + sky_std**2)
        arg0 = np.where(sigmap == 0)
        sigmap[arg0] = 1e20
        fits.writeto(outpath + basename.replace(".fits", "_" + band + "sigma.fits"), sigmap, imgheader, overwrite=True)


def dist_ellipse(
    n: Union[int, Tuple[int, int], List[int]], xc: float, yc: float, ratio: float, angle: float, double: bool = False
) -> np.ndarray:
    """
    Compute the elliptical distance from a center for each pixel in a grid.

    For a given grid size, this function calculates for each pixel the Euclidean
    distance in a coordinate system that is rotated and stretched according to
    the ellipse parameters. The resulting value can be interpreted as the
    "elliptical radius" (i.e., the distance from the center in the transformed
    space where the ellipse becomes a circle). Pixels lying on the ellipse
    defined by the parameters will have a value equal to the semi‑major axis
    length? Not exactly – the returned array gives the distance in the stretched
    coordinates, so it can be compared to a threshold to define elliptical masks.

    Parameters
    ----------
    n : int or tuple/list of two ints
        Dimensions of the output array. If an integer, a square array of size
        `n x n` is created. If a tuple/list of two integers `(nx, ny)`, the
        output shape will be `(ny, nx)` (rows = ny, columns = nx).
    xc : float
        X‑coordinate of the ellipse center (in pixel units, 0‑based from the
        left edge of the array).
    yc : float
        Y‑coordinate of the ellipse center (in pixel units, 0‑based from the
        top? – note that the function treats y increasing downward, consistent
        with image arrays).
    ratio : float
        Stretch factor along the rotated x‑axis. Typically this is the ratio of
        the semi‑major to semi‑minor axis (or vice‑versa) depending on convention.
        The code computes `sqrt((xtemp * ratio)**2 + ytemp**2)`, so larger
        `ratio` makes the ellipse more elongated in the direction of the rotated
        x‑axis.
    angle : float
        Rotation angle of the ellipse in degrees. The rotation is applied
        counter‑clockwise from the positive x‑axis (standard mathematical sense).
    double : bool, optional
        If True, use double‑precision (`float64`) for coordinates and the output
        array. If False (default), use single‑precision (`float32`). The choice
        affects memory usage and speed; single precision is usually sufficient
        for masking.

    Returns
    -------
    arr : 2D `~numpy.ndarray`
        Array of shape `(ny, nx)` containing the elliptical distance for each
        pixel. Values are floating point numbers.

    Raises
    ------
    ValueError
        If `n` is not an integer or a tuple/list of length 2.

    Examples
    --------
    >>> # Create a 100x100 array of distances from center (50,50) with axis ratio 2,
    >>> # rotated by 30 degrees, using single precision.
    >>> dist = dist_ellipse((100,100), 50, 50, 2.0, 30.0)
    >>> mask = dist <= 50   # mask of pixels inside an ellipse of "radius" 50
    """

    cosang = np.cos(angle * np.pi / 180)
    sinang = np.sin(angle * np.pi / 180)

    if isinstance(n, (tuple, list)) and len(n) == 2:
        nx = n[0]
        ny = n[1]
    elif isinstance(n, int):
        ny = nx = n
    else:
        raise ValueError("n must be an integer or a length-2 tuple/list")

    if double:
        # double-precision coords
        x = np.arange(nx, dtype=np.float64) - xc
        y = np.arange(ny, dtype=np.float64) - yc
        arr = np.empty((ny, nx), dtype=np.float64)
    else:
        # single-precision coords
        x = np.arange(nx, dtype=np.float32) - xc
        y = np.arange(ny, dtype=np.float32) - yc
        arr = np.empty((ny, nx), dtype=np.float32)

    # Rotate pixels to match ellipse orientation
    xcosang = x * cosang
    xsinang = x * sinang

    for i in range(ny):
        xtemp = xcosang + y[i] * sinang
        ytemp = -xsinang + y[i] * cosang
        arr[i, :] = np.sqrt((xtemp * ratio) ** 2 + ytemp**2)

    return arr



def caldelmax(image_stamp_list: List[str], cutsize_arcsec: float) -> Tuple[float, float]:
    """
    Compute the maximum RA and Dec offset (in arcseconds) from the image center
    to the corners of a cutout region defined by a given angular size.

    The function takes the first image from the list, determines its WCS,
    and calculates the celestial coordinates of its center and four corners
    after applying a square cutout of side `cutsize_arcsec` (centered on the
    image). The offsets in RA and Dec are then computed, and the maximum
    absolute offset in each coordinate is returned (converted to arcseconds).

    Parameters
    ----------
    image_stamp_list : list of str
        List of paths to image FITS files. Only the first element is used.
    cutsize_arcsec : float
        Desired size of the cutout square in arcseconds (full width).

    Returns
    -------
    maxra_arcsec : float
        Maximum absolute RA offset from the center to any corner, in arcseconds.
    maxdec_arcsec : float
        Maximum absolute Dec offset from the center to any corner, in arcseconds.

    Notes
    -----
    - The pixel scale is assumed to be 0.03 arcsec/pixel (hard‑coded).
    - The cutout half‑size in pixels is computed as `cutsize_arcsec / 0.03 / 2`.
    - The corner coordinates are obtained by transforming pixel coordinates
      to world coordinates using the WCS of the first image.
    """

    thumbrad = cutsize_arcsec / 0.03
    hdu = fits.open(image_stamp_list[0])[0]
    header = fits.open(image_stamp_list[0])[0].header
    xmax = header["NAXIS1"]
    ymax = header["NAXIS2"]
    xcen0 = int((xmax + 1) / 2) - 1
    ycen0 = int((ymax + 1) / 2) - 1
    wcs = WCS(hdu)

    xmin_1 = int(round(xcen0 - thumbrad))
    xmax_1 = int(round(xcen0 + thumbrad))
    ymin_1 = int(round(ycen0 - thumbrad))
    ymax_1 = int(round(ycen0 + thumbrad))

    ra0, dec0 = wcs.all_pix2world(xcen0, ycen0, 0)
    ra1, dec1 = wcs.all_pix2world(xmin_1, ymin_1, 0)
    ra2, dec2 = wcs.all_pix2world(0, ymax_1 + 1, 0)  # top left
    ra3, dec3 = wcs.all_pix2world(xmax_1 + 1, 0, 0)  # bottom right
    ra4, dec4 = wcs.all_pix2world(xmax_1 + 1, ymax_1 + 1, 0)  # top right

    #    ra0,dec0=wcs.all_pix2world(xcen0,ycen0,0)
    #    ra1,dec1=wcs.all_pix2world(0,0,0)
    #    ra2,dec2=wcs.all_pix2world(0,ymax-1,0)              top left
    #    ra3,dec3=wcs.all_pix2world(xmax-1,0,0)              bottom right
    #    ra4,dec4=wcs.all_pix2world(xmax-1,ymax-1,0)         top right

    delra1 = abs(ra1 - ra0)
    delra2 = abs(ra2 - ra0)
    delra3 = abs(ra3 - ra0)
    delra4 = abs(ra4 - ra0)

    deldec1 = abs(dec1 - dec0)
    deldec2 = abs(dec2 - dec0)
    deldec3 = abs(dec3 - dec0)
    deldec4 = abs(dec4 - dec0)

    maxra = max(delra1, delra2, delra3, delra4)
    maxdec = max(deldec1, deldec2, deldec3, deldec4)
    return maxra * 3600, maxdec * 3600

def __fetch_sqe(label_list, targ):
    for i, label in enumerate(label_list):
        if targ == label:
            return i
    raise ValueError("there no targ in list.")

def Fnu_to_Fl(Fnu, lambd):
    #mJy to erg/A 
    c=2.9979246e5
    mJy=1e-26
    a=Fnu*mJy
    Fl=a*(c*1e13)/(lambd**2)
    return Fl

def ABmag_to_covf(mzp_AB, wave):
    flux_mJy = 10**(-0.4*(mzp_AB))*3631*1000
    flux_fl_fuv = Fnu_to_Fl(flux_mJy,wave)
    return 1/flux_fl_fuv

def phys_to_image(object_band, pixsc):
    pixsr_ar = (pixsc/3600/180*np.pi)**2
    ZP_GALFIT = 2.5*np.log10(3631/(pixsr_ar*1e6))
    magab = ZP_GALFIT
    return ABmag_to_covf(magab, effective_wave[object_band])
    

def photometry_to_img(flux, flux_err, z, outputname,band=None,effectivewave=None, unit = 'mJy'):
    '''
    converting a photometry point to a image with a given band
    first, convert the flux to a Luminosity, in unit of 1e38erg/s/A
    unit: mJy or flambda, or MagAB
    '''
    data = np.ones((1,1))
    if effectivewave is not None:
        wave = effectivewave
    else:
        wave = effective_wave[band]
    cosmo = FlatLambdaCDM(H0=67.8 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.308)
    d=cosmo.luminosity_distance(z)/(1+z)
    dc=d.to(u.cm)
    dis=dc.value
    C_unit=1e38/(4*np.pi*dis**2)
    #C_unit=1
    if unit == 'mJy':
        flux = Fnu_to_Fl(flux,wave)
        flux_err = Fnu_to_Fl(flux_err,wave)
    elif unit == 'MagAB':
        flux = 1/ABmag_to_covf(flux, wave)
        flux_err = flux*flux_err*np.log(10)/2.5
    elif unit == 'flambda':
        flux = flux
        flux_err = flux_err
    L = flux/C_unit
    L_err = flux_err/C_unit
    if unit == 'L38':
        L = flux
        L_err = flux_err
    hdu_temp = fits.PrimaryHDU(np.float32(L*data), header = None)
    hdu_temp.header['band'] = band
    hdu_temp.header['photop'] = 1
    hdu_temp2 = fits.ImageHDU(0*np.float32(data))
    hdu_temp3 = fits.ImageHDU(np.float32(L_err*data))
    hdu_temp4 = fits.ImageHDU(np.float32(data))
    hdul = fits.HDUList([hdu_temp,hdu_temp2,hdu_temp3,hdu_temp4])
    hdul.writeto(outputname, overwrite=True)
    return 

def gen_pSed_data_lyric(cat_path, z_cat_path, cutout_dir, mock_dir, 
                   label_list, filter_list, zp_list, ebv = None):
    
    flux_cat = Table.read(cat_path, format = "ascii")
    z_cat = Table.read(z_cat_path, format = "ascii")

    for obj_row in flux_cat: # loop for every source in this group
    
        id = obj_row["id"] # 1-index
        print(f"create pSed mock data for {id}")

        mock_subdir = os.path.join(mock_dir,f"{id}")
        os.makedirs(mock_subdir, exist_ok=True)

        z_fit = z_cat[z_cat["id"] == id]["z_peak"][0]

        if z_fit < 0:
            z_fit = 0.001

        for fltr in filter_list: # loop for every filter.

            if obj_row[f"{fltr}_flux"] == -99.0:
                continue ## skip the lost band.

            mock_impath = os.path.join(mock_subdir, f"{fltr}.fits")
            flux_mjy = obj_row[f"{fltr}_flux"] * 1e-3
            fluxerr_mjy = obj_row[f"{fltr}_fluxerr"] * 1e-3
            photometry_to_img(flux = flux_mjy, flux_err = fluxerr_mjy, z = z_fit, outputname = mock_impath, band = fltr, unit='mJy')

        coord_sci_file = os.path.join(cutout_dir, f"obj{id}_{label_list[0]}sci.fits") ## may change to detband.
        hdu = fits.open(coord_sci_file)[0]
        header = hdu.header
        shape = hdu.data.shape
        ra,dec = WCS(header).all_pix2world((shape[0]+1)/2, (shape[1]+1)/2, 1)
        lyric_file = os.path.join(mock_subdir, f"obj{id}_pureSed.lyric")

        imgatlas = []

        param_file = open(lyric_file,'w')
        param_file.write(f"# This is a galfitS configuration file for galaxy {id}\n")
        param_file.write("# The config file provide a galfitS setup to perform a single sersic SED fitting with multi-band images.\n")

        # Region information

        param_file.write("# Region information\n")
        param_file.write(f'R1) obj{id}'+'\n')  # name of the target
        param_file.write('R2) ['+str(ra)+','+str(dec)+']\n')  # sky coordinate of the target [RA, Dec]
        param_file.write('R3) '+str(z_fit)+' \n\n') # redshift of the target
        if ebv is None:
            ebv = gal_ebv(ra = ra, dec = dec)

        imagels = list(string.ascii_lowercase)[:len(filter_list)]

        for i, (fltr, zp) in enumerate(zip(filter_list, zp_list)):

            imagel = imagels[i]

            if obj_row[f"{fltr}_flux"] == -99.0:
                continue
            else:
                imgatlas.append(imagel)
            mockfile = os.path.join(mock_subdir, f'{fltr}.fits') # mock path 

            param_file.write('# Image '+imagel.upper()+' \n')
            param_file.write('I'+imagel+'1)  [' + mockfile + ',0] \n') #sci image
            param_file.write('I'+imagel+f'2)  {fltr}\n') # band name
            param_file.write('I'+imagel+'3)  [' + mockfile + ',2] \n') # sigma image
            param_file.write('I'+imagel+'4)  [' + mockfile + ',3]\n') #psf image
            param_file.write('I'+imagel+'5)  1\n') # PSF fine sampling factor relative to data
            param_file.write('I'+imagel+'6)  [Noimg,0]\n') #mask image
            param_file.write('I'+imagel+'7)  cR\n') # unit of the image
            param_file.write('I'+imagel+'8)  -1 \n') # size to make cutout image region for fitting, unit arcsec
            param_file.write('I'+imagel+'9)  1 \n') # Conversion from image unit to flambda, -1 for default                 ## why ? for pure sed fitting, it must be specified as 1.
            param_file.write('I'+imagel+f'10) {zp}\n') # Magnitude photometric zeropoint                                 ## mag zp calculate
            param_file.write('I'+imagel+'11) uniform\n') # sky model
            param_file.write('I'+imagel+'12) [[0,-0.5,0.5,0.1,0]]\n') # sky parameter, (value, min, max, step)
            param_file.write('I'+imagel+'13) 0\n') # allow relative shifting
            param_file.write('I'+imagel+'14) [[0,-5,5,0.1,0],[0,-5,5,0.1,0]]\n') # [shiftx, shifty]
            param_file.write('I'+imagel+'15) 1\n\n') # Use SED information

        age= round(cosmo.age(z_fit).value,2)-0.2 
        age_list = [0] + list(np.logspace(-1, np.log10(age), 5))

        param_file.write("# Image atlas\n")
        param_file.write("Aa1) 'all'\n") # name of the image atlas
        param_file.write("Aa2) "+str(imgatlas)+"\n") # images in this atlas
        param_file.write('Aa3) 0\n') # whether the images have same pixel size
        param_file.write('Aa4) 0\n') # link relative shiftings
        param_file.write('Aa5) []\n') # spectra
        param_file.write('Aa6) []\n') # aperture size
        param_file.write('Aa7) []\n\n') # references images
        
        param_file.write("# Profile A\n")
        param_file.write(f'Pa1) obj0\n') # name of the component
        param_file.write('Pa2) sersic\n') # profile type
        param_file.write('Pa3) [0,-0.3,0.3,0.1,0]\n') # x-center [arcsec]
        param_file.write('Pa4) [0,-0.3,0.3,0.1,0]\n') # y-center [arcsec]
        param_file.write('Pa5) [0.2,0.1,1.7,0.1,0]\n') # effective radius [arcsec]
        param_file.write('Pa6) [2,0.5,6,0.1,0]\n') # Sersic index
        param_file.write('Pa7) [0,-90,90,1,0]\n') # position angle (PA) [degrees: Up=0, Left=90]
        param_file.write('Pa8) [0.8,0.5,1,0.01,0]\n') # axis ratio (b/a) [0.1=round, 1=flat]
        param_file.write(f'Pa9) [[-2,-8,0,0.1,1],[-2,-8,0,0.1,1],[-2,-8,0,0.1,1],[-2,-8,0,0.1,1],[-2,-8,0,0.1,1]]\n') # contemporary log star formation fraction         ## sfr
        param_file.write(f'Pa10) [{round(age_list[0],2)}, {round(age_list[1],2)}, {round(age_list[2], 2)}, {round(age_list[3],2)}, {round(age_list[4],2)}, {round(age_list[5],2)}]\n') # burst stellar age [Gyr]          ## age 
        param_file.write('Pa11) [[0.001,0.001,0.04,0.001,1]]\n') # metallicity [Z=0.02=Solar]
        param_file.write('Pa12) [[0.7,0,5.1,0.1,1]]\n') # Av dust extinction [mag]
        param_file.write('Pa13) [100,40,200,1,0]\n') # stellar velocity dispersion
        param_file.write('Pa14) [9,6,12,0.1,1]\n') # log stellar mass
        param_file.write('Pa15) bins \n') # star formation history type: burst/conti                    ## change to bins 
        param_file.write('Pa16) [-2,-4,-2,0.1,0]\n') # logU nebular ionization parameter
        param_file.write('Pa26) [3,0,5,0.1,1]\n') # amplitude of the 2175A bump on extinction curve
        param_file.write('Pa27) 0\n') # SED model, 0: full; 1: stellar only; 2: nebular only; 3: dust only
        param_file.write('Pa28) [8.14,4.5,10,0.1,0]\n') # log dust mass
        param_file.write('Pa29) [1.0, 0.1, 50, 0.1, 0]\n') # Umin, minimum radiation field
        param_file.write('Pa30) [1.0, 0.47, 7.32, 0.1, 0]\n') # qPAH, mass fraction of PAH
        param_file.write('Pa31) [1.0, 1.0, 3.0, 0.1, 0]\n') # alpha, powerlaw slope of U
        param_file.write('Pa32) [0.1, 0, 1.0, 0.1, 0]\n\n') # gamma, fraction illuminated by star forming region

        # Galaixes
        param_file.write("# Galaxy A\n")
        param_file.write('Ga1) mygal\n') # name of the galaxy
        param_file.write("Ga2) ['a']\n") # profile component
        param_file.write('Ga3) ['+str(z_fit)+',0.01,12.0,0.01,0]\n') # galaxy redshift
        param_file.write(f'Ga4) {ebv:.3f}\n') # the EB-V of Galactic dust reddening 
        param_file.write('Ga5) [1.0,0.5,2,0.05,0]\n') # normalization of spectrum when images+spec fitting
        param_file.write('Ga6) []\n') # narrow lines in nebular
        param_file.write('Ga7) 1\n\n') # number of components for narrow lines

        param_file.close()


_plot_stop = 100

def _process_one(i):
    objid = _g_ids[i]
    mass = 10**(_g_masses[i] - 9)
    idx_sfh = _g_objid_to_idx[objid]
    sfh = _g_sfh_data[idx_sfh]
    i50 = sfh.shape[0] // 2
    tbins = sfh.shape[1]
    age = np.logspace(1, _g_agebins_max[idx_sfh], tbins) / 1e9
    ssfr = sfh[i50, :] / mass
    ssfr_interp = np.interp(_g_age_list_binc, age, ssfr)
    logfcont_list = []
    for j in range(len(ssfr_interp)):
        logfcont_bin = np.log10(ssfr_interp[j] * (_g_age_list_obj[j+1] - _g_age_list_obj[j]))
        logfcont_list.append(logfcont_bin)

    if i < _plot_stop:
        return (logfcont_list, ssfr_interp), (age, ssfr)
    return (logfcont_list, ssfr_interp), None


def cal_sfh_prior(z, age_list_obj, SPS_catalog_path, sfhs_path, figout_path):
    '''
    z: redshift.
    '''

    z_lb = z - 1
    z_ub = z + 1

    fsfh = np.load(sfhs_path, allow_pickle = True)
    sps_cat = Table.read(SPS_catalog_path)
    z_list = sps_cat["z_50"]
    z_fltr = (z_list < z_ub) & (z_list > z_lb)
    sps_hz_cat = sps_cat[z_fltr]

    print(f"there are {len(sps_hz_cat)} galaxy samples for sfh prior.")

    # preload the datas.
    _ids = np.array(sps_hz_cat["id"])
    _masses = np.array(sps_hz_cat["mstar_50"])
    _sfh_data = fsfh["sfh"]
    _agebins_max = fsfh["agebins_max"]

    age_list_obj = np.array(age_list_obj)
    age_list_binc = 0.5 * (age_list_obj[1:] + age_list_obj[:-1])
    objid_all = fsfh["objid"]
    objid_to_idx = {oid: idx for idx, oid in enumerate(objid_all)}

    stop = len(sps_hz_cat)

    def _init_worker(ids, masses, oid2idx, sfh_data, agebins_max, age_obj, age_binc):
        global _g_ids, _g_masses, _g_objid_to_idx, _g_sfh_data, _g_agebins_max, _g_age_list_obj, _g_age_list_binc
        _g_ids = ids
        _g_masses = masses
        _g_objid_to_idx = oid2idx
        _g_sfh_data = sfh_data
        _g_agebins_max = agebins_max
        _g_age_list_obj = age_obj
        _g_age_list_binc = age_binc

    with Pool(4, initializer=_init_worker,
              initargs=(_ids, _masses, objid_to_idx, _sfh_data, _agebins_max, age_list_obj, age_list_binc)) as pool:
        results = list(tqdm(pool.imap_unordered(_process_one, range(stop), chunksize=64), total=stop))

    ssfr_list = []
    logf_cont_list = []

    fig, ax = plt.subplots(figsize = (8,6))
    for (logf_cont, ssfr_interp), plot_data in results:
        ssfr_list.append(ssfr_interp)
        logf_cont_list.append(logf_cont)

        if plot_data is not None:
            age, ssfr = plot_data
            ax.step(age, ssfr, lw=0.5, color="grey", alpha=0.5)



    ssfr_list = np.array(ssfr_list)
    logf_cont_list = np.array(logf_cont_list)
    logf_cont_median, logf_cont_std = [], []
    ssfr_median, ssfr_std = [], []

    for i in range(len(age_list_binc)):

        logf_cont_list_bin = logf_cont_list[:, i] ## f_cont: mass fraction of this bins.
        pos_fltr = np.isfinite(logf_cont_list_bin) & (logf_cont_list_bin < 0)
        logf_cont_median.append(np.nanmedian(logf_cont_list_bin[pos_fltr]))
        logf_cont_std.append(np.nanstd(logf_cont_list_bin[pos_fltr]))

        ssfr_list_bin = ssfr_list[:, i]
        pos_fltr = (ssfr_list_bin > 0) & (np.isfinite(ssfr_list_bin))
        ssfr_median.append(np.nanmedian(ssfr_list_bin[pos_fltr]))
        ssfr_std.append(np.nanstd(ssfr_list_bin[pos_fltr]))

    print(logf_cont_std)
    ax.errorbar(age_list_binc, ssfr_median, yerr=ssfr_std, fmt="o", color="r", label="median")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Lookback time(Gyr)", fontsize = 18)
    ax.set_ylabel(r"ssfr($Gyr^{-1}$)", fontsize = 18)
    plt.savefig(figout_path)

    logf_cont_median = np.array(logf_cont_median)
    logf_cont_std = np.array(logf_cont_std)
    ssfr_median = np.array(ssfr_median)
    ssfr_std = np.array(ssfr_std)


    return logf_cont_median, logf_cont_std

def write_sfh(sfh_path):

    with open(sfh_path, "w") as f:
        f.write

def prepare_galfits(
        lyric_path:str,
        prior_path: str,  
        cat_file: str,
        objects: list, 
        det_label: str, 

        sci_list: List[str],
        psf_list: List[str],

        zero_list: List[str], 
        pixscl_list: List[str], 
        label_list: List[str], 
        filter_list: List[str],
        geo_smdir: str,
        pSed_smdir: str,
        imgSed_smdir: str, 

        SPS_catalog_path: str = None, 
        sfhs_path: str = None, 
        z_list: List[str] = None,  
        sigma_list: List[str] = None, 
        mask_list: List[str] = None, 
        use_sed = 0,
        use_sfh_prior: bool = False,
        convf: bool = False,
        convl: Optional[List[float]] = None,
        ebv:float = None
    ):
    
    n_band = len(label_list)
    if sigma_list is None:
        sigma_list = np.array(["none"] * n_band)
    if mask_list is None:
        mask_list = np.array(["none"] * n_band)

    table = ascii.read(cat_file)


    idx_det = __fetch_sqe(label_list, det_label)
    det_pixscl = pixscl_list[idx_det]
    header = fits.open(sci_list[idx_det])[0].header
    xmax = header["NAXIS1"]
    ymax = header["NAXIS2"]

    id = table[objects[0]]["label"]
    ra = table[objects[0]]["ra"]
    dec = table[objects[0]]["dec"]

    sfh_prior_keys = []
    if ebv is None:
        ebv = gal_ebv(ra = ra, dec = dec)

    if use_sed==0:
        z = [0.4,] * len(objects)
        age_list = [[1,2,3,4,5,6],] * len(objects)
    else:
        if z_list is not None:  
            z = z_list 
        else:
            z = np.array(table[objects]['z_peak'])
        ages= [round(cosmo.age(z[jj]).value,1)-0.2 for jj in range(len(z))]
        age_list = [[0] + list(np.logspace(-1, np.log10(age), 5)) for age in ages]

    cutsize_pix = 5 * max(xmax, ymax)
    cutsize_arcsec = cutsize_pix * det_pixscl

    if not convf:
        convlist = np.array([phys_to_image(filter_list[xx],pixscl_list[xx]) for xx in range(len(filter_list))])
    else:
        convlist = np.array([convl[xx] for xx in range(len(filter_list))])


    with open(lyric_path, "w") as f:
        f.write(f"# galfits config file for galaxy id={id}\n")
        f.write(f"# Region information\n\n")

        f.write(f"R1) obj{id}\n")
        f.write(f"R2) [{ra:.5f},{dec:.5f}]\n")
        f.write(f"R3) {z[0]:.5f}\n\n")

        img_letter = np.array(list(string.ascii_lowercase[:n_band]))  # ['a','b','c'......]

        # check image, if the cutout image is non-detection at the central point, skip the filter.
        argindex = []

        for bb in range(n_band):
            img = fits.open(sci_list[bb])[0].data
            ybb = np.shape(img)[0]
            xbb = np.shape(img)[1]
            xcen0 = int((xbb + 1) / 2) - 1
            ycen0 = int((ybb + 1) / 2) - 1
            print(id, filter_list[bb], np.shape(img), xcen0, ycen0)
            if img[ycen0][xcen0] == 0 or np.isnan(img[ycen0][xcen0]):
                continue
            else:
                argindex.append(int(bb))
        argindex = np.array(argindex)
        print(argindex, len(argindex))
        

        if len(argindex) != 0:
            
            img_letter = img_letter[argindex]
            filter_list = filter_list[argindex]
            label_list = label_list[argindex]
            sci_list = sci_list[argindex]
            sigma_list = sigma_list[argindex]
            psf_list = psf_list[argindex]
            convlist = convlist[argindex]
            zero_list = zero_list[argindex]
            mask_list = mask_list[argindex]
        
        img_letter = [str(item) for item in img_letter]
        print(img_letter)

        for b in range(len(argindex)):

            f.write(f"# input images for {label_list[b]} band\n")

            f.write(f"I{img_letter[b]}1) [{sci_list[b]},0]\n")
            f.write(f"I{img_letter[b]}2) {filter_list[b]}\n")
            f.write(f"I{img_letter[b]}3) [{sigma_list[b]},0]\n")
            f.write(f"I{img_letter[b]}4) [{psf_list[b]},0]\n")
            f.write(f"I{img_letter[b]}5) 1\n")
            f.write(f"I{img_letter[b]}6) [{mask_list[b]},0]\n")
            f.write(f"I{img_letter[b]}7) MJy/sr\n")
            f.write(f"I{img_letter[b]}8) {cutsize_arcsec} \n")
            f.write(f"I{img_letter[b]}9) {convlist[b]} \n")  
            f.write(f"I{img_letter[b]}10) {zero_list[b]}\n")
            f.write(f"I{img_letter[b]}11) uniform\n")
            f.write(f"I{img_letter[b]}12) [[0,-1e5,1e5,0.1,0]]\n")  # sky parameter, (value, min, max, step, fix), 0 for fix.
            f.write(f"I{img_letter[b]}13) 1\n")  # allow relative shifting, 0 for False.
            f.write(f"I{img_letter[b]}14) [[0,-5,5,0.1,1],[0,-5,5,0.1,1]] \n")
            f.write(f"I{img_letter[b]}15) {use_sed}\n\n")  # Use SED information

        f.write(f"# image atlas\n")
        f.write(f"Aa1) 'img list'\n")
        f.write(f"Aa2) {img_letter}\n")
        f.write(f"Aa3)  0\n")
        f.write(f"Aa4) 0\n")
        f.write(f"Aa5) []\n")
        f.write(f"Aa6) []\n")
        f.write(f"Aa7) []\n\n")

        # loop over all (primary & secondary) sources
        obj_letter = list(string.ascii_lowercase[: len(objects)])  # ['a','b','c'......]

        objf = []
        zf = []

        for i in range(len(obj_letter)):

            object_id = str(table[objects[i]]["label"])
            age_list_obj = age_list[i]

            geo_file = os.path.join(geo_smdir, f"obj{object_id}.gssummary")
            pSed_file = os.path.join(pSed_smdir, f"{object_id}/results/obj{object_id}.gssummary")
            imgSed_file = os.path.join(imgSed_smdir, f"obj{object_id}.gssummary")

            if use_sed == 0: # pure Image fitting 
                # for geo parameters, if the geo_file exists, we will use the results of it. If not, we would use the SExtractor results.
                if os.path.exists(geo_file):
                    geo_results = ascii.read(geo_file)
                    delta_ra = geo_results["best_value"][geo_results["pname"] == "obj0_xcen"][0] + (table[objects[i]]["ra"] - ra) * 3600
                    delta_dec = geo_results["best_value"][geo_results["pname"] == "obj0_ycen"][0] + (table[objects[i]]["dec"] - dec) * 3600
                    re_galfit = geo_results["best_value"][geo_results["pname"] == "obj0_Re"][0] 
                    n_galfit = geo_results["best_value"][geo_results["pname"] == "obj0_n"][0]
                    pa_galfit = geo_results["best_value"][geo_results["pname"] == "obj0_ang"][0]
                    q_galfit = geo_results["best_value"][geo_results["pname"] == "obj0_axrat"][0]
                    object_description = "(primary/secondary) object has already been fit, galfits output file exists. Reading result from file"
                    fix_geo = ["0", "0", "0", "0", "0", "0"]

                else:
                    delta_ra = (table[objects[i]]["ra"] - ra) * 3600
                    delta_dec = (table[objects[i]]["dec"] - dec) * 3600
                    re_galfit = (10.0 ** (-0.79) * table[objects[i]]["flux_radius"] ** 1.87) * det_pixscl  # in arcsec
                    re_galfit = round(re_galfit, 2)
                    n_galfit = 2.5
                    if table[objects[i]]["orientation"] > 0:
                        pa_galfit = table[objects[i]]["orientation"] - 90
                    else:
                        pa_galfit = table[objects[i]]["orientation"] + 90
                    pa_galfit = round(pa_galfit, 2)
                    q_galfit = 1 - table[objects[i]]["ellipticity"]
                    q_galfit = round(q_galfit, 3)
                    object_description = "(primary/secondary) object has not yet been fit, output file does not exist"
                    fix_geo = ["1", "1", "1", "1", "1", "1"]  
                    max_ra, max_dec = caldelmax(sci_list, cutsize_arcsec)

                    if abs(delta_ra) > 1.0 * max_ra or abs(delta_dec) > 1.0 * max_dec:
                        object_description = ("(primary/secondary) object has not yet been fit, output file does not exist, outside of frame")
                        fix_geo[0] = "0"
                        fix_geo[1] = "0"
                        fix_geo[4] = "0"
                        fix_geo[5] = "0"

                f_cont_1, f_cont_2, f_cont_3, f_cont_4, f_cont_5 = -2,-2,-2,-2,-2
                f_cont_median_list = [f_cont_1, f_cont_2, f_cont_3, f_cont_4, f_cont_5]
                f_cont_upper_list = [0.] * len(f_cont_median_list)
                f_cont_lower_list = [-8.] * len(f_cont_median_list)
                f_cont_std_list = [0.1] * len(f_cont_median_list)
                metalicity = 0.01
                Av = 0.7
                logM = 9.0
                bump_2175 = 3
                fix_sed = [[0, 0, 0, 0, 0], 0, 0, 0, 0] # for f_cont_array, mentalicity, Av, logM, bump2175 each.

            else: # image + sed fitting
                # in principle, geo_file must exists.
                # but we want to fix the bright source when fainter source is fitted.
                if os.path.exists(imgSed_file):
                    object_description = "(primary/secondary) object has already been fit, galfits output file exists. Reading result from file"
                    imSed_results = ascii.read(imgSed_file)

                    delta_ra = imSed_results["best_value"][imSed_results["pname"] == "obj0_xcen"][0] + (table[objects[i]]["ra"] - ra) * 3600
                    delta_dec = imSed_results["best_value"][imSed_results["pname"] == "obj0_ycen"][0] + (table[objects[i]]["dec"] - dec) * 3600
                    re_galfit = imSed_results["best_value"][imSed_results["pname"] == "obj0_Re"][0] 
                    n_galfit = imSed_results["best_value"][imSed_results["pname"] == "obj0_n"][0]
                    pa_galfit = imSed_results["best_value"][imSed_results["pname"] == "obj0_ang"][0]
                    q_galfit = imSed_results["best_value"][imSed_results["pname"] == "obj0_axrat"][0]

                    f_cont_1 = imSed_results["best_value"][imSed_results["pname"] == "obj0_f_cont_bin1"][0]
                    f_cont_2 = imSed_results["best_value"][imSed_results["pname"] == "obj0_f_cont_bin2"][0]
                    f_cont_3 = imSed_results["best_value"][imSed_results["pname"] == "obj0_f_cont_bin3"][0]
                    f_cont_4 = imSed_results["best_value"][imSed_results["pname"] == "obj0_f_cont_bin4"][0]
                    f_cont_5 = imSed_results["best_value"][imSed_results["pname"] == "obj0_f_cont_bin5"][0]
                    f_cont_median_list = [f_cont_1, f_cont_2, f_cont_3, f_cont_4, f_cont_5]
                    f_cont_upper_list = [0.] * len(f_cont_median_list)
                    f_cont_lower_list = [-8.] * len(f_cont_median_list)
                    f_cont_std_list = [0.1] * len(f_cont_median_list)

                    metalicity = imSed_results["best_value"][imSed_results["pname"] == "obj0_Z_value"][0]
                    Av = imSed_results["best_value"][imSed_results["pname"] == "obj0_Av_value"][0]
                    logM = imSed_results["best_value"][imSed_results["pname"] == "logM_obj0"][0]
                    bump_2175 = imSed_results["best_value"][imSed_results["pname"] == "AVbump_obj0"][0]
                    fix_geo = ["0", "0", "0", "0", "0", "0"]
                    fix_sed = [[0,0,0,0,0], 0,0,0,0]

                else: # read initial guess of geo_file and pSed_file and don't fix it.
                    object_description = "(primary/secondary) object has not yet been fit, output file does not exist"
                    geo_results = ascii.read(geo_file)
                    delta_ra = geo_results["best_value"][geo_results["pname"] == "obj0_xcen"][0] + (table[objects[i]]["ra"] - ra) * 3600
                    delta_dec = geo_results["best_value"][geo_results["pname"] == "obj0_ycen"][0] + (table[objects[i]]["dec"] - dec) * 3600
                    re_galfit = geo_results["best_value"][geo_results["pname"] == "obj0_Re"][0] 
                    n_galfit = geo_results["best_value"][geo_results["pname"] == "obj0_n"][0]
                    pa_galfit = geo_results["best_value"][geo_results["pname"] == "obj0_ang"][0]
                    q_galfit = geo_results["best_value"][geo_results["pname"] == "obj0_axrat"][0]
                    fix_geo = ["1", "1", "1", "1", "1", "1"]

                    sed_results = ascii.read(pSed_file)

                    if use_sfh_prior:
                        if SPS_catalog_path is None or sfhs_path is None:
                            raise ValueError(f"{SPS_catalog_path} or {sfhs_path} is needed for sfh prior calculation.")
            
                        figout_path = os.path.join(imgSed_smdir, f"obj{object_id}_sfh_prior.png")
                        f_cont_median_list, f_cont_std_list = cal_sfh_prior(z[i], age_list_obj, SPS_catalog_path, sfhs_path, figout_path)
                        f_cont_upper_list = f_cont_median_list + 3*f_cont_std_list
                        f_cont_lower_list = f_cont_median_list - 3*f_cont_std_list
                        for j in range(1, len(f_cont_median_list)):
                            sfh_prior_keys.append(f'obj{i}_f_cont_bin_{j}')   
                    else:
                        f_cont_1 = sed_results["best_value"][sed_results["pname"] == "obj0_f_cont_bin1"][0]
                        f_cont_2 = sed_results["best_value"][sed_results["pname"] == "obj0_f_cont_bin2"][0]
                        f_cont_3 = sed_results["best_value"][sed_results["pname"] == "obj0_f_cont_bin3"][0]
                        f_cont_4 = sed_results["best_value"][sed_results["pname"] == "obj0_f_cont_bin4"][0]
                        f_cont_5 = sed_results["best_value"][sed_results["pname"] == "obj0_f_cont_bin5"][0]
                        f_cont_median_list = [f_cont_1, f_cont_2, f_cont_3, f_cont_4, f_cont_5]
                        f_cont_upper_list = [0.] * len(f_cont_median_list)
                        f_cont_lower_list = [-8.] * len(f_cont_median_list)
                        f_cont_std_list = [0.1] * len(f_cont_median_list)

                    metalicity = sed_results["best_value"][sed_results["pname"] == "obj0_Z_value"][0]
                    Av = sed_results["best_value"][sed_results["pname"] == "obj0_Av_value"][0]
                    logM = sed_results["best_value"][sed_results["pname"] == "logM_obj0"][0]
                    bump_2175 = sed_results["best_value"][sed_results["pname"] == "AVbump_obj0"][0]
                    fix_sed = [[1,1,1,1,0], 1,1,1,1] # fix one bin to release degeneracy
    

            max_ra, max_dec = caldelmax(sci_list, cutsize_arcsec)
            if abs(delta_ra) > 1.2 * max_ra or abs(delta_dec) > 1.2 * max_dec:
                continue
            objf.append(obj_letter[i])
            zf.append(z[i])

            f.write("# Sersic function\n")
            f.write(f"# {object_description}\n")
            f.write(f"# {object_id}\n")
            f.write(f"P{obj_letter[i]}1) obj{i}\n")
            f.write(f"P{obj_letter[i]}2) sersic\n")
            f.write(f"P{obj_letter[i]}3) [{delta_ra:.2f},{1*delta_ra-10*det_pixscl:.2f},{1*delta_ra+10*det_pixscl:.2f},{0.5*det_pixscl},{fix_geo[0]}]\n") 
            f.write(f"P{obj_letter[i]}4) [{delta_dec:.2f},{1*delta_dec-10*det_pixscl:.2f},{1*delta_dec+10*det_pixscl:.2f},{0.5*det_pixscl},{fix_geo[1]}]\n")
            f.write(f"P{obj_letter[i]}5) [{re_galfit},{0.3*det_pixscl},{cutsize_arcsec},{0.3*det_pixscl},{fix_geo[2]}] \n")  # Re half-light radius [arcsec]
            f.write(f"P{obj_letter[i]}6) [{n_galfit},0.2,8,0.1,{fix_geo[3]}] \n") 
            f.write(f"P{obj_letter[i]}7) [{pa_galfit},-180,180,1,{fix_geo[4]}] \n")
            f.write(f"P{obj_letter[i]}8) [{q_galfit},0.1,1,0.02,{fix_geo[5]}] \n")
            f.write(f"P{obj_letter[i]}9)  [[{f_cont_median_list[0]},{f_cont_lower_list[0]},{f_cont_upper_list[0]},{f_cont_std_list[0]},{fix_sed[0][0]}],[{f_cont_median_list[1]},{f_cont_lower_list[1]},{f_cont_upper_list[1]},{f_cont_std_list[1]},{fix_sed[0][1]}],[{f_cont_median_list[2]},{f_cont_lower_list[2]},{f_cont_upper_list[2]},{f_cont_std_list[2]},{fix_sed[0][2]}],[{f_cont_median_list[3]},{f_cont_lower_list[3]},{f_cont_upper_list[3]},{f_cont_std_list[3]},{fix_sed[0][3]}],[{f_cont_median_list[4]},{f_cont_lower_list[4]},{f_cont_upper_list[4]},{f_cont_std_list[4]},{fix_sed[0][4]}]]\n")
            f.write(f"P{obj_letter[i]}10) [{round(age_list_obj[0],2)}, {round(age_list_obj[1],2)}, {round(age_list_obj[2], 2)}, {round(age_list_obj[3],2)}, {round(age_list_obj[4],2)}, {round(age_list_obj[5],2)}]\n")
            f.write(f"P{obj_letter[i]}11) [[{metalicity},0.001,0.04,0.001,{fix_sed[1]}]] \n")
            f.write(f"P{obj_letter[i]}12) [[{Av},0.,5.1,0.1,{fix_sed[2]}]] \n")
            f.write(f"P{obj_letter[i]}13) [100,40,200,1,0] \n")
            f.write(f"P{obj_letter[i]}14) [{logM}, 6, 12, 0.1, {fix_sed[3]}]\n")
            f.write(f"P{obj_letter[i]}15) bins\n")
            f.write(f"P{obj_letter[i]}16) [-2,-4,-2,0.1,0] \n")
            f.write(f"P{obj_letter[i]}26) [{bump_2175},0,5,0.1,{fix_sed[4]}] \n")
            f.write(f"P{obj_letter[i]}27) 0 \n")
            f.write(f"P{obj_letter[i]}28) [8.14,4.5,10,0.1,0] \n")
            f.write(f"P{obj_letter[i]}29) [1.0, 0.1, 50, 0.1, 0] \n")
            f.write(f"P{obj_letter[i]}30) [1.0, 0.47, 7.32, 0.1, 0] \n")
            f.write(f"P{obj_letter[i]}31) [1.0, 1.0, 3.0, 0.1, 0]  \n")
            f.write(f"P{obj_letter[i]}32) [0.1, 0, 1.0, 0.1, 1]  \n\n")


        for obj, obj_z in zip(objf, zf):  
             f.write("# all objects to be fitted\n")
             f.write(f"G{obj}1) {obj}\n")
             f.write(f"G{obj}2) {obj}\n")
             f.write(f"G{obj}3) [{obj_z:.5f},0.01,12,0.01,0]\n")      # redshift 
             f.write(f"G{obj}4) {ebv:.3f} \n")
             f.write(f"G{obj}5) [1.,0.8,1.2,0.05,0] \n")      # normalization of spectrum when images+spec fitting
             f.write(f"G{obj}6) [] \n")
             f.write(f"G{obj}7) 1 \n")

    if use_sfh_prior:
        with open(prior_path, "w") as f:
            f.write(f"GP) [{','.join(sfh_prior_keys)}]")
