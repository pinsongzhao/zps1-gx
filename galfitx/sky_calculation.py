"""
Identifier:     galfitx/sky_calculation.py
Name:           sky_calculation.py
Description:    calculate local bkg value
Author:         Chao Ma
Created:        2026-01-19
Modified-History:
    2026-01-19, Chao Ma, created
"""

import numpy as np
from astropy.io import fits, ascii
from astropy.stats import sigma_clipped_stats
from scipy.optimize import curve_fit
import statistics
from typing import Tuple, Union, List


def create_skymap(
    weight_name: str,
    catalog_name: str,
    skymap_name: str,
    scale: float,
    offset: float,
) -> np.ndarray:
    """
    Build a sky‑map array where each pixel records how many scaled Kron ellipses
    (from a SExtractor catalog) cover it.

    The resulting integer array has the same dimensions as the input weight image.
    - Pixels with value `-1` indicate no flux (weight == 0).
    - Pixels with value `0` are blank sky (no overlapping ellipse).
    - Pixels with value `>0` give the count of overlapping ellipses.

    For each source in the catalog, a scaled Kron ellipse is constructed using:
    radius = scale * semimajor_sigma * kron_radius + offset
    The ellipse is oriented according to the source's orientation and ellipticity.
    The bounding box of the ellipse is determined, and the elliptical mask is
    computed via `dist_ellipse`. The mask is then added to the global sky map.

    Parameters
    ----------
    weight_name : str
        Path to the FITS weight image (used to identify no‑flux pixels and to
        obtain the image dimensions and header).
    catalog_name : str
        Path to the ASCII SExtractor catalog. Must contain columns:
        'ellipticity', 'xcentroid', 'ycentroid', 'orientation', 'semimajor_sigma',
        'kron_radius'.
    skymap_name : str
        Output filename for the sky‑map FITS file.
    scale : float
        Scaling factor applied to the Kron radius and sigma.
    offset : float
        Constant offset added to the ellipse size (in pixels).

    Returns
    -------
    skymap : 2D `~numpy.ndarray` of int
        The constructed sky map array (same shape as the weight image).
    """

    wht = fits.open(weight_name)[0].data
    header = fits.open(weight_name)[0].header
    ny, nx = wht.shape

    # initalize skymap to zero
    skymap = np.zeros((ny, nx), dtype=int)

    # read catalog
    cat = ascii.read(catalog_name)
    ellipticity = cat["ellipticity"]
    x_image = cat["xcentroid"]
    y_image = cat["ycentroid"]
    theta_image = cat["orientation"] * np.pi / 180  # convert degrees to radian

    # calcluate semimajor axis of scaled kron ellipses
    rad = scale * cat["semimajor_sigma"] * cat["kron_radius"] + offset

    for i in range(len(cat)):
        xfac = rad[i] * (abs(np.sin(theta_image[i])) + (1 - ellipticity[i]) * abs(np.cos(theta_image[i])))
        yfac = rad[i] * (abs(np.cos(theta_image[i])) + (1 - ellipticity[i]) * abs(np.sin(theta_image[i])))

        xfac = max(xfac, 10)  # enforce at least 10 pixels in the projection of x-axis
        yfac = max(yfac, 10)

        major = max([xfac, yfac])
        minor = min([xfac, yfac])

        angle = theta_image[i] * 180 / np.pi  # convert radian to degrees
        angle %= 360  # contraint to [0,360)
        if angle > 180:
            angle -= 180
        if angle > 90:
            angle -= 180
        if abs(angle) < 45:
            xfac = major
            yfac = minor
        else:
            xfac = minor
            yfac = major

        xlo = min(max(round(x_image[i] - xfac), 0), nx - 1)  # clamp into valid pixel index range [0, nx-1]
        xhi = min(max(round(x_image[i] + xfac), 0), nx - 1)
        ylo = min(max(round(y_image[i] - yfac), 0), ny - 1)
        yhi = min(max(round(y_image[i] + yfac), 0), ny - 1)

        arr = dist_ellipse(
            [xhi - xlo + 1, yhi - ylo + 1],
            x_image[i] - xlo,
            y_image[i] - ylo,
            1.0 / (1 - ellipticity[i]),
            theta_image[i] * 180 / np.pi - 90,
        )
        idx = np.where(arr <= max(rad[i], 5))
        arr = arr * 0
        if idx[0].size > 0:
            arr[idx] = 1

        for j in range(xlo, xhi + 1):
            for k in range(ylo, yhi + 1):
                if arr[k - ylo, j - xlo] == 1:
                    skymap[k, j] += 1

    off = np.where(wht == 0)
    if off[0].size > 0:
        skymap[off] = -1

    fits.writeto(skymap_name, skymap, header, overwrite=True)

    return skymap


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


def getsky(
    obj_idx: int,
    catalog_name: str,
    image_name: str,
    skymap_name: str,
    skyfile: str,
    dstep: int = 8,
    wstep: int = 8,
    gap: int = 8,
    nslope: int = 5,
    global_sky: float = 43.3,
    global_sigsky: float = 585.72,
) -> None:
    """
    Estimate the local sky background for a given object using annular apertures.

    The function reads the image, catalog, and a pre‑computed sky‑map (which
    flags pixels belonging to other sources). For the object specified by
    `obj_idx`, it builds a series of elliptical annuli centred on the object,
    moving outward. For each annulus that contains at least 5 sky pixels
    (pixels with sky‑map value 0), it computes a robust sky value by:
    1. Clipping outliers with sigma‑clipping,
    2. Optionally histogram‑fitting a Gaussian to the pixel distribution,
    3. Recording the mean/peak and its uncertainty.
    The search continues until the slope of sky vs. radius becomes positive
    or the outer radius exceeds the maximum distance to any sky pixel.
    The final sky value is taken as the minimum among all measured annuli
    (to avoid contamination by nearby sources). The result, together with
    its uncertainty, the radius used, the object's magnitude, and a flag,
    is written to `skyfile`.

    Parameters
    ----------
    obj_idx : int
        Index (0‑based) of the object in the catalog.
    catalog_name : str
        Path to the ASCII SExtractor catalog. Must contain columns:
        'ellipticity', 'xcentroid', 'ycentroid', 'orientation',
        'semimajor_sigma', 'kron_radius', 'mag_auto'.
    image_name : str
        Path to the science FITS image.
    skymap_name : str
        Path to the sky‑map FITS image (produced e.g. by `create_skymap`).
        Pixels with value 0 are considered safe for sky estimation.
    skyfile : str
        Output text file where the results will be written. The file will contain
        a single line with five numbers:
        sky_value sky_sigma sky_radius object_mag flag
    dstep : int, optional
        Radial step between successive annuli (pixels). Default 8.
    wstep : int, optional
        Width of each annulus (pixels). Default 8.
    gap : int, optional
        Gap between the Kron radius and the first annulus (pixels). Default 8.
    nslope : int, optional
        Number of successive annuli used to detect a positive slope in sky
        vs. radius. Default 5.
    global_sky : float, optional
        Fallback sky value used when the measurement fails. Default 43.3.
    global_sigsky : float, optional
        Fallback sky uncertainty. Default 585.72.

    Returns
    -------
    None
        The function writes the result to `skyfile`.

    Notes
    -----
    - The function relies on an external `dist_ellipse` function (not defined here)
      to compute elliptical distances.
    - The flag value is a bitmask that encodes various conditions encountered
      during the estimation (e.g., insufficient pixels, slope detection, etc.).
    """

    sky_flag = 0

    hdu = fits.open(image_name)[0]
    data = hdu.data
    ny, nx = data.shape
    x, y = np.arange(nx), np.arange(ny)
    xarr, yarr = np.meshgrid(x, y)

    cat = ascii.read(catalog_name)
    ellipticity = cat["ellipticity"]
    rad = cat["semimajor_sigma"] * cat["kron_radius"]

    skymap = fits.open(skymap_name)[0].data

    # ensure that image have enough sky pixel
    idx = np.where(skymap == 0)
    if idx[0].size == 0:
        with open(skyfile, "w") as f:
            line = f"0 0 0 {cat[obj_idx]['mag_auto']} 32\n"
            f.write(line)

    if idx[0].size < 0:
        raise ValueError("Something really wrong in getsky")

    if idx[0].size > 0:
        max_rad = max(
            np.sqrt((cat[obj_idx]["xcentroid"] - xarr[idx]) ** 2 + (cat[obj_idx]["ycentroid"] - yarr[idx]) ** 2)
        )
        if rad[obj_idx] > max_rad:
            rad[obj_idx] = 0
            sky_flag += 1
        contrib_sky = 1e30

        ################

        # content of contributing sources

        #################

        # array of radius
        nstep = int(max(data.shape) / float(dstep))
        radius = np.arange(nstep) * dstep + rad[obj_idx] + gap

        # array containing sky info in each ring
        ringsky = radius * 0
        ringsigma = radius * 0

        # arrays for the radius, sky, scatter of the last nslope
        sl_sky = np.zeros(nslope)
        sl_rad = np.zeros(nslope)
        sl_sct = np.zeros(nslope)

        last_slope = 1
        slope_change = 0
        min_sky = 1e30
        min_sky_rad = 1e30
        min_sky_sig = 999
        min_sky_flag = 0

        # loop over radii
        for i in range(nstep):

            theta = cat[obj_idx]["orientation"] * np.pi / 180  # covert to radian
            xfac = rad[obj_idx] * (abs(np.sin(theta)) + (1 - ellipticity[obj_idx]) * abs(np.cos(theta)))
            yfac = rad[obj_idx] * (abs(np.cos(theta)) + (1 - ellipticity[obj_idx]) * abs(np.sin(theta)))

            xfac = max(xfac, 10) + dstep * i + wstep
            yfac = max(yfac, 10) + dstep * i + wstep

            major = max([xfac, yfac])
            minor = min([xfac, yfac])

            angle = theta * 180 / np.pi  # convert radian to degrees
            angle %= 360  # contraint to [0,360)

            if angle > 180:
                angle -= 180
            if angle > 90:
                angle -= 180
            if abs(angle) < 45:
                xfac = major
                yfac = minor
            else:
                xfac = minor
                yfac = major

            xlo = min(max(round(cat[obj_idx]["xcentroid"] - xfac), 0), nx - 1)
            xhi = min(max(round(cat[obj_idx]["xcentroid"] + xfac), 0), nx - 1)
            ylo = min(max(round(cat[obj_idx]["ycentroid"] - yfac), 0), ny - 1)
            yhi = min(max(round(cat[obj_idx]["ycentroid"] + yfac), 0), ny - 1)

            arr = dist_ellipse(
                [xhi - xlo + 1, yhi - ylo + 1],
                cat[obj_idx]["xcentroid"] - xlo,
                cat[obj_idx]["ycentroid"] - ylo,
                1.0 / (1 - ellipticity[obj_idx]),
                theta * 180 / np.pi - 90,
            )

            # get the index of all sky pixels within the current annulus
            inner = rad[obj_idx] + dstep * i
            outer = inner + wstep
            submap = skymap[ylo: yhi + 1, xlo: xhi + 1]
            mask = (arr > inner) & (arr <= outer) & (submap == 0)
            ring_empty_idx = np.where(mask)

            # delete temporary array since it’s no longer needed.
            del arr

            if ring_empty_idx[0].size < 5:
                # no pixel to calcuate sky
                ringsky = -1
                ringsigma = 1e30
            else:
                # 1D array of pixel values in the current annulus with submap==0
                skyim = data[ylo: yhi + 1, xlo: xhi + 1][ring_empty_idx]
                n_ring = skyim.size
                mean, median, sigma = sigma_clipped_stats(skyim, sigma=3.0, maxiters=5)
                # survived ring pixel index after clipping
                ring_sig_idx = np.where((skyim > (mean - 3 * sigma)) & (skyim < (mean + 3 * sigma)))
                nw = ring_sig_idx[0].size

                # if enough pixels, keep them
                if nw > 4:
                    skyim = skyim[ring_sig_idx]
                    n_ring = skyim.size
                    # reduce the max size to 250x250 pixels to make procedure faster
                    npix = int(min(np.sqrt(n_ring), 250.0))

                    np.random.seed(0)  # set global seed
                    indices = ((npix**2) * np.random.random(npix**2)).astype(int)
                    skyim = skyim[indices].reshape(npix, npix)

                    # count the disticnt value
                    if np.unique(skyim).size < 5:
                        # Fallback when too few unique points
                        ringsky = float(skyim.mean())
                        ringsigma = 1e30
                    else:
                        bin_width = sigma * 3 * 2 / 50.0
                        finite_mask = np.isfinite(skyim)
                        skyim = skyim[finite_mask]  # remove NAN and INF
                        num_bins = int((skyim.max() - skyim.min()) / bin_width)
                        y, bin_edges = np.histogram(skyim, bins=num_bins)
                        x = (bin_edges[:-1] + bin_edges[1:]) / 2.0

                        def gaussian(x, amplitude, mean, stddev):
                            return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev**2))

                        if len(x) > 3 and np.all(np.isfinite(y)):
                            if np.sum(y) >= 100 and np.count_nonzero(y) >= 4:
                                initial_guess = [y.max(), x[np.argmax(y)], sigma]
                                popt, pcov = curve_fit(gaussian, x, y, p0=initial_guess)
                                ringsky = popt[1]
                                ringsigma = np.sqrt(np.diag(pcov))[1]
                            else:
                                peak_idx = np.argmax(y)
                                ringsky = x[peak_idx]
                                bin_width = bin_edges[1] - bin_edges[0]
                                ringsigma = bin_width * 2
                                sky_flag += 64
                        else:
                            ringsky, median, ringsigma = sigma_clipped_stats(skyim, sigma=3.0, maxiters=5)
                            sky_flag += 64

                else:
                    ringsky = -1
                    ringsigma = 1e30

                del skyim, ring_empty_idx

            sl_rad[i % nslope] = radius[i]
            sl_sky[i % nslope] = ringsky
            sl_sct[i % nslope] = ringsigma

            fit = [0, 0]
            # ensure at least 'nslope' sky values has been calculated, then estimate the slope.
            if i >= (nslope - 1):
                good = np.where(sl_sct < (global_sigsky * 3))

                # ensure at least two good points to estimate the slope
                if good[0].size > 1:
                    slope, intercept = statistics.linear_regression(sl_rad[good], sl_sky[good])
                    fit = (intercept, slope)  # analogous to IDL’s [b, m]

                # calculate a sky value
                idx = np.where(sl_rad > 0)
                if idx[0].size > 3:
                    new_sky = np.mean(sl_sky[idx])
                    new_sky_sig = np.sqrt(np.sum(sl_sct[idx] ** 2)) / (idx[0].size)
                    if not np.isfinite(new_sky_sig):
                        new_sky_sig = 1e30
                    min_sky_flag0 = 0
                else:
                    # too few measurements -> take value from SExtractor
                    min_sky_flag0 = 8
                    new_sky = global_sky
                    new_sky_sig = global_sigsky

                if (new_sky < min_sky) and (new_sky_sig < 1e20):
                    min_sky_flag = min_sky_flag0
                    min_sky = new_sky
                    if idx[0].size > 0:
                        min_sky_rad = float(np.mean(sl_rad[idx]))
                    else:
                        min_sky_rad = radius[i]

                    min_sky_sig = new_sky_sig

                if fit[1] > 0 and slope_change:
                    break
                if fit[1] > 0 and last_slope < 0:
                    slope_change += 1
                last_slope = fit[1]

            if radius[i] > max_rad:
                sky_flag += 4
                break

        # loop over radii done==================

        idx = np.where((sl_rad > 0) & (sl_sct < 1e20))
        if idx[0].size > 3:
            new_sky = np.mean(sl_sky[idx])
            new_sky_sig = np.sqrt(np.sum(sl_sct[idx] ** 2)) / (idx[0].size)
            sky_flag0 = 0
        else:
            sky_flag0 = 8
            new_sky = global_sky
            new_sky_sig = global_sigsky
            fit = [0, 0]

        if idx[0].size > 0:
            sky_rad = float(np.mean(sl_rad[idx]))
        else:
            sky_rad = radius[max(0, i - 1)]

        if new_sky > min_sky:
            sky_flag0 = min_sky_flag
            new_sky = min_sky
            sky_rad = min_sky_rad
            new_sky_sig = min_sky_sig

        sky_flag += sky_flag0

        with open(skyfile, "w") as f:
            line = f"{new_sky} {new_sky_sig} {sky_rad} {cat[obj_idx]['mag_auto']} {sky_flag}\n"
            f.write(line)
