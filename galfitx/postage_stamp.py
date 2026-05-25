"""
Identifier:     galfitx/postage_stamp.py
Name:           postage_stamp.py
Description:    create stamp file and image cutout
Author:         Chao Ma
Created:        2026-01-19
Modified-History:
    2026-01-19, Chao Ma, created
"""

import os
import numpy as np
from astropy.io import fits, ascii
from astropy.wcs import WCS
from typing import Optional, List


def create_stamp_file(
    image_name: str, catalog_name: str, sizefac: float = 2.5, outfile: str = "stamps", pixel_scale: float = 0.03
) -> None:
    """
    Generate a stamp file containing bounding‑box coordinates for each object in the catalog.

    The function reads a FITS image to obtain its dimensions, then reads an ASCII catalog
    with object parameters. For each object, it computes a rectangular stamp region based on
    its size, ellipticity, and orientation, ensuring the region stays within the image bounds.
    The results are written to an output text file with one line per object.

    Parameters
    ----------
    image_name : str
        Path to the FITS image file (detection image). Used only to get the image size.
    sizefac : float, optional
        Scaling factor for the stamp size relative to the object's characteristic radius.
        Default is 2.5.
    outfile : str, optional
        Name of the output text file. Default is "stamps".
    pixel_scale : float, optional
        Pixel scale of the detection image in arcsec/pixel. This value is simply
        written into the output file for reference. Default is 0.03.

    Returns
    -------
    None
        The function writes the stamp information to `outfile` and does not return any value.

    Notes
    -----
    The stamp bounding box is computed as follows:
        - The object's effective radius `rad` = semimajor_sigma * kron_radius.
        - Two trial extents along x and y are computed using orientation and ellipticity.
        - The longer extent is assigned to the major axis direction based on the orientation angle.
        - The final box is centered on (xcentroid, ycentroid) and clipped to the image boundaries.
    The output format is:
        ID   xcentroid ycentroid ra dec  xlo xhi ylo yhi pixel_scale
    """

    # get image size
    header = fits.open(image_name)[0].header
    nx = header["NAXIS1"]
    ny = header["NAXIS2"]

    # read catalog
    cat = ascii.read(catalog_name)
    id = cat["label"]
    theta_image = cat["orientation"] * np.pi / 180  # convert degrees to radian
    rad = cat["semimajor_sigma"] * cat["kron_radius"]
    ellipticity = cat["ellipticity"]
    x_image = cat["xcentroid"] + 1      ## start from 1?
    y_image = cat["ycentroid"] + 1      ## start from 1? 

    ra_l = cat["ra"]
    dec_l = cat["dec"]

    with open(outfile, "w") as f:
        # generate stamp bounding-box edges: xlo, xhi, ylo, yhi
        for i in range(len(cat)):
            xfac = sizefac * rad[i] * (abs(np.sin(theta_image[i])) + (1 - ellipticity[i]) * abs(np.cos(theta_image[i])))
            yfac = sizefac * rad[i] * (abs(np.cos(theta_image[i])) + (1 - ellipticity[i]) * abs(np.sin(theta_image[i])))
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

            xlo = round(x_image[i]) - round(xfac)
            # if xlo < 0:
            #     xlo = 0
            xhi = round(x_image[i]) + round(xfac)
            # if xhi > nx - 1:
            #     xhi = nx - 1

            ylo = round(y_image[i]) - round(yfac)
            # if ylo < 0:
            #     ylo = 0
            yhi = round(y_image[i]) + round(yfac)
            # if yhi > ny - 1:
            #     yhi = ny - 1



            ### all xcen, ycen, xlo, xhi, ylo, yhi start from 1
            ### subtract by 1 and clip to boundary values before you apply them to a python arr. 
            line = (
                f"{id[i]:6d}  "
                f"{x_image[i]:8.3f}  {y_image[i]:8.3f}  "
                f"{ra_l[i]:8.10f}  {dec_l[i]:8.10f}  "
                f"{xlo:6d}  {xhi:6d}  {ylo:6d} {yhi:6d}   " + f"{pixel_scale}" + "\n"
            )
            f.write(line)


def cut_stamps(
    image_name: str,
    outdir: str,
    label: str,
    stampfile: str = "stamps",
    cut_list: Optional[List[int]] = None,
    ps: float = 0.03,
) -> None:
    """
    Cut stamps (sub-images) from an astronomical FITS image

    Parameters
    ----------
    image_name : str
        Input FITS image filename
    outdir : str
        Output directory for saving cutout stamps
    pre : str, optional
        Prefix for output filenames. Default is "".
    post : str, optional
        Suffix for output filenames. Default is "".
    stampfile : str, optional
        Table file containing stamp positions (default: 'stamps')
    cut_list : list of int, optional
        List specifying which sources to cut (1=cut, 0=skip). If None, all sources are cut.
    ps : float, optional
        Pixel scale of input image in arcsec/pixel (default: 0.03)
    """

    # If the directory does not exist, create it
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    hdu = fits.open(image_name)[0]

    wcs = WCS(hdu)

    data = hdu.data
    header = hdu.header
    ny, nx = data.shape

    stamps = ascii.read(stampfile)

    # By default, cut all sources if cut_list=None
    if cut_list is None:
        cut_list = [1] * len(stamps)

    for i in range(len(stamps)):
        
    # for i in [1]:    
        # Determine whether to cut this source: 0=skip, 1=cut
        if cut_list[i] == 1:
            # Extract basic source information
            id = stamps.columns[0][i]
            # if idc!=40:
            #     continue
        
        
        
            #### Remember, all these values are starting from 1, need to -1 when applied to extract poststamps in numpy arr. 
            ximg = stamps.columns[1][i]
            yimg = stamps.columns[2][i]

            ra = stamps.columns[3][i]
            dec = stamps.columns[4][i]

            # define the pixel boundaries inclusive
            # Extract cutout region boundaries
            xlo = stamps.columns[5][i]
            xhi = stamps.columns[6][i]
            ylo = stamps.columns[7][i]
            yhi = stamps.columns[8][i]
            pixel_scale = stamps.columns[9][i]
            # slice the image array

            print('cut stamps0',image_name, xlo,xhi, ylo, yhi)


            # x0, y0 are in unit of image
            x0, y0 = wcs.all_world2pix(ra, dec, 1)
            print('x0, y0', x0, y0)
            xlo1 = int(round(x0 - (ximg - xlo) * pixel_scale / ps))-1  # convert from original pixel scale.
            xhi1 = int(round((xhi - ximg) * pixel_scale / ps + x0))-1  # convert from original pixel scale.
            ylo1 = int(round(y0 - (yimg - ylo) * pixel_scale / ps))-1  # convert from original pixel scale.
            yhi1 = int(round((yhi - yimg) * pixel_scale / ps + y0))-1  # convert from original pixel scale.

            # print(yhi, yimg, pixel_scale, ps, y0)

            print('cut stamps1, match pixelscale', image_name, xlo1,xhi1, ylo1, yhi1)

            # if object not in image
            if xhi1 < 0 or yhi1 < 0 or xlo1 > nx or ylo1 > ny:
                print(f"ID {id}: skipped (outside)")
                continue

            # calculate overlap region
            xlo2 = max(xlo1, 0)
            xhi2 = min(xhi1, nx -1)
            ylo2 = max(ylo1, 0)
            yhi2 = min(yhi1, ny -1)

        
            stamp_nx = xhi2 - xlo2 
            stamp_ny = yhi2 - ylo2 
            stamp_data = np.zeros((stamp_ny, stamp_nx), dtype=data.dtype)



            stamp_data = data[ylo2: yhi2 + 1, xlo2: xhi2 + 1]
            
            
    
            # update cutout header
            stamp_header = header.copy()
            stamp_header["NAXIS1"] = stamp_nx
            stamp_header["NAXIS2"] = stamp_ny
            stamp_header["CRPIX1"] -= xlo2
            stamp_header["CRPIX2"] -= ylo2

            # Add custom keywords to FITS header
            stamp_header.set("OBJ_ID", id, "Object identifier")
            stamp_header.set("RA", ra, "[deg] Right ascension of object")
            stamp_header.set("DEC", dec, "[deg] Declination of object")
            stamp_header.set("ORG_X", ximg, "Original X coordinate in input image")
            stamp_header.set("ORG_Y", yimg, "Original Y coordinate in input image")
            stamp_header.set("CUT_XMIN", xlo2, "Cutout X min in input image (0-based)")
            stamp_header.set("CUT_XMAX", xhi2, "Cutout X max in input image (0-based)")
            stamp_header.set("CUT_YMIN", ylo2, "Cutout Y min in input image (0-based)")
            stamp_header.set("CUT_YMAX", yhi2, "Cutout Y max in input image (0-based)")

            # write cutout
            fitsname = f"obj{id}_{label}sci.fits"
            outname = os.path.join(outdir, fitsname)
            fits.writeto(outname, stamp_data, stamp_header, overwrite=True)