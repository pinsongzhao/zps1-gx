"""
Identifier:     galfitx/csst_pipe.py
Name:           csst_pipe.py
Description:    convert table to fits for tomel
Author:         Chao Ma
Created:        2026-01-19
Modified-History:
    2026-01-19, Chao Ma, created
"""

from csst_dadel.utils import table_to_hdu
from astropy.table import Table, Column
import numpy as np


def tab2fits(outtab, output="output_cat.fits"):
    """
    Convert output table to FITS file.

    Parameters
    ----------
    outtab : Table
         catalog table from source detection
    output : str, optional
        Output FITS filename (default: 'output_cat.fits')

    Returns
    -------
    None
        Writes FITS file to disk
    """
    columns = []

    # Dictionary for descriptions
    desc_dict = {
        "label": "Source label/ID",
        "xcentroid": "X centroid position",
        "ycentroid": "Y centroid position",
        "bbox_xmin": "The minimum x pixel index within the minimal bounding box containing the source segment.",
        "bbox_xmax": "The maximum x pixel index within the minimal bounding box containing the source segment.",
        "bbox_ymin": "The minimum y pixel index within the minimal bounding box containing the source segment.",
        "bbox_ymax": "The maximum y pixel index within the minimal bounding box containing the source segment.",
        "area": "The total unmasked area of the source",
        "semimajor_sigma": "1-sigma along semimajor axis",
        "semiminor_sigma": "1-sigma along semiminor axis",
        "orientation": "Position angle (degree)",
        "eccentricity": "Eccentricity (0=circle, 1=line)",
        "min_value": "minimum pixel value within the source segment",
        "max_value": "maximum pixel value within the source segment",
        "local_background": "Local background value (per pixel) estimated using a rectangular annulus aperture around the source",
        "segment_flux": "sum of the unmasked data values within the source segment",
        "segment_fluxerr": "Error in segment flux",
        "kron_flux": "The flux in the Kron aperture",
        "kron_fluxerr": "Error in Kron flux",
        "background_centroid": "Background at centroid",
        "elongation": "The ratio of the semimajor and semiminor axes.",
        "ellipticity": "Ellipticity (1 - b/a)",
        "fwhm": "Full width at half maximum",
        "kron_radius": "Kron radius",
        "flux_radius": "half-light radius",
        "ra": "Right ascension (J2000)",
        "dec": "Declination (J2000)",
        "cxx": "Second moment xx",
        "cxy": "Second moment xy",
        "cyy": "Second moment yy",
        "gini": "Gini coefficient",
        "segment_area": "Area of segmentation region",
        "mag_auto": "Automatic aperture magnitude",
    }

    # Dictionary for units - EXACTLY following screenshot pattern:
    # - Columns with units: unit='unit_string' (e.g., 'pix', 'deg', 'mag')
    # - Unitless columns: NOT in this dictionary (will get unit='' by default)
    unit_dict = {
        "xcentroid": "pix",
        "ycentroid": "pix",
        "area": "pix^2",
        "semimajor_sigma": "pix",
        "semiminor_sigma": "pix",
        "orientation": "deg",
        "min_value": "counts",
        "max_value": "counts",
        "local_background": "counts",
        "segment_flux": "counts",
        "segment_fluxerr": "counts",
        "kron_flux": "counts",
        "kron_fluxerr": "counts",
        "background_centroid": "counts",
        "fwhm": "pix",
        "flux_radius": "pix",
        "ra": "deg",
        "dec": "deg",
        "cxx": "pix^-2",
        "cxy": "pix^-2",
        "cyy": "pix^-2",
        "segment_area": "pix^2",
        "mag_auto": "mag",
        # All other columns will get unit='' (empty string)
    }

    for col_name in outtab.colnames:
        # Get data and dtype from existing column
        data = outtab[col_name].data
        dtype = outtab[col_name].dtype

        # Get description and unit - EXACTLY like screenshot
        description = desc_dict.get(col_name, col_name)
        unit = unit_dict.get(col_name, "")  # Empty string for unitless columns

        # Create new column with metadata
        new_col = Column(
            name=col_name, data=data, dtype=dtype, unit=unit, description=description  # Will be '' for unitless columns
        )
        columns.append(new_col)

    # Create the table
    tab = Table(columns)

    # Convert to HDU and write to FITS
    hdu = table_to_hdu(tab)
    hdu.writeto(output, overwrite=True)
    print(f"Table with {len(columns)} columns written to {output}")
    return tab
