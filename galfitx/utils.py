"""
Identifier:     galfitx/utils.py
Name:           utils.py
Description:    some useful functions
Author:         Chao Ma
Created:        2026-01-19
Modified-History:
    2026-01-19, Chao Ma, created
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits, ascii
from astropy.visualization import simple_norm, AsinhStretch
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from typing import Optional, Tuple, List


def scaled_kron(
    outtab: Table,
    scale: float = 1.5,
    offset: float = 0,
    pixel_scale: float = 0.06,
) -> None:
    """
    Generate a DS9 region file with scaled Kron ellipses for each source.

    For each source in the input table, an ellipse is created using the
    formula:
    major_aper = (scale * semimajor_sigma * kron_radius + offset) * pixel_scale
    minor_aper = (scale * semiminor_sigma * kron_radius + offset * (1 - ellipticity)) * pixel_scale
    The position angle is adjusted by subtracting a constant offset (-49.458529°).
    The resulting ellipses are written to a file named 'scaled_kron.reg' in
    FK5 celestial coordinates, with each ellipse labelled by the source ID.

    Parameters
    ----------
    outtab : `~astropy.table.Table`
        Input source catalog. Must contain the following columns:
        'label', 'ra', 'dec', 'kron_radius', 'semimajor_sigma',
        'semiminor_sigma', 'orientation', 'ellipticity'.
    scale : float, optional
        Scaling factor applied to the Kron radius and sigma axes. Default 1.5.
    offset : float, optional
        Additional offset (in arcseconds) added to the major axis, and
        scaled by (1 - ellipticity) for the minor axis. Default 0.0.
    pixel_scale : float, optional
        Pixel scale in arcseconds per pixel, used to convert pixel‑based
        sizes to arcseconds. Default 0.06.

    Returns
    -------
    None
        The region file is written to the current working directory.
    """

    id = outtab["label"]
    ra = outtab["ra"]
    dec = outtab["dec"]
    kron_radius = np.array(outtab["kron_radius"])
    A = np.array(outtab["semimajor_sigma"])
    B = np.array(outtab["semiminor_sigma"])
    pa = np.array(outtab["orientation"]) - 49.458529
    ell = outtab["ellipticity"]
    major_aper = (scale * A * kron_radius + offset) * pixel_scale
    minor_aper = (scale * B * kron_radius + offset * (1 - ell)) * pixel_scale

    header_reg = 'global color=yellow font="helvetica 10 normal" select=1 edit=1 move=1 delete=1 include=1 fixed=0 source=1 \nfk5'

    output = []
    for i in range(len(id)):
        ellipse = 'ellipse({},{},{}",{}",{}) #text={{{}}}'.format(
            ra[i], dec[i], major_aper[i], minor_aper[i], pa[i], id[i]
        )
        output.append(ellipse)
    np.savetxt("scaled_kron.reg", output, fmt="%s", header=header_reg, comments="")


def combine_models(
    original_fits: str, stamp_catalog: str, galfit_dir: str, output_model_file: str, background_value: float = 41
) -> None:
    """
    Combine individual GALFIT model components into a full model image.

    For each source listed in the stamp catalog, the function reads the
    corresponding GALFIT output FITS file (assumed to be located in
    `galfit_dir` with name `output<ID>.fits`) and adds its model component
    (extension 4) to the appropriate region of the final model image. The
    combined model image is then saved to `output_model_file`. All model
    components are added onto a base image initialised with the given
    `background_value`.

    Parameters
    ----------
    original_fits : str
        Path to the original FITS image. Its header and data shape are used
        for the output model.
    stamp_catalog : str
        Path to an ASCII catalog containing stamp coordinates. Expected columns:
        ID, (unused?), x1, x2, y1, y2 (zero‑based inclusive pixel indices).
        The catalog is read with `np.loadtxt`; it must have at least 7 columns.
    galfit_dir : str
        Directory containing the GALFIT output files. The function expects
        files named `output<ID>.fits` where `<ID>` is the integer identifier
        from the stamp catalog.
    output_model_file : str
        Path where the combined model FITS image will be saved.
    background_value : float, optional
        Value used to initialise the model image (added to every pixel).
        This value is also subtracted from the original data to create a
        residual? (Not done in this function). Default is 41.

    Returns
    -------
    None
        The combined model image is written to `output_model_file`.

    Notes
    -----
    - The function assumes that the GALFIT model is stored in the 5th
      extension (index 4) of the FITS file.
    - If a component file does not exist, a warning is printed and the
      source is skipped.
    - The model image is built with the same data type as the original image.
    - The output uses the header from `original_fits`; no WCS or other
      metadata is modified.
    """
    # Read original data and header
    with fits.open(original_fits) as hdul:
        data0 = hdul[0].data
        header = hdul[0].header

    # Create base model image initialized with background value
    model_data = np.full(data0.shape, background_value, dtype=data0.dtype)

    # Read stamp coordinates
    stamp = np.loadtxt(stamp_catalog)
    id = stamp[:, 0].astype(np.int64)
    x1 = stamp[:, 3].astype(int)
    x2 = stamp[:, 4].astype(int)
    y1 = stamp[:, 5].astype(int)
    y2 = stamp[:, 6].astype(int)

    # Add each GALFIT component to the model
    for i in range(len(id)):

        comp_path = f"{galfit_dir}output{id[i]}.fits"

        if os.path.exists(comp_path):
            with fits.open(comp_path) as comp_hdul:
                comp_data = comp_hdul[4].data

            # Convert component data to match model's data type
            comp_data = comp_data.astype(model_data.dtype, copy=False)

            # Add component to its position in the model
            model_data[y1[i]: y2[i] + 1, x1[i]: x2[i] + 1] += comp_data

        else:
            print(f"{comp_path} does not exist")

    # Save combined model
    fits.writeto(output_model_file, model_data, header=header, overwrite=True)


def plot_oneband_example(
    id: int,
    fits_dir: str = "./galfit/",
    fits_prename: str = "output",
    vmin: float = -1615.9,
    vmax: float = 13771.5,
    add_scaled_kron: bool = True,
    catalog_name: str = "./sex/outcat",
    scale: float = 1.1,
    offset: float = 4,
) -> None:
    """
    Plot the GALFIT fitting result for a single source (primary) in one band.

    The function displays four panels:
    1. Original data cutout with optional scaled Kron ellipses overlaid.
    2. Best‑fit model image.
    3. Residual image.
    4. Mask image.
    If `add_scaled_kron` is True, ellipses from the detection catalog are drawn
    on the data and mask panels, and source IDs are labelled.

    Parameters
    ----------
    id : int
        Source identifier (used to construct the GALFIT output filename and mask).
    fits_dir : str, optional
        Directory containing the GALFIT output files. Default "./galfit/".
    fits_prename : str, optional
        Prefix of the GALFIT output FITS file (before the ID). Default "output".
    vmin : float, optional
        Minimum display value for the image stretch. Default -1615.9.
    vmax : float, optional
        Maximum display value for the image stretch. Default 13771.5.
    add_scaled_kron : bool, optional
        If True, overlay Kron ellipses and source IDs from the detection catalog.
        Default True.
    catalog_name : str, optional
        Path to the detection catalog (ASCII, e.g., SExtractor output). Required
        columns: 'label', 'ra', 'dec', 'kron_radius', 'ellipticity',
        'semimajor_sigma', 'semiminor_sigma', 'orientation', 'mag_auto',
        'flux_radius'. Default "./sex/outcat".
    scale : float, optional
        Scaling factor for the Kron ellipse size (used when `add_scaled_kron=True`).
        Default 1.1.
    offset : float, optional
        Additional offset (in pixels) added to the ellipse axes. Default 4.

    Returns
    -------
    None
        The function displays the plot using `plt.show()` and does not save it.
    """
    fits_path = fits_dir + fits_prename + str(id) + ".fits"
    hdu = fits.open(fits_path)
    data = hdu[0].data
    model = hdu[1].data
    resi = hdu[2].data
    header = hdu[0].header
    wcs = WCS(header)
    norm = simple_norm(data, stretch="asinh", min_cut=vmin, max_cut=vmax)
    shape = data.shape

    ax1 = plt.subplot(141)
    ax1.imshow(data, cmap="gray", norm=norm, origin="lower")

    if add_scaled_kron:

        catalog = ascii.read(catalog_name)
        corners = np.array([[0, 0], [0, shape[1] - 1], [shape[0] - 1, 0], [shape[0] - 1, shape[1] - 1]])

        # Get SkyCoord object and extract RA/Dec
        skycoord_corners = wcs.pixel_to_world(corners[:, 1], corners[:, 0])
        ra_corners = np.array([c.ra.deg for c in skycoord_corners])
        dec_corners = np.array([c.dec.deg for c in skycoord_corners])
        ra_min, ra_max = min(ra_corners), max(ra_corners)
        dec_min, dec_max = min(dec_corners), max(dec_corners)

        # Get pixel scale in degrees (assuming square pixels)
        # Try to get pixel scale from CDELT or CD matrix
        if "CD1_1" in header:
            deg_per_pixel = np.sqrt(header["CD1_1"] ** 2 + header["CD1_2"] ** 2)
        elif "CDELT1" in header:
            deg_per_pixel = abs(header["CDELT1"])
        else:
            # Estimate from WCS if standard keywords missing
            deg_per_pixel = np.mean(np.abs(wcs.proj_plane_pixel_scales().to(u.deg).value))

        # Calculate maximum ellipse radius in the catalog
        max_ellipse_radius_deg = max(catalog["kron_radius"] * catalog["semimajor_sigma"]) * deg_per_pixel

        # Expand search area to include potential overlapping sources
        buffer = 1.5 * max_ellipse_radius_deg  # 50% buffer
        ra_min_exp = ra_min - buffer
        ra_max_exp = ra_max + buffer
        dec_min_exp = dec_min - buffer
        dec_max_exp = dec_max + buffer

        # Pre-filter sources within expanded area
        in_expanded = (
            (catalog["ra"] >= ra_min_exp)
            & (catalog["ra"] <= ra_max_exp)
            & (catalog["dec"] >= dec_min_exp)
            & (catalog["dec"] <= dec_max_exp)
        )
        filtered_catalog = catalog[in_expanded]
        # print(f"Found {len(filtered_catalog)} potential sources near cutout")
        x_cutout, y_cutout = wcs.world_to_pixel(SkyCoord(filtered_catalog["ra"], filtered_catalog["dec"], unit="deg"))
        x_cutout = x_cutout.astype(float)
        y_cutout = y_cutout.astype(float)

        # Plot Kron ellipses for all sources in expanded area
        for i, source in enumerate(filtered_catalog):
            x = x_cutout[i]
            y = y_cutout[i]

            # Calculate ellipse dimensions
            kron_radius = source["kron_radius"]
            ell = source["ellipticity"]
            width = 2 * (scale * kron_radius * source["semimajor_sigma"] + offset)
            height = 2 * (scale * kron_radius * source["semiminor_sigma"] + offset * (1 - ell))
            orientation = source["orientation"]

            # create ellipse
            ellipse = Ellipse(
                (x, y),
                width,
                height,
                angle=orientation,
                edgecolor="lime",
                facecolor="none",
                lw=1.0,
                alpha=0.8,
                transform=ax1.transData,
            )
            ax1.add_patch(ellipse)

            # Add source ID label at center if inside cutout
            if 0 <= x < shape[1] and 0 <= y < shape[0]:
                source_id = source["label"]
                ax1.text(
                    x,
                    y + 5,
                    str(source_id),
                    color="red",
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    alpha=0.8,
                )

        # extract source extraction parameter
        title = []
        title.append(f"mag = {catalog['mag_auto'][id-1]:.3f}")
        title.append(f"re = {catalog['flux_radius'][id-1]:.3f}")
        title.append(f"q = {1 - catalog['ellipticity'][id-1]:.3f}")
        title_str = "\n".join(title)
        ax1.set_title(title_str, fontsize=9, pad=8)

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ax2 = plt.subplot(142)
    # Extract Sersic parameters of primary sources
    header1 = hdu[1].header
    sersic_params = {}
    param_keys = ["2_MAG", "2_RE", "2_N", "2_AR"]
    for key in param_keys:
        if key in header1:
            value_str = header1[key]
        # Remove brackets if present
        if value_str.startswith("[") and value_str.endswith("]"):
            value_str = value_str[1:-1]
        # Remove asterisks if present
        value_str = value_str.replace("*", "")
        parts = value_str.split()
        sersic_params[key] = parts[0]

    title = []
    title.append(f"mag = {sersic_params['2_MAG']}")
    title.append(f"re = {sersic_params['2_RE']}")
    title.append(f"n = {sersic_params['2_N']}")
    title.append(f"q = {sersic_params['2_AR']}")

    title_str = "\n".join(title)
    ax2.set_title(title_str, fontsize=9, pad=8)
    ax2.imshow(model, cmap="gray", norm=norm, origin="lower")
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    ax3 = plt.subplot(143)
    ax3.imshow(resi, cmap="gray", norm=norm, origin="lower")
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    ax4 = plt.subplot(144)
    mask_path = fits_dir + "mask" + str(id) + ".fits"
    mask = fits.open(mask_path)[0].data
    ax4.imshow(mask, cmap="gray", origin="lower")

    if add_scaled_kron:
        for i, source in enumerate(filtered_catalog):
            x = x_cutout[i]
            y = y_cutout[i]
            kron_radius = source["kron_radius"]
            ell = source["ellipticity"]
            width = 2 * (scale * kron_radius * source["semimajor_sigma"] + offset)
            height = 2 * (scale * kron_radius * source["semiminor_sigma"] + offset * (1 - ell))
            orientation = source["orientation"]

            ellipse = Ellipse(
                (x, y),
                width,
                height,
                angle=orientation,
                edgecolor="lime",
                facecolor="none",
                lw=1.0,
                alpha=0.8,
                transform=ax4.transData,
            )

            ax4.add_patch(ellipse)

    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)

    plt.tight_layout()
    # plt.savefig('aaa.png',dpi=150)
    plt.show()


def create_rgb_image(
    red_file: str,
    green_file: str,
    blue_file: str,
    vmin: float = -1615.9,
    vmax: float = 13771.5,
    color_factors: Tuple[float, float, float] = (1.0, 0.8, 1.0),
    a_stretch: float = 0.1,
    show: bool = True,
    dpi: int = 200,
) -> Tuple[np.ndarray, plt.Figure]:
    """
    Create an RGB composite image from three FITS files with customizable processing.

    Each band is clipped to [vmin, vmax], normalised, and then stretched using
    an asinh transformation (`AsinhStretch`). Colour scaling factors are applied
    per band, and the three channels are stacked to form an RGB image.

    Parameters
    ----------
    red_file : str
        Path to the FITS file for the red channel.
    green_file : str
        Path to the FITS file for the green channel.
    blue_file : str
        Path to the FITS file for the blue channel.
    vmin : float, optional
        Minimum value for clipping (default -1615.9).
    vmax : float, optional
        Maximum value for clipping (default 13771.5).
    color_factors : tuple of float, optional
        Scaling factors for (R, G, B) channels (default (1.0, 0.8, 1.0)).
    a_stretch : float, optional
        Asinh stretch parameter (default 0.1).
    show : bool, optional
        If True, display the image using `plt.show()`. Default True.
    dpi : int, optional
        Resolution (dots per inch) for the figure (default 200).

    Returns
    -------
    rgb_image : `~numpy.ndarray`
        3‑dimensional array of shape (ny, nx, 3) containing the RGB image.
    fig : `~matplotlib.figure.Figure`
        The matplotlib figure object used for display (can be used for further
        customisation or saving).
    """
    r_data = fits.open(red_file)[0].data
    g_data = fits.open(green_file)[0].data
    b_data = fits.open(blue_file)[0].data

    def prepare_band(data: np.ndarray, vmin: float, vmax: float, a: float) -> np.ndarray:
        """
        Process single band with clipping and Asinh stretching
        Equivalent to: simple_norm(data, stretch='asinh', vmin=vmin, vmax=vmax)
        """
        data_clipped = np.clip(data, vmin, vmax)
        data_normalized = (data_clipped - vmin) / (vmax - vmin)
        stretched_data = AsinhStretch(a=a)(data_normalized)
        return stretched_data

    # Process each band
    r_processed = prepare_band(r_data, vmin, vmax, a_stretch)
    g_processed = prepare_band(g_data, vmin, vmax, a_stretch)
    b_processed = prepare_band(b_data, vmin, vmax, a_stretch)

    # Apply color scaling
    r_scaled = color_factors[0] * r_processed
    g_scaled = color_factors[1] * g_processed
    b_scaled = color_factors[2] * b_processed

    # Create RGB composite
    rgb_image = np.dstack((r_scaled, g_scaled, b_scaled))
    plt.imshow(rgb_image, origin="lower")
    plt.show()

def combine_catalogs(
    filter_list: List[str],
    sex_cat: str,
    gs_dir: str,
    eazy_dir: str,
    gssed_dir: str,
    output_file: str = 'combined_catalog.fits'
)-> None:
    """
    Combine catalogs from the five pipeline stages:
    1. Source detection
    2. Pure imaging fitting (galfits, no SED)
    3. Photometric redshift (EAZy grouped outputs)
    4. Image+SED fitting (galfits with SED)
    
    The reference catalog is the source detection catalog (sex_cat). All other
    results are merged object identifier (label column). 
    Missing values are filled with -999.0.    
    
    Parameters
    ----------
    filter_list : list of str
        List of filter names, e.g. ['nircam_f444w', ...]
    sex_cat : str
        Path to source detection catalog (ASCII, e.g. './sex/outcat').
        Must contain a column named 'label' that serves as the object identifier.        
    gs_dir : str
        Directory containing pure imaging fitting .gssummary files (e.g. './galfits/pureImage').
        Files are expected to be named 'obj<label>.gssummary'.
    eazy_dir : str
        Root directory containing EAZy grouped outputs. Inside, subdirectories 
        named by group IDs (numeric names) contain a 'photz.zout' file with 
        columns 'id', 'z_peak', 'z_spec'.        
    gssed_dir : str
        Directory containing SED fitting .gssummary files (e.g. './galfits/image_sed').
        Files are expected to be named 'obj<label>.gssummary'.
    output_file : str, optional
        Filename for the output combined FITS catalog.
        Default is 'combined_catalog.fits'.
    
    Returns
    -------
    None
        The combined catalog is written to disk as a FITS file.
    """
    missing_value = -999.0
    
    # claim galfits pure imaging fitting parameters
    mag_params = [f'Mag_obj0_{filter}' for filter in filter_list]
    geo_params = ['obj0_Re', 'obj0_n', 'obj0_ang', 'obj0_axrat']
    sersic_params = mag_params + geo_params
        
    # claim photz paramters
    photz_params = ['z_peak', 'z_spec']
    
    # claim physical parameters
    phy_params = ['obj0_Z_value', 'obj0_Av_value', 'logM_obj0', 'AVbump_obj0', 'obj0_f_cont_bin1', 'obj0_f_cont_bin2', 'obj0_f_cont_bin3', 'obj0_f_cont_bin4', 'obj0_f_cont_bin5', 'logsfr_obj0']

    
    # ----------------------------------------------------------------------
    # 1. Source detection catalog (reference)
    # ----------------------------------------------------------------------
    # Create the combined table, starting with all columns from the reference catalog
    table_ref = Table.read(sex_cat, format='ascii')
    combined = Table()
    for col in table_ref.colnames:
        combined[col] = table_ref[col]
    
    # ----------------------------------------------------------------------
    # 2. Pure imaging fitting (galfits, no SED)
    # ----------------------------------------------------------------------
    # Initialize Sérsic paramters columns with missing value
    for col in sersic_params:
        combined[col] = [missing_value] * len(combined)
        
    # ----------------------------------------------------------------------
    # Loop over objects, read .gssummary if it exists
    # ----------------------------------------------------------------------
    for i, obj_id in enumerate(combined['label']):
        gssummary_file = os.path.join(gs_dir, f'obj{obj_id}.gssummary')
        if not os.path.exists(gssummary_file):
            continue # keep missing_value -999, if gssummary file not exits
    
        # Read the file and build a simple dictionary
        result = ascii.read(gssummary_file)
        param_dict = {}
        for row in result:
            param_dict[row['pname']] = row['best_value']
    
        # Fill parameters that are present
        for col in sersic_params:
            if col in param_dict:
                combined[col][i] = param_dict[col]
            # else remains missing_value

    # ----------------------------------------------------------------------
    # 3. Photometric redshifts (grouped EAZy outputs)
    # ----------------------------------------------------------------------
    combined['z_peak'] = [missing_value] * len(combined)
    combined['z_spec'] = [missing_value] * len(combined)

    id_all = []
    photz_all = []
    specz_all = []

    for entry in os.listdir(eazy_dir):
        subdir = os.path.join(eazy_dir, entry)
        if not os.path.isdir(subdir):
            continue
        if not entry.isdigit():
            continue

        zout_file = os.path.join(subdir, "photz.zout")
        if not os.path.isfile(zout_file):
            continue

        table = ascii.read(zout_file)
        id_arr = table['id'].astype(int)
        photz = table['z_peak']
        specz = table['z_spec']

        id_all = np.append(id_all, id_arr).astype(int)
        photz_all = np.append(photz_all, photz)
        specz_all = np.append(specz_all, specz)

    if len(id_all) > 0:
        sort_idx = np.argsort(id_all)
        id_all = id_all[sort_idx]
        photz_all = photz_all[sort_idx]
        specz_all = specz_all[sort_idx]

        for i, obj_id in enumerate(combined['label']):
            idx = np.searchsorted(id_all, obj_id)
            if idx < len(id_all) and id_all[idx] == obj_id:
                combined['z_peak'][i] = photz_all[idx]
                combined['z_spec'][i] = specz_all[idx]
            # else stays missing_value
    # else: if no data was found, all values remain missing_value (already -999.0)
    
    
    # ----------------------------------------------------------------------
    # 4. Image+SED fitting (galfits with SED)
    # ----------------------------------------------------------------------
    # Initialize physical parameters (from image+SED fitting) with missing value
    for col in phy_params:
        combined[col] = [missing_value] * len(combined)

    # ----------------------------------------------------------------------
    # Loop over objects, read .gssummary if it exists
    # ----------------------------------------------------------------------
    for i, obj_id in enumerate(combined['label']):
        gssummary_file = os.path.join(gssed_dir, f'obj{obj_id}.gssummary')
        if not os.path.exists(gssummary_file):
            continue # keep missing_value -999, if gssummary file not exits
    
        # Read the file and build a simple dictionary
        result = ascii.read(gssummary_file)
        param_dict = {}
        for row in result:
            param_dict[row['pname']] = row['best_value']
    
        # Fill parameters that are present
        for col in phy_params:
            if col in param_dict:
                combined[col][i] = param_dict[col]
            # else remains missing_value

    # ----------------------------------------------------------------------
    # Save the combined catalog
    # ----------------------------------------------------------------------
    combined.write(output_file, format='fits', overwrite=True)
    print(f"Saved combined catalog with {len(combined)} rows and {len(combined.colnames)} columns")
