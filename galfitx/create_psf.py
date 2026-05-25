import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.visualization import simple_norm, AsinhStretch
from astropy.table import Table, Column
from astropy.stats import sigma_clipped_stats
from psfr.psfr import stack_psf
from psfr.util import oversampled2regular
from photutils.psf import fit_fwhm
from photutils.profiles import RadialProfile
from photutils.centroids import centroid_2dg
from typing import Optional, Tuple, List


def build_stacked_psf(
    sci_image: str,
    catalog_file: str,
    seg_file: str,
    output_dir: str = "./star_cutouts",
    mag_bright_limit: float = 19.0,
    mag_faint_limit: float = 21.0,
    cutout_size: int = 71,
    oversampling: int = 3,
    n_recenter: int = 10,
    num_iteration: int = 20,
    save_fits: bool = True,
    save_figures: bool = True,
    plot_overview: bool = True,
    plot_radial_profile: bool = True,
)-> None:
    """
    Build a stacked PSF from selected stars in a science image.

    Parameters
    ----------
    sci_image : str
        Path to the science FITS image.
    catalog_file : str
        Path to the SExtractor output catalogue (ASCII).
    seg_file : str
        Path to the SExtractor segmentation map.
    output_dir : str, optional
        Directory where cutouts and figures are saved.
    mag_bright_limit, mag_faint_limit : float, optional
        Magnitude limits for star selection.
    cutout_size : int, optional
        Size of the square stamp cutout (pixels).
    oversampling : int, optional
        Oversampling factor for PSF stacking.
    n_recenter : int, optional
        Number of recentering iterations during stacking.
    num_iteration : int, optional
        Number of iterations for the stacking algorithm.
    save_fits : bool, optional
        If True, save individual star cutouts and the final PSF as FITS files.
    save_figures : bool, optional
        If True, save the overview and radial profile figures as PNG files.
    plot_overview : bool, optional
        If True, generate the overview plot of all star cutouts.
    plot_radial_profile : bool, optional
        If True, generate the 1D radial profile plot.
        
    Examples
    --------
    >>> build_stacked_psf(
    ...     sci_image='sci_i.fits',
    ...     catalog_file='./sex/outcat',
    ...     seg_file='./sex/outseg.fits',
    ...     output_dir='./stars_output',
    ...     plot_overview=True,
    ...     plot_radial_profile=True,
    ...     save_fits=True,
    ...     save_figures=True,
    ... )
    17 sources satisfy stellar criteria.
    14 out of 17 stars are fully within the image and have been cut out.
    Cropped star images saved to ./stars_output/
    Overview image saved to ./stars_output/star_cutouts_overview.png
    FWHM = [1.93538863]
    Radial profile saved to ./stars_output/radial_profiles.png        
    """
    # --- Read catalogue & select stars ---
    outtab = ascii.read(catalog_file)
    outtab1 = outtab[
        (outtab['elongation'] < 1.5) &
        (outtab['class_star'] > 0.9) &
        (outtab['combined_flags'] < 2) &
        (outtab['mag_auto'] > mag_bright_limit) &
        (outtab['mag_auto'] < mag_faint_limit)
    ]
    print(f"{len(outtab1)} sources satisfy stellar criteria.")

    # --- Extract coordinates (1‑indexed) ---
    star_ids = outtab1['label']
    x_image = np.round(outtab1['xcentroid'] + 1, 3)
    y_image = np.round(outtab1['ycentroid'] + 1, 3)
    mag_auto = outtab1['mag_auto']  # 保存星等信息用于颜色编码

    # --- Create output directory ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Load images ---
    sci = fits.open(sci_image)[0].data
    seg = fits.open(seg_file)[0].data
    ny, nx = sci.shape
    half = cutout_size // 2

    # --- Cutout container ---
    valid_ids = []       # stars IDs that passed the boundary check
    sci_cutout_list = [] # cutout sci image data
    seg_cutout_list = [] # cutout seg image data
    mask_list = []       # boolean mask: True = use pixel (background + target star)
    valid_coords = []    # Coordinates of valid stars (cutout centers)
    valid_mags = []      # Magnitudes of valid stars for color coding

    for idx, (xc, yc, mag) in enumerate(zip(x_image, y_image, mag_auto)):
        # Round to nearest integer pixel center
        xc_int = round(xc)
        yc_int = round(yc)
        
        # Compute slice boundaries (left-inclusive, right-exclusive)
        xmin = xc_int - half
        xmax = xc_int + half + 1
        ymin = yc_int - half
        ymax = yc_int + half + 1
        
        # Boundary check: the cutout must lie entirely within the image
        if xmin >= 0 and xmax <= nx and ymin >= 0 and ymax <= ny:
            obj_id = star_ids[idx]
            sci_cutout = sci[ymin:ymax, xmin:xmax].copy()
            seg_cutout = seg[ymin:ymax, xmin:xmax].copy()

            valid_ids.append(obj_id)
            sci_cutout_list.append(sci_cutout)
            seg_cutout_list.append(seg_cutout)
            valid_coords.append((xc, yc))
            valid_mags.append(mag)  # Save magnitude for color coding

            # mask = (seg==0) OR (seg==obj_id)  -> True = keep
            mask = (seg_cutout == 0) | (seg_cutout == obj_id)
            mask_list.append(mask.astype(int)) # mask area: the area to be used to stack
        else:
            print(f"Star ID {star_ids[idx]} at ({xc:.1f}, {yc:.1f}) too close to edge, skipped.")

    n_stars = len(valid_ids)
    print(f"{n_stars} out of {len(x_image)} stars are fully within the image and have been cut out.")

    # Save coordinates of valid stars only (those entirely within the image)
    if valid_coords:
        x_valid = np.array([c[0] for c in valid_coords])
        y_valid = np.array([c[1] for c in valid_coords])
        ids = np.array(valid_ids)
        np.savetxt('star_coordinates.txt', np.column_stack((x_valid, y_valid, ids)),
                   fmt=['%10.3f', '%10.3f', '%6d'])

    if n_stars == 0:
        raise RuntimeError("No valid stars found within the field of view.")

    # --- Save FITS cutouts (optional) ---
    if save_fits:
        for obj_id, star_data, seg_data, (xc, yc) in zip(
            valid_ids, sci_cutout_list, seg_cutout_list, valid_coords
        ):
            # Science cutout
            hdu = fits.PrimaryHDU(star_data)
            hdu.header['OBJID'] = (obj_id, 'Original ID')
            hdu.header['CUTX'] = (xc, 'Original x centroid (1-indexed)')
            hdu.header['CUTY'] = (yc, 'Original y centroid (1-indexed)')
            hdu.header['SIZE'] = (cutout_size, 'Cutout size in pixels')
            outname = os.path.join(output_dir, f"star{obj_id}.fits")
            hdu.writeto(outname, overwrite=True)

            # Segmentation cutout
            hdu_seg = fits.PrimaryHDU(seg_data.astype(np.int32))
            hdu_seg.header['OBJID'] = (obj_id, 'Original SExtractor ID')
            hdu_seg.header['CUTX'] = (xc, 'Original x centroid (1-indexed)')
            hdu_seg.header['CUTY'] = (yc, 'Original y centroid (1-indexed)')
            hdu_seg.header['SIZE'] = (cutout_size, 'Cutout size in pixels')
            outname_seg = os.path.join(output_dir, f"star{obj_id}_seg.fits")
            hdu_seg.writeto(outname_seg, overwrite=True)
        print(f"Cropped star images saved to {output_dir}/")

    # --- Overview plot (optional) ---
    if save_figures and plot_overview:
        # Determine grid size (square, enough cells for all stars)
        grid_size = int(np.ceil(np.sqrt(n_stars)))
        fig, axes = plt.subplots(grid_size, grid_size,
                                 figsize=(3*grid_size, 3*grid_size))
        if grid_size == 1:
            axes = np.array([axes])  # Make it iterable
        else:
            axes = axes.ravel()
        
        def get_norm(sci, n_min:float = 3, n_max:float = 10):
            """Compute an asinh normalisation for display."""
            _, median, std = sigma_clipped_stats(sci, sigma=3.0, maxiters=5)
            norm = simple_norm(sci, 'asinh', min_cut = median - n_min*std, max_cut = median +  n_max*std)
            return norm
        
        for idx in range(grid_size * grid_size):
            ax = axes[idx]
            if idx < n_stars:
                obj_id = valid_ids[idx]
                cutout = sci_cutout_list[idx]
                norm = get_norm(cutout)
                ax.imshow(cutout, norm=norm, origin='lower', cmap='Greys_r')
                ax.set_title(f"Star {obj_id}")
            ax.axis('off')

        plt.tight_layout()
        out_plot = os.path.join(output_dir, "star_cutouts_overview.png")
        fig.savefig(out_plot, dpi=150, bbox_inches='tight')
        plt.close(fig) # closes the figure without opening a window
        print(f"Overview image saved to {out_plot}")

    # --- Build stacked PSF ---
    result = stack_psf(
        sci_cutout_list,
        oversampling=oversampling,
        mask_list=mask_list,
        error_map_list=None,
        saturation_limit=None,
        num_iteration=num_iteration,
        n_recenter=n_recenter
    )
    psf_oversampled = result[0]
    psf = oversampled2regular(psf_oversampled, oversampling)

    fits.writeto(os.path.join(output_dir, 'psf.fits'), psf, overwrite=True)

    # --- Calculate FWHM for stacked PSF ---
    fwhm_stacked = fit_fwhm(psf)[0]
    print(f"Stacked PSF FWHM = {fwhm_stacked}")

    # --- Calculate FWHM for each input star ---
    fwhm_list = []
    for cutout, mask_avail in zip(sci_cutout_list, mask_list):
        mask_phot = (mask_avail == 0)
        fwhm_star = fit_fwhm(cutout, mask=mask_phot)[0]
        fwhm_list.append(fwhm_star)

    print(f"Input star FWHM: {fwhm_list}")

    # --- Radial profile plot (optional) ---
    if save_figures and plot_radial_profile:
        size = cutout_size
        max_radius = size // 2                # avoid nans outside the image
        radii = np.linspace(0, max_radius, int(max_radius / 1.0) + 1) # step=1.0

        # Setup colormap for magnitude
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(vmin=min(valid_mags), vmax=max(valid_mags))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        fig, ax = plt.subplots(figsize=(7, 5))
        for i, (cutout, seg_cutout, obj_id, mag) in enumerate(zip(sci_cutout_list, seg_cutout_list, valid_ids, valid_mags)):
            # mask_phot: True = mask for RadialProfile (other objects)
            mask_phot = (seg_cutout != 0) & (seg_cutout != obj_id)

            # Find peak within target segment only
            cutout_for_search = cutout.copy()
            # Set pixels not in target segment to -inf
            cutout_for_search[seg_cutout != obj_id] = -np.inf
            peak_idx = np.argmax(cutout_for_search)
            star_center = (peak_idx % cutout.shape[1], peak_idx // cutout.shape[1])

            rp = RadialProfile(cutout, star_center, radii, mask=mask_phot)
            profile = rp.profile
            peak = np.nanmax(profile)        # robust against potential nans
            if peak > 0:
                profile = profile / peak

            color = cmap(norm(mag))
            ax.plot(rp.radius, profile, color=color, alpha=0.7, linewidth=0.8)

        # Stacked PSF - find peak in stacked PSF
        peak_idx = np.argmax(psf)
        psf_center = (peak_idx % psf.shape[1], peak_idx // psf.shape[1])
        rp_psf = RadialProfile(psf, psf_center, radii)
        psf_profile = rp_psf.profile
        peak_psf = np.nanmax(psf_profile)
        if peak_psf > 0:
            psf_profile = psf_profile / peak_psf
        ax.plot(rp_psf.radius, psf_profile, color='red', linewidth=2,
                label='Stacked PSF')

        # Add colorbar for magnitude
        cbar = fig.colorbar(sm, ax=ax, label='MAG_AUTO')

        ax.set_xlabel('Radius [pix]')
        ax.set_ylabel('Normalized flux')
        ax.set_title(f'1D Radial Profiles')
        ax.legend()
        ax.set_yscale('log')
        plt.tight_layout()
        out_profile = os.path.join(output_dir, 'radial_profiles.png')
        fig.savefig(out_profile, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Radial profile saved to {out_profile}")



def build_input_ldac(
    sci_image: str,
    catalog_file: str,
    seg_file: str,
    output_ldac: str,
    mag_bright_limit: float = 19.0,
    mag_faint_limit: float = 21.0,
    cutout_size: int = 71,
    mask_value: float = -1e30,
    save_cutouts: bool = True,
    cutouts_dir: str = "./star_cutouts",
    template_file: str = "./output_assoc_temp.cat",
    fwhm_arcsec: float = 0.13,
    mag_zeropoint: float = 28.0,
    ref_pixel_scale: float = 0.074,
) -> None:
    """
    Build a PSFEx input LDAC catalog from selected stars.

    This function reads a SExtractor catalogue, selects suitable stars,
    extracts cutouts, and creates a FITS_LDAC file (with updated header
    keywords) that can be directly used by PSFEx.

    Parameters
    ----------
    sci_image : str
        Path to the science FITS image.
    catalog_file : str
        Path to the SExtractor output catalogue (ASCII format).
    seg_file : str
        Path to the SExtractor segmentation map.
    output_ldac : str
        Path for the output PSFEx LDAC catalog.
    mag_bright_limit : float, optional (default=19.0)
        Bright magnitude limit for star selection.
    mag_faint_limit : float, optional (default=21.0)
        Faint magnitude limit for star selection.
    cutout_size : int, optional (default=71)
        Size of the square cutout (in pixels). Must match the VIGNET
        dimensions expected in the template (or be smaller if the
        template VIGNET is larger; the column definition will be
        adjusted automatically).
    mask_value : float, optional (default=-1e30)
        Value used to mask neighbouring sources in the cutout.
    save_cutouts : bool, optional (default=True)
        If True, save individual star cutouts as FITS files.
    cutouts_dir : str, optional (default="./star_cutouts")
        Directory where cutout FITS files will be saved (if
        `save_cutouts` is True).
    template_file : str, optional (defalut="./output_assoc_temp.cat")
        Path to an existing SExtractor ASSOC output (LDAC template).
        The template must have the same column structure and a VIGNET
        size matching `cutout_size` (or larger, though the VIGNET column
        will be rebuilt automatically).        
    fwhm_arcsec : float, optional (default=0.13)
        Approximate FWHM of stars in arcsec, used to update the
        ``SEXSFWHM`` keyword in the LDAC IMHEAD.
    mag_zeropoint : float, optional (default=28.0)
        Magnitude zero-point, used to update the ``SEXMGZPT`` keyword.
    ref_pixel_scale : float, optional (default=0.074)
        Pixel scale in arcsec/pixel, used to update the ``SEXPXSCL``
        keyword.

    Notes
    -----
    The function also updates several other SExtractor keywords
    (``NAXIS1``, ``NAXIS2``, ``SEXBKGND``, ``SEXBKDEV``,
    ``SEXTHLD``, ``SEXATHLD``, ``SEXNFIN``, ``SEXPXSCL``,
    ``SEXSFWHM``, ``SEXMGZPT``) in the LDAC IMHEAD so that they
    match the current image and star selection.

    Examples
    --------
    >>> build_input_ldac(
    ...     sci_image='sci_i.fits',
    ...     catalog_file='./sex/outcat',
    ...     seg_file='./sex/outseg.fits',
    ...     output_ldac='psfex_input_assoc.cat',
    ...     mag_bright_limit=19.0,
    ...     mag_faint_limit=21.0,
    ...     cutout_size=71,
    ...     save_cutouts=True,
    ...     cutouts_dir='./star_cutouts',
    ...     template_file='output_assoc_temp.cat',    
    ... )
    17 source satisfy stellar criteria.
    Star ID 6 at (1648.6, 34.0) too close to edge, skipped.
    ...
    14 out of 17 stars are fully within the image and have been cut out.
    Cropped star images saved to ./star_cutouts/
    LDAC 文件已生成: psfex_input_assoc.cat
    已将 NAXIS1/2 更新为 4091, 4091
    """
    # --------------------------------------------------------------------
    # 1. Read catalogue and select stars
    # --------------------------------------------------------------------
    outtab = ascii.read(catalog_file)
    outtab1 = outtab[
        (outtab['elongation'] < 1.5) &
        (outtab['class_star'] > 0.93) &
        (outtab['combined_flags'] < 2) &
        (outtab['mag_auto'] > mag_bright_limit) &
        (outtab['mag_auto'] < mag_faint_limit)
    ]
    print(f"{len(outtab1)} source satisfy stellar criteria.")

    # --------------------------------------------------------------------
    # 2. Extract coordinates (1‑indexed)
    # --------------------------------------------------------------------
    star_ids = outtab1['label']
    x_image = np.round(outtab1['xcentroid'] + 1, 3) # 1-indexed
    y_image = np.round(outtab1['ycentroid'] + 1, 3)

    # --------------------------------------------------------------------
    # 3. Load images
    # --------------------------------------------------------------------
    sci = fits.open(sci_image)[0].data
    seg = fits.open(seg_file)[0].data
    ny, nx = sci.shape
    half = cutout_size // 2

    # --------------------------------------------------------------------
    # 4. Extract cutouts
    # --------------------------------------------------------------------
    # Containers for valid star cutouts
    valid_ids = []         # stars IDs that passed the boundary check
    sci_cutout_list = []   # cutout sci image data
    seg_cutout_list = []   # cutout seg image data
    valid_coords = []      # Coordinates of valid stars (cutout centers)

    for idx, (xc, yc) in enumerate(zip(x_image, y_image)):
        # Round to nearest integer pixel center
        xc_int = round(xc)
        yc_int = round(yc)
        
        # Compute slice boundaries (left-inclusive, right-exclusive)
        xmin = xc_int - half
        xmax = xc_int + half + 1
        ymin = yc_int - half
        ymax = yc_int + half + 1

        # Boundary check: the cutout must lie entirely within the image
        if xmin >= 0 and xmax <= nx and ymin >= 0 and ymax <= ny:
            obj_id = star_ids[idx]

            # Cut out science and segmentation arrays
            sci_cutout = sci[ymin:ymax, xmin:xmax].copy()
            seg_cutout = seg[ymin:ymax, xmin:xmax].copy()

            # Mask neighbouring sources
            # Any pixel where seg_cut is not 0 AND not equal to the target ID
            # gets replaced by 'mask_value'.
            mask = (seg_cutout != 0) & (seg_cutout != obj_id)
            sci_cutout[mask] = mask_value

            valid_ids.append(obj_id)
            sci_cutout_list.append(sci_cutout)
            seg_cutout_list.append(seg_cutout)
            valid_coords.append((xc, yc))
        else:
            print(f"Star ID {star_ids[idx]} at ({xc:.1f}, {yc:.1f}) too close to edge, skipped.")

    n_stars = len(valid_ids)
    print(f"{n_stars} out of {len(x_image)} stars are fully within the image and have been cut out.")

    # Save coordinates of valid stars only (those entirely within the image)
    if valid_coords:
        x_valid = np.array([c[0] for c in valid_coords])
        y_valid = np.array([c[1] for c in valid_coords])
        ids = np.array(valid_ids)
        np.savetxt('star_coordinates.txt', np.column_stack((x_valid, y_valid, ids)),
                   fmt=['%10.3f', '%10.3f', '%6d'])

    if n_stars == 0:
        raise RuntimeError("No valid stars found within the field of view.")

    # --------------------------------------------------------------------
    # 5. Save cutouts as individual FITS files (optional)
    # --------------------------------------------------------------------
    if save_cutouts:
        os.makedirs(cutouts_dir, exist_ok=True)
        for obj_id, star_data, seg_data, (xc, yc) in zip(
            valid_ids, sci_cutout_list, seg_cutout_list, valid_coords
        ):
            hdu = fits.PrimaryHDU(star_data)
            hdu.header['OBJID'] = (obj_id, 'Original  ID')
            hdu.header['CUTX'] = (xc, 'Original x centroid (1-indexed)')
            hdu.header['CUTY'] = (yc, 'Original y centroid (1-indexed)')
            hdu.header['SIZE'] = (cutout_size, 'Cutout size in pixels')
            outname = os.path.join(cutouts_dir, f"star{obj_id}.fits")
            hdu.writeto(outname, overwrite=True)

            # Segmentation cutout
            hdu_seg = fits.PrimaryHDU(seg_data.astype(np.int32))
            hdu_seg.header['OBJID'] = (obj_id, 'Original SExtractor ID')
            hdu_seg.header['CUTX'] = (xc, 'Original x centroid (1-indexed)')
            hdu_seg.header['CUTY'] = (yc, 'Original y centroid (1-indexed)')
            hdu_seg.header['SIZE'] = (cutout_size, 'Cutout size in pixels')
            outname_seg = os.path.join(cutouts_dir, f"star{obj_id}_seg.fits")
            hdu_seg.writeto(outname_seg, overwrite=True)
        print(f"Cropped star images saved to {cutouts_dir}/")

    # --------------------------------------------------------------------
    # 6. Prepare data arrays for the LDAC table
    # --------------------------------------------------------------------
    x_images = []
    y_images = []
    ra = []
    dec = []
    flux_radii = []
    elongations = []
    flags = []
    flux_apers = []      # FLUX_APER
    fluxerr_apers = []   # FLUXERR_APER

    for obj_id, sci_cutout, (xc, yc) in zip(
        valid_ids, sci_cutout_list, valid_coords
    ):
        row = outtab1[outtab1['label'] == obj_id]
        if len(row) != 1:
            raise ValueError(f"Star ID {obj_id} not found uniquely in outtab1.")
        x_images.append(xc)   # Use xc directly from valid_coords
        y_images.append(yc)   # Use yc directly from valid_coords
        ra.append(row['ra'][0])
        dec.append(row['dec'][0])
        flux_radii.append(row['flux_radius'][0])
        flags.append(row['combined_flags'][0])
        elongations.append(row['elongation'][0])
        flux_apers.append(row['kron_flux'][0])
        fluxerr_apers.append(row['kron_fluxerr'][0])  # 绝对误差，与 flux 同单位

    x_arr = np.array(x_images, dtype=np.float32)
    y_arr = np.array(y_images, dtype=np.float32)
    ra_arr = np.array(ra, dtype=np.float64)
    dec_arr = np.array(dec, dtype=np.float64)
    flux_arr = np.array(flux_radii, dtype=np.float32)
    elong_arr = np.array(elongations, dtype=np.float32)
    flags_arr = np.array(flags, dtype=np.int16)
    vign_arr = np.array(sci_cutout_list, dtype=np.float32)  # (n_stars, cutout_size, cutout_size)

    # Use actual flux measurements from SExtractor
    fap_arr = np.array(flux_apers, dtype=np.float32)         # FLUX_APER (absolute flux)
    ferr_arr = np.array(fluxerr_apers, dtype=np.float32)     # FLUXERR_APER (absolute error)
    snr_arr = fap_arr / ferr_arr                             # SNR_WIN = flux / flux_err
    print(f"SNR values: {snr_arr}")
    vect_assoc = np.array(x_images, dtype=np.float64)
    num_assoc = np.ones(n_stars, dtype=np.int32)

    # --------------------------------------------------------------------
    # 7. Copy template and modify hdu2 (LDAC_OBJECTS)
    # --------------------------------------------------------------------
    shutil.copy(template_file, output_ldac)

    with fits.open(output_ldac, mode='update') as hdul:
        hdu2 = hdul[2]

        # Build new column definitions, preserving everything except VIGNET
        new_cols = []
        for col in hdu2.columns:
            if col.name == 'VIGNET':
                n_pix = cutout_size * cutout_size
                new_format = f'{n_pix}E'
                new_col = fits.Column(name=col.name, format=new_format,
                                      unit=col.unit, disp=col.disp)
            else:
                new_col = fits.Column(name=col.name, format=col.format,
                                      unit=col.unit, disp=col.disp)
            new_cols.append(new_col)

        # Create new HDU with the correct number of rows
        new_hdu = fits.BinTableHDU.from_columns(
            fits.ColDefs(new_cols),
            header=hdu2.header,
            nrows=n_stars,
            fill=True,
            name='LDAC_OBJECTS'
        )

        # Fill with our data
        data = new_hdu.data
        data['X_IMAGE'][:]       = x_arr
        data['Y_IMAGE'][:]       = y_arr
        data['ALPHA_J2000'][:]   = ra_arr
        data['DELTA_J2000'][:]   = dec_arr
        data['FLUX_RADIUS'][:]   = flux_arr
        data['ELONGATION'][:]    = elong_arr
        data['FLAGS'][:]         = flags_arr
        data['VIGNET'][:]        = vign_arr.reshape(n_stars, -1)  # flattened
        data['FLUX_APER'][:]     = fap_arr
        data['FLUXERR_APER'][:]  = ferr_arr
        data['SNR_WIN'][:]       = snr_arr
        data['VECTOR_ASSOC'][:]  = vect_assoc
        data['NUMBER_ASSOC'][:]  = num_assoc

        # Update header keywords
        vign_idx = new_hdu.columns.names.index('VIGNET') + 1
        new_hdu.header[f'TDIM{vign_idx}'] = f'({cutout_size}, {cutout_size})'
        new_hdu.header['NAXIS2'] = n_stars

        # Recompute NAXIS1 (bytes per row)
        row_bytes = 0
        type_bytes = {'E':4, 'I':2, 'J':4, 'D':8, 'L':1, 'B':1, 'A':1, 'K':8}
        for col in new_hdu.columns:
            fmt = col.format.strip()
            if fmt[-1] in type_bytes:
                if fmt[:-1] == '':
                    repeat = 1
                else:
                    repeat = int(fmt[:-1])
                row_bytes += repeat * type_bytes[fmt[-1]]
        new_hdu.header['NAXIS1'] = row_bytes
        new_hdu.header['TFIELDS'] = len(new_hdu.columns)

        # Replace hdu2
        hdul[2] = new_hdu
        hdul.flush()

    print(f"LDAC 文件已生成: {output_ldac}")

    # --------------------------------------------------------------------
    # 8. Binary patch hdu1 (LDAC_IMHEAD) with up-to-date values
    # --------------------------------------------------------------------
    # Get global_rms from SExtractor catalog (already contains the correct value)
    global_rms = outtab['global_rms'][0]
    _, median_bkg, std_bkg = sigma_clipped_stats(sci, sigma=3.0, maxiters=5)

    with fits.open(sci_image) as sci_hdul:
        true_naxis1 = sci_hdul[0].header['NAXIS1']
        true_naxis2 = sci_hdul[0].header['NAXIS2']

    with open(output_ldac, 'rb') as f:
        raw = f.read()

    # Replacement pairs (old_bytes -> new_bytes) – must be exact length match
    raw = raw.replace(
        b'NAXIS1  =                 2000',
        f'NAXIS1  = {true_naxis1:20d}'.encode()
    )
    raw = raw.replace(
        b'NAXIS2  =                 2000',
        f'NAXIS2  = {true_naxis2:20d}'.encode()
    )
    raw = raw.replace(
        b'SEXBKGND=   4.254986066371E-03 / Median background level (ADU)',
        f'SEXBKGND= {median_bkg:20.10E} / Median background level (ADU)'.encode()
    )
    # Replace SEXBKDEV with global_rms from SExtractor catalog (critical for chi2 calculation)
    raw = raw.replace(
        b'SEXBKDEV=   1.305840630084E-02 / Median background RMS (ADU)',
        f'SEXBKDEV= {global_rms:20.10E} / Median background RMS (ADU)'.encode()
    )
    raw = raw.replace(
        b'SEXTHLD =   3.519492149353E-01 / Extraction threshold (ADU)',
        f'SEXTHLD = {std_bkg:20.10E} / Extraction threshold (ADU)'.encode()
    )
    raw = raw.replace(
        b'SEXATHLD=   3.519492149353E-01 / Analysis threshold (ADU)',
        f'SEXATHLD= {std_bkg:20.10E} / Analysis threshold (ADU)'.encode()
    )
    raw = raw.replace(
        b'SEXNFIN =                   17 / Final number of extracted sources',
        f'SEXNFIN = {n_stars:20d} / Final number of extracted sources'.encode()
    )
    raw = raw.replace(
        b'SEXPXSCL=   7.400000095367E-02 / Pixel scale used for measurements (arcsec)',
        f'SEXPXSCL= {ref_pixel_scale:20.10E} / Pixel scale used for measurements (arcsec)'.encode()
    )
    raw = raw.replace(
        b'SEXSFWHM=   1.299999952316E-01 / Source FWHM used for measurements (arcsec)',
        f'SEXSFWHM= {fwhm_arcsec:20.10E} / Source FWHM used for measurements (arcsec)'.encode()
    )
    raw = raw.replace(
        b'SEXMGZPT=          28.00000000 / Zero-point used for magnitudes',
        f'SEXMGZPT= {mag_zeropoint:20.8f} / Zero-point used for magnitudes'.encode()
    )
    raw = raw.replace(
        b'SEXGAIN =   1.985576562500E+04 / Gain used (e-/ADU)',
        f'SEXGAIN = {1.0:20.10E} / Gain used (e-/ADU)'.encode()
    )

    with open(output_ldac, 'wb') as f:
        f.write(raw)

    print(f"已将 NAXIS1/2 更新为 {true_naxis1}, {true_naxis2}")


def psfex_config(
    output_config: str = "config.psfex",
    *,
    psf_sampling: float = 0.5,
    psf_size: Tuple[int, int] = (141, 141),
    sample_autoselect: bool = True,
    sample_fwhmrange: str = "1, 5",
    sample_minsn: int = 10,
    psfvar_degrees: int = 0,
) -> None:
    """
    Generate a PSFEx configuration file.

    Only the most commonly modified parameters are exposed; the remaining
    parameters use sensible defaults.  Comments are placed at the end of
    each line, following the PSFEx conventions.

    Parameters
    ----------
    output_config : str, optional (default "config.psfex")
        Path where the configuration file will be written.
    psf_sampling : float, optional (default 0.5)
        Sampling step in pixel units (0.0 = auto).
    psf_size : tuple[int, int], optional (default (141, 141))
        Image size of the PSF model.
    sample_autoselect : bool, optional (default True)
        Automatically select the FWHM? (Y/N)
    sample_fwhmrange : str, optional (default "1, 5")
        Allowed FWHM range for source selection.
    sample_minsn : int, optional (default 10)
        Minimum S/N for a source to be used.
    psfvar_degrees : int, optional (default 0)
        Polynomial degree for spatial PSF variation. 0 = constant PSF,
        1 = linear, 2 = quadratic. Use with `run_psfex(sample_positions=...)`.

    Examples
    --------
    >>> # Constant PSF (default)
    >>> psfex_config(
    ...     output_config='my_psfex.psfex',
    ...     psf_size=(101, 101),
    ... )
    >>> # Position-dependent PSF
    >>> psfex_config(
    ...     output_config='my_psfex.psfex',
    ...     psfvar_degrees=1,
    ... )
    Configuration written to my_psfex.psfex
    """
    autoselect = "Y" if sample_autoselect else "N"

    lines = [
        "# Default configuration file for PSFEx 3.21.1",
        "",
        "#-------------------------------- PSF model ----------------------------------",
        "",
        "BASIS_TYPE      PIXEL_AUTO      # NONE, PIXEL, GAUSS-LAGUERRE or FILE  PIXEL_AUTO",
        "BASIS_NUMBER    20              # Basis number or parameter",
        "BASIS_NAME      basis.fits      # Basis filename (FITS data-cube)",
        "BASIS_SCALE     1.0             # Gauss-Laguerre beta parameter",
        "NEWBASIS_TYPE   NONE            # Create new basis: NONE, PCA_INDEPENDENT",
        "                                # or PCA_COMMON",
        "NEWBASIS_NUMBER 8               # Number of new basis vectors",
        f"PSF_SAMPLING    {psf_sampling}             # Sampling step in pixel units (0.0 = auto)",
        "PSF_PIXELSIZE   1.0             # Effective pixel size in pixel step units",
        "PSF_ACCURACY    0.01            # Accuracy to expect from PSF \"pixel\" values",
        f"PSF_SIZE        {psf_size[0]}, {psf_size[1]}          # Image size of the PSF model",
        "PSF_RECENTER    Y               # Allow recentering of PSF-candidates Y/N ?",
        "MEF_TYPE        INDEPENDENT     # INDEPENDENT or COMMON",
        "",
        "#------------------------- Point source measurements -------------------------",
        "",
        "CENTER_KEYS     X_IMAGE,Y_IMAGE # Catalogue parameters for source pre-centering",
        "PHOTFLUX_KEY    FLUX_APER(1)    # Catalogue parameter for photometric norm.",
        "PHOTFLUXERR_KEY FLUXERR_APER(1) # Catalogue parameter for photometric error",
        "",
        "#----------------------------- PSF variability -------------------------------",
        "",
        "PSFVAR_KEYS     X_IMAGE,Y_IMAGE # Catalogue or FITS (preceded by :) params",
        "PSFVAR_GROUPS   1,1             # Group tag for each context key",
        f"PSFVAR_DEGREES  {psfvar_degrees}             # Polynom degree for each group (0=constant)",
        "PSFVAR_NSNAP    9               # Number of PSF snapshots per axis",
        "HIDDENMEF_TYPE  COMMON          # INDEPENDENT or COMMON",
        "STABILITY_TYPE  EXPOSURE        # EXPOSURE or SEQUENCE",
        "",
        "#----------------------------- Sample selection ------------------------------",
        "",
        f"SAMPLE_AUTOSELECT  {autoselect}            # Automatically select the FWHM (Y/N) ?",
        "SAMPLEVAR_TYPE     SEEING       # File-to-file PSF variability: NONE or SEEING",
        f"SAMPLE_FWHMRANGE   {sample_fwhmrange}",
        "SAMPLE_VARIABILITY 0.5         # Allowed FWHM variability (1.0 = 100%)",
        f"SAMPLE_MINSN       {sample_minsn}          # Minimum S/N for a source to be used",
        "SAMPLE_MAXELLIP    0.3          # Maximum (A-B)/(A+B) for a source to be used",
        "SAMPLE_FLAGMASK    0x00fe       # Rejection mask on SExtractor FLAGS",
        "SAMPLE_WFLAGMASK   0x0000       # Rejection mask on SExtractor FLAGS_WEIGHT",
        "SAMPLE_IMAFLAGMASK 0x0          # Rejection mask on SExtractor IMAFLAGS_ISO",
        "BADPIXEL_FILTER    N            # Filter bad-pixels in samples (Y/N) ?",
        "BADPIXEL_NMAX      0           # Maximum number of bad pixels allowed",
        "",
        "#----------------------- PSF homogeneisation kernel --------------------------",
        "",
        "HOMOBASIS_TYPE     NONE         # NONE or GAUSS-LAGUERRE",
        "HOMOBASIS_NUMBER   10           # Kernel basis number or parameter",
        "HOMOBASIS_SCALE    1.0          # GAUSS-LAGUERRE beta parameter",
        "HOMOPSF_PARAMS     2.0, 3.0     # Moffat parameters of the idealised PSF",
        "HOMOKERNEL_DIR                  # Where to write kernels (empty=same as input)",
        "HOMOKERNEL_SUFFIX  .homo   # Filename extension for homogenisation kernels",
        "",
        "#----------------------------- Output catalogs -------------------------------",
        "",
        "OUTCAT_TYPE        ASCII_HEAD         # NONE, ASCII_HEAD, ASCII, FITS_LDAC",
        "OUTCAT_NAME        psfex_cat.txt           # Output catalog filename",
        "",
        "#------------------------------- Check-plots ----------------------------------",
        "",
        "CHECKPLOT_DEV       PDF         # NULL, XWIN, TK, PS, PSC, XFIG, PNG,",
        "                                # JPEG, AQT, PDF or SVG",
        "CHECKPLOT_RES       0           # Check-plot resolution (0 = default)",
        "CHECKPLOT_ANTIALIAS Y           # Anti-aliasing using convert (Y/N) ?",
        "",
        "CHECKPLOT_TYPE NONE",
        "",
        "CHECKIMAGE_TYPE NONE",
        "",
        "#------------------------------ Check-Images ---------------------------------",
        "#",
        "#CHECKIMAGE_TYPE SAMPLES,SNAPSHOTS_IMRES         # CHI,PROTOTYPES,SAMPLES,RESIDUALS,SNAPSHOTS",
        "#                                # or MOFFAT,-MOFFAT,-SYMMETRICAL",
        "#CHECKIMAGE_NAME results/diagnostics/samp.fits,   results/diagnostics/snap_imres.fits       #chi.fits,proto.fits,samp.fits,resi.fits,snap.fits",
        "#                                # Check-image filenames",
        "CHECKIMAGE_CUBE Y               # Save check-images as datacubes (Y/N) ?",
        "#",
        "",
        "#----------------------------- Miscellaneous ---------------------------------",
        "",
        "PSF_DIR                       # Where to write PSFs (empty=same as input)",
        "PSF_SUFFIX      .psf          # Filename extension for output PSF filename",
        "VERBOSE_TYPE    NORMAL          # can be QUIET,NORMAL,LOG or FULL",
        "WRITE_XML       N               # Write XML file (Y/N)?",
        "XML_NAME                        # Filename for XML output",
        "XSL_URL         file:///Users/mingyang/anaconda3/envs/py39/share/PSFEx/PSFEx.xsl",
        "                                # Filename for XSL style-sheet",
        "NTHREADS        1               # Number of simultaneous threads for",
        "                                # the SMP version of PSFEx",
        "                                # 0 = automatic",
    ]

    with open(output_config, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Configuration written to {output_config}")



def interpolate_psf_at_position(
    psf_file: str,
    x: float,
    y: float,
    normalize: bool = True,
) -> np.ndarray:
    """
    Reconstruct the PSF at a specific position from a PSFEx model.

    This function reads a PSFEx .psf file and reconstructs the PSF at the
    specified image coordinates by interpolating the polynomial basis
    coefficients stored in the file. Supports arbitrary polynomial order.

    Parameters
    ----------
    psf_file : str
        Path to the PSFEx output file (.psf).
    x : float
        X coordinate in image pixels (column), **1-indexed** (same as
        SExtractor's X_IMAGE).
    y : float
        Y coordinate in image pixels (row), **1-indexed** (same as
        SExtractor's Y_IMAGE).
    normalize : bool, optional (default=True)
        If True, normalize the PSF to unit flux.

    Returns
    -------
    psf : np.ndarray
        Reconstructed PSF as a 2D numpy array.

    Raises
    ------
    FileNotFoundError
        If psf_file does not exist.
    RuntimeError
        If the PSFEx file has unexpected structure.
    ValueError
        If POLNAXIS != 2 (only 2D spatial variations are supported).

    Examples
    --------
    >>> psf = interpolate_psf_at_position('psfex.psf', x=100.5, y=200.3)
    >>> psf = interpolate_psf_at_position('psfex.psf', x=100, y=200, normalize=False)

    Notes
    -----
    The interpolation uses the polynomial basis stored in the PSFEx file.
    For a polynomial of degree d, the basis terms are:
        [x^nx * y^ny for ny in range(d+1) for nx in range(d+1-ny)]
    which for d=1 gives [1, x, y], and for d=2 gives [1, x, x^2, y, xy, y^2].

    The PSF is reconstructed as:
        PSF(x, y) = Σ P[i] * PSF_MASK[i]
    where P[i] are the polynomial terms evaluated at the normalized
    coordinates ( (x-POLZERO1)/POLSCAL1, (y-POLZERO2)/POLSCAL2 ).
    """
    if not os.path.isfile(psf_file):
        raise FileNotFoundError(f"PSFEx file not found: {psf_file}")

    with fits.open(psf_file) as hdul:
        h = hdul[1].header

        # Check PSFEx file structure
        pol_naxis = h.get('POLNAXIS', 0)
        if pol_naxis != 2:
            raise ValueError(f"Expected POLNAXIS=2, got {pol_naxis}")

        pol_deg = h.get('POLDEG1', 0)  # Polynomial degree
        x_zero = h.get('POLZERO1', 0.0)  # X coordinate zero point
        y_zero = h.get('POLZERO2', 0.0)  # Y coordinate zero point
        x_scale = h.get('POLSCAL1', 1.0)  # X coordinate scale
        y_scale = h.get('POLSCAL2', 1.0)  # Y coordinate scale

        # Read PSF_MASK basis data
        psf_mask = hdul[1].data['PSF_MASK'][0]  # shape: (n_terms, ny, nx)
        n_terms = psf_mask.shape[0]

        # Verify n_terms matches expected polynomial basis size
        expected_terms = (pol_deg + 1) * (pol_deg + 2) // 2
        if n_terms != expected_terms:
            raise RuntimeError(
                f"PSF_MASK has {n_terms} terms, expected {expected_terms} "
                f"for degree {pol_deg}"
            )

    # Normalize coordinates
    xn = (x - x_zero) / x_scale
    yn = (y - y_zero) / y_scale

    # Compute polynomial powers efficiently (like GalSim's _define_xto)
    x_pows = np.empty(pol_deg + 1)
    x_pows[0] = 1.0
    for i in range(1, pol_deg + 1):
        x_pows[i] = xn * x_pows[i - 1]

    y_pows = np.empty(pol_deg + 1)
    y_pows[0] = 1.0
    for i in range(1, pol_deg + 1):
        y_pows[i] = yn * y_pows[i - 1]

    # Generate polynomial terms: [x^nx * y^ny for ny in range(d+1) for nx in range(d+1-ny)]
    poly = np.array([
        x_pows[nx] * y_pows[ny]
        for ny in range(pol_deg + 1)
        for nx in range(pol_deg + 1 - ny)
    ])

    # Reconstruct PSF: PSF = Σ poly[i] * PSF_MASK[i]
    psf = np.tensordot(poly, psf_mask, axes=([0], [0])).astype(np.float32)

    # Normalize to unit flux (if requested)
    if normalize:
        total = np.sum(psf)
        if total > 0:
            psf = psf / total

    return psf


def run_psfex(
    catalog: str,
    config: str,
    output_dir: str = "./",
    save_psf_fits: bool = True,
    psf_sampling: float = 0.5,
    plot_radial_profile: bool = True,
    sample_positions: Optional[List[Tuple[int, int]]] = None,
) -> str:
    """
    Run PSFEx on an LDAC catalog.

    The output PSF model follows the PSFEx default convention: it is
    named after the input catalog (e.g. ``output_ldac.psf`` for
    ``output_ldac.cat``) and placed in *output_dir*.

    Parameters
    ----------
    catalog : str
        Path to the input FITS_LDAC catalogue (as produced by
        `build_input_ldac`).
    config : str
        Path to the PSFEx configuration file.
    output_dir : str, optional (default="./")
        Directory for output files.
    save_psf_fits : bool, optional (default=True)
        If True, save the extracted PSF to a FITS file. For
        position-dependent PSF (POLDEG1 > 0), this saves PSFs at
        the positions specified in *sample_positions*.
    psf_sampling : float, optional (default=0.5)
        PSF sampling step used to compute oversampling factor.
        oversampling = 1 / psf_sampling.
    plot_radial_profile : bool, optional (default=True)
        If True, generate and save the 1D radial profile plot.
        For position-dependent PSF, plots the first sampled PSF.
    sample_positions : list of tuple, optional (default=None)
        Positions to sample PSF at when POLDEG1 > 0 (position-dependent PSF).
        Each tuple is (x, y) in 1-indexed image pixels. If None and
        POLDEG1 > 0, no FITS PSF files are saved (use
        interpolate_psf_at_position instead). For constant PSF (POLDEG1 == 0),
        this parameter is ignored.

    Returns
    -------
    psf_file : str
        Full path to the generated PSF model file (.psf). Use
        `interpolate_psf_at_position` to reconstruct PSF at any position.

    Raises
    ------
    FileNotFoundError
        If *catalog* or *config* does not exist.
    RuntimeError
        If PSFEx returns a non-zero exit status.

    Examples
    --------
    >>> # Constant PSF (PSFVAR_DEGREES=0)
    >>> psf_file = run_psfex(
    ...     catalog="psfex_input_assoc.cat",
    ...     config="config.psfex",
    ...     output_dir="./psf_results",
    ... )
    >>> # Position-dependent PSF (PSFVAR_DEGREES=1)
    >>> psf_file = run_psfex(
    ...     catalog="psfex_input_assoc.cat",
    ...     config="config.psfex",
    ...     sample_positions=[(100, 100), (2000, 2000), (100, 2000), (2000, 100)],
    ... )
    >>> # Later: reconstruct PSF at any position
    >>> psf = interpolate_psf_at_position(psf_file, x=500, y=800)
    """
    # Validate inputs
    if not os.path.isfile(catalog):
        raise FileNotFoundError(f"Catalog not found: {catalog}")
    if not os.path.isfile(config):
        raise FileNotFoundError(f"Config not found: {config}")

    os.makedirs(output_dir, exist_ok=True)

    # Build command
    cmd = f"psfex {catalog} -c {config} -PSF_DIR {output_dir}"

    # Execute
    print("Running PSFEx...")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"PSFEx exited with error code {ret}")

    # Determine output PSF file name following PSFEx convention:
    # same basename as catalog, with .psf extension.
    base = os.path.splitext(os.path.basename(catalog))[0]
    psf_file = os.path.join(output_dir, f"{base}.psf")

    if not os.path.isfile(psf_file):
        raise FileNotFoundError(f"PSF model {psf_file} not found.")

    print(f"PSF model written to {psf_file}")

    # Read PSFEx header to check if PSF is position-dependent
    with fits.open(psf_file) as hdul:
        pol_deg = hdul[1].header.get('POLDEG1', 0)
        actual_sampling = hdul[0].header.get('PSF_SAMP', psf_sampling)
        oversampling = int(1.0 / actual_sampling)

    is_position_dependent = (pol_deg > 0)

    if is_position_dependent:
        print(f"Position-dependent PSF detected (POLDEG1={pol_deg})")

    # Read the constant term (0th basis function) for radial profile plotting
    with fits.open(psf_file) as hdul:
        # PSFEx stores data as hdul[1].data[0][0][0] - the constant term
        const_psf_oversampled = hdul[1].data[0][0][0]

        if oversampling > 1:
            const_psf_data = oversampled2regular(const_psf_oversampled, oversampling)
        else:
            const_psf_data = const_psf_oversampled

        # Normalize to unit flux
        const_psf_data = const_psf_data / np.sum(const_psf_data)

    # Step 1: Extract and save PSF data to FITS
    psf_fits_path = None
    if save_psf_fits:
        if not is_position_dependent:
            # Constant PSF: save the single PSF from PSFEx
            psf_fits_path = os.path.join(output_dir, f"{base}_psf.fits")
            fits.writeto(psf_fits_path, const_psf_data, overwrite=True)
            print(f"PSF FITS saved to {psf_fits_path}")
        else:
            # Position-dependent PSF: sample at specified positions
            if sample_positions is None:
                print("No sample_positions provided. Use interpolate_psf_at_position() to reconstruct PSF.")
            else:
                print(f"Sampling PSF at {len(sample_positions)} positions...")
                for x, y in sample_positions:
                    psf_oversampled = interpolate_psf_at_position(psf_file, x, y, normalize=False)

                    if oversampling > 1:
                        psf_data = oversampled2regular(psf_oversampled, oversampling)
                    else:
                        psf_data = psf_oversampled

                    # Normalize to unit flux
                    psf_data = psf_data / np.sum(psf_data)

                    pos_suffix = f"_x{x}_y{y}"
                    psf_fits_path = os.path.join(output_dir, f"{base}_psf{pos_suffix}.fits")
                    fits.writeto(psf_fits_path, psf_data, overwrite=True)
                    print(f"  PSF at ({x}, {y}) saved to {psf_fits_path}")
                print(f"Saved PSF FITS files for {len(sample_positions)} positions.")

    # Step 2: Plot radial profile (requires FITS file)
    if plot_radial_profile:
        psf_data_for_plot = None

        if is_position_dependent:
            # Use the constant term (field average PSF)
            psf_data_for_plot = const_psf_data
            print("Plotting radial profile for field-averaged PSF (0th basis term)")
        else:  # 常数 PSF
            psf_data_for_plot = const_psf_data

        if psf_data_for_plot is not None:
            ny, nx = psf_data_for_plot.shape
            max_radius = min(nx, ny) // 2
            radii = np.linspace(0, max_radius, int(max_radius / 1.0) + 1)

            # Read star cutouts from input catalog
            with fits.open(catalog) as cat_hdul:
                vignets = cat_hdul[2].data['VIGNET']
                snr_win = cat_hdul[2].data['SNR_WIN']
                n_stars = vignets.shape[0]

            # Setup colormap for SNR
            cmap = plt.get_cmap('viridis')
            norm = plt.Normalize(vmin=snr_win.min(), vmax=snr_win.max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            fig, ax = plt.subplots(figsize=(8, 5))

            # Plot individual star radial profiles, color-coded by SNR
            for i in range(n_stars):
                cutout = vignets[i]
                # Create mask for pixels to ignore (mask_value = -1e30)
                mask = (cutout <= -1e29)

                # Find peak position within valid (non-masked) region only
                # Set masked pixels to -inf so they won't be selected by argmax
                cutout_for_search = cutout.copy()
                cutout_for_search[mask] = -np.inf
                peak_idx = np.argmax(cutout_for_search)
                star_center = (peak_idx % cutout.shape[1], peak_idx // cutout.shape[1])

                rp = RadialProfile(cutout, star_center, radii, mask=mask)
                profile = rp.profile
                peak = np.nanmax(profile)
                if peak > 0:
                    profile = profile / peak

                color = cmap(norm(snr_win[i]))
                ax.plot(rp.radius, profile, color=color, alpha=0.5, linewidth=0.8)

            # Plot PSFEx PSF radial profile
            # Find peak position in PSF
            peak_idx = np.argmax(psf_data_for_plot)
            psf_center = (peak_idx % psf_data_for_plot.shape[1], peak_idx // psf_data_for_plot.shape[1])
            rp_psf = RadialProfile(psf_data_for_plot, psf_center, radii)
            psf_profile = rp_psf.profile
            peak_psf = np.nanmax(psf_profile)
            if peak_psf > 0:
                psf_profile = psf_profile / peak_psf

            ax.plot(rp_psf.radius, psf_profile, color='red', linewidth=2,
                    label='PSFEx PSF')

            # Add colorbar for SNR
            cbar = fig.colorbar(sm, ax=ax, label='SNR_WIN')

            ax.set_xlabel('Radius [pix]')
            ax.set_ylabel('Normalized flux')
            ax.set_title('1D Radial Profiles')
            ax.legend()
            ax.set_yscale('log')
            plt.tight_layout()
            out_profile = os.path.join(output_dir, f'{base}_radial_profile.png')
            fig.savefig(out_profile, dpi=250, bbox_inches='tight')
            plt.close(fig)
            print(f"Radial profile saved to {out_profile}")

            # Calculate and print FWHM
            fwhm_val = fit_fwhm(psf_data_for_plot)[0]
            print(f"FWHM = {fwhm_val}")

    return psf_file
