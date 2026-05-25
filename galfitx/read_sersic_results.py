"""
Identifier:     galfitx/read_sersic_results.py
Name:           read_sersic_results.py
Description:    reading existing fitting results
Author:         Chao Ma
Created:        2026-01-19
Modified-History:
    2026-01-19, Chao Ma, created
"""

import os
import sys
import time
import numpy as np
from astropy.io import fits
from typing import List, Optional, Union, Any


def derive_primary_chi2(obj_file: str, gal_exe: str) -> None:
    """
    Run a quick Galfit fit for only the primary source and extract χ² statistics.

    This function modifies the input Galfit configuration file to create a version
    where only the primary source is fitted. It does so by:
    - Changing the output filename (B) line) to a temporary file.
    - Constructing new mask files (F) line) that combine the original mask with a primary‑only mask (so that only the primary is unmasked).
    - Setting all non‑primary component fluxes to zero (P) lines) and keeping only the essential header lines (A, W).
    It then runs Galfit with the `-o2` option to quickly compute the best‑fit
    χ² without actually saving the model image. The resulting χ², χ² per degree
    of freedom, and number of degrees of freedom are read from the FITS output
    and written to an ASCII file `{obj_file}_primary_fit_info`. All temporary
    files are deleted afterwards.

    Parameters
    ----------
    obj_file : str
        Path to the original Galfit configuration file (e.g., `obj123`).
    gal_exe : str
        Path to the Galfit executable (e.g., `galfitm` or `galfit`).

    Returns
    -------
    None
        The function writes a small ASCII file and does not return any value.
    """

    obj = obj_file
    infolder = os.getcwd()
    if "/" in obj:  # does obj contain a slash?
        # everything up to the last slash, the directory path portion of obj.
        workfolder = os.path.dirname(obj)
        os.chdir(workfolder)
        # And keep only the filename portion
        obj = os.path.basename(obj)
        print(obj)

    # count number of bands first and get band names
    with open(obj, "r") as f:
        for line in f:
            if line.strip().startswith("A) "):
                break
        # Read the very next line (the one labelled "A1)")
        line = f.readline()
        # Find where the “)” is, then skip past “)  ”
    start = line.find(")") + 2
    end = line.find("#")
    content = line[start:end].strip()
    # Split on commas and strip each name
    bandnames = [bn.strip() for bn in content.split(",")]
    nband = len(bandnames)

    # start reading input file line by line
    infile = open(obj, "r")
    # Prepare to write to a new output file
    newfile = obj + "_with_primary_mask"
    outfile = open(newfile, "w")
    for line in infile:
        # Strip the newline
        line = line.rstrip("\n")
        content_numbers = " "  # ' '?
        content_descriptor = " "
        idx = line.find(")") + 2
        start = line[:idx].strip()  # e.g 'A)'
        # check if comment behind the content
        if line.find("#") != -1:
            # Content between ')' and '#'
            content = line[line.find(")") + 2: line.find("#")].strip()
            comment = line[line.find("#"):].strip()  # e.g, '#ssss..'
        else:
            # No comment
            content = line[line.find(")") + 2:].strip()
            comment = ""

        if line.strip().startswith("B) "):
            # Change output filename for B) records
            # Take everything up through '.fits', append '_primary_only.fits'
            new_output_name = content.split(".fits")[0] + "_primary_only.fits"  # './galfit/t.9_gf_primary_only.fits'
            outfile.write(f"B) {new_output_name}\n")

        elif line.strip().startswith("F) "):
            # change masks used and create new masks that combine primary and normal mask!
            # create new masks
            # find primary mask first
            content_elements = [el.strip() for el in content.split(",")]
            file_folder = content_elements[0][: content_elements[0].rfind("/") + 1]
            file_name = content_elements[0][len(file_folder):]
            file_base = file_name.split("_mask.fits")[0]  # e.g. 't.9_H'
            file_base = file_base[: file_base.rfind("_")]  # e.g. 't.9'
            # pmask_file = file_folder+file_base+'_mask_primary.fits' # '..../galfit/t.9_mask_primary.fits'
            pmask_file = "mask_primary" + obj[6:-10] + ".fits"
            pmask, pmaskhd = fits.getdata(pmask_file, header=True)

            # build all new masks and record their filenames
            new_masks_all = []
            for b in range(nband):
                file_folder = content_elements[b][: content_elements[b].rfind("/") + 1]
                file_name = content_elements[b][len(file_folder):]
                file_base = file_name.split("_mask.fits")[0]  # e.g. 't.9_H'
                file_base = file_base[: file_base.rfind("_")]  # e.g. 't.9'
                new_mask_file = file_folder + "primary_" + file_name  # e.g. './galfit/primary_t.9_H_mask.fits'

                # load original mask image
                mask, maskhd = fits.getdata(content_elements[b], header=True)

                # start with all ones
                new_mask = np.ones_like(mask)

                # unmask primary
                new_mask[pmask == 1] = 0

                # mask anything masked in the original mask
                omasked_idx = np.where(mask != 0)
                if omasked_idx[0].size > 0:
                    new_mask[omasked_idx] = 1

                fits.writeto(new_mask_file, new_mask, header=maskhd, overwrite=True)
                new_masks_all.append(new_mask_file)

            # join the new mask filenames with commas
            outfile.write("F) " + ",".join(new_masks_all) + "\n")

        elif line.strip().startswith("W) "):
            # Write only start + comment
            outfile.write(f"{start}  {comment}\n")

        elif line.strip().startswith("P) "):
            # # Write start, a zero, then comment
            outfile.write(f"{start} 0 {comment}\n")

        else:
            # All other lines pass through unchanged
            outfile.write(line + "\n")

    infile.close()
    outfile.close()

    # run galfitm with "-o2"
    os.system(f"{gal_exe} -o2 {newfile} > null")
    # give galfit a moment to finish writing its FITS
    time.sleep(0.2)

    # open the FITS and read the FIT_INFO extension
    hdul = fits.open(new_output_name)
    fit_info = hdul["FIT_INFO"].data
    ndof = fit_info["NDOF"][0]
    chisq = fit_info["CHISQ"]
    chi2nu = fit_info["CHI2NU"]
    hdul.close()

    # write out the three values into a little ASCII file
    ascii_name = obj_file + "_primary_fit_info"
    filew_prime = open(ascii_name, "w")
    filew_prime.write(f"{ndof} {chisq} {chi2nu}\n")
    filew_prime.close()

    time.sleep(0.1)

    # delete all mask files and new start file to save disk space, only keep output file for readout of Chi^2
    if os.path.exists("null"):
        os.remove("null")
    for new_mask_file in new_masks_all:
        if os.path.exists(new_mask_file):
            os.remove(new_mask_file)
    #  Remove the primary‐only FITS that Galfit just produced
    if os.path.exists(new_output_name):
        os.remove(new_output_name)

    os.chdir(infolder)


def read_sersic_results(
    obj: str, nband: int, setup: Any, bd: Optional[bool] = None, final: Optional[Any] = None
) -> Any:
    """
    Read and parse the output FITS file from a Galfit / GalfitM fit, and return
    a feedback object containing all fit results.

    This function reads a Galfit output FITS file (e.g., `t.266_gf.fits`),
    extracts the best‑fit parameters, uncertainties, fit statistics, and metadata.
    It also handles the case where the fit did not complete (file missing) by
    returning default placeholder values. Depending on the `bd` flag, the
    returned feedback object contains results for either a normal fit (one
    primary component) or a bright‑object deblending (BD) fit (two components:
    diffuse `d` and bulge `b`). Additionally, it calls `derive_primary_chi2` if
    the auxiliary file with the primary‑only χ² statistics is missing.

    Parameters
    ----------
    obj : str
        Path to the Galfit output FITS file (e.g., `./galfit/t.266_gf.fits`).
    nband : int
        Number of bands used in the fit.
    setup : object
        A `Setup` instance (from `read_setup()`) containing configuration
        parameters, especially the path to the Galfit executable.
    bd : bool, optional
        If True, the fit was a bright‑object deblending (BD) run with two
        components (diffuse and bulge). If False or None, assume a normal
        fit with one primary component. Default is None.
    final : any, optional
        Unused parameter (kept for backward compatibility). Default None.

    Returns
    -------
    feedback : object
        A dynamically created object containing all fit results. Its attributes
        differ depending on `bd` and whether the file exists. See the code
        for the exact list of attributes. Typical attributes include:
        - For normal fit: `.mag_galfit`, `.re_galfit`, `.n_galfit`, ... (scalars for the first band) and `.mag_galfit_band`, `.re_galfit_band`, ...
        (arrays per band).
        - For BD fit: attributes with `_d` (diffuse) and `_b` (bulge) suffixes.
        - Fit statistics: `.chisq_galfit`, `.ndof_galfit`, `.chi2nu_galfit`,
        `.flag_galfit` (1 = missing file, 2 = success).
        - Primary‑only χ²: `.chisq_galfit_prime`, `.ndof_galfit_prime`, etc.
        - Chebyshev polynomial results: attributes ending in `_cheb`.
        - Metadata: `.initfile`, `.logfile`, `.psf_galfit_band`, etc.
    """

    if os.path.exists(obj) is True:
        with fits.open(obj) as hdul:
            result = hdul["FINAL_BAND"].data
            res_cheb = hdul["FINAL_CHEB"].data
            fit_info = hdul["FIT_INFO"].data
            band_info = hdul["BAND_INFO"].data

            comp = 1
            while True:
                comp += 1
                if f"COMP{comp}_MAG" not in result.dtype.names:
                    break

            if "NGOOD" in band_info.dtype.names:
                # '_g' might stands for "global", indicating a metric aggregated across all bands
                ngood_g = band_info["NGOOD"][0]
            else:
                ngood_g = -99

            if "NMASK" in band_info.dtype.names:
                nmask_g = band_info["NMASK"][0]
            else:
                nmask_g = -99

            fit_info_primary_file = (
                fit_info["LOGFILE"].strip()[0] + "_primary_fit_info"
            )  # logfile, e.g.,path + 't.266_gf.galfit.01'
            if not os.path.exists(fit_info_primary_file):
                if not os.path.exists(fit_info["LOGFILE"].strip()[0]):
                    print("galfit restart file missing although galfit output file exists")
                    print(fit_info["LOGFILE"].strip()[0])
                    sys.exit(1)  # Exit with error code
                else:
                    derive_primary_chi2(fit_info["LOGFILE"].strip()[0], setup.galexe)
                    time.sleep(0.2)

            # read out these values from ascii file
            infile = open(fit_info_primary_file, "r")
            line = infile.readline().strip()  # e.g. "123 4567.8 12.34"
            infile.close()
            ndof_prime = int(line.split()[0])
            chi2_prime = float(line.split()[1][1:-1])
            chi2nu_prime = float(line.split()[2][1:-1])

            # del feedback

            if bd is None:

                class feedback:
                    def __init__(self):
                        # Primary component (COMP2), for the first band
                        self.mag_galfit = result[0]["COMP2_MAG"]
                        self.magerr_galfit = result[0]["COMP2_MAG_ERR"]
                        self.re_galfit = result[0]["COMP2_RE"]
                        self.reerr_galfit = result[0]["COMP2_RE_ERR"]
                        self.n_galfit = result[0]["COMP2_N"]
                        self.nerr_galfit = result[0]["COMP2_N_ERR"]
                        self.q_galfit = result[0]["COMP2_AR"]
                        self.qerr_galfit = result[0]["COMP2_AR_ERR"]
                        self.pa_galfit = result[0]["COMP2_PA"]
                        self.paerr_galfit = result[0]["COMP2_PA_ERR"]
                        self.x_galfit = result[0]["COMP2_XC"]
                        self.xerr_galfit = result[0]["COMP2_XC_ERR"]
                        self.y_galfit = result[0]["COMP2_YC"]
                        self.yerr_galfit = result[0]["COMP2_YC_ERR"]
                        # PSF and sky
                        self.psf_galfit = band_info[0]["PSF"].strip()
                        self.sky_galfit = result[0]["COMP1_SKY"]

                        # Primary component (COMP2), all bands
                        self.mag_galfit_band = result["COMP2_MAG"]
                        self.magerr_galfit_band = result["COMP2_MAG_ERR"]
                        self.re_galfit_band = result["COMP2_RE"]
                        self.reerr_galfit_band = result["COMP2_RE_ERR"]
                        self.n_galfit_band = result["COMP2_N"]
                        self.nerr_galfit_band = result["COMP2_N_ERR"]
                        self.q_galfit_band = result["COMP2_AR"]
                        self.qerr_galfit_band = result["COMP2_AR_ERR"]
                        self.pa_galfit_band = result["COMP2_PA"]
                        self.paerr_galfit_band = result["COMP2_PA_ERR"]
                        self.x_galfit_band = result["COMP2_XC"]
                        self.xerr_galfit_band = result["COMP2_XC_ERR"]
                        self.y_galfit_band = result["COMP2_YC"]
                        self.yerr_galfit_band = result["COMP2_YC_ERR"]
                        self.sky_galfit_band = result["COMP1_SKY"]

                        # Chebyshev
                        self.mag_galfit_cheb = res_cheb["COMP2_MAG"]
                        self.magerr_galfit_cheb = res_cheb["COMP2_MAG_ERR"]
                        self.re_galfit_cheb = res_cheb["COMP2_RE"]
                        self.reerr_galfit_cheb = res_cheb["COMP2_RE_ERR"]
                        self.n_galfit_cheb = res_cheb["COMP2_N"]
                        self.nerr_galfit_cheb = res_cheb["COMP2_N_ERR"]
                        self.q_galfit_cheb = res_cheb["COMP2_AR"]
                        self.qerr_galfit_cheb = res_cheb["COMP2_AR_ERR"]
                        self.pa_galfit_cheb = res_cheb["COMP2_PA"]
                        self.paerr_galfit_cheb = res_cheb["COMP2_PA_ERR"]
                        self.x_galfit_cheb = res_cheb["COMP2_XC"]
                        self.xerr_galfit_cheb = res_cheb["COMP2_XC_ERR"]
                        self.y_galfit_cheb = res_cheb["COMP2_YC"]
                        self.yerr_galfit_cheb = res_cheb["COMP2_YC_ERR"]
                        self.sky_galfit_cheb = res_cheb["COMP1_SKY"]

                        # Fit metadata
                        self.initfile = fit_info["INITFILE"][0]
                        self.logfile = fit_info["LOGFILE"][0]
                        self.constrnt = fit_info["CONSTRNT"][0]
                        self.fitsect = fit_info["FITSECT"][0]
                        self.convbox = fit_info["CONVBOX"][0]
                        self.psf_galfit_band = band_info["PSF"]

                        # Fit statistics
                        self.chisq_galfit = fit_info["CHISQ"][0]
                        self.ndof_galfit = fit_info["NDOF"][0]
                        self.nfree_galfit = fit_info["NFREE"][0]
                        self.ngood_galfit_band = ngood_g
                        self.nmask_galfit_band = nmask_g
                        self.nfix_galfit = fit_info["NFIX"][0]
                        self.cputime_setup_galfit = fit_info["CPUTIME_SETUP"][0]
                        self.cputime_fit_galfit = fit_info["CPUTIME_FIT"][0]
                        self.cputime_total_galfit = fit_info["CPUTIME_TOTAL"][0]
                        self.chi2nu_galfit = fit_info["CHI2NU"][0]
                        self.niter_galfit = fit_info["NITER"][0]
                        self.galfit_version = fit_info["VERSION"][0]
                        self.firstcon_galfit = fit_info["FIRSTCON"][0]
                        self.lastcon_galfit = fit_info["LASTCON"][0]
                        self.neigh_galfit = comp - 3
                        self.flag_galfit = 2

                        # Degree totals from cheb fits
                        self.x_galfit_deg = sum(res_cheb["COMP2_XC_FIT"])
                        self.y_galfit_deg = sum(res_cheb["COMP2_YC_FIT"])
                        self.mag_galfit_deg = sum(res_cheb["COMP2_MAG_FIT"])
                        self.re_galfit_deg = sum(res_cheb["COMP2_RE_FIT"])
                        self.n_galfit_deg = sum(res_cheb["COMP2_N_FIT"])
                        self.q_galfit_deg = sum(res_cheb["COMP2_AR_FIT"])
                        self.pa_galfit_deg = sum(res_cheb["COMP2_PA_FIT"])

                        # Prime-fit ASCII stats
                        self.ndof_galfit_prime = ndof_prime
                        self.chisq_galfit_prime = chi2_prime
                        self.chi2nu_galfit_prime = chi2nu_prime

                # Instantiate
                feedback = feedback()

            if bd is not None:

                class feedback:
                    def __init__(self):
                        self.mag_galfit_d = result[0]["COMP2_MAG"]
                        self.magerr_galfit_d = result[0]["COMP2_MAG_ERR"]
                        self.re_galfit_d = result[0]["COMP2_RE"]
                        self.reerr_galfit_d = result[0]["COMP2_RE_ERR"]
                        self.n_galfit_d = result[0]["COMP2_N"]
                        self.nerr_galfit_d = result[0]["COMP2_N_ERR"]
                        self.q_galfit_d = result[0]["COMP2_AR"]
                        self.qerr_galfit_d = result[0]["COMP2_AR_ERR"]
                        self.pa_galfit_d = result[0]["COMP2_PA"]
                        self.paerr_galfit_d = result[0]["COMP2_PA_ERR"]
                        self.x_galfit_d = result[0]["COMP2_XC"]
                        self.xerr_galfit_d = result[0]["COMP2_XC_ERR"]
                        self.y_galfit_d = result[0]["COMP2_YC"]
                        self.yerr_galfit_d = result[0]["COMP2_YC_ERR"]

                        self.mag_galfit_b = result[0]["COMP3_MAG"]
                        self.magerr_galfit_b = result[0]["COMP3_MAG_ERR"]
                        self.re_galfit_b = result[0]["COMP3_RE"]
                        self.reerr_galfit_b = result[0]["COMP3_RE_ERR"]
                        self.n_galfit_b = result[0]["COMP3_N"]
                        self.nerr_galfit_b = result[0]["COMP3_N_ERR"]
                        self.q_galfit_b = result[0]["COMP3_AR"]
                        self.qerr_galfit_b = result[0]["COMP3_AR_ERR"]
                        self.pa_galfit_b = result[0]["COMP3_PA"]
                        self.paerr_galfit_b = result[0]["COMP3_PA_ERR"]
                        self.x_galfit_b = result[0]["COMP3_XC"]
                        self.xerr_galfit_b = result[0]["COMP3_XC_ERR"]
                        self.y_galfit_b = result[0]["COMP3_YC"]
                        self.yerr_galfit_b = result[0]["COMP3_YC_ERR"]

                        self.psf_galfit_bd = band_info[0]["PSF"].strip()
                        self.sky_galfit_bd = result[0]["COMP1_SKY"]

                        self.mag_galfit_band_d = result["COMP2_MAG"]
                        self.magerr_galfit_band_d = result["COMP2_MAG_ERR"]
                        self.re_galfit_band_d = result["COMP2_RE"]
                        self.reerr_galfit_band_d = result["COMP2_RE_ERR"]
                        self.n_galfit_band_d = result["COMP2_N"]
                        self.nerr_galfit_band_d = result["COMP2_N_ERR"]
                        self.q_galfit_band_d = result["COMP2_AR"]
                        self.qerr_galfit_band_d = result["COMP2_AR_ERR"]
                        self.pa_galfit_band_d = result["COMP2_PA"]
                        self.paerr_galfit_band_d = result["COMP2_PA_ERR"]
                        self.x_galfit_band_d = result["COMP2_XC"]
                        self.xerr_galfit_band_d = result["COMP2_XC_ERR"]
                        self.y_galfit_band_d = result["COMP2_YC"]
                        self.yerr_galfit_band_d = result["COMP2_YC_ERR"]

                        self.mag_galfit_band_b = result["COMP3_MAG"]
                        self.magerr_galfit_band_b = result["COMP3_MAG_ERR"]
                        self.re_galfit_band_b = result["COMP3_RE"]
                        self.reerr_galfit_band_b = result["COMP3_RE_ERR"]
                        self.n_galfit_band_b = result["COMP3_N"]
                        self.nerr_galfit_band_b = result["COMP3_N_ERR"]
                        self.q_galfit_band_b = result["COMP3_AR"]
                        self.qerr_galfit_band_b = result["COMP3_AR_ERR"]
                        self.pa_galfit_band_b = result["COMP3_PA"]
                        self.paerr_galfit_band_b = result["COMP3_PA_ERR"]
                        self.x_galfit_band_b = result["COMP3_XC"]
                        self.xerr_galfit_band_b = result["COMP3_XC_ERR"]
                        self.y_galfit_band_b = result["COMP3_YC"]
                        self.yerr_galfit_band_b = result["COMP3_YC_ERR"]

                        self.sky_galfit_band_bd = result["COMP1_SKY"]

                        self.mag_galfit_cheb_d = res_cheb["COMP2_MAG"]
                        self.magerr_galfit_cheb_d = res_cheb["COMP2_MAG_ERR"]
                        self.re_galfit_cheb_d = res_cheb["COMP2_RE"]
                        self.reerr_galfit_cheb_d = res_cheb["COMP2_RE_ERR"]
                        self.n_galfit_cheb_d = res_cheb["COMP2_N"]
                        self.nerr_galfit_cheb_d = res_cheb["COMP2_N_ERR"]
                        self.q_galfit_cheb_d = res_cheb["COMP2_AR"]
                        self.qerr_galfit_cheb_d = res_cheb["COMP2_AR_ERR"]
                        self.pa_galfit_cheb_d = res_cheb["COMP2_PA"]
                        self.paerr_galfit_cheb_d = res_cheb["COMP2_PA_ERR"]
                        self.x_galfit_cheb_d = res_cheb["COMP2_XC"]
                        self.xerr_galfit_cheb_d = res_cheb["COMP2_XC_ERR"]
                        self.y_galfit_cheb_d = res_cheb["COMP2_YC"]
                        self.yerr_galfit_cheb_d = res_cheb["COMP2_YC_ERR"]

                        self.mag_galfit_cheb_b = res_cheb["COMP3_MAG"]
                        self.magerr_galfit_cheb_b = res_cheb["COMP3_MAG_ERR"]
                        self.re_galfit_cheb_b = res_cheb["COMP3_RE"]
                        self.reerr_galfit_cheb_b = res_cheb["COMP3_RE_ERR"]
                        self.n_galfit_cheb_b = res_cheb["COMP3_N"]
                        self.nerr_galfit_cheb_b = res_cheb["COMP3_N_ERR"]
                        self.q_galfit_cheb_b = res_cheb["COMP3_AR"]
                        self.qerr_galfit_cheb_b = res_cheb["COMP3_AR_ERR"]
                        self.pa_galfit_cheb_b = res_cheb["COMP3_PA"]
                        self.paerr_galfit_cheb_b = res_cheb["COMP3_PA_ERR"]
                        self.x_galfit_cheb_b = res_cheb["COMP3_XC"]
                        self.xerr_galfit_cheb_b = res_cheb["COMP3_XC_ERR"]
                        self.y_galfit_cheb_b = res_cheb["COMP3_YC"]
                        self.yerr_galfit_cheb_b = res_cheb["COMP3_YC_ERR"]

                        self.sky_galfit_cheb_bd = res_cheb["COMP1_SKY"]

                        self.initfile_bd = fit_info["INITFILE"][0]
                        self.logfile_bd = fit_info["LOGFILE"][0]
                        self.constrnt_bd = fit_info["CONSTRNT"][0]
                        self.psf_galfit_band_bd = band_info["PSF"]
                        self.chisq_galfit_bd = fit_info["CHISQ"][0]
                        self.ndof_galfit_bd = fit_info["NDOF"][0]
                        self.nfree_galfit_bd = fit_info["NFREE"][0]
                        self.nfix_galfit_bd = fit_info["NFIX"][0]
                        self.cputime_setup_galfit_bd = fit_info["CPUTIME_SETUP"][0]
                        self.cputime_fit_galfit_bd = fit_info["CPUTIME_FIT"][0]
                        self.cputime_total_galfit_bd = fit_info["CPUTIME_TOTAL"][0]
                        self.chi2nu_galfit_bd = fit_info["CHI2NU"][0]
                        self.niter_galfit_bd = fit_info["NITER"][0]
                        self.galfit_version_bd = fit_info["VERSION"][0]
                        self.firstcon_galfit_bd = fit_info["FIRSTCON"][0]
                        self.lastcon_galfit_bd = fit_info["LASTCON"][0]
                        self.neigh_galfit_bd = comp - 4
                        self.flag_galfit_bd = 2

                        self.x_galfit_deg_d = sum(res_cheb["COMP2_XC_FIT"])
                        self.y_galfit_deg_d = sum(res_cheb["COMP2_YC_FIT"])
                        self.mag_galfit_deg_d = sum(res_cheb["COMP2_MAG_FIT"])
                        self.re_galfit_deg_d = sum(res_cheb["COMP2_RE_FIT"])
                        self.n_galfit_deg_d = sum(res_cheb["COMP2_N_FIT"])
                        self.q_galfit_deg_d = sum(res_cheb["COMP2_AR_FIT"])
                        self.pa_galfit_deg_d = sum(res_cheb["COMP2_PA_FIT"])

                        self.x_galfit_deg_b = sum(res_cheb["COMP3_XC_FIT"])
                        self.y_galfit_deg_b = sum(res_cheb["COMP3_YC_FIT"])
                        self.mag_galfit_deg_b = sum(res_cheb["COMP3_MAG_FIT"])
                        self.re_galfit_deg_b = sum(res_cheb["COMP3_RE_FIT"])
                        self.n_galfit_deg_b = sum(res_cheb["COMP3_N_FIT"])
                        self.q_galfit_deg_b = sum(res_cheb["COMP3_AR_FIT"])
                        self.pa_galfit_deg_b = sum(res_cheb["COMP3_PA_FIT"])

                        self.ndof_galfit_bd_prime = ndof_prime
                        self.chisq_galfit_bd_prime = chi2_prime
                        self.chi2nu_galfit_bd_prime = chi2nu_prime

                feedback = feedback()

    else:
        # obj not exists
        psf = ["none"] * nband
        if bd is None:

            class feedback:
                def __init__(self):
                    self.mag_galfit = -999.0
                    self.magerr_galfit = 99999.0
                    self.re_galfit = -99.0
                    self.reerr_galfit = 99999.0
                    self.n_galfit = -99.0
                    self.nerr_galfit = 99999.0
                    self.q_galfit = -99.0
                    self.qerr_galfit = 99999.0
                    self.pa_galfit = 0.0
                    self.paerr_galfit = 99999.0
                    self.x_galfit = 0.0
                    self.xerr_galfit = 99999.0
                    self.y_galfit = 0.0
                    self.yerr_galfit = 99999.0

                    self.psf_galfit = "none"
                    self.sky_galfit = -999.0

                    self.mag_galfit_band = [-999.0] * nband
                    self.magerr_galfit_band = [99999.0] * nband
                    self.re_galfit_band = [-99.0] * nband
                    self.reerr_galfit_band = [99999.0] * nband
                    self.n_galfit_band = [-99.0] * nband
                    self.nerr_galfit_band = [99999.0] * nband
                    self.q_galfit_band = [-99.0] * nband
                    self.qerr_galfit_band = [99999.0] * nband
                    self.pa_galfit_band = [0.0] * nband
                    self.paerr_galfit_band = [99999.0] * nband
                    self.x_galfit_band = [0.0] * nband
                    self.xerr_galfit_band = [99999.0] * nband
                    self.y_galfit_band = [0.0] * nband
                    self.yerr_galfit_band = [99999.0] * nband

                    self.sky_galfit_band = [-999.0] * nband

                    self.mag_galfit_cheb = [-999.0] * nband
                    self.magerr_galfit_cheb = [99999.0] * nband
                    self.re_galfit_cheb = [-99.0] * nband
                    self.reerr_galfit_cheb = [99999.0] * nband
                    self.n_galfit_cheb = [-99.0] * nband
                    self.nerr_galfit_cheb = [99999.0] * nband
                    self.q_galfit_cheb = [-99.0] * nband
                    self.qerr_galfit_cheb = [99999.0] * nband
                    self.pa_galfit_cheb = [0.0] * nband
                    self.paerr_galfit_cheb = [99999.0] * nband
                    self.x_galfit_cheb = [0.0] * nband
                    self.xerr_galfit_cheb = [99999.0] * nband
                    self.y_galfit_cheb = [0.0] * nband
                    self.yerr_galfit_cheb = [99999.0] * nband

                    self.sky_galfit_cheb = [-999.0] * nband

                    self.initfile = ""
                    self.logfile = ""
                    self.constrnt = ""
                    self.psf_galfit_band = psf
                    self.chisq_galfit = -99.0
                    self.ndof_galfit = -99
                    self.nfree_galfit = -99
                    self.nfix_galfit = -99
                    self.cputime_setup_galfit = -99.0
                    self.cputime_fit_galfit = -99.0
                    self.cputime_total_galfit = -99.0
                    self.chi2nu_galfit = -99.0
                    self.niter_galfit = -99.0
                    self.galfit_version = "crash"
                    self.firstcon_galfit = -99.0
                    self.lastcon_galfit = -99.0
                    self.neigh_galfit = -99
                    self.flag_galfit = 1
                    self.x_galfit_deg = -99
                    self.y_galfit_deg = -99
                    self.mag_galfit_deg = -99
                    self.re_galfit_deg = -99
                    self.n_galfit_deg = -99
                    self.q_galfit_deg = -99
                    self.pa_galfit_deg = -99
                    self.ndof_galfit_prime = -99
                    self.chisq_galfit_prime = -99.0
                    self.chi2nu_galfit_prime = -99.0

            feedback = feedback()

        if bd is not None:

            class feedback:
                def __init__(self):
                    self.mag_galfit_d = -999.0
                    self.magerr_galfit_d = 99999.0
                    self.re_galfit_d = -99.0
                    self.reerr_galfit_d = 99999.0
                    self.n_galfit_d = -99.0
                    self.nerr_galfit_d = 99999.0
                    self.q_galfit_d = -99.0
                    self.qerr_galfit_d = 99999.0
                    self.pa_galfit_d = 0.0
                    self.paerr_galfit_d = 99999.0
                    self.x_galfit_d = 0.0
                    self.xerr_galfit_d = 99999.0
                    self.y_galfit_d = 0.0
                    self.yerr_galfit_d = 99999.0

                    self.mag_galfit_b = -999.0
                    self.magerr_galfit_b = 99999.0
                    self.re_galfit_b = -99.0
                    self.reerr_galfit_b = 99999.0
                    self.n_galfit_b = -99.0
                    self.nerr_galfit_b = 99999.0
                    self.q_galfit_b = -99.0
                    self.qerr_galfit_b = 99999.0
                    self.pa_galfit_b = 0.0
                    self.paerr_galfit_b = 99999.0
                    self.x_galfit_b = 0.0
                    self.xerr_galfit_b = 99999.0
                    self.y_galfit_b = 0.0
                    self.yerr_galfit_b = 99999.0

                    self.psf_galfit_bd = "none"
                    self.sky_galfit_bd = -999.0

                    self.mag_galfit_band_d = [-999.0] * nband
                    self.magerr_galfit_band_d = [99999.0] * nband
                    self.re_galfit_band_d = [-99.0] * nband
                    self.reerr_galfit_band_d = [99999.0] * nband
                    self.n_galfit_band_d = [-99.0] * nband
                    self.nerr_galfit_band_d = [99999.0] * nband
                    self.q_galfit_band_d = [-99.0] * nband
                    self.qerr_galfit_band_d = [99999.0] * nband
                    self.pa_galfit_band_d = [0.0] * nband
                    self.paerr_galfit_band_d = [99999.0] * nband
                    self.x_galfit_band_d = [0.0] * nband
                    self.xerr_galfit_band_d = [99999.0] * nband
                    self.y_galfit_band_d = [0.0] * nband
                    self.yerr_galfit_band_d = [99999.0] * nband

                    self.mag_galfit_band_b = [-999.0] * nband
                    self.magerr_galfit_band_b = [99999.0] * nband
                    self.re_galfit_band_b = [-99.0] * nband
                    self.reerr_galfit_band_b = [99999.0] * nband
                    self.n_galfit_band_b = [-99.0] * nband
                    self.nerr_galfit_band_b = [99999.0] * nband
                    self.q_galfit_band_b = [-99.0] * nband
                    self.qerr_galfit_band_b = [99999.0] * nband
                    self.pa_galfit_band_b = [0.0] * nband
                    self.paerr_galfit_band_b = [99999.0] * nband
                    self.x_galfit_band_b = [0.0] * nband
                    self.xerr_galfit_band_b = [99999.0] * nband
                    self.y_galfit_band_b = [0.0] * nband
                    self.yerr_galfit_band_b = [99999.0] * nband

                    self.sky_galfit_band_bd = [-999.0] * nband

                    self.mag_galfit_cheb_d = [-999.0] * nband
                    self.magerr_galfit_cheb_d = [99999.0] * nband
                    self.re_galfit_cheb_d = [-99.0] * nband
                    self.reerr_galfit_cheb_d = [99999.0] * nband
                    self.n_galfit_cheb_d = [-99.0] * nband
                    self.nerr_galfit_cheb_d = [99999.0] * nband
                    self.q_galfit_cheb_d = [-99.0] * nband
                    self.qerr_galfit_cheb_d = [99999.0] * nband
                    self.pa_galfit_cheb_d = [0.0] * nband
                    self.paerr_galfit_cheb_d = [99999.0] * nband
                    self.x_galfit_cheb_d = [0.0] * nband
                    self.xerr_galfit_cheb_d = [99999.0] * nband
                    self.y_galfit_cheb_d = [0.0] * nband
                    self.yerr_galfit_cheb_d = [99999.0] * nband

                    self.mag_galfit_cheb_b = [-999.0] * nband
                    self.magerr_galfit_cheb_b = [99999.0] * nband
                    self.re_galfit_cheb_b = [-99.0] * nband
                    self.reerr_galfit_cheb_b = [99999.0] * nband
                    self.n_galfit_cheb_b = [-99.0] * nband
                    self.nerr_galfit_cheb_b = [99999.0] * nband
                    self.q_galfit_cheb_b = [-99.0] * nband
                    self.qerr_galfit_cheb_b = [99999.0] * nband
                    self.pa_galfit_cheb_b = [0.0] * nband
                    self.paerr_galfit_cheb_b = [99999.0] * nband
                    self.x_galfit_cheb_b = [0.0] * nband
                    self.xerr_galfit_cheb_b = [99999.0] * nband
                    self.y_galfit_cheb_b = [0.0] * nband
                    self.yerr_galfit_cheb_b = [99999.0] * nband

                    self.sky_galfit_cheb_bd = [-999.0] * nband

                    self.initfile_bd = ""
                    self.logfile_bd = ""
                    self.constrnt_bd = ""
                    self.psf_galfit_band_bd = psf
                    self.chisq_galfit_bd = -99.0
                    self.ndof_galfit_bd = -99
                    self.nfree_galfit_bd = -99
                    self.nfix_galfit_bd = -99
                    self.cputime_setup_galfit_bd = -99.0
                    self.cputime_fit_galfit_bd = -99.0
                    self.cputime_total_galfit_bd = -99.0
                    self.chi2nu_galfit_bd = -99.0
                    self.niter_galfit_bd = -99
                    self.galfit_version_bd = "crash"
                    self.firstcon_galfit_bd = -99
                    self.lastcon_galfit_bd = -99
                    self.neigh_galfit_bd = -99
                    self.flag_galfit_bd = 1

                    self.x_galfit_deg_d = -99
                    self.y_galfit_deg_d = -99
                    self.mag_galfit_deg_d = -99
                    self.re_galfit_deg_d = -99
                    self.n_galfit_deg_d = -99
                    self.q_galfit_deg_d = -99
                    self.pa_galfit_deg_d = -99

                    self.x_galfit_deg_b = -99
                    self.y_galfit_deg_b = -99
                    self.mag_galfit_deg_b = -99
                    self.re_galfit_deg_b = -99
                    self.n_galfit_deg_b = -99
                    self.q_galfit_deg_b = -99
                    self.pa_galfit_deg_b = -99

                    self.ndof_galfit_bd_prime = -99
                    self.chisq_galfit_bd_prime = -99.0
                    self.chi2nu_galfit_bd_prime = -99.0

            feedback = feedback()

    return feedback
