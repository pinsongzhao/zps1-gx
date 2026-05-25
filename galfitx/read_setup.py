"""
Identifier:     galfitx/read_setup.py
Name:           read_setup.py
Description:    read config parameters
Author:         Chao Ma
Created:        2026-01-19
Modified-History:
    2026-01-19, Chao Ma, created
"""

import os
import shutil
import numpy as np
from typing import List, Union, Optional


def set_trailing_slash(path: str) -> str:
    """
    Ensure a filesystem path ends with the OS‑specific trailing slash.

    If the input path already ends with the separator (e.g., '/' on Unix,
    '\\' on Windows), it is returned unchanged. Otherwise, the separator is
    appended.

    Parameters
    ----------
    path : str
        The directory path to process.

    Returns
    -------
    str
        The input path with a trailing separator guaranteed.
    """
    sep = os.path.sep
    if path.endswith(sep):
        result = path
    else:
        result = path + sep
    return result


def valid_num(s: str) -> bool:
    """
    Check if a string represents a valid number (integer or float).

    Parameters
    ----------
    s : str
        The input string to test.

    Returns
    -------
    bool
        True if the string can be successfully converted to a float,
        False otherwise (e.g., if it contains non‑numeric characters).
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_setup(setup_file: str):
    """
    Read and parse a GALAPAGOS‑style setup file.

    The setup file uses a line format with a four‑character code (e.g. "A00)")
    followed by a value. This function reads the file, interprets each line,
    and populates a `Setup` object with the corresponding attributes. Default
    values are applied for any missing parameters.

    Parameters
    ----------
    setup_file : str
        Path to the setup file.

    Returns
    -------
    Setup
        An object containing all parameters read from the setup file, with
        sensible defaults where values are missing.

    Raises
    ------
    SystemExit
        If the setup file does not exist.
    ValueError
        If an unrecognised setup code is encountered.
    """

    if os.path.isfile(setup_file) is False:
        print("input file does not exist")
        raise SystemExit(1)

    len_num = 4  # length of the numbering scheme, e.g. 4 for 'A00)'

    class Setup:
        """Container for all GALAPAGOS setup parameters."""

        def __init__(self) -> None:
            # Input / output files and directories
            self.files: str = ""
            self.outdir: str = ""

            # SExtractor options
            self.dosex: int = 0
            self.sexexe: str = ""
            self.sexout: str = ""
            self.cold: str = ""
            self.coldcat: str = ""
            self.coldseg: str = ""
            self.hot: str = ""
            self.hotcat: str = ""
            self.hotseg: str = ""
            self.enlarge = -1.0
            self.outcat: str = ""
            self.outseg: str = ""
            self.outparam: str = ""
            self.check: str = ""
            self.chktype: str = ""
            self.sex_rms: int = 0
            self.exclude: str = ""
            self.exclude_rad: float = -1.0
            self.outonly: int = 0
            self.bad: str = ""
            self.sexcomb: str = ""

            # Stamp creation options
            self.dostamps: int = 0
            self.stampfile: str = ""
            self.stamp_pre: List[str] = [""]
            self.stampsize: float = -1.0

            # Sky estimation options
            self.dosky: int = 0
            self.skymap: str = ""
            self.outsky: str = ""
            self.skyscl: float = -1.0
            self.neiscl: float = -1.0
            self.skyoff: float = -1.0
            self.dstep: int = -1
            self.wstep: int = -1
            self.gap: int = -1
            self.cut: float = 0.0
            self.nobj_max: int = 0
            self.power: float = 0.0
            self.nslope: int = 0
            self.stel_slope: float = 0.0
            self.stel_zp: float = 0.0
            self.maglim_gal: float = -1.0
            self.maglim_star: float = -1.0
            self.nneighb: int = 0
            self.max_proc: int = 0
            self.min_dist: float = 0.0
            self.min_dist_block: float = -1.0
            self.srclist: str = ""
            self.srclistrad: float = 0.0

            # GalfitM options
            self.galexe: str = ""
            self.gal_output: str = ""
            self.gal_kill_time: Union[float, str] = 0.0
            self.batch: str = ""
            self.obj: str = ""
            self.galfit_out: str = ""
            self.psf: str = ""
            self.mask: str = ""
            self.constr: str = ""
            self.convbox: int = 0
            self.zp: float = 0.0
            self.platescl: float = 0.0
            self.expt: float = 0.0
            self.conmaxre: float = 0.0
            self.conminm: float = 0.0
            self.conmaxm: float = 0.0
            self.conminn: float = 0.2
            self.conmaxn: float = 8.0
            self.nice: int = 0
            self.version: float = 4.4
            self.cheb: List[Union[int, float, str]] = [-1] * 7
            self.galfit_out_path: str = " "
            self.do_restrict: int = 0
            self.restrict_frac_primary: float = 20.0
            self.mindeg: int = 0

            # BD decompositiom
            self.dobd: int = 0
            self.cheb_b: List[Union[int, float, str]] = [-1] * 7
            self.cheb_d: List[Union[int, float, str]] = [-1] * 7
            self.bd_label: str = " "
            self.gal_output_bd: str = ""
            self.bd_hpc: int = 0
            self.bd_hpc_path: str = " "
            self.bd_srclist: str = ""
            self.bd_srclistrad: float = 0.0
            self.bd_maglim: float = 99.0
            self.bd_psf_corr: List[str] = ["", ""]

            self.docombine: int = 0
            self.docombinebd: int = 0
            self.cat: str = ""
            self.galfitoutput: int = 0

    setup = Setup()
    setup.stel_slope = 1e20
    setup.stel_zp = 1e20
    setup.srclistrad = -1
    setup.galfit_out_path = ""

    # check format for backwards compatibility
    block_bd = 0
    line = ""
    with open(setup_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            # Skip comment or empty lines
            if line.startswith("#") or len(line) == 0:
                continue

            # comment at end of line?
            pos = line.find("#")  # if do not find, return -1
            if pos == -1:
                pos = len(line)

            # Extract content
            content = line[len_num:pos].strip()
            if line[:len_num].upper() == "G00)":
                block_bd = 1

    line = ""
    with open(setup_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("#") or len(line) == 0:
                continue
            pos = line.find("#")
            if pos == -1:
                pos = len(line)
            content = line[len_num:pos].strip()
            value = line[:len_num].upper()

            match value:
                case "A00)":
                    setup.files = content
                case "A01)":
                    setup.outdir = set_trailing_slash(content)

                case "B00)":
                    setup.dosex = 1 if content == "execute" else 0
                case "B01)":
                    setup.sexexe = content
                case "B02)":
                    setup.sexout = content
                case "B03)":
                    setup.cold = content
                case "B04)":
                    setup.coldcat = content
                case "B05)":
                    setup.coldseg = content
                case "B06)":
                    setup.hot = content
                case "B07)":
                    setup.hotcat = content
                case "B08)":
                    setup.hotseg = content
                case "B09)":
                    setup.enlarge = float(content)
                case "B10)":
                    setup.outcat = content
                case "B11)":
                    setup.outseg = content
                case "B12)":
                    setup.outparam = content
                case "B13)":
                    setup.check = "" if content in ("none", "") else content
                case "B14)":
                    setup.chktype = content
                case "B15)":
                    setup.sex_rms = 1 if content == "rms" else 0
                case "B16)":
                    setup.exclude = content
                case "B17)":
                    setup.exclude_rad = float(content)
                case "B18)":
                    setup.outonly = 1 if content == "outonly" else 0
                case "B19)":
                    setup.bad = content
                case "B20)":
                    setup.sexcomb = content

                case "C00)":
                    setup.dostamps = 1 if content == "execute" else 0
                case "C01)":
                    setup.stampfile = content
                case "C02)":
                    setup.stamp_pre = content
                case "C03)":
                    setup.stampsize = float(content)

                case "D00)":
                    setup.dosky = 1 if content == "execute" else 0
                case "D01)":
                    setup.skymap = content
                case "D02)":
                    setup.outsky = content
                case "D03)":
                    setup.skyscl = float(content)
                case "D04)":
                    setup.neiscl = float(content)
                case "D05)":
                    setup.skyoff = float(content)
                case "D06)":
                    setup.dstep = int(float(content))
                case "D07)":
                    setup.wstep = int(float(content))
                case "D08)":
                    setup.gap = int(float(content))
                case "D09)":
                    setup.cut = float(content)
                case "D10)":
                    setup.nobj_max = int(float(content))
                case "D11)":
                    setup.power = float(content)
                case "D12)":
                    setup.nslope = int(float(content))
                case "D13)":
                    setup.stel_slope = float(content)
                case "D14)":
                    setup.stel_zp = float(content)
                case "D15)":
                    setup.maglim_gal = float(content)
                case "D16)":
                    setup.maglim_star = float(content)
                case "D17)":
                    setup.nneighb = int(float(content))
                case "D18)":
                    setup.max_proc = int(float(content))
                case "D19)":
                    setup.min_dist = float(content)
                case "D20)":
                    setup.min_dist_block = float(content)
                case "D21)":
                    setup.srclist = "" if content in ("none", "") else content
                case "D22)":
                    setup.srclistrad = float(content)

                case "E00)":
                    setup.galexe = content
                case "E01)":
                    setup.batch = content
                case "E02)":
                    setup.obj = content
                case "E03)":
                    setup.galfit_out = content
                case "E04)":
                    setup.psf = content
                case "E05)":
                    setup.mask = content
                case "E06)":
                    setup.constr = content
                case "E07)":
                    setup.convbox = int(float(content))
                case "E08)":
                    setup.zp = float(content)
                case "E09)":
                    setup.platescl = float(content)
                case "E10)":
                    setup.expt = float(content)
                case "E11)":
                    setup.conmaxre = float(content)
                case "E12)":
                    setup.conminm = float(content)
                case "E13)":
                    setup.conmaxm = float(content)
                case "E14)":
                    setup.conminn = float(content)
                case "E15)":
                    setup.conmaxn = float(content)
                case "E16)":
                    setup.nice = 1 if content == "nice" else 0
                case "E17)":
                    setup.version = float(content)
                case "E18)":
                    setup.gal_output = content
                case "E19)":
                    setup.gal_kill_time = content
                case "E20)":
                    parts = content.split(",")
                    for n in range(7):
                        setup.cheb[n] = parts[n]
                case "E21)":
                    if content == "":
                        setup.galfit_out_path = content
                    else:
                        setup.galfit_out_path = set_trailing_slash(content)
                case "E22)":
                    setup.do_restrict = 1 if content == "restrict" else 0
                case "E23)":
                    setup.restrict_frac_primary = content
                case "E24)":
                    setup.mindeg = int(content)

                case "F00)":
                    if block_bd == 1:
                        setup.dobd = 1 if content == "execute" else 0
                    else:
                        setup.docombine = 1 if content == "execute" else 0

                case "F01)":
                    if block_bd == 1:
                        parts = content.split(",")
                        for n in range(7):
                            setup.cheb_b[n] = parts[n]
                    else:
                        setup.docombinebd = 1 if content == "execute" else 0

                case "F02)":
                    if block_bd == 1:
                        parts = content.split(",")
                        for n in range(7):
                            setup.cheb_d[n] = parts[n]
                    else:
                        setup.cat = content

                case "F03)":
                    setup.bd_label = content
                case "F04)":
                    setup.bd_srclist = "" if content in ("none", "") else content
                case "F05)":
                    setup.bd_srclistrad = float(content)
                case "F06)":
                    setup.bd_maglim = float(content) if valid_num(content) else 99.0
                case "F07)":
                    setup.gal_output_bd = content
                case "F08)":
                    setup.bd_hpc = 1 if content == "HPC" else 0
                case "F09)":
                    setup.bd_hpc_path = set_trailing_slash(content)
                case "F10)":
                    parts = content.split(",", 1)
                    setup.bd_psf_corr[0] = parts[0].strip()
                    setup.bd_psf_corr[1] = parts[1].strip() if len(parts) > 1 else ""

                case "G00)":
                    setup.docombine = 1 if content == "execute" else 0
                case "G01)":
                    setup.docombinebd = 1 if content == "execute" else 0
                case "G02)":
                    setup.cat = content

                case _:
                    raise ValueError(f"Unrecognized setup value: {value!r}")

    # set default values
    if setup.enlarge == -1:
        setup.enlarge = 1.1
    if setup.exclude_rad == -1:
        setup.exclude_rad = 2.0
    if setup.check == "":
        setup.chktype = "none"
    if setup.stampsize == -1:
        setup.stampsize = 2.5
    if setup.skyscl == -1:
        setup.skyscl = 3.0
    if setup.neiscl == -1:
        setup.neiscl = 1.5
    if setup.skyoff == -1:
        setup.skyoff = 20
    if setup.dstep == -1:
        setup.dstep = 30
    if setup.wstep == -1:
        setup.wstep == 60
    if setup.gap == -1:
        setup.gap = 30
    if setup.stel_slope == 1e20:
        setup.stel_slope = -0.3
    if setup.stel_zp == 1e20:
        setup.stel_zp = 6.8
    if setup.maglim_gal == -1:
        setup.maglim_gal = 5
    if setup.maglim_star == -1:
        setup.maglim_star = 2
    if setup.min_dist_block == -1:
        setup.min_dist_block = setup.min_dist / 3.0
    if setup.cheb[0] == -1:
        for n in range(len(setup.cheb)):
            setup.cheb[n] = 0
    if setup.mindeg < 1:
        setup.mindeg = 1

    # check whether excutables exist
    exe_path = setup.sexexe.strip()
    if os.path.exists(exe_path) is False and setup.dosex == 1:
        print("SExtractor executable does not exist")

    galexe_path = setup.galexe.strip()
    if os.path.exists(galexe_path) is False and (setup.dosky == 1 or setup.dobd == 1):
        print("Galfit executable does not exist")

    return setup


def read_image_files(setup, save_folder, silent=False, nocheck_read=None):

    lineone = ""
    with open(setup.files, "r") as f:
        lineone = f.readline().strip()  # read the first line
    columnsf = lineone.split()
    ncolf = len(columnsf)
    # if number of columns eq 4 then assume 1 band survey and fits files
    if (ncolf == 4) or (ncolf == 5):
        if silent is False:
            print("assuming one band dataset. Are all files within A00) fits images?")

        if ncolf == 4:
            images, weights, outpath, outpre = [], [], [], []
            f = open(setup.files, "r")
            for line in f:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                cols = line.split()
                images.append(cols[0])
                weights.append(cols[1])
                outpath.append(cols[2])
                outpre.append(cols[3])
            f.close()

        if ncolf == 5:
            images, weights, sigmaps, outpath, outpre = [], [], [], [], []
            f = open(setup.files, "r")
            for line in f:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                cols = line.split()
                images.append(cols[0])
                weights.append(cols[1])
                sigmaps.append(cols[2])
                outpath.append(cols[3])
                outpre.append(cols[4])
            f.close()

        nband = 1
        images = images.strip()

    if ncolf == 6:
        if silent is False:
            print("assuming multi-wavelength dataset. Assuming first line to be for SExtractor, rest for fitting!")
        if setup.version < 4.0:
            print(
                "you seem to be using mulit-wavelength data, but the GALFIT version you have specified only supports one-band data"
            )
            print(
                "Multi-band fitting needs GALFITM in order to be able to read out the fitting parameters (output has to be in a fits table)"
            )
            raise RuntimeError("GALFIT version too old for multi-band fitting")
        band = []
        wavelength = []
        mag_offset = []
        filelist = []
        zeropoint = []
        exptime = []
        f = open(setup.files, "r")
        for line in f:
            line = line.strip()
            if line == "" or line[0] == "#":
                continue
            cols = line.split()
            band.append(cols[0])
            wavelength.append(float(cols[1]))
            mag_offset.append(float(cols[2]))
            filelist.append(cols[3])
            zeropoint.append(float(cols[4]))
            exptime.append(float(cols[5]))
        f.close()

        # copy the setup files to the folder
        # Ensure the target folder exists
        if os.path.isdir(save_folder) is False:
            os.makedirs(save_folder)
        # Copy each file into save_folder
        for src in filelist:
            basename = os.path.basename(src)
            dst = os.path.join(save_folder, basename)
            shutil.copy(src, dst)

        nband = len(nband) - 1

        # read first (sextractor) file: sex_gala
        hlpimages, hlpweights, hlpoutpath, hlpoutpre = [], [], [], []
        f = open(filelist[0], "r")
        for line in f:
            line = line.strip()
            if line == "" or line[0] == "#":
                continue
            cols = line.split()
            hlpimages.append(cols[0])
            hlpweights.append(cols[1])
            hlpoutpath.append(cols[2])
            hlpoutpre.append(cols[3])
        f.close()

        cnt = [0] * (nband + 1)
        hlpimages = [s.strip() for s in hlpimages]

        # create arrays in setup needed to store all the data
        setup.images = np.full((nband + 1, len(hlpimages)), "", dtype="<U32")
        setup.weights = np.full((nband + 1, len(hlpimages)), "", dtype="<U32")
        setup.sigmaps = np.full((nband + 1, len(hlpimages)), "", dtype="<U32")
        setup.sigflags = np.zeros(nband + 1, dtype=int)
        setup.outpath = np.full((nband + 1, len(hlpimages)), "", dtype="<U32")
        setup.outpath_band = np.full((nband + 1, len(hlpimages)), "", dtype="<U32")
        setup.outpre = np.full((nband + 1, len(hlpimages)), "", dtype="<U32")
        setup.nband = nband

        # define addtional parameters
        setup.stamp_pre = band
        setup.wavelength = wavelength
        setup.mag_offset = mag_offset
        setup.zp = zeropoint
        setup.expt = exptime
        del hlpimages, hlpweights, hlpoutpath, hlpoutpre

        # read first (sextractor) file again: sex_gala
        hlpimages, hlpweights, hlpoutpath, hlpoutpre = [], [], [], []
        f = open(filelist[0], "r")
        for line in f:
            line = line.strip()
            if line == "" or line[0] == "#":
                continue
            cols = line.split()
            hlpimages.append(cols[0])
            hlpweights.append(cols[1])
            hlpoutpath.append(cols[2])
            hlpoutpre.append(cols[3])
        f.close()

        cnt[0] = len(hlpimages)
        setup.images[0] = hlpimages
        setup.weights[0] = hlpweights
        setup.sigmaps[0] = "none"
        setup.sigflags[0] = 0
        setup.outpre[0] = hlpoutpre
        setup.outpath[0] = [(set_trailing_slash(outdir) + h.strip() + os.path.sep) for h in hlpoutpath]
        setup.outpath_band[0] = [(i.strip() + band[0].strip()) for i in setup.outpath[0]]
        del hlpimages, hlpweights, hlpoutpath, hlpoutpre

        # read other bands, 2 columns: image+weight; 3 columns: image+weight+noisemap
        for b in range(1, nband + 1):
            ncolfb = 0
            lineone = ""
            with open(filelist[b], "r") as f:
                lineone = f.readline().strip()  # read the first line
            columnsf = lineone.split()
            ncolfb = len(columnsf)

            if ncolfb == 2:
                hlpimages, hlpweights = [], []
                f = open(filelist[b], "r")
                for line in f:
                    line = line.strip()
                    if line == "" or line[0] == "#":
                        continue
                    cols = line.split()
                    hlpimages.append(cols[0])
                    hlpweights.append(cols[1])
                f.close()

            if ncolfb == 3:
                hlpimages, hlpweights, hlpsigmaps = [], [], []
                f = open(filelist[b], "r")
                for line in f:
                    line = line.strip()
                    if line == "" or line[0] == "#":
                        continue
                    cols = line.split()
                    hlpimages.append(cols[0])
                    hlpweights.append(cols[1])
                f.close()

            cnt[b] = len(hlpimages)
            if cnt[b] != cnt[0] and not silent:
                print(
                    f"input list {band[b].strip()} contains a wrong number of entries (tiles), compared to SExtractor list"
                )
                raise RuntimeError(f"Mismatched entry count: {cnt[b]} vs {cnt[0]}")

            setup.images[b] = hlpimages
            setup.weights[b] = hlpweights
            setup.sigmaps[b] = "none"
            setup.sigflags[b] = 0
            if ncolfb == 3:
                setup.sigmaps[b] = hlpsigmaps
                setup.sigflags[b] = 1
                print(f"sigma maps used for band {band[b].strip()}")
            setup.outpre[b] = setup.outpre[0]
            setup.outpath[b] = setup.outpath[0]
            setup.outpath_band[b] = [(i.strip() + band[b].strip()) for i in setup.outpath[0]]
            del hlpimages, hlpweights
            if ncolfb != 2 and ncolfb != 3:
                print(f"Invalid Entry in {filelist[b]}")

    if ncolf != 6 and ncolf != 5 and ncolf != 4:
        print("Invalid Entry in " + setup.files)

    # now check whether all images exist
    if not nocheck_read:
        # Only perform checks if any of the read flags is set
        if (setup.dosex + setup.dostamps + setup.dosky + setup.dobd) >= 1:
            # Vectorize os.path.exists to apply it element-wise
            path_exists = np.vectorize(os.path.exists)
            # Check existence for images and weights
            image_exist = path_exists(np.char.strip(setup.images))
            weight_exist = path_exists(np.char.strip(setup.weights))
            im_non_exist = np.where(image_exist is False)
            wh_non_exist = np.where(weight_exist is False)
            cntimne = im_non_exist[0].size
            cntwhne = wh_non_exist[0].size
            if cntimne != 0:
                stopnow = 1
                print(" ")
                print("there is at least one image missing as currently defined (typo?)")
                print("missing images:")
                print(setup.images[im_non_exist])
            if cntwhne != 0:
                stopnow = 1
                print(" ")
                print("there is at least one weight image missing as currently defined (typo?)")
                print("missing images:")
                print(setup.weights[wh_non_exist])

            if (cntimne != 0) or (cntwhne != 0):
                raise RuntimeError(f"Some images are missed....")
            if (cntimne == 0) or (cntwhne == 0):
                print("all images and weights have been checked to exist")
