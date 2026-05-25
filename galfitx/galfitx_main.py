# from csst_common import CsstResult, CsstStatus
from astropy.table import Table
import os
import subprocess
import numpy as np
import shutil
from astropy.io import fits, ascii
from galfitx.source_detection import SExtractor, SExtractor_HDR
from galfitx.postage_stamp import create_stamp_file, cut_stamps
from galfitx.mask import create_mask, prepare_galfitm
from galfitx.sky_calculation import getsky, create_skymap
from galfitx.read_setup import read_setup, set_trailing_slash
from galfitx.utils import scaled_kron
from galfitx.create_setup_gs import prepare_galfits, process_psf
from galfitx.gx_gsutils import calconvfactor, effective_wave
import datetime
import toml
# from galfitx.csst_pipe import tab2fits
from typing import List

__all__ = ["core_galfitx"]


def core_galfitx(
    image_name: str = "/path/to/image",
    weight_name: str = "/path/to/weightimage",
    kernel_name: str = "/path/to/kernel",
    image_list: List[str] = ["/path/to/image"],
    outdir_list: List[str] = [],
    filter_list: List[str] = [],
    label_list: List[str] = [],
    psf_list: List[str] = [],
    zero_list: List[float] = [],
    config: str = "./csst.toml",
) -> int:
    """
    GalfitX pipeline

    Galfitx pipeline for multi-band imaging measurement

    Parameters
    ----------
    image_name : str
        path to detection image.
    weight_name : str
        path to weight image.
    kernel_name : str
        path to convolution kernel file.
    image_list : list[str]
        path to multi-band science images
    outdir_list : list[str]
        path to store multi-band postage stamps
    filter_list : list[str]
        filter identifier used for GalfitS
    label_list : list[str]
        label to each filter
    psf_list : list[str]
        path to multi-band input psf images
    zero_list : list[float]
        multi-band magnitude zeropoints
    config : str
        path to config file

    Returns
    -------
    CsstResult
        A result object containing the status, and products.
    """

    kernel = np.loadtxt(kernel_name, skiprows=1)
    param_dict = toml.load(config)

    outtab, outsegm = SExtractor_HDR(
        image_name,
        kernel=(kernel, kernel),
        detect_minarea=(param_dict["detect_minarea_cold"], param_dict["detect_minarea_hot"]),
        detect_thresh=(param_dict["detect_thresh_cold"], param_dict["detect_thresh_hot"]),
        deblend=(param_dict["deblend"], param_dict["deblend"]),
        deblend_nthresh=(param_dict["deblend_nthresh_cold"], param_dict["deblend_nthresh_hot"]),
        deblend_mincont=(param_dict["deblend_mincont_cold"], param_dict["deblend_mincont_hot"]),
        clean=(param_dict["clean"], param_dict["clean"]),
        back_type=(param_dict["back_type"], param_dict["back_type"]),
        back_value=(param_dict["back_value"], param_dict["back_value"]),
        back_size=(param_dict["back_size_cold"], param_dict["back_size_hot"]),
        back_filtersize=(param_dict["back_filtersize"], param_dict["back_filtersize"]),
        weight_type=param_dict["BACKGROUND"],
        scale_factor=param_dict["scale_factor"],
        pixel_scale=param_dict["pixel_scale"],
        mag_zeropoint=zero_list[0],
        verbose=True,
    )

    catalog_name = "./sex/outcat"
    outtab = ascii.read(catalog_name)
    # tab2fits(outtab)

    create_stamp_file(image_name, catalog_name, sizefac=param_dict["sizefac"], outfile="stamps")

    for i in range(len(image_list)):
        filename = image_list[i]
        outdir = outdir_list[i]
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.mkdir(outdir)
        cut_stamps(filename, outdir, stampfile="stamps")

    sorted_idx = np.argsort(outtab["mag_auto"])  # mag index in ascending order

    path = "./galfits/"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    image_list = np.array(image_list)
    outdir_list = np.array(outdir_list)
    label_list = np.array(label_list)
    filter_list = np.array(filter_list)
    psf_list = np.array(psf_list)

    for idx in sorted_idx:
        skyfile = [path + "skyfile" + label_list[j] + str(idx + 1) for j in range(len(image_list))]
        corner, objects = create_mask(
            weight_name,
            "./sex/outseg.fits",
            catalog_name,
            "stamps",
            path + "mask" + str(idx + 1),
            path + "mask_primary" + str(idx + 1),
            idx,
            param_dict["scale"],
            param_dict["offset"],
            param_dict["limgal"],
            param_dict["b"],
        )

        setup = read_setup(param_dict["setup"])
        image_stamp_list = np.array([outdir_list[j] + str(idx + 1) + ".fits" for j in range(image_list)])
        psf_file_used = process_psf(idx + 1, len(image_list), psf_list, image_stamp_list, label_list)

        prepare_galfits(
            image_stamp_list,
            catalog_name,
            path + "obj" + str(idx + 1),
            corner,
            np.array([path + "mask" + str(idx + 1)] * nband),
            "stamps",
            psf_file_used,
            skyfile,
            zero_list,
            label_list,
            filter_list,
            len(image_list),
            objects=objects,
            setup=setup,
            sky0=True,
        )

        cmd = [
            "python",
            "/home/machao/opt/GalfitS-main/src/galfits/galfitS.py",
            "--config",
            path + "obj" + str(idx + 1),
            "--workplace",
            "./galfits/",
            "--saveimgs",
            "--num_steps",
            "8000",
        ]

        subprocess.run(cmd)

    return 1
