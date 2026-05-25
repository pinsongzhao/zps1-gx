"""
Identifier:     galfitx/source_detection.py
Name:           source_detection.py
Description:    source extraction
Author:         Chao Ma
Created:        2026-01-19
Modified-History:
    2026-01-19, Chao Ma, created

Script Name: Source Detection
License: MIT License (or your chosen license)
Description:
Briefly describe what this script does.

Created: <2024-12-28> by Jinyi Shangguan
Last Updated: <2025-05-12> by Bingcheng Jin

Dependencies:
- List any required packages or modules.
scipy (1.14.1)
scikit-image (0.25.0)
numpy (1.26.0)
astropy (6.1.7)
photutils (2.0.2 or later)
tqdm (4.62.3)

Usage:
    Imported by other modules
"""

# from __future__ import annotations
import os
import sys
import shutil
from typing import List, Optional, Union, Literal, Any, Tuple
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count, get_context
import warnings
import pandas as pd
from tqdm import tqdm
import heapq
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, sum_labels
from scipy.ndimage import label as ndi_label
from scipy.spatial import cKDTree
from astropy.io import fits, ascii
from astropy.table import Table, vstack
from astropy.convolution import convolve, convolve_fft
from astropy.wcs import WCS
from astropy.utils.decorators import deprecated
from astropy.visualization import simple_norm
from photutils.background import Background2D, BackgroundBase
from photutils.background.core import SExtractorBackground, StdBackgroundRMS
from photutils.segmentation import (
    make_2dgaussian_kernel,
    detect_sources,
    deblend_sources,
)
from photutils.segmentation import SegmentationImage, SourceCatalog
from photutils.segmentation.deblend import (
    _SingleSourceDeblender,
    _DeblendParams,
    _get_labels,
)
from photutils.segmentation.utils import _make_binary_structure
from photutils.segmentation.detect import _detect_sources
from photutils.utils import calc_total_error
from dataclasses import dataclass
import datetime
import subprocess
import multiprocessing as mp
from astropy.io import fits
from functools import reduce
from functools import partial
from configparser import ConfigParser
from astropy.table import Table, vstack
import codecs
from glob import glob

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
# from . import derive_class_star_from_segmentation as ss


__all__ = [
    "SExtractor",
    "SExtractor_dualmode",
    "SExtractor_HDR",  # Highly-Assembled API
    "se_background",
    "se_clean",
    "se_deblend",  # Base-level Function
    "cat2tab",
    "ds9reg",
    "sexcomb",
    "se_make_kronmask",  # Utility Function
]  # External APIs

KERNEL_DEFAULT = np.array(
    [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
)  # SExtractor defualt convolution kernel (default.conv): 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels


@dataclass
class TileInfo:
    """分块信息"""

    id: int
    y_start: int
    y_end: int
    x_start: int
    x_end: int
    has_overlap: bool = True


class sex:

    __all__ = [
        "SExtractor",
        "SExtractor_dualmode",
        "SExtractor_HDR",  # Highly-Assembled API
        "se_background",
        "se_clean",
        "se_deblend",  # Base-level Function
        "cat2tab",
        "ds9reg",
        "sexcomb",
        "se_make_kronmask",  # Utility Function
    ]  # External APIs

    KERNEL_DEFAULT = np.array(
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    )  # SExtractor defualt convolution kernel (default.conv): 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels

    def __init__(
        self,
        filename,  # Image data
        catalog_name="default_gx_cat",  # output catalog table
        detect_minarea: int = 5,
        detect_thresh: float = 1.5,
        kernel: np.ndarray = KERNEL_DEFAULT,
        detect_connectivity: {8, 4} = 8,  # Detection parameters
        deblend: bool = False,
        deblend_nthresh: float = 32,
        deblend_mincont: float = 0.005,
        deblend_mode: {"exponential", "linear", "log", "asinh"} = "exponential",
        nproc: int = 1,  # Deblending parameters
        clean: bool = False,
        clean_param: float = 1.0,  # Cleaning parameters
        mask=None,
        coverage_mask=None,  # Mask parameters
        back_type: bool = False,
        back_value=0.0,
        back_size: int = 64,
        back_filtersize: int = 3,
        bkg_estimator=SExtractorBackground(),  # Background parameters
        weight_type: {
            "NONE",
            "BACKGROUND",
            "MAP_RMS",
            "MAP_WEIGHT",
            "MAP_VAR",
        } = "NONE",
        weight_name=None,  # Error image
        checkimage_type: list[str] = [],
        phot_apertures=None,
        phot_autoparams: list[float] = [2.5, 1.4],
        phot_petroparams: list[float] = [2.0, 3.5],
        mag_zeropoint: float = 38.951,  # Photometry parameters
        wcs: WCS = None,
        gain: float = 0.0,
        pixel_scale: float = 0.03,  # image level parameters
        MEMORY_BUFSIZE: int = 1024,
        nnw_sex: str = "./default.nnw",
        fwhm_arcsec: float = 0.16,
        verbose: bool = False,
    ):
        self.nnw_sex = nnw_sex
        self.nproc = nproc
        self.deblend_mode = deblend_mode
        self.filename = filename
        self.catalog_name = catalog_name
        self.detect_minarea = detect_minarea
        self.detect_thresh = detect_thresh
        self.detect_connectivity = detect_connectivity
        self.deblend_nthresh = deblend_nthresh
        self.deblend_mincont = deblend_mincont
        self.clean_param = clean_param
        self.back_size = back_size
        self.back_type = back_type
        self.back_value = back_value
        self.kernel = kernel
        self.back_filtersize = back_filtersize
        self.pixel_scale = pixel_scale
        self.weight_name = weight_name
        self.memory_bufsize = MEMORY_BUFSIZE
        self.bkg_estimator = bkg_estimator
        self.coverage_mask = coverage_mask
        self.weight_type = weight_type
        self.verbose = verbose
        self.mask = mask
        self.deblend = deblend
        self.clean = clean
        self.gain = gain
        self.phot_apertures = phot_apertures
        self.mag_zeropoint = mag_zeropoint
        self.fwhm = fwhm_arcsec
        self.phot_autoparams=phot_autoparams
        # 处理统计
        self.stats = {
            "total_tiles": 0,
            "processed_tiles": 0,
            "failed_tiles": 0,
            "total_sources": 0,
            "processing_time": 0,
            "memory_used_mb": 0,
        }

    def prepare_img(self):
        print('+++++Warining!!!!')
        print('Current Buff have not dealed with several parameters in the catalog.')
        print('1) global rms')
        print('2) chid, obj_merged, child_to_parent, parent_to_children = compute_obj_merged_from_parent_child(segm.data,segm.data)  ')
        print('The first segm.data should use segm before deblending.')
        
        print("process ++++++++ prepare_img")
        hdu = fits.open(self.filename)[0]
        header = hdu.header
        data = hdu.data

        wcs = WCS(header)

        imgshape = np.shape(data)

        if self.weight_name is not None:
            wht_img = fits.open(self.weight_name)[0].data

            dataproducts = {
                "data": data,
                "header": header,
                "wht": wht_img,
                "wcs": wcs,
                "shape": imgshape,
            }
        else:
            dataproducts = {
                "data": data,
                "header": header,
                "wht": None,
                "wcs": wcs,
                "shape": imgshape,
            }

        if self.mask is not None:
            mask_img = fits.open(self.mask)[0].data
            dataproducts["mask"] = mask_img
        else:
            dataproducts["mask"] = None

        if dataproducts.get("wht") is not None:
            weight_image = dataproducts["wht"]  # weight for tile
            if self.mask is not None:
                self.mask |= weight_image == 0
            else:
                self.mask = weight_image == 0
        else:
            weight_image = None

        bkg, bkg_rms, global_rms = se_background(
            data,
            back_size=self.back_size,
            back_filtersize=self.back_filtersize,
            bkg_estimator=self.bkg_estimator,
            mask=self.mask,
            coverage_mask=self.coverage_mask,
            weight_type=self.weight_type,
            weight_image=weight_image,  # for different weight types, the background and background RMS are different
            verbose=self.verbose,
        )  # bkg_rms will be used in thresolding, cleaning, and fluxerr
        # dataproducts['bkg'] = bkg.background
        dataproducts["bkg_rms"] = bkg_rms

        if self.back_type:  # Automatic background subtraction
            background = bkg.background
            image_bkgsub = data - background
            dataproducts["data"] = image_bkgsub
            dataproducts["bkg"] = bkg.background
            dataproducts["bkg_type"] = "2d"
            # if verbose:
            #     print('Automatic Background Estimation and Subtraction Finished')
            #     print(f'  background size = {back_size}')
            #     print(f'  background filter size = {back_filtersize}')
            #     print('\n'+'=='*20+'\n')
        else:  # Manual background subtraction
            if np.isscalar(self.back_value):
                background = np.full_like(data, self.back_value)
                image_bkgsub = data - background
                dataproducts["data"] = image_bkgsub
                dataproducts["bkg"] = bkg.background
                dataproducts["bkg_type"] = "2d"

            elif (
                isinstance(self.back_value, np.ndarray)
                and self.back_value.shape == data.shape
            ):
                background = self.back_value

                image_bkgsub = data - background
                dataproducts["data"] = image_bkgsub
                dataproducts["bkg"] = background
                dataproducts["bkg_type"] = "1d"

            else:
                raise ValueError(
                    "back_value should be a scalar or a 2D array with the same shape as the image."
                )

        if self.kernel is not None:
            image_convolved = convolve_fft(image_bkgsub, self.kernel, mask=self.mask,preserve_nan=True)
            dataproducts["conv"] = image_convolved
        else:
            image_convolved = image_bkgsub
            dataproducts["conv"] = image_convolved

        if self.weight_type != "NONE":
            # threshold = detect_thresh * convolve_fft(bkg_rms, detect_filter, mask=mask) # make sure the detection threshold is convolved with the same kernel as the image
            if self.kernel is not None:
                threshold = self.detect_thresh * convolve_fft(
                    bkg_rms, self.kernel, mask=self.mask,preserve_nan=True
                )
            else:
                threshold = self.detect_thresh * bkg_rms
        else:
            threshold = self.detect_thresh * bkg_rms

        dataproducts["threshold"] = threshold

        print("process ++++++++ prepare_img, done")
        return dataproducts

    def create_tiles(self, dataproducts):

        print("process ++++++++ create_tiles, create tile information")

        height, width = dataproducts["shape"]
        tile_height = self.memory_bufsize
        tile_width = width

        # overlap_y = int(tile_height/3)

        overlap_y = 10

        overlap_x = 0

        # 计算分块数量
        n_tiles_y = max(
            1, (height + tile_height - overlap_y - 1) // (tile_height - overlap_y)
        )
        n_tiles_x = max(
            1, (width + tile_width - overlap_x - 1) // (tile_width - overlap_x)
        )

        # 调整分块大小以避免最后的小分块
        actual_tile_height = (height + overlap_y * (n_tiles_y - 1)) // n_tiles_y
        actual_tile_width = (width + overlap_x * (n_tiles_x - 1)) // n_tiles_x

        tiles = []
        tile_id = 0

        for i in range(n_tiles_y):
            for j in range(n_tiles_x):
                # 计算分块起始和结束位置
                y_start = i * (actual_tile_height - overlap_y)
                y_end = min(y_start + actual_tile_height, height)

                x_start = j * (actual_tile_width - overlap_x)
                x_end = min(x_start + actual_tile_width, width)

                # 调整起始位置，确保分块大小一致
                if y_end - y_start < actual_tile_height and i > 0:
                    y_start = max(0, y_end - actual_tile_height)
                if x_end - x_start < actual_tile_width and j > 0:
                    x_start = max(0, x_end - actual_tile_width)

                # 创建分块信息
                tile_info = TileInfo(
                    id=tile_id,
                    y_start=y_start,
                    y_end=y_end,
                    x_start=x_start,
                    x_end=x_end,
                )
                tiles.append(tile_info)
                tile_id += 1

        self.stats["total_tiles"] = len(tiles)

        # if self.verbose:
        #     print(f"创建了 {len(tiles)} 个分块")
        #     for i, tile in enumerate(tiles[:3]):  # 只显示前3个
        #         print(f"  分块 {i}: ({tile.y_start}:{tile.y_end}, {tile.x_start}:{tile.x_end})")
        #     if len(tiles) > 3:
        #         print(f"  ... 和 {len(tiles) - 3} 个更多分块")
        print("process ++++++++ create_tiles, done")
        return tiles

    def extract_tile(self, dataproducts, tiles, verbose: bool = False, savetiles=False):

        print("process ++++++++ extract_tile, extract tile subsets")

        header = dataproducts["header"]
        # wcs = dataproducts['wcs']

        if dataproducts.get("wht") is not None:
            wht = dataproducts["wht"]

        if dataproducts.get("mask") is not None:
            mask = dataproducts["mask"]

        label = 0

        tilefiles = []
        for tile_info in tiles:

            hh = header.copy()

            crpix1 = hh["CRPIX1"]
            crpix2 = hh["CRPIX2"]
            img = dataproducts["data"]

            # print(crpix1,crpix2)

            # 提取分块
            x_start = tile_info.x_start
            # x_end   = tile_info.x_end
            y_start = tile_info.y_start
            # y_end   = tile_info.y_end
            img_cut = img[
                tile_info.y_start : tile_info.y_end, tile_info.x_start : tile_info.x_end
            ]

            hh["CRPIX1"] = crpix1 - x_start
            hh["CRPIX2"] = crpix2 - y_start

            # tilehdu = fits.HDUList(fits.PrimaryHDU(img_cut,header=header))
            # tilehdu ='/mnt/f/test_buffer/temp2/tile'+str(label)+'.fits'
            # tilefile.append(tilehdu)

            if dataproducts.get("wht") is not None:
                wht_cut = wht[
                    tile_info.y_start : tile_info.y_end,
                    tile_info.x_start : tile_info.x_end,
                ]
                tilefile = {
                    "image": img_cut,
                    "header": hh,
                    "wht": wht_cut,
                }  # need to add wht

            else:
                tilefile = {
                    "image": img_cut,
                    "header": hh,
                    "wht": None,
                }  # need to add wht

            if self.mask is not None:
                mask_cut = self.mask[
                    tile_info.y_start : tile_info.y_end,
                    tile_info.x_start : tile_info.x_end,
                ]
                tilefile["mask"] = mask_cut
            else:
                tilefile["mask"] = None

            if dataproducts["bkg"] is not None:
                if dataproducts["bkg_type"] == "2d":
                    bkg_cut = dataproducts["bkg"][
                        tile_info.y_start : tile_info.y_end,
                        tile_info.x_start : tile_info.x_end,
                    ]
                    tilefile["bkg"] = bkg_cut
                    tilefile["bkg_type"] = "2d"
                else:
                    tilefile["bkg"] = self.back_value
                    tilefile["bkg_type"] = "1d"
            else:
                tilefile["bkg"] = None

            if dataproducts["bkg_rms"] is not None:
                bkgrms_cut = dataproducts["bkg_rms"][
                    tile_info.y_start : tile_info.y_end,
                    tile_info.x_start : tile_info.x_end,
                ]
                tilefile["bkg_rms"] = bkgrms_cut
            else:
                tilefile["bkg_rms"] = None

            if dataproducts["conv"] is not None:
                conv_cut = dataproducts["conv"][
                    tile_info.y_start : tile_info.y_end,
                    tile_info.x_start : tile_info.x_end,
                ]
                tilefile["conv"] = conv_cut
            else:
                tilefile["conv"] = None

            thre_cut = dataproducts["threshold"][
                tile_info.y_start : tile_info.y_end, tile_info.x_start : tile_info.x_end
            ]
            tilefile["threshold"] = thre_cut

            tilefiles.append(tilefile)

            if savetiles:
                fits.writeto(
                    "/mnt/f/test_buffer/temp2/tile" + str(label) + ".fits",
                    img_cut,
                    hh,
                    overwrite=True,
                )

            label = label + 1

        print("process ++++++++ extract_tile, done")
        return tilefiles

    def _sextractor_tile(self, dataproducts, tile_nn, tile_index):

        wcs = WCS(tile_nn["header"])
        image = tile_nn["image"]
        mask = tile_nn["mask"]

        background = tile_nn["bkg"]
        bkg_rms = tile_nn["bkg_rms"]
        image_convolved = tile_nn["conv"]

        image_bkgsub = tile_nn["image"]

        # 1. Calculate the background
        if dataproducts.get("wht") is not None:
            weight_image = tile_nn["wht"]  # weight for tile
            if self.mask is not None:
                mask |= weight_image == 0
            else:
                mask = weight_image == 0
        else:
            weight_image = None

        threshold = tile_nn["threshold"]
        # print('threshold',threshold)
        segment_img = detect_sources(
            data=image_convolved,
            threshold=threshold,
            npixels=self.detect_minarea,
            connectivity=self.detect_connectivity,
            mask=mask,
        )

        # print('detection takes '+str(round((time1-time0)/60,4))+' mins')
        print("detection done for tile" + str(tile_index))

        # 4. Deblend the sources using Photutils.segmentation.deblend_sources if required
        if self.deblend:
            segment_img = se_deblend(
                data=image_convolved,
                segment_img=segment_img,
                convolved_data=image_convolved,
                threshold=threshold,
                detect_minarea=self.detect_minarea,
                deblend_nthresh=self.deblend_nthresh,
                deblend_mincont=self.deblend_mincont,
                detect_connectivity=self.detect_connectivity,
                mode=self.deblend_mode,
                nproc=self.nproc,
                verbose=self.verbose,
            )

        # print('deblend takes '+str(round((time1-time0)/60,4))+' mins')
        print("deblend done for tile" + str(tile_index))

        # 5. Clean spurious detection
        if self.clean:
            # Q ? will self.mask be modified?
            cat_spur = SourceCatalog(
                data=image_bkgsub,
                segment_img=segment_img,
                convolved_data=image_convolved,
                error=None,
                mask=mask,
                background=background,
            )
            segment_img_cln = se_clean(
                cat_spur,
                threshold=threshold,
                clean_param=self.clean_param,
                detect_minarea=self.detect_minarea,
                verbose=self.verbose,
            )
            segment_img_cln.relabel_consecutive()
        else:
            segment_img_cln = segment_img

        # print('clean takes '+str(round((time1-time0)/60,4))+' mins')
        print("clean done for tile" + str(tile_index))

        # 6. Create the SExtractor source catalog based on Photutils.segmentation.SourceCatalog
        error = calc_total_error(image_bkgsub, bkg_rms, self.gain)
        cat = SourceCatalog(
            data=image_bkgsub,
            segment_img=segment_img_cln,
            convolved_data=image_convolved,
            error=error,
            mask=mask,
            background=background,
            wcs=wcs,
        )
        cat.add_extra_property("dthresh", _dthresh(cat, threshold))

        # print('create catalogs takes '+str(round((time1-time0)/60,4))+' mins')
        print("create catalog done for tile" + str(tile_index))

        # 7. convert SourceCatalog to Table, and add other measurement parameters
        # tab = cat.to_table(columns=DEFAULT_COLUMNS)
        tab = cat.to_table()
        # tab = cat2tab(cat, phot_apertures=self.phot_apertures, mag_zeropoint=self.mag_zeropoint)

        # print('convert table takes '+str(round((time1-time0)/60,4))+' mins')
        print("convert table done for tile" + str(tile_index))

        return tab, segment_img_cln, cat

        # return segment_img

    def sextractor_tile(self, dataproducts, tilefiles):

        ntiles = len(tilefiles)
        tablist = []
        segmlist = []
        catlist = []
        for jj in tqdm(range(ntiles)):

            print("process ++++ sextractor_tile, do source detection on tiles")
            print("processing tile" + str(jj))

            tile_jj = tilefiles[jj]
            tab_jj, segment_img_cln_jj, cat_jj = self._sextractor_tile(
                dataproducts, tile_jj, jj
            )

            print("sex tile " + str(jj) + " done")

            tablist.append(tab_jj)
            segmlist.append(segment_img_cln_jj)
            catlist.append(cat_jj)

            # fits.writeto('/mnt/f/test_buffer/temp2/tile'+str(jj)+'_segm_cln.fits',segment_img_cln_jj.data,overwrite=True)
            # print ('file saved')

            # print('tile'+str(jj)+' done, temp files saved.')

        return tablist, segmlist, catlist

    import numpy as np
    from typing import List, Dict, Any

    def relabel_segmentation(self, seg_map):

        max_label = seg_map.max()

        if max_label == 0:
            return seg_map.copy()

        lut = np.zeros(max_label + 1, dtype=seg_map.dtype)

        unique_labels = np.unique(seg_map)
        unique_labels = unique_labels[unique_labels != 0]

        for idx, label in enumerate(unique_labels, start=1):
            lut[label] = idx

        return lut[seg_map]

    def merge_tiles(
        self,
        dataproducts: Dict[str, Any],
        tablist: List[Dict],
        segmlist: List[Any],
        tiles: List[Any],
    ) -> np.ndarray:

        # 初始化变量
        up_segmlist = [segmlist[0].data]
        segmcomb = np.zeros(dataproducts["shape"], dtype=np.int32)
        segmtemp = np.zeros(dataproducts["shape"], dtype=np.int32)

        lenn = 0
        total_tiles = len(tiles)

        for nn in range(total_tiles - 1):
            # 获取当前tile和下一个tile的信息
            tile0_info = tiles[nn]
            tile1_info = tiles[nn + 1]

            tile0_segm = up_segmlist[nn]  # 使用最新的标签
            tile1_segm = segmlist[nn + 1].data

            # 统计标签数量
            len0 = len(tablist[nn]["label"])
            lenn += len0

            # 将当前tile的分割图放到组合图像中
            segmcomb[
                tile0_info.y_start : tile0_info.y_end,
                tile0_info.x_start : tile0_info.x_end,
            ] = tile0_segm

            # 将下一个tile的分割图放到临时图像中
            segmtemp[
                tile1_info.y_start : tile1_info.y_end,
                tile1_info.x_start : tile1_info.x_end,
            ] = tile1_segm

            # 提取重叠区域
            y_overlap_start = tile1_info.y_start
            y_overlap_end = min(tile0_info.y_end, tile1_info.y_end)
            x_overlap_start = tile1_info.x_start
            x_overlap_end = tile1_info.x_end

            overlap0 = segmcomb[
                y_overlap_start:y_overlap_end, x_overlap_start:x_overlap_end
            ]
            overlap1 = segmtemp[
                y_overlap_start:y_overlap_end, x_overlap_start:x_overlap_end
            ]

            # 使用向量化操作找到重叠区域的标签对应关系
            # 获取overlap1中非零像素的位置
            mask = overlap1 >= 1

            if np.any(mask):
                # 获取对应的标签对
                labels1 = overlap1[mask]
                labels0 = overlap0[mask]

                # 创建映射字典 (tile1标签 -> tile0标签)
                # 使用np.unique去除重复，保持第一次出现的映射
                unique_pairs = np.column_stack([labels1, labels0])
                # 使用字典推导式创建映射，后面的值会覆盖前面的相同key
                label_map = {int(k): int(v) for k, v in unique_pairs}

                # 获取唯一的标签列表
                oldlist = np.array(list(label_map.keys()))
                newlist = np.array(list(label_map.values()))

                # print("Old labels:", oldlist)
                # print("New labels:", newlist)

                # 创建新的分割图
                segmtemp3 = np.zeros_like(tile1_segm)

                # 向量化处理：先创建所有像素的映射
                tile1_flat = tile1_segm.flatten()
                segmtemp3_flat = segmtemp3.flatten()

                # 创建布尔掩码用于向量化操作
                non_zero_mask = tile1_flat >= 1

                if np.any(non_zero_mask):
                    # 获取非零像素的标签
                    non_zero_labels = tile1_flat[non_zero_mask]

                    # 创建结果数组
                    result_labels = np.zeros_like(non_zero_labels)

                    # 找到在映射表中的标签
                    in_map_mask = np.isin(non_zero_labels, oldlist)

                    if np.any(in_map_mask):
                        # 处理在映射表中的标签
                        in_map_labels = non_zero_labels[in_map_mask]
                        # 使用向量化方式查找映射
                        for i, label in enumerate(in_map_labels):
                            idx = np.where(oldlist == label)[0]
                            if len(idx) > 0:
                                result_labels[np.where(in_map_mask)[0][i]] = newlist[
                                    idx[0]
                                ]

                    # 处理不在映射表中的标签
                    not_in_map_mask = ~in_map_mask
                    if np.any(not_in_map_mask):
                        result_labels[not_in_map_mask] = (
                            non_zero_labels[not_in_map_mask] + lenn
                        )

                    # 将结果放回segmtemp3
                    segmtemp3_flat[non_zero_mask] = result_labels

                # 重置segmtemp3为正确的形状
                segmtemp3 = segmtemp3_flat.reshape(tile1_segm.shape)

                # 将重叠区域置为0（因为已经用overlap0处理了）
                overlap_height = y_overlap_end - y_overlap_start
                segmtemp3[:overlap_height, :] = 0

                up_segmlist.append(segmtemp3)
            else:
                # 没有重叠区域，直接使用原始分割图（加上偏移）
                segmtemp3 = np.where(tile1_segm >= 1, tile1_segm + lenn, 0)
                up_segmlist.append(segmtemp3)

            # 重置临时图像
            segmtemp[
                tile1_info.y_start : tile1_info.y_end,
                tile1_info.x_start : tile1_info.x_end,
            ] = 0

        # 创建最终的分割图
        segmcomb = np.zeros(dataproducts["shape"], dtype=np.int32)
        for nn, segm in enumerate(up_segmlist):
            tile_info = tiles[nn]
            segmcomb[
                tile_info.y_start : tile_info.y_end, tile_info.x_start : tile_info.x_end
            ] += segm

            # 保存中间结果（调试用）
            # if hasattr(self, 'debug') and self.debug:
            #     fits.writeto(f'./temp2/upsegm_buffer4_{nn}.fits', segmcomb, overwrite=True)

        segup = self.relabel_segmentation(segmcomb)

        return segup

    def finalcat(self, dataproducts, segcomb):

        catalogname = self.catalog_name

        image_bkgsub = dataproducts["data"]
        bkg_rms = dataproducts["bkg_rms"]
        mask = dataproducts["mask"]
        image_convolved = dataproducts["conv"]
        weight_image = dataproducts['wht']

        threshold = dataproducts["threshold"]
        background = dataproducts["bkg"]
        wcs = dataproducts["wcs"]


        ## this is a cleaned segm image 
        segm = SegmentationImage((segcomb))

        # 6. Create the SExtractor source catalog based on Photutils.segmentation.SourceCatalog
        error = calc_total_error(image_bkgsub, bkg_rms, self.gain)
        cat = SourceCatalog(
            data=image_bkgsub,
            segment_img=segm,
            convolved_data=image_convolved,
            error=error,
            mask=mask,
            background=background,
            wcs=wcs,
        )


        ## the first segm should be raw segm without deblending 
        chid, obj_merged, child_to_parent, parent_to_children = compute_obj_merged_from_parent_child(segm.data,segm.data)  
        # print(chid)
        cat.add_extra_property("obj_merged", obj_merged)
        
        idl = cat.label
        x0l = cat.xcentroid   ### star from 0 
        y0l = cat.ycentroid
        a_imagel=cat.semimajor_sigma
        b_imagel=cat.semiminor_sigma
        theta_degl = cat.orientation
        kron_radiusl = cat.kron_radius*self.phot_autoparams[0]
    
        weight_bad_threshold = 1e9
        obj_crowded=[]
        for nn in range(len(idl)):
            crowded, frac_bad, area, areab = kron_crowded_flag(
                segm.data, weight_image, idl[nn],
                x0l[nn], y0l[nn],
                a_imagel[nn].value, b_imagel[nn].value, theta_degl[nn].value,
                kron_radiusl[nn].value,
                weight_bad_threshold=weight_bad_threshold,
            )
            obj_crowded.append(crowded)
        
        cat.add_extra_property("obj_crowded", obj_crowded)
            
        
        print('obj_crowded',obj_crowded)
        flagl = np.array(obj_merged).astype(int)
        flagf = np.array(obj_crowded).astype(int)
        # for ff1  in range(len(flagf)):
            
        #     flagf[ff1]=update_flag_bitwise(flagf[ff1],flagl[ff1])
        # cat.add_extra_property("combined_flags", flagf)
        
        
        print(flagf)
        # a_int = a.astype(int)
        # b_int = b.astype(int)
        # c_int = c.astype(int)
        # d_int = d.astype(int)
        
        # 位运算组合：d c b a 分别对应 3 2 1 0 位
        # 公式: (d<<3) | (c<<2) | (b<<1) | a
        #result = (d_int << 3) | (c_int << 2) | (b_int << 1) | a_int
        combined_flags= (flagl << 1) | flagf
        cat.add_extra_property("combined_flags", combined_flags)

        threl = _dthresh(cat, threshold)
        cat.add_extra_property("dthresh", threl)

        ### calculate star_class parameter.
        idlist, cstar, peak = class_star_from_segmentation(
            seg=segm.data,
            img_bkgsub=image_bkgsub,
            analysis_thresh=threl,
            fwhm_arcsec=self.fwhm,
            pixscale_arcsec=self.pixel_scale,
            nnw_path=self.nnw_sex,
        )
        print("cstar", cstar)
        cat.add_extra_property("class_star", cstar)

        global_rms = 0
        cat.add_extra_property("global_rms", [global_rms]*len(cat))

        # 7. convert SourceCatalog to Table, and add other measurement parameters
        # tab = cat.to_table(columns=DEFAULT_COLUMNS)
        tab = cat2tab(
            cat, phot_apertures=self.phot_apertures, mag_zeropoint=self.mag_zeropoint
        )

        ascii.write(tab, catalogname, overwrite=True)

        return tab


##########################################
##########################################


def SExtractor(
    filename: str,  # Image data
    catalog_name: str = "default_gx_cat",  # output catalog table
    detect_minarea: int = 5,
    detect_thresh: float = 1.5,
    kernel: Optional[np.ndarray] = KERNEL_DEFAULT,
    detect_connectivity: Literal[4, 8] = 8,  # Detection parameters
    deblend: bool = False,
    deblend_nthresh: float = 32,
    deblend_mincont: float = 0.005,
    deblend_mode: Literal["exponential", "linear", "log", "asinh"] = "exponential",
    nproc: int = 1,  # Deblending parameters
    clean: bool = False,
    clean_param: float = 1.0,  # Cleaning parameters
    mask: Optional[np.ndarray] = None,
    coverage_mask: Optional[np.ndarray] = None,  # Mask parameters
    back_type: bool = False,
    back_value: Union[float, np.ndarray] = 0.0,
    back_size: int = 64,
    back_filtersize: int = 3,
    bkg_estimator: BackgroundBase = SExtractorBackground(),  # Background parameters
    weight_type: Literal[
        "NONE", "BACKGROUND", "MAP_RMS", "MAP_WEIGHT", "MAP_VAR"
    ] = "NONE",
    weight_name: Optional[str] = None,  # Error image
    checkimage_type: List[str] = [],
    phot_apertures: Optional[Union[float, List[float]]] = None,
    phot_autoparams: List[float] = [2.5, 1.4],
    phot_petroparams: List[float] = [2.0, 3.5],
    mag_zeropoint: float = 38.951,  # Photometry parameters
    wcs: Optional[WCS] = None,
    gain: float = 0.0,
    pixel_scale: float = 0.06,  # image level parameters
    nnw_sex: str = "./default.nnw",
    fwhm_arcsec: float = 0.16,
    verbose: bool = False,
) -> tuple[Table, SegmentationImage, SourceCatalog]:
    """
    Python equivilant to perform source detection following the SExtractor methodology.
    All parameters are named after the SExtractor parameters, except for new features added by Photutils.

    Parameters
    ----------
    filename: str
        The path to the image fits file.
    catalog_name : str, optional
        Output filename for the ASCII catalog (default: "default_gx_cat")
    detect_minarea : int, optional
        The minimum number of pixels required for a detection, (default: 5) .
    detect_thresh : float, optional
        The threshold value for detection (default: 1.5).
    kernel: np.ndarray or None, optional
        Convolution kernel for filtering before detection.
        If None, no convolution is applied. Default is a 3×3 Gaussian-like kernel.
    detect_connectivity : {8, 4} (default: 8)
        The type of pixel connectivity used in determining how pixels are
        grouped into a detected source. The options are 4 or 8 (default).
        4-connected pixels touch along their edges. 8-connected pixels touch
        along their edges or corners.
    deblend : bool, optional
        Whether to perform deblending (default: False).
    deblend_nthresh : float, optional
        The number of thresholds used for object deblending (default: 32).
    deblend_mincont : float, optional
        The minimum contrast ratio used for object deblending (default: 0.005).
    deblend_mode: {'exponential', 'linear', 'log', 'asinh'}, optional
        The mode used for deblending (default: 'exponential').
    nproc : int, optional
        The number of processors used for parallel deblending (default: 1).
    clean : bool, optional
        Clean spurious detections. Perform cleaning if True (default: False).
    clean_param : float, optional
        Cleaning efficiency parameter (default: 1.0) .
    mask : np.ndarray or None, optional
        Boolean mask of bad pixels (True = masked). Default is None.
    coverage_mask : np.ndarray or None, optional
        Boolean mask for coverage (used in background estimation). Default is None.
    back_type : bool, optional
        Perform automatic background estimation if True. Otherwise, the background is subtracted manually by back_value.
    back_value : float or np.ndarray, optional
        The background value (scalar or 2D array) for manual background subtraction,  when `back_type=False`.
        Default is 0.0.
    back_size : int, optional
        The size of the background mesh box (default: 64).
    back_filtersize : int, optional
        The filter size for background smoothing (default: 3).
    bkg_estimator : BackgroundEstimator, optional
        Photutils background estimator (default: SExtractorBackground()).
    weight_type : {'NONE', 'BACKGROUND', 'MAP_RMS', 'MAP_WEIGHT', 'MAP_VAR'}
        The type of weighting to apply (default: 'NONE').
    weight_image : str or None, optional
        Filename of weight map (required if `weight_type` != 'NONE').
    checkimage_type: list of str, optional
        Types of check images to produce (e.g., 'background', 'segmentation').
        Valid types are listed in the code. Default is empty list.
    phot_apertures : float or list of float, optional
        The radius of the circle in pixels for apeture photometry. Default is None.
    phot_autoparams : list of float, optional
        The parameters [factor, minradius] used for the AUTO photometry (default: [2.5, 3.5]).
    phot_petroparams : list of float, optional
        [nsigma, minradius] for Petrosian photometry (default: [2.0, 3.5]).
    mag_zeropoint : float, optional
        The magnitude zeropoint (default: 38.951).
    wcs : WCS or None, optional
        World Coordinate System object. If None, extracted from FITS header.
    gain : float, optional
        Gain (e⁻/ADU) for error propagation (default: 0.0).
    pixel_scale : float, optional
        The pixel scale of the image (default: 0.06).
    verbose : bool, optional
        If True, print progress information (default: False).

    Returns
    -------
    table : astropy.table.Table
        Source catalog containing SExtractor‑like measurements (e.g., fluxes,
        magnitudes, shapes) for each detected object.
    segment_img : photutils.segmentation.SegmentationImage
        Final segmentation image after detection and optional deblending/cleaning.
        Each source is assigned a unique positive integer label.
    catalog : photutils.segmentation.SourceCatalog
        Photutils sourcecatalog object
    """

    # 8 different type of check image, more can be added in the future
    all_checkimage_type = [
        "background",
        "minibackground",
        "background_rms",
        "miniback_rms",
        "-background",
        "filtered",
        "segmentation",
        "apertures",
    ]
    # if there are invalid checktype, return the error;
    if checkimage_type is not None:
        invalid = [x for x in checkimage_type if x not in all_checkimage_type]
        if invalid:
            raise ValueError(f"{invalid} is unknown checktype keywords")

    if verbose:
        print("\n" + "Source Extraction Begins.......")
        print(f"detection image: {filename}")
        print("\n" + "==" * 20 + "\n")

    # 0. Reading the fits file; image:2D array
    image, header = fits.getdata(filename, header=True)
    if wcs is None:
        wcs = WCS(header)

    # 1. Calculate the background
    if weight_name is not None:
        weight_image = fits.getdata(weight_name)
        if mask is not None:
            mask |= weight_image == 0
        else:
            mask = weight_image == 0
    else:
        weight_image = None

    bkg, bkg_rms, global_rms = se_background(
        image,
        back_size=back_size,
        back_filtersize=back_filtersize,
        bkg_estimator=bkg_estimator,
        mask=mask,
        coverage_mask=coverage_mask,
        weight_type=weight_type,
        weight_image=weight_image,  # for different weight types, the background and background RMS are different
        verbose=verbose,
    )  # bkg_rms will be used in thresolding, cleaning, and fluxerr

    if verbose:
        print("Weight Image Preprocessing Finished")
        print(f"  weight_type = {weight_type}")
        # print(f'  scale_factor = {scale_factor}')
        print("\n" + "==" * 20 + "\n")

    if back_type:  # Automatic background subtraction
        background = bkg.background
        image_bkgsub = image - background
        if verbose:
            print("Automatic Background Estimation and Subtraction Finished")
            print(f"  background size = {back_size}")
            print(f"  background filter size = {back_filtersize}")
            print("\n" + "==" * 20 + "\n")
    else:  # Manual background subtraction
        if np.isscalar(back_value):
            background = np.full_like(image, back_value)
        elif isinstance(back_value, np.ndarray) and back_value.shape == image.shape:
            background = back_value
        else:
            raise ValueError(
                "back_value should be a scalar or a 2D array with the same shape as the image."
            )
        image_bkgsub = image - background
        if verbose:
            print("Manual Background Subtraction Finished")
            if np.isscalar(back_value):
                print(f"  Background value = {back_value}")
            print("\n" + "==" * 20 + "\n")

    if "background" in checkimage_type:
        if back_type:
            fits.writeto(
                "background.fits", bkg.background, header=header, overwrite=True
            )
        else:
            fits.writeto(
                "background.fits",
                np.ones_like(image) * back_value,
                header=header,
                overwrite=True,
            )
    if "-background" in checkimage_type:
        fits.writeto("background_sub.fits", image_bkgsub, header=header, overwrite=True)

    if "minibackground" in checkimage_type:
        fits.writeto(
            "minibackground.fits", bkg.background_mesh, header=header, overwrite=True
        )
    if "background_rms" in checkimage_type:
        fits.writeto(
            "background_rms.fits", bkg.background_rms, header=header, overwrite=True
        )
    if "miniback_rms" in checkimage_type:
        fits.writeto(
            "minibackground_rms.fits",
            bkg.background_rms_mesh,
            header=header,
            overwrite=True,
        )

    # 2. Convolve the image with a small kernel
    if kernel is not None:
        image_convolved = convolve(image_bkgsub, kernel, mask=mask)
    else:
        image_convolved = image_bkgsub
    # image_convolved = convolve_fft(image_bkgsub, detect_filter, mask=mask)
    if "filtered" in checkimage_type:
        fits.writeto("filtered.fits", image_convolved, header=header, overwrite=True)

    # 3. Detect the sources using Photutils.segmentation.detect_sources
    if weight_type != "NONE":
        # threshold = detect_thresh * convolve_fft(bkg_rms, detect_filter, mask=mask) # make sure the detection threshold is convolved with the same kernel as the image
        if kernel is not None:
            threshold = detect_thresh * convolve(bkg_rms, kernel, mask=mask)
        else:
            threshold = detect_thresh * bkg_rms
    else:
        threshold = detect_thresh * bkg_rms


    ### segm before deblend
    segment_img = detect_sources(
        data=image_convolved,
        threshold=threshold,
        npixels=detect_minarea,
        connectivity=detect_connectivity,
        mask=mask,
    )
    
    segm_raw = segment_img
    
    # fits.writeto('seg_raw.fits',segment_img.data,overwrite=True)

    # avoid non detections if det_thresh is too high.
    attempt = 0
    max_attempts = 100
    while segment_img is None and attempt < max_attempts:
        print(
            "Warning: "
            + f"  detect_thresh = {detect_thresh}"
            + " maybe too high and no dectection found."
        )
        print("Reducing the detect_thresh, trying " + str(round(detect_thresh / 2, 1)))
        detect_thresh = round(detect_thresh / 2, 1)
        threshold = threshold / 2
        segment_img = detect_sources(
            data=image_convolved,
            threshold=threshold,
            npixels=detect_minarea,
            connectivity=detect_connectivity,
            mask=mask,
        )
        attempt += 1

    if verbose:
        print("Detection Finished")
        # print(f"  kernel_size = {detect_filtersize}")
        print(f"  detect_minarea = {detect_minarea}")
        print(f"  detect_thresh = {detect_thresh}")
        print(f"  threshold (above background) = {np.nanmedian(threshold)}")
        print(f"Found {segment_img.nlabels} sources.")
        print("\n" + "==" * 20 + "\n")

    # 4. Deblend the sources using Photutils.segmentation.deblend_sources if required
    if deblend:

        segment_img = se_deblend(
            data=image_convolved,
            segment_img=segment_img,
            convolved_data=image_convolved,
            threshold=threshold,
            detect_minarea=detect_minarea,
            deblend_nthresh=deblend_nthresh,
            deblend_mincont=deblend_mincont,
            detect_connectivity=detect_connectivity,
            mode=deblend_mode,
            nproc=nproc,
            verbose=verbose,
        )
      

    if verbose:
        if deblend:
            print("Deblending Finished")
            print(f"  deblend_nthresh = {deblend_nthresh}")
            print(f"  deblend_mincont = {deblend_mincont}")
            print(f"  deblend_connectivity = {detect_connectivity}")
            print(f"  deblend_mode = {deblend_mode}")
            print(f"  nproc = {nproc}")
            print(f"Found {segment_img.nlabels} sources after deblending.")
        else:
            print("Deblending skipped.")
        print("\n" + "==" * 20 + "\n")

    # 5. Clean spurious detection
    if clean:
        cat_spur = SourceCatalog(
            data=image_bkgsub,
            segment_img=segment_img,
            convolved_data=image_convolved,
            error=None,
            mask=mask,
            background=background,
        )
        segment_img_cln = se_clean(
            cat_spur,
            threshold=threshold,
            clean_param=clean_param,
            detect_minarea=detect_minarea,
            verbose=verbose,
        )
        segment_img_cln.relabel_consecutive()
    else:
        segment_img_cln = segment_img


    ### segment_img_cln is our final segmt image


    if verbose:
        if clean:
            print("Cleaning Finished")
            print(f"  clean_param = {clean_param}")
            print(
                f"Removed {segment_img.nlabels - segment_img_cln.nlabels} spurious sources by cleaning."
            )
        else:
            print("Cleaning skipped.")
        print("\n" + "==" * 20 + "\n")
        print(f"There are {segment_img_cln.nlabels} objects in the catalog." + "\n")

    if "segmentation" in checkimage_type:
        fits.writeto(
            "segmentation.fits", segment_img_cln.data, header=header, overwrite=True
        )

    # 6. Create the SExtractor source catalog based on Photutils.segmentation.SourceCatalog
    error = calc_total_error(image_bkgsub, bkg_rms, gain)
    cat = SourceCatalog(
        data=image_bkgsub,
        segment_img=segment_img_cln,
        convolved_data=image_convolved,
        error=error,
        mask=mask,
        background=background,
        wcs=wcs,
        kron_params= phot_autoparams
    )



    chid, obj_merged, child_to_parent, parent_to_children = compute_obj_merged_from_parent_child(segm_raw.data,segment_img_cln.data)
    # print(chid)
    cat.add_extra_property("obj_merged", obj_merged)
    
    idl = cat.label
    x0l = cat.xcentroid   ### star from 0 
    y0l = cat.ycentroid
    a_imagel=cat.semimajor_sigma
    b_imagel=cat.semiminor_sigma
    theta_degl = cat.orientation
    kron_radiusl = cat.kron_radius*phot_autoparams[0]

    weight_bad_threshold = 1e9
    obj_crowded=[]
    for nn in range(len(idl)):
        crowded, frac_bad, area, areab = kron_crowded_flag(
            segment_img_cln.data, weight_image, idl[nn],
            x0l[nn], y0l[nn],
            a_imagel[nn].value, b_imagel[nn].value, theta_degl[nn].value,
            kron_radiusl[nn].value,
            weight_bad_threshold=weight_bad_threshold,
        )
        obj_crowded.append(crowded)
    
    cat.add_extra_property("obj_crowded", obj_crowded)
        
    
    print('obj_crowded',obj_crowded)
    flagl = np.array(obj_merged).astype(int)
    flagf = np.array(obj_crowded).astype(int)
    # for ff1  in range(len(flagf)):
        
    #     flagf[ff1]=update_flag_bitwise(flagf[ff1],flagl[ff1])
    # cat.add_extra_property("combined_flags", flagf)
    
    
    print(flagf)
    # a_int = a.astype(int)
    # b_int = b.astype(int)
    # c_int = c.astype(int)
    # d_int = d.astype(int)
    
    # 位运算组合：d c b a 分别对应 3 2 1 0 位
    # 公式: (d<<3) | (c<<2) | (b<<1) | a
    #result = (d_int << 3) | (c_int << 2) | (b_int << 1) | a_int
    combined_flags= (flagl << 1) | flagf
    cat.add_extra_property("combined_flags", combined_flags)
    
    
    threl = _dthresh(cat, threshold)
    cat.add_extra_property("dthresh", threl)
    # cat.add_extra_property("dthresh", _dthresh(cat, threshold))

    if "apertures" in checkimage_type:
        _check_apertures(
            image_bkgsub=image_bkgsub,
            cat=cat,
            header=header,
            ellipse_value=30 * np.nanmedian(bkg.background_rms_mesh),
            save_jpg=False,
        )

    """
    if 'apertures' in checkimage_type:
        ellipse_layer = np.zeros_like(image_bkgsub)
        ellipse_value = 30 * np.nanmedian(bkg.background_rms_mesh)
        height, width = image_bkgsub.shape
        for source in cat:
            if np.isnan(source.kron_radius):
               continue
            x0 = source.xcentroid
            y0 = source.ycentroid
            theta = source.orientation.value
            a = 2.5 * source.kron_radius.value * source.semimajor_sigma.value
            b = 2.5 * source.kron_radius.value * source.semiminor_sigma.value
            phi = np.linspace(0, 2*np.pi, 200)
            theta_rad = np.deg2rad(theta)
            x = x0 + a * np.cos(phi) * np.cos(theta_rad) - b * np.sin(phi) * np.sin(theta_rad)
            y = y0 + a * np.cos(phi) * np.sin(theta_rad) + b * np.sin(phi) * np.cos(theta_rad)
            x_int = np.round(x).astype(int)
            y_int = np.round(y).astype(int)
            valid = (x_int >= 0) & (x_int < width) & (y_int >= 0) & (y_int < height)
            coords = set(zip(x_int[valid], y_int[valid]))
            for xi, yi in coords:
                ellipse_layer[yi, xi] = ellipse_value
        fits.writeto('apertures.fits', image_bkgsub+ellipse_layer, header=header, overwrite=True)
        #generate jpg version of apertures.fits
        scale = 3
        plt.figure(figsize=(width//np.gcd(width,height)*scale, height//np.gcd(width,height)*scale))
        ax = plt.subplot(111)
        vmin = np.percentile(image_bkgsub, 1)
        vmax = np.percentile(image_bkgsub, 99)
        norm = simple_norm(image_bkgsub, stretch='asinh', vmin=vmin, vmax=vmax)
        ax.imshow(image_bkgsub, cmap='gray', norm=norm, origin='lower', interpolation='nearest')
        cat.plot_kron_apertures(axes=ax, color='yellow', lw=0.6)
        plt.savefig('apertures.jpg')
    """

    ### calculate star_class parameter.
    idlist, cstar, peak = class_star_from_segmentation(
        seg=segment_img_cln,
        img_bkgsub=image_bkgsub,
        analysis_thresh=threl,
        fwhm_arcsec=fwhm_arcsec,
        pixscale_arcsec=pixel_scale,
        nnw_path=nnw_sex,
    )
    print("cstar", cstar)
    cat.add_extra_property("class_star", cstar)
    
    cat.add_extra_property("global_rms", [global_rms]*len(cat))

    # 7. convert SourceCatalog to Table, and add other measurement parameters
    tab = cat2tab(cat, phot_apertures=phot_apertures, mag_zeropoint=mag_zeropoint, kron_fact = phot_autoparams[0])

    # 8. generate the output catalog file
    ascii.write(tab, catalog_name, overwrite=True)

    # return source Table and segmentation image even though it is accessible from the catalog via cat._segment_img
    return tab, segment_img_cln, cat


def _check_apertures(
    image_bkgsub: np.ndarray,
    cat: SourceCatalog,
    header: fits.Header,
    ellipse_value: float,
    save_jpg: bool = False,
    norm: Optional[simple_norm] = None,
) -> None:
    """
    Parameters
    ----------
    image_bkgsub: 2D `~numpy.ndarray`
        The background subtracted image
    cat: SourceCatalog
        The final detected source catalog
    header: fits.Header
        Fits header to be written into the output `apertures.fits` file.
    ellipse_value: float
        Values to be filled to aperture edges.
    save_jpg: bool, optional
        If True, generate a JPEG preview (`apertures.jpg`) with the apertures
        plotted in yellow. Default is False.
    norm: `~astropy.visualization.simple_norm` or None, optional
        Normalization object for the image display in the JPEG. If None,
        a default asinh stretch with 1st–99th percentile clipping is used.

    Returns
    -------
    None
        The function writes `apertures.fits` and optionally `apertures.jpg`
        to the current working directory.
    """

    ellipse_layer = np.zeros_like(cat._data)
    height, width = image_bkgsub.shape
    for source in cat:
        if np.isnan(source.kron_radius):
            continue
        x0 = source.xcentroid
        y0 = source.ycentroid
        theta = source.orientation.value
        a = 2.5 * source.kron_radius.value * source.semimajor_sigma.value
        b = 2.5 * source.kron_radius.value * source.semiminor_sigma.value
        phi = np.linspace(0, 2 * np.pi, 200)
        theta_rad = np.deg2rad(theta)
        x = (
            x0
            + a * np.cos(phi) * np.cos(theta_rad)
            - b * np.sin(phi) * np.sin(theta_rad)
        )
        y = (
            y0
            + a * np.cos(phi) * np.sin(theta_rad)
            + b * np.sin(phi) * np.cos(theta_rad)
        )
        x_int = np.round(x).astype(int)
        y_int = np.round(y).astype(int)
        valid = (x_int >= 0) & (x_int < width) & (y_int >= 0) & (y_int < height)
        coords = set(zip(x_int[valid], y_int[valid]))
        for xi, yi in coords:
            ellipse_layer[yi, xi] = ellipse_value
    fits.writeto(
        "apertures.fits", image_bkgsub + ellipse_layer, header=header, overwrite=True
    )

    if save_jpg:
        scale = 3
        plt.figure(
            figsize=(
                width // np.gcd(width, height) * scale,
                height // np.gcd(width, height) * scale,
            )
        )
        ax = plt.subplot(111)
        if norm is None:
            norm = simple_norm(
                image_bkgsub, stretch="asinh", min_percent=1, max_percent=99
            )
        ax.imshow(
            image_bkgsub,
            cmap="gray",
            norm=norm,
            origin="lower",
            interpolation="nearest",
        )
        cat.plot_kron_apertures(axes=ax, color="yellow", lw=0.6)
        plt.savefig("apertures.jpg")
        plt.close()


def se_background(
    image: np.ndarray,
    back_size: int = 64,
    back_filtersize: int = 3,
    bkg_estimator: BackgroundBase = SExtractorBackground(),
    mask: Optional[np.ndarray] = None,
    coverage_mask: Optional[np.ndarray] = None,
    weight_type: Literal[
        "NONE", "BACKGROUND", "MAP_RMS", "MAP_WEIGHT", "MAP_VAR"
    ] = "NONE",
    weight_image: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[Background2D, np.ndarray]:
    """
    Calculate the background image and background RMS using the SExtractor method.

    This function uses `photutils.background.Background2D` to estimate the
    background and its RMS. The treatment of the weight map (if any) follows
    the logic used by SExtractor for different `weight_type` options.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        Input image data.
    back_size : int, optional
        Size of the background mesh boxes (in pixels). Default is 64.
    back_filtersize : int, optional
        Size of the median filter applied to the background mesh. Default is 3.
    bkg_estimator : `~photutils.background.BackgroundBase`, optional
        The background estimator (e.g., `SExtractorBackground`, `MMMBackground`).
    mask : 2D `~numpy.ndarray` (bool) or None, optional
        Boolean mask of bad pixels (True = masked). Default is None.
    coverage_mask : 2D `~numpy.ndarray` (bool) or None, optional
        Boolean coverage mask (True = no coverage). Default is None.
    weight_type : {'NONE', 'BACKGROUND', 'MAP_RMS', 'MAP_WEIGHT', 'MAP_VAR'}, optional
        Type of weight map to use. Determines how the background RMS is computed.
        - 'NONE' : use the median of the low‑resolution background RMS as a global value.
        - 'BACKGROUND' : use the bicubic‑interpolated background RMS mesh.
        - 'MAP_RMS' : use the provided `weight_image` directly as the RMS map.
        - 'MAP_WEIGHT' : `weight_image` is a weight map (1/var). The RMS is derived
        by scaling it to match the background RMS.
        - 'MAP_VAR' : `weight_image` is a variance map. The RMS is derived similarly.
        Default is 'NONE'.
    weight_image : 2D `~numpy.ndarray` or None, optional
        Weight map (type depends on `weight_type`). Required for types other than
        'NONE' and 'BACKGROUND'. Default is None.
    verbose : bool, optional
        If True, print additional information (e.g., scaling factor). Default is False.

    Returns
    -------
    bkg : `~photutils.background.Background2D`
        The Background2D object containing the background mesh, full‑resolution
        background, and related metadata.
    bkg_rms : 2D `~numpy.ndarray`
        Background RMS map. Depending on `weight_type`, this may be a constant
        (2D array filled with the median RMS) or a pixel‑wise map derived from
        the weight image. It has the same shape as `image`.

    Notes
    -----
    - This function requires Python ≥3.10 because it uses `match` statement.
    - The scaling factor for `weight_type = 'MAP_WEIGHT'` or `'MAP_VAR'` is
      computed as the median ratio between the background RMS mesh and the
      square‑root of the background‑estimated variance mesh. This ensures that
      the final RMS map is correctly normalized.
    """

    bkg = Background2D(
        image,
        (back_size, back_size),
        mask=mask,
        coverage_mask=coverage_mask,
        filter_size=(back_filtersize, back_filtersize),
        bkg_estimator=bkg_estimator,
    )
    
    # The median value of the low-resolution background RMS
    # This is used for PSFex fitting
    global_rms = bkg.background_rms_median

    if sys.version_info >= (
        3,
        10,
    ):  # match syntax is only available in Python 3.10 or later
        match weight_type:

            case "NONE":  # Case 1: weight_type = 'NONE'
                bkg_rms = np.full_like(
                    image, bkg.background_rms_median
                )  # SExtractor uses the median value of the low-resolution background RMS as the global RMS

            case "BACKGROUND":  # Case 2: weight_type = 'BACKGROUND'
                bkg_rms = (
                    bkg.background_rms
                )  # bicubic-spline-interpolated background RMS mesh (minibkgrms)

            case "MAP_RMS":  # Case 3: weight_type = 'MAP_RMS'
                bkg_rms = weight_image

            case "MAP_WEIGHT":  # Case 4: weight_type = 'MAP_WEIGHT'
                # convert the weight image to the variance first to obtain the weight background
                wbkg = Background2D(
                    1 / weight_image,
                    (back_size, back_size),
                    mask=mask,
                    coverage_mask=coverage_mask,
                    filter_size=(back_filtersize, back_filtersize),
                    bkg_estimator=bkg_estimator,
                )
                ratiop = bkg.background_rms_mesh / np.sqrt(
                    wbkg.background_mesh
                )  # compute normalization

                # ratiop_sorted = np.sort(ratiop)
                # num_neg = (ratiop<0).sum()
                # sigfac = np.median(ratiop_sorted[num_neg:-num_neg])
                sigfac = np.nanmedian(ratiop)  # mean weight scaling factor
                if verbose:
                    print("mean weight scaling factor=", sigfac)

                bkg_rms = sigfac / np.sqrt(weight_image)

            case "MAP_VAR":  # Case 5: weight_type = 'MAP_VAR'
                wbkg = Background2D(
                    weight_image,
                    (back_size, back_size),
                    mask=mask,
                    coverage_mask=coverage_mask,
                    filter_size=(back_filtersize, back_filtersize),
                    bkg_estimator=bkg_estimator,
                )

                ratiop = bkg.background_rms_mesh / np.sqrt(
                    wbkg.background_mesh
                )  # compute normalization

                sigfac = np.nanmedian(ratiop)  # mean weight scaling factor
                if verbose:
                    print("median weight scaling factor=", sigfac)

                bkg_rms = sigfac * np.sqrt(weight_image)

            case _:
                raise NotImplementedError(
                    f"weight_type={weight_type} has not been implemented yet."
                )

    else:
        raise NotImplementedError(
            f"Python version {sys.version} is not supported yet. Please use Python 3.10 or later."
        )

    return bkg, bkg_rms, global_rms


def _dthresh(cat: SourceCatalog, threshold: np.ndarray) -> np.ndarray:
    """
    Calculate the detection threshold for each source in the catalog.

    For each source, the function extracts the cutout of the threshold image
    corresponding to the source's bounding box, masks out pixels belonging to
    other sources (using the source's total mask), and returns the minimum
    value of the remaining pixels. This minimum serves as the detection
    threshold for that source, analogous to SExtractor's behaviour.

    Parameters
    ----------
    cat : SourceCatalog
        Source catalog created from a segmentation image. The catalog must
        have the private attributes `_slices_iter` (list of slices for each
        source) and `_cutout_total_masks` (list of boolean masks where True
        indicates pixels belonging to other sources or background).
    threshold : 2D `~numpy.ndarray`
        Threshold image (e.g., detection threshold map) with the same shape
        as the original image from which the catalog was built.

    Returns
    -------
    dthresh : 1D `~numpy.ndarray`
        Array of detection thresholds, one per source. The value is the
        minimum threshold value within the source's unmasked region.

    Notes
    -----
    This function relies on private attributes of `SourceCatalog` and may
    break in future versions of Photutils. It is intended for internal use
    only. The commented alternative using `cat._get_values` would be faster
    but does not exactly match SExtractor's behaviour.
    """

    thresh_cutout_list = [threshold[slc] for slc in cat._slices_iter]
    return np.array(
        [
            np.min(thresh_cutout[~mask])
            for thresh_cutout, mask in zip(thresh_cutout_list, cat._cutout_total_masks)
        ]
    )

    # the following is not exactly following SExtractor, but it is faster
    # return np.array([np.min(array)
    #                 for array in cat._get_values(cat.convdata_ma)]) # SE uses the minimum value of the convolved data as the detection threshold


def _mthresh(
    cat: SourceCatalog, dthresh: Union[float, np.ndarray] = 0.0, detect_minarea: int = 5
) -> np.ndarray:
    """
    Calculate the mthresh parameter in the SExtractor clean.c script.
    mthresh is the maximum detection threshold for the detection to be considered as a source.
    As the detection threshold increases, the detected area decreases. When detected area is smaller than detect_minarea, the detection is invalid.
    Therefore, there exists a maximum detection threshold for a valid detection for every given detection.

    Parameters
    ----------
    cat : SourceCatalog
        Source catalog created from a segmentation image. The catalog must have
        the private attribute `_get_values` that returns per‑source pixel values
        from a masked array (e.g., `cat.convdata_ma`).
    dthresh : float or 1D `~numpy.ndarray`, optional
        Detection threshold value(s). If a scalar, the same value is used for all
        sources. If an array, it must have length equal to the number of sources.
        Default is 0.0.
    detect_minarea : int, optional
        The minimum number of pixels required for a valid detection.

    Returns
    -------
    mthresh : 1D `~numpy.ndarray`
        Maximum threshold possible for detection. (same as THRESHOLDMAX in sextractor, see more details in https://github.com/astromatic/sextractor/issues/87)

    Notes
    -----
    This function relies on the private method `cat._get_values()` and the
    masked array `cat.convdata_ma`, which may change in future Photutils
    versions. It is intended for internal use only.
    """

    if np.isscalar(dthresh):
        dthreshs = np.full(cat.nlabels, dthresh)
    else:
        dthreshs = dthresh
    cpixvals: list = cat._get_values(
        cat.convdata_ma
    )  # every element in the list is the 1D array of pixel values in the convolved data for the detection
    mthresh = np.array(
        [
            heapq.nlargest(detect_minarea, arr)[-1] - dthresh_iter
            for (arr, dthresh_iter) in zip(cpixvals, dthreshs)
        ]
    )
    # mthresh is the detect_minarea-th largest value in the segment minus detection threshold

    return mthresh


def _abcor(cat: SourceCatalog, dthreshs: np.ndarray) -> np.ndarray:
    """
    Calculate the abcor parameter in the SExtractor clean.c script.
    abcor is the correction factor for the detection area.
    It is used to correct the area of the detection based on the detection threshold.
    The correction factor is calculated based on the ratio of the area between two thresholds.

    Parameters
    ----------
    cat : SourceCatalog
        The source catalog.
    dthreshs : 1D `~numpy.ndarray`
        Detection threshold values for each source (same as `dthresh` from
        `_dthresh` or similar). Must have length equal to the number of sources.

    Returns
    -------
    abcor : 1D `~numpy.ndarray`
        The correction factor for the detection area, clipped to a maximum of 1.0.
        Values greater than 1.0 are set to 1.0.

    Notes
    -----
    This function relies on private Photutils methods and may break in future
    releases. It is intended for internal use only. The computation follows:

        1. thresh2 = 0.5 * (source_max_value + dthresh)
        2. t1t2 = dthresh / thresh2
        3. area2 = number of source pixels above thresh2
        4. dnpix = number of source pixels above dthresh
        5. darea = area2 - dnpix
        6. abcor = (darea if darea >=0 else -1.0) / (2π * ln( max(t1t2,0.99) ) * a * b)

    where a and b are the semi‑major and semi‑minor Gaussian sigmas.
    """

    thresh2 = 0.5 * (cat.max_value + dthreshs)
    t1t2 = dthreshs / thresh2

    pixvals: list = cat._get_values(
        cat.data_ma
    )  # every element in the list is the 1D array of pixel values in the raw data for the detection
    area2 = np.array(
        [np.sum(arr > thresh2_) for (arr, thresh2_) in zip(pixvals, thresh2)]
    )  # number of pixel above thresh2
    dnpix = np.array(
        [np.sum(arr > dthresh) for (arr, dthresh) in zip(pixvals, dthreshs)]
    )  # number of pixel above thresh

    darea = area2 - dnpix  # number of pixel between thresh and thresh2

    abcor = np.choose(darea < 0.0, [-1.0, darea]) / (
        2
        * np.pi
        * np.log(np.choose(t1t2 < 1.0, [0.99, t1t2]))
        * cat.semimajor_sigma.value
        * cat.semiminor_sigma.value
    )  # see scan.c line 1066-1077

    return np.choose(abcor > 1.0, [abcor, 1.0])


def cat2tab(
    cat: SourceCatalog,
    phot_apertures: Optional[Union[float, List[float]]] = None,
    mag_zeropoint: float = 38.951,
    kron_fact: float=2.5,
) -> Table:
    """
    Convert a Photutils SourceCatalog to an Astropy Table and add additional measurement parameters.

    The function first converts the catalog to a base table using `cat.to_table()`,
    then appends extra columns such as elongation, ellipticity, FWHM, Kron radius,
    flux radius, celestial coordinates, shape parameters (cxx, cxy, cyy), Gini coefficient,
    segment area, and an AUTO magnitude. Optionally, it performs circular aperture
    photometry for one or more radii and adds the resulting fluxes and flux errors.

    Parameters
    ----------
    cat : SourceCatalog
        The source catalog from which to extract measurements.
    phot_apertures : float or list of float, optional
        Radius (in pixels) of circular apertures for additional photometry.
        If a list, each element must be a float. If None, no aperture photometry
        is performed. Default is None.
    mag_zeropoint : float, optional
        Magnitude zeropoint used to convert Kron flux to magnitude.
        Default is 38.951 (typical for some HST filters).

    Returns
    -------
    tab : `~astropy.table.Table`
        Table containing all source measurements, including the extra columns
        described above and, if requested, aperture photometry columns.

    Notes
    -----
    - The Kron radius is multiplied by 2.5 to match SExtractor's convention.
    - The `flux_radius` is the half‑light radius computed from the radial profile.
    - The `background_centroid` column is added as a direct property from the catalog.
    - If `phot_apertures` is provided, columns named `aperN_flux` and `aperN_fluxerr`
      are added for each aperture (N = 1, 2, ...).
    """

    print("to_table start")
    # convert to Table, but only contain 20 parameters by defalut
    tab = cat.to_table()

    # Remove the sky_centroid column if it exists
    if "sky_centroid" in tab.colnames:
        tab.remove_column("sky_centroid")
    print("to_table done")

    # Below are the desired parameters to be added to Table.
    background_centroid = cat.background_centroid

    print("background_centroid=", background_centroid)

    elongation = cat.elongation
    ellipticity = cat.ellipticity
    fwhm = cat.fwhm
    kron_radius = cat.kron_radius * kron_fact  # to be consistent with SExtractor
    flux_radius = cat.fluxfrac_radius(0.5)  # check into photutils for similar outputs.
    ra = cat.sky_centroid.ra.deg
    dec = cat.sky_centroid.dec.deg
    cxx = cat.cxx
    cxy = cat.cxy
    cyy = cat.cyy
    gini = cat.gini
    segment_area = cat.segment_area
    mag_auto = np.around(-2.5 * np.log10(cat.kron_flux) + mag_zeropoint, 4)

    cstar = cat.class_star
    obj_crowded=cat.obj_crowded
    obj_merged=cat.obj_merged
    combined_flags = cat.combined_flags
    global_rms = cat.global_rms
    
    print("add_columns start")
    tab.add_columns(
        [
            background_centroid,
            elongation,
            ellipticity,
            fwhm,
            kron_radius,
            flux_radius,
            ra,
            dec,
            cxx,
            cxy,
            cyy,
            gini,
            segment_area,
            mag_auto,
            cstar,
            # obj_crowded,
            # obj_merged,
            combined_flags,
            global_rms
        ],
        names=[
            "background_centroid",
            "elongation",
            "ellipticity",
            "fwhm",
            "kron_radius",
            "flux_radius",
            "ra",
            "dec",
            "cxx",
            "cxy",
            "cyy",
            "gini",
            "segment_area",
            "mag_auto",
            "class_star",
            # "obj_crowded",
            # "obj_merged",
            'combined_flags',
            'global_rms'
        ],
        copy=False,
    )

    print("phot_apertures", phot_apertures)

    # peform circular aperture photometry
    if phot_apertures is not None:
        if isinstance(phot_apertures, (int, float)):
            phot_apertures = [phot_apertures]
        elif isinstance(phot_apertures, list):
            # Validate all elements in list are numbers
            for ap in phot_apertures:
                if not isinstance(ap, (int, float)):
                    raise ValueError("All elements in phot_apertures must be numbers.")
        else:
            raise TypeError("phot_apertures must be a number or a list of numbers.")

        # peform photometry for each circular aperture, and then add to tab
        for i in range(len(phot_apertures)):
            col_name = f"aper{i+1}"
            cat.circular_photometry(phot_apertures[i], name=col_name)
            flux = getattr(cat, col_name + "_flux")
            fluxerr = getattr(cat, col_name + "_fluxerr")
            tab.add_column(flux, name=col_name + "_flux")
            tab.add_column(fluxerr, name=col_name + "_fluxerr")

    return tab


def se_clean(
    cat_spur: SourceCatalog,
    threshold: np.ndarray,
    clean_param=1.0,
    detect_minarea: int = 8,
    verbose: bool = False,
) -> SegmentationImage:
    """
    Clean the segmentation image using Moffat profile according to the SExtractor clean.c script.

    This function implements the cleaning algorithm from SExtractor's `clean.c`.
    It iteratively evaluates whether brighter sources (in terms of integrated flux)
    "eat" fainter neighbouring sources based on their Moffat profile amplitudes
    and a distance criterion. Sources that are eaten are removed from the
    segmentation image.

    Parameters
    ----------
    cat_spur : SourceCatalog
        Source catalog containing the initial (spurious) detections. Must have
        the following private attributes accessible: `_segment_img`, `convdata_ma`,
        `data_ma`, and standard properties like `xcentroid`, `semimajor_sigma`,
        `cxx`, etc.
    threshold : 2D `~numpy.ndarray`
        Detection threshold image (same shape as the original data). Used to
        compute per‑source detection thresholds (`dthresh`) and other cleaning
        quantities.
    clean_param : float, optional
        Cleaning parameter (β in the Moffat profile). Corresponds to SExtractor's
        `CLEAN` parameter. Default is 1.0.
    detect_minarea : int, optional
        Minimum number of pixels required for a valid detection. Used in the
        computation of `mthresh`. Default is 8.
    verbose : bool, optional
        If True, show a progress bar during cleaning. Default is False.

    Returns
    -------
    segm_cleaned : SegmentationImage
        A new segmentation image with the spurious sources removed (their pixels
        set to 0). Sources that survive cleaning retain their original labels.

    Notes
    -----
    - The cleaning zone radius is hard‑coded to 10.0 times the sum of the
      semi‑major axes of the two sources (`CLEAN_ZONE = 10.0`).
    - The function relies on several private Photutils methods and attributes
      (e.g., `_segment_img._data`, `_get_values`, `convdata_ma`). It may break
      with future Photutils versions.
    - The cleaning logic follows SExtractor's `clean.c` lines 1066‑1077 and the
      surrounding context. The `mthresh` and `abcor` quantities are computed via
      the helper functions `_dthresh` and `_abcor`.
    """

    CLEAN_ZONE = 10.0  # not sure if this should be a parameter
    beta = clean_param
    # segm_spur = cat_spur._segment_img.copy()
    segm_spur_data = cat_spur._segment_img._data.copy()
    dthreshs = _dthresh(cat=cat_spur, threshold=threshold)

    # treatment on cdpix (convolved detected pixels)
    cpixvals: list = cat_spur._get_values(
        cat_spur.convdata_ma
    )  # every element in the list is the 1D array of pixel values in the convolved data for the detection
    mthresh = np.array(
        [
            heapq.nlargest(detect_minarea, arr)[-1] - dthresh
            for (arr, dthresh) in zip(cpixvals, dthreshs)
        ]
    )  # mthresh is the detect_minarea-th largest value in the segment minus detection threshold
    fdflux = np.array([np.sum(arr) for arr in cpixvals])  # integrated ext. flux
    abcor = _abcor(
        cat_spur, dthreshs=dthreshs
    )  # to derive abcor, see scan.c line 1066-1077

    # disattach units from these lazy properties for faster computation:
    smaj, smin = cat_spur.semimajor_sigma.value, cat_spur.semiminor_sigma.value
    cxx, cxy, cyy = cat_spur.cxx.value, cat_spur.cxy.value, cat_spur.cyy.value
    segm_area = cat_spur.segment_area.value

    survives = np.ones_like(
        cat_spur.labels, dtype=bool
    )  # initialize all sources as survived:[True,True......]
    iis = (
        tqdm(range(0, cat_spur.nlabels), desc="Cleaning")
        if verbose
        else range(0, cat_spur.nlabels)
    )  # tqdm is used to show the progress bar
    for ii in iis:  # iterate over every detection
        if not survives[ii]:
            continue

        unitareain = np.pi * smaj[ii] * smin[ii]
        ampin = fdflux[ii] / (2 * unitareain * abcor[ii])
        alphain = (
            (np.power(ampin / dthreshs[ii], 1.0 / beta) - 1.0)
            * unitareain
            / segm_area[ii]
        )

        # loop over remaining sources
        for jj in range(ii + 1, cat_spur.nlabels):
            if not survives[jj]:
                continue
            # calculate the distance between the two sources
            dx = (
                cat_spur.xcentroid[ii] - cat_spur.xcentroid[jj]
            )  # xcentroid is computed as the center of mass of the unmasked pixels within the source segment
            dy = (
                cat_spur.ycentroid[ii] - cat_spur.ycentroid[jj]
            )  # ycentroid is computed as the center of mass of the unmasked pixels within the source segment
            rlim = smaj[ii] + smaj[jj]
            # if the distance is larger than the limit, skip
            if dx**2 + dy**2 > (rlim * CLEAN_ZONE) ** 2:
                continue

            # if segment ii is brighter than segment jj, see if it eats segment jj
            if fdflux[ii] > fdflux[jj]:
                val = 1 + alphain * (
                    cxx[ii] * dx**2 + cxy[ii] * dx * dy + cyy[ii] * dy**2
                )
                if (val > 1.0) and (
                    (ampin * np.power(val, -beta) if val < 1e10 else 0.0) > mthresh[jj]
                ):  # detect_minarea-th largest - moffat(r) < dthresh
                    survives[jj] = False  # segment jj is eaten by segment ii
                    # print(f'{ii} eats {jj}')
                    # segm_spur.reassign_label(cat_spur.labels[jj], 0)
                    segm_spur_data[segm_spur_data == cat_spur.labels[jj]] = 0
                    # segm_spur.reassign_label(cat_spur.labels[jj], cat_spur.labels[ii]) # reassign the label of segment jj to segment ii

            # if segment jj is brighter than segment ii, see if it eats segment ii
            else:
                unitarea = np.pi * smaj[jj] * smin[jj]
                amp = fdflux[jj] / (2 * unitarea * abcor[jj])

                alpha = (
                    (np.power(amp / dthreshs[jj], 1.0 / beta) - 1.0)
                    * unitarea
                    / segm_area[jj]
                )
                val = 1 + alpha * (
                    cxx[jj] * dx**2 + cxy[jj] * dx * dy + cyy[jj] * dy**2
                )
                if (val > 1.0) and (
                    (amp * np.power(val, -beta) if val < 1e10 else 0.0) > mthresh[ii]
                ):
                    survives[ii] = False
                    # print(f'{jj} eats {ii}')
                    # segm_spur.reassign_label(cat_spur.labels[ii], 0)
                    segm_spur_data[segm_spur_data == cat_spur.labels[ii]] = 0
                    # segm_spur.reassign_label(cat_spur.labels[ii], cat_spur.labels[jj]) # reassign the label of segment ii to segment jj
                    break  # if segment ii has been eaten, then exit the inner loop
            # inner loop ends
        # print(f"{ii}'s loop ends")
        # outer loop ends

    # return segm_spur
    return SegmentationImage(segm_spur_data)


def se_clean2(
    cat_spur: SourceCatalog, threshold: np.ndarray, clean_param=1.0, detect_minarea: int = 8, verbose: bool = False
) -> SegmentationImage:
    """
    Clean the segmentation image using Moffat profile according to the SExtractor clean.c script.

    Use the cKDTree to acclerate the computation.

    This function implements the cleaning algorithm from SExtractor's `clean.c`.
    It iteratively evaluates whether brighter sources (in terms of integrated flux)
    "eat" fainter neighbouring sources based on their Moffat profile amplitudes
    and a distance criterion. Sources that are eaten are removed from the
    segmentation image.

    Parameters
    ----------
    cat_spur : SourceCatalog
        Source catalog containing the initial (spurious) detections. Must have
        the following private attributes accessible: `_segment_img`, `convdata_ma`,
        `data_ma`, and standard properties like `xcentroid`, `semimajor_sigma`,
        `cxx`, etc.
    threshold : 2D `~numpy.ndarray`
        Detection threshold image (same shape as the original data). Used to
        compute per‑source detection thresholds (`dthresh`) and other cleaning
        quantities.
    clean_param : float, optional
        Cleaning parameter (β in the Moffat profile). Corresponds to SExtractor's
        `CLEAN` parameter. Default is 1.0.
    detect_minarea : int, optional
        Minimum number of pixels required for a valid detection. Used in the
        computation of `mthresh`. Default is 8.
    verbose : bool, optional
        If True, show a progress bar during cleaning. Default is False.

    Returns
    -------
    segm_cleaned : SegmentationImage
        A new segmentation image with the spurious sources removed (their pixels
        set to 0). Sources that survive cleaning retain their original labels.

    Notes
    -----
    - The cleaning zone radius is hard‑coded to 10.0 times the sum of the
      semi‑major axes of the two sources (`CLEAN_ZONE = 10.0`).
    - The function relies on several private Photutils methods and attributes
      (e.g., `_segment_img._data`, `_get_values`, `convdata_ma`). It may break
      with future Photutils versions.
    - The cleaning logic follows SExtractor's `clean.c` lines 1066‑1077 and the
      surrounding context. The `mthresh` and `abcor` quantities are computed via
      the helper functions `_dthresh` and `_abcor`.
    """

    CLEAN_ZONE = 10.0  # not sure if this should be a parameter
    beta = clean_param
    # segm_spur = cat_spur._segment_img.copy()
    segm_spur_data = cat_spur._segment_img._data.copy()
    dthreshs = _dthresh(cat=cat_spur, threshold=threshold)

    # treatment on cdpix (convolved detected pixels)
    cpixvals: list = cat_spur._get_values(
        cat_spur.convdata_ma
    )  # every element in the list is the 1D array of pixel values in the convolved data for the detection
    mthresh = np.array(
        [heapq.nlargest(detect_minarea, arr)[-1] - dthresh for (arr, dthresh) in zip(cpixvals, dthreshs)]
    )  # mthresh is the detect_minarea-th largest value in the segment minus detection threshold
    fdflux = np.array([np.sum(arr) for arr in cpixvals])  # integrated ext. flux
    abcor = _abcor(cat_spur, dthreshs=dthreshs)  # to derive abcor, see scan.c line 1066-1077

    # disattach units from these lazy properties for faster computation:
    smaj, smin = cat_spur.semimajor_sigma.value, cat_spur.semiminor_sigma.value
    cxx, cxy, cyy = cat_spur.cxx.value, cat_spur.cxy.value, cat_spur.cyy.value
    segm_area = cat_spur.segment_area.value

    survives = np.ones_like(cat_spur.labels, dtype=bool)  # initialize all sources as survived:[True,True......]

    xc = cat_spur.xcentroid
    yc = cat_spur.ycentroid
    coords = np.column_stack([xc, yc])
    # max_radius = CLEAN_ZONE * 2 * np.max(smaj)   
    sorted_smaj = np.sort(smaj)
    max_radius = CLEAN_ZONE * (sorted_smaj[-1] + sorted_smaj[-2]) if len(smaj) >= 2 else 0.0

    tree = cKDTree(coords)
    candidate_pairs = tree.query_pairs(r=max_radius, output_type='ndarray')

    pair_iter = tqdm(candidate_pairs, desc="Cleaning pairs") if verbose else candidate_pairs
    # pair_iter = tqdm(valid_pairs, desc="Cleaning pairs") if verbose else valid_pairs
    for ii, jj in pair_iter:
    # for ii, jj in candidate_pairs:
        if not survives[ii]:
            continue
        if not survives[jj]:
            continue

        unitareain = np.pi * smaj[ii] * smin[ii]
        ampin = fdflux[ii] / (2 * unitareain * abcor[ii])
        alphain = (np.power(ampin / dthreshs[ii], 1.0 / beta) - 1.0) * unitareain / segm_area[ii]

        # calculate the distance between the two sources
        dx = (
            cat_spur.xcentroid[ii] - cat_spur.xcentroid[jj]
        )  # xcentroid is computed as the center of mass of the unmasked pixels within the source segment
        dy = (
            cat_spur.ycentroid[ii] - cat_spur.ycentroid[jj]
        )  # ycentroid is computed as the center of mass of the unmasked pixels within the source segment
        rlim = smaj[ii] + smaj[jj]
        # if the distance is larger than the limit, skip
        if dx**2 + dy**2 > (rlim * CLEAN_ZONE) ** 2:
            continue

        # if segment ii is brighter than segment jj, see if it eats segment jj
        if fdflux[ii] > fdflux[jj]:
            val = 1 + alphain * (cxx[ii] * dx**2 + cxy[ii] * dx * dy + cyy[ii] * dy**2)
            if (val > 1.0) and (
                (ampin * np.power(val, -beta) if val < 1e10 else 0.0) > mthresh[jj]
            ):  # detect_minarea-th largest - moffat(r) < dthresh
                survives[jj] = False  # segment jj is eaten by segment ii
                # segm_spur_data[segm_spur_data == cat_spur.labels[jj]] = 0

        # if segment jj is brighter than segment ii, see if it eats segment ii
        else:
            unitarea = np.pi * smaj[jj] * smin[jj]
            amp = fdflux[jj] / (2 * unitarea * abcor[jj])

            alpha = (np.power(amp / dthreshs[jj], 1.0 / beta) - 1.0) * unitarea / segm_area[jj]
            val = 1 + alpha * (cxx[jj] * dx**2 + cxy[jj] * dx * dy + cyy[jj] * dy**2)
            if (val > 1.0) and ((amp * np.power(val, -beta) if val < 1e10 else 0.0) > mthresh[ii]):
                survives[ii] = False
                # segm_spur_data[segm_spur_data == cat_spur.labels[ii]] = 0
                # break  # if segment ii has been eaten, then exit the inner loop

    labels_to_remove = cat_spur.labels[~survives]
    if len(labels_to_remove) > 0:
        mask = np.isin(segm_spur_data, labels_to_remove)
        segm_spur_data[mask] = 0

    # return segm_spur
    return SegmentationImage(segm_spur_data)


def _preanalyze(
    data: np.ndarray,
    segm: SegmentationImage,
    convolved_data: np.ndarray,
    threshold: np.ndarray,
) -> Tuple[
    np.ndarray,  # xcentroid
    np.ndarray,  # ycentroid
    np.ndarray,  # cxx
    np.ndarray,  # cxy
    np.ndarray,  # cyy
    np.ndarray,  # semimajor_sigma (values)
    np.ndarray,  # semiminor_sigma (values)
    np.ndarray,  # abcor
    np.ndarray,  # segment_area
    np.ndarray,  # max_value
]:
    """
    Pre‑analyze sources to extract parameters needed for deblending.

    This internal function creates a temporary `SourceCatalog` from the provided
    data and segmentation image, then computes per‑source quantities required by
    the deblending algorithm. The returned tuple contains arrays of these
    quantities, one element per source.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        Original (background‑subtracted) image data.
    segm : `~photutils.segmentation.SegmentationImage`
        Segmentation image defining the sources.
    convolved_data : 2D `~numpy.ndarray`
        Convolved version of the image (used for detection).
    threshold : 2D `~numpy.ndarray`
        Threshold map (same shape as `data`) used for detection.

    Returns
    -------
    xcentroid : `~numpy.ndarray`
        X‑coordinates of source centroids.
    ycentroid : `~numpy.ndarray`
        Y‑coordinates of source centroids.
    cxx : `~numpy.ndarray`
        Ellipse Cxx parameter for each source.
    cxy : `~numpy.ndarray`
        Ellipse Cxy parameter for each source.
    cyy : `~numpy.ndarray`
        Ellipse Cyy parameter for each source.
    semimajor_sigma : `~numpy.ndarray`
        Semi‑major axis Gaussian sigma (values, without units).
    semiminor_sigma : `~numpy.ndarray`
        Semi‑minor axis Gaussian sigma (values, without units).
    abcor : `~numpy.ndarray`
        Area correction factor computed by `_abcor`.
    segment_area : `~numpy.ndarray`
        Area (in pixels) of each source's segment.
    max_value : `~numpy.ndarray`
        Maximum pixel value within each source.
    """

    cat = SourceCatalog(data, segm, convolved_data=convolved_data)
    dthreshs = _dthresh(cat, threshold)
    return (
        cat.xcentroid,
        cat.ycentroid,
        cat.cxx,
        cat.cxy,
        cat.cyy,
        cat.semimajor_sigma.value,
        cat.semiminor_sigma.value,
        _abcor(cat, dthreshs=dthreshs),
        cat.segment_area.value,
        cat.max_value,
    )


def se_deblend(
    data: np.ndarray,
    segment_img: SegmentationImage,
    convolved_data: np.ndarray,
    threshold: np.ndarray,
    detect_minarea: int = 5,
    deblend_nthresh: int = 32,
    deblend_mincont: float = 0.001,
    detect_connectivity: Literal[4, 8] = 8,
    mode: Literal["exponential", "linear"] = "exponential",
    nproc: int = 1,
    verbose: bool = False,
) -> SegmentationImage:
    """
    Deblend the sources using Multi-Thresholding + Gather Up method.
    The method is based on the SExtractor deblending algorithm.

    This function implements the SExtractor deblending algorithm, which recursively
    applies decreasing thresholds to split blended sources. It processes each
    source independently, optionally in parallel.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        Original (background‑subtracted) image data.
    segment_img : `~photutils.segmentation.SegmentationImage`
        Input segmentation image (before deblending).
    convolved_data : 2D `~numpy.ndarray`
        Convolved version of the image (used internally for thresholding).
    threshold : 2D `~numpy.ndarray`
        Detection threshold map (same shape as `data`).
    detect_minarea : int, optional
        Minimum number of pixels for a detection. Sources smaller than
        `2 * detect_minarea` are not deblended. Default is 5.
    deblend_nthresh : int, optional
        Number of thresholds to use in the multi‑thresholding process.
        Default is 32.
    deblend_mincont : float, optional
        Minimum contrast ratio for deblending. A child must have a peak
        at least this fraction of the parent's peak to be considered a
        separate source. Default is 0.001.
    detect_connectivity : {4, 8}, optional
        Pixel connectivity used for grouping pixels. 4‑connected pixels
        touch edges; 8‑connected also touch corners. Default is 8.
    mode : {'exponential', 'linear'}, optional
        Scaling of the thresholds:
        - 'exponential' : thresholds are spaced exponentially.
        - 'linear' : thresholds are spaced linearly.
        Default is 'exponential'.
    nproc : int, optional
        Number of parallel processes to use. If 1, run serially; if >1,
        use a multiprocessing pool. Must not exceed available CPUs.
        Default is 1.
    verbose : bool, optional
        If True, show a progress bar during serial processing. Default is False.

    Returns
    -------
    segm_final : `~photutils.segmentation.SegmentationImage`
        Deblended segmentation image, with consecutive labels starting from 1.
    """

    footprint = _make_binary_structure(data.ndim, detect_connectivity)
    deblend_params = _DeblendParams(
        detect_minarea, footprint, deblend_nthresh, deblend_mincont, mode=mode
    )
    dthreshs = _dthresh(
        cat=SourceCatalog(data, segment_img, convolved_data=convolved_data),
        threshold=threshold,
    )

    max_label = segment_img.max_label
    marker_final = segment_img.data.copy()
    if nproc == 1:  # serial processing
        count = 0

        for idx in tqdm(
            range(0, segment_img.nlabels), desc="Deblending", disable=not verbose
        ):  # tqdm here to show the progress bar
            # if the area is too small to be deblended, skip
            label_iter = segment_img.labels[idx]
            slc_iter = segment_img.slices[idx]

            if segment_img.areas[idx] < detect_minarea * 2:
                ##############
                mask = segment_img.data[slc_iter] == label_iter
                marker_final[slc_iter][mask] = 1 + count
                count = count + 1
                continue

            marker_slc, _ = _se_deblend_worker(
                label_iter,
                slc_iter,
                dthreshs[idx],
                data,
                segment_img,
                convolved_data,
                threshold,
                deblend_params,
            )

            # FIXME: there should be a more elegant way to do the following
            source_mask = marker_slc > 0
            # marker_final[slc_iter][source_mask] = marker_slc[source_mask] + max_label
            # max_label = max_label + marker_slc.max()
            if (
                marker_slc[source_mask].min() == marker_slc[source_mask].max()
            ):  # if no deblend, set the origin idx to 1.
                marker_slc[source_mask] = 1
            marker_final[slc_iter][source_mask] = marker_slc[source_mask] + count
            count = count + marker_slc.max()
        segm_final = SegmentationImage(marker_final)
        segm_final.relabel_consecutive()

    elif nproc > 1:  # parallel processing
        if nproc > cpu_count():
            raise ValueError("Plz check if nproc is valid")
        # assemble the arguements together for parallel input
        args_all = zip(segment_img.labels, segment_img.slices, dthreshs, strict=True)
        # Create a partial function to pass the deblend_params to the worker function
        worker = partial(
            _se_deblend_worker,
            data=data,
            segment_img=segment_img,
            convolved_data=convolved_data,
            threshold=threshold,
            deblend_params=deblend_params,
        )

        # Prepare to store futures and results to preserve the input order of the labels when using as_completed()
        results = [None] * segment_img.nlabels
        with Pool(nproc) as p:
            results = p.starmap(worker, args_all)

        for result in results:
            marker_slc, source_slc = result
            source_mask = marker_slc > 0
            marker_final[source_slc][source_mask] = marker_slc[source_mask] + max_label
            max_label = max_label + marker_slc.max()

        segm_final = SegmentationImage(marker_final)
        segm_final.relabel_consecutive()

    return segm_final


def _se_deblend_worker(
    label_iter: int,
    slc_iter: Tuple[slice, slice],
    dthresh_iter: float,
    data: np.ndarray,
    segment_img: SegmentationImage,
    convolved_data: np.ndarray,
    threshold: np.ndarray,
    deblend_params: _DeblendParams,
) -> Tuple[np.ndarray, Tuple[slice, slice]]:
    """
    Worker function for deblending a single source.

    This function is called by the main deblending routine and is designed
    to be used in parallel processing. It extracts the cutout for one source,
    sets up a `_SingleSourceDeblender` with patched attributes needed for the
    SExtractor‑style deblending, and then calls the core deblending function
    `_se_deblend`. The resulting deblended marker array (within the cutout)
    and the slice are returned.

    Parameters
    ----------
    label_iter : int
        Label of the source to be deblended.
    slc_iter : tuple of slice
        Slice tuple defining the bounding box of the source in the full image.
    dthresh_iter : float
        Detection threshold value for this source (computed by `_dthresh`).
    data : 2D `~numpy.ndarray`
        Original (background‑subtracted) image data.
    segment_img : SegmentationImage
        Segmentation image containing all sources.
    convolved_data : 2D `~numpy.ndarray`
        Convolved version of the image.
    threshold : 2D `~numpy.ndarray`
        Threshold map used for detection.
    deblend_params : _DeblendParams
        Parameters controlling the deblending process (minimum area, number of
        thresholds, contrast, connectivity, mode).

    Returns
    -------
    marker_slc : 2D `~numpy.ndarray`
        Deblended marker array for the cutout region. Pixels belonging to
        different deblended children are assigned distinct positive integers
        (starting from 1). Background pixels are 0.
    slc_iter : tuple of slice
        The same slice tuple passed in, needed to place the result back into
        the full image.

    Notes
    -----
    This function patches three additional attributes onto the `deblender`
    instance (`data_raw`, `threshold`, and `source_min`) to adapt Photutils'
    `_SingleSourceDeblender` to the SExtractor deblending logic implemented
    in `_se_deblend`.
    """

    img_conv_slc = convolved_data[
        slc_iter
    ]  # no need to copy as we are not changing the data
    img_slc = data[slc_iter]
    threshold_slc = threshold[slc_iter]
    marker_iter = segment_img.data[
        slc_iter
    ].copy()  # create a new instance to avoid changing the original one

    # only deblend the current label
    mask = marker_iter == label_iter  # find mask for the current label
    marker_iter[~mask] = 0  # if there's other label in the cutout, set it to 0
    segm_slc = SegmentationImage(marker_iter)

    deblender = _SingleSourceDeblender(
        img_conv_slc, segm_slc, label_iter, deblend_params
    )
    deblender.data_raw = img_slc  # patch a new attribute to the deblender instance for the gatherup method
    deblender.threshold = threshold_slc  # patch a new attribute to the deblender instance for the gatherup method
    deblender.source_min = dthresh_iter  # a small difference btw SE and photutils in determining thresholds

    return _se_deblend(deblender, return_ini=False), slc_iter


def _se_deblend(
    deblender: _SingleSourceDeblender, return_ini: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Deblend a single source following the SExtractor method.

    This is a low‑level internal function that implements the multi‑threshold
    deblending algorithm used by SExtractor. It operates on a cutout containing
    one source (or a blended group) and returns a marker array where each
    deblended child receives a unique positive integer label.

    The algorithm proceeds in two main stages:
        1. Multi‑threshold segmentation: a series of thresholds is applied,
           producing a stack of segmentation images.
        2. Branch cutting: children that do not meet a flux contrast criterion
           are merged back.
        3. Gather‑up step: pixels that remain unassigned after the multi‑threshold
           process are assigned to the nearest child based on a distance‑weighted
           probability.

    Parameters
    ----------
    deblender : _SingleSourceDeblender
        A Photutils deblender object, patched with additional attributes
        required for the SExtractor‑style deblending:
            - `data_raw` : original (unconvolved) image cutout
            - `threshold` : threshold map cutout
            - `source_min` : detection threshold value for this source
        The deblender must also contain the usual attributes (`data`, `segment_data`,
        `npixels`, `footprint`, `segment_mask`, `contrast`, `source_sum`).
    return_ini : bool, optional
        If True, return a tuple `(marker_lower, marker_ini)` where `marker_ini`
        is the marker array before the gather‑up step (for debugging). Default is False.

    Returns
    -------
    marker_lower : 2D `~numpy.ndarray`
        Final deblended marker array for the cutout. Positive integers label
        distinct deblended children; 0 indicates background.
    marker_ini : 2D `~numpy.ndarray`, optional
        Returned only if `return_ini=True`. The marker array after multi‑threshold
        and branch cutting, before the gather‑up step.
    """

    # check if the deblender object has the raw data, otherwise, use the current data
    if not hasattr(deblender, "data_raw"):
        warnings.warn(
            "The deblender object does not have the raw data. The deblending will be performed on the current data."
        )
        deblender.data_raw = deblender.data

    # define the initial markers (possible segements)
    thresholds = deblender.compute_thresholds()[
        :
    ]  # dthresh is not included in the list
    segms = [
        deblender.segment_data
    ]  # list of segmentations with multiple thresholds, starts with the initial segmentation (dthresh)
    for threshold in thresholds:
        _segm = _detect_sources(
            deblender.data,
            threshold,
            deblender.npixels,
            deblender.footprint,
            deblender.segment_mask,
            relabel=False,
            return_segmimg=True,
        )
        if _segm is None:  # iterate until no more segments are detected
            break
        segms.append(_segm)
    marker_lower = segms[0].data.copy()  # the marker starts from lowest threshold

    # cut the branches according to the flux criterion
    for ii in range(len(segms) - 1):
        marker_upper = segms[ii + 1].data  # the marker at the upper level
        threshold = thresholds[ii]  # the threshold at the upper level

        new_markers = False  # flag to update labels
        labels_lower = _get_labels(marker_lower)
        for label in labels_lower:  # iterate every label in the lower level
            mask = marker_lower == label  # find mask for the current label
            upper_labels = _get_labels(
                marker_upper[mask]
            )  # find label mapping from the lower to upper level

            fluxlist = sum_labels(deblender.data, marker_upper, upper_labels)
            arealist = sum_labels(
                np.ones_like(deblender.data), marker_upper, upper_labels
            )
            survives = (
                fluxlist - threshold * arealist
                > deblender.contrast * deblender.source_sum
            )  # surviving criterion, which is different from photutils
            if survives.sum() > 1:  # new child markers found
                new_markers = True  # update labels
                # to remove child markers that cannot survive
                segm_survived = SegmentationImage(marker_upper)
                segm_survived.reassign_labels(upper_labels[~survives], 0)
                # now we have the updated markers
                marker_survived = segm_survived.data
                marker_lower[mask] = marker_survived[mask].astype(
                    bool
                )  # update marker_lower

        # update labels
        if new_markers:
            marker_lower = ndi_label(marker_lower, structure=deblender.footprint)[0]

    if return_ini:  # return the initial marker if required
        marker_ini = marker_lower.copy()

    segm_dbd = SegmentationImage(marker_lower)  # the initial markers to analyze
    nmarkers = (
        segm_dbd.nlabels
    )  # number of final markers which is determined through previous process

    # Once we have the final markers, apply gatherup
    # prepare for gatherup
    pix_tba = np.bitwise_and(
        segms[0].data.astype(bool), ~(marker_lower.astype(bool))
    )  # PIXels To Be Assigned labels (upper cases)
    dthresh = deblender.source_min  # should be dthresh for SE method
    xc, yc, cxx, cxy, cyy, smaj, smin, abcor, fdnpix, fdpeak = _preanalyze(
        deblender.data_raw, segm_dbd, deblender.data, deblender.threshold
    )
    dist_ = fdnpix / (2 * np.pi * abcor * smaj * smin)

    # To be honest, this part is hard to understand: why 70 here?
    amp = np.choose(dist_ < 70, [4 * fdpeak, dthresh * np.exp(dist_)])
    amp = np.choose(amp < 4 * fdpeak, [4 * fdpeak, amp])

    dist = np.zeros((nmarkers, marker_lower.shape[0], marker_lower.shape[1]))
    cdf = np.zeros((nmarkers + 1, marker_lower.shape[0], marker_lower.shape[1]))
    X, Y = np.meshgrid(
        np.arange(marker_lower.shape[1]), np.arange(marker_lower.shape[0])
    )
    for ii in range(nmarkers):
        dx, dy = X - xc[ii], Y - yc[ii]
        dist[ii] = (cxx[ii] * dx**2 + cyy[ii] * dy**2 + cxy[ii] * dx * dy) / (
            2 * abcor[ii]
        )  # r^2/2
        cdf[ii + 1] = cdf[ii] + np.choose(
            dist[ii] < 70, [0, amp[ii] * np.exp(-dist[ii])]
        )  # why 70 again?

    iclst = np.argmin(dist, axis=0)  # 0-indexed
    np.random.seed(0)
    rand_arr = np.random.rand(marker_lower.shape[0], marker_lower.shape[1])
    drand = cdf[-1, :, :] * rand_arr  # random numbers for each pixel
    for px, py in zip(*np.where(pix_tba)):
        if cdf[-1, px, py] == 0:
            idx_tba = iclst[px, py]
        else:
            ia = np.searchsorted(cdf[:, px, py], drand[px, py])  # sample the cdf
            idx_tba = ia - 1 if ia < nmarkers else iclst[px, py]
        label_tba = segm_dbd.labels[idx_tba]
        marker_lower[px, py] = label_tba

    if return_ini:
        return marker_lower, marker_ini
    else:
        return marker_lower




from typing import Tuple, Optional, Literal
import numpy as np
from astropy.table import Table, vstack
from photutils.segmentation import SegmentationImage
from scipy.spatial import cKDTree

# Columns needed for elliptical-distance computation
_COLD_COMP_COLS = ["label", "xcentroid", "ycentroid", "cxx", "cxy", "cyy", "kron_radius"]
_HOT_COMP_COLS = ["label", "xcentroid", "ycentroid"]

def _extract_comp(table, cols):
    """Return a view of *table* containing only *cols*, normalised to lower-case."""
    col_map = {c.strip().lower(): c for c in table.colnames}
    return Table({col: table[col_map[col]] for col in cols})



def sexcomb(
    coldtab: Table,
    hottab: Table, 
    coldsegm: SegmentationImage,
    hotsegm: SegmentationImage,
    scale_factor: float = 1.1,
    chunk_size: int = 1000,
    verbose: bool = False,
) -> Tuple[Table, SegmentationImage]:
    """
    sexcomb_memory_efficient
    
    Memory-efficient sexcomb for very large catalogs.

    Preserves ALL columns from the input catalogs in the output table.
    """
    cold_comp = _extract_comp(coldtab, _COLD_COMP_COLS)
    hot_comp = _extract_comp(hottab, _HOT_COMP_COLS)

    if verbose:
        print("Computing cold source parameters...")

    xcold = np.asarray(cold_comp["xcentroid"] + 1, dtype=np.float64)
    ycold = np.asarray(cold_comp["ycentroid"] + 1, dtype=np.float64)
    cxx_image = np.asarray(cold_comp["cxx"], dtype=np.float64)
    cxy_image = np.asarray(cold_comp["cxy"], dtype=np.float64)
    cyy_image = np.asarray(cold_comp["cyy"], dtype=np.float64)
    kron_radius = np.asarray(cold_comp["kron_radius"], dtype=np.float64)

    zero_mask = kron_radius == 0
    if np.any(zero_mask):
        kron_median = np.median(kron_radius[~zero_mask]) if np.any(~zero_mask) else 3.5
        kron_radius[zero_mask] = kron_median

    cxx = cxx_image / kron_radius ** 2
    cyy = cyy_image / kron_radius ** 2
    cxy = cxy_image / kron_radius ** 2

    n_hot = len(hottab)
    if verbose:
        print(f"Cold sources: {len(coldtab)}")
        print(f"Hot sources: {n_hot}")
        print(f"Processing in chunks of {chunk_size}...")

    keep_indices = []

    for i in range(0, n_hot, chunk_size):
        end_idx = min(i + chunk_size, n_hot)
        xhot_chunk = np.asarray(hot_comp["xcentroid"][i:end_idx] + 1, dtype=np.float64)
        yhot_chunk = np.asarray(hot_comp["ycentroid"][i:end_idx] + 1, dtype=np.float64)

        delta_x = xhot_chunk[None, :] - xcold[:, None]
        delta_y = yhot_chunk[None, :] - ycold[:, None]

        distances = (
            cxx[:, None] * delta_x ** 2
            + cyy[:, None] * delta_y ** 2
            + cxy[:, None] * delta_x * delta_y
        )

        chunk_keep = np.all(distances > scale_factor ** 2, axis=0)
        keep_indices.extend(np.arange(i, end_idx)[chunk_keep])

        if verbose and (i // chunk_size) % 10 == 0:
            print(f"Processed {end_idx}/{n_hot} hot sources...")

    if verbose:
        print(f"Hot sources kept: {len(keep_indices)}")

    # â”€â”€ filter FULL hot table â”€â”€
    hottab_filtered = hottab[keep_indices]
    hot_labels = np.asarray(hot_comp["label"][keep_indices])

    off = len(coldtab) + 1
    new_labels = np.arange(off, off + len(hottab_filtered))

    # Update label column in catalog
    hottab_filtered["label"] = new_labels.astype(np.int64)

    cold_data = coldsegm.data
    hot_data = hotsegm.data

    # Map old segmentation labels -> new sequential labels
    label_map = dict(zip(hot_labels, new_labels))
    hot_pixels_mask = np.isin(hot_data, hot_labels)

    if len(hot_labels) > 0:
        max_label = max(hot_labels.max(), off + len(hottab_filtered))
        lookup = np.full(max_label + 1, -1, dtype=np.int64)
        for old_label, new_label in label_map.items():
            lookup[old_label] = new_label
        cold_data[hot_pixels_mask] = lookup[hot_data[hot_pixels_mask]]

    outsegm = SegmentationImage(cold_data)
    outtab = vstack((coldtab, hottab_filtered))

    return outtab, outsegm





def SExtractor_HDR(
    filename: str,
    catalog_name: Tuple[str, str, str] = ("coldcat", "hotcat", "outcat"),
    segmap_name: Tuple[str, str, str] = ("coldseg.fits", "hotseg.fits", "outseg.fits"),
    path: str = "./sex/",
    kernel: Tuple[np.ndarray, np.ndarray] = (KERNEL_DEFAULT, KERNEL_DEFAULT),
    detect_minarea: Tuple[int, int] = (5, 5),
    detect_thresh: Tuple[float, float] = (3, 2),
    detect_connectivity: Tuple[Literal[4, 8], Literal[4, 8]] = (
        8,
        8,
    ),  # Detection parameters
    deblend: Tuple[bool, bool] = (False, False),
    deblend_nthresh: Tuple[float, float] = (32, 32),
    deblend_mincont: Tuple[float, float] = (0.002, 0.005),  # Deblending parameters
    clean: Tuple[bool, bool] = (False, False),
    clean_param: Tuple[float, float] = (1.0, 1.0),  # Cleaning parameters
    mask: Optional[np.ndarray] = None,
    coverage_mask: Optional[np.ndarray] = None,
    back_type: Tuple[bool, bool] = (False, False),
    back_value: Tuple[Union[float, np.ndarray], Union[float, np.ndarray]] = (0.0, 0.0),
    back_size: Tuple[int, int] = (128, 32),
    back_filtersize: Tuple[int, int] = (3, 3),
    bkg_estimator: SExtractorBackground = SExtractorBackground(),
    bkgrms_estimator: StdBackgroundRMS = StdBackgroundRMS(),
    weight_type: Literal[
        "NONE", "BACKGROUND", "MAP_RMS", "MAP_WEIGHT", "MAP_VAR"
    ] = "NONE",
    weight_name: Optional[str] = None,
    checkimage_type: List[str] = [],
    phot_apertures: Optional[Union[float, List[float]]] = None,
    phot_autoparams: list[float] = [2.5, 1.4],
    scale_factor: float = 1.1,
    wcs: Optional[WCS] = None,
    pixel_scale: float = 0.06,
    nnw_sex: str = "default.nnw",
    fwhm_arcsec: float = 0.16,
    gain: float = 0.0,
    verbose: bool = False,
    **kwargs: Any,
) -> Tuple[Table, SegmentationImage]:
    """
    Perform HDR (High Dynamic Range) source extraction following the method
    described in "GALAPAGOS: from pixels to parameters" (Barden et al. 2012).

    The function runs two SExtractor passes: a "cold" detection (low‑threshold)
    and a "hot" detection (high‑threshold). The cold detection identifies
    large, birght object, while the hot detection picks up small, faint sources.
    The two catalogs and segmentation maps are then
    combined using `sexcomb`, which retains hot sources that lie outside all
    cold source ellipses (scaled by `scale_factor`) and overlays them onto the
    cold segmentation map. The final output consists of a combined catalog and
    segmentation image, together with DS9 region files for inspection.

    Parameters
    ----------
    filename : str
        Path to the input FITS image.
    catalog_name : tuple of str, optional
        Names for the cold, hot, and combined output catalogs.
        Default is ("coldcat", "hotcat", "outcat").
    segmap_name : tuple of str, optional
        Names for the cold, hot, and combined segmentation FITS files.
        Default is ("coldseg.fits", "hotseg.fits", "outseg.fits").
    path : str, optional
        Output directory. If it exists, it will be removed and recreated.
        Default is "./sex/".
    kernel : tuple of np.ndarray, optional
        Convolution kernels for the cold and hot passes. Default is
        (KERNEL_DEFAULT, KERNEL_DEFAULT).
    detect_minarea : tuple of int, optional
        Minimum number of pixels for detection (cold, hot). Default (5,5).
    detect_thresh : tuple of float, optional
        Detection threshold in units of background RMS (cold, hot). Default (3,2).
    detect_connectivity : tuple of {4,8}, optional
        Pixel connectivity (4 or 8) for source grouping. Each element must be
        either 4 or 8. Default (8,8).
    deblend : tuple of bool, optional
        Whether to deblend sources in each pass. Default (False, False).
    deblend_nthresh : tuple of float, optional
        Number of deblending thresholds. Default (32,32).
    deblend_mincont : tuple of float, optional
        Minimum contrast ratio for deblending. Default (0.002,0.005).
    clean : tuple of bool, optional
        Whether to clean spurious detections. Default (False, False).
    clean_param : tuple of float, optional
        Cleaning efficiency parameter. Default (1.0,1.0).
    mask : np.ndarray or None, optional
        Boolean mask of bad pixels. Default None.
    coverage_mask : np.ndarray or None, optional
        Boolean coverage mask. Default None.
    back_type : tuple of bool, optional
        If True, perform automatic background estimation; if False, use manual
        `back_value`. Default (False, False).
    back_value : tuple of float or np.ndarray, optional
        Background value(s) for manual subtraction. Can be scalars or 2D arrays
        matching the image shape. Default (0.0,0.0).
    back_size : tuple of int, optional
        Background mesh box size. Default (128,32).
    back_filtersize : tuple of int, optional
        Background filter size. Default (3,3).
    bkg_estimator : SExtractorBackground, optional
        Background estimator for the cold/hot passes. Default SExtractorBackground().
    bkgrms_estimator : StdBackgroundRMS, optional
        Background RMS estimator. Default StdBackgroundRMS().
    weight_type : Literal["NONE","BACKGROUND","MAP_RMS","MAP_WEIGHT","MAP_VAR"], optional
        Type of weight map. Default "NONE".
    weight_name : str or None, optional
        Filename of weight map. Required if weight_type != "NONE". Default None.
    checkimage_type : list of str, optional
        Types of check images to produce (passed to SExtractor). Default [].
    phot_apertures : float or list of float, optional
        Radii for aperture photometry. Default None.
    phot_autoparams : list of float, optional
        Parameters for Kron‑like AUTO photometry: [factor, minradius].
        Default [2.5, 1.4].
    scale_factor : float, optional
        Scaling factor for the elliptical distance threshold used in `sexcomb`.
        A hot source is kept only if its scaled elliptical distance to **every**
        cold source is > `scale_factor**2`. Default 1.1.
    wcs : WCS or None, optional
        World Coordinate System object. If None, extracted from FITS header.
    pixel_scale : float, optional
        Pixel scale in arcsec/pixel. Default 0.06.
    gain : float, optional
        Gain (e⁻/ADU) for error propagation. Default 0.0.
    verbose : bool, optional
        If True, print progress information. Default False.
    **kwargs : dict
        Additional keyword arguments passed to both SExtractor calls.

    Returns
    -------
    outtab : `~astropy.table.Table`
        Combined source catalog containing all cold sources and retained hot
        sources, with SExtractor‑like measurements.
    outsegm : `~photutils.segmentation.SegmentationImage`
        Combined segmentation image where hot sources have been inserted into
        the cold segmentation, overwriting any overlapping cold labels.

    Notes
    -----
    The output directory `path` is **removed and recreated** at the beginning
    of the function to ensure a clean workspace. Be cautious not to point it
    to an important existing directory.
    """

    full_path = os.path.abspath(path)  # get absolute path
    current_dir = os.path.abspath(os.getcwd())
    if full_path == current_dir:
        raise ValueError("Error: Trying to delete current directory! Aborting.")
    if os.path.exists(full_path):
        shutil.rmtree(full_path)
    os.mkdir(full_path)

    header = fits.open(filename)[0].header
    if verbose:
        print("*" * 10 + "Cold Detection" + "*" * 10)
    coldtab, coldsegm, coldcat = SExtractor(
        filename,
        catalog_name=path + catalog_name[0],
        kernel=kernel[0],
        detect_minarea=detect_minarea[0],
        detect_thresh=detect_thresh[0],
        detect_connectivity=detect_connectivity[0],
        deblend=deblend[0],
        deblend_nthresh=deblend_nthresh[0],
        deblend_mincont=deblend_mincont[0],
        clean=clean[0],
        clean_param=clean_param[0],
        mask=mask,
        coverage_mask=coverage_mask,
        back_type=back_type[0],
        back_value=back_value[0],
        back_size=back_size[0],
        back_filtersize=back_filtersize[0],
        bkg_estimator=bkg_estimator,
        weight_type=weight_type,
        weight_name=weight_name,
        checkimage_type=checkimage_type,
        phot_apertures=phot_apertures,
        phot_autoparams=phot_autoparams,
        wcs=wcs,
        pixel_scale=pixel_scale,
        nnw_sex=nnw_sex,
        fwhm_arcsec=fwhm_arcsec,
        gain=gain,
        verbose=verbose,
        **kwargs,
    )

    hottab, hotsegm, hotcat = SExtractor(
        filename,
        catalog_name=path + catalog_name[1],
        kernel=kernel[1],
        detect_minarea=detect_minarea[1],
        detect_thresh=detect_thresh[1],
        detect_connectivity=detect_connectivity[1],
        deblend=deblend[1],
        deblend_nthresh=deblend_nthresh[1],
        deblend_mincont=deblend_mincont[1],
        clean=clean[1],
        clean_param=clean_param[1],
        mask=mask,
        coverage_mask=coverage_mask,
        back_type=back_type[1],
        back_value=back_value[1],
        back_size=back_size[1],
        back_filtersize=back_filtersize[1],
        bkg_estimator=bkg_estimator,
        weight_type=weight_type,
        weight_name=weight_name,
        checkimage_type=checkimage_type,
        phot_apertures=phot_apertures,
        phot_autoparams=phot_autoparams,
        wcs=wcs,
        pixel_scale=pixel_scale,
        nnw_sex=nnw_sex,
        fwhm_arcsec=fwhm_arcsec,
        gain=gain,
        verbose=verbose,
        **kwargs,
    )

    fits.writeto(path + segmap_name[0], coldsegm.data, header=header, overwrite=True)
    fits.writeto(path + segmap_name[1], hotsegm.data, header=header, overwrite=True)

    if verbose:
        print("*" * 10 + "HDR Combination" + "*" * 10)
    outtab, outsegm = sexcomb(
        coldtab,
        hottab,
        coldsegm,
        hotsegm,
        scale_factor=scale_factor,
        verbose=verbose,
    )

    fits.writeto(path + segmap_name[2], outsegm.data, header=header, overwrite=True)
    ascii.write(outtab, path + catalog_name[2], overwrite=True)
    ds9reg(
        coldtab, hottab, outtab, pixel_scale=pixel_scale, path=path, fits_image=filename
    )

    if verbose:
        print("*" * 10 + "HDR SExtractor Finished" + "*" * 10)
        print(f"Found {outsegm.nlabels} sources after HDR SExtractor.")
        print("==" * 20)

    return outtab, outsegm


def ds9reg(
    coldtab: Table,
    hottab: Table,
    outtab: Table,
    pixel_scale: float = 0.06,
    path: str = "./sex/",
    fits_image: str = None,
) -> None:
    """
    generate the ds9 region files for cold, hot, and combined catalogs

    parameters
    ----------
    coldtab : astropy.table.Table
        The cold source table. Must contain columns:
        'label', 'ra', 'dec', 'kron_radius', 'semimajor_sigma',
        'semiminor_sigma', 'orientation' (position angle in degrees).
    hottab : astropy.table.Table
        The hot source table. Same required columns as `coldtab`.
    outtab : astropy.table.Table
        Combined catalog. Same required columns as above.
    pixel_scale : float, optional
        Pixel scale of the image in arcsec/pixel. Used to convert pixel sizes
        to arcseconds. Default is 0.06.
    path : str, optional
        Directory where region files will be saved. Default is "./sex/".
    fits_image: str, optional
        Path to the FITS image file. If provided, the WCS rotation angle
        (from image x‑axis to celestial North) is extracted and subtracted
        from the `orientation` column to produce the correct sky position
        angle for DS9. If `None`, no rotation correction is applied.
        Default is None.            

    Returns
    -------
    None
        The function writes three region files (`cold.reg`, `hot.reg`,
        `outcat.reg`) in the specified directory and prints a confirmation
        message. No value is returned.

    """

    def _get_rotation_from_fits(fits_filename: str) -> float:
        """
        Returns the angle (in degrees) from image x‑axis to celestial North.
        This angle should be ADDED to photutils' `orientation` to get the
        DS9 sky position angle (east of north).        
        This has been tested on JWST/SMACS, JWST/CEERS, HST/EGS, and HST/HFF.
        It seems that for sinlge pointing field (like SMACS and HFF), this is not
        necessary.
        """
        with fits.open(fits_filename) as hdul:
            wcs = WCS(hdul[0].header)
            # Get the CD matrix (or PC + CDELT)
            if hasattr(wcs.wcs, "cd"):
                cd = wcs.wcs.cd
            else:
                cd = wcs.wcs.pc * wcs.wcs.cdelt
            # rotation angle = arctan2(CD1_2, CD1_1)
            rot_rad = np.arctan2(cd[0, 1], cd[0, 0])
            rot_deg = np.degrees(rot_rad)
            return rot_deg

    # ---------- compute rotation angle if FITS file is provided ----------
    if fits_image is not None:
        rotation_angle = _get_rotation_from_fits(fits_image)
        print(f"Detected image rotation angle: {rotation_angle:.2f}°")
    else:
        rotation_angle = 0.0
        print("No FITS image provided; using uncorrected orientation.")

    # generate cold region file
    id = coldtab["label"]
    ra = coldtab["ra"]
    dec = coldtab["dec"]
    kron_radius = np.array(coldtab["kron_radius"])
    A = np.array(coldtab["semimajor_sigma"])
    B = np.array(coldtab["semiminor_sigma"])
    pa = (np.array(coldtab["orientation"]) - rotation_angle) % 360
    major_aper = A * kron_radius * pixel_scale
    minor_aper = B * kron_radius * pixel_scale
    header_reg = 'global color=cyan font="helvetica 10 normal" select=1 edit=1 move=1 delete=1 include=1 fixed=0 source=1 \nfk5'
    output = []
    for i in range(len(id)):
        ellipse = 'ellipse({},{},{}",{}",{}) #text={{}}'.format(
            ra[i], dec[i], major_aper[i], minor_aper[i], pa[i]
        )
        output.append(ellipse)
    np.savetxt(path + "cold.reg", output, fmt="%s", header=header_reg, comments="")

    # generate hot region file
    id = hottab["label"]
    ra = hottab["ra"]
    dec = hottab["dec"]
    kron_radius = np.array(hottab["kron_radius"])
    A = np.array(hottab["semimajor_sigma"])
    B = np.array(hottab["semiminor_sigma"])
    pa = (np.array(hottab["orientation"]) - rotation_angle) % 360
    major_aper = A * kron_radius * pixel_scale
    minor_aper = B * kron_radius * pixel_scale
    header_reg = 'global color=orange font="helvetica 10 normal" select=1 edit=1 move=1 delete=1 include=1 fixed=0 source=1 \nfk5'
    output = []
    for i in range(len(id)):
        ellipse = 'ellipse({},{},{}",{}",{}) #text={{}}'.format(
            ra[i], dec[i], major_aper[i], minor_aper[i], pa[i]
        )
        output.append(ellipse)
    np.savetxt(path + "hot.reg", output, fmt="%s", header=header_reg, comments="")

    # generate combined region file
    id = outtab["label"]
    ra = outtab["ra"]
    dec = outtab["dec"]
    kron_radius = np.array(outtab["kron_radius"])
    A = np.array(outtab["semimajor_sigma"])
    B = np.array(outtab["semiminor_sigma"])
    pa = (np.array(outtab["orientation"]) - rotation_angle) % 360
    major_aper = A * kron_radius * pixel_scale
    minor_aper = B * kron_radius * pixel_scale
    header_reg = 'global color=yellow font="helvetica 10 normal" select=1 edit=1 move=1 delete=1 include=1 fixed=0 source=1 \nfk5'
    output = []
    for i in range(len(id)):
        ellipse = 'ellipse({},{},{}",{}",{}) #text={{{}}}'.format(
            ra[i], dec[i], major_aper[i], minor_aper[i], pa[i], id[i]
        )
        output.append(ellipse)
    np.savetxt(path + "outcat.reg", output, fmt="%s", header=header_reg, comments="")

    print("The cold, hot and combined region files has been created")


def SExtractor_dualmode(
    image_list: list[str],
    segm_detect: SegmentationImage,
    detection_cat: SourceCatalog,
    detection_tab: Table,  # input
    catalog_name: Optional[List[str]] = None,
    error_list: Optional[list[str]] = None,
    mask_list: Optional[list[str]] = None,
    coverage_mask: Optional[np.ndarray] = None,
    back_type: bool = False,
    back_value: Union[float, np.ndarray] = 0.0,
    back_size: int = 64,
    back_filtersize: int = 3,
    bkg_estimator: SExtractorBackground = SExtractorBackground(),  # Background parameters
    phot_apertures: Optional[Union[float, List[float]]] = None,
    phot_autoparams: list[float] = [2.5, 1.4],
    mag_zeropoint: float = 0.0,  # Photometry parameters
    wcs: Optional[WCS] = None,
    gain: float = 0.0,
    pixel_scale: float = 0.06,  # image level parameters
    verbose: bool = False,
) -> List[Table]:
    """
    SExtractor dual-mode photometry for multiple images.

    In dual‑mode, source detection is performed only once (on a detection image),
    and the resulting segmentation map is used to measure photometry on one or
    more measurement images. This function takes a pre‑computed segmentation
    image and a detection catalog, then measures fluxes and magnitudes
    on each image provided in `image_list`. Background subtraction is applied
    independently to each measurement image. The results are returned as a list
    of Astropy Tables, one per measurement image, plus the input detection table
    as the first element.

    Parameters
    ----------
    image_list :  list of str
        Paths to the measurement FITS images.
    segm_detect : SegmentationImage
        The segmentation image of detection.
    detection_cat : SourceCatalog
        Source catalog created from the detection image. Used internally to
        initialize measurement catalogs.
    detection_tab : Table
        The detection catalog in Astropy.Table.
    catalog_name : list of str or None, optional
        If provided, a list of output filenames (one per measurement image) to
        which the corresponding measurement tables will be written as ASCII
        catalogs. Default is None (no writing).
    error_list : list of str or None, optional
        Paths to error (variance or RMS) images for each measurement image.
        Must have the same length as `image_list`. If None, no error is used.
        Default None.
    mask_list : list of str or None, optional
        Paths to mask images (boolean, 1 = masked) for each measurement image.
        Must have the same length as `image_list`. If None, no mask is applied.
        Default None.
    coverage_mask : 2D `~numpy.ndarray` (bool) or None, optional
        Boolean coverage mask applied during background estimation. Default None.
    back_type : bool, optional
        If True, perform automatic background estimation using `Background2D`.
        If False, use manual subtraction with `back_value`. Default False.
    back_value : float or 2D `~numpy.ndarray`, optional
        Background value for manual subtraction. If a scalar, the same value is
        subtracted from the whole image; if a 2D array, it must match the image
        shape. Default 0.0.
    back_size : int, optional
        Size of background mesh boxes for `Background2D`. Default 64.
    back_filtersize : int, optional
        Filter size for background smoothing. Default 3.
    bkg_estimator : `~photutils.background.BackgroundBase`, optional
        Background estimator used by `Background2D`. Default SExtractorBackground().
    phot_apertures : float or list of float, optional
        Radii for circular aperture photometry (in pixels). Default None.
    phot_autoparams : list of float, optional
        Parameters [factor, min_radius] for Kron‑like AUTO photometry.
        Default [2.5, 1.4].
    mag_zeropoint : float, optional
        Magnitude zeropoint for converting fluxes to magnitudes. Default 0.0.
    wcs : WCS or None, optional
        World Coordinate System object. If None, not used. Default None.
    gain : float, optional
        Gain (e⁻/ADU) for error propagation. Default 0.0.
    pixel_scale : float, optional
        Pixel scale in arcsec/pixel. Default 0.06.
    verbose : bool, optional
        If True, print progress information. Default False.

    Returns
    -------
    measurement_tab_list : list of `~astropy.table.Table`
        A list of Astropy Tables. The first element is the input `detection_tab`.
        Subsequent elements correspond to each measurement image, in the same
        order as `image_list`. Each table contains measured photometry (fluxes,
        magnitudes, etc.) for the sources defined by `segm_detect`.
    """

    if error_list is None:
        error_list = [None] * len(image_list)
    if mask_list is None:
        mask_list = [None] * len(image_list)

    if verbose:
        print("*" * 10 + "SExtractor Dual Mode" + "*" * 10)
        print(f"Number of images to be measured: {len(image_list)}")
        print("No detection will be performed on measurement images.")
        print("==" * 20)
    measurement_tab_list = [
        detection_tab
    ]  # List to hold measurement catalogs for each image

    for ii in range(len(image_list)):
        image = fits.getdata(image_list[ii])
        error = None if error_list[ii] is None else fits.getdata(error_list[ii])
        mask = None if mask_list[ii] is None else fits.getdata(mask_list[ii])

        bkg = Background2D(
            image,
            (back_size, back_size),
            mask=mask,
            coverage_mask=coverage_mask,
            filter_size=(back_filtersize, back_filtersize),
            bkg_estimator=bkg_estimator,
        )
        if back_type:  # Automatic background subtraction
            background = bkg.background
            image_bkgsub = image - background
            if verbose:
                print("Automatic Background Estimation and Subtraction Finished")
                print(f"  background size = {back_size}")
                print(f"  background filter size = {back_filtersize}")
                print("==" * 20)
        else:  # Manual background subtraction
            if np.isscalar(back_value):
                background = np.full_like(image, back_value)
            elif isinstance(back_value, np.ndarray) and back_value.shape == image.shape:
                background = back_value
            else:
                raise ValueError(
                    "back_value should be a scalar or a 2D array with the same shape as the image."
                )
            image_bkgsub = image - background
            if verbose:
                print("Manual Background Subtraction Finished")
                if np.isscalar(back_value):
                    print(f"  Background value = {back_value}")
                print("==" * 20)

        # image_convolved = convolve_fft(image_bkgsub, detect_filter, mask=mask)

        measurement_cat = SourceCatalog(
            image_bkgsub,
            segm_detect,
            error=error,
            background=background,
            detection_cat=detection_cat,
        )
        measurement_tab = cat2tab(measurement_cat, phot_apertures=phot_apertures)
        if catalog_name is not None:
            ascii.write(measurement_tab, catalog_name[ii], overwrite=True)
        measurement_tab_list.append(measurement_tab)

        if verbose:
            print(f"Finished measurement for image {ii+1}")
            print("==" * 20)

    if verbose:
        print("*" * 6 + "SExtractor Dual Mode Finished" + "*" * 6)

    return measurement_tab_list


def se_make_kronmask(
    cat: SourceCatalog, kron_params: Tuple[float, float] = (2.5, 1.4)
) -> np.ndarray:
    """
    Create a boolean mask covering all Kron apertures for sources in the catalog.

    The mask is built by iterating over each source's Kron aperture (as defined by
    `kron_params`), converting it to a pixel mask using the ``'center'`` method,
    and combining them via logical OR. The resulting mask has the same dimensions
    as the original image from which the catalog was constructed.

    Parameters
    ----------
    cat : SourceCatalog
        The source catalog.
    kron_params : tuple of float, optional
        Parameters for the Kron aperture: (factor, min_radius). Default is (2.5, 1.4),
        matching SExtractor's AUTO aperture settings.

    Returns
    -------
    kronmask : 2D `~numpy.ndarray` of bool
        Boolean array of the same shape as the original image. Pixels belonging
        to any Kron aperture are set to `True`; all others are `False`.
    """
    kronaper = cat.make_kron_apertures(kron_params=kron_params)
    kronmask = np.zeros(cat._data.shape, dtype=bool)
    for i in range(len(kronaper)):
        slices_large, slices_small = (
            kronaper[i].to_mask(method="center").get_overlap_slices(cat._data.shape)
        )
        kronmask[slices_large] |= (
            kronaper[i].to_mask(method="center").data[slices_small].astype(bool)
        )

    return kronmask


# from __future__ import annotations

from dataclasses import dataclass
from math import log10
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class SExtractorNNW:
    """
    NumPy-backed implementation of SExtractor's CLASS_STAR neural net.

    Supports arbitrary layer counts, but default.nnw is typically:
      10 inputs -> 10 hidden -> 1 output
    """

    nn: tuple[int, ...]  # neurons per layer
    inbias: np.ndarray  # (nin,)
    inscale: np.ndarray  # (nin,)
    W: tuple[np.ndarray, ...]  # each (n_from, n_to)
    b: tuple[np.ndarray, ...]  # each (n_to,)
    outbias: np.ndarray  # (nout,)
    outscale: np.ndarray  # (nout,)

    @property
    def nin(self) -> int:
        return self.nn[0]

    @property
    def nout(self) -> int:
        return self.nn[-1]

    @staticmethod
    def load_nnw(path: str | Path) -> "SExtractorNNW":
        """
        Parse an SExtractor .nnw file (e.g. config/default.nnw).

        The file is a token stream:
          - header "NNW"
          - layersnb and nn[...]
          - inbias[nin], inscale[nin]
          - for each layer l: for each neuron j in layer l+1:
                weights nn[l] then bias
          - outbias[nout], outscale[nout]
        """
        path = Path(path)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines or not lines[0].lstrip().startswith("NNW"):
            raise ValueError(f"{path} is NOT a NNW table (missing 'NNW' header)")

        tokens: list[str] = []
        for line in lines[1:]:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tokens.extend(s.split())

        pos = 0
        layersnb = int(tokens[pos])
        pos += 1
        nn = tuple(int(tokens[pos + i]) for i in range(layersnb))
        pos += layersnb

        nin = nn[0]
        nout = nn[-1]

        inbias = np.array(
            [float(tokens[pos + i]) for i in range(nin)], dtype=np.float64
        )
        pos += nin
        inscale = np.array(
            [float(tokens[pos + i]) for i in range(nin)], dtype=np.float64
        )
        pos += nin

        W_list: list[np.ndarray] = []
        b_list: list[np.ndarray] = []

        for l in range(layersnb - 1):
            n_from = nn[l]
            n_to = nn[l + 1]

            W = np.empty((n_from, n_to), dtype=np.float64)
            b = np.empty((n_to,), dtype=np.float64)

            for j in range(n_to):
                # nn[l] weights then 1 bias
                W[:, j] = [float(tokens[pos + i]) for i in range(n_from)]
                pos += n_from
                b[j] = float(tokens[pos])
                pos += 1

            W_list.append(W)
            b_list.append(b)

        outbias = np.array(
            [float(tokens[pos + i]) for i in range(nout)], dtype=np.float64
        )
        pos += nout
        outscale = np.array(
            [float(tokens[pos + i]) for i in range(nout)], dtype=np.float64
        )
        pos += nout

        if pos != len(tokens):
            raise ValueError(
                f"{path}: trailing tokens after parsing (pos={pos}, len={len(tokens)})"
            )

        return SExtractorNNW(
            nn=nn,
            inbias=inbias,
            inscale=inscale,
            W=tuple(W_list),
            b=tuple(b_list),
            outbias=outbias,
            outscale=outscale,
        )

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        """
        X: (batch, nin)
        returns: (batch, nout)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.nin:
            raise ValueError(f"X must have shape (batch, {self.nin}), got {X.shape}")

        a = X * self.inscale + self.inbias
        for W, b in zip(self.W, self.b):
            a = _sigmoid(a @ W + b)
        return (a - self.outbias) / self.outscale

    def forward_one(self, x: np.ndarray) -> np.ndarray:
        """
        x: (nin,)
        returns: (nout,)
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.nin,):
            raise ValueError(f"x must have shape ({self.nin},), got {x.shape}")
        return self.forward_batch(x[None, :])[0]


def _log10_safe(v: np.ndarray) -> np.ndarray:
    # v is positive by construction (we clamp), so log10 is safe
    return np.log10(v)


def build_class_star_features_batch(
    *,
    iso: np.ndarray,
    peak: np.ndarray,
    field_thresh: np.ndarray,
    fwhm_arcsec: np.ndarray,
    pixscale_arcsec: np.ndarray,
    nin: int = 10,
) -> np.ndarray:
    """
    Vectorized replication of SExtractor analyse.c feature building.

    Inputs:
      iso: (batch, n_iso)  where n_iso should typically be 8 (A0..A7)
      peak: (batch,)
      field_thresh: (batch,)  (field->thresh per object/field)
      fwhm_arcsec: (batch,)
      pixscale_arcsec: (batch,)

    Output:
      X: (batch, nin)  (nin typically 10)
    """
    iso = np.asarray(iso, dtype=np.float64)
    peak = np.asarray(peak, dtype=np.float64)
    field_thresh = np.asarray(field_thresh, dtype=np.float64)
    fwhm_arcsec = np.asarray(fwhm_arcsec, dtype=np.float64)
    pixscale_arcsec = np.asarray(pixscale_arcsec, dtype=np.float64)

    if iso.ndim != 2:
        raise ValueError(f"iso must be 2D (batch, n_iso), got {iso.shape}")
    batch = iso.shape[0]
    for arr, name in [
        (peak, "peak"),
        (field_thresh, "field_thresh"),
        (fwhm_arcsec, "fwhm_arcsec"),
        (pixscale_arcsec, "pixscale_arcsec"),
    ]:
        if arr.shape != (batch,):
            raise ValueError(f"{name} must have shape ({batch},), got {arr.shape}")

    if np.any(fwhm_arcsec <= 0.0) or np.any(pixscale_arcsec <= 0.0):
        raise ValueError("fwhm_arcsec and pixscale_arcsec must be > 0")

    # fac2 = (fwhm/pixscale)^2  (per object)
    fac2 = (fwhm_arcsec / pixscale_arcsec) ** 2  # (batch,)

    # Normalize iso by fac2 with broadcasting
    iso_norm = iso / fac2[:, None]  # (batch, n_iso)

    # SExtractor uses defaults: if iso[i]==0 => 0.01; if peak<=0 => 0.1; if thresh<=0 => -1
    iso_norm = np.where(iso > 0.0, iso_norm, 0.01)

    # Build X
    X = np.empty((batch, nin), dtype=np.float64)

    # slot 0: log10(iso0/fac2 or 0.01)
    X[:, 0] = _log10_safe(iso_norm[:, 0])

    # slot 1: log10(peak/thresh) with defaults, else -1 if thresh<=0
    peak_over_thresh = np.where(peak > 0.0, peak / field_thresh, 0.1)
    X[:, 1] = np.where(field_thresh > 0.0, _log10_safe(peak_over_thresh), -1.0)

    # slots 2..nin-2: iso[1..] (pad with log10(0.01) if not enough iso columns)
    fill = log10(0.01)
    X[:, 2 : nin - 1] = fill

    # how many iso-derived slots do we actually fill?
    # We need (nin-1) total pre-fwhm slots; already used 2 => remaining = (nin-1)-2 = nin-3
    want = nin - 3
    have = max(0, iso.shape[1] - 1)  # iso[1..]
    take = min(want, have)
    if take > 0:
        X[:, 2 : 2 + take] = _log10_safe(iso_norm[:, 1 : 1 + take])

    # last slot: log10(fwhm)
    X[:, -1] = np.log10(fwhm_arcsec)

    return X


def class_star_predict_batch(
    *,
    nn: SExtractorNNW,
    iso: np.ndarray,
    peak: np.ndarray,
    field_thresh: np.ndarray,
    fwhm_arcsec: np.ndarray,
    pixscale_arcsec: np.ndarray,
) -> np.ndarray:
    """
    Returns CLASS_STAR for each object: shape (batch,)
    """
    X = build_class_star_features_batch(
        iso=iso,
        peak=peak,
        field_thresh=field_thresh,
        fwhm_arcsec=fwhm_arcsec,
        pixscale_arcsec=pixscale_arcsec,
        nin=nn.nin,
    )
    y = nn.forward_batch(X)  # (batch, 1) for default.nnw
    return y[:, 0]


def class_star_predict_one(
    *,
    nn: SExtractorNNW,
    iso: Sequence[float],
    peak: float,
    field_thresh: float,
    fwhm_arcsec: float,
    pixscale_arcsec: float,
) -> float:
    """
    Convenience wrapper for a single object (still uses NumPy path).
    """
    iso_arr = np.asarray(iso, dtype=np.float64)[None, :]
    y = class_star_predict_batch(
        nn=nn,
        iso=iso_arr,
        peak=np.asarray([peak], dtype=np.float64),
        field_thresh=np.asarray([field_thresh], dtype=np.float64),
        fwhm_arcsec=np.asarray([fwhm_arcsec], dtype=np.float64),
        pixscale_arcsec=np.asarray([pixscale_arcsec], dtype=np.float64),
    )
    return float(y[0])


# import numpy as np

# from sextractor_class_star import SExtractorNNW, class_star_predict_batch


def compute_iso_areas_A0_A7(
    seg: np.ndarray,
    img: np.ndarray,
    analysis_thresh: np.ndarray | float,
    n_iso: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute iso areas A0..A7 and peak for each object id in seg.

    Parameters:
        seg: 分割图像，值为对象ID
        img: 背景扣除后的图像
        analysis_thresh: 阈值，可以是标量或一维数组 (nobj,)
                        如果是一维数组，每个对象使用自己的阈值
        n_iso: 等照度环数量，默认为8

    Returns:
        iso: (Nobj, n_iso) float64, areas in pixels (counts of pixels)
        peak: (Nobj,) float64, peak value above background within segmentation
    """
    seg = np.asarray(seg)
    img = np.asarray(img, dtype=np.float64)
    if seg.shape != img.shape:
        raise ValueError("seg and img must have same shape")

    obj_ids = np.unique(seg)
    obj_ids = obj_ids[obj_ids > 0]
    nobj = obj_ids.size

    # 处理analysis_thresh参数
    if np.isscalar(analysis_thresh):
        # 如果是标量，为所有对象创建相同的阈值数组
        analysis_thresh = np.full(nobj, float(analysis_thresh), dtype=np.float64)
    else:
        # 确保是一维数组
        analysis_thresh = np.asarray(analysis_thresh, dtype=np.float64).ravel()
        if analysis_thresh.shape[0] != nobj:
            raise ValueError(
                f"analysis_thresh must have size {nobj} (number of objects) or be scalar, "
                f"got shape {analysis_thresh.shape}"
            )

    iso = np.zeros((nobj, n_iso), dtype=np.float64)
    peak = np.zeros((nobj,), dtype=np.float64)

    # 预计算每个标签的像素索引（通过排序，对于大图像比N个掩码更快）
    flat_id = seg.ravel()
    flat_img = img.ravel()

    order = np.argsort(flat_id, kind="mergesort")
    flat_id_sorted = flat_id[order]
    flat_img_sorted = flat_img[order]

    # 找到每个标签的运行
    # (跳过标签0的运行)
    unique_ids, start_idx, counts = np.unique(
        flat_id_sorted, return_index=True, return_counts=True
    )
    # 映射 label->(start,count)
    run = {
        int(i): (int(s), int(c))
        for i, s, c in zip(unique_ids, start_idx, counts)
        if i > 0
    }

    for k, oid in enumerate(obj_ids):
        s, c = run[int(oid)]
        vals = flat_img_sorted[s : s + c]

        imax = vals.max() if vals.size else 0.0
        peak[k] = imax

        # 获取当前对象的阈值
        thresh = analysis_thresh[k]

        # 如果峰值低于阈值，面积基本最小
        if imax <= thresh or thresh <= 0:
            # SExtractor代码在下游使用 iso[i]? ... : 0.01；这里保持0，让默认值应用
            continue

        # 在thresh和peak之间指数间隔的级别
        # L_0 = T, L_7 = Imax
        ratio = imax / thresh
        # levels = thresh * (ratio ** (np.arange(n_iso, dtype=np.float64) / (n_iso - 1)))
        levels = thresh * (ratio ** (np.arange(n_iso, dtype=np.float64) / n_iso))

        # 在对象足迹内每个等照度线以上的面积
        # 使用严格的">"与典型阈值处理一致；精确的相等很少重要
        # vals[:,None] 创建 (npix, n_iso); np.sum 给出 (n_iso,)
        iso[k, :] = (vals[:, None] > levels[None, :]).sum(axis=0)

    return iso, peak


def class_star_from_segmentation(
    *,
    seg: np.ndarray,
    img_bkgsub: np.ndarray,
    analysis_thresh: np.ndarray | float,
    fwhm_arcsec: float,
    pixscale_arcsec: float,
    nnw_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    End-to-end stellarity (CLASS_STAR) derived from seg + bkg-subtracted image.

    Parameters:
        seg: 分割图像
        img_bkgsub: 背景扣除后的图像
        analysis_thresh: 阈值，可以是标量或一维数组 (nobj,)
        fwhm_arcsec: FWHM (角秒)
        pixscale_arcsec: 像素尺度 (角秒/像素)
        nnw_path: NNW权重文件路径

    Returns:
        obj_ids: (Nobj,) object labels
        class_star: (Nobj,) float
    """
    nn = SExtractorNNW.load_nnw(nnw_path)

    obj_ids = np.unique(seg)
    obj_ids = obj_ids[obj_ids > 0]

    # 计算等照度面积和峰值
    iso, peak = compute_iso_areas_A0_A7(seg, img_bkgsub, analysis_thresh, n_iso=8)

    # 处理analysis_thresh参数
    nobj = len(obj_ids)
    if np.isscalar(analysis_thresh):
        # 如果是标量，为所有对象创建相同的阈值数组
        field_thresh = np.full(nobj, float(analysis_thresh), dtype=np.float64)
    else:
        # 确保是一维数组
        field_thresh = np.asarray(analysis_thresh, dtype=np.float64).ravel()
        if field_thresh.shape[0] != nobj:
            raise ValueError(
                f"analysis_thresh must have size {nobj} (number of objects) or be scalar, "
                f"got shape {field_thresh.shape}"
            )

    fwhm = np.full((nobj,), float(fwhm_arcsec), dtype=np.float64)
    pixscale = np.full((nobj,), float(pixscale_arcsec), dtype=np.float64)

    class_star = class_star_predict_batch(
        nn=nn,
        iso=iso,
        peak=peak,
        field_thresh=field_thresh,
        fwhm_arcsec=fwhm,
        pixscale_arcsec=pixscale,
    )

    return obj_ids, class_star, peak











############### Building flags.
# obj[j].flag |= OBJ_MERGED	/* Merge flag on */
#   | ((OBJ_ISO_PB|OBJ_APERT_PB|OBJ_OVERFLOW)
#   &debobjlist2.obj[0].flag);
def compute_obj_merged_from_parent_child(segm_parent, segm_child):
    """
    Compute an OBJ_MERGED-like flag for each child object, given:
      - segm_parent: segmentation BEFORE deblending (parent islands)
      - segm_child:  segmentation AFTER deblending (child components)

    Returns
    -------
    child_ids : 1D array of child labels (excluding 0)
    merged_flag : 1D bool array aligned with child_ids
        True if child's parent produced >1 child.
    child_to_parent : dict {child_id: parent_id}
    parent_to_children : dict {parent_id: set(child_ids)}
    """
    segm_parent = np.asarray(segm_parent)
    segm_child = np.asarray(segm_child)
    if segm_parent.shape != segm_child.shape:
        raise ValueError(f"Shape mismatch: parent {segm_parent.shape} vs child {segm_child.shape}")

    # Only pixels that are labeled in the child map matter.
    mask = segm_child > 0
    if not np.any(mask):
        return np.array([], dtype=int), np.array([], dtype=bool), {}, {}

    parent_labels = segm_parent[mask].astype(np.int64)
    child_labels  = segm_child[mask].astype(np.int64)

    # If your deblending ever creates child pixels outside the original parent islands,
    # parent_labels could be 0 for some child pixels; we handle that below.
    pairs = np.stack([parent_labels, child_labels], axis=1)

    # Deduplicate (parent, child) pairs so we can build mappings efficiently.
    pairs = np.unique(pairs, axis=0)

    parent_to_children = {}
    child_to_parent = {}

    for p, c in pairs:
        if c == 0:
            continue
        parent_to_children.setdefault(int(p), set()).add(int(c))

    # Define child->parent.
    # Usually each child belongs to exactly one parent. If a child overlaps multiple parents,
    # you should resolve it (majority vote). Here we use majority vote by pixel counts.
    # We'll implement majority vote in a separate step for robustness.
    child_to_parent = majority_parent_for_each_child(segm_parent, segm_child)

    child_ids = np.array(sorted(child_to_parent.keys()), dtype=int)

    merged_flag = np.zeros(child_ids.shape[0], dtype=bool)
    for i, c in enumerate(child_ids):
        p = child_to_parent[c]
        children = parent_to_children.get(p, set())
        merged_flag[i] = (len(children) > 1)

    return child_ids, merged_flag, child_to_parent, parent_to_children


def majority_parent_for_each_child(segm_parent, segm_child):
    """
    Robustly map each child label to a single parent label by majority pixel overlap.
    Returns dict {child_id: parent_id}. Background parent_id can be 0.
    """
    segm_parent = np.asarray(segm_parent)
    segm_child = np.asarray(segm_child)

    mask = segm_child > 0
    parent_labels = segm_parent[mask].astype(np.int64)
    child_labels  = segm_child[mask].astype(np.int64)

    # Count overlaps for each (child, parent) pair
    # We'll encode pairs into a single integer key to use bincount efficiently.
    max_parent = int(parent_labels.max()) if parent_labels.size else 0
    key = child_labels * (max_parent + 1) + parent_labels
    counts = np.bincount(key)

    child_to_parent = {}
    child_ids = np.unique(child_labels)
    for c in child_ids:
        # keys for this child are in [c*(max_parent+1) ... c*(max_parent+1)+max_parent]
        start = c * (max_parent + 1)
        block = counts[start:start + (max_parent + 1)]
        p = int(block.argmax())
        child_to_parent[int(c)] = p

    return child_to_parent





##### flag obj_crowded.
CROWD_THRESHOLD = 0.1
def kron_crowded_flag(
    segm, weight, obj_id,
    x0, y0,  # pixel coordinates; must be consistent with segm/weight indexing
    a_image, b_image, theta_deg,
    kron_radius,
    weight_bad_threshold=None,
):
    """
    Compute OBJ_CROWDED-like flag for one object.

    Parameters
    ----------
    segm : 2D int array
        Segmentation image. Pixels are 0 (background) or object IDs.
    weight : 2D float array or None
        Weight/variance-like map. If provided, pixels are marked bad if weight >= weight_bad_threshold
        (to mimic SExtractor's 'var >= wthresh' logic). If your map is inverse-variance or weight,
        you may need to invert the comparison (see notes below).
    obj_id : int
        Object ID in segm.
    x0, y0 : float
        Object center in pixel coords (0-based). If your catalog is 1-based (FITS convention),
        pass x0-1, y0-1.
    a_image, b_image : float
        Semi-major/minor axes (in pixels) corresponding to A_IMAGE, B_IMAGE.
    theta_deg : float
        Position angle in degrees. (SExtractor's THETA_IMAGE is typically degrees.)
    kron_radius : float
        KRON_RADIUS scaling. SExtractor uses ellipse size ~ A_IMAGE*kron_radius, B_IMAGE*kron_radius
        for MAG_AUTO aperture.
    weight_bad_threshold : float or None
        If not None and weight is not None: mark pixel bad when weight >= threshold.

    Returns
    -------
    crowded : bool
        True if areab/area > 0.1
    frac_bad : float
        areab/area
    area : int
        Pixels inside ellipse
    areab : int
        "Bad" pixels inside ellipse
    """
    h, w = segm.shape

    # Kron ellipse semi-axes (pixels)
    a = float(a_image) * float(kron_radius)
    b = float(b_image) * float(kron_radius)
    if a <= 0 or b <= 0:
        return False, 0.0, 0, 0

    # Build a tight bounding box around the ellipse
    # Conservative half-size: use max(a,b) in both axes
    r = int(np.ceil(max(a, b) + 2))
    x_min = max(int(np.floor(x0 - r)), 0)
    x_max = min(int(np.ceil(x0 + r)) + 1, w)
    y_min = max(int(np.floor(y0 - r)), 0)
    y_max = min(int(np.ceil(y0 + r)) + 1, h)

    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    dx = xx - x0
    dy = yy - y0

    # Rotate by theta to align with ellipse axes
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    # x' =  dx*c + dy*s ; y' = -dx*s + dy*c  (standard image rotation)
    xp =  dx * c + dy * s
    yp = -dx * s + dy * c

    inside = (xp*xp)/(a*a) + (yp*yp)/(b*b) <= 1.0
    area = int(np.count_nonzero(inside))
    if area == 0:
        return False, 0.0, 0, 0

    seg_cut = segm[y_min:y_max, x_min:x_max]

    # Define "bad" pixels:
    # (1) pixels that belong to some OTHER object (close companions) inside this object's Kron ellipse
    bad = inside & (seg_cut != 0) & (seg_cut != obj_id)

    # (2) optionally weight-based bad pixels
    if weight is not None and weight_bad_threshold is not None:
        w_cut = weight[y_min:y_max, x_min:x_max]
        bad |= inside & (w_cut >= weight_bad_threshold)    ## a small weight value means pixel with high noise.

    areab = int(np.count_nonzero(bad))
    frac_bad = areab / area
    crowded = frac_bad > CROWD_THRESHOLD
    return crowded, frac_bad, area, areab


def update_flag_bitwise(total_flag, new_flag):
    if new_flag:
        total_flag = (total_flag << 1) | 1
    return total_flag




