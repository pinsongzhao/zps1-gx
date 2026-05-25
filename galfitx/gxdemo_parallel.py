import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import sys
galfitS_path = "/share/pszhao/GalfitS-main/src/galfits/galfitS.py"
import subprocess
import numpy as np
import shutil
from astropy.io import fits, ascii
from astropy.table import Table
from galfitx.source_detection import SExtractor, SExtractor_HDR
from galfitx.postage_stamp import create_stamp_file, cut_stamps
from galfitx.sky_calculation import getsky, create_skymap
from galfitx.read_setup import read_setup, set_trailing_slash
from galfitx.utils import scaled_kron
from galfitx.create_setup_gs import prepare_galfits, create_mask, process_psf, reproject_segm, gen_pSed_data_lyric
from galfitx.gx_gsutils import calconvfactor, effective_wave, Union_Set
from galfitx.model_isoflux import process_multi_band_photometry, PhotometryConfig, PhotometryPipeline
from galfitx.eazy_utils import zphot_config, translate_config, run_eazy, show_all_fitting
from galfitx.pureSed import pureSed_class
import datetime
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
template_dir = "./sps_templates/"
SPS_catalog_path = os.path.join(template_dir, "UNCOVER_DR4_SPS_catalog.fits")
sfhs_path = os.path.join(template_dir, "sfhs_SPS_DR4.npz")

print(SPS_catalog_path)
print(sfhs_path)

public_galfit_path = '/home/zps/galfitm/galfitm-1.4.4-linux-x86_64'
# =============================================================================
# Multi-GPU Parallel Worker Function
# =============================================================================

def process_group(i_group, group, object_dict, outtab_dict, bg_params_dict,
                  nband, sciname_list, label_list, filter_list, psf_list,
                  zero_list, pixel_scale_list, expt_list, mjsr_list, gain_list,
                  det_label, catalog_name, cutout_dir, gs_pureImage_dir,
                  gs_pureSed_dir, gs_image_sed_dir, gmoutdir, skynoisepath,
                  stampfile, psf_file, try_filterlist, apers_list,
                  kron_scale, ref_pixel_scale, eazy_path, num_gpus,
                  whtname_list,weight_name):
    """
    Worker function to process a single galaxy group on an assigned GPU.

    Parameters
    ----------
    i_group : int
        Group index
    group : list
        List of object indices in this group
    object_dict : dict
        Dictionary mapping object index to list of co-fitted objects
    outtab_dict : dict
        Catalog data as dictionary (for pickling)
    bg_params_dict : dict
        Background noise parameters as dictionary (for pickling)
    num_gpus : int
        Total number of GPUs available for round-robin assignment
    ... (other parameters are configuration values)

    Returns
    -------
    str
        Status message indicating completion
    """

    # Assign GPU for this worker using round-robin
    gpu_id = i_group % num_gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"[Group {i_group}] Assigned to GPU {gpu_id}")

    # Convert dict back to DataFrame for bg_params
    import pandas as pd
    bg_df = pd.DataFrame(bg_params_dict)
    # perfer to use public path in server 206
    if os.path.exists(public_galfit_path):
        galfit_path = public_galfit_path
    else:
        # fallback to dynamic lookup
        galfit_path = shutil.which('galfitm')
        if galfit_path is None:
            raise FileNotFoundError("No galfitm found: neither public path nor in PATH.")

    seglist = np.array([sciname.replace(".fits", "_seg.fits") for sciname in sciname_list])

    iso_config = PhotometryConfig(
                galfit_path = galfit_path, sky_noise_path = skynoisepath, gmoutdir = gmoutdir,
                label_list = label_list, filter_labels = filter_list, wht_file_list = whtname_list,
                galaxy_catalog = catalog_name, image_list = sciname_list, kron_scale = kron_scale,
                save_mask = True, apertures_list = apers_list, pixel_scales = pixel_scale_list, save_regions = True,
                plot_histograms = True, psf_file = psf_file, segmentation_map_list=seglist, cutout_dir=cutout_dir,
                gains = gain_list,
                exptimes=expt_list, mjsr_list=mjsr_list, zero_list=zero_list, try_filterlist = try_filterlist,
                ref_pixel_scale = ref_pixel_scale,
                stampfile = stampfile,detwht_file=weight_name,external_mask_file_list=external_mask_file_list
                )
    iso_pipeline = PhotometryPipeline(iso_config)

    # Sort group by magnitude
    sorted_idx_group = sorted(group, key = lambda x: outtab_dict["mag_auto"][x])
    sorted_idx_group_1idx = [value + 1 for value in sorted_idx_group]

    # ========================================================================
    # Part 1: Pure Image Fitting with GALFITS
    # ========================================================================
    for idx in sorted_idx_group:

        ## cofitting objects.
        objects = object_dict[str(idx)]

        #prepare the fitting materials.
        image_stamp_list = []
        mask_file_list = []

        for i in range(nband):
            label = label_list[i]
            maskfile = os.path.join(cutout_dir, f"obj{idx+1}_{label}mask.fits")
            image_file = os.path.join(cutout_dir, f"obj{idx+1}_{label}sci.fits")
            image_stamp_list.append(image_file)
            mask_file_list.append(maskfile)

        image_stamp_list = np.array(image_stamp_list)
        mask_file_list = np.array(mask_file_list)

        # cut psf file to suitable size
        psf_file_used = process_psf(idx+1, nband, psf_list, image_stamp_list, label_list, processed_psf_dir) # cut psf file to suitable size
       
        ## output lyric file.
        lyric_path = os.path.join(gs_pureImage_dir, f"obj_{idx+1}.lyric")
        prior_path = os.path.join(gs_pureImage_dir, f"obj_{idx+1}_sfh.prior")

        prepare_galfits(lyric_path = lyric_path, cat_file = catalog_name, prior_path = prior_path,objects = objects,
                        det_label = det_label, sci_list = image_stamp_list, psf_list = psf_file_used,
                        zero_list = zero_list, pixscl_list = pixel_scale_list,
                        label_list = label_list, filter_list=filter_list,
                        geo_smdir = gs_pureImage_dir, pSed_smdir = gs_pureSed_dir, imgSed_smdir = gs_image_sed_dir,
                        mask_list = mask_file_list, use_sed = 0, convf = False)


        command = f"python {galfitS_path} --config \
        {lyric_path} --workplace {gs_pureImage_dir} --fit_method ES --num_generations 10000"

        os.system(command)

    # ========================================================================
    # Part 2: Compute Iso Flux and Errors
    # ========================================================================
    gs_flux_outfile = os.path.join(gmoutdir, f"gs_iso_flux_group{i_group}.cat")
    gs_flux_err_outfile = os.path.join(gmoutdir, f"gs_iso_flux_err_group{i_group}.cat")
    iso_pipeline._compute_model_fluxes_for_galaxies(
            galids = sorted_idx_group_1idx, filter_list = filter_list, image_list = sciname_list,
        gsdir = gs_pureImage_dir, outdir = gmoutdir, output_file = gs_flux_outfile,
    )
    iso_pipeline.step4_compute_isophotal_errors(bg_df = bg_df, flux_df_path  = gs_flux_outfile, gs_flux_err_outfile = gs_flux_err_outfile)

    # ========================================================================
    # Part 3: EAZY Photometric Redshift Fitting
    # ========================================================================
    # generate eazy config and translate file

    # please check the best error setting for your photometry catalog
    # suggestions: temperr in [0.00, 0.20]; syserr in [0.00,0.20]
    # the additinal template error will included in fitting
    temperr = 0.01
    # the additinal systematic error will added in fitting
    syserr = 0.01
    eazy_out_path = f"./eazy/{i_group}"
    config_file_param = os.path.join(eazy_out_path, "zphot.param")
    os.makedirs(eazy_out_path, exist_ok=True)
    # generate the config for EAZY
    zphot_config(gs_flux_err_outfile, eazy_out_path, temperr, syserr, eazypath=eazy_path, configfile = config_file_param)
    # generate the translate config bewteen photometry catalog and EAZY
    config_file_translate = os.path.join(eazy_out_path,"zphot.translate")
    translate_config(gs_flux_err_outfile, configfile = config_file_translate)

    # User can use this to run eazy,
    # or modify eazy config in terminal for more parameters and run eazy in terminal with command "eazy_path/src/eazy -t zphot.translate"
    run_eazy(eazypath=eazy_path, configfile = config_file_param, translatefile=config_file_translate)

    # show fitting results of EAZY, including photz-specz comparison figure and sed-fitting, pdf, and galfits fitting figures
    figure_output = os.path.join(eazy_out_path, "figures")
    os.makedirs(figure_output, exist_ok=True)
    show_all_fitting(eazy_out_path, gs_pureImage_dir, output_path = figure_output)

    # ========================================================================
    # Part 4: Pure SED Fitting
    # ========================================================================
    ebv = 0.01 ## will automately search using package.

    os.makedirs(eazy_out_path, exist_ok=True)
    z_out_path = os.path.join(eazy_out_path, "photz.zout")

    gen_pSed_data_lyric(cat_path = gs_flux_err_outfile, z_cat_path = z_out_path, 
                        cutout_dir=cutout_dir, mock_dir = gs_pureSed_dir, label_list = label_list, 
                        filter_list = filter_list, zp_list = zero_list, ebv = ebv, ) # gen mock data and write lyric file.

    for idx in sorted_idx_group:

        mock_subdir = os.path.join(gs_pureSed_dir,f"{idx+1}")
        lyric_file = os.path.join(mock_subdir, f"obj{idx+1}_pureSed.lyric")
        workplace = os.path.join(mock_subdir, "results")
        command = f"CUDA_VISIBLE_DEVICES=7 python {galfitS_path} --config \
        {lyric_file} --workplace {workplace} --fit_method ES --num_generations 10000"

        os.system(command)

    # ========================================================================
    # Part 5: Image+SED Fitting
    # ========================================================================
    z_cat = Table.read(z_out_path, format = "ascii")

    for idx in sorted_idx_group:

        image_stamp_list = []
        mask_file_list = []

        for i in range(nband):
            # claim input material for galfits config file
            image_stamp_list.append(os.path.join(cutout_dir, f"obj{idx+1}_{label_list[i]}sci.fits"))
            mask_file_list.append(os.path.join(cutout_dir, f"obj{idx+1}_{label_list[i]}mask.fits"))

        mask_file_list = np.array(mask_file_list)
        image_stamp_list = np.array(image_stamp_list)
        lyric_path = os.path.join(gs_image_sed_dir, f"obj_{idx+1}.lyric")
        prior_path = os.path.join(gs_image_sed_dir, f"obj_{idx+1}_sfh.prior")
        objects = object_dict[str(idx)]

        z_list = []
        for obj in objects:
            index = obj+1
            z_idx = np.where(z_cat["id"] == index)[0][0]
            if z_cat[z_idx]["z_peak"]<0:
               z_list.append(0.001) 
            else:    
               z_list.append(z_cat[z_idx]["z_peak"])

        psf_file_used = process_psf(idx+1, nband, psf_list, image_stamp_list, label_list, processed_psf_dir) # cut psf file to suitable size
       
        prepare_galfits(lyric_path = lyric_path, cat_file = catalog_name, objects = objects,prior_path = prior_path,
                        det_label = det_label, sci_list = image_stamp_list, psf_list = psf_file_used,
                        zero_list = zero_list, pixscl_list = pixel_scale_list,
                        label_list = label_list, filter_list=filter_list,
                        geo_smdir = gs_pureImage_dir, pSed_smdir = gs_pureSed_dir, imgSed_smdir = gs_image_sed_dir,
                        mask_list = mask_file_list,SPS_catalog_path = SPS_catalog_path,sfhs_path=sfhs_path, 
                        z_list = z_list, use_sed = 1, convf = False,use_sfh_prior=True)

        command = f"python {galfitS_path} --config \
        {lyric_path} --workplace {gs_image_sed_dir} --fit_method ES --num_generations 10000 --prior {prior_path}"

        os.system(command)

    return f"Group {i_group} completed on GPU {gpu_id}"


# =============================================================================
# Main Program
# =============================================================================

begin_time=datetime.datetime.now()

# multi-band input science images
sciname_list = np.array(['F444W_sci.fits',
                         'F356W_sci.fits',
                         'F277W_sci.fits',
                         'F200W_sci.fits',
                         'F150W_sci.fits',
                         'F115W_sci.fits'])

# corresponding weight images
whtname_list = np.array(['F444W_wht.fits',
                        'F356W_wht.fits',
                        'F277W_wht.fits',
                        'F200W_wht.fits',
                        'F150W_wht.fits',
                        'F115W_wht.fits'])

# directory to store image cutout for each band
banddir_list = np.array(['./F444W/',
                        './F356W/',
                        './F277W/',
                        './F200W/',
                        './F150W/',
                        './F115W'])

# band label for image cutout
label_list = np.array(['F444W',
                    'F356W',
                    'F277W',
                    'F200W',
                    'F150W',
                    'F115W'])

# specific label for each band (used for galfits), do not change it
filter_list = np.array(['nircam_f444w',
                        'nircam_f356w',
                        'nircam_f277w',
                        'nircam_f200w',
                        'nircam_f150w',
                        'nircam_f115w'])

# input psf files
psf_list = np.array(['./psfs/f444w_psf.fits',
                     './psfs/f356w_psf.fits',
                     './psfs/f277w_psf.fits',
                     './psfs/f200w_psf.fits',
                     './psfs/f150w_psf.fits',
                     './psfs/f115w_psf.fits'])

# mag zeropoints
zero_list = np.array([27.462, 27.462, 27.462, 28.967,28.967, 28.967])
expt_list = np.array([2834.5, 3092.18, 3092.18, 2834.5, 3092.18, 6184.38]) # exptime
pixel_scale_list = np.array([0.04, 0.04, 0.04, 0.02, 0.02, 0.02])
mjsr_list = np.array([0.4, 0.42, 0.49, 1.96, 2.33, 3.1]) # photmjsr parameter in the header
gain_list = np.array([1.8, 1.8, 1.8, 1.8, 1.8, 1.8])


covermask_list = np.array([sciname.replace("sci.fits", "cover_mask.fits") for sciname in sciname_list])
nband = len(sciname_list)


detname = sciname_list[0] # detection(reference) band, i.e, F444W
det_label = label_list[0]
weight_name = whtname_list[0] # weight map of detection band

# smoothing kernel for source detection
kernel = np.loadtxt('gauss_4.0_7x7.conv', skiprows=1)
SEx_dir = "./SEx/"
os.makedirs(SEx_dir, exist_ok=True)
ref_pixel_scale = pixel_scale_list[0] # the pixel scale of detection band
det_segname = "HDR_outseg.fits" # combined segmap, it will be created below.
stampfile = './stamps.txt'
nnw_sex = 'default.nnw'
fwhm_arcsec = 0.16

## store cutouts
cutout_dir = "./cutout" ## store mask
os.makedirs(cutout_dir, exist_ok=True)

# cold+hot detection, all the output results will be saved into './sex/'
outtab, outsegm=SExtractor_HDR(detname, kernel=(kernel, kernel),path = SEx_dir ,
                segmap_name = ("cold_seg.fits", "hot_seg.fits", det_segname), catalog_name = ("cold.cat", "hot.cat", "HDR_out.cat"),
                detect_minarea=(12, 15), detect_thresh=(3, 1.8), deblend=(True, True),
                deblend_nthresh=(32, 64), deblend_mincont=(0.01, 0.001), clean=(True, True), back_type=(True, True), back_value=(0., 0.), back_size=(128, 32),
                back_filtersize=(3, 3), weight_type='MAP_WEIGHT', weight_name=weight_name, scale_factor=1.1, pixel_scale=ref_pixel_scale, nnw_sex=nnw_sex,
                fwhm_arcsec=fwhm_arcsec, mag_zeropoint=zero_list[0], verbose=True)

det_img = fits.getdata(detname)
coverage_mask = np.isnan(det_img) # coverage mask for detect img.                                                       
coverage_header = fits.getheader(detname, 0)                                                                            
coverage_mask_output = f"./{det_label}_cover_mask_0.fits"                                                               
fits.writeto(filename = coverage_mask_output, data = coverage_mask.astype(np.int8), header = coverage_header, overwrite = True)



catalog_name = os.path.join(SEx_dir, 'HDR_out.cat') # combined catalog
outtab = ascii.read(catalog_name)

# create stampfile, which will be used for cutting image
create_stamp_file(detname, catalog_name, sizefac=2.5, outfile=stampfile, pixel_scale=ref_pixel_scale)


# cut image for all bands
for i in range(nband):
    sciname = sciname_list[i]
    label = label_list[i]
    cut_stamps(sciname, cutout_dir, label = label, stampfile=stampfile, ps=pixel_scale_list[i])  # set ps value


## create segm images if the input image has different pixel scales from your reference image.
## The input are seg images (has the same pixel scale of ref band) and sci image.
det_seg_fullpath = os.path.join(SEx_dir, det_segname)

seglist = np.array([sciname.replace(".fits", "_seg.fits") for sciname in sciname_list])

for sciname, seg_name, cover_mask_name in zip(sciname_list, seglist, covermask_list):

    reproject_segm(det_seg_fullpath, sciname, output = seg_name, )
    reproject_segm(coverage_mask_output, sciname, output = cover_mask_name, type = "mask")




# sort magnitdue from bright to faint
sorted_idx = np.argsort(outtab['mag_auto'])

gsdir='./galfits/'
os.makedirs(gsdir, exist_ok=True)

gs_pureImage_dir = os.path.join(gsdir, "pureImage")
gs_pureSed_dir = os.path.join(gsdir, "pureSed")
gs_image_sed_dir =  os.path.join(gsdir, "image_sed")

os.makedirs(gsdir, exist_ok=True)
os.makedirs(gs_pureImage_dir, exist_ok = True)
os.makedirs(gs_pureSed_dir, exist_ok=True)
os.makedirs(gs_image_sed_dir, exist_ok=True)

object_dict = {} ## key is 0-indexed
object_path = "./objects.json"

# create mask file and objects.json
for i in range(nband):
 hdusci = fits.open(sciname_list[i])[0]
 
 cover_data = fits.getdata(covermask_list[i])
 segdata = fits.getdata(seglist[i])

 print(sciname_list[i]+ ' create mask start')
 for idx in sorted_idx:
    hduin = hdusci.copy()
    key = str(idx)
    # create multi-band mask stamp
    label = label_list[i]
    maskfile = os.path.join(cutout_dir, f"obj{idx+1}_{label}mask.fits")
    corner, objects = create_mask(scihdu = hduin,
                                  seg_data = segdata,
                                  cover_data = cover_data,
                                  catalog_name = catalog_name,
                                  paramfile = stampfile,
                                  mask_file = maskfile,
                                  current = idx,
                                  ps=pixel_scale_list[i])
        
    obj_stored = [int(obj) for obj in objects]
    object_dict[key] = obj_stored

print('start objects dump')
with open(object_path, "w") as f:
    json.dump(object_dict, f, indent = 4)
print('objects dump ends')

with open(object_path, "r") as f:
    object_dict = json.load(f)

print('grouping starts')
groups = Union_Set(object_dict)
print('grouping ends')
## define path issues


psf_file = psf_list[0] # reference psf (F444W)
try_filterlist = [0]
# define apertures to measure noise
apers_list = np.array([5, 10, 15, 20, 25, 30, 35, 40]) * ref_pixel_scale
skynoisepath = "./sky_noise/"
gmoutdir = "./galfitm/"
os.makedirs(gmoutdir, exist_ok=True)
out_gs_flux_catfile = os.path.join(gmoutdir,"gs_iso_flux.cat")
out_gs_flux_error_catfile ="./gsflux_isoerr.cat"


processed_psf_dir = os.path.join(cutout_dir, "processed_psf") # where to store the processed psfs.
os.makedirs(processed_psf_dir, exist_ok=True)

# perfer to use public path in server 206
if os.path.exists(public_galfit_path):
    galfit_path = public_galfit_path
else:
    # fallback to dynamic lookup
    galfit_path = shutil.which('galfitm')
    if galfit_path is None:
        raise FileNotFoundError("No galfitm found: neither public path nor in PATH.")

#galfit_path = shutil.which('galfitm')
#if galfit_path is None:
#    raise FileNotFoundError("no galfitm path found.")
kron_scale = 1.5 # the factor to enlarge kron radius to make mask
specz_cat = None
max_sep = 0



external_mask_file_list=[None,]*7
## initial config for iso flux measurement.
iso_config = PhotometryConfig(
            galfit_path = galfit_path, sky_noise_path = skynoisepath, gmoutdir = gmoutdir,
            label_list = label_list, filter_labels = filter_list, wht_file_list = whtname_list,
            galaxy_catalog = catalog_name, image_list = sciname_list, kron_scale = kron_scale,
            save_mask = True,apertures_list = apers_list, pixel_scales = pixel_scale_list, save_regions = True,
            plot_histograms = True, psf_file = psf_file, segmentation_map_list=seglist, cutout_dir=cutout_dir, gains = gain_list,
            exptimes=expt_list, mjsr_list=mjsr_list, zero_list=zero_list, try_filterlist = try_filterlist, ref_pixel_scale = 0.04,
            stampfile = stampfile,detwht_file=weight_name,external_mask_file_list=external_mask_file_list,
            )
iso_pipeline = PhotometryPipeline(iso_config)
iso_pipeline.step1_determine_background_positions() ## pre
bg_params=iso_pipeline.step2_fit_background_noise()

# =============================================================================
# Multi-GPU Parallel Processing
# =============================================================================

# Number of GPUs available for parallel processing
num_gpus = 8  # Adjust based on your system

# Convert outtab (astropy Table) and bg_params (DataFrame) to dict for pickling
outtab_dict = {col: outtab[col].tolist() for col in outtab.colnames}
bg_params_dict = bg_params.to_dict('list')

# EAZY path
eazy_path = '/home/zps/eazy-photoz-master'

print(f"\n{'='*60}")
print(f"Starting parallel processing with {num_gpus} GPUs")
print(f"Total groups: {len(groups)}")
print(f"{'='*60}\n")

# Process groups in parallel using ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=num_gpus) as executor:
    futures = []
    for i_group, group in enumerate(groups):
        
        future = executor.submit(
            process_group,
            i_group=i_group,
            group=group,
            object_dict=object_dict,
            outtab_dict=outtab_dict,
            bg_params_dict=bg_params_dict,
            nband=nband,
            sciname_list=sciname_list,
            label_list=label_list,
            filter_list=filter_list,
            psf_list=psf_list,
            zero_list=zero_list,
            pixel_scale_list=pixel_scale_list,
            expt_list=expt_list,
            mjsr_list=mjsr_list,
            gain_list=gain_list,
            det_label=det_label,
            catalog_name=catalog_name,
            cutout_dir=cutout_dir,
            gs_pureImage_dir=gs_pureImage_dir,
            gs_pureSed_dir=gs_pureSed_dir,
            gs_image_sed_dir=gs_image_sed_dir,
            gmoutdir=gmoutdir,
            skynoisepath=skynoisepath,
            stampfile=stampfile,
            psf_file=psf_file,
            try_filterlist=try_filterlist,
            apers_list=apers_list,
            kron_scale=kron_scale,
            ref_pixel_scale=ref_pixel_scale,
            eazy_path=eazy_path,
            num_gpus=num_gpus,
            whtname_list=whtname_list,
            weight_name=weight_name,
        )
        futures.append(future)

    # Wait for all tasks to complete and show progress
    print("Submitted all groups to workers. Waiting for completion...\n")
    for i, future in enumerate(futures):
        result = future.result()
        print(f"[{i+1}/{len(futures)}] {result}")

print(f"\n{'='*60}")
print(f"All processing completed!")
print(f"Total time: {datetime.datetime.now() - begin_time}")
print(f"{'='*60}\n")
