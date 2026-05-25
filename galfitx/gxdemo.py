import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
galfitS_path = "/home/zhongyi/Softwares/GalfitS_fork/src/galfits/galfitS.py" # public path env in 206
eazy_path = '/home/zhongyi/Softwares/eazy-photoz'
public_galfit_path = "/home/zhongyi/Softwares/galfitm"  # public path in server 206
template_dir = "/home/zhongyi/Softwares/GalfitX_fork/demo_group/templates"
SPS_catalog_path = os.path.join(template_dir, "UNCOVER_DR4_SPS_catalog.fits")
sfhs_path = os.path.join(template_dir, "sfhs_SPS_DR4.npz")
# perfer to use public path in server 206
if os.path.exists(public_galfit_path):
    galfit_path = public_galfit_path
else:
    # fallback to dynamic lookup
    galfit_path = shutil.which('galfitm')
    if galfit_path is None:
        raise FileNotFoundError("No galfitm found: neither public path nor in PATH.")

import sys
import numpy as np
import shutil
from astropy.io import ascii, fits
from astropy.table import Table
from galfitx.source_detection import SExtractor_HDR
from galfitx.postage_stamp import create_stamp_file, cut_stamps
from galfitx.create_setup_gs import prepare_galfits, create_mask, process_psf, reproject_segm, gen_pSed_data_lyric
from galfitx.gx_gsutils import Union_Set
from galfitx.model_isoflux import PhotometryConfig, PhotometryPipeline
from galfitx.eazy_utils import zphot_config, translate_config, run_eazy, show_all_fitting
import json
from galfitx.utils import combine_catalogs
import datetime

begin_time=datetime.datetime.now()

# multi-band input science images
sciname_list = np.array(['F444W_sci.fits', 'F356W_sci.fits', 'F277W_sci.fits', 'F200W_sci.fits', 'F150W_sci.fits', 'F115W_sci.fits'])
covermask_list = np.array([sciname.replace("sci.fits", "cover_mask.fits") for sciname in sciname_list])
seglist = np.array([sciname.replace(".fits", "_seg.fits") for sciname in sciname_list])

# corresponding weight images
whtname_list = np.array(['F444W_wht.fits', 'F356W_wht.fits', 'F277W_wht.fits', 'F200W_wht.fits', 'F150W_wht.fits', 'F115W_wht.fits'])

# directory to store image cutout for each band
banddir_list = np.array(['./F444W/', './F356W/', './F277W/', './F200W/', './F150W/', './F115W/'])

# band label for image cutout
label_list = np.array(['F444W', 'F356W', 'F277W', 'F200W', 'F150W', 'F115W'])

# specific label for each band (used for galfits), do not change it
filter_list = np.array(['nircam_f444w', 'nircam_f356w', 'nircam_f277w', 'nircam_f200w', 'nircam_f150w', 'nircam_f115w'])

# input psf files
psf_list = np.array(['./psfs/f444w_psf.fits', './psfs/f356w_psf.fits', './psfs/f277w_psf.fits', './psfs/f200w_psf.fits', './psfs/f150w_psf.fits', './psfs/f115w_psf.fits'])

# mag zeropoints
zero_list = np.array([27.462, 27.462, 27.462, 28.967, 28.967, 28.967])
expt_list = np.array([2834.5, 3092.18, 3092.18, 2834.5, 3092.18, 6184.38]) # exptime
pixel_scale_list = np.array([0.04, 0.04, 0.04, 0.02, 0.02, 0.02])
mjsr_list = np.array([0.4, 0.42, 0.49, 1.96, 2.33, 3.1]) # photmjsr parameter in the header
gain_list = np.array([1.8, 1.8, 1.8, 1.8, 1.8, 1.8])


nband = len(sciname_list)
detname = sciname_list[0] # detection(reference) band, i.e, F444W
det_label = label_list[0]
weight_name = whtname_list[0] # weight map of detection band
# smoothing kernel for source detection
kernel = np.loadtxt('gauss_4.0_7x7.conv', skiprows=1)
SEx_dir = "./SEx/"
os.makedirs(SEx_dir, exist_ok=True)
ref_pixel_scale = pixel_scale_list[0] # the pixel scale of detection band
det_segname = 'outseg.fits' # combined segmap, it will be created below.
stampfile = './stamps.txt'
nnw_sex = 'default.nnw'
fwhm_arcsec = 0.16

# store cutouts
cutout_dir = "./cutout" # store mask
os.makedirs(cutout_dir, exist_ok=True)
processed_psf_dir = os.path.join(cutout_dir, "processed_psf") # where to store the processed psfs.
os.makedirs(processed_psf_dir, exist_ok=True)

# cold+hot detection, all the output results will be saved into './sex/'
outtab, outsegm=SExtractor_HDR(detname,
                               path = SEx_dir,
                               kernel=(kernel, kernel),
                               detect_minarea=(25, 15),
                               detect_thresh=(3, 1.8),
                               deblend=(True, True),
                               deblend_nthresh=(32, 64),
                               deblend_mincont=(0.01, 0.001),
                               clean=(True, True),
                               back_type=(True, True),
                               back_value=(0., 0.),
                               back_size=(128, 32),
                               back_filtersize=(3, 3),
                               weight_type='MAP_WEIGHT',
                               weight_name=weight_name,
                               scale_factor=0.8,
                               pixel_scale=ref_pixel_scale,
                               nnw_sex=nnw_sex,
                               fwhm_arcsec=fwhm_arcsec,
                               mag_zeropoint=zero_list[0],
                               verbose=True)
det_img = fits.getdata(detname, 0)
coverage_mask = np.isnan(det_img) # coverage mask for detect img.
coverage_header = fits.getheader(detname, 0)
coverage_mask_output = f"./{det_label}_cover_mask_0.fits"
fits.writeto(filename = coverage_mask_output, data = coverage_mask.astype(np.int8), header = coverage_header, overwrite = True)

catalog_name = os.path.join(SEx_dir, 'outcat') # combined catalog
outtab = ascii.read(catalog_name)

# create stampfile, which will be used for cutting image
create_stamp_file(detname, catalog_name, sizefac=2.5, outfile=stampfile, pixel_scale=ref_pixel_scale)

# cut image for all bands
for i in range(nband):
    sciname = sciname_list[i]
    label = label_list[i]
    cut_stamps(sciname, cutout_dir, label=label, stampfile=stampfile, ps=pixel_scale_list[i])  # set ps value

## create segm images if the input image has different pixel scales from your reference image.
## The input are seg images (has the same pixel scale of ref band) and sci image.
det_seg_fullpath = os.path.join(SEx_dir, det_segname)

for sciname, seg_name, cover_mask_name in zip(sciname_list, seglist, covermask_list):

    reproject_segm(det_seg_fullpath, sciname, output = seg_name, )
    reproject_segm(coverage_mask_output, sciname, output = cover_mask_name, type = "mask")


# sort magnitdue from bright to faint
sorted_idx = np.argsort(outtab['mag_auto'])

gsdir='./galfits/'
os.makedirs(gsdir, exist_ok=True)

gs_pureImage_dir = os.path.join(gsdir, "pureImage")
gs_pureSed_dir = os.path.join(gsdir, "pureSed")
gs_image_sed_dir = os.path.join(gsdir, "image_sed")

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

with open(object_path, "w") as f:
    json.dump(object_dict, f, indent = 4)
    
with open(object_path, "r") as f:
    object_dict = json.load(f)
    
groups = Union_Set(object_dict)

psf_file = psf_list[0] # reference psf (F444W)
try_filterlist = [0]
# define apertures to measure noise
apers_list = np.array([5, 10, 15, 20, 25, 30, 35, 40]) * ref_pixel_scale
skynoisepath = "./sky_noise/"
gmoutdir = "./galfitm/"
os.makedirs(gmoutdir, exist_ok=True)
out_gs_flux_catfile = os.path.join(gmoutdir,"gs_iso_flux.cat")
out_gs_flux_error_catfile ="./gsflux_isoerr.cat"


external_mask_file_list = ['./sky_noise/nircam_f444w_kron_scale1.5_integrated_mask.fits', None, None, None, None, None]
kron_scale = 1.5 # the factor to enlarge kron radius to make mask
specz_cat = None
max_sep = 0.15

# initial config for iso flux measurement.
iso_config = PhotometryConfig(# image and catalog paths
                              image_list = sciname_list,
                              galaxy_catalog = catalog_name,
                              psf_file = psf_file,
                              segmentation_map_list=seglist,
                              cutout_dir=cutout_dir,
                              
                              # Photometry parameters
                              label_list = label_list,
                              filter_labels = filter_list,
                              gains = gain_list,
                              exptimes=expt_list,
                              mjsr_list=mjsr_list,
                              zero_list=zero_list,
                              
                              # Aperture settings
                              apertures_list = apers_list,
                              pixel_scales = pixel_scale_list,
                              ref_pixel_scale = ref_pixel_scale,
                              # Specz settings
                              
                              # GALFIT settings
                              galfit_path = galfit_path,
                              stampfile = stampfile,
                              try_filterlist = try_filterlist,
                              
                              # Output settings
                              sky_noise_path = skynoisepath,
                              gmoutdir = gmoutdir,
                              
                              # Processing options
                              save_mask = True,
                              save_regions = True,
                              plot_histograms = True,
                              
                              # mask related
                              kron_scale = kron_scale,
                              detwht_file = weight_name,
                              wht_file_list = whtname_list,
                              external_mask_file_list = external_mask_file_list,
                              )

iso_pipeline = PhotometryPipeline(iso_config)
iso_pipeline.step1_determine_background_positions() # pre
bg_params=iso_pipeline.step2_fit_background_noise()

for i_group, group in enumerate(groups):

    sorted_idx_group = sorted(group, key = lambda x: outtab["mag_auto"][x])
    sorted_idx_group_1idx = [value + 1 for value in sorted_idx_group]

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
        
        psf_file_used = process_psf(idx+1, nband, psf_list, image_stamp_list, label_list, processed_psf_dir) # cut psf file to suitable size
        lyric_path = os.path.join(gs_pureImage_dir, f"obj{idx+1}.lyric") # output lyric file.
        prior_path = os.path.join(gs_pureImage_dir, f"obj{idx+1}_sfh.prior")

        prepare_galfits(lyric_path = lyric_path, prior_path = prior_path, cat_file = catalog_name, objects = objects,
                        det_label = det_label, sci_list = image_stamp_list, psf_list = psf_file_used,
                        zero_list = zero_list, pixscl_list = pixel_scale_list,
                        label_list = label_list, filter_list=filter_list,
                        geo_smdir = gs_pureImage_dir, pSed_smdir = gs_pureSed_dir, imgSed_smdir = gs_image_sed_dir,
                        mask_list = mask_file_list, use_sed = 0, convf = False)

        command = f"CUDA_VISIBLE_DEVICES=7 python {galfitS_path} --config \
        {lyric_path} --workplace {gs_pureImage_dir} --fit_method ES --num_generations 10000"
        
        os.system(command)


    # below is to derive model ISO(segment) flux and errors for each object,
    # which will be used as input for deriving photz
    # This important function will generate out_gs_flux_error_catfile,
    # which contain multi-band ISO flux and error for each object.

    gs_flux_outfile = os.path.join(gmoutdir, f"gs_iso_flux_group{i_group}.cat")
    gs_flux_err_outfile = os.path.join(gmoutdir, f"gs_iso_flux_err_group{i_group}.cat")
    iso_pipeline._compute_model_fluxes_for_galaxies(galids = sorted_idx_group_1idx,
                                                    filter_list = filter_list,
                                                    image_list = sciname_list,
                                                    gsdir = gs_pureImage_dir,
                                                    outdir = gmoutdir,
                                                    output_file = gs_flux_outfile)
    iso_pipeline.step4_compute_isophotal_errors(bg_df = bg_params,
                                                flux_df_path  = gs_flux_outfile,
                                                gs_flux_err_outfile = gs_flux_err_outfile)



    # please check the best error setting for your photometry catalog
    # suggestions: temperr in [0.00, 0.20]; syserr in [0.00,0.20]
    # the additinal template error will included in fitting
    temperr = 0.03
    # the additinal systematic error will added in fitting
    syserr = 0.03
    eazy_out_path = f"./eazy/{i_group}"
    config_file_param = os.path.join(eazy_out_path, "zphot.param")
    os.makedirs(eazy_out_path, exist_ok=True)
    # generate the config for EAZY
    zphot_config(gs_flux_err_outfile, eazy_out_path, temperr, syserr, eazypath=eazy_path, configfile = config_file_param)
    # generate the translate config bewteen photometry catalog and EAZY
    config_file_translate = os.path.join(eazy_out_path, "zphot.translate")
    translate_config(gs_flux_err_outfile, configfile = config_file_translate)

    # User can use this to run eazy,
    # or modify eazy config in terminal for more parameters and run eazy in terminal with command "eazy_path/src/eazy -t zphot.translate"
    run_eazy(eazypath=eazy_path, configfile = config_file_param, translatefile=config_file_translate)

    # show fitting results of EAZY, including photz-specz comparison figure and sed-fitting, pdf, and galfits fitting figures
    figure_output = os.path.join(eazy_out_path, "figures")
    os.makedirs(figure_output, exist_ok=True)
    # show_all_fitting(eazy_out_path, gs_pureImage_dir, output_path = figure_output)

    # Pure sed fitting
    ebv = None ## will automately search using package.

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

    # Image + sed fitting
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
        lyric_path = os.path.join(gs_image_sed_dir, f"obj{idx+1}.lyric")
        prior_path = os.path.join(gs_image_sed_dir, f"obj{idx+1}_sfh.prior")
        objects = object_dict[str(idx)]

        z_list = []
        
        for obj in objects:
            index = obj+1

            z_idx = np.where(z_cat["id"] == index)[0][0]
            z_curr = z_cat[z_idx]["z_peak"]

            if z_curr < 0:
                z_curr = 0.001

            z_list.append(z_curr)

        psf_file_used = process_psf(idx+1, nband, psf_list, image_stamp_list, label_list, processed_psf_dir)

        prepare_galfits(lyric_path = lyric_path, prior_path = prior_path, cat_file = catalog_name, objects = objects,
                        det_label = det_label, sci_list = image_stamp_list, psf_list = psf_file_used,
                        zero_list = zero_list, pixscl_list = pixel_scale_list,
                        label_list = label_list, filter_list=filter_list,
                        geo_smdir = gs_pureImage_dir, pSed_smdir = gs_pureSed_dir, imgSed_smdir = gs_image_sed_dir,
                        SPS_catalog_path = SPS_catalog_path, sfhs_path = sfhs_path, 
                        mask_list = mask_file_list,
                        z_list = z_list, use_sed = 1, use_sfh_prior = True, convf = False, ebv = ebv)

        command = f"CUDA_VISIBLE_DEVICES=7 python {galfitS_path} --config \
        {lyric_path} --workplace {gs_image_sed_dir} --fit_method ES --num_generations 10000 --prior {prior_path}"

        os.system(command)

# combine all catalog into a fits file
combine_catalogs(filter_list, catalog_name, gs_pureImage_dir, './eazy', gs_image_sed_dir, output_file='combined_catalog.fits')

print (datetime.datetime.now() - begin_time)
