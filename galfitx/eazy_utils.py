"""
EAZY Photometric Redshift Analysis Package
=========================================

A comprehensive package for running and analyzing photometric redshifts using EAZY.

Modules:
- Eazybox: Core functionality for configuring and running EAZY
- run_eazy: Main execution functions for running EAZY
- EAzy_analysis: Analysis and visualization of EAZY outputs

Main Features:
1. Configuration generation for EAZY runs
2. Execution of EAZY photometric redshift code
3. Analysis and visualization of results including:
   - Redshift comparison plots
   - SED fitting visualization
   - Probability distribution functions
   - Statistical analysis of results

Reference: 2023 CSST summer school: eazy_toolbox.py from Lipin Fu's Group.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.table import Table
import matplotlib.image as mpimg
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator

def run_eazy(eazypath='/Path/to/eazy',configfile='zphot.param',translatefile='zphot.translate',zeropointfile='zphot.zeropoint'):

    """
    Execute EAZY photometric redshift code with specified parameters.
    
    Parameters
    ----------
    eazypath : str, optional
        Path to the Eazy executable directory (default: '/Path/to/eazy')
    configfile : str, optional
        Name of configuration file to generate (default: 'zphot.param')
    translatefile : str, optional
        Filter translation file (default: 'zphot.translate')
    zeropointfile : str, optional
        Zeropoint file for calibration (default: 'zphot.zeropoint')
    """
    
    if not os.path.exists(eazypath+'/src/eazy'): #??? redundant '/'
        print(('Eazy executable not found in path: %s' %(eazypath)))#????
        return 0

    if not os.path.exists(configfile):
        print(('Configuration file, %s, not found.' %(configfile)))
        print('Please generate a configuration file using zphot_config function.')
        return 0
    
    if not os.path.exists(translatefile):
        print(('Filter translation file, %s, not found.' %(translatefile)))
        print('Please generate a filter translation file using zphot_config function.')
        return 0
    
    #if not os.path.exists(zeropointfile):
    #    print(('Zeropoint file, %s, not found.' %(zeropointfile)))
    #    return 0
    
    #cmd = eazypath+'/src/eazy -p '+configfile+' -t '+translatefile+' -z '+zeropointfile
    cmd = eazypath+'/src/eazy -p '+configfile+' -t '+translatefile
    #cmd = f'eazy -p {configfile}' # not work
    os.system(cmd)


def zphot_config(catfile, 
                 outdir,
                 temperr,
                 syserr,
                 eazypath='/Path/to/eazy',
                 template_file='templates/fsps_full/tweak_fsps_QSF_12_v3.param',
                 template_combos = 99,
                 zmax=12.000,
                 prior=0,
                 prior_filter='205',
                 prior_file = 'templates/prior_F160W_TAO.dat',
                 zp_offsets=0,
                 fixspecz=0,
                 not_obs_threshold = -90.000,
                 configfile='./zphot.param'):
    """
    Generate EAZY configuration file with specified parameters.
    
    Creates a congfig file that controls all aspects of the EAZY run including:
    - Filter definitions and properties
    - Template sets and fitting options
    - Input/output file specifications
    - Redshift grid parameters
    - Priors and zeropoint corrections
    
    Parameters
    ----------
    catfile : str
        Path to input photometric catalog file
    outdir : str
        Output directory for results
    temperr : float
        Template error amplitude (typically 0.01-0.1)
    syserr : float
        Systematic flux error as fraction of flux (typically 0.01-0.1)
    template_file : str, optional
        Template definition file (default: 'templates/tweak_fsps_QSF_12_v3.param')
    template_combos : int, optional
        Template combination method (1=single, 2=double, 99=full set) (default: 99)
    zmax : float, optional
        Maximum redshift (default: 12.0)
    prior : int, optional
        Apply magnitude prior (0=No, 1=Yes) (default: 0)
    prior_filter : str, optional
        Filter for prior calculation (default: '205' for WFC3 F160W)
    prior_file : str, optional
        Magnitude prior definition file (default: 'templates/prior_F160W_TAO.dat')
    zp_offsets : int, optional
        Compute zeropoint offsets (0=No, 1=Yes) (default: 0)
    fixspecz : int, optional
        Fix redshift to spectroscopic value (0=No, 1=Yes) (default: 0)
    not_obs_threshold : float, optional
        Threshold for non-observed fluxes (default: -90.0)
    configfile : str, optional
        Name of configuration file to create (default: './zphot.param')
        
    Configuration Parameters
    ------------------------
    FILTERS_RES : str
        Filter transmission curves file
    TEMPLATES_FILE : str
        Template definition file
    TEMPLATE_COMBOS : int
        Template combination method (99 = full set)
    TEMP_ERR_A2 : float
        Template error amplitude
    SYS_ERR : float
        Systematic flux error
    APPLY_IGM : int
        Apply IGM absorption (1=Yes)
    CATALOG_FILE : str
        Input catalog file
    OUTPUT_DIRECTORY : str
        Directory for output files
    MAIN_OUTPUT_FILE : str
        Base name for output files
    APPLY_PRIOR : int
        Apply magnitude prior
    FIX_ZSPEC : int
        Fix to spectroscopic redshift
    Z_MIN/Z_MAX/Z_STEP : float
        Redshift grid parameters
    GET_ZP_OFFSETS : int
        Compute zeropoint offsets
        
    Notes
    -----
    The function generates a complete EAZY configuration file with sensible defaults
    for most parameters. Users typically only need to specify the required parameters
    (catfile, outdir, temperr, syserr) and optionally adjust the commonly changed
    parameters (zmax, prior, etc.).
    
    Examples
    --------
    >>> # Basic usage with minimum required parameters
    >>> zphot_config('catalog.cat', 'output/', 0.03, 0.03)
    
    >>> # With custom redshift range and priors
    >>> zphot_config('catalog.cat', 'output/', 0.03, 0.03, 
    ...              zmax=6.0, prior=1, prior_filter='236')
    """

    # Default configuration parameters
    filter_res = eazypath+'/filters/FILTER.RES.latest'    # Filter transmission data file
    n_min_colors = '4'                  # Minimum number of bands required for fit
    zp_ab = 16.4                        # AB zeropoint for magnitude calculations
    zmin = 0.01                         # Minimum redshift
    zstep = 0.005                       # Redshift step size
    zstep_type = '1'                    # 0=linear, 1=relative (1+z) steps
    
    if not os.path.exists('./templates'):
        os.system('cp -r '+eazypath+'/templates ./')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    # Create and write configuration file
    zphot_file = open(configfile, 'w')

    # Header section
    zphot_file.write('################   Run parameters (can feed this file back to EAZY)  #################### \n')

    # Filter configuration
    zphot_file.write('## Filters \n')
    zphot_file.write('FILTERS_RES          '+filter_res+'  # Filter transmission data \n')
    zphot_file.write('FILTER_FORMAT        1                  # Format of FILTERS_RES file -- 0: energy-  1: photon-counting detector \n')
    zphot_file.write('SMOOTH_FILTERS       0                  # Smooth filter curves with Gaussian \n')
    zphot_file.write('SMOOTH_SIGMA         50.00              # Gaussian sigma (in Angstroms) to smooth filters \n')

    # Template configuration
    zphot_file.write('\n## Templates \n')
    zphot_file.write('TEMPLATES_FILE       '+template_file+' # Template definition file \n')
    zphot_file.write('TEMPLATE_COMBOS      '+str(template_combos)+'                 # Template combination options:  \n')
    zphot_file.write('NMF_TOLERANCE        1.00e-04           # Tolerance for non-negative combinations (TEMPLATE_COMBOS=a) \n')
    zphot_file.write('WAVELENGTH_FILE      templates/uvista_nmf/lambda.def # Wavelength grid definition file \n')
    zphot_file.write('TEMP_ERR_FILE        templates/uvista_nmf/template_error_10.def # Template error definition file \n')
    zphot_file.write('TEMP_ERR_A2          '+str(temperr)+'              # Template error amplitude \n')
    zphot_file.write('SYS_ERR              '+str(syserr)+'              # Systematic flux error (% of flux) \n')
    zphot_file.write('APPLY_IGM            1                  # Apply IGM absorption (1/y=Inoue2014, 2/x=Madau1995) \n')
    zphot_file.write('LAF_FILE             templates/LAFcoeff.txt # File containing the Lyman alpha forest data from Inoue(2014) \n')
    zphot_file.write('DLA_FILE             templates/DLAcoeff.txt # File containing the damped Lyman absorber data from Inoue(2014) \n')
    zphot_file.write('SCALE_2175_BUMP      0.000              # Scaling of 2175A bump.  Values 0.13 (0.27) absorb ~10 (20) % at peak. \n')
    zphot_file.write('DUMP_TEMPLATE_CACHE  0                  # Write binary template cache \n')
    zphot_file.write('USE_TEMPLATE_CACHE   0                  # Load in template cache \n')
    zphot_file.write('CACHE_FILE           photz.tempfilt     # Template cache file (in OUTPUT_DIRECTORY) \n')
    
    # Input file configuration
    zphot_file.write('\n## Input Files \n')
    zphot_file.write('CATALOG_FILE         '+catfile+'       # Catalog data file \n')
    zphot_file.write('MAGNITUDES           0                  # Catalog photometry in magnitudes rather than f_nu fluxes \n')
    zphot_file.write('NOT_OBS_THRESHOLD    '+str(not_obs_threshold)+'            # Ignore flux point if <NOT_OBS_THRESH \n')
    zphot_file.write('N_MIN_COLORS         '+n_min_colors+'                  # Require N_MIN_COLORS to fit \n')
    
     # Output file configuration
    zphot_file.write('\n## Output Files \n')
    zphot_file.write('OUTPUT_DIRECTORY     '+outdir+'       # Directory to put output files in \n')
    zphot_file.write('MAIN_OUTPUT_FILE     photz              # Main output file, .zout \n')
    zphot_file.write('PRINT_ERRORS         1                  # Print 68, 95 and 99% confidence intervals \n')
    zphot_file.write('CHI2_SCALE           1.000              # Scale ML Chi-squared values to improve confidence intervals \n')
    zphot_file.write('VERBOSE_LOG          1                  # Dump information from the run into [MAIN_OUTPUT_FILE].param \n')
    zphot_file.write('OBS_SED_FILE         0                  # Write out observed SED/object, .obs_sed \n')
    zphot_file.write('TEMP_SED_FILE        0                  # Write out best template fit/object, .temp_sed \n')
    zphot_file.write('POFZ_FILE            0                  # Write out Pofz/object, .pz \n')
    zphot_file.write('BINARY_OUTPUT        1                  # Save OBS_SED, TEMP_SED, PZ in binary format to read with e.g IDL \n')
    
    # Prior configuration
    zphot_file.write('\n## Redshift / Mag prior \n')
    zphot_file.write('APPLY_PRIOR          '+str(prior)+'                  # Apply apparent magnitude prior \n')
    zphot_file.write('PRIOR_FILE           '+prior_file+' # File containing prior grid \n')
    zphot_file.write('PRIOR_FILTER         '+prior_filter+'                # Filter from FILTER_RES corresponding to the columns in PRIOR_FILE \n')
    zphot_file.write('PRIOR_ABZP           '+str(zp_ab)+'            # AB zeropoint of fluxes in catalog.  Needed for calculating apparent mags! \n')
    
    # Redshift grid configuration
    zphot_file.write('\n## Redshift Grid \n')
    zphot_file.write('FIX_ZSPEC            '+str(fixspecz)+'                  # Fix redshift to catalog zspec \n')
    zphot_file.write('Z_MIN                '+str(zmin)+'             # Minimum redshift \n')
    zphot_file.write('Z_MAX                '+str(zmax)+'            # Maximum redshift \n')
    zphot_file.write('Z_STEP               '+str(zstep)+'             # Redshift step size \n')
    zphot_file.write('Z_STEP_TYPE          '+zstep_type+'                  #  0 = ZSTEP, 1 = Z_STEP*(1+z) \n')
    
    # Zeropoint offset configuration
    zphot_file.write('\n## Zeropoint Offsets \n')
    zphot_file.write('GET_ZP_OFFSETS       '+str(zp_offsets)+'                  # Look for zphot.zeropoint file and compute zeropoint offsets \n')
    zphot_file.write('ZP_OFFSET_TOL        1.000e-04          # Tolerance for iterative fit for zeropoint offsets [not implemented] \n')
    
    # Rest-frame color configuration
    zphot_file.write('\n## Rest-frame colors \n')
    zphot_file.write('REST_FILTERS         ---                # Comma-separated list of rest frame filters to compute \n')
    zphot_file.write('RF_PADDING           1000               # Padding (Ang) for choosing observed filters around specified rest-frame pair. \n')
    zphot_file.write('RF_ERRORS            1                  # Compute RF color errors from p(z) \n')
    zphot_file.write('Z_COLUMN             z_peak             # Redshift to use for rest-frame color calculation (z_a, z_p, z_m1, z_m2, z_peak) \n')
    zphot_file.write('USE_ZSPEC_FOR_REST   1                  # Use z_spec when available for rest-frame colors \n')
    zphot_file.write('READ_ZBIN            no                 # Get redshifts from OUTPUT_DIRECTORY/MAIN_OUTPUT_FILE.zbin rather than fitting them. \n')
    
    # Cosmology configuration
    zphot_file.write('\n## Cosmology \n')
    zphot_file.write('H0                   70.000             # Hubble constant (km/s/Mpc) \n')
    zphot_file.write('OMEGA_M              0.300              # Omega_matter \n')
    zphot_file.write('OMEGA_L              0.700              # Omega_lambda \n')


catalog_filters = {
    'acs_f435w':'233',
    'acs_f606w':'236',
    'acs_f814w':'239',
    'wfc3_f105w':'202',
    'wfc3_f125w':'203',
    'wfc3_f140w':'204',
    'wfc3_f160w':'205',
    'nircam_f090w':'363',
    'nircam_f115w':'364',
    'nircam_f150w':'365',
    'nircam_f200w':'366',
    'nircam_f277w':'375',
    'nircam_f356w':'376',
    'nircam_f410m':'383',
    'nircam_f444w':'377',
    'miri_f770w':'396',
    'miri_f1000w':'397',
    'miri_f1500w':'400',
    'miri_f1800w':'401',
    'csst_nuv':'425',
    'csst_u':'426',
    'csst_g':'427',
    'csst_r':'428',
    'csst_i':'429',
    'csst_z':'430',
    'csst_y':'431'
}

def translate_config(catfile,configfile='zphot.translate'):
    """
    Generate EAZY filter translation file based on catalog column names.
    
    This function creates a translation file that maps catalog column names
    to EAZY filter identifiers. It automatically detects flux columns
    (ending with '_flux') and their corresponding error columns
    (assumed to be named as filtername+'_fluxerr').
    
    Parameters
    ----------
    catalog : astropy.table.Table or pandas.DataFrame
        The photometric catalog containing the source data.
        Column names should follow the convention: 
        - ID column: 'id'
        - Specz column: 'z_spec'
        - Flux columns: '[filtername]_flux' (e.g., 'acs_f435w_flux')
        - Error columns: '[filtername]_fluxerr' (e.g., 'acs_f435w_fluxerr')
    configfile : str, optional
        Name of the translation file to create (default: 'zphot.translate')
        
    Notes
    -----
    The function uses the global [catalog_filters] dictionary to map filter names
    to EAZY filter numbers. If a filter is not found in this dictionary,
    a warning message is printed.
    
    The translation file format follows EAZY conventions:
    - Flux columns are prefixed with 'F' followed by the filter number
    - Error columns are prefixed with 'E' followed by the filter number
    
    Example
    -------
    For a catalog with column 'acs_f435w_flux':
    - If 'acs_f435w' is in [catalog_filters] with value '233'
    - Output lines: 
      'acs_f435w_flux F233'
      'acs_f435w_fluxerr E233'
      
    See Also
    --------
    catalog_filters : Dictionary mapping filter names to EAZY filter numbers
    """

    catalog = pd.read_csv(catfile)
    translate_file = open(configfile, 'w')
    translate_file.write('################   Translate column names in catalog and in EAZY filters. (Can feed this file back to EAZY)  #################### \n')
    
    for column in catalog.columns:
        print(column)
        if column.endswith('_flux'):
            if column[:-5] in catalog_filters:
                # Write flux column mapping
                translate_file.write(column+' F'+catalog_filters[column[:-5]]+' \n')
                # Write corresponding error column mapping
                translate_file.write(column[:-5]+'_fluxerr E'+catalog_filters[column[:-5]]+' \n')
            else:
                # Print warning for unknown filters
                print('Please find the filter for '+column[:-5]+' in FILTER.RES.latest.info.')



def show_all_fitting(outdir: str, gsdir: str, output_path: str):
    
    zout= EAzy_analysis(outputdir=outdir, outputfile='photz', cache_file=None)
    # show statistical comparison of photoz and specz
    zout.show_photz_compare(set1=zout.z_spec, set2=zout.z_best, set1label=r'$z_{spec}$', set2label=r'$z_{phot}$', distin_chi2=False, distin_qz=False, s=10)
    plt.savefig(outdir+'/photz_specz.png', dpi=300, bbox_inches='tight')

    # Show the EAZY-fitted SED and redshift probability distribution function (PDF), together with the GALFITS-modeled image.
    for idx in range(len(zout.id)):
        galaxy_id = zout.id[idx]
        # 创建图形 - 使用2行2列的布局
        fig = plt.figure(figsize=(14, 10))
        
        # 上方：galfits图（占据整个第一行，旋转-90度）
        ax_galfits = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        galfits_file = os.path.join(gsdir, f'obj{galaxy_id}image_fit.png')
        
        if os.path.exists(galfits_file):
            # 读取galfits图像
            img = mpimg.imread(galfits_file)
            # 旋转-90度（顺时针旋转90度）并显示
            ax_galfits.imshow(np.rot90(img, k=-1), aspect='equal')
            ax_galfits.axis('off')
            ax_galfits.set_title(f'Galaxy ID: {galaxy_id}', fontsize=14, fontweight='bold')
        else:
            ax_galfits.text(0.5, 0.5, f'Galfit image not found\nfor ID {galaxy_id}', 
                        transform=ax_galfits.transAxes, ha='center', va='center', fontsize=12)
            ax_galfits.axis('off')
        
        # 下方：photoz fitting图（SED + PDF）
        # SED图在左下，PDF图在右下
        ax_sed = plt.subplot2grid((2, 2), (1, 0))
        ax_pdf = plt.subplot2grid((2, 2), (1, 1))
        
        # 使用Eazybox的show_fitting方法，传入自定义axes
        try:
            axs = [ax_sed, ax_pdf]
            zout.show_fitting(idx, axs=axs, specz=zout.z_spec[idx], individual_templates=False)
            ax_sed.set_title('Photo-z SED Fitting', fontsize=12, fontweight='bold')
            ax_pdf.set_title('Probability Density Function', fontsize=12, fontweight='bold')
        except Exception as e:
            ax_sed.text(0.5, 0.5, f'Error plotting photo-z fitting:\n{str(e)}', 
                    transform=ax_sed.transAxes, ha='center', va='center', fontsize=10)
            ax_sed.axis('off')
            ax_pdf.axis('off')
        
        # 调整子图间距
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # 保存图像
        output_file = os.path.join(output_path, f'{galaxy_id}_fitting.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        if (idx + 1) % 100 == 0:
            print(f'已处理 {idx + 1}/{len(zout.id)} 个星系')


# Binary file reading functions would go here...
def read_eazy_binary(outputdir='',outputfile='photz',cache_file=None):
    """
    Read Eazy binary files and return a dictionary with the data.
    
    Parameters
    ----------
    outputdir : str
        Path to the Eazy output directory.
    outputfile : str
        Name of the output file, default is 'photz'.
    cache_file : str
        Name of the cache file.
    
    Returns
    -------
    photz_info : dict={'NFILT':NFILT, 'NTEMP':NTEMP, 'NZ':NZ, 'NOBJ':NOBJ}
    tempfilt_out: dict={'tempfilt':tempfilt, 'lc':lc, 'zgrid':zgrid, 'fnu':fnu, 'efnu':efnu}
    coeff_out   : dict={'coeffs':coeffs, 'izbest':izbest, 'tnorm':tnorm}
    tempsed_out : dict={'tempseds':tempseds, 'templambda':templambda, 'da':da, 'db':db}
    pz_out      : dict={'chi2fit':chi2fit, 'kbins':kbins, 'priorzk':priorzk, 'kidx':kidx, 'NK':NK}
    """
    photz_info,tempfilt_out = read_tempfilt_binary(outputdir=outputdir,outputfile=outputfile,cache_file=cache_file)
    coeff_out                = read_coeff_binary(outputdir=outputdir,outputfile=outputfile)
    tempsed_out              = read_tempsed_binary(outputdir=outputdir,outputfile=outputfile)
    pz_out                   = read_pz_binary(outputdir=outputdir,outputfile=outputfile)
    return photz_info,tempfilt_out,coeff_out,tempsed_out,pz_out

def read_tempfilt_binary(outputdir='',outputfile='photz',cache_file=None):
    """
    Read the binary tempfilt file from EAZY and return the data in a dictionary.

    Parameters
    ----------
    outputdir : str
        Path to the Eazy output directory.
    outputfile : str
        Name of the output file, default is 'photz'.
    cache_file : str
        Name of the cache file.

    Returns
    -------
    photz_info : dict={'NFILT':NFILT, 'NTEMP':NTEMP, 'NZ':NZ, 'NOBJ':NOBJ}
    tempfilt_out : dict={'tempfilt':tempfilt, 'lc':lc, 'zgrid':zgrid, 'fnu':fnu, 'efnu':efnu}
    """
    path = os.path.join(outputdir, outputfile)
    if cache_file:
        tempfilt_path = os.path.join(output_dir, cache_file)
    else:
        tempfilt_name = path + '.tempfilt'

    with open(tempfilt_name, 'rb') as tempfilt_file:
        tempfilt_info = np.fromfile(tempfilt_file, dtype=np.int32, count=4)
        NFILT = tempfilt_info[0] # number of filters
        NTEMP = tempfilt_info[1] # number of templates
        NZ = tempfilt_info[2]    # number of redshifts
        NOBJ = tempfilt_info[3]  # number of objects
        
        # tempfilt: fluxes derievd from templates at different bandpass in different redshift
        # lc      : central wavelength of filters, unit: Angstrom
        # zgrid   : redshift grid 
        # fnu     : fluxes of objects at different bandpass
        # efnu    : flux_errors of objects at different bandpass
        tempfilt = np.fromfile(file=tempfilt_file, dtype=np.double, count=NFILT*NTEMP*NZ).reshape((NZ, NTEMP, NFILT)) 
        lc = np.fromfile(file=tempfilt_file, dtype=np.double, count=NFILT)
        zgrid = np.fromfile(file=tempfilt_file, dtype=np.double, count=NZ)
        fnu = np.fromfile(file=tempfilt_file, dtype=np.double, count=NFILT*NOBJ).reshape((NOBJ, NFILT))
        efnu = np.fromfile(file=tempfilt_file, dtype=np.double, count=NFILT*NOBJ).reshape((NOBJ, NFILT)) 
        tempfilt_out = {'tempfilt':tempfilt, 'lc':lc, 'zgrid':zgrid, 'fnu':fnu, 'efnu':efnu}
    photz_info = {'NFILT':NFILT, 'NTEMP':NTEMP, 'NZ':NZ, 'NOBJ':NOBJ}
    return photz_info,tempfilt_out

def read_coeff_binary(outputdir='',outputfile=''):
    """
    Read the binary coeff(OBS_SED) file from EAZY and return the data in a dictionary.

    Parameters
    ----------
    outputdir : str
        Path to the Eazy output directory.
    outputfile : str
        Name of the output file, default is 'photz'.

    Returns
    -------
    coeff_out = {'coeffs':coeffs, 'izbest':izbest, 'tnorm':tnorm}
    """
    path = os.path.join(outputdir, outputfile)
    coeff_name = path + '.coeff'
    
    with open(coeff_name, 'rb') as coeff_file:
        coeff_info = np.fromfile(coeff_file, dtype=np.int32, count=4)
        NFILT = coeff_info[0]
        NTEMP = coeff_info[1]
        NZ = coeff_info[2]
        NOBJ = coeff_info[3]
        
        # .coeff file contains coefficients of templates
        # coeffs: coefficients of each template for each object
        # izbest: the index of the best redshift in zgrid
        # tnorm : normalize coefficient for the templates?
        coeffs = np.fromfile(file=coeff_file, dtype=np.double, count=NTEMP*NOBJ).reshape((NOBJ, NTEMP))
        izbest = np.fromfile(file=coeff_file, dtype=np.int32, count=NOBJ)
        tnorm = np.fromfile(file=coeff_file, dtype=np.double, count=NTEMP)
        coeff_out = {'coeffs':coeffs, 'izbest':izbest, 'tnorm':tnorm}
    return coeff_out

def read_tempsed_binary(outputdir='',outputfile='photz'):
    """
    Read the binary Full templates(TEMP_SED) file from EAZY and return the data in a dictionary.

    Parameters
    ----------
    outputdir : str
        Path to the Eazy output directory.
    outputfile : str
        Name of the output file, default is 'photz'.

    Returns
    -------
    tempsed_out = {'tempseds':tempseds, 'templambda':templambda, 'da':da, 'db':db}
    """
    path = os.path.join(outputdir, outputfile)
    tempsed_name = path + '.temp_sed'
    with open(tempsed_name, 'rb') as tempsed_file:
        tempsed_info = np.fromfile(tempsed_file, dtype=np.int32, count=3)
        NTEMP = tempsed_info[0]
        NTEMPL = tempsed_info[1] # NTEMPL: length of template wavelength array
        NZ = tempsed_info[2]
        
        # templambda  : templates wavelength array
        # temp_seds   : flambda of all the templates
        # da          : Da
        # db          : Db
        templambda = np.fromfile(file=tempsed_file, dtype=np.double, count=NTEMPL)
        tempseds = np.fromfile(file=tempsed_file, dtype=np.double, count=NTEMPL*NTEMP).reshape((NTEMP, NTEMPL))
        da = np.fromfile(file=tempsed_file, dtype=np.double, count=NZ)
        db = np.fromfile(file=tempsed_file, dtype=np.double, count=NZ)
        tempsed_out = {'tempseds':tempseds, 'templambda':templambda, 'da':da, 'db':db}
    return tempsed_out

def read_pz_binary(outputdir='',outputfile=''):
    """
    Read the binary P of Z file from EAZY and return the data in a dictionary.

    Parameters
    ----------
    outputdir : str
        Path to the Eazy output directory.
    outputfile : str
        Name of the output file, default is 'photz'.

    Returns
    -------
    pz_out = {'chi2fit':chi2fit, 'kbins':kbins, 'priorzk':priorzk, 'kidx':kidx, 'NK':NK}
    """
    path = os.path.join(outputdir, outputfile)
    pz_name = path + '.pz'
    with open(pz_name, 'rb') as pz_file:
        pz_info = np.fromfile(pz_file, dtype=np.int32, count=2)
        NZ = pz_info[0]
        NOBJ = pz_info[1]
        
        # chi2fit : chi2 of the best fit
        # pz      : posterior
        chi2fit = np.fromfile(file=pz_file, dtype=np.double, count=NZ*NOBJ).reshape((NOBJ, NZ))
        
        #may break if APPLY_PRIOR is False
        nk = np.fromfile(file=pz_file, dtype=np.int32, count=1)
        if len(nk) > 0:
            NK = nk[0]
            kbins = np.fromfile(file=pz_file, dtype=np.double, count=NK)
            priorzk = np.fromfile(file=pz_file, dtype=np.double, count=NZ*NK).reshape((NK,NZ))
            kidx = np.fromfile(file=pz_file, dtype=np.int32,  count=NOBJ)
            pz_out = {'chi2fit':chi2fit, 'kbins':kbins, 'priorzk':priorzk, 'kidx':kidx, 'NK':NK}
        else:
            pz_out = {'chi2fit':chi2fit, 'kbins':None, 'priorzk':None, 'kidx':None, 'NK':None}
    return pz_out

def zout_analysis(z_spec, z_best):
    """
    Return dz, fraction of outliers and sigma
    """
    compareidx = (z_best > 0) & (z_spec > 0)
    dz = (z_best[compareidx]-z_spec[compareidx]) / (1+z_spec[compareidx])
    idx_outliers = np.fabs(dz) >= 0.15
    f_out = idx_outliers.sum() / compareidx.sum() * 100
    sigma = 1.48 * np.median(np.abs(dz) - np.median(dz))
    f_5sigma = (np.fabs(dz) >= 5*sigma).sum() / compareidx.sum() * 100
    return dz, f_out, sigma, f_5sigma, compareidx

def show_z_compare(set1=None,set2=None,set1label=None,set2label=None,distin_qz=0.95,distin_chi2=50,errorbar=False,zmax=12,deltaz=0.5,s=1):
        """
        Plot the comparison between two sets of redshifts.

        Parameters
        ----------
        set1 : array
            First set of redshifts, default: z_spec.
        set2 : array
            Second set of redshifts, default: z_best.
        set1label : str
            Label of the first set of redshifts, default: None.
        set2label : str 
            Label of the second set of redshifts, default: None.
        distin_qz : bool
            Whether to distinguish the q_z>0.95, default: 0.95, if do not want to distinguish, set to 0.
        distin_chi2 : bool
            Whether to distinguish the chi2>50, default: 50, if do not want to distinguish, set to 0.
        errorbar : bool
            Whether to show errorbar, default: False.
        zmax : float
            Maximum redshift to plot, default: 12.

        Returns
        ----------
        axs : list
            List of axes, axs[0] is the z-z scatter plot, axs[1] is the dz-z scatter plot.
        """
        
        dz, f_out, sigma, f_5sigma, compareidx = zout_analysis(set1,set2)

        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(5*1.2,6*1.2),gridspec_kw={'height_ratios':[8,2]},sharex=True)
        
        ax1.scatter(set1[compareidx],set2[compareidx],s=s,alpha=0.5,marker='+',c='k',edgecolors=None)
        
        ax1.plot([0,zmax],[0,zmax],c='k')
        ax1.plot([0,zmax],[0.15,zmax*1.15],c='k',ls='--',alpha=0.5)
        ax1.plot([0,zmax],[-0.15,zmax*0.85],c='k',ls='--',alpha=0.5)
        ax1.text(0.65,0.23,'N: '+str(compareidx.sum()),fontsize=15,transform=ax1.transAxes)
        ax1.text(0.65,0.11,r'$\sigma_{\rm nMAD}$: '+str(round(sigma,4)),fontsize=15,transform=ax1.transAxes)
        ax1.text(0.65,0.17,r'$f_{out}:  $'+str(round(f_out,2))+'%',fontsize=15,transform=ax1.transAxes)
        ax1.text(0.65,0.05,r'$n(>5\sigma_{\rm nMAD}):  $'+str(round(f_5sigma,2))+'%',fontsize=15,transform=ax1.transAxes)

        if set2label is None:
            ax1.set_ylabel(r'$z_{best}$',fontsize=15)
        else:
            ax1.set_ylabel(set2label,fontsize=15)
        ax1.set_xlim(0,zmax)
        ax1.set_ylim(0,zmax)
        ax2.scatter(set1[compareidx],dz,s=s,c='k')
        ax2.axhline(0,c='k')
        ax2.axhline(0.1,c='k',ls='--',alpha=0.5)
        ax2.axhline(-0.1,c='k',ls='--',alpha=0.5)
        if deltaz!=0:
            ax2.set_ylim(-deltaz,deltaz)
        if set1 is None:
            ax2.set_xlabel(r'$z_{spec}$',fontsize=15)
        else:
            ax2.set_xlabel(set1label,fontsize=15)
        ax2.set_ylabel(r'$\frac{\Delta z}{1+z}$',fontsize=15)
        plt.subplots_adjust(hspace=0)
        return fig, [ax1,ax2]

class EAzy_analysis():
    def __init__(self,outputdir,outputfile,cache_file):
        """
        Can read eazy output files and plot figures.
        # Parameters\n
        outputdir: str 
          output directory
        outputfile: str 
          output file name, NO file extension
        cache_file: str 
          cache file name, file extension is needed
        
        # Example\n
          Either of the following usage is ok
        >>> zout = EAzy_analysis(outputdir='',outputfile='photz',cache_file=None)

          after loading the output file the following functions can be used

        >>> zout.show_photz_compare()
            Plot the comparison between two sets of redshifts, default is z_spec and z_best
        >>> zout.show_zhist1d()
            Plot the histogram of the estimated redshifts.
        >>> zout.show_zhist2d()
            Plot the 2D histogram of the estimated redshifts.
        >>> zout.get_sed_fitting()
            Get the SED fitting of the object.
        >>> zout.show_sed_fitting()
            Plot the SED fitting of the object.
        >>> zout.get_pdf()
            Get the PDF of the object. 
        >>> zout.show_pdf()
            Plot the PDF of the object. 
        >>> zout.show_fitting()
            Plot the SED fitting and PDF of the object.     
        """
        eazy_out = read_eazy_binary(outputdir=outputdir,outputfile=outputfile,cache_file=cache_file)
        photz_info = eazy_out[0]
        tempfilt_out = eazy_out[1]
        coeff_out = eazy_out[2]
        tempsed_out = eazy_out[3]
        pz_out = eazy_out[4]

        self.NOBJ = photz_info['NOBJ']
        self.NZ = photz_info['NZ']

        self.lc = tempfilt_out['lc']
        self.tempfilt = tempfilt_out['tempfilt']
        self.zgrid = tempfilt_out['zgrid']
        self.fnu = tempfilt_out['fnu']
        self.efnu = tempfilt_out['efnu']

        self.izbest = coeff_out['izbest']
        self.coeffs = coeff_out['coeffs']
        
        self.tempseds = tempsed_out['tempseds']
        self.da = tempsed_out['da']
        self.db = tempsed_out['db']
        self.templambda = tempsed_out['templambda']

        # IGM correction for SED
        self.lim1 = np.where(self.templambda < 912)
        self.lim2 = np.where((self.templambda >= 912) & (self.templambda < 1026))
        self.lim3 = np.where((self.templambda >= 1026) & (self.templambda < 1216))

        self.NK = pz_out['NK']
        self.chi2fit = pz_out['chi2fit']
        self.kidx = pz_out['kidx']
        self.priorzk = pz_out['priorzk']

        zout_path = os.path.join(outputdir, outputfile+'.zout')
        zout = Table.read(zout_path, format='ascii.commented_header')
        zout_col = list(zout.columns)
        if 'z_peak' not in zout_col:
            self.z_best = zout['z_a']
            print('No z_best in eazy output, we use z_a as final photoz.')
        else:
            self.z_best = zout['z_peak']
            self.qz = zout['q_z']
        self.z_spec = zout['z_spec']
        self.id = zout['id']
        self.u68 = zout["u68"]
        self.l68 = zout["l68"]

        self.nfilt = zout['nfilt']
        # diffenent types of templates combo have different output file
        # need to be classified before load template information
        if 'temp_p' in zout_col:        # single template fitting with prior
            self.temp = zout['temp_p']
            self.chi2 = zout['chi_p']
            self.combo = 1
        elif 'temp_1' in zout_col:      # single template fitting without prior
            self.temp = zout['temp_1']
            self.chi2 = zout['chi_1']
            self.combo = 1
        elif 'temp_pa' in zout_col:     # double template fitting with prior
            self.temp = [zout['temp_pa'], zout['temp_pb']]
            self.chi2 = zout['chi_p']
            self.combo = 2
        elif 'temp2a' in zout_col:      # double template fitting without prior
            self.temp = [zout['temp2a'], zout['temp_2b']]
            self.chi2 = zout['chi_2']
            self.combo = 2
        elif 'chi_p' in zout_col:       # full template set fitting with prior
            self.temp = np.zeros(len(zout), dtype=int)
            self.chi2 = zout['chi_p']
            self.combo = 99
        else:                           # full template set fitting without prior
            self.temp = np.zeros(len(zout), dtype=int)
            self.chi2 = zout['chi_a']
            self.combo = 99

    def show_photz_compare(self,set1=None,set2=None,set1label=None,set2label=None,s=10,distin_qz=0.95,distin_chi2=50,
                           errorbar=False,zmax=12,deltaz=0.5,ax=None):
        """
        Plot the comparison between two sets of redshifts.

        Parameters
        ----------
        set1 : array
            First set of redshifts, default: z_spec.
        set2 : array
            Second set of redshifts, default: z_best.
        set1label : str
            Label of the first set of redshifts, default: None.
        set2label : str 
            Label of the second set of redshifts, default: None.
        distin_qz : bool
            Whether to distinguish the q_z>0.95, default: 0.95, if do not want to distinguish, set to 0.
        distin_chi2 : bool
            Whether to distinguish the chi2>50, default: 50, if do not want to distinguish, set to 0.
        errorbar : bool
            Whether to show errorbar, default: False.
        zmax : float
            Maximum redshift to plot, default: 12.

        Returns
        ----------
        ax : list
            The z-z scatter plot and dz-z scatter plot.
        """
        
        if set1 is None:
            set1 = self.z_spec
        if set2 is None:
            set2 = self.z_best
            
        dz, f_out, sigma, f_5sigma, compareidx = zout_analysis(set1,set2)

        if not ax:
            if deltaz!=0:
                plt.figure(figsize=(7, 9))
            else:
                plt.figure(figsize=(7, 7))
            ax = plt.subplot(111)
            
        # show errorbar        
        if errorbar:
            u_err = self.z_u68 - self.z_best
            l_err = self.z_best - self.z_l68
            ax.errorbar(set1, set2, yerr=[l_err, u_err], fmt='o', ms=1, color=c, elinewidth=.25, alpha=.8)   
        
        if distin_chi2 != 0:
            chi2_cut = self.chi2 > distin_chi2
            ax.scatter(set1[chi2_cut],set2[chi2_cut],s=s,alpha=1,marker='o',c='b',)
        if distin_qz != 0:
            qz_cut = self.qz > distin_qz
            ax.scatter(set1[qz_cut],set2[qz_cut],s=s,alpha=0.5,marker='o',c='r',)

        #ax.scatter(set1[compareidx],set2[compareidx],s=s,alpha=0.5,marker='+',c='k',edgecolors=None)
        ax.plot(set1[compareidx], set2[compareidx], 'k.')
        ax.plot([0,zmax],[0,zmax],c='k')
        ax.plot([0,zmax],[0.15,zmax*1.15],c='k',ls='--',alpha=0.5)
        ax.plot([0,zmax],[-0.15,zmax*0.85],c='k',ls='--',alpha=0.5)
        ax.text(0.05,0.95,'N: '+str(compareidx.sum()),fontsize=15,transform=ax.transAxes)
        ax.text(0.05,0.83,r'$\sigma_{\rm NMAD}$: '+str(round(sigma,4)),fontsize=15,transform=ax.transAxes)
        ax.text(0.05,0.89,r'$f_{out}:  $'+str(round(f_out,2))+'%',fontsize=15,transform=ax.transAxes)
        ax.text(0.05,0.77,r'$n(>5\sigma_{\rm NMAD}):  $'+str(round(f_5sigma,2))+'%',fontsize=15,transform=ax.transAxes)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.xaxis.set_tick_params(direction='in', which='both', top=True, bottom=True, labelsize=15)
        ax.yaxis.set_tick_params(direction='in', which='both', left=True, right=True, labelsize=15)

        if set2label is None:
            ax.set_ylabel(r'$z_{best}$', fontsize=16)
        else:
            ax.set_ylabel(set2label, fontsize=16)
        ax.set_xlim(0,zmax)
        ax.set_ylim(0,zmax)

        if deltaz!=0:
            ax.xaxis.set_tick_params(labelbottom=False)
            divider = make_axes_locatable(ax)
            ax_bottom = divider.append_axes('bottom', size=1, pad=.1, sharex=ax)
        
            ax_bottom.scatter(set1[compareidx], dz, s=s, c='k')
            ax_bottom.axhline(0,c='k')
            ax_bottom.axhline(0.15,c='k',ls='--',alpha=0.5)
            ax_bottom.axhline(-0.15,c='k',ls='--',alpha=0.5)
            # axis settings
            ax_bottom.set_ylim(-deltaz, deltaz)
            ax_bottom.set_ylabel(r'$\frac{\Delta z}{1+z}$', fontsize=16)
            ax_bottom.set_xlabel(r'z$_{\rm spec}$', fontsize=16)
            # 设置tick方向向内，并在上下左右都显示
            ax_bottom.xaxis.set_tick_params(direction='in', which='both', top=True, bottom=True,labelsize=15)
            ax_bottom.yaxis.set_tick_params(direction='in', which='both', left=True, right=True,labelsize=15)
            ax_bottom.grid(True, alpha=0.3, linestyle='--')

        plt.subplots_adjust(hspace=0)
        plt.show()
        return ax

    def show_zhist1d(self,zmax=12,ax=None):
        """
        Plot the histogram of the estimated redshifts.

        Parameters
        ----------
        zmax : float
            Maximum redshift to plot, default: 12.
        ax : matplotlib.axes
            Axes to plot on, default: None.

        Returns
        ----------
        ax : matplotlib.axes
            Axes to plot on.
        """

        if not ax:
            plt.figure(figsize=(8,6))
            ax = plt.subplot(111)
        ax.hist(self.z_best,bins=50,range=(0,zmax),lw=2)
        ax.set_xlabel(r'$z_{best}$')
        ax.set_ylabel('N')
        ax.set_xlim(0,zmax)
        return ax

    def show_zhist2d(self,set1=None,set2=None,set1label=None,set2label=None,zmax=12,ax=None):
        """
        Plot the 2D histogram of the estimated redshifts.

        Parameters
        ----------
        set1 : array
            First set of redshifts, default: z_spec.
        set2 : array
            Second set of redshifts, default: z_best.
        set1label : str
            Label of the first set of redshifts, default: None.
        set2label : str
            Label of the second set of redshifts, default: None.
        zmax : float
            Maximum redshift to plot, default: 12.
        ax : matplotlib.axes
            Axes to plot on, default: None.

        Returns
        ----------
        ax : matplotlib.axes
            Axes to plot on.
        """
        if set1 is None:
            set1 = self.z_spec
        if set2 is None:
            set2 = self.z_best

        compareidx = (set1 > 0) & (set2 > 0)
        if not ax:
            plt.figure(figsize=(6,6))
            ax = plt.subplot(111)
        ax.hist2d(set1[compareidx],set2[compareidx],bins=50,cmap='Blues_r',norm=LogNorm(),range=[[0,zmax],[0,zmax]])
        if set1label is None:
            ax.set_xlabel(r'$z_{spec}$')
        else:
            ax.set_xlabel(set1label)
        if set2label is None:
            ax.set_xlabel(r'$z_{best}$')
        else:
            ax.set_xlabel(set2label)
        ax.set_xlim(0,zmax)
        ax.set_ylim(0,zmax)
        return ax

    def get_sed_fitting(self,idx,):
        """
        Get the SED fitting of the object.

        Parameters
        ----------
        idx : int
            Index of the object in the catalog.

        Returns
        ----------
        lambdaz : array
            Wavelength array of the SED fitting.
        itemp_sed : array
            SED fitting of the templates.
        iobs_sed : array
            SED fitting of the observed fluxes.
        fobs : array
            Observed fluxes.
        efobs : array
            Observed flux errors.
        """
        z = self.zgrid[self.izbest[idx]]
        lambdaz = (1+z) * self.templambda

        #flambda_coeff = 3e18*1e-29 #for catalog input in muJy

        iobs_sed = np.dot(self.tempfilt[self.izbest[idx]].T, self.coeffs[idx]) #/self.lc**2*flambda_coeff
        #itemp_sed = np.dot(tempseds.T, coeffs[idx])
        itemp_sed = self.tempseds.T * self.coeffs[idx] / (1+z)**2 #*flambda_coeff/5500**2
        
        if self.lim1[0].size > 0: 
            itemp_sed[self.lim1] *= 0.
        if self.lim2[0].size > 0: 
            itemp_sed[self.lim2] *= 1.-self.db[self.izbest[idx]]
        if self.lim3[0].size > 0: 
            itemp_sed[self.lim3] *= 1.-self.da[self.izbest[idx]]   

        # convert observed f_nu (from the catalog) to f_lambda
        fobs = self.fnu[idx] #/self.lc**2*flambda_coeff
        efobs = self.efnu[idx] #/self.lc**2*flambda_coeff

        return lambdaz,itemp_sed,iobs_sed,fobs,efobs

    def show_sed_fitting(self,idx,xrange=(2000,50000),yrange=None,log_x=True,individual_templates=True,
                        ltext=['U','G','R','I','Z','F606W','F814W','F125W','F140W','F160W','Ks','CH1','CH2'],fluxtype='f_lambda',ax=False):
        """
        Plot the SED fitting of the object.

        Parameters
        ----------
        idx : int
            Index of the object in the catalog.
        xrange : tuple
            x range of the SED fitting plot, default: (2000,50000).
        yrange : tuple
            y range of the SED fitting plot, default: None, e.g. (0, max(iobs_sed+efnu)*1.1).
        log_x : bool
            Whether to use log scale in x axis, default: True.
        individual_templates : bool
            Whether to plot individual templates, default: True.
        ltext : list
            List of the filter names, default: ['U','G','R','I','Z','F606W','F814W','F125W','F140W','F160W','Ks','CH1','CH2'].
        ax : matplotlib.axes
            Axes to plot on, default: None.

        Returns
        ----------
        ax : matplotlib.axes
            Axes to plot on.
        """
        
        if not ax:
            plt.figure(figsize=(8,6))
            ax = plt.subplot(111)

        if self.izbest[idx] < 0:
            if self.nfilt[idx] == -99.:
                string = 'less than 5 filters'
            else:
                string = str(self.nfilt[idx])+' filters'
            ax.text(0.5,0.5,'idx: '+str(idx)+'\n'+string+' has SNR>1 \n No good fit!',transform=ax.transAxes,fontsize=20,ha='center',va='center')
            return ax
        
        lambdaz,itemp_sed,iobs_sed,fobs,efobs = self.get_sed_fitting(idx,)
        
        if fluxtype == 'f_lambda':

            flambda_coeff = 3e18*1e-29 #for catalog input in muJy
            fobs = fobs/self.lc**2*flambda_coeff
            efobs = efobs/self.lc**2*flambda_coeff
            iobs_sed = iobs_sed/self.lc**2*flambda_coeff
            itemp_sed = itemp_sed/5500**2*flambda_coeff
            temp_sed = itemp_sed.sum(axis=1)
            sed_ylabel = r'$f_{\lambda}$'
            texty=0.16
        elif fluxtype == 'f_nu':
            print('here')
            #temp_sed = itemp_sed.sum(axis=1)*lambdaz**2/5500**2
            itemp_sed = (itemp_sed.T*lambdaz**2/5500**2).T
            temp_sed = itemp_sed.sum(axis=1) 
            sed_ylabel = r'$f_{\nu}$'
            texty = 0.75
        highsnr = (fobs/efobs > 2)&(efobs > 0)
        lowsnr = (fobs/efobs < 2)&(efobs > 0)
        neg_err = efobs < 0
        
        ax.scatter(self.lc,iobs_sed,c='r',s=100,marker='o',alpha=1,zorder=2)
        ax.plot(lambdaz,temp_sed,c='y',alpha=1,zorder=1,lw=0.5)
        if individual_templates:
            ax.plot(lambdaz,itemp_sed,c='y',alpha=0.3,zorder=1,lw=0.5)
            
        s = 'zpeak: %.3f\n'%self.zgrid[self.izbest[idx]]+r'$\chi^2$: %.3f'%self.chi2fit[idx][self.izbest[idx]]
        ax.text(0.03,texty-0.13,s,transform=ax.transAxes,fontsize=12,ha='left',va='bottom')
        s = 'zbest: %.3f'%self.z_best[idx]
        ax.text(0.03,texty,s,transform=ax.transAxes,fontsize=12,ha='left',va='bottom')
        ax.errorbar(self.lc[highsnr], fobs[highsnr], yerr=efobs[highsnr], ecolor='b', color='b', fmt='^', alpha=1, markeredgecolor='b', 
                            markerfacecolor='None', markeredgewidth=1, elinewidth=1.5, ms=8, zorder=3,label='SNR>2')
        ax.errorbar(self.lc[lowsnr], fobs[lowsnr], yerr=efobs[lowsnr], ecolor='g', color='g',fmt='o', alpha=1, markeredgecolor='g', 
                            markerfacecolor='None', markeredgewidth=1, elinewidth=1.5, ms=8, zorder=3,label='SNR<2')
        ax.errorbar(self.lc[neg_err], fobs[neg_err], yerr=0, ecolor='k', color='k',fmt='x', alpha=1, markeredgecolor='k', 
                            markerfacecolor='None', markeredgewidth=1, elinewidth=1.5, ms=8, zorder=3,label='ERROR<0')
        ax.legend()
        ax.set_yscale('log')
        ax.set_xlim(xrange[0], xrange[1])
        if yrange:
            ax.set_ylim(yrange[0], yrange[1])
        else:
            yrange = (0.9*min(min(fobs[fobs>0]),min(iobs_sed)),1.3*(max(max(fobs+efobs),max(iobs_sed))))
            ax.set_ylim(yrange[0], yrange[1])
        ax.set_xlabel(r'Wavelength ($\AA$)',fontsize=12)
        ax.set_ylabel(sed_ylabel,fontsize=12)
        if log_x:
            ax.semilogx()

        if ltext:
            for i,istr in enumerate(ltext):
                ax.text(self.lc[i],yrange[1],istr,fontsize=12,rotation=90,va='bottom',ha='center')
            #ax.text(0.05,0.9,ltext,transform=ax.transAxes,fontsize=12)
        return ax

    def get_pdf(self,):
        """
        Get the PDF of the object.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        # convert chi2fit to p(z)
        coeff = np.ones(self.NZ)
        min_chi2fit = np.multiply(np.min(self.chi2fit, axis=1).reshape(self.NOBJ,1), coeff)

        if self.NK:
            nofitidx = (self.kidx >= self.NK) | (self.kidx < 0)
            self.kidx[nofitidx] = 0
            self.priorz = self.priorzk[self.kidx,:]
        else:
            self.priorz = np.ones([self.NOBJ, self.NZ]) # means no prior
        
        self.noprior_pzout = np.exp(-0.5*(self.chi2fit-min_chi2fit)) # likelihood
        self.pzout = np.exp(-0.5*(self.chi2fit-min_chi2fit)) * self.priorz # likelihood*prior
        
        self.noprior_pzout = self.noprior_pzout/np.multiply(np.trapezoid(self.noprior_pzout,self.zgrid,axis=1).reshape(self.NOBJ,1), coeff) # normalization
        self.pzout = self.pzout/np.multiply(np.trapezoid(self.pzout,self.zgrid,axis=1).reshape(self.NOBJ,1), coeff) # normalization
        
    def show_pdf(self,idx,zrange=(0,12),ax=False,specz=0):
        """
        Plot the PDF of the object.

        Parameters
        ----------
        idx : int
            Index of the object in the catalog.
        zrange : tuple
            x range of the PDF plot, default: (0,12).
        ax : matplotlib.axes
            Axes to plot on, default: None.
        specz : float
            Spectroscopic redshift, default: 0, it means no spectroscopic redshift plotted in the figure.

        Returns
        ----------
        ax : matplotlib.axes
            Axes to plot on.
        """

        if not ax:
            plt.figure(figsize=(8,6))
            ax = plt.subplot(111)

        if self.izbest[idx] < 0:
            if self.nfilt[idx] == -99.:
                string = 'less than 5 filters'
            else:
                string = str(self.nfilt[idx])+' filters'
            ax.text(0.5,0.5,'idx: '+str(idx)+'\n'+string+' has SNR>1 \n No good fit!',transform=ax.transAxes,fontsize=20,ha='center',va='center')
            return ax
        
        self.get_pdf()

        ax.plot(self.zgrid,self.noprior_pzout[idx],c='gray',alpha=1,lw=1)
        ax.fill_between(self.zgrid,self.noprior_pzout[idx],0,color='gray',alpha=0.2)

        ax.plot(self.zgrid,self.pzout[idx],c='orange',alpha=1,lw=1)
        ax.fill_between(self.zgrid,self.pzout[idx],0,color='orange',alpha=0.2)
        ax.axvline(self.zgrid[self.izbest[idx]],color='k',linewidth=1, label='$z_{peak}$', alpha=0.8)
        if specz != 0:
            ax.axvline(specz,color='r',linewidth=1, label='$z_{spec}=$'+str(specz), alpha=0.8)
            plt.legend()
        ax.set_xlabel('z')
        ax.set_ylabel('p(z)')
        ax.set_xlim(zrange[0],zrange[1])
        ax.set_ylim(0,max(self.pzout[idx])*1.1)
        return ax

    def show_fitting(self,idx,xrange=(2000,50000),yrange=None,log_x=True,individual_templates=True,specz=0,
                        ltext=None,zrange=(0,12),fluxtype='f_lambda',axs=None):
        """
        Plot the SED fitting and PDF in one figure.

        Parameters
        ----------
        idx : int
            Index of the object in the catalog.
        xrange : tuple
            x range of the SED fitting plot, default: (2000,50000).
        yrange : tuple
            y range of the SED fitting plot, default: None, e.g. (0, max(iobs_sed+efnu)*1.1).
        log_x : bool
            Whether to use log scale in x axis, default: True.
        individual_templates : bool
            Whether to plot individual templates, default: True.
        ltext : list
            List of the filter names, default: ['U','G','R','I','Z','F606W','F814W','F125W','F140W','F160W','Ks','CH1','CH2'].
        zrange : tuple
            x range of the PDF plot, default: (0,12).
        axs : list
            List of the axes, default: None.
        specz : float
            Spectroscopic redshift, default: 0, it means no spectroscopic redshift plotted in the figure.

        Returns
        ----------
        axs : list
            List of the axes, ax0 is the SED fitting plot, ax1 is the PDF plot.
        """
        if axs is None:
            fig,axs = plt.subplots(1,2,figsize=(15,6))
        
        ax_sed = axs[0]
        ax_pdf = axs[1]
        ax_sed = self.show_sed_fitting(idx,xrange=xrange,yrange=yrange,individual_templates=individual_templates,log_x=log_x,ax=ax_sed,ltext=ltext,fluxtype=fluxtype)
        ax_pdf = self.show_pdf(idx,zrange=zrange,ax=ax_pdf,specz=specz)
        return axs

