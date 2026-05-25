import os
import numpy as np


import sys

# sys.path.append(r"/mnt/d/Ubuntu/software/MIRA/MIRA-dev_se/mira/")
sys.path.append(r"/mnt/d/Ubuntu/software/galfitx2")
import galfitx

import galfitx.source_detection as se
import galfitx.Eazybox_llm as eazyllm


catfile = "/mnt/d/Ubuntu/software/Limins_code/eazy_gx/testeazy/smacs_update_iso_err_zspec.cat"
eazyllm.translate_config(catfile)

eazyllm.zphot_config(
    catfile,
    "./test",
    temperr=0.03,
    syserr=0.01,
    prior=0,
    zp_offsets=0,
    fixspecz=0,
    configfile="./zphot.param",
    eazypath="/mnt/d/Ubuntu/software/eazy/eazy-photoz-master/",
    template_file="/mnt/d/Ubuntu/software/eazy/eazy-photoz-master/templates/fsps_full/tweak_fsps_QSF_12_v3.param",
)


eazyllm.run_eazy(eazypath="/mnt/d/Ubuntu/software/eazy/eazy-photoz-master")
