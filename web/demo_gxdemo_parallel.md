# Demo: Multi-GPU Parallel Pipeline (gxdemo_parallel.py)

This walkthrough covers `galfitx/gxdemo_parallel.py`, which extends the sequential `gxdemo.py` pipeline with multi-GPU parallel processing. The core analysis is identical -- six-band JWST/NIRCam structural analysis through pure-image fitting, photo-z estimation, pure-SED fitting, and joint image+SED fitting -- but the per-group work is distributed across multiple GPUs using Python's `ProcessPoolExecutor`.

The sections below focus on what is **different** from `gxdemo.py`. Read the [full pipeline demo](demo_gxdemo.md) first for the shared concepts.

---

## Prerequisites

Same as the sequential pipeline, plus:

- **Multiple NVIDIA GPUs** available on the system (the demo configures 8)
- **CUDA toolkit** installed so that `CUDA_VISIBLE_DEVICES` works correctly
- Enough GPU memory per device to hold at least one GalfitS fitting problem

---

## Section 1: Additional Imports

```python
import datetime
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
```

Compared to `gxdemo.py`, this script imports:
- **`ProcessPoolExecutor`**: Manages a pool of worker processes, each running on a separate GPU
- **`functools.partial`**: Available but not directly used in the final version (the `process_group` function takes all arguments explicitly)

---

## Section 2: SPS Template Paths

```python
template_dir = "./sps_templates/"
SPS_catalog_path = os.path.join(template_dir, "UNCOVER_DR4_SPS_catalog.fits")
sfhs_path = os.path.join(template_dir, "sfhs_SPS_DR4.npz")
```

These paths are defined at the top of the file because they are needed both in the main process (for the photometry configuration) and inside the worker function (for SFH prior computation). Keeping them as module-level variables makes them accessible to `process_group()`.

---

## Section 3: The `process_group()` Worker Function

This is the central addition. All per-group logic that was inline in `gxdemo.py` is encapsulated in a single function that can be dispatched to any GPU.

### Function signature

```python
def process_group(i_group, group, object_dict, outtab_dict, bg_params_dict,
                  nband, sciname_list, label_list, filter_list, psf_list,
                  zero_list, pixel_scale_list, expt_list, mjsr_list, gain_list,
                  det_label, catalog_name, cutout_dir,
                  gs_pureImage_dir, gs_pureSed_dir, gs_image_sed_dir,
                  gmoutdir, skynoisepath, stampfile, psf_file,
                  try_filterlist, apers_list, kron_scale, ref_pixel_scale,
                  eazy_path, num_gpus, whtname_list, weight_name):
```

The function takes the group index, the list of object indices in the group, and every configuration parameter as flat arguments. This is necessary because `ProcessPoolExecutor` uses `pickle` for serialization, and complex objects like `astropy.table.Table` or `pandas.DataFrame` may not pickle cleanly.

### GPU assignment

```python
    gpu_id = i_group % num_gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"[Group {i_group}] Assigned to GPU {gpu_id}")
```

Round-robin GPU assignment: group 0 goes to GPU 0, group 1 to GPU 1, ..., group `num_gpus` to GPU 0 again, and so on. Setting `CUDA_VISIBLE_DEVICES` in the worker process environment restricts JAX (used by GalfitS) to see only the assigned GPU. This is the standard way to partition GPUs among workers in a multi-process setup.

### DataFrame conversion for pickling

```python
    import pandas as pd
    bg_df = pd.DataFrame(bg_params_dict)
```

`bg_params` is a pandas DataFrame in the main process. It is converted to a dictionary (`bg_params_dict`) before submission to the executor, and reconstructed inside the worker. This avoids potential pickle issues with DataFrames crossing process boundaries.

### GALFITM binary resolution

```python
    if os.path.exists(public_galfit_path):
        galfit_path = public_galfit_path
    else:
        galfit_path = shutil.which('galfitm')
        if galfit_path is None:
            raise FileNotFoundError("No galfitm found")
```

Each worker independently locates the GALFITM binary. This is necessary because `shutil.which()` resolves at runtime and the PATH may differ across processes.

### The Five Stages

The worker function contains all five analysis stages for one group:

#### Stage 1: Pure Image Fitting

```python
    for idx in sorted_idx_group:
        objects = object_dict[str(idx)]
        # ... gather stamps, masks, PSFs ...
        prepare_galfits(lyric_path=lyric_path, ..., use_sed=0, convf=False)
        command = f"python {galfitS_path} --config {lyric_path} ..."
        os.system(command)
```

Identical to the sequential version. For each source in the group (sorted brightest-first), stamps and masks are gathered, PSFs are cropped, a `.lyric` file is generated, and GalfitS is run in pure-image mode.

#### Stage 2: Isophotal Flux and Errors

```python
    iso_pipeline._compute_model_fluxes_for_galaxies(
        galids=sorted_idx_group_1idx, ...)
    iso_pipeline.step4_compute_isophotal_errors(
        bg_df=bg_df, flux_df_path=gs_flux_outfile, ...)
```

The photometry pipeline is reconstructed inside the worker (from the config parameters) and used to measure model fluxes and errors. Output files are suffixed with `_group{i_group}` to avoid name collisions between workers.

#### Stage 3: EAZY Photo-z

```python
    temperr = 0.01
    syserr  = 0.01
    zphot_config(gs_flux_err_outfile, eazy_out_path, temperr, syserr, ...)
    translate_config(gs_flux_err_outfile, ...)
    run_eazy(eazypath=eazy_path, ...)
```

EAZY is run per-group with slightly different error settings (`temperr=0.01, syserr=0.01`) compared to the sequential version (`0.03`). Each group gets its own EAZY output directory (`./eazy/{i_group}/`).

#### Stage 4: Pure SED Fitting

```python
    gen_pSed_data_lyric(cat_path=gs_flux_err_outfile, z_cat_path=z_out_path, ...)
    for idx in sorted_idx_group:
        command = f"CUDA_VISIBLE_DEVICES=7 python {galfitS_path} ..."
        os.system(command)
```

Note: the `CUDA_VISIBLE_DEVICES=7` in this command is a leftover from the sequential script. In practice, the GPU is already set by the environment variable at the top of `process_group()`, so this hardcoded value should ideally be removed or replaced. The environment-level assignment takes precedence for JAX, but explicit values in subcommands can override it.

#### Stage 5: Image+SED Fitting

```python
    for idx in sorted_idx_group:
        # ... gather z_list from photo-z ...
        prepare_galfits(lyric_path=lyric_path, ..., use_sed=1, use_sfh_prior=True, ...)
        command = f"python {galfitS_path} --config {lyric_path} ... --prior {prior_path}"
        os.system(command)
```

Joint image+SED fitting with SFH priors, identical logic to the sequential version.

### Return value

```python
    return f"Group {i_group} completed on GPU {gpu_id}"
```

A simple status string that is printed by the main process when the future resolves.

---

## Section 4: Main Program -- Parallel Dispatch

### Configuration (identical to gxdemo.py)

The main program defines all the same input lists (`sciname_list`, `whtname_list`, `filter_list`, etc.), runs `SExtractor_HDR`, creates stamp files, cuts stamps, builds masks, and runs `Union_Set`. This portion is unchanged from the sequential version.

### Serialization preparation

```python
outtab_dict   = {col: outtab[col].tolist() for col in outtab.colnames}
bg_params_dict = bg_params.to_dict('list')
```

Before submitting work to the executor, two potentially problematic objects are converted to plain Python dictionaries:
- **`outtab`** (astropy Table) is converted column-by-column to `{column_name: [values]}`.
- **`bg_params`** (pandas DataFrame) is converted using `.to_dict('list')`.

Both conversions produce simple `{str: list}` structures that pickle reliably.

### Parallel execution

```python
num_gpus = 8  # Adjust based on your system

print(f"\n{'='*60}")
print(f"Starting parallel processing with {num_gpus} GPUs")
print(f"Total groups: {len(groups)}")
print(f"{'='*60}\n")

with ProcessPoolExecutor(max_workers=num_gpus) as executor:
    futures = []
    for i_group, group in enumerate(groups):
        future = executor.submit(
            process_group,
            i_group=i_group, group=group,
            object_dict=object_dict,
            outtab_dict=outtab_dict,
            bg_params_dict=bg_params_dict,
            # ... all other parameters ...
            num_gpus=num_gpus,
        )
        futures.append(future)

    print("Submitted all groups to workers. Waiting for completion...\n")
    for i, future in enumerate(futures):
        result = future.result()
        print(f"[{i+1}/{len(futures)}] {result}")

print(f"\n{'='*60}")
print(f"All processing completed!")
print(f"Total time: {datetime.datetime.now() - begin_time}")
print(f"{'='*60}\n")
```

Key details:
- **`max_workers=num_gpus`**: One worker process per GPU. Setting this higher than the GPU count would cause contention; setting it lower would leave GPUs idle.
- **All groups are submitted immediately**: The executor queues them internally. When a worker finishes one group, it picks up the next.
- **`future.result()` blocks**: The main process waits for each future in order. If group 3 finishes before group 1, its result is held until groups 1 and 2 have also finished. This is for orderly printing only -- the workers run independently.
- **No shared state**: Each worker gets copies of all input data. There is no inter-worker communication or shared memory. This is safe but uses more RAM than a threading approach.

---

## Section 5: Performance Considerations

### Scaling

With `N` groups and `G` GPUs, the wall-clock time is approximately `N/G * T_group`, where `T_group` is the time for one group. In practice, groups vary in size (number of sources), so load balancing is not perfect. Groups with many overlapping sources take longer.

### Memory

Each worker process loads its own copies of the science images, catalogs, and segmentation maps. For six large NIRCam mosaics, this can consume significant RAM. Monitor system memory if running on a shared machine.

### GPU utilization

The round-robin assignment is simple but may leave some GPUs idle if groups are uneven. A more sophisticated scheduler could assign groups to the least-busy GPU, but the current implementation prioritizes simplicity.

### Error handling

If one worker raises an exception, `future.result()` will re-raise it in the main process. The other workers continue unaffected. However, partial results from the failed group are lost. Consider adding try/except blocks inside `process_group()` for production use.

---

## Section 6: Differences from gxdemo.py -- Summary

| Aspect | gxdemo.py | gxdemo_parallel.py |
|---|---|---|
| Execution | Sequential loop over groups | Parallel via `ProcessPoolExecutor` |
| GPU usage | Single GPU (hardcoded) | Multiple GPUs (round-robin) |
| Per-group logic | Inline in main script | Encapsulated in `process_group()` |
| Data serialization | Direct object references | Tables/DataFrames converted to dicts |
| Output file naming | Generic names | Suffixed with `_group{i_group}` |
| EAZY error settings | `temperr=0.03, syserr=0.03` | `temperr=0.01, syserr=0.01` |
| `combine_catalogs()` | Called at the end | Not called (add manually after all groups finish) |
| `ebv` value | `None` (auto-detect) | `0.01` (fixed low value) |

Note: The parallel version does not call `combine_catalogs()` at the end. You should add this call manually after all groups complete, or run it as a separate step.
