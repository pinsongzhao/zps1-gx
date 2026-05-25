# Read Setup

::: galfitx.read_setup

Module for parsing GALAPAGOS-style setup files and reading multi-band image file lists.

---

## `set_trailing_slash`

```python
set_trailing_slash(path)
```

Ensure a filesystem path ends with the OS-specific directory separator (`/` on Unix, `\\` on Windows). If the path already ends with a separator, it is returned unchanged.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | `str` | *(required)* | The directory path to process. |

### Returns

**`str`** -- The input path with a trailing separator guaranteed.

### Example

```python
from galfitx.read_setup import set_trailing_slash

set_trailing_slash("/home/user/data")
# Returns: "/home/user/data/"

set_trailing_slash("/home/user/data/")
# Returns: "/home/user/data/"
```

---

## `valid_num`

```python
valid_num(s)
```

Check if a string represents a valid number (integer or float).

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `s` | `str` | *(required)* | The input string to test. |

### Returns

**`bool`** -- `True` if the string can be converted to a float, `False` otherwise.

### Example

```python
from galfitx.read_setup import valid_num

valid_num("3.14")    # True
valid_num("42")      # True
valid_num("hello")   # False
valid_num("1e-3")    # True
```

---

## `read_setup`

```python
read_setup(setup_file)
```

Read and parse a GALAPAGOS-style setup file. The setup file uses a line format with a 4-character code (e.g., `A00)`, `B01)`, `E17)`) followed by a value. Lines starting with `#` are treated as comments. The function returns a `Setup` object populated with all parameters, applying sensible defaults for any missing values.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `setup_file` | `str` | *(required)* | Path to the GALAPAGOS setup file. |

### Returns

**`Setup`** -- An object containing all parsed parameters as attributes. See below for the full attribute listing.

### Raises

| Exception | Condition |
|-----------|-----------|
| `SystemExit` | If the setup file does not exist. |
| `ValueError` | If an unrecognized setup code is encountered. |

### Setup Codes and Attributes

The setup file is organized into sections identified by the first letter of the 4-character code:

#### Section A -- Files and Directories

| Code | Attribute | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `A00)` | `files` | `str` | `""` | Path to the image file list. |
| `A01)` | `outdir` | `str` | `""` | Output directory (trailing slash ensured). |

#### Section B -- SExtractor Options

| Code | Attribute | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `B00)` | `dosex` | `int` | `0` | Set to `1` if `"execute"`. |
| `B01)` | `sexexe` | `str` | `""` | SExtractor executable path. |
| `B02)` | `sexout` | `str` | `""` | SExtractor output directory. |
| `B03)` | `cold` | `str` | `""` | Cold detection image name. |
| `B04)` | `coldcat` | `str` | `""` | Cold catalog name. |
| `B05)` | `coldseg` | `str` | `""` | Cold segmentation map name. |
| `B06)` | `hot` | `str` | `""` | Hot detection image name. |
| `B07)` | `hotcat` | `str` | `""` | Hot catalog name. |
| `B08)` | `hotseg` | `str` | `""` | Hot segmentation map name. |
| `B09)` | `enlarge` | `float` | `1.1` | Enlargement factor. |
| `B10)` | `outcat` | `str` | `""` | Combined output catalog name. |
| `B11)` | `outseg` | `str` | `""` | Combined output segmentation name. |
| `B12)` | `outparam` | `str` | `""` | Output parameter file. |
| `B13)` | `check` | `str` | `""` | Check image name. |
| `B14)` | `chktype` | `str` | `"none"` | Check image type. |
| `B15)` | `sex_rms` | `int` | `0` | Set to `1` if `"rms"`. |
| `B16)` | `exclude` | `str` | `""` | Exclusion file. |
| `B17)` | `exclude_rad` | `float` | `2.0` | Exclusion radius. |
| `B18)` | `outonly` | `int` | `0` | Set to `1` if `"outonly"`. |
| `B19)` | `bad` | `str` | `""` | Bad pixel file. |
| `B20)` | `sexcomb` | `str` | `""` | Combined SExtractor output. |

#### Section C -- Stamp Creation

| Code | Attribute | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `C00)` | `dostamps` | `int` | `0` | Set to `1` if `"execute"`. |
| `C01)` | `stampfile` | `str` | `""` | Stamp file name. |
| `C02)` | `stamp_pre` | `list[str]` | `[""]` | Band prefix list. |
| `C03)` | `stampsize` | `float` | `2.5` | Stamp size factor. |

#### Section D -- Sky Estimation

| Code | Attribute | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `D00)` | `dosky` | `int` | `0` | Set to `1` if `"execute"`. |
| `D01)` | `skymap` | `str` | `""` | Sky-map file name. |
| `D02)` | `outsky` | `str` | `""` | Output sky file name. |
| `D03)` | `skyscl` | `float` | `3.0` | Sky scale factor. |
| `D04)` | `neiscl` | `float` | `1.5` | Neighbour scale factor. |
| `D05)` | `skyoff` | `float` | `20` | Sky offset in pixels. |
| `D06)` | `dstep` | `int` | `30` | Radial step for sky annuli. |
| `D07)` | `wstep` | `int` | `60` | Annulus width for sky estimation. |
| `D08)` | `gap` | `int` | `30` | Gap between Kron radius and first annulus. |
| `D09)` | `cut` | `float` | `0.0` | Cut parameter. |
| `D10)` | `nobj_max` | `int` | `0` | Maximum number of objects. |
| `D11)` | `power` | `float` | `0.0` | Power parameter. |
| `D12)` | `nslope` | `int` | `0` | Number of slope points. |
| `D13)` | `stel_slope` | `float` | `-0.3` | Stellar locus slope. |
| `D14)` | `stel_zp` | `float` | `6.8` | Stellar locus zeropoint. |
| `D15)` | `maglim_gal` | `float` | `5` | Galaxy magnitude limit. |
| `D16)` | `maglim_star` | `float` | `2` | Star magnitude limit. |
| `D17)` | `nneighb` | `int` | `0` | Number of neighbours. |
| `D18)` | `max_proc` | `int` | `0` | Maximum number of processes. |
| `D19)` | `min_dist` | `float` | `0.0` | Minimum distance. |
| `D20)` | `min_dist_block` | `float` | `min_dist / 3` | Minimum distance for blocking. |
| `D21)` | `srclist` | `str` | `""` | Source list file. |
| `D22)` | `srclistrad` | `float` | `-1` | Source list radius. |

#### Section E -- GALFITM Options

| Code | Attribute | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `E00)` | `galexe` | `str` | `""` | GALFITM executable path. |
| `E01)` | `batch` | `str` | `""` | Batch system type. |
| `E02)` | `obj` | `str` | `""` | Object selection. |
| `E03)` | `galfit_out` | `str` | `""` | GALFIT output file name. |
| `E04)` | `psf` | `str` | `""` | PSF file path. |
| `E05)` | `mask` | `str` | `""` | Mask file path. |
| `E06)` | `constr` | `str` | `""` | Constraint file path. |
| `E07)` | `convbox` | `int` | `0` | Convolution box size. |
| `E08)` | `zp` | `float` | `0.0` | Magnitude zeropoint. |
| `E09)` | `platescl` | `float` | `0.0` | Plate scale. |
| `E10)` | `expt` | `float` | `0.0` | Exposure time. |
| `E11)` | `conmaxre` | `float` | `0.0` | Maximum half-light radius constraint. |
| `E12)` | `conminm` | `float` | `0.0` | Minimum magnitude constraint. |
| `E13)` | `conmaxm` | `float` | `0.0` | Maximum magnitude constraint. |
| `E14)` | `conminn` | `float` | `0.2` | Minimum Sersic index constraint. |
| `E15)` | `conmaxn` | `float` | `8.0` | Maximum Sersic index constraint. |
| `E16)` | `nice` | `int` | `0` | Set to `1` if `"nice"`. |
| `E17)` | `version` | `float` | `4.4` | GALFIT version number. |
| `E18)` | `gal_output` | `str` | `""` | GALFIT output format. |
| `E19)` | `gal_kill_time` | `float` or `str` | `0.0` | Time limit for GALFIT process. |
| `E20)` | `cheb` | `list` | `[0]*7` | Chebyshev polynomial orders for (x, y, mag, re, n, q, pa). |
| `E21)` | `galfit_out_path` | `str` | `""` | GALFIT output directory path. |
| `E22)` | `do_restrict` | `int` | `0` | Set to `1` if `"restrict"`. Restricts polynomial degree based on bad pixels. |
| `E23)` | `restrict_frac_primary` | `float` | `20.0` | Bad-pixel fraction threshold (percent) for restricting polynomial degree. |
| `E24)` | `mindeg` | `int` | `1` | Minimum polynomial degree. |

#### Section F -- BD Decomposition / Combine

| Code | Attribute | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `F00)` | `dobd` / `docombine` | `int` | `0` | BD decomposition or combination toggle. Interpretation depends on whether `G00)` is present in the file. |
| `F01)` | `cheb_b` / `docombinebd` | `list` / `int` | `[-1]*7` / `0` | Bulge Chebyshev orders or BD combination toggle. |
| `F02)` | `cheb_d` / `cat` | `list` / `str` | `[-1]*7` / `""` | Disk Chebyshev orders or catalog path. |
| `F03)` | `bd_label` | `str` | `" "` | BD label. |
| `F04)` | `bd_srclist` | `str` | `""` | BD source list file. |
| `F05)` | `bd_srclistrad` | `float` | `0.0` | BD source list radius. |
| `F06)` | `bd_maglim` | `float` | `99.0` | BD magnitude limit. |
| `F07)` | `gal_output_bd` | `str` | `""` | BD GALFIT output path. |
| `F08)` | `bd_hpc` | `int` | `0` | Set to `1` if `"HPC"`. |
| `F09)` | `bd_hpc_path` | `str` | `" "` | HPC path for BD. |
| `F10)` | `bd_psf_corr` | `list[str]` | `["", ""]` | PSF correction files. |

#### Section G -- Alternative Combine Section

| Code | Attribute | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `G00)` | `docombine` | `int` | `0` | Set to `1` if `"execute"`. |
| `G01)` | `docombinebd` | `int` | `0` | BD combination toggle. |
| `G02)` | `cat` | `str` | `""` | Catalog path. |

### Example

```python
from galfitx.read_setup import read_setup

setup = read_setup("setup_gALAPAGOS.ini")

print(setup.files)          # image file list path
print(setup.outdir)         # output directory
print(setup.dosex)          # run SExtractor?
print(setup.stampsize)      # stamp size factor
print(setup.skyscl)         # sky scale
print(setup.cheb)           # Chebyshev orders
```

---

## `read_image_files`

```python
read_image_files(setup, save_folder, silent=False, nocheck_read=None)
```

Read multi-band image file lists from the file specified in `setup.files`. Supports single-band (4 or 5 columns) and multi-band (6 columns) formats. For multi-band data, the setup file list is copied to `save_folder` and all image paths are validated.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `setup` | `Setup` | *(required)* | A `Setup` object returned by `read_setup`. The `setup.files` attribute must point to a valid file list. |
| `save_folder` | `str` | *(required)* | Directory where the multi-band file lists will be copied. Created if it does not exist. |
| `silent` | `bool` | `False` | If `False`, print informational messages during processing. |
| `nocheck_read` | `bool` or `None` | `None` | If truthy, skip the image-existence check. Useful for testing or when images are on remote storage. |

### Returns

`None` -- Populates the `setup` object in-place with the following new attributes (multi-band only):

| Attribute | Type | Description |
|-----------|------|-------------|
| `setup.images` | `np.ndarray` (2D, str) | Image paths indexed by `[band, tile]`. |
| `setup.weights` | `np.ndarray` (2D, str) | Weight map paths indexed by `[band, tile]`. |
| `setup.sigmaps` | `np.ndarray` (2D, str) | Sigma map paths (or `"none"`). |
| `setup.sigflags` | `np.ndarray` (1D, int) | Flag indicating whether sigma maps are present per band. |
| `setup.outpath` | `np.ndarray` (2D, str) | Output paths per band and tile. |
| `setup.outpath_band` | `np.ndarray` (2D, str) | Output paths with band labels appended. |
| `setup.outpre` | `np.ndarray` (2D, str) | Output prefixes per band and tile. |
| `setup.nband` | `int` | Number of bands. |
| `setup.stamp_pre` | `list[str]` | Band labels. |
| `setup.wavelength` | `list[float]` | Wavelengths per band. |
| `setup.mag_offset` | `list[float]` | Magnitude offsets per band. |
| `setup.zp` | `list[float]` | Zeropoints per band. |
| `setup.expt` | `list[float]` | Exposure times per band. |

### File List Formats

**Single-band (4 columns):**

```
image_path  weight_path  output_path  output_prefix
```

**Single-band with sigma maps (5 columns):**

```
image_path  weight_path  sigma_path  output_path  output_prefix
```

**Multi-band (6 columns):**

```
band  wavelength  mag_offset  filelist_path  zeropoint  exptime
```

The first line is used for SExtractor detection; subsequent lines are for measurement bands. Each `filelist_path` points to a separate file with 2 or 3 columns:

```
image_path  weight_path  [sigma_path]
```

### Raises

| Exception | Condition |
|-----------|-----------|
| `RuntimeError` | GALFIT version < 4.0 with multi-band data. |
| `RuntimeError` | Mismatched tile count between bands. |
| `RuntimeError` | Image or weight files not found on disk. |

### Example

```python
from galfitx.read_setup import read_setup, read_image_files

setup = read_setup("setup.ini")
read_image_files(setup, save_folder="./work/", silent=False)

# Access multi-band image paths
print(setup.nband)              # number of bands
print(setup.images[0, 0])       # SExtractor detection image for tile 0
print(setup.images[1, 0])       # Band 1 measurement image for tile 0
print(setup.zp)                 # zeropoints per band
```
