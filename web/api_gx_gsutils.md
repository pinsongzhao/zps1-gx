# Helper Utilities (`galfitx/gx_gsutils.py`)

Helper utilities for preparing GalfitS inputs. Provides unit conversions, conversion factor calculations, Union-Find grouping for co-fitting objects, and multi-band image stacking.

---

## `effective_wave`

A dictionary mapping filter/instrument names (strings) to their effective wavelengths in Angstroms. This is the same dictionary found in `create_setup_gs.py`. Covers GALEX, Pan-STARRS, SDSS, DESI, HSC, DECam, Subaru FOCAS, JWST NIRCam/MIRI, HST ACS/WFC3, WISE, Spitzer-IRAC, 2MASS, Keck, Swift UVOT, XMM-OM, and CSST.

---

## `Fnu_to_Fl`

Convert flux density from millijanskys (mJy) to erg/s/cm^2/Angstrom.

The conversion formula is:
```
Fl = Fnu * 1e-26 * (c * 1e13) / lambda^2
```
where `c = 2.9979246e5 km/s` and `1 mJy = 1e-26 erg/s/cm^2/Hz`.

```python
Fnu_to_Fl(Fnu, lambd)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `Fnu` | `float` or `np.ndarray` | (required) | Flux density in millijanskys (mJy). |
| `lambd` | `float` or `np.ndarray` | (required) | Wavelength in Angstroms. |

**Returns**

| Type | Description |
|---|---|
| `float` or `np.ndarray` | Flux density in erg/s/cm^2/A. Same shape as input. |

---

## `ABmag_to_covf`

Convert an AB magnitude zeropoint to a conversion factor (the inverse of the flux density in erg/s/cm^2/A). This factor can be multiplied by instrumental counts to yield physical flux.

The steps are:
1. Convert AB magnitude to flux density in mJy: `flux_mJy = 10^(-0.4 * mzp_AB) * 3631 * 1000`
2. Convert mJy to erg/s/cm^2/A using `Fnu_to_Fl`
3. Return the reciprocal

```python
ABmag_to_covf(mzp_AB, wave)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `mzp_AB` | `float` or `np.ndarray` | (required) | AB magnitude zeropoint(s). |
| `wave` | `float` or `np.ndarray` | (required) | Wavelength(s) in Angstroms. Should be broadcastable with `mzp_AB`. |

**Returns**

| Type | Description |
|---|---|
| `float` or `np.ndarray` | Conversion factor(s) with units of (erg/s/cm^2/A)^-1. |

---

## `calconvfactor`

Calculate conversion factors from instrumental counts to physical flux (erg/s/cm^2/A) for a set of filters. Iterates over zeropoint/wavelength pairs and calls `ABmag_to_covf` for each.

```python
calconvfactor(zpab_list, wave_list, image_stamp_list="none", filter_list="none")
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `zpab_list` | `list[float]` | (required) | AB magnitude zeropoints for each filter. |
| `wave_list` | `list[float]` | (required) | Central wavelengths in Angstroms for each filter. |
| `image_stamp_list` | `str` or `list[str]` | `"none"` | Paths to image stamp files (currently unused in the active code path). |
| `filter_list` | `str` or `list[str]` | `"none"` | Filter labels (currently unused in the active code path). |

**Returns**

| Type | Description |
|---|---|
| `np.ndarray` | 1D array of conversion factors with units of (erg/s/cm^2/A)^-1, one per filter. |

**Notes**

The commented-out section in the source shows an extended version that reads image headers to derive zeropoints for specific instruments (JWST NIRCam/MIRI, HST ACS/WFC3). This is kept for future extension.

---

## `Union_Set`

Group objects into connected components using a Union-Find algorithm. Given a dictionary mapping each object index to its list of co-fitting objects, returns groups of objects that should be fitted together.

```python
Union_Set(objects)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `objects` | `dict[int, list]` | (required) | Dictionary mapping each object index to a list of co-fitting object indices. For example, `{0: [1, 2], 1: [0], 2: [0, 3], 3: [2]}` means objects 0, 1, 2, and 3 are all connected. |

**Returns**

| Type | Description |
|---|---|
| `list[set]` | Sorted list of groups (sets of object indices). Sorted by group size (largest first), then by member values. |

**Example**

```python
>>> Union_Set({0: [1, 2], 1: [0], 2: [0, 3], 3: [2], 4: [5], 5: [4]})
[{0, 1, 2, 3}, {4, 5}]
```

---

## `stackimg`

Multi-band image stacking with support for inverse-variance weighting or SNR averaging. Reads science, sigma, and weight images from FITS files, stacks them, and optionally saves the results.

```python
stackimg(
    imglist,
    siglist,
    whtlist,
    mode="whtm",
    savepath=["./detection.fits", "detection_wht.fits"],
    saveresult=True,
)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `imglist` | `list[str]` | (required) | List of paths to science image FITS files. |
| `siglist` | `list[str]` | (required) | List of paths to sigma/noise image FITS files. |
| `whtlist` | `list[str]` | (required) | List of paths to weight map FITS files. |
| `mode` | `"whtm"` or `"snr"` | `"whtm"` | Stacking mode. `"whtm"` = inverse-variance weighted mean, `"snr"` = SNR average. |
| `savepath` | `list[str]` | `["./detection.fits", "detection_wht.fits"]` | Output file paths: `[science_output, weight_output]`. |
| `saveresult` | `bool` | `True` | If True, write the stacked images to disk. |

**Returns**

| Type | Description |
|---|---|
| `list[np.ndarray or None]` | `[stacked_science, stacked_weight]`. In `"snr"` mode, `stacked_weight` is `None`. |

**Mode Details**

- **`"whtm"` (inverse-variance weighted):** `stack = sum(sci * wht) / sum(wht)`. This is the standard noise-equalized stacking method. Pixels with zero total weight are set to 0.
- **`"snr"` (SNR average):** `stack = sum(sci / sig) / sqrt(n_bands)`. Each band is divided by its sigma image before averaging. This produces a detection image already weighted by noise.

**Raises**

- `ValueError` -- If any image has a shape mismatch with the template (first image in `imglist`).
