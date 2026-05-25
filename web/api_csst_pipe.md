# csst_pipe -- CSST Pipeline Utilities

Module: `galfitx.csst_pipe`

Utility functions for the CSST (Chinese Space Station Telescope) data pipeline. Currently provides table-to-FITS conversion with rich metadata annotations for standard SExtractor-style source catalogs.

---

## tab2fits

```python
from galfitx.csst_pipe import tab2fits
```

```
tab2fits(
    outtab: Table,
    output: str = "output_cat.fits",
) -> Table
```

Converts an `astropy.table.Table` (typically from source detection) to a FITS file with descriptive metadata. For every column, the function looks up a human-readable description and a physical unit from internal dictionaries, then creates a new `Column` with that metadata attached. The resulting table is written as a FITS binary table via `csst_dadel.utils.table_to_hdu`.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `outtab` | `astropy.table.Table` | *required* | Source catalog table. Columns should use standard SExtractor-style names (see below). |
| `output` | `str` | `"output_cat.fits"` | Output FITS file path. |

### Returns

`astropy.table.Table` -- the table with descriptions and units attached, also written to disk.

### Supported columns

The function recognises the following column names and annotates them automatically. Columns not listed below receive the column name as the description and an empty unit string.

| Column name | Description | Unit |
|---|---|---|
| `label` | Source label/ID | -- |
| `xcentroid` | X centroid position | `pix` |
| `ycentroid` | Y centroid position | `pix` |
| `bbox_xmin` | Minimum x pixel index within the minimal bounding box containing the source segment | -- |
| `bbox_xmax` | Maximum x pixel index within the minimal bounding box containing the source segment | -- |
| `bbox_ymin` | Minimum y pixel index within the minimal bounding box containing the source segment | -- |
| `bbox_ymax` | Maximum y pixel index within the minimal bounding box containing the source segment | -- |
| `area` | Total unmasked area of the source | `pix^2` |
| `semimajor_sigma` | 1-sigma along semimajor axis | `pix` |
| `semiminor_sigma` | 1-sigma along semiminor axis | `pix` |
| `orientation` | Position angle (degree) | `deg` |
| `eccentricity` | Eccentricity (0=circle, 1=line) | -- |
| `min_value` | Minimum pixel value within the source segment | `counts` |
| `max_value` | Maximum pixel value within the source segment | `counts` |
| `local_background` | Local background value (per pixel) estimated using a rectangular annulus aperture around the source | `counts` |
| `segment_flux` | Sum of the unmasked data values within the source segment | `counts` |
| `segment_fluxerr` | Error in segment flux | `counts` |
| `kron_flux` | Flux in the Kron aperture | `counts` |
| `kron_fluxerr` | Error in Kron flux | `counts` |
| `background_centroid` | Background at centroid | `counts` |
| `elongation` | Ratio of the semimajor and semiminor axes | -- |
| `ellipticity` | Ellipticity (1 - b/a) | -- |
| `fwhm` | Full width at half maximum | `pix` |
| `kron_radius` | Kron radius | -- |
| `flux_radius` | Half-light radius | `pix` |
| `ra` | Right ascension (J2000) | `deg` |
| `dec` | Declination (J2000) | `deg` |
| `cxx` | Second moment xx | `pix^-2` |
| `cxy` | Second moment xy | `pix^-2` |
| `cyy` | Second moment yy | `pix^-2` |
| `gini` | Gini coefficient | -- |
| `segment_area` | Area of segmentation region | `pix^2` |
| `mag_auto` | Automatic aperture magnitude | `mag` |

### Notes

- Columns with recognised names receive their standard description and unit. Unrecognised columns keep their column name as the description and an empty (`""`) unit.
- The output FITS is overwritten if it already exists (`overwrite=True`).
- A short summary is printed to stdout: `Table with N columns written to <output>`.
