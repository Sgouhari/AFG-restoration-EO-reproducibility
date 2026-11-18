# Reproducibility code for the Afghanistan restoration monitoring analysis

This repository contains the analytical code supporting the manuscript **“Monitoring restoration interventions using Earth observation in a post-conflict forest and watershed programme in Afghanistan.”**

The code follows the final analysis used in the manuscript. It covers Sentinel-2 processing, polygon-level indicator extraction, reference matching, BACI analysis, robustness checks, MSAVI2 sensitivity analysis, and hydro-climatic context.

## Contents

The public code package contains:

```text
README.md
requirements.txt

scripts/
├── 01_sentinel2_ndvi_bsi_gee.js
├── 02_ndvi_bsi_polygon_processing.py
├── 03_reference_matching.py
├── 04_build_baci_panel.py
├── 05_baci_primary_analysis.py
├── 06_baci_robustness.py
├── 07_msavi2_gee.js
├── 08_msavi2_sensitivity.py
├── 09_chirps_monthly_gee.js
└── 10_spi_context.py
```

Keep the ten scripts inside the `scripts/` folder. The Python scripts use the parent of this folder as the repository root.

## Analysis covered by the code

The final BACI sample contains:

- 19 treatment polygons
- 39 unique reference polygons
- 58 unique polygons
- 17 anonymised site groups (`SG01`–`SG17`)
- 42 treatment-reference links
- annual observations from 2017 to 2024
- peak season composites for June to August
- late season composites for September to October

The BACI periods are:

```text
Baseline:             2017–2018
Implementation:       2019–2021
Post-implementation:  2022–2024
```

The hydro-climatic analysis also uses 1981–2018 as a long-term climate reference period.

## Script order

| Step | Script | Main task |
|---|---|---|
| 1 | `01_sentinel2_ndvi_bsi_gee.js` | Create seasonal Sentinel-2 NDVI and BSI composites for 2017–2024. |
| 2 | `02_ndvi_bsi_polygon_processing.py` | Reproject the rasters to EPSG:32642 at 10 m and extract polygon statistics. |
| 3 | `03_reference_matching.py` | Prepare terrain variables, calculate baseline-informed matching scores, rank references, and create the final matched sample. |
| 4 | `04_build_baci_panel.py` | Build the final 58-polygon BACI panel. |
| 5 | `05_baci_primary_analysis.py` | Run the primary mixed-effects BACI models, fixed-effects checks, and baseline diagnostic. |
| 6 | `06_baci_robustness.py` | Run the leave-one-post-year-out and 17-site-group robustness analyses. |
| 7 | `07_msavi2_gee.js` | Create seasonal MSAVI2 composites. |
| 8 | `08_msavi2_sensitivity.py` | Extract MSAVI2 for the final polygons and run the BACI sensitivity analysis. |
| 9 | `09_chirps_monthly_gee.js` | Extract monthly CHIRPS rainfall for the seven project districts. |
| 10 | `10_spi_context.py` | Calculate SPI-12 and summarise the hydro-climatic periods used in the manuscript. |

## Main input data

The workflow uses the following source data:

- Sentinel-2 Surface Reflectance Harmonized: `COPERNICUS/S2_SR_HARMONIZED`
- CHIRPS Daily precipitation: `UCSB-CHG/CHIRPS/DAILY`
- Copernicus DEM GLO-30
- seven project district boundaries
- candidate treatment and reference polygons

The local Python scripts expect the following working folders:

```text
data/
├── sentinel2_exports/
├── msavi2_exports/
├── dem_tiles/
├── district_boundaries_utm42n/
└── spatial/
    └── candidate_polygons_utm42n.shp

outputs/
```

The candidate polygon layer must be in WGS 84 / UTM Zone 42N (`EPSG:32642`) and contain:

```text
POLYGON_ID
SITE_GROUP
SITE_ROLE
```

`SITE_ROLE` should identify each polygon as `treatment` or `reference`.

The seven projected district boundary files expected by the ArcPy scripts are:

```text
Bamyan_district_Bamyan.shp
Sayghan_district_Bamyan.shp
Miramor_district_Daykundi.shp
Shahristan_district_Daykundi.shp
HazarSumuch_district_Takhar.shp
Kalfagan_district_Takhar.shp
Rustaq_district_Takhar.shp
```

Some filenames retain the original working spellings. Manuscript labels use **Saighan** and **Daikundi**.

## Google Earth Engine exports

Scripts 01, 07, and 09 run in the Google Earth Engine JavaScript Code Editor.

Before running them, replace:

```javascript
var GEE_ASSET_ROOT = 'projects/YOUR_GEE_PROJECT/assets/';
```

with the Earth Engine asset location containing the seven district boundaries.

After Script 01 finishes, place the downloaded NDVI and BSI GeoTIFFs in:

```text
data/sentinel2_exports/
```

After Script 07 finishes, place the downloaded MSAVI2 GeoTIFFs in:

```text
data/msavi2_exports/
```

After Script 09 finishes, place the downloaded CHIRPS table at:

```text
outputs/chirps/AFG_CHIRPS_monthly_7districts_1981_2025.csv
```

## Reference matching

Reference matching uses pre-intervention information only. The matching score uses 2017–2018 NDVI and BSI, elevation, slope, northness, eastness, polygon area, and centroid distance. Post-implementation NDVI and BSI are not used for matching.

The final sample retains the two lowest-scoring references for each treatment plus three prespecified additional reference polygons.

The three additional references are listed directly in `03_reference_matching.py` using anonymous site-group and polygon IDs (`SG05_R03`, `SG06_R03`, and `SG13_R03`).

Do not publish the private crosswalk linking anonymous IDs to original local site names.

## Statistical analysis

The primary mixed-effects BACI model is fitted separately for peak and late NDVI and BSI:

```text
outcome ~ treat * C(phase_3, Treatment('baseline'))
```

The model includes a site-group random intercept and a polygon-level variance component. Maximum-likelihood estimation uses the `lbfgs` optimiser.

Script 05 also runs fixed-effects checks and a two-year baseline diagnostic. Model warnings are saved in the diagnostics output.

Script 06 repeats the primary model after excluding 2022, 2023, and 2024 one at a time. It also calculates one baseline-to-post BACI contrast for each of the 17 site groups and summarises those contrasts using a one-sample t test, Wilcoxon signed-rank test, and exact binomial sign test.

The 17-site-group analysis is a robustness check. It does not fit a temporal autocorrelation model.

## MSAVI2 sensitivity analysis

MSAVI2 uses Sentinel-2 B4 and B8. Both bands are multiplied by `0.0001` before the MSAVI2 calculation because the equation contains the constant term 1.

The MSAVI2 analysis uses the same final polygons, years, seasons, and BACI phases as the NDVI analysis.

## Hydro-climatic context

Script 09 creates monthly CHIRPS rainfall estimates for all seven project districts from 1981 to 2025.

Script 10 calculates SPI-12 from 12-month precipitation accumulation. Gamma distributions are fitted separately for each calendar month using the 1981–2025 record. Manuscript summaries use data through 2024.

The reported periods are:

```text
Long-term climate reference: 1981–2018
BACI baseline:               2017–2018
Implementation:              2019–2021
Post-implementation:         2022–2024
```

Rainfall and SPI are used as contextual information and are not included as covariates in the BACI models.

## Data access

Sentinel-2, CHIRPS, and Copernicus DEM are public source datasets.

The treatment and reference polygon geometry may be subject to project, partner, or location-sensitivity restrictions. Anonymous polygon IDs do not hide the coordinates in a spatial file. The real polygon layer and the private site-name crosswalk should only be shared where permission allows.

The code can still be used with an authorised polygon layer that follows the field structure described above.

## Software

The workflow uses:

```text
Google Earth Engine JavaScript Code Editor
ArcGIS Pro
ArcPy
Spatial Analyst
Python
NumPy
pandas
SciPy
statsmodels
```

`requirements.txt` lists the Python packages that are installed separately from ArcGIS Pro. ArcPy is supplied with ArcGIS Pro and is not included in `requirements.txt`.

Exact package versions can be added if they are available from the analysis environment.

## Checks included in the scripts

The scripts include checks for the main analytical structure, including:

- 19 treatment polygons
- 39 unique reference polygons
- 58 unique polygons
- 17 site groups
- 42 treatment-reference links
- 928 polygon-year-season observations
- complete 2017–2024 peak and late seasonal coverage
- duplicate polygon-year-season records
- consistency between the primary BACI panel and the MSAVI2 sensitivity panel

Script 06 also checks that its full mixed-model estimates match the corresponding estimates produced by Script 05.

## Note on the public code package

These public scripts were prepared from the final analysis workflow used during manuscript revision and were cleaned for sharing. Python syntax checks were completed for the Python scripts during preparation. A full clean rerun of all ten public scripts was not completed after the final code consolidation because several stages require new Earth Engine exports and long raster-processing runs.

The scripts include internal QA checks so that users running the workflow can identify differences in sample size, missing data, model completion, or expected analytical structure.

## Citation

Please cite the associated manuscript and the archived code release. Add the final repository DOI when available.

```text
[Authors]. [Year]. Reproducibility code for “Monitoring restoration interventions
using Earth observation in a post-conflict forest and watershed programme in
Afghanistan”. [Repository]. [DOI].
```
