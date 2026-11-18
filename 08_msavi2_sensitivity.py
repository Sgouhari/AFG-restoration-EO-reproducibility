# -*- coding: utf-8 -*-
"""
MSAVI2 polygon extraction and BACI sensitivity analysis.

This script reproduces the soil-background sensitivity analysis reported in
Supplementary Table S6 of the Afghanistan restoration monitoring study.

Workflow
--------
1. Read the final 58-polygon BACI panel produced by 04_build_baci_panel.py.
2. Select exactly those polygons from the anonymised UTM Zone 42N polygon layer.
3. Reproject the seasonal MSAVI2 GeoTIFFs from 07_msavi2_gee.js to
   EPSG:32642 at 10 m using bilinear resampling.
4. Extract polygon-level MSAVI2 statistics.
5. Verify an exact polygon-year-season match with the final BACI panel.
6. Refit the primary mixed-effects BACI model with MSAVI2 as the response.
7. Calculate one baseline-to-post MSAVI2 BACI contrast per site group and
   summarise the number of positive contrasts.

Only the vegetation index changes relative to the primary NDVI analysis.
The final polygons, site groups, years, seasons and BACI phase definitions
are inherited from the final BACI panel.

Requirements
------------
ArcGIS Pro Python environment with:
    arcpy
    Spatial Analyst
    numpy
    pandas
    statsmodels

Inputs
------
data/msavi2_exports/
data/district_boundaries_utm42n/
data/spatial/candidate_polygons_utm42n.shp
outputs/baci/baci_master_panel_2017_2024.csv
"""

from pathlib import Path
import traceback
import warnings

import arcpy
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from arcpy.sa import ZonalStatisticsAsTable


# -----------------------------------------------------------------------------
# 1. Repository paths
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_ROOT = SCRIPT_DIR.parent

DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"

MSAVI2_RAW_DIR = DATA_DIR / "msavi2_exports"
DISTRICT_AOI_DIR = DATA_DIR / "district_boundaries_utm42n"
POLYGON_FC = DATA_DIR / "spatial" / "candidate_polygons_utm42n.shp"

BACI_MASTER = OUTPUT_DIR / "baci" / "baci_master_panel_2017_2024.csv"

MSAVI2_DIR = OUTPUT_DIR / "msavi2"
WORK_DIR = OUTPUT_DIR / "_working_msavi2"

MSAVI2_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

OUT_PANEL = MSAVI2_DIR / "msavi2_polygon_panel_2017_2024.csv"
OUT_EXTRACTION_QA = MSAVI2_DIR / "msavi2_extraction_qa.csv"
OUT_MODEL_RESULTS = MSAVI2_DIR / "msavi2_baci_results.csv"
OUT_MODEL_DIAGNOSTICS = MSAVI2_DIR / "msavi2_model_diagnostics.csv"
OUT_SITE_CONTRASTS = MSAVI2_DIR / "msavi2_site_group_contrasts.csv"
OUT_DIRECTION = MSAVI2_DIR / "msavi2_site_group_direction_summary.csv"
OUT_QA = MSAVI2_DIR / "msavi2_sensitivity_qa.txt"


# -----------------------------------------------------------------------------
# 2. Analysis settings
# -----------------------------------------------------------------------------

TARGET_EPSG = 32642
TARGET_CELL_SIZE = 10

VALID_YEARS = list(range(2017, 2025))
VALID_SEASONS = ["peak", "late"]

BASELINE_YEARS = [2017, 2018]
POST_YEARS = [2022, 2023, 2024]

PHASE_ORDER = ["baseline", "implementation", "post_implementation"]

EXPECTED_TREATMENTS = 19
EXPECTED_REFERENCES = 39
EXPECTED_POLYGONS = 58
EXPECTED_SITE_GROUPS = 17
EXPECTED_ROWS = 928
EXPECTED_ROWS_PER_SEASON = 464
EXPECTED_OBS_PER_POLYGON = 16
EXPECTED_RAW_RASTERS = 7 * len(VALID_YEARS) * len(VALID_SEASONS)

POLYGON_ID_FIELD = "POLYGON_ID"

STRICT_CHECKS = True

DISTRICTS = [
    {
        "province": "Bamyan",
        "district": "Bamyan",
        "file_district": "Bamyan",
        "aoi_file": "Bamyan_district_Bamyan.shp",
    },
    {
        "province": "Bamyan",
        "district": "Saighan",
        "file_district": "Saighan",
        "aoi_file": "Sayghan_district_Bamyan.shp",
    },
    {
        "province": "Daikundi",
        "district": "Miramor",
        "file_district": "Miramor",
        "aoi_file": "Miramor_district_Daykundi.shp",
    },
    {
        "province": "Daikundi",
        "district": "Shahristan",
        "file_district": "Shahristan",
        "aoi_file": "Shahristan_district_Daykundi.shp",
    },
    {
        "province": "Takhar",
        "district": "Hazar Sumuch",
        "file_district": "HazarSumuch",
        "aoi_file": "HazarSumuch_district_Takhar.shp",
    },
    {
        "province": "Takhar",
        "district": "Kalfagan",
        "file_district": "Kalfagan",
        "aoi_file": "Kalfagan_district_Takhar.shp",
    },
    {
        "province": "Takhar",
        "district": "Rustaq",
        "file_district": "Rustaq",
        "aoi_file": "Rustaq_district_Takhar.shp",
    },
]


# -----------------------------------------------------------------------------
# 3. ArcGIS environment and helpers
# -----------------------------------------------------------------------------

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True
TARGET_SR = arcpy.SpatialReference(TARGET_EPSG)


def safe_delete(path):
    try:
        if path and arcpy.Exists(str(path)):
            arcpy.management.Delete(str(path))
    except Exception:
        pass


def reset_raster_environment():
    arcpy.env.snapRaster = None
    arcpy.env.cellSize = None
    arcpy.env.mask = None
    arcpy.env.extent = None
    arcpy.env.outputCoordinateSystem = None


def require_file(path, label):
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_feature_class(path, label):
    if not arcpy.Exists(str(path)):
        raise FileNotFoundError(f"{label} not found: {path}")


def require_columns(df, columns, label):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {label}: {missing}")


def require_fields(feature_class, fields):
    existing = {field.name for field in arcpy.ListFields(str(feature_class))}
    missing = [field for field in fields if field not in existing]
    if missing:
        raise ValueError(
            f"Missing fields in {feature_class}: {missing}"
        )


def require_epsg(dataset, expected_epsg):
    sr = arcpy.Describe(str(dataset)).spatialReference
    if sr.factoryCode != expected_epsg:
        raise ValueError(
            f"{dataset} must use EPSG:{expected_epsg}; "
            f"found {sr.name} (WKID {sr.factoryCode})."
        )


def raster_property(raster, property_name):
    return arcpy.management.GetRasterProperties(
        str(raster),
        property_name,
    ).getOutput(0)


def raster_cell_area(raster):
    x = float(raster_property(raster, "CELLSIZEX"))
    y = float(raster_property(raster, "CELLSIZEY"))
    return abs(x * y)


def expected_raster_path(cfg, season, year):
    filename = (
        f"{cfg['province']}_{cfg['file_district']}"
        f"_MSAVI2_{season}_{year}.tif"
    )
    return MSAVI2_RAW_DIR / filename


def standardise_site_type(value):
    value = "" if pd.isna(value) else str(value).strip().lower()

    if value in {"treatment", "treated", "intervention"}:
        return "treatment"

    if value in {"control", "reference", "matched control", "matched reference"}:
        return "control"

    return value


def role_display(site_type):
    return "Treatment" if site_type == "treatment" else "Reference"


def extract_model_coefficient(fit, term, season, phase):
    if term not in fit.params.index:
        raise RuntimeError(
            f"Expected model term not found for {season}: {term}"
        )

    ci = fit.conf_int().loc[term]

    return {
        "model": "mixed_baci",
        "outcome": "MSAVI2",
        "season": season,
        "effect_phase": phase,
        "term": term,
        "coef": float(fit.params[term]),
        "se": float(fit.bse[term]),
        "t_or_z": float(fit.tvalues[term]),
        "p_value": float(fit.pvalues[term]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "n_obs": int(fit.nobs),
    }


def pass_fail(condition):
    return "PASS" if condition else "CHECK"


def round_numeric(df, digits=6):
    result = df.copy()
    columns = result.select_dtypes(include=[np.number]).columns
    result[columns] = result[columns].round(digits)
    return result


# -----------------------------------------------------------------------------
# 4. Validate the final BACI master
# -----------------------------------------------------------------------------

require_file(BACI_MASTER, "Final BACI master panel")
require_feature_class(POLYGON_FC, "Candidate polygon feature class")
require_fields(POLYGON_FC, [POLYGON_ID_FIELD])
require_epsg(POLYGON_FC, TARGET_EPSG)

master = pd.read_csv(
    BACI_MASTER,
    encoding="utf-8-sig",
).copy()

required_master = [
    "POLYGON_ID",
    "POLYGON_NUM_ID",
    "SITE_GROUP",
    "SITE_TYPE",
    "SITE_ROLE",
    "year",
    "season",
    "phase_3",
    "period_baci",
    "terrain_valid",
]

require_columns(master, required_master, "final BACI master")

master["POLYGON_ID"] = master["POLYGON_ID"].astype(str).str.strip()
master["SITE_GROUP"] = master["SITE_GROUP"].astype(str).str.strip().str.upper()
master["SITE_TYPE"] = master["SITE_TYPE"].apply(standardise_site_type)
master["SITE_ROLE"] = master["SITE_ROLE"].astype(str).str.strip().str.lower()
master["season"] = master["season"].astype(str).str.strip().str.lower()
master["phase_3"] = master["phase_3"].astype(str).str.strip().str.lower()
master["period_baci"] = master["period_baci"].astype(str).str.strip().str.lower()

for column in ["year", "POLYGON_NUM_ID", "terrain_valid"]:
    master[column] = pd.to_numeric(master[column], errors="coerce")

master = master.dropna(
    subset=["year", "POLYGON_NUM_ID", "terrain_valid"]
).copy()

master["year"] = master["year"].astype(int)
master["POLYGON_NUM_ID"] = master["POLYGON_NUM_ID"].astype(int)

master = master[
    (master["terrain_valid"] == 1)
    & master["year"].isin(VALID_YEARS)
    & master["season"].isin(VALID_SEASONS)
    & master["SITE_TYPE"].isin(["treatment", "control"])
].copy()

n_master_rows = len(master)
n_master_polygons = int(master["POLYGON_ID"].nunique())
n_master_sites = int(master["SITE_GROUP"].nunique())
n_master_treatments = int(
    master.loc[
        master["SITE_TYPE"] == "treatment",
        "POLYGON_ID",
    ].nunique()
)
n_master_references = int(
    master.loc[
        master["SITE_TYPE"] == "control",
        "POLYGON_ID",
    ].nunique()
)

master_duplicate_keys = int(
    master.duplicated(
        subset=["POLYGON_ID", "year", "season"]
    ).sum()
)

master_checks = {
    "master_rows": n_master_rows == EXPECTED_ROWS,
    "master_polygons": n_master_polygons == EXPECTED_POLYGONS,
    "master_treatments": n_master_treatments == EXPECTED_TREATMENTS,
    "master_references": n_master_references == EXPECTED_REFERENCES,
    "master_site_groups": n_master_sites == EXPECTED_SITE_GROUPS,
    "master_unique_keys": master_duplicate_keys == 0,
}

if STRICT_CHECKS and not all(master_checks.values()):
    failed = [
        key for key, value in master_checks.items()
        if not value
    ]
    raise RuntimeError(
        "Final BACI master failed MSAVI2 input checks: "
        f"{failed}"
    )

metadata_columns = [
    "POLYGON_ID",
    "POLYGON_NUM_ID",
    "SITE_GROUP",
    "SITE_TYPE",
    "SITE_ROLE",
]

if "province" in master.columns:
    metadata_columns.append("province")

if "district" in master.columns:
    metadata_columns.append("district")

final_meta = (
    master[metadata_columns]
    .drop_duplicates()
    .copy()
)

if final_meta["POLYGON_ID"].duplicated().any():
    raise RuntimeError(
        "A final BACI polygon has inconsistent metadata."
    )

meta_lookup = (
    final_meta
    .set_index("POLYGON_ID")
    .to_dict(orient="index")
)

final_polygon_ids = set(final_meta["POLYGON_ID"])


# -----------------------------------------------------------------------------
# 5. Select the exact final 58 polygons
# -----------------------------------------------------------------------------

working_polygons = WORK_DIR / "msavi2_final58_polygons.shp"
safe_delete(working_polygons)

candidate_layer = "msavi2_candidate_layer"
safe_delete(candidate_layer)

arcpy.management.MakeFeatureLayer(
    str(POLYGON_FC),
    candidate_layer,
)

field = arcpy.AddFieldDelimiters(
    str(POLYGON_FC),
    POLYGON_ID_FIELD,
)

escaped_ids = [
    "'" + polygon_id.replace("'", "''") + "'"
    for polygon_id in sorted(final_polygon_ids)
]

selection_sql = (
    f"{field} IN ({','.join(escaped_ids)})"
)

arcpy.management.SelectLayerByAttribute(
    candidate_layer,
    "NEW_SELECTION",
    selection_sql,
)

selected_count = int(
    arcpy.management.GetCount(candidate_layer)[0]
)

if selected_count != EXPECTED_POLYGONS:
    raise RuntimeError(
        "The anonymised polygon layer does not contain the exact "
        f"58-polygon BACI sample. Selected: {selected_count}"
    )

arcpy.management.CopyFeatures(
    candidate_layer,
    str(working_polygons),
)

safe_delete(candidate_layer)

selected_ids = set()

with arcpy.da.SearchCursor(
    str(working_polygons),
    [POLYGON_ID_FIELD],
) as cursor:
    for row in cursor:
        selected_ids.add(str(row[0]).strip())

if selected_ids != final_polygon_ids:
    raise RuntimeError(
        "Selected polygon IDs do not exactly match the BACI master."
    )


# -----------------------------------------------------------------------------
# 6. Check MSAVI2 raster inputs
# -----------------------------------------------------------------------------

expected_rasters = []

for cfg in DISTRICTS:
    for year in VALID_YEARS:
        for season in VALID_SEASONS:
            expected_rasters.append(
                expected_raster_path(cfg, season, year)
            )

missing_rasters = [
    path for path in expected_rasters
    if not path.exists()
]

if missing_rasters:
    preview = "\n".join(str(path) for path in missing_rasters[:20])
    raise FileNotFoundError(
        "Required MSAVI2 GeoTIFFs are missing. "
        f"Missing {len(missing_rasters)} of "
        f"{EXPECTED_RAW_RASTERS} expected files.\n{preview}"
    )


# -----------------------------------------------------------------------------
# 7. Extract MSAVI2 for the final polygons
# -----------------------------------------------------------------------------

final_layer = "msavi2_final_layer"
safe_delete(final_layer)

arcpy.management.MakeFeatureLayer(
    str(working_polygons),
    final_layer,
)

all_rows = []
extraction_rows = []

for district_index, cfg in enumerate(DISTRICTS, start=1):
    aoi = DISTRICT_AOI_DIR / cfg["aoi_file"]

    require_feature_class(aoi, "District boundary")
    require_epsg(aoi, TARGET_EPSG)

    arcpy.management.SelectLayerByAttribute(
        final_layer,
        "CLEAR_SELECTION",
    )

    arcpy.management.SelectLayerByLocation(
        in_layer=final_layer,
        overlap_type="INTERSECT",
        select_features=str(aoi),
        selection_type="NEW_SELECTION",
    )

    district_polygon_count = int(
        arcpy.management.GetCount(final_layer)[0]
    )

    # Districts with no retained polygons do not contribute to the
    # MSAVI2 sensitivity analysis.
    if district_polygon_count == 0:
        continue

    district_fc = f"memory/msavi2_district_{district_index}"
    safe_delete(district_fc)

    arcpy.management.CopyFeatures(
        final_layer,
        district_fc,
    )

    district_ids = set()

    with arcpy.da.SearchCursor(
        district_fc,
        [POLYGON_ID_FIELD],
    ) as cursor:
        for row in cursor:
            district_ids.add(str(row[0]).strip())

    for year in VALID_YEARS:
        for season in VALID_SEASONS:
            raw_raster = expected_raster_path(
                cfg,
                season,
                year,
            )

            temp_raster = WORK_DIR / (
                f"tmp_{cfg['province']}_{cfg['file_district']}"
                f"_MSAVI2_{season}_{year}_utm42n.tif"
            )

            temp_table = (
                f"memory/msavi2_zs_{district_index}_{year}_{season}"
            )

            safe_delete(temp_raster)
            safe_delete(temp_table)
            reset_raster_environment()

            qa_row = {
                "province": cfg["province"],
                "district": cfg["district"],
                "year": year,
                "season": season,
                "source_raster": raw_raster.name,
                "expected_polygons": district_polygon_count,
                "extracted_polygons": 0,
                "projected_wkid": "",
                "cell_size_x": "",
                "cell_size_y": "",
                "status": "failed",
                "message": "",
            }

            try:
                arcpy.management.ProjectRaster(
                    in_raster=str(raw_raster),
                    out_raster=str(temp_raster),
                    out_coor_system=TARGET_SR,
                    resampling_type="BILINEAR",
                    cell_size=str(TARGET_CELL_SIZE),
                )

                if not arcpy.Exists(str(temp_raster)):
                    raise RuntimeError(
                        "Temporary projected raster was not created."
                    )

                desc = arcpy.Describe(str(temp_raster))
                wkid = desc.spatialReference.factoryCode

                cell_x = float(
                    raster_property(temp_raster, "CELLSIZEX")
                )
                cell_y = float(
                    raster_property(temp_raster, "CELLSIZEY")
                )

                if wkid != TARGET_EPSG:
                    raise RuntimeError(
                        f"Projected raster WKID is {wkid}, "
                        f"expected {TARGET_EPSG}."
                    )

                if not (
                    abs(cell_x - TARGET_CELL_SIZE) < 0.05
                    and abs(cell_y - TARGET_CELL_SIZE) < 0.05
                ):
                    raise RuntimeError(
                        "Projected raster does not have an "
                        f"approximately 10 m cell size: "
                        f"{cell_x}, {cell_y}"
                    )

                arcpy.env.snapRaster = str(temp_raster)
                arcpy.env.cellSize = str(temp_raster)

                ZonalStatisticsAsTable(
                    in_zone_data=district_fc,
                    zone_field=POLYGON_ID_FIELD,
                    in_value_raster=str(temp_raster),
                    out_table=temp_table,
                    ignore_nodata="DATA",
                    statistics_type="ALL",
                )

                table_fields = {
                    field.name.upper(): field.name
                    for field in arcpy.ListFields(temp_table)
                }

                required_zonal = [
                    POLYGON_ID_FIELD.upper(),
                    "COUNT",
                    "MIN",
                    "MAX",
                    "MEAN",
                ]

                missing_zonal = [
                    name for name in required_zonal
                    if name not in table_fields
                ]

                if missing_zonal:
                    raise RuntimeError(
                        "Zonal-statistics table is missing fields: "
                        f"{missing_zonal}"
                    )

                optional_fields = [
                    name for name in ["RANGE", "STD", "SUM"]
                    if name in table_fields
                ]

                read_upper = required_zonal + optional_fields
                read_actual = [
                    table_fields[name]
                    for name in read_upper
                ]

                cell_area = raster_cell_area(temp_raster)
                rows_this_raster = 0

                with arcpy.da.SearchCursor(
                    temp_table,
                    read_actual,
                ) as cursor:
                    for values in cursor:
                        record = dict(zip(read_upper, values))

                        polygon_id = str(
                            record[POLYGON_ID_FIELD.upper()]
                        ).strip()

                        if polygon_id not in district_ids:
                            raise RuntimeError(
                                "Unexpected polygon ID in zonal output: "
                                f"{polygon_id}"
                            )

                        meta = meta_lookup[polygon_id]
                        pixel_count = record.get("COUNT")

                        all_rows.append({
                            "POLYGON_ID": polygon_id,
                            "POLYGON_NUM_ID":
                                meta["POLYGON_NUM_ID"],
                            "SITE_GROUP":
                                meta["SITE_GROUP"],
                            "SITE_TYPE":
                                meta["SITE_TYPE"],
                            "SITE_ROLE":
                                meta["SITE_ROLE"],
                            "province":
                                cfg["province"],
                            "district":
                                cfg["district"],
                            "year":
                                year,
                            "season":
                                season,
                            "source_raster":
                                raw_raster.name,
                            "pixel_count":
                                pixel_count,
                            "pixel_area_m2":
                                (
                                    float(pixel_count) * cell_area
                                    if pixel_count is not None
                                    else np.nan
                                ),
                            "msavi2_min":
                                record.get("MIN"),
                            "msavi2_max":
                                record.get("MAX"),
                            "msavi2_range":
                                record.get("RANGE"),
                            "msavi2_mean":
                                record.get("MEAN"),
                            "msavi2_std":
                                record.get("STD"),
                            "msavi2_sum":
                                record.get("SUM"),
                        })

                        rows_this_raster += 1

                qa_row.update({
                    "extracted_polygons": rows_this_raster,
                    "projected_wkid": wkid,
                    "cell_size_x": cell_x,
                    "cell_size_y": cell_y,
                    "status":
                        (
                            "pass"
                            if rows_this_raster == district_polygon_count
                            else "check"
                        ),
                })

            except Exception as exc:
                qa_row["message"] = (
                    f"{exc} | {traceback.format_exc(limit=1)}"
                )

            finally:
                extraction_rows.append(qa_row)
                safe_delete(temp_table)
                safe_delete(temp_raster)
                reset_raster_environment()

    safe_delete(district_fc)


# -----------------------------------------------------------------------------
# 8. Build and verify the MSAVI2 polygon panel
# -----------------------------------------------------------------------------

msavi = pd.DataFrame(all_rows)

if msavi.empty:
    raise RuntimeError("No MSAVI2 polygon observations were extracted.")

msavi["year"] = pd.to_numeric(msavi["year"], errors="coerce")
msavi["msavi2_mean"] = pd.to_numeric(
    msavi["msavi2_mean"],
    errors="coerce",
)

keys = ["POLYGON_ID", "year", "season"]

duplicate_msavi_keys = int(
    msavi.duplicated(keys).sum()
)

n_rows = len(msavi)
n_polygons = int(msavi["POLYGON_ID"].nunique())
n_sites = int(msavi["SITE_GROUP"].nunique())
n_treatments = int(
    msavi.loc[
        msavi["SITE_TYPE"] == "treatment",
        "POLYGON_ID",
    ].nunique()
)
n_references = int(
    msavi.loc[
        msavi["SITE_TYPE"] == "control",
        "POLYGON_ID",
    ].nunique()
)

missing_msavi2 = int(msavi["msavi2_mean"].isna().sum())

invalid_msavi2 = int(
    (
        (msavi["msavi2_mean"] < -1)
        | (msavi["msavi2_mean"] > 1)
    ).sum()
)

polygon_completeness = (
    msavi.groupby(
        ["POLYGON_ID", "SITE_GROUP", "SITE_TYPE"],
        as_index=False,
    )
    .agg(
        observations=("msavi2_mean", "size"),
        nonmissing_msavi2=("msavi2_mean", "count"),
        min_year=("year", "min"),
        max_year=("year", "max"),
    )
)

polygon_completeness["complete"] = (
    (polygon_completeness["observations"] == EXPECTED_OBS_PER_POLYGON)
    & (
        polygon_completeness["nonmissing_msavi2"]
        == EXPECTED_OBS_PER_POLYGON
    )
    & (polygon_completeness["min_year"] == 2017)
    & (polygon_completeness["max_year"] == 2024)
)

master_keys = (
    master[keys]
    .drop_duplicates()
    .copy()
)

msavi_keys = (
    msavi[keys]
    .drop_duplicates()
    .copy()
)

key_comparison = master_keys.merge(
    msavi_keys,
    on=keys,
    how="outer",
    indicator=True,
)

key_mismatches = key_comparison[
    key_comparison["_merge"] != "both"
]

failed_extraction_jobs = sum(
    row["status"] != "pass"
    for row in extraction_rows
)

extraction_checks = {
    "rows": n_rows == EXPECTED_ROWS,
    "polygons": n_polygons == EXPECTED_POLYGONS,
    "treatments": n_treatments == EXPECTED_TREATMENTS,
    "references": n_references == EXPECTED_REFERENCES,
    "site_groups": n_sites == EXPECTED_SITE_GROUPS,
    "unique_polygon_year_season": duplicate_msavi_keys == 0,
    "no_missing_msavi2": missing_msavi2 == 0,
    "msavi2_range": invalid_msavi2 == 0,
    "all_polygons_complete":
        bool(polygon_completeness["complete"].all()),
    "exact_key_match_with_baci":
        len(key_mismatches) == 0,
    "all_used_raster_jobs_passed":
        failed_extraction_jobs == 0,
}

if STRICT_CHECKS and not all(extraction_checks.values()):
    failed = [
        key for key, value in extraction_checks.items()
        if not value
    ]
    raise RuntimeError(
        "MSAVI2 extraction failed QA checks: "
        f"{failed}"
    )

msavi = msavi.sort_values(
    ["SITE_GROUP", "SITE_TYPE", "POLYGON_ID", "year", "season"]
).reset_index(drop=True)

msavi.to_csv(
    OUT_PANEL,
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame(extraction_rows).to_csv(
    OUT_EXTRACTION_QA,
    index=False,
    encoding="utf-8-sig",
)


# -----------------------------------------------------------------------------
# 9. Join MSAVI2 to the authoritative BACI phase structure
# -----------------------------------------------------------------------------

baci_meta = master[
    [
        "POLYGON_ID",
        "year",
        "season",
        "SITE_GROUP",
        "SITE_TYPE",
        "SITE_ROLE",
        "phase_3",
        "period_baci",
        "terrain_valid",
    ]
].copy()

analysis = msavi.merge(
    baci_meta,
    on=keys,
    how="inner",
    suffixes=("_msavi", "_baci"),
    validate="one_to_one",
)

group_mismatch = (
    analysis["SITE_GROUP_msavi"].str.upper()
    != analysis["SITE_GROUP_baci"].str.upper()
)

type_mismatch = (
    analysis["SITE_TYPE_msavi"].apply(standardise_site_type)
    != analysis["SITE_TYPE_baci"].apply(standardise_site_type)
)

role_mismatch = (
    analysis["SITE_ROLE_msavi"].str.lower()
    != analysis["SITE_ROLE_baci"].str.lower()
)

if (
    group_mismatch.any()
    or type_mismatch.any()
    or role_mismatch.any()
):
    raise RuntimeError(
        "MSAVI2 extraction metadata do not match the final BACI panel."
    )

analysis["SITE_GROUP"] = (
    analysis["SITE_GROUP_baci"].str.upper()
)
analysis["SITE_TYPE"] = (
    analysis["SITE_TYPE_baci"].apply(standardise_site_type)
)
analysis["SITE_ROLE"] = (
    analysis["SITE_ROLE_baci"].str.lower()
)

analysis["phase_3"] = analysis["phase_3"].str.lower()

analysis["treat"] = (
    analysis["SITE_TYPE"] == "treatment"
).astype(int)

analysis["role_display"] = (
    analysis["SITE_TYPE"].apply(role_display)
)

analysis["phase_3"] = pd.Categorical(
    analysis["phase_3"],
    categories=PHASE_ORDER,
    ordered=True,
)


# -----------------------------------------------------------------------------
# 10. Primary mixed-effects BACI model with MSAVI2
# -----------------------------------------------------------------------------

model_rows = []
diagnostic_rows = []

for season in VALID_SEASONS:
    subset = analysis[
        (analysis["season"] == season)
        & analysis["msavi2_mean"].notna()
    ].copy()

    if len(subset) != EXPECTED_ROWS_PER_SEASON:
        raise RuntimeError(
            f"{season}: expected {EXPECTED_ROWS_PER_SEASON} "
            f"MSAVI2 observations, found {len(subset)}."
        )

    formula = (
        "msavi2_mean ~ "
        "treat * C(phase_3, Treatment('baseline'))"
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        model = smf.mixedlm(
            formula=formula,
            data=subset,
            groups=subset["SITE_GROUP"],
            re_formula="1",
            vc_formula={
                "polygon": "0 + C(POLYGON_ID)"
            },
        )

        fit = model.fit(
            method="lbfgs",
            reml=False,
            maxiter=500,
        )

    terms = [
        (
            "treat:C(phase_3, Treatment('baseline'))"
            "[T.implementation]",
            "implementation",
        ),
        (
            "treat:C(phase_3, Treatment('baseline'))"
            "[T.post_implementation]",
            "post_implementation",
        ),
    ]

    for term, phase in terms:
        model_rows.append(
            extract_model_coefficient(
                fit,
                term,
                season,
                phase,
            )
        )

    diagnostic_rows.append({
        "model": "mixed_baci",
        "outcome": "MSAVI2",
        "season": season,
        "status": "success",
        "converged": bool(fit.converged),
        "n_obs": int(fit.nobs),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "llf": float(fit.llf),
        "warning_count": len(caught_warnings),
        "warnings": " | ".join(
            str(item.message)
            for item in caught_warnings
        ),
    })

model_results = round_numeric(
    pd.DataFrame(model_rows)
)

model_diagnostics = round_numeric(
    pd.DataFrame(diagnostic_rows)
)

model_results.to_csv(
    OUT_MODEL_RESULTS,
    index=False,
    encoding="utf-8-sig",
)

model_diagnostics.to_csv(
    OUT_MODEL_DIAGNOSTICS,
    index=False,
    encoding="utf-8-sig",
)


# -----------------------------------------------------------------------------
# 11. Site-group baseline-to-post MSAVI2 contrasts
# -----------------------------------------------------------------------------

contrast_rows = []

for season in VALID_SEASONS:
    baseline = analysis[
        analysis["year"].isin(BASELINE_YEARS)
        & (analysis["season"] == season)
    ].copy()

    post = analysis[
        analysis["year"].isin(POST_YEARS)
        & (analysis["season"] == season)
    ].copy()

    baseline_means = (
        baseline
        .groupby(
            ["SITE_GROUP", "role_display"],
            as_index=False,
        )["msavi2_mean"]
        .mean()
        .rename(columns={
            "msavi2_mean": "baseline_mean"
        })
    )

    post_means = (
        post
        .groupby(
            ["SITE_GROUP", "role_display"],
            as_index=False,
        )["msavi2_mean"]
        .mean()
        .rename(columns={
            "msavi2_mean": "post_mean"
        })
    )

    period = baseline_means.merge(
        post_means,
        on=["SITE_GROUP", "role_display"],
        how="inner",
    )

    period["change_post_minus_baseline"] = (
        period["post_mean"] - period["baseline_mean"]
    )

    treatment = (
        period[
            period["role_display"] == "Treatment"
        ]
        .drop(columns=["role_display"])
        .rename(columns={
            "baseline_mean": "treatment_baseline_mean",
            "post_mean": "treatment_post_mean",
            "change_post_minus_baseline":
                "treatment_change",
        })
    )

    reference = (
        period[
            period["role_display"] == "Reference"
        ]
        .drop(columns=["role_display"])
        .rename(columns={
            "baseline_mean": "reference_baseline_mean",
            "post_mean": "reference_post_mean",
            "change_post_minus_baseline":
                "reference_change",
        })
    )

    contrasts = treatment.merge(
        reference,
        on="SITE_GROUP",
        how="inner",
        validate="one_to_one",
    )

    contrasts["baci_contrast"] = (
        contrasts["treatment_change"]
        - contrasts["reference_change"]
    )

    contrasts["season"] = season
    contrasts["outcome"] = "MSAVI2"

    contrast_rows.append(contrasts)

site_contrasts = pd.concat(
    contrast_rows,
    ignore_index=True,
)

site_contrasts = site_contrasts.sort_values(
    ["season", "SITE_GROUP"]
).reset_index(drop=True)

site_contrasts = round_numeric(site_contrasts)

site_contrasts.to_csv(
    OUT_SITE_CONTRASTS,
    index=False,
    encoding="utf-8-sig",
)

direction_rows = []

for season in VALID_SEASONS:
    subset = site_contrasts[
        site_contrasts["season"] == season
    ].copy()

    n_groups = int(subset["SITE_GROUP"].nunique())
    n_positive = int((subset["baci_contrast"] > 0).sum())
    n_negative = int((subset["baci_contrast"] < 0).sum())
    n_zero = int((subset["baci_contrast"] == 0).sum())

    direction_rows.append({
        "outcome": "MSAVI2",
        "season": season,
        "site_groups": n_groups,
        "positive_baci_groups": n_positive,
        "negative_baci_groups": n_negative,
        "zero_baci_groups": n_zero,
        "percent_positive":
            (
                n_positive / n_groups * 100.0
                if n_groups > 0
                else np.nan
            ),
    })

direction_summary = round_numeric(
    pd.DataFrame(direction_rows),
    digits=3,
)

direction_summary.to_csv(
    OUT_DIRECTION,
    index=False,
    encoding="utf-8-sig",
)


# -----------------------------------------------------------------------------
# 12. Final QA
# -----------------------------------------------------------------------------

model_checks = {
    "four_primary_interaction_results":
        len(model_results) == 4,
    "two_model_diagnostics":
        len(model_diagnostics) == 2,
    "models_converged":
        bool(model_diagnostics["converged"].all()),
    "464_observations_per_model":
        bool(
            (
                model_diagnostics["n_obs"]
                == EXPECTED_ROWS_PER_SEASON
            ).all()
        ),
}

site_checks = {
    "34_site_group_contrasts":
        len(site_contrasts)
        == EXPECTED_SITE_GROUPS * len(VALID_SEASONS),
    "17_groups_each_season":
        bool(
            (
                site_contrasts.groupby("season")["SITE_GROUP"]
                .nunique()
                == EXPECTED_SITE_GROUPS
            ).all()
        ),
    "two_direction_summary_rows":
        len(direction_summary) == 2,
}

all_checks = {
    **master_checks,
    **extraction_checks,
    **model_checks,
    **site_checks,
}

with open(
    OUT_QA,
    "w",
    encoding="utf-8",
) as qa:
    qa.write("MSAVI2 sensitivity analysis QA\n")
    qa.write("=" * 72 + "\n\n")

    qa.write("Analytical design\n")
    qa.write("-" * 72 + "\n")
    qa.write(
        "MSAVI2 replaces NDVI while the final 58 polygons, "
        "17 site groups, 2017-2024 period, seasonal windows "
        "and BACI phases are retained.\n"
    )
    qa.write(
        "MSAVI2 rasters are projected to EPSG:32642 at 10 m "
        "using bilinear resampling before polygon extraction.\n\n"
    )

    qa.write("Final sample\n")
    qa.write("-" * 72 + "\n")
    qa.write(f"Rows: {n_rows}\n")
    qa.write(f"Unique polygons: {n_polygons}\n")
    qa.write(f"Treatment polygons: {n_treatments}\n")
    qa.write(f"Reference polygons: {n_references}\n")
    qa.write(f"Site groups: {n_sites}\n")
    qa.write(
        f"Raster jobs used: {len(extraction_rows)} "
        f"(of {EXPECTED_RAW_RASTERS} district-season-year "
        "rasters generated by Script 07)\n\n"
    )

    qa.write("Primary model\n")
    qa.write("-" * 72 + "\n")
    qa.write(
        "msavi2_mean ~ "
        "treat * C(phase_3, Treatment('baseline'))\n"
    )
    qa.write("Random intercept: SITE_GROUP\n")
    qa.write(
        "Polygon-level variance component: "
        "0 + C(POLYGON_ID)\n"
    )
    qa.write("Estimator: ML, lbfgs\n\n")

    qa.write("Site-group summary\n")
    qa.write("-" * 72 + "\n")
    for row in direction_summary.itertuples(index=False):
        qa.write(
            f"{row.season}: "
            f"{int(row.positive_baci_groups)}/"
            f"{int(row.site_groups)} positive contrasts\n"
        )

    qa.write("\nChecks\n")
    qa.write("-" * 72 + "\n")
    for key, value in all_checks.items():
        qa.write(f"{key}: {pass_fail(value)}\n")

    qa.write("\nOutputs\n")
    qa.write("-" * 72 + "\n")
    for path in [
        OUT_PANEL,
        OUT_EXTRACTION_QA,
        OUT_MODEL_RESULTS,
        OUT_MODEL_DIAGNOSTICS,
        OUT_SITE_CONTRASTS,
        OUT_DIRECTION,
    ]:
        qa.write(f"{path.name}\n")


if STRICT_CHECKS and not all(all_checks.values()):
    failed = [
        key for key, value in all_checks.items()
        if not value
    ]
    raise RuntimeError(
        "MSAVI2 sensitivity analysis failed QA checks: "
        f"{failed}. See {OUT_QA}"
    )


safe_delete(final_layer)
arcpy.CheckInExtension("Spatial")

print("MSAVI2 sensitivity analysis complete")
print(f"Polygon panel: {OUT_PANEL}")
print(f"Model results: {OUT_MODEL_RESULTS}")
print(f"Site-group contrasts: {OUT_SITE_CONTRASTS}")
print(f"Direction summary: {OUT_DIRECTION}")
print(f"QA: {OUT_QA}")
