# -*- coding: utf-8 -*-
"""
Prepare terrain covariates and select matched untreated references.

This script consolidates the terrain-processing and reference-matching steps
used in the Afghanistan restoration monitoring analysis.

Workflow
--------
1. Mosaic Copernicus DEM GLO-30 tiles and project to EPSG:32642.
2. Create district-level DEM, slope and aspect rasters.
3. Extract mean elevation, slope and aspect for candidate polygons.
4. Derive northness and eastness from mean aspect.
5. Calculate treatment-reference area and distance diagnostics within site groups.
6. Summarise 2017-2018 baseline NDVI and BSI from Script 02 output.
7. Calculate the published baseline-informed matching score.
8. Rank candidate references for each treatment.
9. Retain ranks 1 and 2 plus the prespecified additional ranked references.
10. Export the final treatment-reference links and unique polygon list.

Requirements
------------
ArcGIS Pro Python environment with:
    arcpy
    Spatial Analyst
    numpy
    pandas

Expected candidate-polygon fields
---------------------------------
POLYGON_ID : anonymised unique polygon identifier
SITE_GROUP : anonymised site-group code used in the Supplementary Information
             (SG01-SG17)
SITE_ROLE  : "treatment" or "reference"

Additional reference selections
-------------------------------
The final study retained the two lowest-scoring references per treatment plus
three prespecified additional ranked reference polygons in selected site groups.

For the public repository, the three additional selections are defined directly
in the configuration block below using anonymised site-group and polygon IDs.
A listed reference is retained for every treatment-reference candidate link
involving that reference within its site group, matching the final selection
logic. The private site-name crosswalk is not used in this script.
"""

from pathlib import Path
import math
import traceback

import arcpy
import numpy as np
import pandas as pd
from arcpy.sa import Aspect, ExtractByMask, SetNull, Slope, ZonalStatisticsAsTable


# -----------------------------------------------------------------------------
# 1. Repository paths
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_ROOT = SCRIPT_DIR.parent

DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"

DEM_TILE_DIR = DATA_DIR / "dem_tiles"
DISTRICT_AOI_DIR = DATA_DIR / "district_boundaries_utm42n"
POLYGON_FC = DATA_DIR / "spatial" / "candidate_polygons_utm42n.shp"

# Output from 02_ndvi_bsi_polygon_processing.py
INDEX_LONG_CSV = OUTPUT_DIR / "polygon_indices_long_2017_2024.csv"

TERRAIN_DIR = OUTPUT_DIR / "terrain"
MATCHING_DIR = OUTPUT_DIR / "reference_matching"
WORK_DIR = OUTPUT_DIR / "_working_reference_matching"

OUT_TERRAIN = MATCHING_DIR / "polygon_terrain_covariates.csv"
OUT_RANKED = MATCHING_DIR / "reference_matching_ranked.csv"
OUT_SELECTED_LINKS = MATCHING_DIR / "selected_treatment_reference_links.csv"
OUT_SELECTED_POLYGONS = MATCHING_DIR / "selected_polygons_for_baci.csv"
OUT_QA = MATCHING_DIR / "reference_matching_qa.txt"

for folder in [OUTPUT_DIR, TERRAIN_DIR, MATCHING_DIR, WORK_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. Analysis settings
# -----------------------------------------------------------------------------

TARGET_EPSG = 32642
DEM_CELL_SIZE = 30

BASELINE_YEARS = [2017, 2018]
BASELINE_SEASONS = ["peak", "late"]

# Scaling thresholds used in the final baseline-informed matching score.
# These are scaling thresholds, not hard exclusion rules.
THRESHOLDS = {
    "ndvi": 0.05,
    "bsi": 0.05,
    "elevation": 300.0,
    "slope": 10.0,
    "northness": 0.50,
    "eastness": 0.50,
    "area": 75.0,
    "distance": 10000.0,
}

WEIGHTS = {
    "ndvi": 3.0,
    "bsi": 2.0,
    "elevation": 2.0,
    "slope": 2.0,
    "northness": 1.0,
    "eastness": 1.0,
    "area": 0.75,
    "distance": 0.75,
}

CORE_MAX_RANK = 2

# Three prespecified additional reference polygons retained in the final study.
# These are public anonymous identifiers. The private mapping to original
# site/reference names is intentionally not included in the repository.
ADDITIONAL_REFERENCE_SELECTIONS = [
    {
        "site_group": "SG05",
        "reference_id": "SG05_R03",
    },
    {
        "site_group": "SG06",
        "reference_id": "SG06_R03",
    },
    {
        "site_group": "SG13",
        "reference_id": "SG13_R03",
    },
]

EXPECTED_TREATMENTS = 19
EXPECTED_REFERENCES = 39
EXPECTED_SITE_GROUPS = 17
EXPECTED_LINKS = 42
STRICT_FINAL_SAMPLE_CHECK = True

POLYGON_ID_FIELD = "POLYGON_ID"
SITE_GROUP_FIELD = "SITE_GROUP"
SITE_ROLE_FIELD = "SITE_ROLE"

VALID_ROLES = {"treatment", "reference"}

# District labels are public study-area information. AOI filenames preserve the
# original local file naming where needed.
DISTRICTS = [
    {
        "district_key": "Bamyan_district_Bamyan",
        "aoi_file": "Bamyan_district_Bamyan.shp",
    },
    {
        "district_key": "Sayghan_district_Bamyan",
        "aoi_file": "Sayghan_district_Bamyan.shp",
    },
    {
        "district_key": "Miramor_district_Daykundi",
        "aoi_file": "Miramor_district_Daykundi.shp",
    },
    {
        "district_key": "Shahristan_district_Daykundi",
        "aoi_file": "Shahristan_district_Daykundi.shp",
    },
    {
        "district_key": "HazarSumuch_district_Takhar",
        "aoi_file": "HazarSumuch_district_Takhar.shp",
    },
    {
        "district_key": "Kalfagan_district_Takhar",
        "aoi_file": "Kalfagan_district_Takhar.shp",
    },
    {
        "district_key": "Rustaq_district_Takhar",
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


def reset_env():
    arcpy.env.mask = None
    arcpy.env.extent = None
    arcpy.env.snapRaster = None
    arcpy.env.cellSize = None
    arcpy.env.outputCoordinateSystem = None


def require_feature_class(path, label):
    if not arcpy.Exists(str(path)):
        raise FileNotFoundError(f"{label} not found: {path}")


def require_fields(feature_class, fields):
    existing = {f.name for f in arcpy.ListFields(str(feature_class))}
    missing = [field for field in fields if field not in existing]
    if missing:
        raise ValueError(
            f"Missing required fields in {feature_class}: {missing}"
        )


def require_epsg(feature_class, expected_epsg):
    sr = arcpy.Describe(str(feature_class)).spatialReference
    if sr.factoryCode != expected_epsg:
        raise ValueError(
            f"{feature_class} must use EPSG:{expected_epsg}; "
            f"found {sr.name} (WKID {sr.factoryCode})."
        )


def normalised_score(value, threshold):
    if pd.isna(value):
        return np.nan
    return min(abs(float(value)) / float(threshold), 1.0)


def weighted_mean_score(row, components):
    numerator = 0.0
    denominator = 0.0

    for column, weight in components.items():
        value = row.get(column, np.nan)
        if pd.notna(value):
            numerator += float(value) * float(weight)
            denominator += float(weight)

    return numerator / denominator if denominator > 0 else np.nan


def aspect_to_northness(aspect_deg):
    if aspect_deg is None or pd.isna(aspect_deg):
        return np.nan
    return math.cos(math.radians(float(aspect_deg)))


def aspect_to_eastness(aspect_deg):
    if aspect_deg is None or pd.isna(aspect_deg):
        return np.nan
    return math.sin(math.radians(float(aspect_deg)))


def terrain_valid(elevation, slope, aspect):
    if any(v is None or pd.isna(v) for v in [elevation, slope, aspect]):
        return 0
    if float(elevation) <= 0 or float(slope) < 0:
        return 0
    return 1


# -----------------------------------------------------------------------------
# 4. Prepare DEM, slope and aspect
# -----------------------------------------------------------------------------

require_feature_class(POLYGON_FC, "Candidate polygon feature class")
require_fields(
    POLYGON_FC,
    [POLYGON_ID_FIELD, SITE_GROUP_FIELD, SITE_ROLE_FIELD],
)
require_epsg(POLYGON_FC, TARGET_EPSG)

dem_tiles = sorted(DEM_TILE_DIR.glob("*.tif"))
if not dem_tiles:
    raise FileNotFoundError(f"No DEM tiles found in {DEM_TILE_DIR}")

mosaic_raw = TERRAIN_DIR / "dem_mosaic_raw.tif"
mosaic_utm = TERRAIN_DIR / "dem_mosaic_utm42n.tif"

safe_delete(mosaic_raw)
safe_delete(mosaic_utm)

arcpy.management.MosaicToNewRaster(
    input_rasters=[str(p) for p in dem_tiles],
    output_location=str(TERRAIN_DIR),
    raster_dataset_name_with_extension=mosaic_raw.name,
    coordinate_system_for_the_raster="",
    pixel_type="32_BIT_FLOAT",
    cellsize="",
    number_of_bands=1,
    mosaic_method="MEAN",
    mosaic_colormap_mode="FIRST",
)

arcpy.management.ProjectRaster(
    in_raster=str(mosaic_raw),
    out_raster=str(mosaic_utm),
    out_coor_system=TARGET_SR,
    resampling_type="BILINEAR",
    cell_size=str(DEM_CELL_SIZE),
)

arcpy.management.CalculateStatistics(str(mosaic_utm))

terrain_products = {}

for cfg in DISTRICTS:
    district_key = cfg["district_key"]
    aoi = DISTRICT_AOI_DIR / cfg["aoi_file"]

    require_feature_class(aoi, "District boundary")
    require_epsg(aoi, TARGET_EPSG)

    district_dir = TERRAIN_DIR / district_key
    district_dir.mkdir(parents=True, exist_ok=True)

    dem_out = district_dir / f"{district_key}_dem_utm42n.tif"
    slope_out = district_dir / f"{district_key}_slope_deg_utm42n.tif"
    aspect_out = district_dir / f"{district_key}_aspect_deg_utm42n.tif"
    temp_clip = f"memory/dem_clip_{district_key}"

    for path in [dem_out, slope_out, aspect_out, temp_clip]:
        safe_delete(path)

    reset_env()
    arcpy.env.snapRaster = str(mosaic_utm)
    arcpy.env.cellSize = str(mosaic_utm)
    arcpy.env.extent = str(aoi)

    extracted = ExtractByMask(str(mosaic_utm), str(aoi))
    extracted.save(temp_clip)

    cleaned = SetNull(
        (arcpy.Raster(temp_clip) <= -10000)
        | (arcpy.Raster(temp_clip) >= 10000),
        arcpy.Raster(temp_clip),
    )
    cleaned.save(str(dem_out))
    arcpy.management.CalculateStatistics(str(dem_out))

    slope_raster = Slope(
        in_raster=str(dem_out),
        output_measurement="DEGREE",
        z_factor=1,
    )
    slope_raster.save(str(slope_out))
    arcpy.management.CalculateStatistics(str(slope_out))

    aspect_raster = Aspect(str(dem_out))
    aspect_raster.save(str(aspect_out))
    arcpy.management.CalculateStatistics(str(aspect_out))

    terrain_products[district_key] = {
        "aoi": aoi,
        "dem": dem_out,
        "slope": slope_out,
        "aspect": aspect_out,
    }

    safe_delete(temp_clip)

reset_env()


# -----------------------------------------------------------------------------
# 5. Extract polygon-level terrain covariates
# -----------------------------------------------------------------------------

working_polygons = WORK_DIR / "candidate_polygons_terrain.shp"
safe_delete(working_polygons)
arcpy.management.CopyFeatures(str(POLYGON_FC), str(working_polygons))
arcpy.management.RepairGeometry(str(working_polygons))

polygon_layer = "matching_polygon_layer"
safe_delete(polygon_layer)
arcpy.management.MakeFeatureLayer(str(working_polygons), polygon_layer)

terrain_rows = {}

for i, cfg in enumerate(DISTRICTS, start=1):
    district_key = cfg["district_key"]
    products = terrain_products[district_key]

    arcpy.management.SelectLayerByAttribute(
        polygon_layer, "CLEAR_SELECTION"
    )
    arcpy.management.SelectLayerByLocation(
        in_layer=polygon_layer,
        overlap_type="INTERSECT",
        select_features=str(products["aoi"]),
        selection_type="NEW_SELECTION",
    )

    if int(arcpy.management.GetCount(polygon_layer)[0]) == 0:
        continue

    selected_fc = f"memory/terrain_polygons_{i}"
    dem_table = f"memory/terrain_dem_{i}"
    slope_table = f"memory/terrain_slope_{i}"
    aspect_table = f"memory/terrain_aspect_{i}"

    for item in [selected_fc, dem_table, slope_table, aspect_table]:
        safe_delete(item)

    arcpy.management.CopyFeatures(polygon_layer, selected_fc)

    ZonalStatisticsAsTable(
        selected_fc, POLYGON_ID_FIELD, str(products["dem"]),
        dem_table, "DATA", "MEAN"
    )
    ZonalStatisticsAsTable(
        selected_fc, POLYGON_ID_FIELD, str(products["slope"]),
        slope_table, "DATA", "MEAN"
    )
    ZonalStatisticsAsTable(
        selected_fc, POLYGON_ID_FIELD, str(products["aspect"]),
        aspect_table, "DATA", "MEAN"
    )

    def table_means(table):
        values = {}
        with arcpy.da.SearchCursor(
            table, [POLYGON_ID_FIELD, "MEAN"]
        ) as cursor:
            for polygon_id, mean_value in cursor:
                values[str(polygon_id)] = mean_value
        return values

    dem_means = table_means(dem_table)
    slope_means = table_means(slope_table)
    aspect_means = table_means(aspect_table)

    with arcpy.da.SearchCursor(
        selected_fc,
        [POLYGON_ID_FIELD, SITE_GROUP_FIELD, SITE_ROLE_FIELD],
    ) as cursor:
        for polygon_id, site_group, site_role in cursor:
            polygon_id = str(polygon_id)
            role = str(site_role).strip().lower()

            if role not in VALID_ROLES:
                raise ValueError(
                    f"Unexpected SITE_ROLE '{site_role}' for {polygon_id}."
                )

            elevation = dem_means.get(polygon_id)
            slope = slope_means.get(polygon_id)
            aspect = aspect_means.get(polygon_id)

            terrain_rows[polygon_id] = {
                "polygon_id": polygon_id,
                "site_group": str(site_group).strip(),
                "site_role": role,
                "district_key": district_key,
                "elevation_m": elevation,
                "slope_deg": slope,
                "aspect_deg": aspect,
                "northness": aspect_to_northness(aspect),
                "eastness": aspect_to_eastness(aspect),
                "terrain_valid": terrain_valid(elevation, slope, aspect),
            }

    for item in [selected_fc, dem_table, slope_table, aspect_table]:
        safe_delete(item)

terrain = pd.DataFrame(terrain_rows.values())
terrain = terrain.sort_values(
    ["site_group", "site_role", "polygon_id"]
).reset_index(drop=True)
terrain.to_csv(OUT_TERRAIN, index=False, encoding="utf-8-sig")


# -----------------------------------------------------------------------------
# 6. Read polygon geometry and calculate pairwise area/distance diagnostics
# -----------------------------------------------------------------------------

polygon_geometry = {}

with arcpy.da.SearchCursor(
    str(working_polygons),
    [
        POLYGON_ID_FIELD,
        SITE_GROUP_FIELD,
        SITE_ROLE_FIELD,
        "SHAPE@",
        "SHAPE@AREA",
    ],
) as cursor:
    for polygon_id, site_group, site_role, geometry, area_m2 in cursor:
        role = str(site_role).strip().lower()
        if role not in VALID_ROLES:
            continue

        polygon_geometry[str(polygon_id)] = {
            "polygon_id": str(polygon_id),
            "site_group": str(site_group).strip(),
            "site_role": role,
            "geometry": geometry,
            "area_ha": float(area_m2) / 10000.0,
        }

pair_rows = []

site_groups = sorted(
    {x["site_group"] for x in polygon_geometry.values()}
)

for site_group in site_groups:
    treatments = [
        x for x in polygon_geometry.values()
        if x["site_group"] == site_group
        and x["site_role"] == "treatment"
    ]
    references = [
        x for x in polygon_geometry.values()
        if x["site_group"] == site_group
        and x["site_role"] == "reference"
    ]

    for treatment in treatments:
        for reference in references:
            t_geom = treatment["geometry"]
            r_geom = reference["geometry"]

            # Polygon centroids returned by ArcPy are Point objects. Calculate
            # planar centroid-to-centroid distance directly from UTM coordinates
            # (EPSG:32642), so the result is in metres.
            t_centroid = t_geom.centroid
            r_centroid = r_geom.centroid

            centroid_distance = math.hypot(
                t_centroid.X - r_centroid.X,
                t_centroid.Y - r_centroid.Y,
            )

            edge_distance = t_geom.distanceTo(r_geom)

            t_area = treatment["area_ha"]
            r_area = reference["area_ha"]

            area_diff_pct = (
                ((r_area - t_area) / t_area) * 100.0
                if t_area > 0 else np.nan
            )

            pair_rows.append({
                "site_group": site_group,
                "treatment_id": treatment["polygon_id"],
                "reference_id": reference["polygon_id"],
                "treatment_area_ha": t_area,
                "reference_area_ha": r_area,
                "area_diff_pct_reference_minus_treatment": area_diff_pct,
                "abs_area_diff_pct": abs(area_diff_pct)
                    if pd.notna(area_diff_pct) else np.nan,
                "area_ratio_reference_to_treatment":
                    r_area / t_area if t_area > 0 else np.nan,
                "centroid_distance_m": centroid_distance,
                "edge_distance_m": edge_distance,
            })

pairs = pd.DataFrame(pair_rows)

if pairs.empty:
    raise RuntimeError(
        "No treatment-reference candidate pairs were generated."
    )


# -----------------------------------------------------------------------------
# 7. Add terrain covariates to treatment-reference pairs
# -----------------------------------------------------------------------------

terrain_t = terrain.rename(columns={
    "polygon_id": "treatment_id",
    "elevation_m": "treatment_elevation_m",
    "slope_deg": "treatment_slope_deg",
    "northness": "treatment_northness",
    "eastness": "treatment_eastness",
    "terrain_valid": "treatment_terrain_valid",
})

terrain_r = terrain.rename(columns={
    "polygon_id": "reference_id",
    "elevation_m": "reference_elevation_m",
    "slope_deg": "reference_slope_deg",
    "northness": "reference_northness",
    "eastness": "reference_eastness",
    "terrain_valid": "reference_terrain_valid",
})

terrain_t = terrain_t[[
    "treatment_id",
    "treatment_elevation_m",
    "treatment_slope_deg",
    "treatment_northness",
    "treatment_eastness",
    "treatment_terrain_valid",
]]

terrain_r = terrain_r[[
    "reference_id",
    "reference_elevation_m",
    "reference_slope_deg",
    "reference_northness",
    "reference_eastness",
    "reference_terrain_valid",
]]

pairs = pairs.merge(terrain_t, on="treatment_id", how="left")
pairs = pairs.merge(terrain_r, on="reference_id", how="left")

pairs["abs_elevation_diff_m"] = (
    pairs["treatment_elevation_m"] - pairs["reference_elevation_m"]
).abs()

pairs["abs_slope_diff_deg"] = (
    pairs["treatment_slope_deg"] - pairs["reference_slope_deg"]
).abs()

pairs["abs_northness_diff"] = (
    pairs["treatment_northness"] - pairs["reference_northness"]
).abs()

pairs["abs_eastness_diff"] = (
    pairs["treatment_eastness"] - pairs["reference_eastness"]
).abs()

pairs["terrain_pair_valid"] = np.where(
    (pairs["treatment_terrain_valid"] == 1)
    & (pairs["reference_terrain_valid"] == 1),
    1,
    0,
)


# -----------------------------------------------------------------------------
# 8. Summarise 2017-2018 baseline NDVI and BSI
# -----------------------------------------------------------------------------

if not INDEX_LONG_CSV.exists():
    raise FileNotFoundError(
        f"Script 02 output not found: {INDEX_LONG_CSV}"
    )

indices = pd.read_csv(INDEX_LONG_CSV, encoding="utf-8-sig")

required_index_cols = {
    "polygon_id",
    "site_group",
    "site_role",
    "year",
    "season",
    "ndvi_mean",
    "bsi_mean",
}

missing = required_index_cols - set(indices.columns)
if missing:
    raise ValueError(
        f"Missing required columns in {INDEX_LONG_CSV}: {sorted(missing)}"
    )

indices["polygon_id"] = indices["polygon_id"].astype(str)
indices["year"] = pd.to_numeric(indices["year"], errors="coerce")
indices["season"] = indices["season"].astype(str).str.strip().str.lower()

baseline = indices[
    indices["year"].isin(BASELINE_YEARS)
    & indices["season"].isin(BASELINE_SEASONS)
].copy()

baseline_summary = (
    baseline.groupby("polygon_id", as_index=False)
    .agg(
        ndvi_baseline_mean=("ndvi_mean", "mean"),
        bsi_baseline_mean=("bsi_mean", "mean"),
        baseline_observations=("year", "size"),
    )
)

for season in BASELINE_SEASONS:
    season_summary = (
        baseline[baseline["season"] == season]
        .groupby("polygon_id", as_index=False)
        .agg(
            **{
                f"ndvi_baseline_{season}": ("ndvi_mean", "mean"),
                f"bsi_baseline_{season}": ("bsi_mean", "mean"),
            }
        )
    )
    baseline_summary = baseline_summary.merge(
        season_summary,
        on="polygon_id",
        how="left",
    )

baseline_t = baseline_summary.rename(columns={
    "polygon_id": "treatment_id",
    "ndvi_baseline_mean": "treatment_ndvi_baseline_mean",
    "bsi_baseline_mean": "treatment_bsi_baseline_mean",
    "ndvi_baseline_peak": "treatment_ndvi_baseline_peak",
    "ndvi_baseline_late": "treatment_ndvi_baseline_late",
    "bsi_baseline_peak": "treatment_bsi_baseline_peak",
    "bsi_baseline_late": "treatment_bsi_baseline_late",
    "baseline_observations": "treatment_baseline_observations",
})

baseline_r = baseline_summary.rename(columns={
    "polygon_id": "reference_id",
    "ndvi_baseline_mean": "reference_ndvi_baseline_mean",
    "bsi_baseline_mean": "reference_bsi_baseline_mean",
    "ndvi_baseline_peak": "reference_ndvi_baseline_peak",
    "ndvi_baseline_late": "reference_ndvi_baseline_late",
    "bsi_baseline_peak": "reference_bsi_baseline_peak",
    "bsi_baseline_late": "reference_bsi_baseline_late",
    "baseline_observations": "reference_baseline_observations",
})

pairs = pairs.merge(baseline_t, on="treatment_id", how="left")
pairs = pairs.merge(baseline_r, on="reference_id", how="left")


# -----------------------------------------------------------------------------
# 9. Published baseline-informed matching score
# -----------------------------------------------------------------------------

pairs["abs_ndvi_baseline_diff"] = (
    pairs["treatment_ndvi_baseline_mean"]
    - pairs["reference_ndvi_baseline_mean"]
).abs()

pairs["abs_bsi_baseline_diff"] = (
    pairs["treatment_bsi_baseline_mean"]
    - pairs["reference_bsi_baseline_mean"]
).abs()

pairs["baseline_pair_valid"] = np.where(
    pairs["treatment_ndvi_baseline_mean"].notna()
    & pairs["reference_ndvi_baseline_mean"].notna()
    & pairs["treatment_bsi_baseline_mean"].notna()
    & pairs["reference_bsi_baseline_mean"].notna()
    & (pairs["terrain_pair_valid"] == 1),
    1,
    0,
)

pairs["score_ndvi"] = pairs["abs_ndvi_baseline_diff"].apply(
    lambda x: normalised_score(x, THRESHOLDS["ndvi"])
)
pairs["score_bsi"] = pairs["abs_bsi_baseline_diff"].apply(
    lambda x: normalised_score(x, THRESHOLDS["bsi"])
)
pairs["score_elevation"] = pairs["abs_elevation_diff_m"].apply(
    lambda x: normalised_score(x, THRESHOLDS["elevation"])
)
pairs["score_slope"] = pairs["abs_slope_diff_deg"].apply(
    lambda x: normalised_score(x, THRESHOLDS["slope"])
)
pairs["score_northness"] = pairs["abs_northness_diff"].apply(
    lambda x: normalised_score(x, THRESHOLDS["northness"])
)
pairs["score_eastness"] = pairs["abs_eastness_diff"].apply(
    lambda x: normalised_score(x, THRESHOLDS["eastness"])
)
pairs["score_area"] = pairs["abs_area_diff_pct"].apply(
    lambda x: normalised_score(x, THRESHOLDS["area"])
)
pairs["score_distance"] = pairs["centroid_distance_m"].apply(
    lambda x: normalised_score(x, THRESHOLDS["distance"])
)

score_components = {
    "score_ndvi": WEIGHTS["ndvi"],
    "score_bsi": WEIGHTS["bsi"],
    "score_elevation": WEIGHTS["elevation"],
    "score_slope": WEIGHTS["slope"],
    "score_northness": WEIGHTS["northness"],
    "score_eastness": WEIGHTS["eastness"],
    "score_area": WEIGHTS["area"],
    "score_distance": WEIGHTS["distance"],
}

pairs["match_score"] = pairs.apply(
    lambda row: weighted_mean_score(row, score_components),
    axis=1,
)

pairs.loc[
    pairs["baseline_pair_valid"] == 0,
    "match_score",
] = np.nan

pairs["match_score"] = pairs["match_score"].round(5)

# Edge-to-edge distance is retained as a proximity diagnostic only.
pairs["reference_rank"] = (
    pairs.groupby(["site_group", "treatment_id"])["match_score"]
    .rank(method="first", ascending=True)
    .astype("Int64")
)

pairs = pairs.sort_values(
    ["site_group", "treatment_id", "reference_rank", "reference_id"]
).reset_index(drop=True)

pairs.to_csv(OUT_RANKED, index=False, encoding="utf-8-sig")


# -----------------------------------------------------------------------------
# 10. Final reference selection
# -----------------------------------------------------------------------------

# The two lowest-scoring references are retained for every treatment.
core = pairs[
    (pairs["reference_rank"] <= CORE_MAX_RANK)
    & pairs["match_score"].notna()
].copy()

core["selection_tier"] = "rank_1_2"

# The original final-selection workflow also retained three specific additional
# reference polygons. Keeping these selections explicit is important: they are
# not replaced here by a generic "rank 3" rule because one additional reference
# can be linked to more than one treatment within the same site group.
additional = pd.DataFrame(ADDITIONAL_REFERENCE_SELECTIONS).copy()

required_additional = {"site_group", "reference_id"}
missing = required_additional - set(additional.columns)

if missing:
    raise ValueError(
        "ADDITIONAL_REFERENCE_SELECTIONS is missing required fields: "
        f"{sorted(missing)}"
    )

additional["site_group"] = (
    additional["site_group"].astype(str).str.strip().str.upper()
)
additional["reference_id"] = (
    additional["reference_id"].astype(str).str.strip()
)

if (additional["reference_id"] == "").any():
    raise ValueError(
        "ADDITIONAL_REFERENCE_SELECTIONS contains an empty reference_id."
    )

invalid_additional_groups = sorted(
    set(additional["site_group"])
    - set(pairs["site_group"].astype(str).str.upper())
)

if invalid_additional_groups:
    raise ValueError(
        "Additional-reference site groups are not present in the ranked "
        f"candidate table: {invalid_additional_groups}"
    )

if additional.duplicated(
    subset=["site_group", "reference_id"]
).any():
    raise ValueError(
        "Duplicate entries found in ADDITIONAL_REFERENCE_SELECTIONS."
    )

additional_rows = []

for row in additional.itertuples(index=False):
    selected_extra = pairs[
        (pairs["site_group"].astype(str).str.upper() == row.site_group)
        & (pairs["reference_id"].astype(str) == row.reference_id)
        & pairs["match_score"].notna()
    ].copy()

    if selected_extra.empty:
        raise ValueError(
            "Configured additional reference was not found among valid ranked "
            f"candidates: {row.site_group} | {row.reference_id}"
        )

    # Retain every treatment-reference link involving the configured reference
    # within the site group, matching the original final-selection script.
    selected_extra["selection_tier"] = "additional_ranked_reference"
    additional_rows.append(selected_extra)

extras = (
    pd.concat(additional_rows, ignore_index=True)
    if additional_rows
    else pd.DataFrame(columns=core.columns)
)

selected_links = pd.concat(
    [core, extras],
    ignore_index=True,
)

selected_links = selected_links.drop_duplicates(
    subset=["site_group", "treatment_id", "reference_id"],
    keep="first",
)

selected_links = selected_links.sort_values(
    ["site_group", "treatment_id", "reference_rank", "reference_id"]
).reset_index(drop=True)

selected_links.to_csv(
    OUT_SELECTED_LINKS,
    index=False,
    encoding="utf-8-sig",
)


# -----------------------------------------------------------------------------
# 11. Unique final polygon list
# -----------------------------------------------------------------------------

treatment_ids = (
    selected_links[["site_group", "treatment_id"]]
    .drop_duplicates()
    .rename(columns={"treatment_id": "polygon_id"})
)
treatment_ids["site_role"] = "treatment"

reference_ids = (
    selected_links[["site_group", "reference_id"]]
    .drop_duplicates()
    .rename(columns={"reference_id": "polygon_id"})
)
reference_ids["site_role"] = "reference"

selected_polygons = pd.concat(
    [treatment_ids, reference_ids],
    ignore_index=True,
).drop_duplicates(
    subset=["polygon_id", "site_role"]
)

selected_polygons = selected_polygons.merge(
    baseline_summary,
    on="polygon_id",
    how="left",
)

selected_polygons = selected_polygons.merge(
    terrain[[
        "polygon_id",
        "elevation_m",
        "slope_deg",
        "northness",
        "eastness",
        "terrain_valid",
    ]],
    on="polygon_id",
    how="left",
)

selected_polygons = selected_polygons.sort_values(
    ["site_group", "site_role", "polygon_id"]
).reset_index(drop=True)

selected_polygons.to_csv(
    OUT_SELECTED_POLYGONS,
    index=False,
    encoding="utf-8-sig",
)


# -----------------------------------------------------------------------------
# 12. Final checks and QA
# -----------------------------------------------------------------------------

n_links = len(selected_links)
n_treatments = selected_links["treatment_id"].nunique()
n_references = selected_links["reference_id"].nunique()
n_sites = selected_links["site_group"].nunique()

checks = {
    "treatments": n_treatments == EXPECTED_TREATMENTS,
    "references": n_references == EXPECTED_REFERENCES,
    "site_groups": n_sites == EXPECTED_SITE_GROUPS,
    "links": n_links == EXPECTED_LINKS,
}

with open(OUT_QA, "w", encoding="utf-8") as qa:
    qa.write("Reference matching QA\n")
    qa.write("=" * 72 + "\n\n")

    qa.write("Published matching specification\n")
    qa.write("-" * 72 + "\n")
    qa.write(f"Baseline years: {BASELINE_YEARS}\n")
    qa.write(f"Baseline seasons: {BASELINE_SEASONS}\n")
    qa.write(f"Scaling thresholds: {THRESHOLDS}\n")
    qa.write(f"Weights: {WEIGHTS}\n")
    qa.write(
        "Edge-to-edge distance retained as a diagnostic and not included "
        "in the final matching score.\n"
    )
    qa.write(
        "Additional references are configured directly in this script using "
        "anonymous site-group and polygon IDs.\n"
    )
    for item in ADDITIONAL_REFERENCE_SELECTIONS:
        qa.write(
            f"  additional reference: {item['site_group']} | "
            f"{item['reference_id']}\n"
        )
    qa.write("\n")

    qa.write("Final sample\n")
    qa.write("-" * 72 + "\n")
    qa.write(f"Treatment-reference links: {n_links}\n")
    qa.write(f"Unique treatments: {n_treatments}\n")
    qa.write(f"Unique references: {n_references}\n")
    qa.write(f"Site groups: {n_sites}\n\n")

    qa.write("Checks\n")
    qa.write("-" * 72 + "\n")
    for key, value in checks.items():
        qa.write(f"{key}: {'PASS' if value else 'CHECK'}\n")

if STRICT_FINAL_SAMPLE_CHECK and not all(checks.values()):
    failed = [key for key, value in checks.items() if not value]
    raise RuntimeError(
        "Final reference-selection counts do not match the published "
        f"analytical sample. Failed checks: {failed}. "
        f"See {OUT_QA}"
    )

print("Reference matching complete")
print(f"Ranked candidates: {OUT_RANKED}")
print(f"Selected links: {OUT_SELECTED_LINKS}")
print(f"Selected polygons: {OUT_SELECTED_POLYGONS}")
print(f"QA: {OUT_QA}")

arcpy.CheckInExtension("Spatial")
