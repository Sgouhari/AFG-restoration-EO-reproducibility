# -*- coding: utf-8 -*-
"""
Reproject Sentinel-2 NDVI/BSI rasters and extract polygon-level statistics.

This script is the second step in the reproducibility workflow. It takes the
seasonal GeoTIFFs exported by 01_sentinel2_ndvi_bsi_gee.js, projects them to
WGS 84 / UTM Zone 42N (EPSG:32642) at 10 m, clips them to district boundaries,
and extracts zonal statistics for the candidate treatment/reference polygons.

Requirements
------------
ArcGIS Pro Python environment with:
    arcpy
    Spatial Analyst
    pandas

Expected polygon fields
-----------------------
POLYGON_ID : anonymised unique polygon identifier
SITE_GROUP : anonymised matched-landscape/site-group identifier
SITE_ROLE  : "treatment" or "reference"

The polygon and district-boundary feature classes should be in EPSG:32642.
"""

from pathlib import Path
import traceback

import arcpy
import pandas as pd
from arcpy.sa import ExtractByMask, ZonalStatisticsAsTable


# -----------------------------------------------------------------------------
# 1. Repository paths and analysis settings
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_ROOT = SCRIPT_DIR.parent

DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"

RAW_RASTER_DIR = DATA_DIR / "sentinel2_exports"
DISTRICT_AOI_DIR = DATA_DIR / "district_boundaries_utm42n"
POLYGON_FC = DATA_DIR / "spatial" / "candidate_polygons_utm42n.shp"

PROCESSED_RASTER_DIR = OUTPUT_DIR / "processed_indices"
WORK_DIR = OUTPUT_DIR / "_working_polygon_processing"

OUT_LONG = OUTPUT_DIR / "polygon_indices_long_2017_2024.csv"
OUT_RASTER_QA = OUTPUT_DIR / "polygon_processing_raster_qa.csv"
OUT_EXTRACTION_QA = OUTPUT_DIR / "polygon_processing_extraction_qa.csv"

TARGET_EPSG = 32642
TARGET_CELL_SIZE = 10
YEARS = list(range(2017, 2025))
SEASONS = ["peak", "late"]
INDICATORS = ["NDVI", "BSI"]

POLYGON_ID_FIELD = "POLYGON_ID"
SITE_GROUP_FIELD = "SITE_GROUP"
SITE_ROLE_FIELD = "SITE_ROLE"

VALID_ROLES = {"treatment", "reference"}

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
# 2. ArcGIS environment
# -----------------------------------------------------------------------------

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True

for folder in [OUTPUT_DIR, PROCESSED_RASTER_DIR, WORK_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TARGET_SR = arcpy.SpatialReference(TARGET_EPSG)


# -----------------------------------------------------------------------------
# 3. Utility functions
# -----------------------------------------------------------------------------

def reset_raster_environment():
    """Clear raster environments between processing operations."""
    arcpy.env.snapRaster = None
    arcpy.env.cellSize = None
    arcpy.env.mask = None
    arcpy.env.extent = None
    arcpy.env.outputCoordinateSystem = None


def safe_delete(path):
    """Delete an ArcGIS dataset if it exists."""
    try:
        if path and arcpy.Exists(str(path)):
            arcpy.management.Delete(str(path))
    except Exception:
        pass


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
            f"{feature_class} must be EPSG:{expected_epsg}; "
            f"found {sr.name} (WKID {sr.factoryCode})."
        )


def raster_property(raster, property_name):
    return arcpy.management.GetRasterProperties(
        str(raster), property_name
    ).getOutput(0)


def raster_cell_area(raster):
    x = float(raster_property(raster, "CELLSIZEX"))
    y = float(raster_property(raster, "CELLSIZEY"))
    return abs(x * y)


def processed_raster_path(indicator, province, district, season, year):
    folder = PROCESSED_RASTER_DIR / indicator.lower()
    folder.mkdir(parents=True, exist_ok=True)
    name = (
        f"{province}_{district}_{indicator}_{season}_{year}"
        f"_utm42n_clip.tif"
    )
    return folder / name


def raw_raster_path(indicator, province, district, season, year):
    name = f"{province}_{district}_{indicator}_{season}_{year}.tif"
    return RAW_RASTER_DIR / name


def validate_output_raster(raster):
    if not arcpy.Exists(str(raster)):
        return False, "output raster not created"

    rows = int(float(raster_property(raster, "ROWCOUNT")))
    cols = int(float(raster_property(raster, "COLUMNCOUNT")))

    if rows <= 0 or cols <= 0:
        return False, "output raster has zero rows or columns"

    minimum = raster_property(raster, "MINIMUM")
    maximum = raster_property(raster, "MAXIMUM")

    if minimum in [None, ""] or maximum in [None, ""]:
        return False, "output raster contains no valid statistics"

    return True, "ok"


# -----------------------------------------------------------------------------
# 4. Raster projection and district clipping
# -----------------------------------------------------------------------------

def process_raster(indicator, district_cfg, season, year):
    """Project one GEE raster to UTM 42N, clip it, and return QA metadata."""
    province = district_cfg["province"]
    file_district = district_cfg["file_district"]
    aoi = DISTRICT_AOI_DIR / district_cfg["aoi_file"]

    source = raw_raster_path(
        indicator, province, file_district, season, year
    )
    output = processed_raster_path(
        indicator, province, file_district, season, year
    )
    temp = WORK_DIR / (
        f"tmp_{province}_{file_district}_{indicator}_{season}_{year}.tif"
    )

    row = {
        "indicator": indicator,
        "province": province,
        "district": district_cfg["district"],
        "year": year,
        "season": season,
        "source_raster": str(source),
        "processed_raster": str(output),
        "status": "failed",
        "message": "",
    }

    try:
        if not source.exists():
            row["message"] = "source raster not found"
            return None, row

        require_feature_class(aoi, "District boundary")
        require_epsg(aoi, TARGET_EPSG)

        safe_delete(temp)
        safe_delete(output)
        reset_raster_environment()

        arcpy.management.ProjectRaster(
            in_raster=str(source),
            out_raster=str(temp),
            out_coor_system=TARGET_SR,
            resampling_type="BILINEAR",
            cell_size=str(TARGET_CELL_SIZE),
        )

        arcpy.env.snapRaster = str(temp)
        arcpy.env.cellSize = str(temp)
        arcpy.env.mask = str(aoi)
        arcpy.env.extent = str(aoi)

        clipped = ExtractByMask(str(temp), str(aoi))
        clipped.save(str(output))

        arcpy.management.CalculateStatistics(str(output))

        ok, message = validate_output_raster(output)
        if not ok:
            row["message"] = message
            return None, row

        desc = arcpy.Describe(str(output))
        row.update({
            "output_crs": desc.spatialReference.name,
            "rows": raster_property(output, "ROWCOUNT"),
            "cols": raster_property(output, "COLUMNCOUNT"),
            "cellsize_x": raster_property(output, "CELLSIZEX"),
            "cellsize_y": raster_property(output, "CELLSIZEY"),
            "minimum": raster_property(output, "MINIMUM"),
            "maximum": raster_property(output, "MAXIMUM"),
            "mean": raster_property(output, "MEAN"),
            "std": raster_property(output, "STD"),
            "status": "ok",
            "message": "",
        })

        return output, row

    except Exception as exc:
        row["message"] = f"{exc} | {traceback.format_exc(limit=1)}"
        return None, row

    finally:
        safe_delete(temp)
        reset_raster_environment()


# -----------------------------------------------------------------------------
# 5. Candidate-polygon preparation
# -----------------------------------------------------------------------------

require_feature_class(POLYGON_FC, "Candidate polygon feature class")
require_fields(
    POLYGON_FC,
    [POLYGON_ID_FIELD, SITE_GROUP_FIELD, SITE_ROLE_FIELD],
)
require_epsg(POLYGON_FC, TARGET_EPSG)

working_polygons = WORK_DIR / "candidate_polygons_working.shp"
safe_delete(working_polygons)
arcpy.management.CopyFeatures(str(POLYGON_FC), str(working_polygons))
arcpy.management.RepairGeometry(str(working_polygons))

polygon_layer = "candidate_polygon_layer"
safe_delete(polygon_layer)
arcpy.management.MakeFeatureLayer(str(working_polygons), polygon_layer)


# -----------------------------------------------------------------------------
# 6. Zonal-statistics extraction
# -----------------------------------------------------------------------------

def extract_statistics(
    raster,
    indicator,
    district_cfg,
    season,
    year,
    district_index,
):
    """Extract polygon-level statistics for one processed raster."""
    aoi = DISTRICT_AOI_DIR / district_cfg["aoi_file"]

    arcpy.management.SelectLayerByAttribute(
        polygon_layer, "CLEAR_SELECTION"
    )
    arcpy.management.SelectLayerByLocation(
        in_layer=polygon_layer,
        overlap_type="INTERSECT",
        select_features=str(aoi),
        selection_type="NEW_SELECTION",
    )

    n_selected = int(arcpy.management.GetCount(polygon_layer)[0])

    qa = {
        "indicator": indicator,
        "province": district_cfg["province"],
        "district": district_cfg["district"],
        "year": year,
        "season": season,
        "selected_polygons": n_selected,
        "rows_output": 0,
        "status": "failed",
        "message": "",
    }

    if n_selected == 0:
        qa["status"] = "no_polygons"
        qa["message"] = "no candidate polygons intersect district boundary"
        return [], qa

    selected_fc = f"memory/selected_polygons_{district_index}"
    safe_delete(selected_fc)
    arcpy.management.CopyFeatures(polygon_layer, selected_fc)

    attributes = {}
    fields = [POLYGON_ID_FIELD, SITE_GROUP_FIELD, SITE_ROLE_FIELD]

    with arcpy.da.SearchCursor(selected_fc, fields) as cursor:
        for polygon_id, site_group, site_role in cursor:
            role = str(site_role).strip().lower()
            if role not in VALID_ROLES:
                raise ValueError(
                    f"Unexpected SITE_ROLE '{site_role}' for polygon "
                    f"{polygon_id}. Expected treatment/reference."
                )

            attributes[str(polygon_id)] = {
                "polygon_id": str(polygon_id),
                "site_group": str(site_group).strip(),
                "site_role": role,
            }

    table = f"memory/zonal_{indicator.lower()}_{district_index}_{year}_{season}"
    safe_delete(table)

    try:
        arcpy.env.snapRaster = str(raster)
        arcpy.env.cellSize = str(raster)
        arcpy.env.mask = None
        arcpy.env.extent = None

        ZonalStatisticsAsTable(
            in_zone_data=selected_fc,
            zone_field=POLYGON_ID_FIELD,
            in_value_raster=str(raster),
            out_table=table,
            ignore_nodata="DATA",
            statistics_type="ALL",
        )

        field_lookup = {
            f.name.upper(): f.name for f in arcpy.ListFields(table)
        }

        required = [POLYGON_ID_FIELD.upper(), "COUNT", "MIN", "MAX", "MEAN"]
        missing = [field for field in required if field not in field_lookup]
        if missing:
            raise RuntimeError(
                f"Missing zonal-statistics fields: {missing}"
            )

        optional = [
            field for field in ["RANGE", "STD", "SUM"]
            if field in field_lookup
        ]
        read_upper = required + optional
        read_actual = [field_lookup[field] for field in read_upper]

        cell_area = raster_cell_area(raster)
        rows = []

        with arcpy.da.SearchCursor(table, read_actual) as cursor:
            for values in cursor:
                record = dict(zip(read_upper, values))
                polygon_id = str(record[POLYGON_ID_FIELD.upper()])
                attrs = attributes.get(polygon_id)

                if attrs is None:
                    raise RuntimeError(
                        f"Polygon ID {polygon_id} missing from attribute lookup."
                    )

                count = record.get("COUNT")
                prefix = indicator.lower()

                rows.append({
                    "polygon_id": attrs["polygon_id"],
                    "site_group": attrs["site_group"],
                    "site_role": attrs["site_role"],
                    "province": district_cfg["province"],
                    "district": district_cfg["district"],
                    "year": year,
                    "season": season,
                    f"{prefix}_source_raster": Path(raster).name,
                    f"{prefix}_pixel_count": count,
                    f"{prefix}_pixel_area_m2":
                        float(count) * cell_area if count is not None else None,
                    f"{prefix}_min": record.get("MIN"),
                    f"{prefix}_max": record.get("MAX"),
                    f"{prefix}_range": record.get("RANGE"),
                    f"{prefix}_mean": record.get("MEAN"),
                    f"{prefix}_std": record.get("STD"),
                    f"{prefix}_sum": record.get("SUM"),
                })

        qa["rows_output"] = len(rows)
        qa["status"] = "ok" if rows else "empty"
        qa["message"] = ""
        return rows, qa

    except Exception as exc:
        qa["message"] = f"{exc} | {traceback.format_exc(limit=1)}"
        return [], qa

    finally:
        safe_delete(table)
        safe_delete(selected_fc)
        reset_raster_environment()


# -----------------------------------------------------------------------------
# 7. Run processing and extraction
# -----------------------------------------------------------------------------

indicator_rows = {"NDVI": [], "BSI": []}
raster_qa = []
extraction_qa = []

for district_index, district_cfg in enumerate(DISTRICTS, start=1):
    for year in YEARS:
        for season in SEASONS:
            for indicator in INDICATORS:
                raster, raster_status = process_raster(
                    indicator=indicator,
                    district_cfg=district_cfg,
                    season=season,
                    year=year,
                )
                raster_qa.append(raster_status)

                if raster is None:
                    extraction_qa.append({
                        "indicator": indicator,
                        "province": district_cfg["province"],
                        "district": district_cfg["district"],
                        "year": year,
                        "season": season,
                        "selected_polygons": 0,
                        "rows_output": 0,
                        "status": "not_run",
                        "message": "raster processing failed",
                    })
                    continue

                rows, extraction_status = extract_statistics(
                    raster=raster,
                    indicator=indicator,
                    district_cfg=district_cfg,
                    season=season,
                    year=year,
                    district_index=district_index,
                )
                indicator_rows[indicator].extend(rows)
                extraction_qa.append(extraction_status)


# -----------------------------------------------------------------------------
# 8. Combine NDVI and BSI polygon statistics
# -----------------------------------------------------------------------------

ndvi = pd.DataFrame(indicator_rows["NDVI"])
bsi = pd.DataFrame(indicator_rows["BSI"])

keys = [
    "polygon_id",
    "site_group",
    "site_role",
    "province",
    "district",
    "year",
    "season",
]

if ndvi.empty:
    raise RuntimeError("No NDVI polygon statistics were extracted.")

if bsi.empty:
    raise RuntimeError("No BSI polygon statistics were extracted.")

if ndvi.duplicated(keys).any():
    raise RuntimeError("Duplicate NDVI polygon-year-season records detected.")

if bsi.duplicated(keys).any():
    raise RuntimeError("Duplicate BSI polygon-year-season records detected.")

combined = ndvi.merge(
    bsi,
    on=keys,
    how="outer",
    validate="one_to_one",
    indicator="indicator_coverage",
)

combined["has_ndvi"] = combined["ndvi_mean"].notna().astype(int)
combined["has_bsi"] = combined["bsi_mean"].notna().astype(int)

combined = combined.sort_values(
    ["site_group", "site_role", "polygon_id", "year", "season"]
).reset_index(drop=True)

combined.to_csv(OUT_LONG, index=False, encoding="utf-8-sig")
pd.DataFrame(raster_qa).to_csv(
    OUT_RASTER_QA, index=False, encoding="utf-8-sig"
)
pd.DataFrame(extraction_qa).to_csv(
    OUT_EXTRACTION_QA, index=False, encoding="utf-8-sig"
)

matched_rows = int((combined["indicator_coverage"] == "both").sum())
ndvi_only = int((combined["indicator_coverage"] == "left_only").sum())
bsi_only = int((combined["indicator_coverage"] == "right_only").sum())

print("Polygon processing complete")
print(f"Combined rows: {len(combined)}")
print(f"Rows with both NDVI and BSI: {matched_rows}")
print(f"NDVI-only rows: {ndvi_only}")
print(f"BSI-only rows: {bsi_only}")
print(f"Output: {OUT_LONG}")

arcpy.CheckInExtension("Spatial")
