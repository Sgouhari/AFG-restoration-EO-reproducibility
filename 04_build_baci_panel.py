# -*- coding: utf-8 -*-
"""
Build the final BACI analytical panel.

This script filters the combined NDVI/BSI polygon observations to the final
matched sample selected in 03_reference_matching.py and creates the
2017–2024 panel used in the BACI analyses.

The panel is built from the unique selected-polygon list, not from the
treatment-reference link table. References linked to more than one treatment
therefore remain single analytical units.

Inputs
------
outputs/polygon_indices_long_2017_2024.csv
outputs/reference_matching/selected_polygons_for_baci.csv
outputs/reference_matching/selected_treatment_reference_links.csv

Main output
-----------
outputs/baci/baci_master_panel_2017_2024.csv

Expected final sample
---------------------
19 treatment polygons
39 unique reference polygons
58 unique polygons
17 site groups (SG01–SG17)
42 treatment-reference links
2017–2024
peak and late seasons
928 polygon-year-season observations
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Repository paths
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_ROOT = SCRIPT_DIR.parent

OUTPUT_DIR = REPO_ROOT / "outputs"
MATCHING_DIR = OUTPUT_DIR / "reference_matching"
BACI_DIR = OUTPUT_DIR / "baci"
BACI_DIR.mkdir(parents=True, exist_ok=True)

INDEX_LONG_CSV = OUTPUT_DIR / "polygon_indices_long_2017_2024.csv"
SELECTED_POLYGONS_CSV = MATCHING_DIR / "selected_polygons_for_baci.csv"
SELECTED_LINKS_CSV = MATCHING_DIR / "selected_treatment_reference_links.csv"

OUT_MASTER = BACI_DIR / "baci_master_panel_2017_2024.csv"
OUT_POLYGON_LOOKUP = BACI_DIR / "baci_polygon_lookup.csv"
OUT_QA_SUMMARY = BACI_DIR / "baci_panel_qa_summary.csv"
OUT_QA = BACI_DIR / "baci_panel_qa.txt"


# -----------------------------------------------------------------------------
# 2. Analysis settings
# -----------------------------------------------------------------------------

VALID_YEARS = list(range(2017, 2025))
VALID_SEASONS = ["peak", "late"]

BASELINE_YEARS = [2017, 2018]
IMPLEMENTATION_YEARS = [2019, 2020, 2021]
POST_IMPLEMENTATION_YEARS = [2022, 2023, 2024]

EXPECTED_TREATMENTS = 19
EXPECTED_REFERENCES = 39
EXPECTED_POLYGONS = 58
EXPECTED_SITE_GROUPS = 17
EXPECTED_LINKS = 42
EXPECTED_ROWS = 928
EXPECTED_SHARED_REFERENCES = 3

EXPECTED_SITE_CODES = [f"SG{i:02d}" for i in range(1, 18)]

STRICT_SAMPLE_CHECK = True


# -----------------------------------------------------------------------------
# 3. Helpers
# -----------------------------------------------------------------------------

def require_file(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_columns(df, columns, label):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {label}: {missing}")


def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def standardise_role(value):
    role = clean_text(value).lower()
    if role in {"treatment", "treated", "intervention"}:
        return "treatment"
    if role in {"reference", "control", "untreated"}:
        return "reference"
    return role


def assign_phase(year):
    if year in BASELINE_YEARS:
        return "baseline"
    if year in IMPLEMENTATION_YEARS:
        return "implementation"
    if year in POST_IMPLEMENTATION_YEARS:
        return "post_implementation"
    return "outside_period"


def assign_baci_period(year):
    if year in BASELINE_YEARS:
        return "before"
    if year in IMPLEMENTATION_YEARS:
        return "implementation"
    if year in POST_IMPLEMENTATION_YEARS:
        return "after"
    return "outside_period"


def stable_numeric_id(series):
    values = sorted(series.dropna().astype(str).unique())
    lookup = {value: i + 1 for i, value in enumerate(values)}
    return series.astype(str).map(lookup)


def pass_fail(condition):
    return "PASS" if condition else "CHECK"


# -----------------------------------------------------------------------------
# 4. Load and validate inputs
# -----------------------------------------------------------------------------

require_file(INDEX_LONG_CSV, "Combined NDVI/BSI polygon table")
require_file(SELECTED_POLYGONS_CSV, "Selected polygon table")
require_file(SELECTED_LINKS_CSV, "Selected treatment-reference links")

indices = pd.read_csv(INDEX_LONG_CSV, encoding="utf-8-sig").copy()
selected = pd.read_csv(SELECTED_POLYGONS_CSV, encoding="utf-8-sig").copy()
links = pd.read_csv(SELECTED_LINKS_CSV, encoding="utf-8-sig").copy()

require_columns(
    indices,
    [
        "polygon_id",
        "site_group",
        "site_role",
        "year",
        "season",
        "ndvi_mean",
        "bsi_mean",
    ],
    "polygon indices",
)

require_columns(
    selected,
    [
        "polygon_id",
        "site_group",
        "site_role",
        "terrain_valid",
    ],
    "selected polygons",
)

require_columns(
    links,
    [
        "site_group",
        "treatment_id",
        "reference_id",
    ],
    "selected treatment-reference links",
)


# -----------------------------------------------------------------------------
# 5. Standardise identifiers and roles
# -----------------------------------------------------------------------------

for frame in [indices, selected]:
    frame["polygon_id"] = frame["polygon_id"].apply(clean_text)
    frame["site_group"] = frame["site_group"].apply(clean_text).str.upper()
    frame["site_role"] = frame["site_role"].apply(standardise_role)

for column in ["site_group", "treatment_id", "reference_id"]:
    links[column] = links[column].apply(clean_text)

links["site_group"] = links["site_group"].str.upper()

indices["year"] = pd.to_numeric(indices["year"], errors="coerce")
indices["season"] = indices["season"].apply(clean_text).str.lower()

indices = indices[
    indices["year"].isin(VALID_YEARS)
    & indices["season"].isin(VALID_SEASONS)
].copy()

indices["year"] = indices["year"].astype(int)

selected["terrain_valid"] = pd.to_numeric(
    selected["terrain_valid"], errors="coerce"
)

valid_roles = {"treatment", "reference"}

if not set(selected["site_role"].unique()).issubset(valid_roles):
    raise ValueError(
        "Selected polygons contain unexpected site_role values: "
        f"{sorted(set(selected['site_role']) - valid_roles)}"
    )

if not set(indices["site_role"].unique()).issubset(valid_roles):
    raise ValueError(
        "Polygon indices contain unexpected site_role values: "
        f"{sorted(set(indices['site_role']) - valid_roles)}"
    )


# -----------------------------------------------------------------------------
# 6. Validate the selected sample and link structure
# -----------------------------------------------------------------------------

duplicate_selected = int(selected.duplicated(subset=["polygon_id"]).sum())
if duplicate_selected:
    duplicates = selected.loc[
        selected.duplicated(subset=["polygon_id"], keep=False),
        ["polygon_id", "site_group", "site_role"],
    ]
    raise ValueError(
        "Selected polygon table must contain one row per unique polygon.\n"
        f"{duplicates.head(20)}"
    )

duplicate_links = int(
    links.duplicated(
        subset=["site_group", "treatment_id", "reference_id"]
    ).sum()
)
if duplicate_links:
    raise ValueError(
        "Duplicate treatment-reference links were found in the final link table."
    )

selected_site_codes = sorted(selected["site_group"].unique().tolist())
invalid_site_codes = [
    code for code in selected_site_codes
    if code not in EXPECTED_SITE_CODES
]

if invalid_site_codes:
    raise ValueError(
        "Site-group codes must use the same SG01-SG17 scheme as the "
        f"Supplementary Information. Invalid values: {invalid_site_codes}"
    )

n_selected_treatments = int(
    selected.loc[
        selected["site_role"] == "treatment", "polygon_id"
    ].nunique()
)
n_selected_references = int(
    selected.loc[
        selected["site_role"] == "reference", "polygon_id"
    ].nunique()
)
n_selected_polygons = int(selected["polygon_id"].nunique())
n_selected_sites = int(selected["site_group"].nunique())

n_links = len(links)
n_link_treatments = int(links["treatment_id"].nunique())
n_link_references = int(links["reference_id"].nunique())

reference_link_counts = (
    links.groupby("reference_id")["treatment_id"]
    .nunique()
    .sort_values(ascending=False)
)
shared_reference_ids = reference_link_counts[
    reference_link_counts > 1
].index.tolist()

shared_reference_site_counts = (
    links[links["reference_id"].isin(shared_reference_ids)]
    .groupby("reference_id")["site_group"]
    .nunique()
)

shared_refs_with_multiple_site_groups = int(
    (shared_reference_site_counts > 1).sum()
)


# -----------------------------------------------------------------------------
# 7. Filter to the unique selected polygons
# -----------------------------------------------------------------------------

selected_ids = set(selected["polygon_id"])
index_ids = set(indices["polygon_id"])

missing_selected_ids = sorted(selected_ids - index_ids)

if missing_selected_ids:
    raise ValueError(
        "Selected polygons are missing from the combined NDVI/BSI table. "
        f"Examples: {missing_selected_ids[:20]}"
    )

index_duplicates = int(
    indices.duplicated(
        subset=["polygon_id", "year", "season"]
    ).sum()
)

if index_duplicates:
    duplicates = indices.loc[
        indices.duplicated(
            subset=["polygon_id", "year", "season"], keep=False
        ),
        ["polygon_id", "year", "season"],
    ]
    raise ValueError(
        "The combined NDVI/BSI table contains duplicate "
        "polygon-year-season records.\n"
        f"{duplicates.head(20)}"
    )

master = indices[
    indices["polygon_id"].isin(selected_ids)
].copy()

selected_meta = selected.copy().rename(columns={
    "site_group": "selected_site_group",
    "site_role": "selected_site_role",
})

metadata_columns = [
    column for column in selected_meta.columns
    if column not in {"site_group", "site_role"}
]

master = master.merge(
    selected_meta[metadata_columns],
    on="polygon_id",
    how="left",
    validate="many_to_one",
)

# Confirm that the identity fields carried from Script 02 agree with the final
# selection produced by Script 03.
site_group_mismatch = (
    master["site_group"].str.upper()
    != master["selected_site_group"].str.upper()
)
site_role_mismatch = (
    master["site_role"].apply(standardise_role)
    != master["selected_site_role"].apply(standardise_role)
)

if site_group_mismatch.any():
    raise ValueError(
        "SITE_GROUP differs between Script 02 observations and "
        "Script 03 selected-polygon metadata."
    )

if site_role_mismatch.any():
    raise ValueError(
        "SITE_ROLE differs between Script 02 observations and "
        "Script 03 selected-polygon metadata."
    )


# -----------------------------------------------------------------------------
# 8. Create final BACI variables
# -----------------------------------------------------------------------------

master["POLYGON_ID"] = master["polygon_id"]
master["POLYGON_NUM_ID"] = stable_numeric_id(master["POLYGON_ID"])

master["SITE_GROUP"] = master["selected_site_group"].str.upper()
master["SITE_ROLE"] = master["selected_site_role"].apply(standardise_role)

# Internal model coding retains "control" for the untreated BACI comparison
# group, while manuscript-facing terminology remains "reference".
master["SITE_TYPE"] = np.where(
    master["SITE_ROLE"] == "treatment",
    "treatment",
    "control",
)

master["treat"] = (master["SITE_ROLE"] == "treatment").astype(int)
master["reference"] = (master["SITE_ROLE"] == "reference").astype(int)

master["phase_3"] = master["year"].apply(assign_phase)
master["period_baci"] = master["year"].apply(assign_baci_period)

master["post"] = (
    master["phase_3"] == "post_implementation"
).astype(int)

master["year_2018"] = (master["year"] == 2018).astype(int)

master["terrain_valid"] = pd.to_numeric(
    master["terrain_valid"], errors="coerce"
)

master["complete_ndvi_bsi"] = (
    master["ndvi_mean"].notna()
    & master["bsi_mean"].notna()
).astype(int)

# Public district label retained for later sensitivity extraction.
if "district" in master.columns:
    master["DISTRICT_KEY"] = master["district"].apply(clean_text)
else:
    master["DISTRICT_KEY"] = ""


# -----------------------------------------------------------------------------
# 9. Order and sort the analytical panel
# -----------------------------------------------------------------------------

front_columns = [
    "POLYGON_NUM_ID",
    "POLYGON_ID",
    "SITE_GROUP",
    "SITE_TYPE",
    "SITE_ROLE",
    "treat",
    "reference",
    "year",
    "season",
    "phase_3",
    "period_baci",
    "post",
    "year_2018",
    "DISTRICT_KEY",
    "province",
    "district",
    "terrain_valid",
    "complete_ndvi_bsi",
    "ndvi_mean",
    "bsi_mean",
    "ndvi_min",
    "ndvi_max",
    "ndvi_range",
    "ndvi_std",
    "bsi_min",
    "bsi_max",
    "bsi_range",
    "bsi_std",
    "ndvi_pixel_count",
    "bsi_pixel_count",
    "ndvi_pixel_area_m2",
    "bsi_pixel_area_m2",
    "elevation_m",
    "slope_deg",
    "northness",
    "eastness",
]

front_columns = [
    column for column in front_columns
    if column in master.columns
]

drop_columns = {
    "polygon_id",
    "site_group",
    "site_role",
    "selected_site_group",
    "selected_site_role",
}

remaining_columns = [
    column for column in master.columns
    if column not in front_columns
    and column not in drop_columns
]

master = master[front_columns + remaining_columns].copy()

master = master.sort_values(
    [
        "SITE_GROUP",
        "SITE_TYPE",
        "POLYGON_ID",
        "year",
        "season",
    ]
).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 10. Final QA
# -----------------------------------------------------------------------------

n_rows = len(master)
n_polygons = int(master["POLYGON_ID"].nunique())
n_sites = int(master["SITE_GROUP"].nunique())

n_treatments = int(
    master.loc[
        master["SITE_TYPE"] == "treatment", "POLYGON_ID"
    ].nunique()
)
n_references = int(
    master.loc[
        master["SITE_TYPE"] == "control", "POLYGON_ID"
    ].nunique()
)

duplicate_panel_rows = int(
    master.duplicated(
        subset=["POLYGON_ID", "year", "season"]
    ).sum()
)

missing_ndvi = int(master["ndvi_mean"].isna().sum())
missing_bsi = int(master["bsi_mean"].isna().sum())
invalid_terrain_rows = int((master["terrain_valid"] != 1).sum())

years_present = sorted(master["year"].unique().tolist())
seasons_present = sorted(master["season"].unique().tolist())
site_codes_present = sorted(master["SITE_GROUP"].unique().tolist())

phase_counts = master["phase_3"].value_counts().to_dict()

expected_phase_counts = {
    "baseline": EXPECTED_POLYGONS * len(BASELINE_YEARS) * len(VALID_SEASONS),
    "implementation":
        EXPECTED_POLYGONS * len(IMPLEMENTATION_YEARS) * len(VALID_SEASONS),
    "post_implementation":
        EXPECTED_POLYGONS * len(POST_IMPLEMENTATION_YEARS) * len(VALID_SEASONS),
}

checks = {
    "selected_treatments": n_selected_treatments == EXPECTED_TREATMENTS,
    "selected_references": n_selected_references == EXPECTED_REFERENCES,
    "selected_polygons": n_selected_polygons == EXPECTED_POLYGONS,
    "selected_site_groups": n_selected_sites == EXPECTED_SITE_GROUPS,
    "link_count": n_links == EXPECTED_LINKS,
    "link_treatment_count": n_link_treatments == EXPECTED_TREATMENTS,
    "link_reference_count": n_link_references == EXPECTED_REFERENCES,
    "shared_reference_count":
        len(shared_reference_ids) == EXPECTED_SHARED_REFERENCES,
    "shared_references_within_site_groups":
        shared_refs_with_multiple_site_groups == 0,
    "panel_rows": n_rows == EXPECTED_ROWS,
    "panel_polygons": n_polygons == EXPECTED_POLYGONS,
    "panel_treatments": n_treatments == EXPECTED_TREATMENTS,
    "panel_references": n_references == EXPECTED_REFERENCES,
    "panel_site_groups": n_sites == EXPECTED_SITE_GROUPS,
    "site_codes_match_supplement":
        site_codes_present == EXPECTED_SITE_CODES,
    "years_complete": years_present == VALID_YEARS,
    "seasons_complete": seasons_present == sorted(VALID_SEASONS),
    "no_duplicate_polygon_year_season": duplicate_panel_rows == 0,
    "no_missing_ndvi": missing_ndvi == 0,
    "no_missing_bsi": missing_bsi == 0,
    "terrain_valid": invalid_terrain_rows == 0,
    "phase_counts":
        all(
            phase_counts.get(key, 0) == value
            for key, value in expected_phase_counts.items()
        ),
}

qa_rows = [
    {"metric": key, "status": pass_fail(value)}
    for key, value in checks.items()
]

qa_rows.extend([
    {"metric": "rows", "value": n_rows},
    {"metric": "unique_polygons", "value": n_polygons},
    {"metric": "treatment_polygons", "value": n_treatments},
    {"metric": "reference_polygons", "value": n_references},
    {"metric": "site_groups", "value": n_sites},
    {"metric": "treatment_reference_links", "value": n_links},
    {"metric": "shared_references", "value": len(shared_reference_ids)},
    {"metric": "missing_ndvi_rows", "value": missing_ndvi},
    {"metric": "missing_bsi_rows", "value": missing_bsi},
    {"metric": "duplicate_panel_rows", "value": duplicate_panel_rows},
])

pd.DataFrame(qa_rows).to_csv(
    OUT_QA_SUMMARY,
    index=False,
    encoding="utf-8-sig",
)


# -----------------------------------------------------------------------------
# 11. Save outputs
# -----------------------------------------------------------------------------

master.to_csv(
    OUT_MASTER,
    index=False,
    encoding="utf-8-sig",
)

polygon_lookup = (
    master[
        [
            "POLYGON_ID",
            "POLYGON_NUM_ID",
            "SITE_GROUP",
            "SITE_TYPE",
            "SITE_ROLE",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        ["SITE_GROUP", "SITE_TYPE", "POLYGON_ID"]
    )
    .reset_index(drop=True)
)

polygon_lookup.to_csv(
    OUT_POLYGON_LOOKUP,
    index=False,
    encoding="utf-8-sig",
)

with open(OUT_QA, "w", encoding="utf-8") as qa:
    qa.write("BACI analytical panel QA\n")
    qa.write("=" * 72 + "\n\n")

    qa.write("Panel construction\n")
    qa.write("-" * 72 + "\n")
    qa.write(
        "The analytical panel was built from the unique selected-polygon "
        "list rather than the treatment-reference link table. Shared "
        "references therefore occur once per polygon-year-season.\n\n"
    )

    qa.write("Final sample\n")
    qa.write("-" * 72 + "\n")
    qa.write(f"Rows: {n_rows}\n")
    qa.write(f"Unique polygons: {n_polygons}\n")
    qa.write(f"Treatment polygons: {n_treatments}\n")
    qa.write(f"Reference polygons: {n_references}\n")
    qa.write(f"Site groups: {n_sites}\n")
    qa.write(f"Treatment-reference links: {n_links}\n")
    qa.write(f"Shared references: {len(shared_reference_ids)}\n")
    qa.write(
        "Shared references assigned across multiple site groups: "
        f"{shared_refs_with_multiple_site_groups}\n\n"
    )

    qa.write("Time structure\n")
    qa.write("-" * 72 + "\n")
    qa.write(f"Years: {years_present}\n")
    qa.write(f"Seasons: {seasons_present}\n")
    qa.write(f"Phase counts: {phase_counts}\n\n")

    qa.write("Completeness\n")
    qa.write("-" * 72 + "\n")
    qa.write(f"Missing NDVI rows: {missing_ndvi}\n")
    qa.write(f"Missing BSI rows: {missing_bsi}\n")
    qa.write(f"Duplicate polygon-year-season rows: {duplicate_panel_rows}\n")
    qa.write(f"Rows with invalid terrain flag: {invalid_terrain_rows}\n\n")

    qa.write("Checks\n")
    qa.write("-" * 72 + "\n")
    for key, value in checks.items():
        qa.write(f"{key}: {pass_fail(value)}\n")

if STRICT_SAMPLE_CHECK and not all(checks.values()):
    failed = [key for key, value in checks.items() if not value]
    raise RuntimeError(
        "The BACI panel does not match the published analytical sample. "
        f"Failed checks: {failed}. See {OUT_QA}"
    )

print("BACI panel construction complete")
print(f"Master panel: {OUT_MASTER}")
print(f"Polygon lookup: {OUT_POLYGON_LOOKUP}")
print(f"QA: {OUT_QA}")
