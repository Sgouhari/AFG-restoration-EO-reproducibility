# -*- coding: utf-8 -*-
"""
Primary BACI analysis for the final matched restoration sample.

This script reproduces the principal statistical analyses reported in the
Afghanistan restoration monitoring study.

Analyses
--------
1. Annual treatment/reference trajectories for NDVI and BSI.
2. Descriptive baseline-to-post BACI contrasts at pooled and site-group level.
3. Primary mixed-effects BACI models.
4. Fixed-effects robustness models using:
   a. all years, and
   b. baseline and post-implementation years only.
5. Baseline sensitivity diagnostic comparing 2017 and 2018.

Input
-----
outputs/baci/baci_master_panel_2017_2024.csv

Expected analytical sample
--------------------------
19 treatment polygons
39 reference polygons
58 polygons
17 site groups
2017–2024
peak and late seasons
928 polygon-year-season observations
"""

from pathlib import Path
import traceback
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# -----------------------------------------------------------------------------
# 1. Repository paths
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_ROOT = SCRIPT_DIR.parent

BACI_DIR = REPO_ROOT / "outputs" / "baci"
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = BACI_DIR / "baci_master_panel_2017_2024.csv"

OUT_ANNUAL = ANALYSIS_DIR / "annual_trajectories.csv"
OUT_CONTRASTS = ANALYSIS_DIR / "baci_contrasts.csv"
OUT_MODEL_RESULTS = ANALYSIS_DIR / "baci_model_results.csv"
OUT_MODEL_DIAGNOSTICS = ANALYSIS_DIR / "baci_model_diagnostics.csv"
OUT_QA = ANALYSIS_DIR / "baci_primary_analysis_qa.txt"


# -----------------------------------------------------------------------------
# 2. Analysis settings
# -----------------------------------------------------------------------------

VALID_YEARS = list(range(2017, 2025))
VALID_SEASONS = ["peak", "late"]

BASELINE_YEARS = [2017, 2018]
POST_YEARS = [2022, 2023, 2024]

VALID_PHASES = ["baseline", "implementation", "post_implementation"]
PHASE_ORDER = ["baseline", "implementation", "post_implementation"]

EXPECTED_TREATMENTS = 19
EXPECTED_REFERENCES = 39
EXPECTED_POLYGONS = 58
EXPECTED_SITE_GROUPS = 17
EXPECTED_ROWS = 928

STRICT_SAMPLE_CHECK = True

OUTCOMES = {
    "NDVI": {
        "column": "ndvi_mean",
        "expected_direction": "positive",
    },
    "BSI": {
        "column": "bsi_mean",
        "expected_direction": "negative",
    },
}


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


def standardise_site_type(value):
    value = "" if pd.isna(value) else str(value).strip().lower()
    if value in {"treatment", "treated", "intervention"}:
        return "treatment"
    if value in {"control", "reference", "matched control", "matched reference"}:
        return "control"
    return value


def prepare_role_fields(df):
    df = df.copy()
    df["SITE_TYPE"] = df["SITE_TYPE"].apply(standardise_site_type)

    if "SITE_ROLE" in df.columns:
        role = df["SITE_ROLE"].astype(str).str.strip().str.lower()
        df["SITE_ROLE"] = role.replace({
            "control": "reference",
            "matched control": "reference",
            "matched reference": "reference",
        })
    else:
        df["SITE_ROLE"] = np.where(
            df["SITE_TYPE"] == "control",
            "reference",
            "treatment",
        )

    df["treat"] = (df["SITE_TYPE"] == "treatment").astype(int)
    df["reference"] = (df["SITE_TYPE"] == "control").astype(int)
    df["role_display"] = np.where(
        df["treat"] == 1,
        "Treatment",
        "Reference",
    )

    return df


def phase_from_year(year):
    if year in BASELINE_YEARS:
        return "baseline"
    if year in POST_YEARS:
        return "post_implementation"
    return "implementation"


def concat_or_empty(frames):
    frames = [
        frame for frame in frames
        if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def round_numeric(df, digits=6):
    if df.empty:
        return df
    result = df.copy()
    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].round(digits)
    return result


def extract_coefficient(
    fit,
    term,
    model,
    model_role,
    outcome,
    season,
    effect_phase,
):
    if fit is None or term not in fit.params.index:
        return None

    ci = fit.conf_int()

    return {
        "model": model,
        "model_role": model_role,
        "outcome": outcome,
        "season": season,
        "effect_phase": effect_phase,
        "term": term,
        "coef": fit.params[term],
        "se": fit.bse[term],
        "t_or_z": fit.tvalues[term],
        "p_value": fit.pvalues[term],
        "ci_low": ci.loc[term, 0],
        "ci_high": ci.loc[term, 1],
        "n_obs": int(fit.nobs),
    }


def diagnostic_row(
    fit,
    model,
    outcome,
    season,
    status,
    message="",
    warning_count=0,
    warning_messages="",
):
    row = {
        "model": model,
        "outcome": outcome,
        "season": season,
        "status": status,
        "message": message,
        "warning_count": int(warning_count),
        "warning_messages": warning_messages,
        "converged": np.nan,
        "n_obs": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "llf": np.nan,
    }

    if fit is None:
        return row

    try:
        row["converged"] = getattr(fit, "converged", np.nan)
    except Exception:
        pass

    try:
        row["n_obs"] = int(fit.nobs)
    except Exception:
        pass

    for attr in ["aic", "bic", "llf"]:
        try:
            row[attr] = getattr(fit, attr)
        except Exception:
            pass

    return row


def pass_fail(condition):
    return "PASS" if condition else "CHECK"


# -----------------------------------------------------------------------------
# 4. Read and validate the final BACI panel
# -----------------------------------------------------------------------------

require_file(INPUT_CSV, "Final BACI master panel")

df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig").copy()

required_columns = [
    "POLYGON_ID",
    "SITE_GROUP",
    "SITE_TYPE",
    "year",
    "season",
    "phase_3",
    "period_baci",
    "ndvi_mean",
    "bsi_mean",
    "terrain_valid",
]

require_columns(df, required_columns, "final BACI master panel")

df["POLYGON_ID"] = df["POLYGON_ID"].astype(str).str.strip()
df["SITE_GROUP"] = df["SITE_GROUP"].astype(str).str.strip().str.upper()
df["SITE_TYPE"] = df["SITE_TYPE"].astype(str).str.strip().str.lower()
df["season"] = df["season"].astype(str).str.strip().str.lower()
df["phase_3"] = df["phase_3"].astype(str).str.strip().str.lower()
df["period_baci"] = df["period_baci"].astype(str).str.strip().str.lower()

if "SITE_ROLE" in df.columns:
    df["SITE_ROLE"] = (
        df["SITE_ROLE"].astype(str).str.strip().str.lower()
    )

for column in [
    "year",
    "ndvi_mean",
    "bsi_mean",
    "terrain_valid",
]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=["year", "terrain_valid"]).copy()
df["year"] = df["year"].astype(int)

df = df[
    (df["terrain_valid"] == 1)
    & df["year"].isin(VALID_YEARS)
    & df["season"].isin(VALID_SEASONS)
    & df["phase_3"].isin(VALID_PHASES)
].copy()

df = prepare_role_fields(df)

df = df[
    df["SITE_TYPE"].isin(["treatment", "control"])
].copy()

df["phase_3"] = pd.Categorical(
    df["phase_3"],
    categories=PHASE_ORDER,
    ordered=True,
)

df["post"] = np.where(
    df["phase_3"].astype(str) == "post_implementation",
    1,
    0,
)

df["year_2018"] = np.where(df["year"] == 2018, 1, 0)


# -----------------------------------------------------------------------------
# 5. Sample QA
# -----------------------------------------------------------------------------

n_rows = len(df)
n_polygons = int(df["POLYGON_ID"].nunique())
n_sites = int(df["SITE_GROUP"].nunique())

n_treatments = int(
    df.loc[df["treat"] == 1, "POLYGON_ID"].nunique()
)
n_references = int(
    df.loc[df["treat"] == 0, "POLYGON_ID"].nunique()
)

years_present = sorted(df["year"].unique().tolist())
seasons_present = sorted(df["season"].unique().tolist())

duplicate_rows = int(
    df.duplicated(
        subset=["POLYGON_ID", "year", "season"]
    ).sum()
)

missing_ndvi = int(df["ndvi_mean"].isna().sum())
missing_bsi = int(df["bsi_mean"].isna().sum())

sample_checks = {
    "row_count": n_rows == EXPECTED_ROWS,
    "polygon_count": n_polygons == EXPECTED_POLYGONS,
    "treatment_count": n_treatments == EXPECTED_TREATMENTS,
    "reference_count": n_references == EXPECTED_REFERENCES,
    "site_group_count": n_sites == EXPECTED_SITE_GROUPS,
    "year_coverage": years_present == VALID_YEARS,
    "season_coverage": seasons_present == sorted(VALID_SEASONS),
    "no_duplicate_polygon_year_season": duplicate_rows == 0,
    "no_missing_ndvi": missing_ndvi == 0,
    "no_missing_bsi": missing_bsi == 0,
}

if STRICT_SAMPLE_CHECK and not all(sample_checks.values()):
    failed = [
        name for name, passed in sample_checks.items()
        if not passed
    ]
    raise RuntimeError(
        "Final BACI panel failed strict sample checks: "
        f"{failed}"
    )


# -----------------------------------------------------------------------------
# 6. Annual treatment/reference trajectories
# -----------------------------------------------------------------------------

annual_frames = []

for outcome, config in OUTCOMES.items():
    column = config["column"]

    summary = (
        df.groupby(
            ["role_display", "year", "season"],
            as_index=False,
        )
        .agg(
            mean_value=(column, "mean"),
            std_value=(column, "std"),
            n_observations=(column, "count"),
            n_polygons=("POLYGON_ID", "nunique"),
        )
    )

    summary["outcome"] = outcome
    summary["phase_3"] = summary["year"].apply(phase_from_year)
    summary["series"] = summary["role_display"]
    summary["expected_restoration_direction"] = (
        config["expected_direction"]
    )

    annual_frames.append(
        summary[
            [
                "outcome",
                "season",
                "year",
                "phase_3",
                "series",
                "mean_value",
                "std_value",
                "n_observations",
                "n_polygons",
                "expected_restoration_direction",
            ]
        ]
    )

    wide = summary.pivot_table(
        index=["year", "season", "phase_3"],
        columns="role_display",
        values="mean_value",
    ).reset_index()

    wide.columns.name = None

    if not {"Treatment", "Reference"}.issubset(wide.columns):
        raise RuntimeError(
            f"Missing treatment/reference annual series for {outcome}"
        )

    wide["outcome"] = outcome
    wide["series"] = "Treatment_minus_Reference"
    wide["mean_value"] = (
        wide["Treatment"] - wide["Reference"]
    )
    wide["std_value"] = np.nan
    wide["n_observations"] = np.nan
    wide["n_polygons"] = np.nan
    wide["expected_restoration_direction"] = (
        config["expected_direction"]
    )

    annual_frames.append(
        wide[
            [
                "outcome",
                "season",
                "year",
                "phase_3",
                "series",
                "mean_value",
                "std_value",
                "n_observations",
                "n_polygons",
                "expected_restoration_direction",
            ]
        ]
    )

annual = concat_or_empty(annual_frames)

annual = annual.sort_values(
    ["outcome", "season", "series", "year"]
).reset_index(drop=True)

annual = round_numeric(annual)
annual.to_csv(OUT_ANNUAL, index=False, encoding="utf-8-sig")


# -----------------------------------------------------------------------------
# 7. Descriptive BACI contrasts
# -----------------------------------------------------------------------------

contrast_frames = []

for outcome, config in OUTCOMES.items():
    column = config["column"]

    baseline = df[df["year"].isin(BASELINE_YEARS)].copy()
    post = df[df["year"].isin(POST_YEARS)].copy()

    # Pooled sample
    baseline_pooled = (
        baseline.groupby(
            ["role_display", "season"],
            as_index=False,
        )[column]
        .mean()
        .rename(columns={column: "baseline_mean"})
    )

    post_pooled = (
        post.groupby(
            ["role_display", "season"],
            as_index=False,
        )[column]
        .mean()
        .rename(columns={column: "post_mean"})
    )

    pooled = baseline_pooled.merge(
        post_pooled,
        on=["role_display", "season"],
        how="inner",
    )

    pooled["change_post_minus_baseline"] = (
        pooled["post_mean"] - pooled["baseline_mean"]
    )

    treatment = pooled[
        pooled["role_display"] == "Treatment"
    ].copy()

    reference = pooled[
        pooled["role_display"] == "Reference"
    ].copy()

    pooled_baci = treatment.merge(
        reference,
        on="season",
        suffixes=("_treatment", "_reference"),
    )

    pooled_baci["baci_contrast"] = (
        pooled_baci["change_post_minus_baseline_treatment"]
        - pooled_baci["change_post_minus_baseline_reference"]
    )

    pooled_baci["summary_level"] = "pooled"
    pooled_baci["site_group"] = "all_sites"
    pooled_baci["outcome"] = outcome
    pooled_baci["expected_restoration_direction"] = (
        config["expected_direction"]
    )

    contrast_frames.append(
        pooled_baci[
            [
                "outcome",
                "season",
                "summary_level",
                "site_group",
                "baseline_mean_treatment",
                "post_mean_treatment",
                "change_post_minus_baseline_treatment",
                "baseline_mean_reference",
                "post_mean_reference",
                "change_post_minus_baseline_reference",
                "baci_contrast",
                "expected_restoration_direction",
            ]
        ]
    )

    # Site-group summaries
    baseline_site = (
        baseline.groupby(
            ["SITE_GROUP", "role_display", "season"],
            as_index=False,
        )[column]
        .mean()
        .rename(columns={column: "baseline_mean"})
    )

    post_site = (
        post.groupby(
            ["SITE_GROUP", "role_display", "season"],
            as_index=False,
        )[column]
        .mean()
        .rename(columns={column: "post_mean"})
    )

    site = baseline_site.merge(
        post_site,
        on=["SITE_GROUP", "role_display", "season"],
        how="inner",
    )

    site["change_post_minus_baseline"] = (
        site["post_mean"] - site["baseline_mean"]
    )

    treatment = site[
        site["role_display"] == "Treatment"
    ].copy()

    reference = site[
        site["role_display"] == "Reference"
    ].copy()

    site_baci = treatment.merge(
        reference,
        on=["SITE_GROUP", "season"],
        suffixes=("_treatment", "_reference"),
    )

    site_baci["baci_contrast"] = (
        site_baci["change_post_minus_baseline_treatment"]
        - site_baci["change_post_minus_baseline_reference"]
    )

    site_baci["summary_level"] = "site_group"
    site_baci["site_group"] = site_baci["SITE_GROUP"]
    site_baci["outcome"] = outcome
    site_baci["expected_restoration_direction"] = (
        config["expected_direction"]
    )

    contrast_frames.append(
        site_baci[
            [
                "outcome",
                "season",
                "summary_level",
                "site_group",
                "baseline_mean_treatment",
                "post_mean_treatment",
                "change_post_minus_baseline_treatment",
                "baseline_mean_reference",
                "post_mean_reference",
                "change_post_minus_baseline_reference",
                "baci_contrast",
                "expected_restoration_direction",
            ]
        ]
    )

contrasts = concat_or_empty(contrast_frames)

contrasts = contrasts.sort_values(
    ["outcome", "season", "summary_level", "site_group"]
).reset_index(drop=True)

contrasts = round_numeric(contrasts)
contrasts.to_csv(
    OUT_CONTRASTS,
    index=False,
    encoding="utf-8-sig",
)


# -----------------------------------------------------------------------------
# 8. Statistical model functions
# -----------------------------------------------------------------------------

def run_mixed_baci(data, outcome, column, season):
    """
    Primary mixed-effects BACI model.

    Fixed effects:
        treatment
        implementation phase
        post-implementation phase
        treatment × implementation
        treatment × post-implementation

    Random structure:
        site-group random intercept
        polygon-level variance component
    """
    subset = data[
        (data["season"] == season)
        & data[column].notna()
    ].copy()

    formula = (
        f"{column} ~ "
        "treat * C(phase_3, Treatment('baseline'))"
    )

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

    rows = []

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
        row = extract_coefficient(
            fit=fit,
            term=term,
            model="mixed_baci",
            model_role="primary",
            outcome=outcome,
            season=season,
            effect_phase=phase,
        )
        if row is not None:
            rows.append(row)

    return fit, pd.DataFrame(rows)


def run_fixed_effects_all_years(
    data,
    outcome,
    column,
    season,
):
    """
    Fixed-effects robustness model using all analytical years.

    outcome ~ treatment × post + polygon fixed effects + year fixed effects

    Standard errors are clustered by site group.
    """
    subset = data[
        (data["season"] == season)
        & data[column].notna()
    ].copy()

    formula = (
        f"{column} ~ "
        "treat:post + C(POLYGON_ID) + C(year)"
    )

    fit = smf.ols(
        formula=formula,
        data=subset,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": subset["SITE_GROUP"]},
    )

    row = extract_coefficient(
        fit=fit,
        term="treat:post",
        model="fixed_effects_did_all_years",
        model_role="robustness",
        outcome=outcome,
        season=season,
        effect_phase="post_implementation",
    )

    return (
        fit,
        pd.DataFrame([row])
        if row is not None
        else pd.DataFrame(),
    )


def run_fixed_effects_baseline_post(
    data,
    outcome,
    column,
    season,
):
    """
    Fixed-effects robustness model excluding implementation years.

    Only 2017–2018 and 2022–2024 observations are retained.
    Standard errors are clustered by site group.
    """
    subset = data[
        (data["season"] == season)
        & data["phase_3"].astype(str).isin(
            ["baseline", "post_implementation"]
        )
        & data[column].notna()
    ].copy()

    formula = (
        f"{column} ~ "
        "treat:post + C(POLYGON_ID) + C(year)"
    )

    fit = smf.ols(
        formula=formula,
        data=subset,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": subset["SITE_GROUP"]},
    )

    row = extract_coefficient(
        fit=fit,
        term="treat:post",
        model="fixed_effects_did_baseline_post_only",
        model_role="robustness",
        outcome=outcome,
        season=season,
        effect_phase="post_implementation",
    )

    return (
        fit,
        pd.DataFrame([row])
        if row is not None
        else pd.DataFrame(),
    )


def run_baseline_sensitivity(
    data,
    outcome,
    column,
    season,
):
    """
    Baseline diagnostic using only 2017 and 2018.

    The treatment × 2018 interaction tests for differential change between
    treatment and reference polygons across the two pre-intervention years.
    With two baseline years, this is a diagnostic and does not establish
    parallel pre-intervention trends.
    """
    subset = data[
        (data["season"] == season)
        & data["year"].isin(BASELINE_YEARS)
        & data[column].notna()
    ].copy()

    formula = f"{column} ~ treat * year_2018"

    fit = smf.ols(
        formula=formula,
        data=subset,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": subset["SITE_GROUP"]},
    )

    row = extract_coefficient(
        fit=fit,
        term="treat:year_2018",
        model="baseline_sensitivity",
        model_role="diagnostic",
        outcome=outcome,
        season=season,
        effect_phase="baseline_2018_vs_2017",
    )

    return (
        fit,
        pd.DataFrame([row])
        if row is not None
        else pd.DataFrame(),
    )


# -----------------------------------------------------------------------------
# 9. Run models
# -----------------------------------------------------------------------------

model_frames = []
diagnostics = []

for outcome, config in OUTCOMES.items():
    column = config["column"]

    for season in VALID_SEASONS:

        model_specs = [
            (
                "mixed_baci",
                run_mixed_baci,
            ),
            (
                "fixed_effects_did_all_years",
                run_fixed_effects_all_years,
            ),
            (
                "fixed_effects_did_baseline_post_only",
                run_fixed_effects_baseline_post,
            ),
            (
                "baseline_sensitivity",
                run_baseline_sensitivity,
            ),
        ]

        for model_name, model_function in model_specs:
            caught_warnings = []

            try:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")

                    fit, result = model_function(
                        df,
                        outcome,
                        column,
                        season,
                    )

                warning_messages = [
                    f"{item.category.__name__}: {item.message}"
                    for item in caught_warnings
                ]

                model_frames.append(result)

                diagnostics.append(
                    diagnostic_row(
                        fit=fit,
                        model=model_name,
                        outcome=outcome,
                        season=season,
                        status="success",
                        warning_count=len(warning_messages),
                        warning_messages=" | ".join(warning_messages),
                    )
                )

            except Exception as exc:
                warning_messages = [
                    f"{item.category.__name__}: {item.message}"
                    for item in caught_warnings
                ]

                diagnostics.append(
                    diagnostic_row(
                        fit=None,
                        model=model_name,
                        outcome=outcome,
                        season=season,
                        status="failed",
                        message=str(exc),
                        warning_count=len(warning_messages),
                        warning_messages=" | ".join(warning_messages),
                    )
                )

                print(
                    f"{model_name} failed for "
                    f"{outcome} | {season}"
                )
                print(traceback.format_exc(limit=2))


model_results = concat_or_empty(model_frames)
model_diagnostics = pd.DataFrame(diagnostics)

term_labels = {
    (
        "treat:C(phase_3, Treatment('baseline'))"
        "[T.implementation]"
    ): "Treatment x Implementation",
    (
        "treat:C(phase_3, Treatment('baseline'))"
        "[T.post_implementation]"
    ): "Treatment x Post",
    "treat:post": "Treatment x Post",
    "treat:year_2018": "Treatment x 2018",
}

if not model_results.empty:
    model_results["term_label"] = (
        model_results["term"].replace(term_labels)
    )

    direction_lookup = {
        outcome: config["expected_direction"]
        for outcome, config in OUTCOMES.items()
    }

    model_results["expected_restoration_direction"] = (
        model_results["outcome"].map(direction_lookup)
    )

    model_results = model_results[
        [
            "model",
            "model_role",
            "outcome",
            "season",
            "effect_phase",
            "term",
            "term_label",
            "coef",
            "se",
            "t_or_z",
            "p_value",
            "ci_low",
            "ci_high",
            "n_obs",
            "expected_restoration_direction",
        ]
    ]

model_results = round_numeric(model_results)
model_diagnostics = round_numeric(model_diagnostics)

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
# 10. Model-output QA
# -----------------------------------------------------------------------------

expected_diagnostic_rows = (
    len(OUTCOMES) * len(VALID_SEASONS) * 4
)

expected_model_result_rows = (
    # Mixed model: two interaction coefficients per outcome-season
    len(OUTCOMES) * len(VALID_SEASONS) * 2
    # Three secondary models: one coefficient per outcome-season
    + len(OUTCOMES) * len(VALID_SEASONS) * 3
)

model_checks = {
    "diagnostic_row_count":
        len(model_diagnostics) == expected_diagnostic_rows,
    "all_models_successful":
        (
            not model_diagnostics.empty
            and (model_diagnostics["status"] == "success").all()
        ),
    "model_result_row_count":
        len(model_results) == expected_model_result_rows,
}

all_checks = {
    **sample_checks,
    **model_checks,
}


# -----------------------------------------------------------------------------
# 11. QA report
# -----------------------------------------------------------------------------

with open(OUT_QA, "w", encoding="utf-8") as qa:
    qa.write("Primary BACI analysis QA\n")
    qa.write("=" * 72 + "\n\n")

    qa.write("Analytical sample\n")
    qa.write("-" * 72 + "\n")
    qa.write(f"Rows: {n_rows}\n")
    qa.write(f"Unique polygons: {n_polygons}\n")
    qa.write(f"Treatment polygons: {n_treatments}\n")
    qa.write(f"Reference polygons: {n_references}\n")
    qa.write(f"Site groups: {n_sites}\n")
    qa.write(f"Years: {years_present}\n")
    qa.write(f"Seasons: {seasons_present}\n\n")

    qa.write("Model specifications\n")
    qa.write("-" * 72 + "\n")
    qa.write(
        "Primary mixed BACI: "
        "outcome ~ treat * C(phase_3, Treatment('baseline'))\n"
    )
    qa.write("Site-group random intercept: SITE_GROUP\n")
    qa.write(
        "Polygon-level variance component: "
        "0 + C(POLYGON_ID)\n"
    )
    qa.write(
        "Fixed effects, all years: "
        "outcome ~ treat:post + C(POLYGON_ID) + C(year)\n"
    )
    qa.write(
        "Fixed effects, baseline-post only: same specification "
        "with implementation years excluded\n"
    )
    qa.write(
        "Baseline diagnostic: "
        "outcome ~ treat * year_2018, using 2017–2018 only\n"
    )
    qa.write(
        "Fixed-effects and baseline-diagnostic standard errors "
        "clustered by SITE_GROUP\n"
    )
    qa.write(
        "Model warnings are captured per fit and retained in "
        "baci_model_diagnostics.csv; warnings are not globally suppressed.\n\n"
    )

    qa.write("Checks\n")
    qa.write("-" * 72 + "\n")
    for name, passed in all_checks.items():
        qa.write(f"{name}: {pass_fail(passed)}\n")

    qa.write("\nModel status\n")
    qa.write("-" * 72 + "\n")

    if model_diagnostics.empty:
        qa.write("No model diagnostics available.\n")
    else:
        status_counts = (
            model_diagnostics
            .groupby(["model", "status"])
            .size()
            .reset_index(name="count")
        )
        for row in status_counts.itertuples(index=False):
            qa.write(
                f"{row.model} | {row.status}: {row.count}\n"
            )

        qa.write("\nCaptured model warnings\n")
        qa.write("-" * 72 + "\n")

        total_warnings = int(
            pd.to_numeric(
                model_diagnostics["warning_count"],
                errors="coerce",
            ).fillna(0).sum()
        )

        qa.write(f"Total warnings captured: {total_warnings}\n")

        warned = model_diagnostics[
            pd.to_numeric(
                model_diagnostics["warning_count"],
                errors="coerce",
            ).fillna(0) > 0
        ]

        if warned.empty:
            qa.write("No model warnings were emitted.\n")
        else:
            for row in warned.itertuples(index=False):
                qa.write(
                    f"{row.model} | {row.outcome} | {row.season} | "
                    f"{int(row.warning_count)} warning(s)\n"
                )
                qa.write(f"  {row.warning_messages}\n")

    qa.write("\nOutputs\n")
    qa.write("-" * 72 + "\n")
    for path in [
        OUT_ANNUAL,
        OUT_CONTRASTS,
        OUT_MODEL_RESULTS,
        OUT_MODEL_DIAGNOSTICS,
    ]:
        qa.write(f"{path.name}\n")


if STRICT_SAMPLE_CHECK and not all(model_checks.values()):
    failed = [
        name for name, passed in model_checks.items()
        if not passed
    ]
    raise RuntimeError(
        "One or more statistical analyses did not complete as expected. "
        f"Failed checks: {failed}. See {OUT_QA}"
    )


print("Primary BACI analysis complete")
print(f"Annual trajectories: {OUT_ANNUAL}")
print(f"BACI contrasts: {OUT_CONTRASTS}")
print(f"Model results: {OUT_MODEL_RESULTS}")
print(f"Model diagnostics: {OUT_MODEL_DIAGNOSTICS}")
print(f"QA: {OUT_QA}")
