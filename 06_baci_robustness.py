# -*- coding: utf-8 -*-
"""
Robustness analyses for the final BACI sample.

This script reproduces two sensitivity analyses reported in Supplementary
Table S5 of the Afghanistan restoration monitoring study:

1. Leave-one-post-year-out sensitivity analysis
   The primary mixed-effects BACI model is refitted after omitting 2022,
   2023 and 2024 in turn.

2. Site-group-level BACI robustness analysis
   Each of the 17 site groups contributes one baseline-to-post BACI contrast
   for each outcome and season. The 17 contrasts are evaluated against zero
   using a one-sample t test, Wilcoxon signed-rank test and exact binomial
   sign test.

The site-group analysis reduces repeated polygon-year observations to one
contrast per site group. It does not fit or imply a residual temporal
autocorrelation model.

Inputs
------
outputs/baci/baci_master_panel_2017_2024.csv
outputs/analysis/baci_model_results.csv

Outputs
-------
outputs/robustness/leave_one_post_year_out.csv
outputs/robustness/site_group_baci_contrasts.csv
outputs/robustness/site_group_robustness_summary.csv
outputs/robustness/baci_robustness_qa.txt
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


# -----------------------------------------------------------------------------
# 1. Repository paths
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_ROOT = SCRIPT_DIR.parent

BACI_DIR = REPO_ROOT / "outputs" / "baci"
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"
ROBUSTNESS_DIR = REPO_ROOT / "outputs" / "robustness"
ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PANEL = BACI_DIR / "baci_master_panel_2017_2024.csv"
PRIMARY_RESULTS = ANALYSIS_DIR / "baci_model_results.csv"

OUT_LEAVE_ONE = ROBUSTNESS_DIR / "leave_one_post_year_out.csv"
OUT_SITE_CONTRASTS = ROBUSTNESS_DIR / "site_group_baci_contrasts.csv"
OUT_SITE_SUMMARY = ROBUSTNESS_DIR / "site_group_robustness_summary.csv"
OUT_QA = ROBUSTNESS_DIR / "baci_robustness_qa.txt"


# -----------------------------------------------------------------------------
# 2. Analysis settings
# -----------------------------------------------------------------------------

VALID_YEARS = list(range(2017, 2025))
VALID_SEASONS = ["peak", "late"]

BASELINE_YEARS = [2017, 2018]
POST_YEARS = [2022, 2023, 2024]

PHASE_ORDER = ["baseline", "implementation", "post_implementation"]

EXPECTED_ROWS = 928
EXPECTED_POLYGONS = 58
EXPECTED_TREATMENTS = 19
EXPECTED_REFERENCES = 39
EXPECTED_SITE_GROUPS = 17
EXPECTED_SITE_CODES = [f"SG{i:02d}" for i in range(1, 18)]

EXPECTED_FULL_SEASON_ROWS = 464
EXPECTED_OMITTED_SEASON_ROWS = 406

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

LEAVE_ONE_MODELS = [
    ("Full model", None),
    ("Exclude 2022", 2022),
    ("Exclude 2023", 2023),
    ("Exclude 2024", 2024),
]

STRICT_CHECKS = True


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


def safe_wilcoxon(values):
    try:
        result = stats.wilcoxon(
            values,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return np.nan, np.nan


def pass_fail(condition):
    return "PASS" if condition else "CHECK"


def round_numeric(df, digits=6):
    result = df.copy()
    columns = result.select_dtypes(include=[np.number]).columns
    result[columns] = result[columns].round(digits)
    return result


# -----------------------------------------------------------------------------
# 4. Read and validate the final analytical panel
# -----------------------------------------------------------------------------

require_file(INPUT_PANEL, "Final BACI panel")

df = pd.read_csv(INPUT_PANEL, encoding="utf-8-sig").copy()

required_columns = [
    "POLYGON_ID",
    "SITE_GROUP",
    "SITE_TYPE",
    "year",
    "season",
    "phase_3",
    "terrain_valid",
    "ndvi_mean",
    "bsi_mean",
]

require_columns(df, required_columns, "BACI panel")

df["POLYGON_ID"] = df["POLYGON_ID"].astype(str).str.strip()
df["SITE_GROUP"] = df["SITE_GROUP"].astype(str).str.strip().str.upper()
df["SITE_TYPE"] = df["SITE_TYPE"].apply(standardise_site_type)
df["season"] = df["season"].astype(str).str.strip().str.lower()
df["phase_3"] = df["phase_3"].astype(str).str.strip().str.lower()

for column in ["year", "terrain_valid", "ndvi_mean", "bsi_mean"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=["year", "terrain_valid"]).copy()
df["year"] = df["year"].astype(int)

df = df[
    (df["terrain_valid"] == 1)
    & df["year"].isin(VALID_YEARS)
    & df["season"].isin(VALID_SEASONS)
    & df["SITE_TYPE"].isin(["treatment", "control"])
].copy()

df["treat"] = (df["SITE_TYPE"] == "treatment").astype(int)

df["phase_3"] = pd.Categorical(
    df["phase_3"],
    categories=PHASE_ORDER,
    ordered=True,
)

n_rows = len(df)
n_polygons = int(df["POLYGON_ID"].nunique())
n_sites = int(df["SITE_GROUP"].nunique())
n_treatments = int(
    df.loc[df["SITE_TYPE"] == "treatment", "POLYGON_ID"].nunique()
)
n_references = int(
    df.loc[df["SITE_TYPE"] == "control", "POLYGON_ID"].nunique()
)

duplicate_rows = int(
    df.duplicated(["POLYGON_ID", "year", "season"]).sum()
)

missing_ndvi = int(df["ndvi_mean"].isna().sum())
missing_bsi = int(df["bsi_mean"].isna().sum())

site_codes = sorted(df["SITE_GROUP"].unique().tolist())

sample_checks = {
    "row_count": n_rows == EXPECTED_ROWS,
    "polygon_count": n_polygons == EXPECTED_POLYGONS,
    "treatment_count": n_treatments == EXPECTED_TREATMENTS,
    "reference_count": n_references == EXPECTED_REFERENCES,
    "site_group_count": n_sites == EXPECTED_SITE_GROUPS,
    "site_group_codes": site_codes == EXPECTED_SITE_CODES,
    "no_duplicate_polygon_year_season": duplicate_rows == 0,
    "no_missing_ndvi": missing_ndvi == 0,
    "no_missing_bsi": missing_bsi == 0,
}

if STRICT_CHECKS and not all(sample_checks.values()):
    failed = [key for key, value in sample_checks.items() if not value]
    raise RuntimeError(
        "BACI panel failed robustness-analysis QA checks: "
        f"{failed}"
    )


# -----------------------------------------------------------------------------
# 5. Leave-one-post-year-out mixed-effects analysis
# -----------------------------------------------------------------------------

def fit_leave_one_model(
    data,
    outcome,
    outcome_column,
    season,
    model_label,
    omitted_year,
):
    subset = data[
        (data["season"] == season)
        & data[outcome_column].notna()
    ].copy()

    if omitted_year is not None:
        subset = subset[
            subset["year"] != omitted_year
        ].copy()

    subset["phase_3"] = pd.Categorical(
        subset["phase_3"].astype(str),
        categories=PHASE_ORDER,
        ordered=True,
    )

    formula = (
        f"{outcome_column} ~ "
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

    post_term = (
        "treat:C(phase_3, Treatment('baseline'))"
        "[T.post_implementation]"
    )

    if post_term not in fit.params.index:
        raise RuntimeError(
            f"Treatment x Post term missing for "
            f"{outcome} | {season} | {model_label}"
        )

    ci = fit.conf_int().loc[post_term]

    warning_messages = [
        str(item.message) for item in caught_warnings
    ]

    expected_n = (
        EXPECTED_FULL_SEASON_ROWS
        if omitted_year is None
        else EXPECTED_OMITTED_SEASON_ROWS
    )

    return {
        "outcome": outcome,
        "season": season,
        "model": model_label,
        "omitted_post_year":
            "" if omitted_year is None else omitted_year,
        "coef": float(fit.params[post_term]),
        "se": float(fit.bse[post_term]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "p_value": float(fit.pvalues[post_term]),
        "n_obs": int(fit.nobs),
        "expected_n_obs": expected_n,
        "n_polygons": int(subset["POLYGON_ID"].nunique()),
        "n_site_groups": int(subset["SITE_GROUP"].nunique()),
        "converged": bool(fit.converged),
        "warning_count": len(warning_messages),
        "warnings": " | ".join(warning_messages),
        "post_years_used": ",".join(
            str(year)
            for year in sorted(
                subset.loc[
                    subset["phase_3"].astype(str)
                    == "post_implementation",
                    "year",
                ].unique()
            )
        ),
    }


leave_one_rows = []

for outcome, config in OUTCOMES.items():
    for season in VALID_SEASONS:
        for model_label, omitted_year in LEAVE_ONE_MODELS:
            leave_one_rows.append(
                fit_leave_one_model(
                    data=df,
                    outcome=outcome,
                    outcome_column=config["column"],
                    season=season,
                    model_label=model_label,
                    omitted_year=omitted_year,
                )
            )

leave_one = pd.DataFrame(leave_one_rows)

leave_one["sample_size_check"] = (
    leave_one["n_obs"] == leave_one["expected_n_obs"]
)

leave_one["polygon_count_check"] = (
    leave_one["n_polygons"] == EXPECTED_POLYGONS
)

leave_one["site_group_count_check"] = (
    leave_one["n_site_groups"] == EXPECTED_SITE_GROUPS
)

leave_one = round_numeric(leave_one)
leave_one.to_csv(
    OUT_LEAVE_ONE,
    index=False,
    encoding="utf-8-sig",
)


# -----------------------------------------------------------------------------
# 6. Verify full-model estimates against Script 05
# -----------------------------------------------------------------------------

require_file(PRIMARY_RESULTS, "Primary BACI model results")

primary = pd.read_csv(
    PRIMARY_RESULTS,
    encoding="utf-8-sig",
).copy()

require_columns(
    primary,
    [
        "model",
        "outcome",
        "season",
        "effect_phase",
        "coef",
    ],
    "primary BACI results",
)

primary_post = primary[
    (primary["model"] == "mixed_baci")
    & (primary["effect_phase"] == "post_implementation")
].copy()

primary_post["coef"] = pd.to_numeric(
    primary_post["coef"],
    errors="coerce",
)

full_rows = leave_one[
    leave_one["model"] == "Full model"
].copy()

full_comparison = full_rows.merge(
    primary_post[
        ["outcome", "season", "coef"]
    ].rename(columns={"coef": "primary_coef"}),
    on=["outcome", "season"],
    how="left",
    validate="one_to_one",
)

full_comparison["matches_primary_model"] = np.isclose(
    full_comparison["coef"],
    full_comparison["primary_coef"],
    atol=1e-6,
    rtol=0,
)


# -----------------------------------------------------------------------------
# 7. Site-group-level BACI contrasts
# -----------------------------------------------------------------------------

role_check = (
    df[["SITE_GROUP", "SITE_TYPE"]]
    .drop_duplicates()
    .groupby("SITE_GROUP")["SITE_TYPE"]
    .apply(set)
)

invalid_role_groups = [
    site_group
    for site_group, roles in role_check.items()
    if not {"treatment", "control"}.issubset(roles)
]

if invalid_role_groups:
    raise RuntimeError(
        "Site groups lacking both treatment and reference polygons: "
        f"{invalid_role_groups}"
    )

contrast_frames = []

for outcome, config in OUTCOMES.items():
    column = config["column"]

    for season in VALID_SEASONS:
        baseline = df[
            df["year"].isin(BASELINE_YEARS)
            & (df["season"] == season)
            & df[column].notna()
        ].copy()

        post = df[
            df["year"].isin(POST_YEARS)
            & (df["season"] == season)
            & df[column].notna()
        ].copy()

        baseline_means = (
            baseline
            .groupby(
                ["SITE_GROUP", "SITE_TYPE"],
                as_index=False,
            )
            .agg(
                baseline_mean=(column, "mean"),
                baseline_n_obs=(column, "count"),
                baseline_n_polygons=("POLYGON_ID", "nunique"),
            )
        )

        post_means = (
            post
            .groupby(
                ["SITE_GROUP", "SITE_TYPE"],
                as_index=False,
            )
            .agg(
                post_mean=(column, "mean"),
                post_n_obs=(column, "count"),
                post_n_polygons=("POLYGON_ID", "nunique"),
            )
        )

        period_means = baseline_means.merge(
            post_means,
            on=["SITE_GROUP", "SITE_TYPE"],
            how="inner",
            validate="one_to_one",
        )

        period_means["change"] = (
            period_means["post_mean"]
            - period_means["baseline_mean"]
        )

        treatment = (
            period_means[
                period_means["SITE_TYPE"] == "treatment"
            ]
            .drop(columns=["SITE_TYPE"])
            .rename(columns={
                "baseline_mean": "treatment_baseline_mean",
                "post_mean": "treatment_post_mean",
                "change": "treatment_change",
                "baseline_n_obs": "treatment_baseline_n_obs",
                "post_n_obs": "treatment_post_n_obs",
                "baseline_n_polygons":
                    "treatment_baseline_n_polygons",
                "post_n_polygons":
                    "treatment_post_n_polygons",
            })
        )

        reference = (
            period_means[
                period_means["SITE_TYPE"] == "control"
            ]
            .drop(columns=["SITE_TYPE"])
            .rename(columns={
                "baseline_mean": "reference_baseline_mean",
                "post_mean": "reference_post_mean",
                "change": "reference_change",
                "baseline_n_obs": "reference_baseline_n_obs",
                "post_n_obs": "reference_post_n_obs",
                "baseline_n_polygons":
                    "reference_baseline_n_polygons",
                "post_n_polygons":
                    "reference_post_n_polygons",
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

        contrasts["outcome"] = outcome
        contrasts["season"] = season
        contrasts["expected_direction"] = (
            config["expected_direction"]
        )

        if config["expected_direction"] == "positive":
            contrasts["in_expected_direction"] = (
                contrasts["baci_contrast"] > 0
            ).astype(int)
        else:
            contrasts["in_expected_direction"] = (
                contrasts["baci_contrast"] < 0
            ).astype(int)

        contrast_frames.append(contrasts)

site_contrasts = pd.concat(
    contrast_frames,
    ignore_index=True,
)

site_contrasts = site_contrasts.sort_values(
    ["outcome", "season", "SITE_GROUP"]
).reset_index(drop=True)

contrast_counts = (
    site_contrasts
    .groupby(["outcome", "season"])["SITE_GROUP"]
    .nunique()
)

if not (contrast_counts == EXPECTED_SITE_GROUPS).all():
    raise RuntimeError(
        "Not every outcome-season combination contains 17 "
        f"site-group contrasts:\n{contrast_counts}"
    )

site_contrasts = round_numeric(site_contrasts)
site_contrasts.to_csv(
    OUT_SITE_CONTRASTS,
    index=False,
    encoding="utf-8-sig",
)


# -----------------------------------------------------------------------------
# 8. Inference across the 17 site-group contrasts
# -----------------------------------------------------------------------------

summary_rows = []

for outcome, config in OUTCOMES.items():
    for season in VALID_SEASONS:
        subset = site_contrasts[
            (site_contrasts["outcome"] == outcome)
            & (site_contrasts["season"] == season)
        ].copy()

        values = subset["baci_contrast"].dropna().to_numpy(dtype=float)

        if len(values) != EXPECTED_SITE_GROUPS:
            raise RuntimeError(
                f"{outcome} | {season} contains "
                f"{len(values)} contrasts, expected 17."
            )

        n = len(values)
        mean_value = float(np.mean(values))
        median_value = float(np.median(values))
        sd_value = float(np.std(values, ddof=1))
        se_value = sd_value / np.sqrt(n)

        t_critical = stats.t.ppf(0.975, df=n - 1)
        ci_low = mean_value - t_critical * se_value
        ci_high = mean_value + t_critical * se_value

        t_result = stats.ttest_1samp(
            values,
            popmean=0,
        )

        wilcoxon_stat, wilcoxon_p = safe_wilcoxon(values)

        if config["expected_direction"] == "positive":
            n_expected = int(np.sum(values > 0))
        else:
            n_expected = int(np.sum(values < 0))

        sign_result = stats.binomtest(
            k=n_expected,
            n=n,
            p=0.5,
            alternative="two-sided",
        )

        summary_rows.append({
            "outcome": outcome,
            "season": season,
            "n_site_groups": n,
            "mean_baci_contrast": mean_value,
            "se": se_value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "median_baci_contrast": median_value,
            "sd_baci_contrast": sd_value,
            "expected_direction": config["expected_direction"],
            "n_expected_direction": n_expected,
            "pct_expected_direction": n_expected / n * 100.0,
            "t_statistic": float(t_result.statistic),
            "t_p_value": float(t_result.pvalue),
            "wilcoxon_statistic": wilcoxon_stat,
            "wilcoxon_p_value": wilcoxon_p,
            "exact_sign_test_p": float(sign_result.pvalue),
        })

site_summary = pd.DataFrame(summary_rows)
site_summary = site_summary.sort_values(
    ["outcome", "season"]
).reset_index(drop=True)

site_summary = round_numeric(site_summary)
site_summary.to_csv(
    OUT_SITE_SUMMARY,
    index=False,
    encoding="utf-8-sig",
)


# -----------------------------------------------------------------------------
# 9. Robustness QA
# -----------------------------------------------------------------------------

leave_one_checks = {
    "expected_result_rows": len(leave_one) == 16,
    "all_models_converged": bool(leave_one["converged"].all()),
    "all_models_keep_58_polygons":
        bool((leave_one["n_polygons"] == EXPECTED_POLYGONS).all()),
    "all_models_keep_17_site_groups":
        bool((leave_one["n_site_groups"] == EXPECTED_SITE_GROUPS).all()),
    "expected_model_sample_sizes":
        bool(leave_one["sample_size_check"].all()),
    "full_models_match_script05":
        bool(full_comparison["matches_primary_model"].all()),
}

site_group_checks = {
    "expected_contrast_rows":
        len(site_contrasts)
        == len(OUTCOMES) * len(VALID_SEASONS) * EXPECTED_SITE_GROUPS,
    "expected_summary_rows": len(site_summary) == 4,
    "17_groups_per_outcome_season":
        bool((contrast_counts == EXPECTED_SITE_GROUPS).all()),
    "all_sg_codes_present":
        sorted(site_contrasts["SITE_GROUP"].unique().tolist())
        == EXPECTED_SITE_CODES,
}

all_checks = {
    **sample_checks,
    **leave_one_checks,
    **site_group_checks,
}

with open(OUT_QA, "w", encoding="utf-8") as qa:
    qa.write("BACI robustness analysis QA\n")
    qa.write("=" * 72 + "\n\n")

    qa.write("Purpose\n")
    qa.write("-" * 72 + "\n")
    qa.write(
        "Leave-one-post-year-out sensitivity and equal-weight "
        "site-group-level BACI robustness analyses.\n"
    )
    qa.write(
        "The site-group analysis does not model residual temporal "
        "autocorrelation; it collapses repeated observations to one "
        "baseline-to-post contrast per site group.\n\n"
    )

    qa.write("Analytical sample\n")
    qa.write("-" * 72 + "\n")
    qa.write(f"Rows: {n_rows}\n")
    qa.write(f"Polygons: {n_polygons}\n")
    qa.write(f"Treatments: {n_treatments}\n")
    qa.write(f"References: {n_references}\n")
    qa.write(f"Site groups: {n_sites}\n\n")

    qa.write("Leave-one-post-year-out specification\n")
    qa.write("-" * 72 + "\n")
    qa.write(
        "outcome ~ treat * C(phase_3, Treatment('baseline'))\n"
    )
    qa.write("Random intercept: SITE_GROUP\n")
    qa.write("Polygon variance component: 0 + C(POLYGON_ID)\n")
    qa.write("Estimator: ML, lbfgs\n")
    qa.write(
        "Full seasonal model n = 464; each omitted-year model n = 406.\n\n"
    )

    qa.write("Full-model comparison with Script 05\n")
    qa.write("-" * 72 + "\n")
    for row in full_comparison.itertuples(index=False):
        qa.write(
            f"{row.outcome} | {row.season} | "
            f"Script06={row.coef:+.6f} | "
            f"Script05={row.primary_coef:+.6f} | "
            f"{pass_fail(row.matches_primary_model)}\n"
        )

    qa.write("\nSite-group robustness specification\n")
    qa.write("-" * 72 + "\n")
    qa.write(
        "BACI = (Treatment_post - Treatment_baseline) "
        "- (Reference_post - Reference_baseline)\n"
    )
    qa.write(f"Baseline years: {BASELINE_YEARS}\n")
    qa.write(f"Post years: {POST_YEARS}\n")
    qa.write(
        "Inference: two-sided one-sample t test, Wilcoxon "
        "signed-rank test and exact binomial sign test.\n\n"
    )

    qa.write("Checks\n")
    qa.write("-" * 72 + "\n")
    for key, value in all_checks.items():
        qa.write(f"{key}: {pass_fail(value)}\n")

    qa.write("\nOutputs\n")
    qa.write("-" * 72 + "\n")
    qa.write(f"{OUT_LEAVE_ONE.name}\n")
    qa.write(f"{OUT_SITE_CONTRASTS.name}\n")
    qa.write(f"{OUT_SITE_SUMMARY.name}\n")


if STRICT_CHECKS and not all(all_checks.values()):
    failed = [
        key for key, value in all_checks.items()
        if not value
    ]
    raise RuntimeError(
        "Robustness analyses failed one or more QA checks: "
        f"{failed}. See {OUT_QA}"
    )


print("BACI robustness analyses complete")
print(f"Leave-one-post-year-out: {OUT_LEAVE_ONE}")
print(f"Site-group contrasts: {OUT_SITE_CONTRASTS}")
print(f"Site-group summary: {OUT_SITE_SUMMARY}")
print(f"QA: {OUT_QA}")
