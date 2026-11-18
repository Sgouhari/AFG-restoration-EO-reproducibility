# -*- coding: utf-8 -*-
"""
Calculate district-level SPI-12 from CHIRPS and summarise hydro-climatic context.

This script is the final stage of the reproducibility workflow for the
Afghanistan restoration monitoring study. It reads the combined monthly
CHIRPS table produced by 09_chirps_monthly_gee.js, derives SPI-12 using the
same gamma-based procedure used in the original analysis, and produces the
hydro-climatic summaries reported in the manuscript.

SPI-12 procedure
----------------
1. Build a continuous monthly precipitation series for each district.
2. Calculate 12-month rolling precipitation accumulation.
3. Fit a gamma distribution separately for each calendar month using the
   full 1981-2025 record.
4. Estimate the probability of zero precipitation.
5. Calculate H(x) = q + (1-q)G(x).
6. Transform H(x) to the standard normal distribution.

The first 11 months for each district are retained with missing SPI-12 because
a complete 12-month accumulation is not yet available.

Hydro-climatic periods
----------------------
Long-term reference period : 1981-2018
BACI baseline              : 2017-2018
Implementation             : 2019-2021
Post-implementation        : 2022-2024

The 1981-2018 period is contextual only. The BACI baseline remains 2017-2018.

Requirements
------------
numpy
pandas
scipy

Input
-----
outputs/chirps/AFG_CHIRPS_monthly_7districts_1981_2025.csv

Outputs
-------
outputs/hydroclimate/chirps_spi12_monthly_1981_2025.csv
outputs/hydroclimate/spi12_gamma_fit_diagnostics.csv
outputs/hydroclimate/spi12_district_period_summary.csv
outputs/hydroclimate/spi12_studywide_period_summary.csv
outputs/hydroclimate/spi12_context_qa.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm


# -----------------------------------------------------------------------------
# 1. Repository paths
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_ROOT = SCRIPT_DIR.parent

OUTPUT_DIR = REPO_ROOT / "outputs"
CHIRPS_DIR = OUTPUT_DIR / "chirps"
HYDRO_DIR = OUTPUT_DIR / "hydroclimate"
HYDRO_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = CHIRPS_DIR / "AFG_CHIRPS_monthly_7districts_1981_2025.csv"

OUT_SPI12 = HYDRO_DIR / "chirps_spi12_monthly_1981_2025.csv"
OUT_FIT = HYDRO_DIR / "spi12_gamma_fit_diagnostics.csv"
OUT_DISTRICT = HYDRO_DIR / "spi12_district_period_summary.csv"
OUT_STUDYWIDE = HYDRO_DIR / "spi12_studywide_period_summary.csv"
OUT_QA = HYDRO_DIR / "spi12_context_qa.txt"


# -----------------------------------------------------------------------------
# 2. Analysis settings
# -----------------------------------------------------------------------------

FIT_START_YEAR = 1981
FIT_END_YEAR = 2025
SPI_WINDOW = 12

EXPECTED_DISTRICTS = 7
EXPECTED_MONTHS_PER_DISTRICT = (
    (FIT_END_YEAR - FIT_START_YEAR + 1) * 12
)
EXPECTED_INPUT_ROWS = (
    EXPECTED_DISTRICTS * EXPECTED_MONTHS_PER_DISTRICT
)

# The first 11 observations in each district cannot have a 12-month
# accumulation.
EXPECTED_NONMISSING_SPI12 = (
    EXPECTED_INPUT_ROWS
    - EXPECTED_DISTRICTS * (SPI_WINDOW - 1)
)

PROB_EPS = 1e-8
MIN_POSITIVE_FIT = 8

PERIODS = {
    "Long-term reference period": (1981, 2018),
    "BACI baseline": (2017, 2018),
    "Implementation": (2019, 2021),
    "Post-implementation": (2022, 2024),
}

PERIOD_ORDER = [
    "Long-term reference period",
    "BACI baseline",
    "Implementation",
    "Post-implementation",
]

DROUGHT_THRESHOLD = -1.0

EXPECTED_BACI_PERIOD_OBSERVATIONS = {
    "BACI baseline": 7 * 24,
    "Implementation": 7 * 36,
    "Post-implementation": 7 * 36,
}

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


def ensure_monthly_continuity(df, district_key):
    """Return a continuous monthly series and fail if rainfall is missing."""
    data = df.copy()

    data["date"] = pd.to_datetime(
        dict(
            year=data["year"],
            month=data["month"],
            day=1,
        )
    )

    data = data.sort_values("date").reset_index(drop=True)

    full_dates = pd.date_range(
        f"{FIT_START_YEAR}-01-01",
        f"{FIT_END_YEAR}-12-01",
        freq="MS",
    )

    full = pd.DataFrame({"date": full_dates})

    metadata_columns = [
        "district_key",
        "province",
        "district",
    ]

    metadata = (
        data[metadata_columns]
        .drop_duplicates()
        .copy()
    )

    if len(metadata) != 1:
        raise ValueError(
            f"{district_key}: district metadata are inconsistent."
        )

    result = full.merge(
        data[["date", "rain_mm"]],
        on="date",
        how="left",
        validate="one_to_one",
    )

    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month

    for column in metadata_columns:
        result[column] = metadata.iloc[0][column]

    missing_rain = int(result["rain_mm"].isna().sum())

    if missing_rain > 0:
        raise ValueError(
            f"{district_key}: {missing_rain} monthly rainfall "
            "values are missing after continuity checking."
        )

    return result


def fit_gamma_and_spi(values):
    """
    Fit the gamma distribution and calculate SPI for one calendar-month group.

    Zero precipitation is handled using:
        H(x) = q + (1-q)G(x)

    where q is the empirical zero probability.
    """
    values = values.astype(float).copy()

    result = pd.Series(
        np.nan,
        index=values.index,
        dtype=float,
    )

    valid = values.dropna()

    if valid.empty:
        return result, {
            "q_zero": np.nan,
            "gamma_shape": np.nan,
            "gamma_scale": np.nan,
            "n_total": 0,
            "n_zero": 0,
            "n_positive_fit": 0,
        }

    n_total = len(valid)
    n_zero = int((valid <= 0).sum())
    q_zero = n_zero / n_total

    positive = valid[valid > 0]

    if len(positive) < MIN_POSITIVE_FIT:
        return result, {
            "q_zero": q_zero,
            "gamma_shape": np.nan,
            "gamma_scale": np.nan,
            "n_total": n_total,
            "n_zero": n_zero,
            "n_positive_fit": len(positive),
        }

    shape, _, scale = gamma.fit(
        positive.values,
        floc=0,
    )

    for index, value in valid.items():
        if value < 0:
            result.loc[index] = np.nan
            continue

        if value == 0:
            h = q_zero
        else:
            gamma_probability = gamma.cdf(
                value,
                a=shape,
                loc=0,
                scale=scale,
            )

            h = (
                q_zero
                + (1.0 - q_zero) * gamma_probability
            )

        h = min(
            max(h, PROB_EPS),
            1.0 - PROB_EPS,
        )

        result.loc[index] = norm.ppf(h)

    return result, {
        "q_zero": q_zero,
        "gamma_shape": shape,
        "gamma_scale": scale,
        "n_total": n_total,
        "n_zero": n_zero,
        "n_positive_fit": len(positive),
    }


def calculate_spi12_for_district(df):
    """
    Calculate SPI-12 for one district.

    Gamma distributions are fitted independently for January, February, ...
    December using all available 12-month accumulations for that calendar
    month over 1981-2025.
    """
    data = df.copy()

    data["accum_12"] = (
        data["rain_mm"]
        .rolling(
            window=SPI_WINDOW,
            min_periods=SPI_WINDOW,
        )
        .sum()
    )

    data["spi12"] = np.nan
    parameter_rows = []

    for month in range(1, 13):
        mask = data["month"] == month

        spi_values, parameters = fit_gamma_and_spi(
            data.loc[mask, "accum_12"]
        )

        data.loc[mask, "spi12"] = spi_values.values

        parameter_rows.append({
            "district_key":
                data["district_key"].iloc[0],
            "province":
                data["province"].iloc[0],
            "district":
                data["district"].iloc[0],
            "month":
                month,
            "window_months":
                SPI_WINDOW,
            **parameters,
        })

    return data, pd.DataFrame(parameter_rows)


def subset_period(df, start_year, end_year):
    return df[
        (df["year"] >= start_year)
        & (df["year"] <= end_year)
    ].copy()


def summarise_spi12(values):
    values = values.dropna()

    n = len(values)

    if n == 0:
        return {
            "n_district_months": 0,
            "mean_spi12": np.nan,
            "sd_spi12": np.nan,
            "min_spi12": np.nan,
            "max_spi12": np.nan,
            "n_spi12_le_minus1": 0,
            "pct_spi12_le_minus1": np.nan,
        }

    drought = values <= DROUGHT_THRESHOLD

    return {
        "n_district_months": n,
        "mean_spi12": float(values.mean()),
        "sd_spi12":
            float(values.std(ddof=1))
            if n > 1 else 0.0,
        "min_spi12": float(values.min()),
        "max_spi12": float(values.max()),
        "n_spi12_le_minus1": int(drought.sum()),
        "pct_spi12_le_minus1":
            float(drought.mean() * 100.0),
    }


def pass_fail(condition):
    return "PASS" if condition else "CHECK"


def round_numeric(df, digits=6):
    result = df.copy()
    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].round(digits)
    return result


# -----------------------------------------------------------------------------
# 4. Read and validate monthly CHIRPS
# -----------------------------------------------------------------------------

require_file(INPUT_CSV, "Combined CHIRPS monthly table")

raw = pd.read_csv(
    INPUT_CSV,
    encoding="utf-8-sig",
).copy()

required_columns = [
    "district_key",
    "province",
    "district",
    "year",
    "month",
    "rain_mm",
]

require_columns(raw, required_columns, "CHIRPS monthly input")

for column in ["district_key", "province", "district"]:
    raw[column] = (
        raw[column]
        .astype(str)
        .str.strip()
    )

raw["year"] = pd.to_numeric(
    raw["year"],
    errors="coerce",
)

raw["month"] = pd.to_numeric(
    raw["month"],
    errors="coerce",
)

raw["rain_mm"] = pd.to_numeric(
    raw["rain_mm"],
    errors="coerce",
)

if raw[["year", "month", "rain_mm"]].isna().any().any():
    raise ValueError(
        "CHIRPS input contains missing or non-numeric "
        "year, month or rainfall values."
    )

raw["year"] = raw["year"].astype(int)
raw["month"] = raw["month"].astype(int)

raw = raw[
    raw["year"].between(
        FIT_START_YEAR,
        FIT_END_YEAR,
    )
    & raw["month"].between(1, 12)
].copy()

duplicate_input_rows = int(
    raw.duplicated(
        ["district_key", "year", "month"]
    ).sum()
)

if duplicate_input_rows:
    raise ValueError(
        "Duplicate district-year-month CHIRPS records found: "
        f"{duplicate_input_rows}"
    )

n_input_rows = len(raw)
n_input_districts = int(raw["district_key"].nunique())

input_checks = {
    "input_rows":
        n_input_rows == EXPECTED_INPUT_ROWS,
    "seven_districts":
        n_input_districts == EXPECTED_DISTRICTS,
    "no_duplicate_district_months":
        duplicate_input_rows == 0,
    "no_negative_rainfall":
        bool((raw["rain_mm"] >= 0).all()),
}

if STRICT_CHECKS and not all(input_checks.values()):
    failed = [
        key for key, value in input_checks.items()
        if not value
    ]
    raise RuntimeError(
        "CHIRPS input failed SPI-12 QA checks: "
        f"{failed}"
    )


# -----------------------------------------------------------------------------
# 5. Calculate SPI-12 for each district
# -----------------------------------------------------------------------------

spi_frames = []
fit_frames = []

for district_key, district_data in raw.groupby(
    "district_key",
    sort=True,
):
    continuous = ensure_monthly_continuity(
        district_data,
        district_key,
    )

    if len(continuous) != EXPECTED_MONTHS_PER_DISTRICT:
        raise RuntimeError(
            f"{district_key}: expected "
            f"{EXPECTED_MONTHS_PER_DISTRICT} months, "
            f"found {len(continuous)}."
        )

    spi_data, fit_parameters = calculate_spi12_for_district(
        continuous
    )

    spi_frames.append(spi_data)
    fit_frames.append(fit_parameters)

spi12 = pd.concat(
    spi_frames,
    ignore_index=True,
)

fit_diagnostics = pd.concat(
    fit_frames,
    ignore_index=True,
)

spi12 = spi12[
    [
        "district_key",
        "province",
        "district",
        "year",
        "month",
        "rain_mm",
        "accum_12",
        "spi12",
    ]
].copy()

spi12 = spi12.sort_values(
    ["province", "district", "year", "month"]
).reset_index(drop=True)

fit_diagnostics = fit_diagnostics.sort_values(
    ["province", "district", "month"]
).reset_index(drop=True)

n_spi_rows = len(spi12)
n_nonmissing_spi = int(spi12["spi12"].notna().sum())

spi_checks = {
    "spi_output_rows":
        n_spi_rows == EXPECTED_INPUT_ROWS,
    "nonmissing_spi12_count":
        n_nonmissing_spi == EXPECTED_NONMISSING_SPI12,
    "twelve_fit_rows_per_district":
        len(fit_diagnostics)
        == EXPECTED_DISTRICTS * 12,
    "all_gamma_fits_available":
        bool(
            fit_diagnostics["gamma_shape"].notna().all()
            and fit_diagnostics["gamma_scale"].notna().all()
        ),
}

if STRICT_CHECKS and not all(spi_checks.values()):
    failed = [
        key for key, value in spi_checks.items()
        if not value
    ]
    raise RuntimeError(
        "SPI-12 calculation failed QA checks: "
        f"{failed}"
    )

round_numeric(spi12).to_csv(
    OUT_SPI12,
    index=False,
    encoding="utf-8-sig",
)

round_numeric(fit_diagnostics).to_csv(
    OUT_FIT,
    index=False,
    encoding="utf-8-sig",
)


# -----------------------------------------------------------------------------
# 6. District-level hydro-climatic period summaries
# -----------------------------------------------------------------------------

district_rows = []

for district_key, district_data in spi12.groupby(
    "district_key",
    sort=True,
):
    province = district_data["province"].iloc[0]
    district = district_data["district"].iloc[0]

    for period, (start_year, end_year) in PERIODS.items():
        subset = subset_period(
            district_data,
            start_year,
            end_year,
        )

        summary = summarise_spi12(
            subset["spi12"]
        )

        district_rows.append({
            "district_key": district_key,
            "province": province,
            "district": district,
            "period": period,
            "start_year": start_year,
            "end_year": end_year,
            **summary,
        })

district_summary = pd.DataFrame(district_rows)

district_summary["period"] = pd.Categorical(
    district_summary["period"],
    categories=PERIOD_ORDER,
    ordered=True,
)

district_summary = district_summary.sort_values(
    ["province", "district", "period"]
).reset_index(drop=True)

round_numeric(district_summary).to_csv(
    OUT_DISTRICT,
    index=False,
    encoding="utf-8-sig",
)


# -----------------------------------------------------------------------------
# 7. Study-wide summaries across the seven districts
# -----------------------------------------------------------------------------
#
# Each district contributes one SPI-12 observation per month. The study-wide
# summaries therefore pool district-month observations across the seven project
# districts, matching the original manuscript analysis.
# -----------------------------------------------------------------------------

studywide_rows = []

for period, (start_year, end_year) in PERIODS.items():
    subset = subset_period(
        spi12,
        start_year,
        end_year,
    )

    summary = summarise_spi12(
        subset["spi12"]
    )

    studywide_rows.append({
        "period": period,
        "start_year": start_year,
        "end_year": end_year,
        "number_of_districts":
            int(subset["district_key"].nunique()),
        "expected_months_per_district":
            (end_year - start_year + 1) * 12,
        **summary,
    })

studywide = pd.DataFrame(studywide_rows)

studywide["period"] = pd.Categorical(
    studywide["period"],
    categories=PERIOD_ORDER,
    ordered=True,
)

studywide = studywide.sort_values(
    "period"
).reset_index(drop=True)

round_numeric(studywide).to_csv(
    OUT_STUDYWIDE,
    index=False,
    encoding="utf-8-sig",
)


# -----------------------------------------------------------------------------
# 8. Final QA
# -----------------------------------------------------------------------------

period_count_checks = {}

for period, expected in EXPECTED_BACI_PERIOD_OBSERVATIONS.items():
    actual_series = studywide.loc[
        studywide["period"].astype(str) == period,
        "n_district_months",
    ]

    period_count_checks[
        f"{period}_observation_count"
    ] = (
        len(actual_series) == 1
        and int(actual_series.iloc[0]) == expected
    )

context_checks = {
    "four_period_summary_rows":
        len(studywide) == 4,
    "seven_districts_in_each_period":
        bool(
            (
                studywide["number_of_districts"]
                == EXPECTED_DISTRICTS
            ).all()
        ),
    **period_count_checks,
}

all_checks = {
    **input_checks,
    **spi_checks,
    **context_checks,
}

with open(
    OUT_QA,
    "w",
    encoding="utf-8",
) as qa:
    qa.write("SPI-12 hydro-climatic context QA\n")
    qa.write("=" * 72 + "\n\n")

    qa.write("SPI-12 calculation\n")
    qa.write("-" * 72 + "\n")
    qa.write(
        "Source: district-level monthly CHIRPS precipitation "
        "from 1981-2025.\n"
    )
    qa.write(
        "Accumulation: rolling 12-month precipitation total.\n"
    )
    qa.write(
        "Gamma fitting: separately for each calendar month "
        "using the full 1981-2025 fitting record.\n"
    )
    qa.write(
        "Zero correction: H(x) = q + (1-q)G(x).\n"
    )
    qa.write(
        "Transformation: standard normal inverse CDF.\n"
    )
    qa.write(
        f"Monthly input rows: {n_input_rows}\n"
    )
    qa.write(
        f"Non-missing SPI-12 observations: {n_nonmissing_spi}\n\n"
    )

    qa.write("Hydro-climatic periods\n")
    qa.write("-" * 72 + "\n")
    for period, (start_year, end_year) in PERIODS.items():
        qa.write(
            f"{period}: {start_year}-{end_year}\n"
        )

    qa.write(
        "\nThe long-term reference period is contextual only. "
        "The BACI baseline is 2017-2018.\n\n"
    )

    qa.write("Study-wide period summaries\n")
    qa.write("-" * 72 + "\n")

    for row in studywide.itertuples(index=False):
        qa.write(
            f"{row.period}\n"
            f"  Mean SPI12: {row.mean_spi12:.6f}\n"
            f"  SPI12 <= -1.0: "
            f"{row.pct_spi12_le_minus1:.3f}%\n"
            f"  District-month observations: "
            f"{int(row.n_district_months)}\n"
            f"  Districts: "
            f"{int(row.number_of_districts)}\n"
        )

    qa.write("\nChecks\n")
    qa.write("-" * 72 + "\n")
    for key, value in all_checks.items():
        qa.write(
            f"{key}: {pass_fail(value)}\n"
        )

    qa.write("\nOutputs\n")
    qa.write("-" * 72 + "\n")
    for path in [
        OUT_SPI12,
        OUT_FIT,
        OUT_DISTRICT,
        OUT_STUDYWIDE,
    ]:
        qa.write(f"{path.name}\n")


if STRICT_CHECKS and not all(all_checks.values()):
    failed = [
        key for key, value in all_checks.items()
        if not value
    ]

    raise RuntimeError(
        "SPI-12 hydro-climatic workflow failed QA checks: "
        f"{failed}. See {OUT_QA}"
    )


print("SPI-12 hydro-climatic context analysis complete")
print(f"Monthly SPI-12: {OUT_SPI12}")
print(f"Gamma-fit diagnostics: {OUT_FIT}")
print(f"District summaries: {OUT_DISTRICT}")
print(f"Study-wide summaries: {OUT_STUDYWIDE}")
print(f"QA: {OUT_QA}")
