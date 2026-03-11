"""
data_collection.py
==================
Downloads four Statistics Canada tables and builds a clean
province × year panel:  research_data_2015_2024.csv

Source tables:
──────────────────────────────────────────────────────────────────
  14-10-0287-01  LFS monthly, seasonally adjusted
                 pid=1410028701
                 https://www150.statcan.gc.ca/n1/tbl/csv/14100287-eng.zip
                 → Unemployment rate, Participation rate, FT/PT employment
                 → Age group filter includes "20 to 24 years", "25 to 29 years"
                 → Monthly → collapsed to annual mean

  14-10-0064-01  Employee wages by industry, annual
                 pid=1410006401
                 https://www150.statcan.gc.ca/n1/tbl/csv/14100064-eng.zip
                 → Average hourly wage rate
                 → Age group filter includes "20 to 24 years", "25 to 29 years"

  14-10-0020-01  Unemployment/participation by educational attainment, annual
                 pid=1410002001
                 https://www150.statcan.gc.ca/n1/tbl/csv/14100020-eng.zip
                 → Part-time share (Full-time / Part-time employment columns)
                 → Note: age breakout is coarser here; used only for PT share

  18-10-0004-01  Consumer Price Index, monthly, not seasonally adjusted
                 pid=1810000401
                 https://www150.statcan.gc.ca/n1/tbl/csv/18100004-eng.zip
                 → All-items CPI, provincial
                 → Monthly → annual mean → 2019-base deflation

Age-group approach
──────────────────
Tables 14100287 and 14100064 publish "20 to 24 years" AND "25 to 29 years".
We take their simple average as the 20–29 cohort — no approximation needed.

Table 14100020 only has coarser groups (15–24, 25–44 etc.), so for Part-Time
Share we take the weighted blend: 40 % "15 to 24 years" + 60 % "25 to 44 years".

Real wage (2019 CAD)
────────────────────
  Real_Wage = Nominal_Wage × ( CPI_2019_province / CPI_year_province )

Stress Index
────────────
  Stress = mean( +z(Unemployment_Rate),
                 −z(Participation_Rate),
                 −z(Real_Wage) )
  Standardised over the full 100-obs panel → mean = 0, SD ≈ 1.

Output columns
──────────────
  Province | Province_Abbr | Year
  Unemployment_Rate | Participation_Rate
  FT_Employment | PT_Employment | Part_Time_Share
  Nominal_Wage | CPI_Index | Real_Wage
  Post2020 | Stress_Index
"""

import io
import os
import sys
import zipfile
import urllib.request
import urllib.error

import numpy as np
import pandas as pd

# ── Download URLs (confirmed from StatCan table-viewer pages) ─────────────────
STATCAN_ZIPS = {
    "LFS":   "https://www150.statcan.gc.ca/n1/tbl/csv/14100287-eng.zip",
    "WAGES": "https://www150.statcan.gc.ca/n1/tbl/csv/14100064-eng.zip",
    "PT":    "https://www150.statcan.gc.ca/n1/tbl/csv/14100020-eng.zip",
    "CPI":   "https://www150.statcan.gc.ca/n1/tbl/csv/18100004-eng.zip",
}

# CSV filename inside each zip matches the table ID
ZIP_CSV = {
    "LFS":   "14100287.csv",
    "WAGES": "14100064.csv",
    "PT":    "14100020.csv",
    "CPI":   "18100004.csv",
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "statcan_cache")

# ── Geography / time constants ─────────────────────────────────────────────────
PROVINCES = [
    "Alberta", "British Columbia", "Manitoba", "New Brunswick",
    "Newfoundland and Labrador", "Nova Scotia", "Ontario",
    "Prince Edward Island", "Quebec", "Saskatchewan",
]
PROV_ABBR = {
    "Alberta": "AB", "British Columbia": "BC", "Manitoba": "MB",
    "New Brunswick": "NB", "Newfoundland and Labrador": "NL",
    "Nova Scotia": "NS", "Ontario": "ON", "Prince Edward Island": "PE",
    "Quebec": "QC", "Saskatchewan": "SK",
}
STUDY_YEARS = list(range(2015, 2025))

# Exact age-group strings for LFS (14-10-0287)
YOUTH_AGES_LFS = ["20 to 24 years", "25 to 29 years"]

# Available age-group string for Wages (14-10-0064)
YOUTH_AGES_WAGE = ["15 to 24 years"]

# Weighted blend for 14100020 (coarser breakout)
PT_AGE_WEIGHTS = {"15 to 24 years": 0.40, "25 to 44 years": 0.60}


# DOWNLOAD + CACHE

def _fetch(key: str, force: bool = False) -> pd.DataFrame:
    """
    Download zip for *key* (if not cached), extract the CSV, return DataFrame.
    Files are cached in ./statcan_cache/ so subsequent runs are instant.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{key}.zip")

    if os.path.exists(cache_path) and not force:
        mb = os.path.getsize(cache_path) / 1_048_576
        print(f"    cache: {os.path.basename(cache_path)}  ({mb:.0f} MB)")
        raw_bytes = open(cache_path, "rb").read()
    else:
        url = STATCAN_ZIPS[key]
        print(f"    ↓ {url}")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (StatCan Research Downloader)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                chunks, done = [], 0
                while chunk := resp.read(1 << 16):
                    chunks.append(chunk)
                    done += len(chunk)
                    if total:
                        sys.stdout.write(f"\r      {done/1e6:6.1f} / {total/1e6:.1f} MB")
                        sys.stdout.flush()
                print()
            raw_bytes = b"".join(chunks)
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"\n✗ Download failed for {url}\n  {exc}\n\n"
                f"  Manual fix: download the zip yourself and save it to:\n"
                f"    {cache_path}"
            ) from exc

        with open(cache_path, "wb") as f:
            f.write(raw_bytes)
        print(f"    saved: {cache_path}  ({len(raw_bytes)/1e6:.0f} MB)")

    # Extract CSV from the zip
    csv_name = ZIP_CSV[key]
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        match = next((n for n in zf.namelist()
                      if n.endswith(csv_name) or n == csv_name), None)
        if match is None:
            raise FileNotFoundError(
                f"Expected '{csv_name}' inside {key} zip. "
                f"Found: {zf.namelist()[:5]}"
            )
        with zf.open(match) as f:
            df = pd.read_csv(f, low_memory=False)

    df.columns = [c.strip() for c in df.columns]
    print(f"      {len(df):,} rows × {len(df.columns)} columns")
    return df



def _year(series: pd.Series) -> pd.Series:
    """Parse REF_DATE → integer year. Handles 'YYYY-MM' and plain integers."""
    sample = str(series.dropna().iloc[0])
    if "-" in sample:
        return pd.to_datetime(series, errors="coerce").dt.year
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _find_col(df: pd.DataFrame, *substrings: str) -> str:
    """Return first column whose lower-case name contains any of the substrings."""
    for sub in substrings:
        for c in df.columns:
            if sub.lower() in c.lower():
                return c
    raise KeyError(f"No column matching {substrings} in: {list(df.columns)}")


# ________________________________________________________________________________
# STEP 1 — LFS  (14-10-0287-01)
# Unemployment rate, Participation rate, FT/PT employment
# Monthly → annual mean, 20–24 and 25–29 averaged → 20–29 cohort


def process_lfs() -> pd.DataFrame:
    print("\n[1/4]  LFS  14-10-0287-01")
    raw = _fetch("LFS")

    age_col  = _find_col(raw, "age group", "age")
    char_col = _find_col(raw, "labour force char", "characteristic", "statistics")
    sex_col  = next((c for c in raw.columns
                     if c.lower() in ("sex", "gender")), None)

    LF_CHARS = [
        "Unemployment rate",
        "Participation rate",
        "Full-time employment",
        "Part-time employment",
    ]

    mask = (
        raw["GEO"].isin(PROVINCES) &
        raw[age_col].isin(YOUTH_AGES_LFS) &
        raw[char_col].isin(LF_CHARS)
    )
    if sex_col:
        mask &= raw[sex_col].str.contains(r"both|total", case=False, na=False)

    lfs = raw[mask].copy()
    lfs["Year"] = _year(lfs["REF_DATE"])
    lfs = lfs[lfs["Year"].isin(STUDY_YEARS)]

    annual = (
        lfs.groupby(["Year", "GEO", char_col])["VALUE"]
        .mean().reset_index()
    )

    pivot = annual.pivot_table(
        index=["Year", "GEO"], columns=char_col, values="VALUE"
    ).reset_index()
    pivot.columns.name = None
    pivot.rename(columns={
        "GEO":                   "Province",
        "Unemployment rate":     "Unemployment_Rate",
        "Participation rate":    "Participation_Rate",
        "Full-time employment":  "FT_Employment",
        "Part-time employment":  "PT_Employment",
    }, inplace=True)

    total = pivot["FT_Employment"] + pivot["PT_Employment"]
    pivot["Part_Time_Share"] = (
        pivot["PT_Employment"] / total.replace(0, np.nan)
    ) * 100

    print(f"    → {len(pivot)} province-year rows")
    return pivot


# ______________________________________________________________________________
# STEP 2 — WAGES  (14-10-0064-01)
# Average hourly wage rate, 20–24 and 25–29 averaged → 20–29 cohort


def process_wages() -> pd.DataFrame:
    print("\n[2/4]  Wages  14-10-0064-01")
    raw = _fetch("WAGES")

    age_col  = _find_col(raw, "age group", "age")
    wage_col = _find_col(raw, "wages", "wage type", "type of wage")

    raw['GEO'] = raw['GEO'].str.strip()
    raw[age_col] = raw[age_col].str.strip()
    raw[wage_col] = raw[wage_col].str.strip()

    work_col = next((c for c in raw.columns
                     if "type of work" in c.lower()), None)
    ind_col  = next((c for c in raw.columns
                     if "naics" in c.lower() or "industry" in c.lower()), None)
    sex_col  = next((c for c in raw.columns
                     if c.lower() in ("sex", "gender")), None)

    mask = (
        raw["GEO"].isin(PROVINCES) &
        raw[age_col].isin(YOUTH_AGES_WAGE) &
        raw[wage_col].str.contains("average hourly", case=False, na=False)
    )
    if work_col:
        mask &= raw[work_col].str.contains(r"both|all", case=False, na=False)
    if ind_col:
        mask &= raw[ind_col].str.contains(r"total|all industries", case=False, na=False)
    if sex_col:
        mask &= raw[sex_col].str.contains(r"both|total", case=False, na=False)

    wages = raw[mask].copy()

    if wages.empty:
        print(f"⚠ Critical Error: No rows found for Wages!")
        print(f"Check 1: GEO matches? {raw['GEO'].unique()[:3]}")
        print(f"Check 2: Age groups match? {raw[age_col].unique()[:5]}")
        return pd.DataFrame(columns=["Year", "Province", "Nominal_Wage"])
    
    wages["Year"] = _year(wages["REF_DATE"])
    wages = wages[wages["Year"].isin(STUDY_YEARS)]

    # Average across the two youth age groups per province-year
    agg = (
        wages.groupby(["Year", "GEO"])["VALUE"]
        .mean().reset_index()
        .rename(columns={"GEO": "Province", "VALUE": "Nominal_Wage"})
    )
    print(f"    → {len(agg)} province-year rows")
    return agg


#_____________________________________________________________________________
# STEP 3 — PART-TIME SHARE  (14-10-0020-01)
# Coarser age groups; weighted blend 15–24 (40%) + 25–44 (60%)
# Only used to fill Part_Time_Share gaps from the LFS table

def process_pt_share() -> pd.DataFrame | None:
    print("\n[3/4]  Part-time share  14-10-0020-01  (supplemental)")
    try:
        raw = _fetch("PT")
    except Exception as exc:
        print(f"    ⚠ Skipped (not critical): {exc}")
        return None

    age_col  = _find_col(raw, "age group", "age")
    char_col = _find_col(raw, "labour force char", "characteristic")
    edu_col  = next((c for c in raw.columns
                     if "education" in c.lower()), None)
    sex_col  = next((c for c in raw.columns
                     if c.lower() in ("sex", "gender")), None)

    PT_AGES = list(PT_AGE_WEIGHTS.keys())
    PT_CHARS = ["Full-time employment", "Part-time employment"]

    mask = (
        raw["GEO"].isin(PROVINCES) &
        raw[age_col].isin(PT_AGES) &
        raw[char_col].isin(PT_CHARS)
    )
    if edu_col:
        mask &= raw[edu_col].str.contains("total", case=False, na=False)
    if sex_col:
        mask &= raw[sex_col].str.contains(r"both|total", case=False, na=False)

    pt = raw[mask].copy()
    pt["Year"] = _year(pt["REF_DATE"])
    pt = pt[pt["Year"].isin(STUDY_YEARS)]
    pt["weight"] = pt[age_col].map(PT_AGE_WEIGHTS)

    def wavg(g):
        v = g["VALUE"].fillna(g["VALUE"].mean())
        return float(np.average(v, weights=g["weight"]))

    blended = (
        pt.groupby(["Year", "GEO", char_col])
        .apply(wavg, include_groups=False)
        .reset_index(name="VALUE")
    )
    pivot = blended.pivot_table(
        index=["Year", "GEO"], columns=char_col, values="VALUE"
    ).reset_index()
    pivot.columns.name = None

    ft_col = next((c for c in pivot.columns if "full" in c.lower()), None)
    pt_col = next((c for c in pivot.columns if "part" in c.lower()), None)
    if ft_col and pt_col:
        total = pivot[ft_col] + pivot[pt_col]
        pivot["Part_Time_Share_Supp"] = (
            pivot[pt_col] / total.replace(0, np.nan)
        ) * 100
        pivot.rename(columns={"GEO": "Province"}, inplace=True)
        print(f"    → {len(pivot)} province-year rows (gap-fill only)")
        return pivot[["Year", "Province", "Part_Time_Share_Supp"]]

    return None


# ________________________________________________________________________________
# STEP 4 — CPI  (18-10-0004-01)
# All-items, provincial, monthly → annual mean

def process_cpi() -> pd.DataFrame:
    print("\n[4/4]  CPI  18-10-0004-01")
    raw = _fetch("CPI")

    prod_col = _find_col(raw, "product")

    mask = (
        raw["GEO"].isin(PROVINCES) &
        (raw[prod_col] == "All-items")
    )
    cpi = raw[mask].copy()
    cpi["Year"] = _year(cpi["REF_DATE"])
    cpi = cpi[cpi["Year"].isin(STUDY_YEARS)]

    agg = (
        cpi.groupby(["Year", "GEO"])["VALUE"]
        .mean().reset_index()
        .rename(columns={"GEO": "Province", "VALUE": "CPI_Index"})
    )
    print(f"    → {len(agg)} province-year rows")
    return agg


# _________________________________________________________________________________
# STEP 5 — MERGE 

def build_panel(
    lfs:    pd.DataFrame,
    wages:  pd.DataFrame,
    cpi:    pd.DataFrame,
    pt_sup: pd.DataFrame | None,
) -> pd.DataFrame:

    print("\n[5/5]  Merging and deriving variables …")

    df = lfs.merge(wages, on=["Year", "Province"], how="left")
    df = df.merge(cpi,   on=["Year", "Province"], how="left")

    # Fill Part_Time_Share gaps from supplemental table
    if pt_sup is not None:
        df = df.merge(pt_sup, on=["Year", "Province"], how="left")
        if "Part_Time_Share" in df.columns:
            df["Part_Time_Share"] = df["Part_Time_Share"].fillna(
                df["Part_Time_Share_Supp"]
            )
        else:
            df.rename(columns={"Part_Time_Share_Supp": "Part_Time_Share"},
                      inplace=True)
        df.drop(columns="Part_Time_Share_Supp", errors="ignore", inplace=True)

    # Province abbreviation
    df["Province_Abbr"] = df["Province"].map(PROV_ABBR)

    # Real wage: deflate to 2019 province-specific prices
    cpi_2019 = (
        cpi[cpi["Year"] == 2019]
        .set_index("Province")["CPI_Index"].to_dict()
    )
    df["_base"] = df["Province"].map(cpi_2019)
    df["Real_Wage"] = df["Nominal_Wage"] * (df["_base"] / df["CPI_Index"])
    df.drop(columns="_base", inplace=True)

    # Post-2020 binary
    df["Post2020"] = (df["Year"] >= 2020).astype(int)

    # Stress Index: three z-score components
    #   Unemployment  → high = more stress  (+)
    #   Participation → low  = more stress  (−)
    #   Real Wage     → low  = more stress  (−)
    _signs = {"Unemployment_Rate": +1, "Participation_Rate": -1, "Real_Wage": -1}
    _zcols = []
    for col, sign in _signs.items():
        mu, sd = df[col].mean(), df[col].std()
        z = f"_z{col}"
        df[z] = sign * (df[col] - mu) / sd
        _zcols.append(z)
    df["Stress_Index"] = df[_zcols].mean(axis=1)
    df.drop(columns=_zcols, inplace=True)

    # Canonical column order
    cols = [
        "Province", "Province_Abbr", "Year",
        "Unemployment_Rate", "Participation_Rate",
        "FT_Employment", "PT_Employment", "Part_Time_Share",
        "Nominal_Wage", "CPI_Index", "Real_Wage",
        "Post2020", "Stress_Index",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values(["Province", "Year"]).reset_index(drop=True)

    print(f"    → {len(df)} rows × {len(df.columns)} columns")
    return df


def build_research_data(
    output_path: str = "research_data_2015_2024.csv",
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Full pipeline: download → parse → merge → derive → save.
    Called by main.py on first launch; runnable standalone too.

    Parameters
    ----------
    output_path     : destination CSV (default next to this script)
    force_download  : ignore cache and re-download everything
    """
    print("=" * 64)
    print("  StatCan Youth Labour Market Panel Builder")
    print("  Tables: 14100287 · 14100064 · 14100020 · 18100004")
    print("  Target: 10 provinces × 2015–2024 = 100 province-years")
    print("=" * 64)

    lfs    = process_lfs()
    wages  = process_wages()
    pt_sup = process_pt_share()
    cpi    = process_cpi()
    panel  = build_panel(lfs, wages, cpi, pt_sup)

    core = ["Unemployment_Rate", "Participation_Rate", "Real_Wage"]
    n_before = len(panel)
    panel.dropna(subset=core, inplace=True)
    if len(panel) < n_before:
        print(f"\n  ⚠ Dropped {n_before - len(panel)} rows with missing core vars.")

    panel.to_csv(output_path, index=False)
    print(f"\n✓  Saved → {output_path}  ({len(panel)} rows)")

    print("\n── Summary statistics ──────────────────────────────────────")
    print(panel[core + ["Stress_Index"]].describe().round(3).to_string())
    print("\n── Province coverage ───────────────────────────────────────")
    cov = panel.groupby("Province")["Year"].agg(["min", "max", "count"])
    cov.columns = ["First", "Last", "N"]
    print(cov.to_string())

    return panel


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Re-download all tables even if cached.")
    p.add_argument("--output", default="research_data_2015_2024.csv")
    args = p.parse_args()
    df = build_research_data(output_path=args.output,
                              force_download=args.force)
    print("\n── First 12 rows ───────────────────────────────────────────")
    print(df.head(12).to_string(index=False))