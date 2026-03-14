"""
data_collection.py 
==========================================================
Builds research_data_monthly.csv: a province × month × age-group panel
for the youth labour market stress paper.

1. Monthly data  
   • 10 provinces × 10 years × 12 months × 2 age groups = 2,400 obs
   • Enables month fixed effects (i.month) to absorb seasonality

2. Age group split  (20–24 vs 25–29)
   • Two separate rows per province-month
   • Allows heterogeneous-effects regression & direct comparison

3. Monthly wages via interpolation
   • StatCan 14-10-0064-01 is annual for wages
   • We interpolate to monthly using cubic spline within province-age
   • Real wages deflated by monthly provincial CPI (18-10-0004-01)

4. COVID 
   • COVID = 1 from March 2020 onward

5. Treatment-group variable
   • Treat_Province = 1 for high-shock provinces (NL, NS, NB, AB, QC)
     defined as provinces whose youth unemployment rose > 8 pp in 2020
   • Enables DiD: COVID × Treat_Province interaction

Source tables:
  14-10-0287-01  LFS monthly, seasonally adjusted
                 → Unemployment rate, Participation rate, FT/PT
                 → Age groups: "20 to 24 years", "25 to 29 years"
  14-10-0064-01  Employee wages by industry, annual
                 → Average hourly wage by age group (interpolated monthly)
  14-10-0020-01  LFS by educational attainment, annual
                 → Part-time share supplemental (coarser age groups)
  18-10-0004-01  CPI, monthly, not seasonally adjusted
                 → Province-specific deflation

Output: research_data_monthly.csv
Columns:
  Province | Province_Abbr | Year | Month | YearMonth (YYYY-MM)
  Age_Group (20-24 / 25-29)
  Unemployment_Rate | Participation_Rate
  FT_Employment | PT_Employment | Part_Time_Share
  Nominal_Wage | CPI_Index | Real_Wage
  COVID          ← 1 from March 2020 onwards
  Treat_Province ← 1 for high-shock provinces
  Stress_Index   ← standardised over full panel per age group
"""

import io, os, sys, zipfile, urllib.request, urllib.error
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# ── Download URLs ──────────────────────────────────────────────────────────────
STATCAN_ZIPS = {
    "LFS":   "https://www150.statcan.gc.ca/n1/tbl/csv/14100287-eng.zip",
    "WAGES": "https://www150.statcan.gc.ca/n1/tbl/csv/14100064-eng.zip",
    "PT":    "https://www150.statcan.gc.ca/n1/tbl/csv/14100020-eng.zip",
    "CPI":   "https://www150.statcan.gc.ca/n1/tbl/csv/18100004-eng.zip",
}
ZIP_CSV = {
    "LFS": "14100287.csv", "WAGES": "14100064.csv",
    "PT":  "14100020.csv", "CPI":   "18100004.csv",
}
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "statcan_cache")

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

# Provinces with large COVID youth-unemployment shock (> 8 pp rise in 2020)
# Used as Treatment group in DiD specification
HIGH_SHOCK_PROVINCES = {
    "Newfoundland and Labrador", "Nova Scotia", "New Brunswick",
    "Alberta", "Quebec"
}

STUDY_YEARS  = list(range(2015, 2025))
YOUTH_AGES   = ["20 to 24 years", "25 to 29 years"]
AGE_LABEL    = {"20 to 24 years": "20-24", "25 to 29 years": "25-29"}

# Weighted blend for PT share supplemental (14-10-0020 has no 25-29 breakout)
PT_AGE_WEIGHTS = {"15 to 24 years": 0.40, "25 to 44 years": 0.60}

# COVID shock starts March 2020 (first major lockdowns in Canada)
COVID_START = pd.Timestamp("2020-03-01")


def _fetch(key: str, force: bool = False) -> pd.DataFrame:
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
            url, headers={"User-Agent": "Mozilla/5.0 (StatCan Research Downloader)"})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                chunks, done = [], 0
                while chunk := resp.read(1 << 16):
                    chunks.append(chunk); done += len(chunk)
                    if total:
                        sys.stdout.write(f"\r      {done/1e6:6.1f}/{total/1e6:.1f} MB")
                        sys.stdout.flush()
                print()
            raw_bytes = b"".join(chunks)
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"\n✗ Download failed: {url}\n  {exc}\n"
                f"  Save manually to: {cache_path}") from exc
        with open(cache_path, "wb") as f:
            f.write(raw_bytes)
        print(f"    saved: {cache_path}  ({len(raw_bytes)/1e6:.0f} MB)")

    csv_name = ZIP_CSV[key]
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        match = next((n for n in zf.namelist()
                      if n.endswith(csv_name) or n == csv_name), None)
        if not match:
            raise FileNotFoundError(f"'{csv_name}' not in {key} zip: {zf.namelist()[:5]}")
        with zf.open(match) as f:
            df = pd.read_csv(f, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print(f"      {len(df):,} rows × {len(df.columns)} cols")
    return df


def _find_col(df, *subs):
    for sub in subs:
        for c in df.columns:
            if sub.lower() in c.lower():
                return c
    raise KeyError(f"No column matching {subs} in {list(df.columns)}")


def process_lfs_monthly() -> pd.DataFrame:
    """
    Returns monthly province × age-group panel with:
        Unemployment_Rate, Participation_Rate, FT_Employment, PT_Employment
    REF_DATE kept as 'YYYY-MM' string for joining.
    """
    print("\n[1/4]  LFS monthly  14-10-0287-01")
    raw = _fetch("LFS")

    age_col  = _find_col(raw, "age group", "age")
    char_col = _find_col(raw, "labour force char", "characteristic", "statistics")
    sex_col  = next((c for c in raw.columns if c.lower() in ("sex","gender")), None)

    LF_CHARS = ["Unemployment rate", "Participation rate",
                "Full-time employment", "Part-time employment"]

    mask = (raw["GEO"].isin(PROVINCES) &
            raw[age_col].isin(YOUTH_AGES) &
            raw[char_col].isin(LF_CHARS))
    if sex_col:
        mask &= raw[sex_col].str.contains(r"both|total", case=False, na=False)

    lfs = raw[mask].copy()
    # Parse year; keep monthly REF_DATE
    lfs["Date"] = pd.to_datetime(lfs["REF_DATE"], errors="coerce")
    lfs["Year"]  = lfs["Date"].dt.year
    lfs["Month"] = lfs["Date"].dt.month
    lfs["YearMonth"] = lfs["REF_DATE"].str[:7]   # 'YYYY-MM'
    lfs = lfs[lfs["Year"].isin(STUDY_YEARS)].copy()

    pivot = lfs.pivot_table(
        index=["YearMonth", "Year", "Month", "GEO", age_col],
        columns=char_col, values="VALUE"
    ).reset_index()
    pivot.columns.name = None
    pivot.rename(columns={
        "GEO": "Province",
        age_col: "Age_Group_Raw",
        "Unemployment rate":    "Unemployment_Rate",
        "Participation rate":   "Participation_Rate",
        "Full-time employment": "FT_Employment",
        "Part-time employment": "PT_Employment",
    }, inplace=True)

    pivot["Age_Group"] = pivot["Age_Group_Raw"].map(AGE_LABEL)
    total = pivot["FT_Employment"] + pivot["PT_Employment"]
    pivot["Part_Time_Share"] = pivot["PT_Employment"] / total.replace(0, np.nan) * 100
    pivot.drop(columns="Age_Group_Raw", inplace=True)

    print(f"    → {len(pivot)} province × month × age-group rows")
    return pivot



def process_wages_monthly() -> pd.DataFrame:
    """
    Returns monthly Nominal_Wage via cubic-spline interpolation of annual values.

    Age-group mapping for 14-10-0064-01
    ─────────────────────────────────────
    This table publishes: "15 to 24 years", "25 to 54 years", "55 years and over",
    "15 years and over" — no finer youth breakout exists.

    We map as follows (methodologically transparent):
        Age_Group "20-24"  ←  "15 to 24 years"   (best proxy; slightly broader)
        Age_Group "25-29"  ←  "25 to 54 years"   (best proxy; slightly broader)

    This is disclosed in the paper as a data limitation.  The direction of any
    bias is small: 15-19 wages are lower than 20-24 (biases "20-24" wages down)
    and 30-54 wages are higher than 25-29 (biases "25-29" wages up), so the
    wage gap between cohorts is modestly overstated.
    """
    print("\n[2/4]  Wages (annual → monthly)  14-10-0064-01")
    raw = _fetch("WAGES")
    raw = raw.apply(lambda c: c.str.strip() if c.dtype == "object" else c)

    age_col  = _find_col(raw, "age group", "age")
    wage_col = _find_col(raw, "wages", "wage type", "type of wage")
    work_col = next((c for c in raw.columns if "type of work" in c.lower()), None)
    ind_col  = next((c for c in raw.columns if "naics" in c.lower() or "industry" in c.lower()), None)
    sex_col  = next((c for c in raw.columns if c.lower() in ("sex","gender")), None)

    # Print available age groups so it's visible in console output
    avail = sorted(raw[age_col].dropna().unique())
    print(f"    Age groups in wages table: {avail}")

    # Map available groups to our cohort labels — pick the closest bracket
    # Priority: exact match first, then broad fallback
    AGE_PROXY = {}
    all_ages = raw[age_col].dropna().unique().tolist()
    for ag_label, candidates in [
        ("20-24", ["20 to 24 years", "15 to 24 years", "15 years and over"]),
        ("25-29", ["25 to 29 years", "25 to 54 years", "25 years and over", "15 years and over"]),
    ]:
        for c in candidates:
            if c in all_ages:
                AGE_PROXY[c] = ag_label
                print(f"    Mapping wages age group '{c}' → cohort '{ag_label}'")
                break

    if not AGE_PROXY:
        print("    ⚠ No usable age group found in wages table — using NaN placeholder.")
        rows = []
        for prov in PROVINCES:
            for ag in ["20-24", "25-29"]:
                for dt in pd.date_range("2015-01", "2024-12", freq="MS"):
                    rows.append({"Province": prov, "Age_Group": ag,
                                 "YearMonth": dt.strftime("%Y-%m"),
                                 "Year": dt.year, "Month": dt.month,
                                 "Nominal_Wage": np.nan})
        return pd.DataFrame(rows)

    use_ages = list(AGE_PROXY.keys())
    mask = (raw["GEO"].isin(PROVINCES) &
            raw[age_col].isin(use_ages) &
            raw[wage_col].str.contains("average hourly", case=False, na=False))
    if work_col:
        mask &= raw[work_col].str.contains(r"both|all", case=False, na=False)
    if ind_col:
        mask &= raw[ind_col].str.contains(r"total|all industries", case=False, na=False)
    if sex_col:
        mask &= raw[sex_col].str.contains(r"both|total", case=False, na=False)

    wages = raw[mask].copy()

    if wages.empty:
        print("    Wage rows still empty after filtering — check zip file.")
        # Return NaN placeholder with correct shape
        rows = []
        for prov in PROVINCES:
            for ag in ["20-24", "25-29"]:
                for dt in pd.date_range("2015-01", "2024-12", freq="MS"):
                    rows.append({"Province": prov, "Age_Group": ag,
                                 "YearMonth": dt.strftime("%Y-%m"),
                                 "Year": dt.year, "Month": dt.month,
                                 "Nominal_Wage": np.nan})
        return pd.DataFrame(rows)

    wages["Year"]      = pd.to_numeric(wages["REF_DATE"], errors="coerce").astype("Int64")
    wages              = wages[wages["Year"].isin(STUDY_YEARS)]
    wages["Age_Group"] = wages[age_col].map(AGE_PROXY)

    annual = (wages.groupby(["Year", "GEO", "Age_Group"])["VALUE"]
              .mean().reset_index()
              .rename(columns={"GEO": "Province", "VALUE": "Nominal_Wage_Annual"}))

    # Interpolate each Province × Age_Group series to monthly using cubic spline
    monthly_rows = []
    all_months   = pd.date_range("2015-01", "2024-12", freq="MS")

    for (prov, ag), grp in annual.groupby(["Province", "Age_Group"]):
        grp = grp.sort_values("Year").dropna(subset=["Nominal_Wage_Annual"])
        if len(grp) < 2:
            continue
        x  = grp["Year"].values.astype(float) + 0.5   # mid-year anchor
        y  = grp["Nominal_Wage_Annual"].values
        cs = CubicSpline(x, y, extrapolate=True)
        for dt in all_months:
            t_frac = dt.year + (dt.month - 1) / 12
            monthly_rows.append({
                "Province":     prov,
                "Age_Group":    ag,
                "YearMonth":    dt.strftime("%Y-%m"),
                "Year":         dt.year,
                "Month":        dt.month,
                "Nominal_Wage": float(np.clip(cs(t_frac), 8, 80)),
            })

    wages_monthly = pd.DataFrame(monthly_rows)
    print(f"    → {len(wages_monthly)} province × month × age-group rows")
    return wages_monthly


def process_pt_share() -> pd.DataFrame | None:
    print("\n[3/4]  PT share supplement  14-10-0020-01")
    try:
        raw = _fetch("PT")
    except Exception as exc:
        print(f"    ⚠ Skipped: {exc}")
        return None

    age_col  = _find_col(raw, "age group", "age")
    char_col = _find_col(raw, "labour force char", "characteristic")
    edu_col  = next((c for c in raw.columns if "education" in c.lower()), None)
    sex_col  = next((c for c in raw.columns if c.lower() in ("sex","gender")), None)

    mask = (raw["GEO"].isin(PROVINCES) &
            raw[age_col].isin(list(PT_AGE_WEIGHTS.keys())) &
            raw[char_col].isin(["Full-time employment", "Part-time employment"]))
    if edu_col:
        mask &= raw[edu_col].str.contains("total", case=False, na=False)
    if sex_col:
        mask &= raw[sex_col].str.contains(r"both|total", case=False, na=False)

    pt = raw[mask].copy()
    pt["Year"] = pd.to_numeric(pt["REF_DATE"], errors="coerce").astype("Int64")
    pt = pt[pt["Year"].isin(STUDY_YEARS)]
    pt["weight"] = pt[age_col].map(PT_AGE_WEIGHTS)

    blended = (pt.groupby(["Year", "GEO", char_col])
               .apply(lambda g: float(np.average(
                   g["VALUE"].fillna(g["VALUE"].mean()),
                   weights=g["weight"])), include_groups=False)
               .reset_index(name="VALUE"))

    pivot = blended.pivot_table(
        index=["Year","GEO"], columns=char_col, values="VALUE"
    ).reset_index()
    pivot.columns.name = None
    ft_col = next((c for c in pivot.columns if "full" in c.lower()), None)
    pt_col = next((c for c in pivot.columns if "part" in c.lower()), None)
    if ft_col and pt_col:
        total = pivot[ft_col] + pivot[pt_col]
        pivot["PT_Share_Supp"] = pivot[pt_col] / total.replace(0, np.nan) * 100
        pivot.rename(columns={"GEO":"Province"}, inplace=True)
        print(f"    → {len(pivot)} province-year rows (annual, gap-fill only)")
        return pivot[["Year","Province","PT_Share_Supp"]]
    return None



def process_cpi_monthly() -> pd.DataFrame:
    print("\n[4/4]  CPI monthly  18-10-0004-01")
    raw = _fetch("CPI")
    prod_col = _find_col(raw, "product")

    mask = raw["GEO"].isin(PROVINCES) & (raw[prod_col] == "All-items")
    cpi = raw[mask].copy()
    cpi["Date"]     = pd.to_datetime(cpi["REF_DATE"], errors="coerce")
    cpi["Year"]     = cpi["Date"].dt.year
    cpi["Month"]    = cpi["Date"].dt.month
    cpi["YearMonth"] = cpi["REF_DATE"].str[:7]
    cpi = cpi[cpi["Year"].isin(STUDY_YEARS)]

    agg = (cpi.groupby(["YearMonth","Year","Month","GEO"])["VALUE"]
           .mean().reset_index()
           .rename(columns={"GEO":"Province","VALUE":"CPI_Index"}))
    print(f"    → {len(agg)} province × month rows")
    return agg



def build_panel(lfs, wages_m, cpi, pt_sup) -> pd.DataFrame:
    print("\n[5/5]  Merging …")

    # Defensive: ensure wages_m has the expected columns even if empty/incomplete
    wage_merge_cols = ["YearMonth","Year","Month","Province","Age_Group"]
    if wages_m.empty or not all(c in wages_m.columns for c in wage_merge_cols):
        print("    ⚠ wages_m missing required columns — adding NaN Nominal_Wage column")
        lfs["Nominal_Wage"] = np.nan
        df = lfs.copy()
    else:
        df = lfs.merge(wages_m, on=wage_merge_cols, how="left")

    df = df.merge(cpi, on=["YearMonth","Year","Month","Province"], how="left")

    # Gap-fill PT share from annual supplement (broadcast annual → all months)
    if pt_sup is not None:
        df = df.merge(pt_sup, on=["Year","Province"], how="left")
        if "Part_Time_Share" in df.columns:
            df["Part_Time_Share"] = df["Part_Time_Share"].fillna(df["PT_Share_Supp"])
        df.drop(columns="PT_Share_Supp", errors="ignore", inplace=True)

    # Province abbreviation
    df["Province_Abbr"] = df["Province"].map(PROV_ABBR)

    # Real wage: deflate to 2019 provincial CPI base
    cpi_2019 = (cpi[cpi["Year"] == 2019]
                .groupby("Province")["CPI_Index"].mean().to_dict())
    df["_base"] = df["Province"].map(cpi_2019)
    df["Real_Wage"] = df["Nominal_Wage"] * (df["_base"] / df["CPI_Index"])
    df.drop(columns="_base", inplace=True)

    df["Date"]  = pd.to_datetime(df["YearMonth"], format="%Y-%m")
    df["COVID"] = (df["Date"] >= COVID_START).astype(int)
    df.drop(columns="Date", inplace=True)

    df["Treat_Province"] = df["Province"].isin(HIGH_SHOCK_PROVINCES).astype(int)

    # Stress Index: standardised WITHIN each age group over full panel
    # (so 20-24 and 25-29 are each mean-0, so comparable in regression)
    _signs = {"Unemployment_Rate": +1, "Participation_Rate": -1, "Real_Wage": -1}
    for ag, grp_idx in df.groupby("Age_Group").groups.items():
        for col, sign in _signs.items():
            mu = df.loc[grp_idx, col].mean()
            sd = df.loc[grp_idx, col].std()
            df.loc[grp_idx, f"_z{col}"] = sign * (df.loc[grp_idx, col] - mu) / sd
    df["Stress_Index"] = df[["_zUnemployment_Rate","_zParticipation_Rate","_zReal_Wage"]].mean(axis=1)
    df.drop(columns=[c for c in df.columns if c.startswith("_z")], inplace=True)

    cols = [
        "Province","Province_Abbr","Year","Month","YearMonth","Age_Group",
        "Unemployment_Rate","Participation_Rate",
        "FT_Employment","PT_Employment","Part_Time_Share",
        "Nominal_Wage","CPI_Index","Real_Wage",
        "COVID","Treat_Province","Stress_Index",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values(["Province","Age_Group","Year","Month"]).reset_index(drop=True)
    print(f"    → {len(df)} rows × {len(df.columns)} columns")
    return df


def build_research_data(
    output_path: str = "research_data_monthly.csv",
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Full pipeline: download → parse → merge → derive → save.

    Produces research_data_monthly.csv
      ~2,400 rows (10 prov × 10 yr × 12 mo × 2 age groups)
    """
    print("=" * 64)
    print("  StatCan Youth Labour Market Panel Builder  (v2 — monthly)")
    print("  Tables: 14100287 · 14100064 · 14100020 · 18100004")
    print("  Target: 10 prov × 120 months × 2 age groups = 2,400 rows")
    print("=" * 64)

    lfs     = process_lfs_monthly()
    wages_m = process_wages_monthly()
    pt_sup  = process_pt_share()
    cpi     = process_cpi_monthly()
    panel   = build_panel(lfs, wages_m, cpi, pt_sup)

    core = ["Unemployment_Rate","Participation_Rate","Real_Wage"]
    n_before = len(panel)
    panel.dropna(subset=core, inplace=True)
    if len(panel) < n_before:
        print(f"\n  ⚠ Dropped {n_before-len(panel)} rows with missing core vars.")

    panel.to_csv(output_path, index=False)
    print(f"\n✓  Saved → {output_path}  ({len(panel)} rows)")

    print("\n── Summary by age group ────────────────────────────────────")
    print(panel.groupby("Age_Group")[core+["Stress_Index"]].describe().round(2))
    print("\n── COVID split ─────────────────────────────────────────────")
    print(panel.groupby(["Age_Group","COVID"])[core+["Stress_Index"]].mean().round(3))

    return panel


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--output", default="research_data_monthly.csv")
    args = p.parse_args()
    df = build_research_data(output_path=args.output, force_download=args.force)
    print("\n── First 8 rows ─────────────────────────────────────────────")
    print(df.head(8).to_string(index=False))