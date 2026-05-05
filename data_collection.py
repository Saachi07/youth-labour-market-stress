"""
data_collection.py  (v3 — full heterogeneity panel)
=====================================================
Builds research_data_monthly.csv: a province × month × age-cohort panel
for the Youth Labour Market Stress paper.

INPUTS (all in ./data/ folder relative to this script, or paths override below)
────────────────────────────────────────────────────────────────────────────────
  lfs_monthly.csv      Custom download from StatCan table 14-10-0287-02
                       URL: see README — filtered to 10 provinces, age 20-24 + 25-29,
                       Jan 2015 – Dec 2024, total gender
                       Variables: Unemployment rate, Participation rate, FT/PT employment

  14100064-eng.zip     Employee wages by industry, annual (table 14-10-0064-01)
                       → wages by age / gender / work-type / industry

  14100020-eng.zip     LFS by educational attainment, annual (table 14-10-0020-01)
                       → FT/PT employment + participation by gender (annual, coarser age)

  18100004-eng.zip     CPI monthly, not seasonally adjusted (table 18-10-0004-01)
                       → provincial price deflation

  17100005-eng.zip     Population estimates by age (table 17-10-0005)
                       → youth population share per province-year

OUTPUT: research_data_monthly.csv
──────────────────────────────────
Columns per province × month × age-group row:
  Province, Province_Abbr, Year, Month, YearMonth, Age_Group (20-24 / 25-29)
  Unemployment_Rate, Participation_Rate
  FT_Employment, PT_Employment, Part_Time_Share
  Nominal_Wage, CPI_Index, Real_Wage
  Gender_Wage_Gap        ← Women / Men avg hourly wage ratio (annual, merged monthly)
  FT_Wage_Premium        ← FT / PT avg hourly wage ratio
  Industry_Exposure      ← (hosp + retail wage) / (2 × all-industries wage) for youth
  Youth_Pop_Share        ← cohort population / total provincial population
  Post2020               ← 1 from January 2020 onward
  Stress_Index           ← standardised within age-cohort

REGRESSION SPECIFICATIONS (see paper §3.2)
───────────────────────────────────────────
Part 1 — Baseline:
  Stress_it = α + β·Unemployment_it + μ_i + τ_t + γ_m + ε_it
  (reported separately for Age_Group = 20-24, 25-29, pooled)

Part 2 — Interaction (main result):
  Stress_it = β₀ + β₁·Unemp_it + β₂·Post_t + β₃·(Unemp×Post)_it
            + μ_i + τ_t + γ_m + ε_it
  (β₃ = decoupling test; μ_i = province FE, τ_t = year FE, γ_m = month FE)

Heterogeneity sub-questions (moderators):
  1. Gender:            Gender_Wage_Gap as interaction moderator
  2. FT/PT:             Part_Time_Share and FT_Wage_Premium
  3. Industry exposure: Industry_Exposure index + separate regressions by NAICS sector
  4. Province age:      Youth_Pop_Share

Run:  python data_collection.py
      python data_collection.py --lfs path/to/lfs_monthly.csv
"""

import io, os, sys, zipfile, argparse
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(_HERE, "data")

def _data(*parts):
    return os.path.join(DATA, *parts)

# ── Constants ──────────────────────────────────────────────────────────────────
PROVINCES = [
    'Alberta', 'British Columbia', 'Manitoba', 'New Brunswick',
    'Newfoundland and Labrador', 'Nova Scotia', 'Ontario',
    'Prince Edward Island', 'Quebec', 'Saskatchewan',
]
PROV_ABBR = {
    'Alberta': 'AB', 'British Columbia': 'BC', 'Manitoba': 'MB',
    'New Brunswick': 'NB', 'Newfoundland and Labrador': 'NL',
    'Nova Scotia': 'NS', 'Ontario': 'ON', 'Prince Edward Island': 'PE',
    'Quebec': 'QC', 'Saskatchewan': 'SK',
}
STUDY_YEARS  = list(range(2015, 2025))
AGE_LABEL    = {'20 to 24 years': '20-24', '25 to 29 years': '25-29'}

# Youth-exposed industries for the industry exposure index
YOUTH_INDUSTRIES = [
    'Accommodation and food services [72]',
    'Wholesale and retail trade [41, 44-45]',
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def _find_col(df, *subs):
    for sub in subs:
        for c in df.columns:
            if sub.lower() in c.lower():
                return c
    raise KeyError(f"No column matching {subs} in {list(df.columns)}")


def _open_zip(zip_path, csv_stem):
    """Open a StatCan zip and return a DataFrame for the matching CSV inside."""
    with zipfile.ZipFile(zip_path) as zf:
        match = next((n for n in zf.namelist()
                      if csv_stem in n and n.endswith('.csv')), None)
        if not match:
            raise FileNotFoundError(
                f"Could not find '{csv_stem}' in {zip_path}. "
                f"Contents: {zf.namelist()[:8]}")
        with zf.open(match) as f:
            df = pd.read_csv(f, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def _strip(df):
    """Strip whitespace from all string columns."""
    for c in df.select_dtypes('object').columns:
        df[c] = df[c].str.strip()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LFS MONTHLY  (custom download CSV from 14-10-0287-02)
# ══════════════════════════════════════════════════════════════════════════════

def process_lfs(lfs_path: str) -> pd.DataFrame:
    """
    Parse the custom-downloaded LFS CSV from StatCan table 14-10-0287-02.

    REQUIRED file contents:
      - GEO: 10 provinces
      - Age group: "20 to 24 years" AND "25 to 29 years"
      - Gender: "Both sexes" or "Total - Gender"
      - Labour force characteristics: Unemployment rate, Participation rate,
                                      Full-time employment, Part-time employment
      - REF_DATE: monthly, Jan 2015 – Dec 2024

    Source table options — use EITHER:
    ──────────────────────────────────
    PREFERRED: Table 14-10-0017-02  (detailed age groups, NOT seasonally adjusted)
      URL: https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=1410001702
      Has 20-24 and 25-29 exact age groups by province.
      Not seasonally adjusted — handled by month fixed effects in regression.

    FALLBACK:  Table 14-10-0287-02  (seasonally adjusted, but only 15-24 and 25-54)
      URL: https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=1410028702
      If you use this, the pipeline will map 15-24 → '20-24' and 25-54 → '25-29'
      and note it as a data limitation in the methodology.

    For either table, set:
      Geography  = all 10 provinces (no Canada)
      Gender     = Both sexes / Total - Gender
      Age group  = select the youth groups (20-24 + 25-29 if available)
      LF chars   = Unemployment rate, Participation rate, FT employment, PT employment
      Date       = January 2015 to December 2024
      Download   = "selected data (for database loading)"
    """
    print(f"\n[1/5]  LFS monthly  →  {lfs_path}")
    if not os.path.exists(lfs_path):
        raise FileNotFoundError(
            f"\n✗ LFS file not found: {lfs_path}\n\n"
            "  Go to: https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=1410028702\n"
            "  Select: 10 provinces, Gender=Both sexes, Age=20-24+25-29,\n"
            "          Characteristics=Unemp rate+Part rate+FT+PT employment,\n"
            "          Date=Jan 2015 to Dec 2024\n"
            "  Download: 'selected data (for database loading)'\n"
            "  Save as: data/lfs_monthly.csv"
        )

    raw = pd.read_csv(lfs_path, low_memory=False)
    raw = _strip(raw)
    print(f"    {len(raw):,} rows × {len(raw.columns)} cols")

    # ── Diagnose the file so errors are instantly understandable ──────────────
    age_col  = _find_col(raw, 'age group', 'age')
    char_col = _find_col(raw, 'labour force char', 'characteristic')
    sex_col  = next((c for c in raw.columns if c.lower() in ('sex','gender')), None)

    print(f"    Age groups found:  {sorted(raw[age_col].dropna().unique())}")
    print(f"    Characteristics:   {sorted(raw[char_col].dropna().unique())}")
    print(f"    Gender values:     {sorted(raw[sex_col].dropna().unique()) if sex_col else 'no gender col'}")
    print(f"    Date range:        {raw['REF_DATE'].min()} → {raw['REF_DATE'].max()}")
    print(f"    Provinces:         {sorted(raw['GEO'].dropna().unique())}")

    # ── Validate and set age mapping ──────────────────────────────────────────
    ages_in_file = set(raw[age_col].dropna().unique())
    exact_ages   = {'20 to 24 years', '25 to 29 years'}
    coarse_ages  = {'15 to 24 years', '25 to 54 years'}

    if exact_ages.issubset(ages_in_file):
        # Best case: exact 20-24 / 25-29 breakout (table 14-10-0017-02)
        active_age_label = AGE_LABEL.copy()   # {'20 to 24 years': '20-24', '25 to 29 years': '25-29'}
        print("    ✓ Exact age groups (20-24, 25-29) found — no proxy needed")
    elif coarse_ages.issubset(ages_in_file):
        # Fallback: coarser breakout (table 14-10-0287-02)
        active_age_label = {'15 to 24 years': '20-24', '25 to 54 years': '25-29'}
        print("    ⚠ Using coarser age proxies: '15 to 24' → '20-24',  '25 to 54' → '25-29'")
        print("      This is a data limitation — note in methodology section §3.1.2")
    else:
        available_youth = [a for a in sorted(ages_in_file)
                           if any(str(n) in a for n in range(15, 35))]
        raise ValueError(
            f"\n✗ No usable youth age groups found in LFS file!\n"
            f"  Found: {sorted(ages_in_file)}\n"
            f"  Youth-ish: {available_youth}\n\n"
            "  Use table 14-10-0017-02 (preferred) or 14-10-0287-02 (fallback).\n"
            "  See docstring for download instructions."
        )

    # Confirm date range covers 2015-2024
    dates = pd.to_datetime(raw['REF_DATE'], errors='coerce').dropna()
    if dates.max().year < 2020:
        raise ValueError(
            f"\n✗ LFS file only covers up to {dates.max().strftime('%Y-%m')}.\n"
            "  Set the date range to Jan 2015 – Dec 2024 when downloading."
        )

    # ── Filter ────────────────────────────────────────────────────────────────
    LF_CHARS = [
        'Unemployment rate', 'Participation rate',
        'Full-time employment', 'Part-time employment',
    ]

    mask = (
        raw['GEO'].isin(PROVINCES) &
        raw[age_col].isin(list(active_age_label.keys())) &
        raw[char_col].isin(LF_CHARS)
    )
    # Gender: accept "Total", "Both sexes", or aggregate Men+ Women+ for counts
    if sex_col:
        has_total = raw[sex_col].str.contains(r'both|total', case=False, na=False).any()
        if has_total:
            mask &= raw[sex_col].str.contains(r'both|total', case=False, na=False)
        else:
            # Only Men+/Women+ available — sum them for employment counts,
            # average for rates (approximate)
            print("    ⚠ No 'Total' gender row — summing Men+ and Women+ for employment, averaging for rates")

    lfs = raw[mask].copy()

    if not has_total and sex_col:
        # Build totals by aggregating both genders
        rate_chars = ['Unemployment rate', 'Participation rate']
        count_chars = ['Full-time employment', 'Part-time employment']
        lfs_all = raw[
            raw['GEO'].isin(PROVINCES) &
            raw[age_col].isin(list(active_age_label.keys())) &
            raw[char_col].isin(LF_CHARS)
        ].copy()
        lfs_rates  = lfs_all[lfs_all[char_col].isin(rate_chars)].groupby(
            ['REF_DATE','GEO',age_col,char_col])['VALUE'].mean().reset_index()
        lfs_counts = lfs_all[lfs_all[char_col].isin(count_chars)].groupby(
            ['REF_DATE','GEO',age_col,char_col])['VALUE'].sum().reset_index()
        lfs = pd.concat([lfs_rates, lfs_counts], ignore_index=True)

    print(f"    After filter: {len(lfs):,} rows")

    # ── Parse date ────────────────────────────────────────────────────────────
    lfs['Date']      = pd.to_datetime(lfs['REF_DATE'], errors='coerce')
    lfs['Year']      = lfs['Date'].dt.year.astype(int)
    lfs['Month']     = lfs['Date'].dt.month.astype(int)
    lfs['YearMonth'] = lfs['REF_DATE'].str[:7]
    lfs = lfs[lfs['Year'].isin(STUDY_YEARS)].copy()

    # ── Pivot: one row per province × month × age-group ───────────────────────
    pivot = lfs.pivot_table(
        index=['YearMonth', 'Year', 'Month', 'GEO', age_col],
        columns=char_col,
        values='VALUE',
        aggfunc='mean',
    ).reset_index()
    pivot.columns.name = None
    pivot.rename(columns={
        'GEO':                   'Province',
        age_col:                 'Age_Group_Raw',
        'Unemployment rate':     'Unemployment_Rate',
        'Participation rate':    'Participation_Rate',
        'Full-time employment':  'FT_Employment',
        'Part-time employment':  'PT_Employment',
    }, inplace=True)

    pivot['Age_Group'] = pivot['Age_Group_Raw'].map(active_age_label)

    # FT/PT share — only compute if both columns exist
    if 'FT_Employment' in pivot.columns and 'PT_Employment' in pivot.columns:
        total = pivot['FT_Employment'] + pivot['PT_Employment']
        pivot['Part_Time_Share'] = pivot['PT_Employment'] / total.replace(0, np.nan) * 100
    else:
        pivot['Part_Time_Share'] = np.nan
        print("    ⚠ FT/PT employment not in file — Part_Time_Share will be NaN")

    pivot.drop(columns='Age_Group_Raw', errors='ignore', inplace=True)

    print(f"    → {len(pivot)} province × month × age-group rows")
    print(f"    Age groups: {sorted(pivot['Age_Group'].dropna().unique())}")
    n_expected = len(PROVINCES) * len(STUDY_YEARS) * 12 * 2
    if len(pivot) < n_expected * 0.9:
        print(f"    ⚠ Expected ~{n_expected} rows, got {len(pivot)} — some data may be missing")
    return pivot


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — WAGES  (14-10-0064-01)
# Annual → interpolated monthly; by gender, work-type, and industry
# ══════════════════════════════════════════════════════════════════════════════

def process_wages() -> pd.DataFrame:
    """
    Returns annual province × age-group panel with:
        Nominal_Wage       (total, both genders, both work types)
        Gender_Wage_Gap    (Women / Men avg hourly wage)
        FT_Wage_Premium    (FT / PT avg hourly wage)
        Industry_Exposure  (hospitality+retail wage / all-industries wage, youth)

    All values interpolated to monthly using cubic spline.
    """
    print("\n[2/5]  Wages  14-10-0064-01")
    zip_path = _data('14100064-eng.zip')
    raw = _open_zip(zip_path, '14100064')
    raw = _strip(raw)
    print(f"    {len(raw):,} rows")

    age_col  = _find_col(raw, 'age group', 'age')
    wage_col = _find_col(raw, 'wages', 'wage type', 'type of wage')
    work_col = next((c for c in raw.columns if 'type of work' in c.lower()), None)
    ind_col  = _find_col(raw, 'naics', 'industry')
    sex_col  = next((c for c in raw.columns if c.lower() in ('sex','gender')), None)

    # Map available age groups to cohort labels
    avail_ages = raw[age_col].dropna().unique().tolist()
    AGE_WAGE_MAP = {}
    for label, candidates in [
        ('20-24', ['20 to 24 years', '15 to 24 years', '15 years and over']),
        ('25-29', ['25 to 29 years', '25 to 54 years', '25 years and over']),
    ]:
        for c in candidates:
            if c in avail_ages:
                AGE_WAGE_MAP[c] = label
                print(f"    Wage age proxy: '{c}' → '{label}'")
                break

    # ── Base filter ────────────────────────────────────────────────────────────
    base_mask = (
        raw['GEO'].isin(PROVINCES) &
        raw[age_col].isin(list(AGE_WAGE_MAP.keys())) &
        raw[wage_col].str.contains('average hourly', case=False, na=False) &
        raw[ind_col].str.contains('total employees, all industries', case=False, na=False)
    )

    # ── A. Total nominal wage ──────────────────────────────────────────────────
    mask_total = base_mask.copy()
    if work_col:
        mask_total &= raw[work_col].str.contains(r'both|all', case=False, na=False)
    if sex_col:
        mask_total &= raw[sex_col].str.contains(r'total|both', case=False, na=False)

    wages_total = raw[mask_total].copy()
    wages_total['Year']      = pd.to_numeric(wages_total['REF_DATE'], errors='coerce').astype('Int64')
    wages_total['Age_Group'] = wages_total[age_col].map(AGE_WAGE_MAP)
    wages_total = wages_total[wages_total['Year'].isin(STUDY_YEARS)]

    annual_total = (
        wages_total.groupby(['Year', 'GEO', 'Age_Group'])['VALUE']
        .mean().reset_index()
        .rename(columns={'GEO': 'Province', 'VALUE': 'Nominal_Wage'})
    )

    # ── B. Gender wage gap ─────────────────────────────────────────────────────
    gap_rows = []
    if sex_col:
        for gender, gname in [('Men+', 'wage_men'), ('Women+', 'wage_women')]:
            gm = base_mask.copy()
            if work_col:
                gm &= raw[work_col].str.contains(r'both|all', case=False, na=False)
            gm &= raw[sex_col] == gender
            sub = raw[gm].copy()
            sub['Year']      = pd.to_numeric(sub['REF_DATE'], errors='coerce').astype('Int64')
            sub['Age_Group'] = sub[age_col].map(AGE_WAGE_MAP)
            sub = sub[sub['Year'].isin(STUDY_YEARS)]
            agg = sub.groupby(['Year', 'GEO', 'Age_Group'])['VALUE'].mean().reset_index()
            agg.rename(columns={'GEO': 'Province', 'VALUE': gname}, inplace=True)
            gap_rows.append(agg)

        if len(gap_rows) == 2:
            gender_df = gap_rows[0].merge(gap_rows[1], on=['Year', 'Province', 'Age_Group'], how='inner')
            gender_df['Gender_Wage_Gap'] = gender_df['wage_women'] / gender_df['wage_men'].replace(0, np.nan)
            gender_df = gender_df[['Year', 'Province', 'Age_Group', 'Gender_Wage_Gap']]
        else:
            gender_df = pd.DataFrame(columns=['Year', 'Province', 'Age_Group', 'Gender_Wage_Gap'])
    else:
        gender_df = pd.DataFrame(columns=['Year', 'Province', 'Age_Group', 'Gender_Wage_Gap'])

    # ── C. FT/PT wage premium ──────────────────────────────────────────────────
    ft_rows = []
    if work_col:
        for wtype, wname in [('Full-time employees', 'wage_ft'), ('Part-time employees', 'wage_pt')]:
            wm = base_mask.copy()
            wm &= raw[work_col].str.contains(wtype[:8], case=False, na=False)
            if sex_col:
                wm &= raw[sex_col].str.contains(r'total|both', case=False, na=False)
            sub = raw[wm].copy()
            sub['Year']      = pd.to_numeric(sub['REF_DATE'], errors='coerce').astype('Int64')
            sub['Age_Group'] = sub[age_col].map(AGE_WAGE_MAP)
            sub = sub[sub['Year'].isin(STUDY_YEARS)]
            agg = sub.groupby(['Year', 'GEO', 'Age_Group'])['VALUE'].mean().reset_index()
            agg.rename(columns={'GEO': 'Province', 'VALUE': wname}, inplace=True)
            ft_rows.append(agg)

        if len(ft_rows) == 2:
            ftpt_df = ft_rows[0].merge(ft_rows[1], on=['Year', 'Province', 'Age_Group'], how='inner')
            ftpt_df['FT_Wage_Premium'] = ftpt_df['wage_ft'] / ftpt_df['wage_pt'].replace(0, np.nan)
            ftpt_df = ftpt_df[['Year', 'Province', 'Age_Group', 'FT_Wage_Premium']]
        else:
            ftpt_df = pd.DataFrame(columns=['Year', 'Province', 'Age_Group', 'FT_Wage_Premium'])
    else:
        ftpt_df = pd.DataFrame(columns=['Year', 'Province', 'Age_Group', 'FT_Wage_Premium'])

    # ── D. Industry exposure index ─────────────────────────────────────────────
    # For each province-year-age: mean(hosp_wage, retail_wage) / all_industries_wage
    # Lower ratio → more youth are in low-wage exposed sectors → higher exposure
    ind_parts = []
    youth_age_for_ind = next(iter(AGE_WAGE_MAP.keys()))  # use finest available youth age

    for ind_name in YOUTH_INDUSTRIES + ['Total employees, all industries']:
        im = (
            raw['GEO'].isin(PROVINCES) &
            (raw[age_col] == youth_age_for_ind) &
            raw[wage_col].str.contains('average hourly', case=False, na=False) &
            raw[ind_col].str.contains(ind_name[:20], case=False, na=False)
        )
        if work_col:
            im &= raw[work_col].str.contains(r'both|all', case=False, na=False)
        if sex_col:
            im &= raw[sex_col].str.contains(r'total|both', case=False, na=False)
        sub = raw[im].copy()
        sub['Year'] = pd.to_numeric(sub['REF_DATE'], errors='coerce').astype('Int64')
        sub = sub[sub['Year'].isin(STUDY_YEARS)]
        agg = (sub.groupby(['Year', 'GEO'])['VALUE'].mean().reset_index()
               .rename(columns={'GEO': 'Province', 'VALUE': ind_name[:20]}))
        ind_parts.append(agg)

    if len(ind_parts) == 3:
        ind_df = ind_parts[0]
        for part in ind_parts[1:]:
            ind_df = ind_df.merge(part, on=['Year', 'Province'], how='outer')
        col_hosp   = YOUTH_INDUSTRIES[0][:20]
        col_retail = YOUTH_INDUSTRIES[1][:20]
        col_all    = 'Total employees, al'
        if all(c in ind_df.columns for c in [col_hosp, col_retail, col_all]):
            ind_df['Industry_Exposure'] = (
                (ind_df[col_hosp] + ind_df[col_retail]) /
                (2 * ind_df[col_all].replace(0, np.nan))
            )
            ind_df = ind_df[['Year', 'Province', 'Industry_Exposure']]
            # Note: industry exposure doesn't vary by age group (only youth wage used)
            # We'll broadcast it to both cohorts during merge
        else:
            ind_df = pd.DataFrame(columns=['Year', 'Province', 'Industry_Exposure'])
    else:
        ind_df = pd.DataFrame(columns=['Year', 'Province', 'Industry_Exposure'])

    # ── Merge all wage components → annual panel ───────────────────────────────
    annual = annual_total.copy()
    for extra_df in [gender_df, ftpt_df]:
        if not extra_df.empty:
            annual = annual.merge(extra_df, on=['Year', 'Province', 'Age_Group'], how='left')
    if not ind_df.empty:
        annual = annual.merge(ind_df, on=['Year', 'Province'], how='left')

    print(f"    Annual wage panel: {len(annual)} rows, cols: {[c for c in annual.columns if c not in ['Year','Province','Age_Group']]}")

    # ── Interpolate annual → monthly using cubic spline ───────────────────────
    all_months   = pd.date_range('2015-01', '2024-12', freq='MS')
    interp_cols  = [c for c in annual.columns
                    if c not in ['Year', 'Province', 'Age_Group']]
    monthly_rows = []

    for (prov, ag), grp in annual.groupby(['Province', 'Age_Group']):
        grp = grp.sort_values('Year').dropna(subset=['Nominal_Wage'])
        if len(grp) < 2:
            continue
        x = grp['Year'].values.astype(float) + 0.5  # mid-year anchor

        row_base = {'Province': prov, 'Age_Group': ag}
        splines  = {}
        for col in interp_cols:
            y = grp[col].fillna(method='ffill').fillna(method='bfill').values
            if len(y) >= 2 and not np.all(np.isnan(y.astype(float))):
                try:
                    splines[col] = CubicSpline(x, y.astype(float), extrapolate=True)
                except Exception:
                    pass

        for dt in all_months:
            t = dt.year + (dt.month - 1) / 12
            row = {**row_base,
                   'YearMonth': dt.strftime('%Y-%m'),
                   'Year':      dt.year,
                   'Month':     dt.month}
            for col, cs in splines.items():
                val = float(cs(t))
                # Clamp wages to plausible range; ratios to [0.3, 1.5]
                if col == 'Nominal_Wage':
                    val = float(np.clip(val, 8, 80))
                elif col in ('Gender_Wage_Gap', 'FT_Wage_Premium', 'Industry_Exposure'):
                    val = float(np.clip(val, 0.3, 2.0))
                row[col] = val
            monthly_rows.append(row)

    wages_monthly = pd.DataFrame(monthly_rows)
    print(f"    → {len(wages_monthly)} province × month × age-group monthly rows")
    return wages_monthly


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CPI MONTHLY  (18-10-0004-01)
# ══════════════════════════════════════════════════════════════════════════════

def process_cpi() -> pd.DataFrame:
    print("\n[3/5]  CPI monthly  18-10-0004-01")
    zip_path = _data('18100004-eng.zip')
    raw = _open_zip(zip_path, '18100004')
    raw = _strip(raw)

    prod_col = _find_col(raw, 'product')
    mask = raw['GEO'].isin(PROVINCES) & (raw[prod_col] == 'All-items')
    cpi = raw[mask].copy()
    cpi['Date']      = pd.to_datetime(cpi['REF_DATE'], errors='coerce')
    cpi['Year']      = cpi['Date'].dt.year.astype(int)
    cpi['Month']     = cpi['Date'].dt.month.astype(int)
    cpi['YearMonth'] = cpi['REF_DATE'].str[:7]
    cpi = cpi[cpi['Year'].isin(STUDY_YEARS)]

    agg = (cpi.groupby(['YearMonth', 'Year', 'Month', 'GEO'])['VALUE']
           .mean().reset_index()
           .rename(columns={'GEO': 'Province', 'VALUE': 'CPI_Index'}))
    print(f"    → {len(agg)} province × month rows")
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — YOUTH POPULATION SHARE  (17-10-0005)
# ══════════════════════════════════════════════════════════════════════════════

def process_population() -> pd.DataFrame | None:
    """
    Returns annual province × age-group Youth_Pop_Share:
        cohort population / total provincial population
    Merged to monthly by broadcast (annual value repeated for each month).
    Returns None gracefully if zip not found (Youth_Pop_Share column is omitted).
    Download from: https://www150.statcan.gc.ca/n1/tbl/csv/17100005-eng.zip
    """
    print("\n[4/5]  Population  17-10-0005")
    zip_path = _data('17100005-eng.zip')
    if not os.path.exists(zip_path):
        print("    ⚠ 17100005-eng.zip not found — Youth_Pop_Share will be omitted.")
        print("    Download from: https://www150.statcan.gc.ca/n1/tbl/csv/17100005-eng.zip")
        print("    Save to: data/17100005-eng.zip  (only ~4MB)")
        return None
    raw = _open_zip(zip_path, '17100005')
    raw = _strip(raw)

    age_col = _find_col(raw, 'age group', 'age')
    sex_col = next((c for c in raw.columns if c.lower() in ('sex','gender')), None)

    sex_filter = sex_col is not None

    # Youth cohort populations
    youth_mask = (
        raw['GEO'].isin(PROVINCES) &
        raw[age_col].isin(['20 to 24 years', '25 to 29 years']) &
        (raw['UOM'] == 'Persons')
    )
    if sex_filter:
        youth_mask &= raw[sex_col].str.contains(r'total|both', case=False, na=False)

    youth = raw[youth_mask].copy()
    youth['Year']      = pd.to_numeric(youth['REF_DATE'], errors='coerce').astype('Int64')
    youth['Age_Group'] = youth[age_col].map(AGE_LABEL)
    youth = youth[youth['Year'].isin(STUDY_YEARS)]

    youth_agg = (youth.groupby(['Year', 'GEO', 'Age_Group'])['VALUE']
                 .mean().reset_index()
                 .rename(columns={'GEO': 'Province', 'VALUE': 'Youth_Pop'}))

    # Total provincial population
    total_mask = (
        raw['GEO'].isin(PROVINCES) &
        (raw[age_col] == 'All ages') &
        (raw['UOM'] == 'Persons')
    )
    if sex_filter:
        total_mask &= raw[sex_col].str.contains(r'total|both', case=False, na=False)

    total = raw[total_mask].copy()
    total['Year'] = pd.to_numeric(total['REF_DATE'], errors='coerce').astype('Int64')
    total = total[total['Year'].isin(STUDY_YEARS)]

    total_agg = (total.groupby(['Year', 'GEO'])['VALUE']
                 .mean().reset_index()
                 .rename(columns={'GEO': 'Province', 'VALUE': 'Total_Pop'}))

    pop = youth_agg.merge(total_agg, on=['Year', 'Province'], how='left')
    pop['Youth_Pop_Share'] = pop['Youth_Pop'] / pop['Total_Pop'].replace(0, np.nan) * 100
    pop = pop[['Year', 'Province', 'Age_Group', 'Youth_Pop_Share']]

    print(f"    → {len(pop)} province × year × age-group rows")
    print(f"    Sample:\n{pop[pop['Year']==2022].sort_values('Province').to_string(index=False)}")
    return pop


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — MERGE + DERIVE
# ══════════════════════════════════════════════════════════════════════════════

def build_panel(lfs, wages_m, cpi, pop) -> pd.DataFrame:  # pop may be None
    print("\n[5/5]  Merging all components …")

    # LFS × wages (both have YearMonth + Province + Age_Group)
    df = lfs.merge(wages_m, on=['YearMonth', 'Year', 'Month', 'Province', 'Age_Group'],
                   how='left')

    # × CPI (YearMonth + Province, no age split)
    df = df.merge(cpi, on=['YearMonth', 'Year', 'Month', 'Province'], how='left')

    # × population (optional — Year + Province + Age_Group, annual → broadcast to months)
    if pop is not None:
        df = df.merge(pop, on=['Year', 'Province', 'Age_Group'], how='left')

    # Province abbreviation
    df['Province_Abbr'] = df['Province'].map(PROV_ABBR)

    # Real wage: deflate to 2019 provincial CPI base
    cpi_2019 = (cpi[cpi['Year'] == 2019]
                .groupby('Province')['CPI_Index'].mean().to_dict())
    df['_cpi_base'] = df['Province'].map(cpi_2019)
    df['Real_Wage']  = df['Nominal_Wage'] * (df['_cpi_base'] / df['CPI_Index'])
    df.drop(columns='_cpi_base', inplace=True)

    # Post-2020 binary (January 2020 onward)
    df['Date']    = pd.to_datetime(df['YearMonth'], format='%Y-%m')
    df['Post2020'] = (df['Date'] >= pd.Timestamp('2020-01-01')).astype(int)
    df.drop(columns='Date', inplace=True)

    # Stress Index: standardised WITHIN each age cohort over the full panel
    #   +z(Unemployment)  − z(Participation)  − z(Real_Wage)
    for ag, idx in df.groupby('Age_Group').groups.items():
        for col, sign in [('Unemployment_Rate', +1),
                          ('Participation_Rate', -1),
                          ('Real_Wage',          -1)]:
            mu = df.loc[idx, col].mean()
            sd = df.loc[idx, col].std()
            if sd > 0:
                df.loc[idx, f'_z_{col}'] = sign * (df.loc[idx, col] - mu) / sd
            else:
                df.loc[idx, f'_z_{col}'] = 0.0
    z_cols = [f'_z_{c}' for c in ['Unemployment_Rate', 'Participation_Rate', 'Real_Wage']]
    df['Stress_Index'] = df[z_cols].mean(axis=1)
    df.drop(columns=z_cols, inplace=True)

    # Canonical column order
    cols = [
        'Province', 'Province_Abbr', 'Year', 'Month', 'YearMonth', 'Age_Group',
        'Unemployment_Rate', 'Participation_Rate',
        'FT_Employment', 'PT_Employment', 'Part_Time_Share',
        'Nominal_Wage', 'CPI_Index', 'Real_Wage',
        'Gender_Wage_Gap', 'FT_Wage_Premium',
        'Industry_Exposure', 'Youth_Pop_Share',
        'Post2020', 'Stress_Index',
    ]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values(['Province', 'Age_Group', 'Year', 'Month']).reset_index(drop=True)

    print(f"    → {len(df):,} rows × {len(df.columns)} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def print_diagnostics(df: pd.DataFrame):
    print("\n" + "=" * 64)
    print("  DIAGNOSTICS")
    print("=" * 64)

    print(f"\n  Shape:  {df.shape}")
    print(f"  Age groups: {sorted(df['Age_Group'].unique())}")
    print(f"  Provinces:  {df['Province'].nunique()}")
    print(f"  Months:     {df['YearMonth'].nunique()} (expect 120)")

    print("\n  Missing values:")
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if len(miss):
        print(miss.to_string())
    else:
        print("  None")

    print("\n  Summary by age group:")
    core = ['Unemployment_Rate', 'Participation_Rate', 'Real_Wage', 'Stress_Index']
    avail = [c for c in core if c in df.columns]
    print(df.groupby('Age_Group')[avail].agg(['mean', 'std']).round(3).to_string())

    print("\n  Pre vs Post-2020 means:")
    print(df.groupby(['Age_Group', 'Post2020'])[avail].mean().round(3).to_string())

    print("\n  Coverage (rows per province):")
    cov = (df.groupby(['Province', 'Age_Group'])
           .agg(N=('YearMonth', 'count'),
                First=('YearMonth', 'min'),
                Last=('YearMonth', 'max'))
           .reset_index())
    print(cov.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def build_research_data(
    lfs_path: str  = None,
    output_path: str = None,
) -> pd.DataFrame:

    lfs_path    = lfs_path    or _data('lfs_monthly.csv')
    output_path = output_path or os.path.join(_HERE, 'research_data_monthly.csv')

    print("=" * 64)
    print("  Youth Labour Market Stress — Panel Builder v3")
    print("  10 provinces × 120 months × 2 age groups = 2,400 rows")
    print("=" * 64)
    print("  Expected files in ./data/:")
    print("    lfs_monthly.csv      (custom StatCan download — see docstring)")
    print("    14100064-eng.zip     (wages)")
    print("    14100020-eng.zip     (LFS educ)")
    print("    18100004-eng.zip     (CPI)")
    print("    17100005-eng.zip     (population — OPTIONAL)")
    print()

    lfs     = process_lfs(lfs_path)
    wages_m = process_wages()
    cpi     = process_cpi()
    pop     = process_population()   # returns None if zip not found — OK
    panel   = build_panel(lfs, wages_m, cpi, pop)

    # Drop rows missing all three core stress components
    core = ['Unemployment_Rate', 'Participation_Rate', 'Real_Wage']
    n_before = len(panel)
    panel.dropna(subset=core, inplace=True)
    if len(panel) < n_before:
        print(f"\n  ⚠ Dropped {n_before - len(panel)} rows with missing core variables.")

    panel.to_csv(output_path, index=False)
    print(f"\n✓  Saved → {output_path}  ({len(panel):,} rows)")

    print_diagnostics(panel)
    return panel


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Build youth labour market stress panel from StatCan tables.')
    ap.add_argument('--lfs',    default=None,
                    help='Path to lfs_monthly.csv (default: data/lfs_monthly.csv)')
    ap.add_argument('--output', default=None,
                    help='Output CSV path (default: research_data_monthly.csv)')
    args = ap.parse_args()
    build_research_data(lfs_path=args.lfs, output_path=args.output)