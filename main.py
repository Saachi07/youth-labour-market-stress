import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# DATA LOAD

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSV  = os.path.join(_HERE, "research_data_monthly.csv")
_CSV_LEGACY = os.path.join(_HERE, "research_data_2015_2024.csv")


def _load() -> pd.DataFrame:
    if os.path.exists(_CSV):
        df = pd.read_csv(_CSV)
        print(f"✓  Loaded {_CSV}  ({len(df)} rows)")
        return df
    if os.path.exists(_CSV_LEGACY):
        # Fallback: load old annual CSV and fake monthly columns so dashboard
        # doesn't crash while user re-runs data_collection.py
        print("⚠  Monthly CSV not found — loading legacy annual CSV as fallback.")
        print("   Run data_collection.py to generate research_data_monthly.csv")
        df = pd.read_csv(_CSV_LEGACY)
        df["Month"] = 7
        df["YearMonth"] = df["Year"].astype(str) + "-07"
        df["COVID"] = (df["Year"] >= 2020).astype(int)
        df["Treat_Province"] = df["Province"].isin(
            {"Newfoundland and Labrador","Nova Scotia","New Brunswick","Alberta","Quebec"}
        ).astype(int)
        if "Age_Group" not in df.columns:
            df["Age_Group"] = "20-29"
        return df
    print("No CSV found — downloading from Statistics Canada …")
    from data_collection import build_research_data
    return build_research_data(_CSV)


df = _load()

# ── KMeans (k=3) ──────────────────────────────────────────────────────────────
_FEATS = [c for c in ["Unemployment_Rate","Participation_Rate","Real_Wage","Part_Time_Share"]
          if c in df.columns]
_X  = StandardScaler().fit_transform(df[_FEATS].fillna(df[_FEATS].mean()))
_km = KMeans(n_clusters=3, random_state=42, n_init=10)
df["_k"] = _km.fit_predict(_X)
_ord = df.groupby("_k")["Stress_Index"].mean().sort_values().index.tolist()
_MAP = {_ord[0]: "Low Stress", _ord[1]: "Medium Stress", _ord[2]: "High Stress"}
df["Cluster"] = df["_k"].map(_MAP)
df.drop(columns="_k", inplace=True)

# ── Static lookups ─────────────────────────────────────────────────────────────
ALL_PROVS  = sorted(df["Province"].unique())
DEF_PROVS  = ["Ontario","Alberta","British Columbia","Quebec","Nova Scotia"]
AGE_GROUPS = sorted(df["Age_Group"].unique()) if "Age_Group" in df.columns else ["All"]

NAVY  = "#1F3864"; BLUE  = "#2F75B6"; RED   = "#C00000"
GREEN = "#1E5631"; GOLD  = "#BF8F00"; GRAY  = "#595959"
PALE  = "#F0F2F6"; WHITE = "#FFFFFF"
CMAP  = {"Low Stress": BLUE, "Medium Stress": GOLD, "High Stress": RED}
AGE_COLORS = {"20-24": RED, "25-29": BLUE, "20-29": NAVY}

INDS = [
    {"label": "Stress Index (composite)",     "value": "Stress_Index"},
    {"label": "Unemployment Rate (%)",         "value": "Unemployment_Rate"},
    {"label": "Participation Rate (%)",        "value": "Participation_Rate"},
    {"label": "Real Hourly Wage (2019 CAD)",   "value": "Real_Wage"},
    {"label": "Part-Time Share (%)",           "value": "Part_Time_Share"},
    {"label": "Nominal Wage (CAD)",            "value": "Nominal_Wage"},
    {"label": "CPI Index",                     "value": "CPI_Index"},
]
ILBL = {o["value"]: o["label"] for o in INDS}

_PL = dict(
    template="plotly_white", font_family="Inter, Arial, sans-serif", font_size=12,
    plot_bgcolor=WHITE, paper_bgcolor=WHITE,
    margin=dict(l=44, r=14, t=34, b=34),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font_size=11),
    hoverlabel=dict(bgcolor=WHITE, font_size=12),
)


# OLS

def _ols_month_fe(data: pd.DataFrame) -> dict:
    """
    Full professor spec:
        Stress ~ COVID×Unemp + COVID + Unemp + month_FE + Participation + RealWage + province_FE
    Returns dict(names, beta, se, t, p, r2, n) or {}
    """
    need = ["Stress_Index","Unemployment_Rate","COVID","Participation_Rate","Real_Wage","Month","Province"]
    sub = data.dropna(subset=[c for c in need if c in data.columns])
    if len(sub) < 20:
        return {}
    try:
    
        months = pd.get_dummies(sub["Month"], prefix="M", drop_first=True)

        provs  = pd.get_dummies(sub["Province"], prefix="P", drop_first=True)

        unemp  = sub["Unemployment_Rate"].values
        covid  = sub["COVID"].astype(float).values
        part   = sub["Participation_Rate"].values
        wage   = sub["Real_Wage"].values

        X = np.column_stack([
            np.ones(len(sub)),          # intercept
            unemp,                      # β₃  Unemployment
            covid,                      # β₂  COVID
            unemp * covid,              # β₁  COVID × Unemployment  ← KEY
            part,                       # β₄  Participation
            wage,                       # β₅  Real Wage
            months.values,              # month FE
            provs.values,               # province FE
        ])

        # Readable names for first 6 coefficients (rest are FE)
        n_month = months.shape[1]
        n_prov  = provs.shape[1]
        names = (
            ["Constant",
             "Unemployment (β₃)",
             "COVID dummy (β₂)",
             "COVID × Unemployment (β₁) ★",
             "Participation Rate (β₄)",
             "Real Wage (β₅)"] +
            [f"Month FE: {m}" for m in months.columns] +
            [f"Province FE: {p}" for p in provs.columns]
        )

        y = sub["Stress_Index"].values
        beta  = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        n, k  = X.shape
        s2    = (resid**2).sum() / (n - k)
        cov   = s2 * np.linalg.inv(X.T @ X)
        se    = np.sqrt(np.diag(cov))
        tv    = beta / se
        pv    = 2 * stats.t.sf(np.abs(tv), df=n - k)
        r2    = 1 - (resid**2).sum() / ((y - y.mean())**2).sum()
        return dict(names=names, beta=beta, se=se, t=tv, p=pv, r2=r2, n=n,
                    n_month=n_month, n_prov=n_prov)
    except (np.linalg.LinAlgError, ValueError):
        return {}


def _ols_simple(data: pd.DataFrame) -> dict:
    """Simple interaction OLS without month FE — for coefficient plot comparisons."""
    need = ["Stress_Index","Unemployment_Rate","COVID","Participation_Rate","Real_Wage"]
    sub = data.dropna(subset=[c for c in need if c in data.columns])
    if len(sub) < 12:
        return {}
    unemp = sub["Unemployment_Rate"].values
    covid = sub["COVID"].astype(float).values if "COVID" in sub.columns else np.zeros(len(sub))
    X = np.column_stack([
        np.ones(len(sub)), unemp, covid, unemp * covid,
        sub["Participation_Rate"].values, sub["Real_Wage"].values,
    ])
    y = sub["Stress_Index"].values
    names = ["Constant","Unemployment (β₃)","COVID (β₂)","COVID×Unemp (β₁) ★",
             "Participation","Real Wage"]
    try:
        beta  = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        n, k  = X.shape
        s2    = (resid**2).sum() / (n - k)
        se    = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        tv    = beta / se
        pv    = 2 * stats.t.sf(np.abs(tv), df=n - k)
        r2    = 1 - (resid**2).sum() / ((y - y.mean())**2).sum()
        return dict(names=names, beta=beta, se=se, t=tv, p=pv, r2=r2, n=n)
    except np.linalg.LinAlgError:
        return {}


def _kpi(title, value, sub="", color=BLUE):
    return dbc.Card(dbc.CardBody([
        html.P(title, className="text-muted mb-1",
               style={"fontSize":"0.74rem","fontWeight":"600",
                      "textTransform":"uppercase","letterSpacing":"0.04em"}),
        html.H4(value, style={"color":color,"fontWeight":"700","margin":0}),
        html.P(sub, className="text-muted mt-1 mb-0",
               style={"fontSize":"0.71rem"}) if sub else None,
    ]), className="shadow-sm border-0 h-100",
        style={"borderLeft":f"5px solid {color}","borderRadius":"8px"})


def _sec(title, sub=""):
    ch = [html.H6(title, style={"color":NAVY,"fontWeight":"700","margin":0})]
    if sub:
        ch.append(html.P(sub, className="text-muted mb-0", style={"fontSize":"0.75rem"}))
    return html.Div(ch, style={"borderBottom":f"2px solid {NAVY}",
                               "paddingBottom":"5px","marginBottom":"12px"})


def _box(*children):
    return html.Div(list(children),
                    style={"background":WHITE,"borderRadius":"8px","padding":"16px",
                           "boxShadow":"0 1px 4px rgba(0,0,0,0.08)"})


# APP LAYOUT

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="Youth Labour Market Stress — Canada",
)
server = app.server   # gunicorn entry point

app.layout = dbc.Container(fluid=True,
    style={"background":PALE,"minHeight":"100vh",
           "fontFamily":"Inter, Arial, sans-serif","padding":0},
children=[

    # Header
    dbc.Row(dbc.Col(html.Div([
        html.H3("Youth Labour Market Stress Dashboard",
                style={"color":WHITE,"fontWeight":"700","margin":"0 0 3px 0"}),
        html.P("Canada 2015–2024  ·  Monthly data  ·  Age groups: 20–24 vs 25–29  ·  "
               "10 provinces  ·  Spec: Stress ~ COVID×Unemp + COVID + Unemp + month_FE + controls",
               style={"color":"rgba(255,255,255,0.72)","fontSize":"0.77rem","margin":0}),
    ], style={"background":NAVY,"padding":"15px 22px","borderRadius":"0 0 8px 8px"})),
    className="mb-3 px-3"),

    # Controls
    dbc.Row([
        dbc.Col([
            html.Label("Provinces", style={"fontWeight":"600","fontSize":"0.80rem"}),
            dcc.Dropdown(id="dd-prov",
                options=[{"label":p,"value":p} for p in ALL_PROVS],
                value=DEF_PROVS, multi=True, clearable=False,
                style={"fontSize":"0.80rem"}),
        ], md=4),
        dbc.Col([
            html.Label("Age Group", style={"fontWeight":"600","fontSize":"0.80rem"}),
            dcc.Dropdown(id="dd-age",
                options=[{"label":"All ages","value":"All"}] +
                        [{"label":a,"value":a} for a in AGE_GROUPS],
                value="All", clearable=False,
                style={"fontSize":"0.80rem"}),
        ], md=2),
        dbc.Col([
            html.Label("Indicator", style={"fontWeight":"600","fontSize":"0.80rem"}),
            dcc.Dropdown(id="dd-ind", options=INDS, value="Stress_Index",
                clearable=False, style={"fontSize":"0.80rem"}),
        ], md=2),
        dbc.Col([
            html.Label("Year Range", style={"fontWeight":"600","fontSize":"0.80rem"}),
            dcc.RangeSlider(id="sl-yr", min=2015, max=2024, step=1,
                value=[2015,2024],
                marks={y:{"label":str(y),"style":{"fontSize":"0.68rem","color":GRAY}}
                       for y in range(2015,2025)},
                tooltip={"placement":"bottom","always_visible":False}),
        ], md=4),
    ], className="mb-3 px-3 py-3 g-2",
       style={"background":WHITE,"borderRadius":"8px",
              "boxShadow":"0 1px 4px rgba(0,0,0,0.07)","margin":"0 12px"}),

    # KPI row
    dbc.Row(id="kpi-row", className="mb-3 px-3 g-2"),

    # Row 1 — Time series | Pre/Post scatter
    dbc.Row([
        dbc.Col(_box(
            _sec("Trend Over Time", "Monthly indicator by province; vertical line = COVID (Mar 2020)"),
            dcc.Graph(id="g-ts", style={"height":"330px"}),
        ), md=8),
        dbc.Col(_box(
            _sec("Pre vs Post-COVID Scatter",
                 "Unemployment → Stress Index; OLS lines per period"),
            dcc.Graph(id="g-sc", style={"height":"330px"}),
        ), md=4),
    ], className="mb-3 px-3 g-2"),

    # Row 2 — Heatmap | Age comparison
    dbc.Row([
        dbc.Col(_box(
            _sec("Stress Index Heatmap",
                 "All 10 provinces × selected years (annual avg) — red = high stress"),
            dcc.Graph(id="g-hm", style={"height":"310px"}),
        ), md=7),
        dbc.Col(_box(
            _sec("Age Group Comparison: 20–24 vs 25–29",
                 "COVID × Unemployment coefficient ± 95% CI by age group"),
            dcc.Graph(id="g-age", style={"height":"310px"}),
        ), md=5),
    ], className="mb-3 px-3 g-2"),

    # Row 3 — Variance decomp | Coefficient plot
    dbc.Row([
        dbc.Col(_box(
            _sec("Variance Decomposition",
                 "Share of partial r² per predictor — pre vs post-COVID"),
            dcc.Graph(id="g-vd", style={"height":"290px"}),
        ), md=5),
        dbc.Col(_box(
            _sec("OLS Coefficient Plot",
                 "COVID × Unemployment Rate → Stress Index  ± 95% CI"),
            dcc.Graph(id="g-co", style={"height":"290px"}),
        ), md=7),
    ], className="mb-3 px-3 g-2"),

    # Row 4 — Regression table (full width)
    dbc.Row([
        dbc.Col(_box(
            _sec("Month-FE Regression",
                 "Stress ~ COVID×Unemp + COVID + Unemp + Σγₘ·Month + Participation + RealWage + Province FE"),
            html.Div(id="tbl-reg"),
        ), md=12),
    ], className="mb-3 px-3 g-2"),

    # Row 5 — Post-COVID Provincial Summary (full width)
    dbc.Row([
        dbc.Col(_box(
            _sec("Post-COVID Provincial Summary",
                 "Averages from March 2020 onward — click any column header to sort"),
            html.Div(id="tbl-pv"),
        ), md=12),
    ], className="mb-4 px-3 g-2"),

    # Footer
    dbc.Row(dbc.Col(html.P(
        "Spec (v2): Stress ~ COVID×Unemp + COVID + Unemp + Σγₘ·Month_m + Participation + RealWage + Province_FE. "
        "COVID = 1 from March 2020. Monthly data: 10 prov × 120 months × 2 age groups. "
        "Wages interpolated cubic-spline from annual StatCan 14-10-0064. "
        "Real Wage = Nominal × (CPI₂₀₁₉_prov / CPI_month_prov). "
        "Stress Index = mean(+z_Unemp − z_Part − z_Wage), standardised within age group.",
        className="text-muted text-center",
        style={"fontSize":"0.70rem","padding":"6px 0 14px"},
    ), className="px-3")),
])



@app.callback(
    [Output("kpi-row","children"),
     Output("g-ts",   "figure"),
     Output("g-sc",   "figure"),
     Output("g-hm",   "figure"),
     Output("g-age",  "figure"),
     Output("g-vd",   "figure"),
     Output("g-co",   "figure"),
     Output("tbl-reg","children"),
     Output("tbl-pv", "children")],
    [Input("dd-prov","value"),
     Input("sl-yr",  "value"),
     Input("dd-ind", "value"),
     Input("dd-age", "value")],
)
def update(sel_provs, yr_range, indicator, sel_age):
    try:
        return _update_inner(sel_provs, yr_range, indicator, sel_age)
    except Exception as exc:
        import traceback; traceback.print_exc()
        empty = go.Figure().update_layout(
            annotations=[dict(text=f"Error: {exc}", xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False, font_size=13)])
        err = html.P(f"Error: {exc}", style={"color":"red","fontSize":"0.80rem"})
        return (html.Div(), empty, empty, empty, empty, empty, empty, err, err)


def _update_inner(sel_provs, yr_range, indicator, sel_age):

    if not sel_provs:
        sel_provs = ALL_PROVS
    y0, y1 = yr_range

    base = df[df["Province"].isin(sel_provs) & df["Year"].between(y0, y1)].copy()

    # Age filter
    if sel_age and sel_age != "All" and "Age_Group" in base.columns:
        filt = base[base["Age_Group"] == sel_age].copy()
    else:
        filt = base.copy()

    pre  = filt[filt["COVID"] == 0] if "COVID" in filt.columns else filt[filt["Year"] < 2020]
    post = filt[filt["COVID"] == 1] if "COVID" in filt.columns else filt[filt["Year"] >= 2020]

    def sm(s):
        v = s.mean()
        return float(v) if pd.notna(v) else 0.0

    # ── KPIs ──────────────────────────────────────────────────────────────────
    pre_s, post_s = sm(pre["Stress_Index"]), sm(post["Stress_Index"])
    delta = ((post_s - pre_s) / abs(pre_s) * 100) if pre_s else 0.0

    kpi = dbc.Row([
        dbc.Col(_kpi("Avg Stress", f"{sm(filt['Stress_Index']):+.2f}",
                     f"{filt['Province'].nunique()} prov · {filt['Year'].nunique()} yrs", NAVY), md=2),
        dbc.Col(_kpi("Stress Δ (post-COVID)", f"{delta:+.1f}%",
                     f"{pre_s:.2f}  →  {post_s:.2f}", RED), md=2),
        dbc.Col(_kpi("Avg Unemployment",
                     f"{sm(filt['Unemployment_Rate']):.1f}%",
                     f"Pre {sm(pre['Unemployment_Rate']):.1f}%  Post {sm(post['Unemployment_Rate']):.1f}%",
                     BLUE), md=2),
        dbc.Col(_kpi("Avg Participation",
                     f"{sm(filt['Participation_Rate']):.1f}%", "", GREEN), md=2),
        dbc.Col(_kpi("Avg Real Wage",
                     f"${sm(filt['Real_Wage']):.2f}", "2019 CAD", GOLD), md=2),
        dbc.Col(_kpi("Observations", str(len(filt)),
                     f"{filt['Province'].nunique()} prov × {filt['Month'].nunique() if 'Month' in filt.columns else '—'} mo",
                     GRAY), md=2),
    ], className="g-2")

    # ── 1. Time series (monthly, annual avg per province) ──────────────────────
    # Aggregate to monthly mean across age groups if "All" selected
    ts_grp = ["YearMonth","Year","Month","Province"] if "YearMonth" in filt.columns else ["Year","Province"]
    ts_data = (filt.groupby(ts_grp, as_index=False)[indicator].mean()
               .sort_values(ts_grp))
    x_col = "YearMonth" if "YearMonth" in ts_data.columns else "Year"

    fig_ts = px.line(ts_data, x=x_col, y=indicator, color="Province",
                     color_discrete_sequence=px.colors.qualitative.Set1,
                     labels={x_col: "", indicator: ILBL.get(indicator, indicator)})

    # add_vline with a string x value crashes in plotly when x-axis is categorical.
    # Use add_shape + add_annotation instead — works for both string and numeric x.
    if x_col == "YearMonth":
        vline_x = "2020-03"
    else:
        vline_x = 2020

    fig_ts.add_shape(
        type="line",
        xref="x", yref="paper",
        x0=vline_x, x1=vline_x, y0=0, y1=1,
        line=dict(color=RED, width=1.5, dash="dot"),
    )
    fig_ts.add_annotation(
        xref="x", yref="paper",
        x=vline_x, y=1.02,
        text="COVID (Mar 2020)",
        showarrow=False,
        font=dict(size=9, color=RED),
        xanchor="left",
    )
    fig_ts.update_layout(**_PL)
    fig_ts.update_xaxes(tickangle=-45, nticks=20)

    # ── 2. Pre/Post scatter (manual numpy trendline — no statsmodels) ──────────
    sc = filt.dropna(subset=["Unemployment_Rate","Stress_Index"]).copy()
    sc["Period"] = sc["COVID"].map({0:"Pre-COVID",1:"Post-COVID"}) if "COVID" in sc.columns \
                  else (sc["Year"] >= 2020).map({False:"Pre-COVID",True:"Post-COVID"})
    fig_sc = px.scatter(sc, x="Unemployment_Rate", y="Stress_Index", color="Period",
                        color_discrete_map={"Pre-COVID":BLUE,"Post-COVID":RED},
                        hover_data=["Province","Year"],
                        labels={"Unemployment_Rate":"Unemployment Rate (%)","Stress_Index":"Stress Index"})
    for period, color in (("Pre-COVID",BLUE),("Post-COVID",RED)):
        s2 = sc[sc["Period"]==period].dropna(subset=["Unemployment_Rate","Stress_Index"])
        if len(s2) > 5:
            xv, yv = s2["Unemployment_Rate"].values, s2["Stress_Index"].values
            m, b_i = np.polyfit(xv, yv, 1)
            xl = np.linspace(xv.min(), xv.max(), 50)
            fig_sc.add_trace(go.Scatter(x=xl, y=m*xl+b_i, mode="lines",
                                        line=dict(color=color,width=2,dash="dash"),
                                        showlegend=False, hoverinfo="skip"))
            r, pv = stats.pearsonr(xv, yv)
            stars = "***" if pv<0.01 else "**" if pv<0.05 else "*"
            ya = 0.96 if period=="Pre-COVID" else 0.85
            fig_sc.add_annotation(text=f"{period}  r = {r:.3f}{stars}",
                                   xref="paper", yref="paper", x=0.02, y=ya,
                                   showarrow=False, font_size=10, font_color=color)
    fig_sc.update_layout(**_PL)

    # ── 3. Heatmap (annual avg across all months and age groups) ─────────────
    # Aggregate monthly data → one value per province × year
    hm_src = (df[df["Year"].between(y0, y1)]
              .dropna(subset=["Province_Abbr", "Stress_Index"])
              .copy())
    hm_src["Year"] = hm_src["Year"].astype(int)
    hm_agg = (hm_src.groupby(["Province_Abbr", "Year"])["Stress_Index"]
              .mean().reset_index()
              .sort_values(["Province_Abbr", "Year"]))
    hm_data = hm_agg.pivot(index="Province_Abbr", columns="Year",
                            values="Stress_Index")
    hm_data.columns = [int(c) for c in hm_data.columns]
    hm_data = hm_data.sort_index()

    # Pass x and y explicitly so plotly uses actual year values on the axis,
    # not positional indices (0,1,2...) which caused only one column to show.
    yr_labels  = [str(y) for y in hm_data.columns]   # x labels
    prov_labels = list(hm_data.index)                  # y labels

    fig_hm = go.Figure(go.Heatmap(
        z=hm_data.values,
        x=yr_labels,
        y=prov_labels,
        colorscale="RdBu_r",
        zmid=0,
        text=[[f"{v:.2f}" if not np.isnan(v) else "" for v in row]
              for row in hm_data.values],
        texttemplate="%{text}",
        textfont={"size": 9},
        colorbar=dict(title="Stress", len=0.7),
        hoverongaps=False,
    ))

    # COVID vertical line — find position of "2020" in x labels
    if "2020" in yr_labels:
        covid_pos = yr_labels.index("2020")
        fig_hm.add_shape(type="line",
                         xref="x", yref="paper",
                         x0=covid_pos - 0.5, x1=covid_pos - 0.5,
                         y0=0, y1=1,
                         line=dict(color="black", width=1.5, dash="dash"))
        fig_hm.add_annotation(xref="x", yref="paper",
                               x=covid_pos - 0.5, y=1.04,
                               text="COVID", showarrow=False,
                               font=dict(size=8, color="black"), xanchor="center")

    fig_hm.update_layout(**_PL,
                          xaxis=dict(title="Year", tickangle=-45, side="bottom"),
                          yaxis=dict(title="", autorange="reversed"))

    # ── 4. Age Group Comparison — COVID×Unemp coefficient ─────────────────────
    fig_age = go.Figure()
    groups = {"20-24": RED, "25-29": BLUE} if "Age_Group" in df.columns \
             else {"All": NAVY}
    for ag, color in groups.items():
        sub_ag = df[df["Province"].isin(sel_provs) & df["Year"].between(y0,y1)]
        if "Age_Group" in sub_ag.columns:
            sub_ag = sub_ag[sub_ag["Age_Group"] == ag]
        res = _ols_simple(sub_ag)
        if not res:
            continue
        # Find the COVID×Unemp interaction coefficient
        idx = next((i for i,n in enumerate(res["names"]) if "COVID×Unemp" in n or "β₁" in n), None)
        if idx is None:
            continue
        b, se = res["beta"][idx], res["se"][idx]
        lo, hi = b - 1.96*se, b + 1.96*se
        fig_age.add_trace(go.Scatter(x=[lo,hi], y=[ag,ag], mode="lines",
                                     line=dict(color=color,width=4), showlegend=False))
        fig_age.add_trace(go.Scatter(x=[b], y=[ag], mode="markers+text",
                                     marker=dict(color=color,size=14,symbol="diamond"),
                                     text=[f"  β₁={b:.3f}  (R²={res['r2']:.2f})"],
                                     textposition="middle right",
                                     textfont=dict(color=color,size=11),
                                     showlegend=False))
    fig_age.add_vline(x=0, line_dash="dash", line_color=GRAY, line_width=1)
    fig_age.update_layout(**_PL,
                           xaxis_title="COVID × Unemployment Coefficient (β₁)",
                           yaxis_title="Age Group",
                           title_text="Larger negative = stronger COVID decoupling in that cohort")

    # ── 5. Variance decomposition (pre/post COVID) ────────────────────────────
    vd_rows = []
    for lbl, mask in [("Pre-COVID\n(≤ Mar 2020)", filt["COVID"]==0 if "COVID" in filt else filt["Year"]<2020),
                       ("Post-COVID\n(> Mar 2020)", filt["COVID"]==1 if "COVID" in filt else filt["Year"]>=2020)]:
        s2 = filt[mask].dropna(subset=["Stress_Index","Unemployment_Rate","Participation_Rate","Real_Wage"])
        if len(s2) < 10:
            continue
        r2u = stats.pearsonr(s2["Unemployment_Rate"],  s2["Stress_Index"])[0]**2
        r2p = stats.pearsonr(s2["Participation_Rate"], s2["Stress_Index"])[0]**2
        r2w = stats.pearsonr(s2["Real_Wage"],          s2["Stress_Index"])[0]**2
        tot = r2u + r2p + r2w or 1
        vd_rows.append({"Period":lbl,"Unemployment":r2u/tot,
                         "Participation":r2p/tot,"Real Wage":r2w/tot})
    if vd_rows:
        vd = pd.DataFrame(vd_rows).melt(id_vars="Period",var_name="Component",value_name="Share")
        fig_vd = px.bar(vd, x="Period", y="Share", color="Component",
                        color_discrete_map={"Unemployment":BLUE,"Participation":GOLD,"Real Wage":GREEN},
                        text=vd["Share"].apply(lambda v:f"{v:.0%}"),
                        labels={"Share":"Share of partial r²","Period":""},
                        barmode="stack")
        fig_vd.update_traces(textposition="inside", textfont_size=10)
        fig_vd.update_yaxes(tickformat=".0%", range=[0,1])
        fig_vd.update_layout(**_PL)
    else:
        fig_vd = go.Figure().update_layout(**_PL)

    # ── 6. Coefficient plot ───────────────────────────────────────────────────
    fig_co = go.Figure()
    for lbl, data, color in [
        ("Full period", filt, NAVY),
        ("Pre-COVID",   pre,  BLUE),
        ("Post-COVID",  post, RED),
    ]:
        res = _ols_simple(data)
        if not res:
            continue
        # COVID×Unemp is index 3; for pre/post subsamples COVID is constant so use Unemp
        idx = 3 if "COVID" in data.columns and data["COVID"].nunique() > 1 else 1
        if idx >= len(res["beta"]):
            continue
        b, se = res["beta"][idx], res["se"][idx]
        lo, hi = b - 1.96*se, b + 1.96*se
        fig_co.add_trace(go.Scatter(x=[lo,hi], y=[lbl,lbl], mode="lines",
                                    line=dict(color=color,width=3), showlegend=False))
        fig_co.add_trace(go.Scatter(x=[b], y=[lbl], mode="markers+text",
                                    marker=dict(color=color,size=14,symbol="diamond"),
                                    text=[f"  {b:.3f}"],
                                    textposition="middle right",
                                    textfont=dict(color=color,size=11),
                                    showlegend=False))
    fig_co.add_vline(x=0, line_dash="dash", line_color=GRAY, line_width=1)
    fig_co.update_layout(**_PL,
                          xaxis_title="Coefficient: COVID×Unemployment → Stress",
                          yaxis_title="")

    # ── 7. Regression table (month FE spec) ───────────────────────────────────
    res = _ols_month_fe(filt)
    if res:
        rows = []
        for i, (nm, b, se, t, pv) in enumerate(
                zip(res["names"], res["beta"], res["se"], res["t"], res["p"])):
            # Only show first 6 coefficients (skip the FE dummies)
            if i >= 6:
                break
            stars = "***" if pv<0.01 else "**" if pv<0.05 else "*" if pv<0.10 else ""
            rows.append({"Variable":nm, "Coef.":f"{b:.3f}{stars}",
                          "Std. Err.":f"({se:.3f})",
                          "t-stat":f"{t:.2f}", "p-value":f"{pv:.3f}"})
        n_mfe = res.get("n_month", 0)
        n_pfe = res.get("n_prov", 0)
        rows.append({"Variable":f"Month FE ({n_mfe+1} dummies)",
                      "Coef.":"Yes","Std. Err.":"","t-stat":"","p-value":""})
        rows.append({"Variable":f"Province FE ({n_pfe+1} dummies)",
                      "Coef.":"Yes","Std. Err.":"","t-stat":"","p-value":""})
        rows.append({"Variable":"R²  |  N",
                      "Coef.":f"{res['r2']:.3f}","Std. Err.":"",
                      "t-stat":"","p-value":str(res["n"])})
        tbl_reg = dash_table.DataTable(
            data=rows,
            columns=[{"name":c,"id":c} for c in rows[0]],
            style_cell={"fontSize":"0.76rem","padding":"5px 7px",
                        "fontFamily":"Inter, Arial","border":"1px solid #eee"},
            style_header={"backgroundColor":NAVY,"color":WHITE,
                           "fontWeight":"700","fontSize":"0.76rem","border":"none"},
            style_data_conditional=[
                {"if":{"row_index":"odd"},"backgroundColor":PALE},
                {"if":{"filter_query":'{Variable} contains "★"'},
                 "backgroundColor":"#FFF3E0","fontWeight":"700"},
            ],
        )
    else:
        tbl_reg = html.P("Select more data for the regression.",
                          className="text-muted", style={"fontSize":"0.80rem"})

    # ── 8. Province summary table ─────────────────────────────────────────────
    post_df = df[df["COVID"]==1] if "COVID" in df.columns else df[df["Year"]>=2020]
    ps = (post_df.groupby("Province")
          .agg(Unemp=("Unemployment_Rate","mean"),
               Part =("Participation_Rate","mean"),
               Wage =("Real_Wage","mean"),
               PT   =("Part_Time_Share","mean"),
               Stress=("Stress_Index","mean"),
               Cluster=("Cluster", lambda x: x.mode().iloc[0]))
          .round(2).reset_index()
          .sort_values("Stress", ascending=False))
    ps.columns = ["Province","Unemp (%)","Part. (%)","Real Wage ($)","PT Share (%)","Stress","Cluster"]
    tbl_pv = dash_table.DataTable(
        data=ps.to_dict("records"),
        columns=[{"name":c,"id":c} for c in ps.columns],
        sort_action="native",
        style_cell={"fontSize":"0.75rem","padding":"5px 7px",
                    "fontFamily":"Inter, Arial","border":"1px solid #eee"},
        style_header={"backgroundColor":NAVY,"color":WHITE,
                       "fontWeight":"700","fontSize":"0.75rem","border":"none"},
        style_data_conditional=[
            {"if":{"row_index":"odd"},"backgroundColor":PALE},
            {"if":{"filter_query":'{Cluster} = "High Stress"'},"backgroundColor":"#FFF0F0"},
            {"if":{"filter_query":'{Cluster} = "Low Stress"'},"backgroundColor":"#EEF6FF"},
        ],
    )

    return (kpi, fig_ts, fig_sc, fig_hm, fig_age, fig_vd, fig_co, tbl_reg, tbl_pv)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)