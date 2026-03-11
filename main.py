import os
import warnings
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

from waitress import serve

warnings.filterwarnings("ignore")

# 1.  DATA LOAD

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSV  = os.path.join(_HERE, "research_data_2015_2024.csv")


def _load() -> pd.DataFrame:
    if os.path.exists(_CSV):
        df = pd.read_csv(_CSV)
        print(f"✓  Loaded {_CSV}  ({len(df)} rows)")
        return df
    print("CSV not found — downloading from Statistics Canada …")
    from data_collection import build_research_data
    return build_research_data(_CSV)


df = _load()

# ── KMeans (k=3) ──────────────────────────────────────────────────────────────
_FEATS = [c for c in
          ["Unemployment_Rate", "Participation_Rate", "Real_Wage", "Part_Time_Share"]
          if c in df.columns]
_X   = StandardScaler().fit_transform(df[_FEATS].fillna(df[_FEATS].mean()))
_km  = KMeans(n_clusters=3, random_state=42, n_init=10)
df["_k"] = _km.fit_predict(_X)
_ord = df.groupby("_k")["Stress_Index"].mean().sort_values().index.tolist()
_MAP = {_ord[0]: "Low Stress", _ord[1]: "Medium Stress", _ord[2]: "High Stress"}
df["Cluster"] = df["_k"].map(_MAP)
df.drop(columns="_k", inplace=True)

# ── Constants ──────────────────────────────────────────────────────────────────
ALL_PROVS = sorted(df["Province"].unique())
DEF_PROVS = ["Ontario", "Alberta", "British Columbia", "Quebec", "Nova Scotia"]

NAVY  = "#1F3864"
BLUE  = "#2F75B6"
RED   = "#C00000"
GREEN = "#1E5631"
GOLD  = "#BF8F00"
GRAY  = "#595959"
PALE  = "#F0F2F6"
WHITE = "#FFFFFF"

CMAP = {"Low Stress": BLUE, "Medium Stress": GOLD, "High Stress": RED}

INDS = [
    {"label": "Stress Index (composite)",       "value": "Stress_Index"},
    {"label": "Unemployment Rate (%)",           "value": "Unemployment_Rate"},
    {"label": "Participation Rate (%)",          "value": "Participation_Rate"},
    {"label": "Real Hourly Wage (2019 CAD)",     "value": "Real_Wage"},
    {"label": "Part-Time Share (%)",             "value": "Part_Time_Share"},
    {"label": "Nominal Wage (CAD)",              "value": "Nominal_Wage"},
    {"label": "CPI Index",                       "value": "CPI_Index"},
]
ILBL = {o["value"]: o["label"] for o in INDS}

_PL = dict(
    template="plotly_white",
    font_family="Inter, Arial, sans-serif",
    font_size=12,
    plot_bgcolor=WHITE,
    paper_bgcolor=WHITE,
    margin=dict(l=44, r=14, t=34, b=34),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0, font_size=11),
    hoverlabel=dict(bgcolor=WHITE, font_size=12),
)



def _ols(data: pd.DataFrame) -> dict:
    """
    OLS: Stress ~ 1 + Unemp + Post2020 + Unemp×Post2020 + Participation + RealWage
    Returns dict(names, beta, se, t, p, r2, n) or {} if insufficient data.
    """
    need = ["Stress_Index", "Unemployment_Rate", "Post2020",
            "Participation_Rate", "Real_Wage"]
    sub = data.dropna(subset=need)
    if len(sub) < 12:
        return {}
    X = np.column_stack([
        np.ones(len(sub)),
        sub["Unemployment_Rate"].values,
        sub["Post2020"].astype(float).values,
        (sub["Unemployment_Rate"] * sub["Post2020"].astype(float)).values,
        sub["Participation_Rate"].values,
        sub["Real_Wage"].values,
    ])
    y = sub["Stress_Index"].values
    names = ["Constant", "Unemployment (β₁)", "Post-2020 (β₂)",
             "Unemp × Post-2020 (β₃) ★", "Participation Rate", "Real Wage"]
    try:
        beta  = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        n, k  = X.shape
        s2    = (resid**2).sum() / (n - k)
        cov   = s2 * np.linalg.inv(X.T @ X)
        se    = np.sqrt(np.diag(cov))
        tv    = beta / se
        pv    = 2 * stats.t.sf(np.abs(tv), df=n - k)
        r2    = 1 - (resid**2).sum() / ((y - y.mean())**2).sum()
        return dict(names=names, beta=beta, se=se, t=tv, p=pv, r2=r2, n=n)
    except np.linalg.LinAlgError:
        return {}



def _kpi(title, value, sub="", color=BLUE):
    return dbc.Card(dbc.CardBody([
        html.P(title, className="text-muted mb-1",
               style={"fontSize": "0.74rem", "fontWeight": "600",
                      "textTransform": "uppercase", "letterSpacing": "0.04em"}),
        html.H4(value, style={"color": color, "fontWeight": "700", "margin": 0}),
        html.P(sub, className="text-muted mt-1 mb-0",
               style={"fontSize": "0.71rem"}) if sub else None,
    ]), className="shadow-sm border-0 h-100",
        style={"borderLeft": f"5px solid {color}", "borderRadius": "8px"})


def _sec(title, sub=""):
    children = [html.H6(title, style={"color": NAVY, "fontWeight": "700", "margin": 0})]
    if sub:
        children.append(html.P(sub, className="text-muted mb-0",
                               style={"fontSize": "0.75rem"}))
    return html.Div(children,
                    style={"borderBottom": f"2px solid {NAVY}",
                           "paddingBottom": "5px", "marginBottom": "12px"})


def _box(*children):
    return html.Div(list(children),
                    style={"background": WHITE, "borderRadius": "8px",
                           "padding": "16px",
                           "boxShadow": "0 1px 4px rgba(0,0,0,0.08)"})



# 4.  APP LAYOUT
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="Youth Labour Market Stress — Canada",
)

app.layout = dbc.Container(fluid=True,
    style={"background": PALE, "minHeight": "100vh",
           "fontFamily": "Inter, Arial, sans-serif", "padding": 0},
children=[

    # Header
    dbc.Row(dbc.Col(html.Div([
        html.H3("Youth Labour Market Stress Dashboard",
                style={"color": WHITE, "fontWeight": "700", "margin": "0 0 3px 0"}),
        html.P(
            "Canada 2015–2024  ·  Age group: 20–29 years (exact)  ·  10 provinces  ·  "
            "Data: StatCan 14-10-0287-01 · 14-10-0064-01 · 14-10-0020-01 · 18-10-0004-01",
            style={"color": "rgba(255,255,255,0.72)", "fontSize": "0.77rem", "margin": 0}),
    ], style={"background": NAVY, "padding": "15px 22px",
              "borderRadius": "0 0 8px 8px"})), className="mb-3 px-3"),

    # Controls
    dbc.Row([
        dbc.Col([
            html.Label("Provinces", style={"fontWeight": "600", "fontSize": "0.80rem"}),
            dcc.Dropdown(id="dd-prov",
                options=[{"label": p, "value": p} for p in ALL_PROVS],
                value=DEF_PROVS, multi=True, clearable=False,
                style={"fontSize": "0.80rem"}),
        ], md=5),
        dbc.Col([
            html.Label("Indicator", style={"fontWeight": "600", "fontSize": "0.80rem"}),
            dcc.Dropdown(id="dd-ind", options=INDS,
                value="Stress_Index", clearable=False,
                style={"fontSize": "0.80rem"}),
        ], md=3),
        dbc.Col([
            html.Label("Year Range", style={"fontWeight": "600", "fontSize": "0.80rem"}),
            dcc.RangeSlider(id="sl-yr", min=2015, max=2024, step=1,
                value=[2015, 2024],
                marks={y: {"label": str(y),
                           "style": {"fontSize": "0.68rem", "color": GRAY}}
                       for y in range(2015, 2025)},
                tooltip={"placement": "bottom", "always_visible": False}),
        ], md=4),
    ], className="mb-3 px-3 py-3 g-2",
       style={"background": WHITE, "borderRadius": "8px",
              "boxShadow": "0 1px 4px rgba(0,0,0,0.07)", "margin": "0 12px"}),

    # KPI row
    dbc.Row(id="kpi-row", className="mb-3 px-3 g-2"),


# Row 1 — Trend | Pre/Post scatter
    dbc.Row([
        dbc.Col(_box(
            _sec("Trend Over Time", "Selected indicator by province"),
            dcc.Graph(id="g-ts", style={"height": "330px"}),
        ), md=8),
        dbc.Col(_box(
            _sec("Pre vs Post-2020",
                 "Unemployment Rate → Stress Index; OLS trend lines per period"),
            dcc.Graph(id="g-sc", style={"height": "330px"}),
        ), md=4),
    ], className="mb-3 px-3 g-2"),


# Row 2 — Heatmap | Cluster
    dbc.Row([
        dbc.Col(_box(
            _sec("Stress Index Heatmap",
                 "All 10 provinces × selected years — red = high stress"),
            dcc.Graph(id="g-hm", style={"height": "310px"}),
        ), md=7),
        dbc.Col(_box(
            _sec("KMeans Cluster Map  (k = 3)",
                 "Axes: Unemployment / Real Wage; bubble size = PT share"),
            dcc.Graph(id="g-cl", style={"height": "310px"}),
        ), md=5),
    ], className="mb-3 px-3 g-2"),


# Row 3 — Variance decomp | Coefficient plot
    dbc.Row([
        dbc.Col(_box(
            _sec("Variance Decomposition",
                 "Share of partial r² per predictor — pre vs post-2020"),
            dcc.Graph(id="g-vd", style={"height": "290px"}),
        ), md=5),
        dbc.Col(_box(
            _sec("OLS Coefficient Plot",
                 "Unemployment Rate → Stress Index  ± 95 % CI"),
            dcc.Graph(id="g-co", style={"height": "290px"}),
        ), md=7),
    ], className="mb-3 px-3 g-2"),

# Row 4 — Regression table | Province table
    dbc.Row([
        dbc.Col(_box(
            _sec("Interaction Regression", 
                 "Stress ~ Unemp + Post2020 + Unemp×Post2020 + Participation + RealWage"),
            html.Div(id="tbl-reg"),
        ), md=12, className="mb-3"),
        
        dbc.Col(_box(
            _sec("Post-2020 Provincial Summary", 
                 "Averages 2020–2024 — click a column header to sort"),
            html.Div(id="tbl-pv"),
        ), md=12),
    ], className="mb-4 px-3 g-2"),

# Footer
    dbc.Row(dbc.Col(html.P(
        "Age: Tables 14-10-0287-01 and 14-10-0064-01 publish '20 to 24 years' and "
        "'25 to 29 years' series, averaged to form the 20–29 cohort. "
        "Table 14-10-0020-01 uses a 40/60 weighted blend of '15 to 24' and '25 to 44' "
        "for part-time share only. "
        "Real Wage = Nominal × (CPI₂₀₁₉_prov / CPI_t_prov). "
        "Stress Index = mean(+z_Unemployment − z_Participation − z_Real_Wage).",
        className="text-muted text-center",
        style={"fontSize": "0.70rem", "padding": "6px 0 14px"},
    ), className="px-3")),

])



@app.callback(
    [Output("kpi-row", "children"),
     Output("g-ts",    "figure"),
     Output("g-sc",    "figure"),
     Output("g-hm",    "figure"),
     Output("g-cl",    "figure"),
     Output("g-vd",    "figure"),
     Output("g-co",    "figure"),
     Output("tbl-reg", "children"),
     Output("tbl-pv",  "children")],
    [Input("dd-prov",  "value"),
     Input("sl-yr",    "value"),
     Input("dd-ind",   "value")],
)
def update(sel_provs, yr_range, indicator):

    if not sel_provs:
        sel_provs = ALL_PROVS
    y0, y1 = yr_range

    filt = df[df["Province"].isin(sel_provs) & df["Year"].between(y0, y1)].copy()
    pre  = filt[filt["Post2020"] == 0]
    post = filt[filt["Post2020"] == 1]

    def sm(s):
        v = s.mean()
        return float(v) if pd.notna(v) else 0.0


# ── KPIs ─────────────────────────────────────────────────────────────────
    pre_s  = sm(pre["Stress_Index"])
    post_s = sm(post["Stress_Index"])
    delta  = ((post_s - pre_s) / abs(pre_s) * 100) if pre_s else 0.0

    kpi = dbc.Row([
        dbc.Col(_kpi("Avg Stress", f"{sm(filt['Stress_Index']):+.2f}",
                     f"{filt['Province'].nunique()} prov · {filt['Year'].nunique()} yrs",
                     NAVY), md=2),
        dbc.Col(_kpi("Stress Δ (post-2020)", f"{delta:+.1f}%",
                     f"{pre_s:.2f}  →  {post_s:.2f}", RED), md=2),
        dbc.Col(_kpi("Avg Unemployment",
                     f"{sm(filt['Unemployment_Rate']):.1f}%",
                     f"Pre {sm(pre['Unemployment_Rate']):.1f}%  "
                     f"Post {sm(post['Unemployment_Rate']):.1f}%", BLUE), md=2),
        dbc.Col(_kpi("Avg Participation",
                     f"{sm(filt['Participation_Rate']):.1f}%", "", GREEN), md=2),
        dbc.Col(_kpi("Avg Real Wage",
                     f"${sm(filt['Real_Wage']):.2f}", "2019 CAD", GOLD), md=2),
        dbc.Col(_kpi("Obs",
                     str(len(filt)),
                     f"{filt['Province'].nunique()} × {filt['Year'].nunique()}",
                     GRAY), md=2),
    ], className="g-2")


# ── 1. Time series ────────────────────────────────────────────────────────
    fig_ts = px.line(
        filt.sort_values("Year"), x="Year", y=indicator, color="Province",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Set1,
        labels={"Year": "Year", indicator: ILBL.get(indicator, indicator)},
    )
    fig_ts.add_vline(x=2020, line_dash="dot", line_color=RED, line_width=1.5,
                     annotation_text="2020", annotation_font_size=10,
                     annotation_font_color=RED)
    fig_ts.update_layout(**_PL)
    fig_ts.update_xaxes(tickformat="d", dtick=1)


# ── 2. Pre/Post scatter ───────────────────────────────────────────────────
    sc = filt.dropna(subset=["Unemployment_Rate", "Stress_Index"]).copy()
    sc["Period"] = sc["Post2020"].map({0: "Pre-2020", 1: "Post-2020"})
    fig_sc = px.scatter(
        sc, x="Unemployment_Rate", y="Stress_Index",
        color="Period", trendline="ols",
        color_discrete_map={"Pre-2020": BLUE, "Post-2020": RED},
        hover_data=["Province", "Year"],
        labels={"Unemployment_Rate": "Unemployment Rate (%)",
                "Stress_Index": "Stress Index"},
    )
    for period, color in (("Pre-2020", BLUE), ("Post-2020", RED)):
        s2 = sc[sc["Period"] == period].dropna(
            subset=["Unemployment_Rate", "Stress_Index"])
        if len(s2) > 3:
            r, pv = stats.pearsonr(s2["Unemployment_Rate"], s2["Stress_Index"])
            stars = "***" if pv < 0.01 else "**" if pv < 0.05 else "*"
            y_a = 0.96 if period == "Pre-2020" else 0.85
            fig_sc.add_annotation(
                text=f"{period}  r = {r:.3f}{stars}",
                xref="paper", yref="paper", x=0.02, y=y_a,
                showarrow=False, font_size=10, font_color=color)
    fig_sc.update_layout(**_PL)


# ── 3. Heatmap ─────────────────────────────────────────────────────────────
    hm = (df[df["Year"].between(y0, y1)]
        .pivot(index="Province_Abbr", columns="Year", values="Stress_Index"))
    hm.columns = [str(int(c)) for c in hm.columns]

    fig_hm = px.imshow(
        hm, color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
        aspect="auto", text_auto=".2f",
        labels={"color": "Stress", "x": "Year", "y": ""},
        template="plotly_white"
    )

    fig_hm.update_xaxes(
        type='category', 
        tickangle=-45,   
        side="bottom",
        dtick=1         
    )

    fig_hm.update_layout(**_PL)
    fig_hm.update_layout(
        margin=dict(l=40, r=20, t=30, b=80),
        coloraxis_colorbar=dict(title="Stress", len=0.7)
    )

    fig_hm.update_xaxes(type='category', tickangle=-45)


# ── 4. Cluster scatter ────────────────────────────────────────────────────
    cl = filt.dropna(subset=["Unemployment_Rate", "Real_Wage"])
    fig_cl = px.scatter(
        cl, x="Unemployment_Rate", y="Real_Wage",
        color="Cluster", size="Part_Time_Share", size_max=20,
        color_discrete_map=CMAP,
        hover_data=["Province", "Year", "Stress_Index"],
        labels={"Unemployment_Rate": "Unemployment (%)",
                "Real_Wage": "Real Wage (2019 CAD)"},
    )
    fig_cl.update_layout(**_PL)


# ── 5. Variance decomposition ─────────────────────────────────────────────
    vd_rows = []
    for lbl, mask in [("Pre-2020\n(2015–2019)", filt["Post2020"] == 0),
                       ("Post-2020\n(2020–2024)", filt["Post2020"] == 1)]:
        s2 = filt[mask].dropna(
            subset=["Stress_Index", "Unemployment_Rate",
                    "Participation_Rate", "Real_Wage"])
        if len(s2) < 5:
            continue
        r2u = stats.pearsonr(s2["Unemployment_Rate"],  s2["Stress_Index"])[0]**2
        r2p = stats.pearsonr(s2["Participation_Rate"], s2["Stress_Index"])[0]**2
        r2w = stats.pearsonr(s2["Real_Wage"],          s2["Stress_Index"])[0]**2
        tot = r2u + r2p + r2w or 1
        vd_rows.append({"Period": lbl,
                         "Unemployment":  r2u / tot,
                         "Participation": r2p / tot,
                         "Real Wage":     r2w / tot})
    if vd_rows:
        vd = pd.DataFrame(vd_rows).melt(
            id_vars="Period", var_name="Component", value_name="Share")
        fig_vd = px.bar(
            vd, x="Period", y="Share", color="Component",
            color_discrete_map={"Unemployment": BLUE,
                                 "Participation": GOLD, "Real Wage": GREEN},
            text=vd["Share"].apply(lambda v: f"{v:.0%}"),
            labels={"Share": "Share of partial r²", "Period": ""},
            barmode="stack",
        )
        fig_vd.update_traces(textposition="inside", textfont_size=10)
        fig_vd.update_yaxes(tickformat=".0%", range=[0, 1])
        fig_vd.update_layout(**_PL)
    else:
        fig_vd = go.Figure().update_layout(**_PL)


# ── 6. Coefficient plot ───────────────────────────────────────────────────
    fig_co = go.Figure()
    for lbl, data, color in [
        ("Full period", filt,               NAVY),
        ("Pre-2020",    pre,                 BLUE),
        ("Post-2020",   post,                RED),
    ]:
        res = _ols(data)
        if not res:
            continue
        idx = next((i for i, n in enumerate(res["names"]) if "β₁" in n), None)
        if idx is None:
            continue
        b, se = res["beta"][idx], res["se"][idx]
        lo, hi = b - 1.96 * se, b + 1.96 * se
        fig_co.add_trace(go.Scatter(
            x=[lo, hi], y=[lbl, lbl], mode="lines",
            line=dict(color=color, width=3), showlegend=False))
        fig_co.add_trace(go.Scatter(
            x=[b], y=[lbl], mode="markers+text",
            marker=dict(color=color, size=14, symbol="diamond"),
            text=[f"  {b:.3f}"],
            textposition="middle right",
            textfont=dict(color=color, size=11),
            showlegend=False))
    fig_co.add_vline(x=0, line_dash="dash", line_color=GRAY, line_width=1)
    fig_co.update_layout(**_PL,
                          xaxis_title="Coefficient: Unemployment → Stress",
                          yaxis_title="")


# ── 7. Regression table ───────────────────────────────────────────────────
    res = _ols(filt)
    if res:
        rows = []
        for nm, b, se, t, pv in zip(
                res["names"], res["beta"], res["se"], res["t"], res["p"]):
            stars = ("***" if pv < 0.01 else "**" if pv < 0.05
                     else "*" if pv < 0.10 else "")
            rows.append({"Variable": nm, "Coef.": f"{b:.3f}{stars}",
                          "Std. Err.": f"({se:.3f})",
                          "t-stat": f"{t:.2f}", "p-value": f"{pv:.3f}"})
        rows.append({"Variable": "R²  |  N", "Coef.": f"{res['r2']:.3f}",
                      "Std. Err.": "", "t-stat": "", "p-value": str(res["n"])})
        tbl_reg = dash_table.DataTable(
            data=rows,
            columns=[{"name": c, "id": c} for c in rows[0]],
            style_cell={"fontSize": "0.76rem", "padding": "5px 7px",
                        "fontFamily": "Inter, Arial", "border": "1px solid #eee"},
            style_header={"backgroundColor": NAVY, "color": WHITE,
                           "fontWeight": "700", "fontSize": "0.76rem",
                           "border": "none"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": PALE},
                {"if": {"filter_query": '{Variable} contains "★"'},
                 "backgroundColor": "#FFF3E0", "fontWeight": "700"},
            ],
        )
    else:
        tbl_reg = html.P("Select more provinces / years for regression.",
                          className="text-muted", style={"fontSize": "0.80rem"})


# ── 8. Province summary table ─────────────────────────────────────────────
    ps = (
        df[df["Post2020"] == 1]
        .groupby("Province")
        .agg(Unemp=("Unemployment_Rate", "mean"),
             Part =("Participation_Rate", "mean"),
             Wage =("Real_Wage",          "mean"),
             PT   =("Part_Time_Share",    "mean"),
             Stress=("Stress_Index",      "mean"),
             Cluster=("Cluster", lambda x: x.mode().iloc[0]))
        .round(2).reset_index()
        .sort_values("Stress", ascending=False)
    )
    ps.columns = ["Province", "Unemp (%)", "Part. (%)",
                   "Real Wage ($)", "PT Share (%)", "Stress", "Cluster"]
    tbl_pv = dash_table.DataTable(
        data=ps.to_dict("records"),
        columns=[{"name": c, "id": c} for c in ps.columns],
        sort_action="native",
        style_cell={"fontSize": "0.75rem", "padding": "5px 7px",
                    "fontFamily": "Inter, Arial", "border": "1px solid #eee"},
        style_header={"backgroundColor": NAVY, "color": WHITE,
                       "fontWeight": "700", "fontSize": "0.75rem",
                       "border": "none"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": PALE},
            {"if": {"filter_query": '{Cluster} = "High Stress"'},
             "backgroundColor": "#FFF0F0"},
            {"if": {"filter_query": '{Cluster} = "Low Stress"'},
             "backgroundColor": "#EEF6FF"},
        ],
    )

    return (kpi, fig_ts, fig_sc, fig_hm, fig_cl,
            fig_vd, fig_co, tbl_reg, tbl_pv)


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8050))
    
    print(f"🚀 Dashboard starting on port {port}...")
    serve(app.server, host="0.0.0.0", port=port)