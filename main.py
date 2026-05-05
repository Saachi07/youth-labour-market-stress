import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# --- 1. DATA LOADING ---
@st.cache_data
def load_data():
    # Load the verbatim research data file[cite: 3, 4]
    df = pd.read_csv("research_data_monthly.csv")
    if "Post2020" in df.columns and "COVID" not in df.columns:
        df.rename(columns={"Post2020": "COVID"}, inplace=True)
    
    # Restoring KMeans Clustering logic for provincial stress tiers
    feats = [c for c in ["Unemployment_Rate","Participation_Rate","Real_Wage","Part_Time_Share"] if c in df.columns]
    X = StandardScaler().fit_transform(df[feats].fillna(df[feats].mean()))
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["_k"] = km.fit_predict(X)
    order = df.groupby("_k")["Stress_Index"].mean().sort_values().index.tolist()
    mapping = {order[0]: "Low Stress", order[1]: "Medium Stress", order[2]: "High Stress"}
    df["Cluster"] = df["_k"].map(mapping)
    return df

df = load_data()

# --- 2. LAYOUT & SIDEBAR ---
st.set_page_config(layout="wide", page_title="Youth Labour Stress")
st.title("Youth Labour Market Stress Dashboard")

with st.sidebar:
    st.header("Controls")
    all_provs = sorted(df["Province"].unique())
    sel_provs = st.multiselect("Provinces", all_provs, default=["Ontario", "Alberta", "Quebec"])
    sel_age = st.selectbox("Age Group", ["All"] + sorted(df["Age_Group"].unique()))
    indicator = st.selectbox("Indicator", ["Stress_Index", "Unemployment_Rate", "Participation_Rate", "Real_Wage"])
    yr_range = st.slider("Years", 2015, 2024, (2015, 2024))

# Filter Data based on UI input
filt = df[df["Province"].isin(sel_provs) & df["Year"].between(yr_range[0], yr_range[1])]
if sel_age != "All":
    filt = filt[filt["Age_Group"] == sel_age]

# --- 3. TRENDS & SCATTERS (Row 1) ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Trend Over Time")
    chart_data = filt.copy()
    
    # 1. Standardize the Time Axis
    if 'Month' in chart_data.columns:
        # Create a datetime object first for proper chronological sorting
        chart_data['Date'] = pd.to_datetime(
            chart_data['Year'].astype(str) + '-' + chart_data['Month'].astype(str) + '-01'
        )
        chart_data = chart_data.sort_values('Date')
        
        # Convert the entire column to STRINGS to satisfy Plotly's sum() logic
        chart_data['x_axis_display'] = chart_data['Date'].dt.strftime('%Y-%m-%d').astype(str)
        vline_pos = "2020-03-01"
    else:
        # If only Year is available, ensure it is treated as a string too
        chart_data = chart_data.sort_values('Year')
        chart_data['x_axis_display'] = chart_data['Year'].astype(str)
        vline_pos = "2020"

    # 2. Plot using the string-only column[cite: 4]
    fig_ts = px.line(
        chart_data, 
        x='x_axis_display', 
        y=indicator, 
        color="Province", 
        template="plotly_white"
    )
    
    # 3. Add the vertical line with a matching string type[cite: 4]
    fig_ts.add_vline(
        x=vline_pos, 
        line_dash="dot", 
        line_color="red", 
        annotation_text="COVID Start"
    )
    
    st.plotly_chart(fig_ts, use_container_width=True)

with col2:
    st.subheader("Pre vs Post-COVID Scatter")
    fig_sc = px.scatter(filt, x="Unemployment_Rate", y="Stress_Index", color="COVID", 
                        color_discrete_map={0: "#2F75B6", 1: "#C00000"}, trendline="ols")
    st.plotly_chart(fig_sc, use_container_width=True)

# --- 4. HEATMAP & AGE COMPARISON (Row 2) ---
col3, col4 = st.columns([1, 1])

with col3:
    st.subheader("Stress Index Heatmap")
    hm_data = df.pivot_table(index="Province", columns="Year", values="Stress_Index", aggfunc="mean")
    fig_hm = px.imshow(hm_data, color_continuous_scale="RdBu_r", aspect="auto")
    st.plotly_chart(fig_hm, use_container_width=True)

with col4:
    st.subheader("Age Group Comparison: 20–24 vs 25–29")
    age_coeffs = []
    for ag in ["20-24", "25-29"]:
        sub = df[df["Age_Group"] == ag]
        if not sub.empty:
            m = smf.ols("Stress_Index ~ COVID * Unemployment_Rate", data=sub).fit()
            b = m.params.get("COVID:Unemployment_Rate", 0)
            se = m.bse.get("COVID:Unemployment_Rate", 0)
            age_coeffs.append({"Age": ag, "Beta": b, "Lo": b-1.96*se, "Hi": b+1.96*se})
    
    if age_coeffs:
        ac_df = pd.DataFrame(age_coeffs)
        fig_age = px.scatter(ac_df, x="Beta", y="Age", error_x="Hi", error_x_minus="Lo", 
                             title="COVID × Unemp Coefficient ± 95% CI")
        st.plotly_chart(fig_age, use_container_width=True)

# --- 5. VARIANCE DECOMPOSITION ---
st.subheader("Variance Decomposition (Unemployment vs Participation)")
vd_list = []
for lbl, mask in [("Pre-COVID", filt["COVID"]==0), ("Post-COVID", filt["COVID"]==1)]:
    s2 = filt[mask].dropna(subset=["Stress_Index", "Unemployment_Rate", "Participation_Rate"])
    if not s2.empty and len(s2) > 1:
        r2u = stats.pearsonr(s2["Unemployment_Rate"], s2["Stress_Index"])[0]**2
        r2p = stats.pearsonr(s2["Participation_Rate"], s2["Stress_Index"])[0]**2
        tot = r2u + r2p
        if tot > 0:
            vd_list.append({"Period": lbl, "Unemployment": r2u/tot, "Participation": r2p/tot})

if vd_list:
    vd_df = pd.DataFrame(vd_list).melt(id_vars="Period")
    fig_vd = px.bar(vd_df, x="Period", y="value", color="variable", barmode="stack")
    st.plotly_chart(fig_vd, use_container_width=True)

# --- 6. REGRESSION TABLES (Fixed Rendering) ---
st.subheader("Month-FE Regression Details")
try:
    full_model = smf.ols("Stress_Index ~ COVID * Unemployment_Rate + Participation_Rate + Real_Wage + C(Month) + C(Province)", data=filt).fit()
    
    # Convert the statsmodels summary table to HTML then to a DataFrame[cite: 4]
    results_df = pd.read_html(full_model.summary().tables[1].as_html(), header=0, index_col=0)[0]
    st.dataframe(results_df) # Renders a clean table instead of object pointers[cite: 4]
except Exception as e:
    st.warning("Insufficient data for fixed-effects regression. Try selecting more provinces.")

st.subheader("Post-COVID Provincial Summary")
summary = filt[filt["COVID"] == 1].groupby("Province").agg({"Stress_Index": "mean", "Cluster": lambda x: x.mode()[0]})
st.table(summary.sort_values("Stress_Index", ascending=False))