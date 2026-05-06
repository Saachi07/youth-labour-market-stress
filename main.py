import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import io

warnings.filterwarnings("ignore")

# --- 1. CONFIG & DATA LOADING ---
st.set_page_config(layout="wide", page_title="Youth Labour Stress")

@st.cache_data
def load_data():
    # Load the verbatim research data file
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
with st.sidebar:
    st.header("Controls")
    all_provs = sorted(df["Province"].unique())
    sel_provs = st.multiselect("Provinces", all_provs, default=["Ontario", "Alberta", "Quebec"])
    sel_age = st.selectbox("Age Group", ["All"] + sorted(df["Age_Group"].unique()))
    indicator = st.selectbox("Indicator", ["Stress_Index", "Unemployment_Rate", "Participation_Rate", "Real_Wage"])
    yr_range = st.slider("Years", 2015, 2024, (2015, 2024))

# Filter Data based on UI input
filt = df[df["Province"].isin(sel_provs) & df["Year"].between(yr_range[0], yr_range[1])].copy()
if sel_age != "All":
    filt = filt[filt["Age_Group"] == sel_age]

st.title("Youth Labour Market Stress Dashboard")

# --- 3. TAB NAVIGATION ---
# Replace columns with tabs
eda_tab, ml_tab = st.tabs(["📊 Exploratory Data Analysis", "🧠 Machine Learning Insights"])

# --- TAB 1: MAIN DASHBOARD (EDA & REGRESSION) ---
with eda_tab:
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Trend Over Time")
        chart_data = filt.copy()
        
        # Use native datetimes and integers instead of strings so Plotly can calculate annotation coordinates
        if 'Month' in chart_data.columns:
            chart_data['Date'] = pd.to_datetime(chart_data['Year'].astype(str) + '-' + chart_data['Month'].astype(str) + '-01')
            chart_data = chart_data.sort_values('Date')
            x_col = 'Date'
            vline_pos = "2020-03-01"
        else:
            chart_data = chart_data.sort_values('Year')
            x_col = 'Year'
            vline_pos = 2020

        fig_ts = px.line(chart_data, x=x_col, y=indicator, color="Province", template="plotly_white")
        
        # Draw the line WITHOUT the buggy annotation_text parameter
        fig_ts.add_vline(x=vline_pos, line_dash="dot", line_color="red")
        
        # Add the text manually using add_annotation
        fig_ts.add_annotation(
            x=vline_pos, 
            y=0.98,          # Height placement (1.0 is the top of the chart)
            yref="paper",    # Aligns 'y' to the chart bounding box rather than data coordinates
            text="COVID Start", 
            showarrow=False,
            font=dict(color="red"),
            xanchor="left",
            xshift=5         # Shifts the text slightly right so it doesn't overlap the line
        )
        
        st.plotly_chart(fig_ts, use_container_width=True)

    with c2:
        st.subheader("Pre vs Post-COVID Scatter")
        fig_sc = px.scatter(filt, x="Unemployment_Rate", y="Stress_Index", color="COVID", 
                            color_discrete_map={0: "#2F75B6", 1: "#C00000"}, trendline="ols")
        st.plotly_chart(fig_sc, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Stress Index Heatmap")
        hm_data = filt.pivot_table(index="Province", columns="Year", values="Stress_Index", aggfunc="mean")
        fig_hm = px.imshow(hm_data, color_continuous_scale="RdBu_r", aspect="auto")
        st.plotly_chart(fig_hm, use_container_width=True)

    with c4:
        st.subheader("Age Group Comparison")
        age_coeffs = []
        for ag in ["20-24", "25-29"]:
            sub = df[df["Age_Group"] == ag] # Use full DF to ensure sufficient degrees of freedom
            if not sub.empty:
                try:
                    m = smf.ols("Stress_Index ~ COVID * Unemployment_Rate", data=sub).fit()
                    b = m.params.get("COVID:Unemployment_Rate", 0)
                    se = m.bse.get("COVID:Unemployment_Rate", 0)
                    age_coeffs.append({"Age": ag, "Beta": b, "Lo": b-1.96*se, "Hi": b+1.96*se})
                except: pass
        if age_coeffs:
            ac_df = pd.DataFrame(age_coeffs)
            fig_age = px.scatter(ac_df, x="Beta", y="Age", error_x="Hi", error_x_minus="Lo", 
                                 title="COVID × Unemp Coefficient ± 95% CI")
            st.plotly_chart(fig_age, use_container_width=True)

    st.subheader("Regression Details")
    try:
        # Removed C(Month) and C(Province) to prevent singular matrix errors on filtered subsets
        full_model = smf.ols("Stress_Index ~ COVID * Unemployment_Rate + Participation_Rate + Real_Wage", data=filt).fit()
        
        # Wrap HTML in io.StringIO for Pandas 2.0+ compatibility
        html_string = full_model.summary().tables[1].as_html()
        results_df = pd.read_html(io.StringIO(html_string), header=0, index_col=0)[0]
        st.dataframe(results_df, use_container_width=True)
    except Exception as e:
        st.warning("Insufficient data for regression. Try selecting more provinces or years.")

# --- TAB 2: MACHINE LEARNING INSIGHTS ---
with ml_tab:
    
    # We can use columns inside the tab to format the ML section nicely
    ml_col1, ml_col2 = st.columns(2)
    
    with ml_col1:
        # 1. Anomaly Detection
        st.subheader("1. Economic Anomalies")
        st.caption("Using Isolation Forest to detect economic shocks")
        anomaly_features = ['Unemployment_Rate', 'CPI_Index', 'Stress_Index', 'Real_Wage']
        ml_df = filt.dropna(subset=anomaly_features).copy()
        
        if not ml_df.empty and len(ml_df) > 10:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(ml_df[anomaly_features])
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            ml_df['Is_Anomaly'] = iso_forest.fit_predict(X_scaled) == -1
            
            fig_anom = px.scatter(ml_df, x='YearMonth', y='Unemployment_Rate', color='Is_Anomaly',
                                  color_discrete_map={False: '#A0C4FF', True: '#D90429'},
                                  labels={"Is_Anomaly": "Anomaly Flag"})
            fig_anom.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_anom, use_container_width=True)
        else:
            st.info("Not enough data to run anomaly detection.")

        # 3. Random Forest - Wage Gap
        st.subheader("3. Wage Gap Predictors")
        st.caption("Random Forest Feature Importances")
        rf_feats = ['Participation_Rate', 'Part_Time_Share', 'FT_Wage_Premium', 'Stress_Index', 'COVID']
        rf_df = filt.dropna(subset=rf_feats + ['Gender_Wage_Gap']).copy()
        
        if len(rf_df) > 20:
            X_rf = rf_df[rf_feats]
            y_rf = rf_df['Gender_Wage_Gap']
            model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
            model_rf.fit(X_rf, y_rf)
            
            imp_df = pd.DataFrame({'Feature': rf_feats, 'Importance': model_rf.feature_importances_})
            imp_df = imp_df.sort_values('Importance', ascending=True)
            
            fig_rf = px.bar(imp_df, x='Importance', y='Feature', orientation='h')
            fig_rf.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_rf, use_container_width=True)
        else:
            st.info("Not enough data to calculate predictors.")

    with ml_col2:
        # 2. Clustering
        st.subheader("2. Employment Profiles")
        st.caption("K-Means Clustering on FT/PT dynamics")
        cluster_features = ['FT_Employment', 'PT_Employment', 'Part_Time_Share']
        clust_df = filt.dropna(subset=cluster_features).copy()
        
        if not clust_df.empty and len(clust_df) > 10:
            X_clust = StandardScaler().fit_transform(clust_df[cluster_features])
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clust_df['Profile'] = kmeans.fit_predict(X_clust).astype(str)
            
            fig_clust = px.scatter(clust_df, x='FT_Employment', y='Part_Time_Share', color='Profile',
                                   color_discrete_sequence=px.colors.qualitative.Vivid)
            fig_clust.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_clust, use_container_width=True)
        else:
            st.info("Not enough data to identify profiles.")

        # 4. Time Series Forecasting
        st.subheader("4. Unemployment Forecast")
        st.caption("Actual vs Predicted Unemployment (Test Sample)")
        
        forecast_df = df[df["Province"].isin(sel_provs)].copy()
        if 'Month' in forecast_df.columns:
            forecast_df['Date'] = pd.to_datetime(forecast_df['Year'].astype(str) + '-' + forecast_df['Month'].astype(str) + '-01')
        else:
             forecast_df['Date'] = pd.to_datetime(forecast_df['Year'].astype(str) + '-01-01')
        forecast_df = forecast_df.sort_values('Date')
        
        # Feature Engineering Shift
        forecast_df['Lag_1_Unemployment'] = forecast_df.groupby(['Province', 'Age_Group'])['Unemployment_Rate'].shift(1)
        ts_feats = ['Lag_1_Unemployment', 'Participation_Rate', 'CPI_Index', 'Stress_Index']
        ts_clean = forecast_df.dropna(subset=ts_feats + ['Unemployment_Rate']).copy()
        
        if len(ts_clean) > 50:
            X_ts = ts_clean[ts_feats]
            y_ts = ts_clean['Unemployment_Rate']
            
            X_train, X_test, y_train, y_test = train_test_split(X_ts, y_ts, test_size=0.2, shuffle=False)
            model_ts = RandomForestRegressor(n_estimators=100, random_state=42)
            model_ts.fit(X_train, y_train)
            preds = model_ts.predict(X_test)
            
            plot_df = pd.DataFrame({'Actual': y_test.values[:50], 'Predicted': preds[:50]})
            fig_ts_pred = go.Figure()
            fig_ts_pred.add_trace(go.Scatter(y=plot_df['Actual'], mode='lines+markers', name='Actual', line=dict(color='blue', dash='solid')))
            fig_ts_pred.add_trace(go.Scatter(y=plot_df['Predicted'], mode='lines', name='Predicted', line=dict(color='orange', dash='dash')))
            fig_ts_pred.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            st.plotly_chart(fig_ts_pred, use_container_width=True)
        else:
            st.info("Not enough data to run Time Series Forecasting.")