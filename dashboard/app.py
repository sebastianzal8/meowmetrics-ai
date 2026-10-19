import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from pathlib import Path

# Allow importing the ai package (shared mock data + copilot) when running
# `streamlit run app.py` from inside dashboard/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Set page config with a cat theme
st.set_page_config(
    page_title="MeowMetrics - Cat Health Analytics",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS styling for sleek dark/glassmorphic look
st.markdown("""
<style>
    .main {
        background-color: #0f111a;
        color: #e2e8f0;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.03);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #fbd38d !important;
    }
    .css-1542z7w {
        background-color: #1a1e2e !important;
    }
    .badge {
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.85em;
        font-weight: bold;
    }
    .badge-excellent { background-color: #276749; color: #c6f6d5; }
    .badge-good { background-color: #2b6cb0; color: #ebf8ff; }
    .badge-attention { background-color: #9c4221; color: #fffaf0; }
    .badge-critical { background-color: #9b2c2c; color: #fff5f5; }
</style>
""", unsafe_allow_html=True)

# DBTITLE 1,Database Connection & Data Loading
# The mock Gold-layer generator was moved to `ai/data_access/mock_data.py` so
# the dashboard, the AI copilot and the eval harness all share one dataset.
from ai.data_access.mock_data import build_mock_gold_data

@st.cache_data
def get_mock_data():
    """Generates mock Gold-layer views for local execution."""
    return build_mock_gold_data()

def load_data():
    """Loads dataset from Snowflake if credentials exist, else returns mock data."""
    # Check for Snowflake environment setup
    sf_user = os.getenv("SF_USER")
    if sf_user:
        try:
            import snowflake.connector
            ctx = snowflake.connector.connect(
                user=os.getenv("SF_USER"),
                password=os.getenv("SF_PASSWORD"),
                account=os.getenv("SF_ACCOUNT"),
                warehouse="MEOW_WH",
                database="MEOW_DB",
                schema="ANALYTICS"
            )
            cs = ctx.cursor()
            cs.execute("SELECT * FROM BI_TABLEAU_VIEWS")
            columns = [col[0].lower() for col in cs.description]
            data = cs.fetchall()
            df = pd.DataFrame(data, columns=columns)
            cs.close()
            ctx.close()
            return df
        except Exception as e:
            st.sidebar.warning(f"Failed to connect to Snowflake: {str(e)}. Falling back to mock data.")
            return get_mock_data()
    else:
        return get_mock_data()

# Load project dataset
df = load_data()

# DBTITLE 2,Sidebar Navigation & Selectors
_LOGO = str(Path(__file__).resolve().parent / "D720F0F5-F61E-4E86-8191-6EED544CC472.jpg")
st.sidebar.image(_LOGO, use_container_width=True)
st.sidebar.title("🐱 MeowMetrics")
st.sidebar.caption("Cat Telemetry & Pipeline Analytics Dashboard")

# Cat filter selector
cat_names = sorted(df["cat_name"].unique())
selected_cat = st.sidebar.selectbox("Select Cat Profile", ["All Cats"] + cat_names)

# Date filter selector
min_date = df["reading_date"].min()
max_date = df["reading_date"].max()
date_range = st.sidebar.slider("Select Date Range", min_date, max_date, (min_date, max_date))

# Filter data
filtered_df = df[(df["reading_date"] >= date_range[0]) & (df["reading_date"] <= date_range[1])]
if selected_cat != "All Cats":
    filtered_df = filtered_df[filtered_df["cat_name"] == selected_cat]

# Connection status message
if os.getenv("SF_USER"):
    st.sidebar.success("Connected to Snowflake Warehouse ✅")
else:
    st.sidebar.info("Running Mock Mode (Snowflake offline) ℹ️")

# DBTITLE 3,Header Analytics Metrics
st.title("🐱 MeowMetrics Analytics Portal")
st.markdown("Exploring Gold-Layer Data Warehouse Metrics from Smart-Collar Telemetry")

if not filtered_df.empty:
    # Aggregated metrics for display
    avg_steps = int(filtered_df["total_daily_steps"].mean())
    avg_sleep = int(filtered_df["estimated_sleep_minutes"].mean())
    avg_purr = round(filtered_df["avg_purr_frequency_hz"].mean(), 1)
    avg_health = int(filtered_df["daily_feline_health_score"].mean())
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🐾 Average Daily Steps", f"{avg_steps:,}", delta=f"{int(avg_steps - 3000)} vs Baseline")
    with col2:
        st.metric("💤 Avg Sleep Duration", f"{avg_sleep // 60}h {avg_sleep % 60}m", delta=f"{int(avg_sleep - 840)} mins vs Baseline")
    with col3:
        st.metric("🔊 Average Purr Frequency", f"{avg_purr} Hz", delta="Stable")
    with col4:
        # Determine status delta direction
        st.metric("❤️ Feline Health Index", f"{avg_health}/100", delta=f"{avg_health - 75}% vs Safety Target")
        
    st.markdown("---")

    # DBTITLE 4,Data Visualizations Layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📈 Health & Step Trends Over Time")
        
        # Step Trends Chart (Plotly)
        fig_steps = px.line(
            filtered_df,
            x="reading_date",
            y="total_daily_steps",
            color="cat_name" if selected_cat == "All Cats" else None,
            title="Total Daily Steps",
            labels={"reading_date": "Date", "total_daily_steps": "Steps"},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_steps.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            title_font_color='#fbd38d'
        )
        st.plotly_chart(fig_steps, use_container_width=True)
        
        # Activity Distribution Chart
        avg_act = filtered_df[["estimated_sleep_minutes", "estimated_active_minutes", "estimated_lazy_minutes"]].mean()
        fig_activity = px.pie(
            names=["Sleep", "Active Play", "Lazy Lounging"],
            values=avg_act.values,
            title="Est. Daily Activity Distribution (Minutes)",
            hole=0.4,
            color_discrete_sequence=["#2b6cb0", "#fbd38d", "#4a5568"]
        )
        fig_activity.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            title_font_color='#fbd38d'
        )
        st.plotly_chart(fig_activity, use_container_width=True)

    with col_right:
        st.subheader("🐱 Cat Profiles & Status")
        
        # Cat Registry Cards
        if selected_cat == "All Cats":
            # Display summary list of registered cats
            unique_cats = filtered_df.drop_duplicates(subset=["cat_id"])
            for idx, row in unique_cats.iterrows():
                # Color code status
                latest_score = filtered_df[filtered_df["cat_id"] == row["cat_id"]].sort_values("reading_date").iloc[-1]
                score_val = latest_score["daily_feline_health_score"]
                status_cat = latest_score["health_status_category"]
                
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.03); padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid #fbd38d;">
                    <h4>{row['cat_name']} ({row['breed']})</h4>
                    <p style="margin: 2px 0;"><b>Age:</b> {row['age_years']} years (Human eq. {row['human_equivalent_age']})</p>
                    <p style="margin: 2px 0;"><b>Weight Category:</b> {row['weight_category']} ({row['weight_lbs']} lbs)</p>
                    <p style="margin: 2px 0;"><b>Latest Health Score:</b> <span class="badge badge-{status_cat.lower().replace(' ', '')}">{score_val}/100 ({status_cat})</span></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Display detailed card for specific cat
            cat_row = filtered_df.iloc[0]
            latest_status = filtered_df.sort_values("reading_date").iloc[-1]
            score_val = latest_status["daily_feline_health_score"]
            status_cat = latest_status["health_status_category"]
            
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.04); padding: 25px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.1);">
                <h3 style="margin-top:0;">{cat_row['cat_name']}</h3>
                <hr style="border-color: rgba(255,255,255,0.1);"/>
                <p><b>Breed:</b> {cat_row['breed']}</p>
                <p><b>Gender:</b> {cat_row['gender'].capitalize()}</p>
                <p><b>Age:</b> {cat_row['age_years']} years old</p>
                <p><b>Human Equivalent Age:</b> {cat_row['human_equivalent_age']} years</p>
                <p><b>Weight:</b> {cat_row['weight_lbs']} lbs ({cat_row['weight_category']})</p>
                <p><b>Latest Health Score:</b></p>
                <div style="text-align: center; padding: 15px 0;">
                    <span style="font-size: 3em; font-weight: bold; color: #fbd38d;">{score_val}</span><span style="font-size: 1.5em; color: #a0aec0;">/100</span>
                    <br/><br/>
                    <span class="badge badge-{status_cat.lower().replace(' ', '')}" style="font-size: 1.1em; padding: 6px 15px;">{status_cat}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Single Heart Rate Gauges
            latest_hr = latest_status["avg_heart_rate_bpm"]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=latest_hr,
                title={'text': "Avg Heart Rate (BPM)"},
                gauge={
                    'axis': {'range': [40, 240]},
                    'bar': {'color': "#fbd38d"},
                    'steps': [
                        {'range': [40, 100], 'color': "#9b2c2c"},
                        {'range': [100, 180], 'color': "#276749"},
                        {'range': [180, 240], 'color': "#9b2c2c"}
                    ]
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0',
                height=250
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

    # DBTITLE 5,Detailed Data View (Raw Table)
    st.subheader("📋 Pipeline Gold Mart Data Table")
    with st.expander("Expand to view raw denormalized BI dataset"):
        st.dataframe(
            filtered_df[[
                "cat_name", "breed", "reading_date", "avg_heart_rate_bpm", 
                "total_daily_steps", "avg_purr_frequency_hz", 
                "estimated_sleep_minutes", "daily_feline_health_score", 
                "health_status_category"
            ]].sort_values("reading_date", ascending=False),
            use_container_width=True
        )
else:
    st.error("No cat behavior records found matching the filters.")

# DBTITLE 6,🆕 AI Layer: Agentic RAG Copilot + LLM Daily Digests
st.markdown("---")
from ai.ui.copilot import render_copilot
from ai.ui.digest import render_digest_section

render_copilot()
st.markdown("---")
render_digest_section()

