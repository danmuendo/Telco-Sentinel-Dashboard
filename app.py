import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Tetranet | Telco-Sentinel Dashboard",
    page_icon="🔋",
    layout="wide"
)

# --- THEME & STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #1E3A8A; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- SETTINGS ---
EOL_THRESHOLD = 1.4  # Capacity (Ah) threshold for replacement
DATA_DIR = "data_sample"    # Folder where individual .csv files are stored

# --- DATA LOADING ---
@st.cache_data
def load_metadata():
    if not os.path.exists('metadata.csv'):
        st.error("Missing 'metadata.csv' in the project folder!")
        return None, None
    
    df = pd.read_csv('metadata.csv')
    df['Capacity'] = pd.to_numeric(df['Capacity'], errors='coerce')
    
    # Process health data (discharge cycles)
    health_df = df[df['type'] == 'discharge'].dropna(subset=['Capacity']).copy()
    health_df = health_df.sort_values(['battery_id', 'test_id'])
    health_df['cycle'] = health_df.groupby('battery_id').cumcount() + 1
    
    return df, health_df

def load_detailed_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# --- LOAD DATA ---
metadata_all, health_data = load_metadata()

if metadata_all is not None:
    # --- SIDEBAR ---
    st.sidebar.title("TETRANET SERVICES")
    st.sidebar.markdown("### Infrastructure Analytics")
    
    battery_list = sorted(health_data['battery_id'].unique())
    selected_id = st.sidebar.selectbox("Select Tower Site (Battery ID)", battery_list)
    
    st.sidebar.divider()
    st.sidebar.info(f"Showing live analytics for Site {selected_id}. This data helps predict power failures before they happen.")

    # --- DATA FILTERING ---
    b_history = health_data[health_data['battery_id'] == selected_id]
    current_cycle = b_history['cycle'].iloc[-1]
    latest_capacity = b_history['Capacity'].iloc[-1]
    start_capacity = b_history['Capacity'].iloc[0]
    soh = (latest_capacity / start_capacity) * 100

    # --- PREDICTION ENGINE (Linear Regression) ---
    X = b_history[['cycle']].values
    y = b_history['Capacity'].values
    model = LinearRegression().fit(X, y)
    
    # Calculate Remaining Useful Life (RUL)
    # y = mx + c  => x = (y - c) / m
    m = model.coef_[0]
    c = model.intercept_
    predicted_end_cycle = (EOL_THRESHOLD - c) / m
    remaining_cycles = int(max(0, predicted_end_cycle - current_cycle))

    # --- MAIN UI ---
    st.title(f"🔋 Asset Health: {selected_id}")
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("State of Health (SoH)", f"{soh:.1f}%", f"{soh-100:.1f}%")
    col2.metric("Current Capacity", f"{latest_capacity:.3f} Ah")
    col3.metric("Remaining Useful Life", f"{remaining_cycles} Cycles")
    col4.metric("Ambient Temp", f"{b_history['ambient_temperature'].iloc[-1]}°C")

    # Tabs
    tab1, tab2 = st.tabs(["📉 Long-term Fleet View", "🔍 Deep-Dive Sensor Logs"])

    with tab1:
        st.subheader("Capacity Degradation & Prediction")
        
        # Plotting actual vs trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=b_history['cycle'], y=b_history['Capacity'], 
                                 mode='lines+markers', name='Observed Capacity', line=dict(color='#1E3A8A')))
        
        # Forecast Line
        future_x = np.array([[1], [predicted_end_cycle + 20]])
        future_y = model.predict(future_x)
        fig.add_trace(go.Scatter(x=future_x.flatten(), y=future_y, 
                                 name='Forecast Trend', line=dict(dash='dash', color='orange')))
        
        fig.add_hline(y=EOL_THRESHOLD, line_color="red", line_dash="dot", 
                      annotation_text="Failure Threshold (1.4Ah)")
        
        fig.update_layout(xaxis_title="Usage Cycles", yaxis_title="Capacity (Ah)", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Alerts
        if remaining_cycles < 15:
            st.error(f"⚠️ **URGENT:** Site {selected_id} battery is nearing End-of-Life. Replacement recommended within {remaining_cycles} cycles.")
        else:
            st.success(f"✅ Site {selected_id} is healthy. Estimated maintenance needed in {remaining_cycles} cycles.")

    with tab2:
        st.subheader("Individual Cycle Analysis")
        st.markdown("Select a specific log file to view the 'Digital Twin' sensor data (Voltage, Temp, Current).")
        
        # Get all files related to this battery (charge, discharge, impedance)
        site_files = metadata_all[metadata_all['battery_id'] == selected_id].sort_values('test_id', ascending=False)
        selected_file = st.selectbox("Select Log File (Chronological)", site_files['filename'])
        
        sensor_df = load_detailed_csv(selected_file)
        
        if sensor_df is not None:
            c_left, c_right = st.columns(2)
            
            # Map column names (NASA datasets use these)
            v_col = 'Voltage_measured' if 'Voltage_measured' in sensor_df.columns else sensor_df.columns[0]
            t_col = 'Temperature_measured' if 'Temperature_measured' in sensor_df.columns else sensor_df.columns[0]
            
            with c_left:
                st.write("**Voltage Discharge Profile**")
                fig_v = px.line(sensor_df, y=v_col, line_shape='spline', render_mode='svg')
                fig_v.update_traces(line_color='#059669')
                st.plotly_chart(fig_v, use_container_width=True)
                
            with c_right:
                st.write("**Thermal Behavior (°C)**")
                fig_t = px.line(sensor_df, y=t_col, line_shape='spline')
                fig_t.update_traces(line_color='#DC2626')
                st.plotly_chart(fig_t, use_container_width=True)
            
            with st.expander("View Raw Sensor Table"):
                st.dataframe(sensor_df, use_container_width=True)
        else:
            st.warning(f"File `{selected_file}` not found in the `/data` directory. Please ensure all archive CSVs are moved there.")

# --- FOOTER ---
st.divider()
st.caption("© 2024 Tetranet Infrastructure Predictive Analytics | Built by Dantech Solutions")