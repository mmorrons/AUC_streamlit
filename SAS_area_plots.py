import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import trapezoid
from bokeh.plotting import figure
from bokeh.models import Span, ColumnDataSource, HoverTool

# Set up the Streamlit page
st.set_page_config(page_title="SAS - AUC", page_icon="🐼", layout="wide")
st.title("SAS - AUC v.110325")
st.write(
    "Upload exactly 4 Excel (.xlsx) files containing your acquisition data. "
    "Each file must have the time data in column 'Time (s)' and the selected measurement in its respective column. "
    "Only rows where both time and the selected data type are filled will be processed."
)

# File uploader
uploaded_files = st.file_uploader("Upload Excel files", type=["xlsx"], accept_multiple_files=True)

# Mapping for data options
data_options = {
    "Theta (°)": "Theta (°)",
    "Velocity (θ/s)": "DistGyr_X (dps)",
    "Force (g)": "DistForce (gram)",
    "Acceleration (θ²/s)": "DistAngAcc_X (dps/s^2)",
    "EMG Gastrocnemius Medial (mV)": "Ch1_GastrocnemiusMedial_GM (mV)",
    "EMG Gastrocnemius Lateralis (mV)": "Ch2_GastrocnemiusLateralis_GL (mV)",
    "EMG Soleus (mV)": "Ch3_Soleus_S (mV)",
    "EMG Tibialis Anterior (mV)": "Ch5_TibialisAnterior_TA (mV)"
}

if uploaded_files and len(uploaded_files) == 4:
    # Process files in a 2x2 grid layout
    for row in range(2):
        cols = st.columns(2)
        for col in range(2):
            index = row * 2 + col
            file = uploaded_files[index]
            with cols[col]:
                file_name = file.name
                base_name = file_name.replace(".xlsx", "").replace(".XLSX", "")
                condition = "Non Spastic" if "NonSpastic" in base_name else "Spastic"
                measurement = "Instant ReTest" if "InstantReTest" in base_name else "Post" if "PostIntervention" in base_name else "Pre"
                number = base_name.split("_")[-1]
                graph_title = f"{measurement} n°{number} {condition}".strip()
                st.subheader(graph_title)
                
                # Read header row to determine columns
                file.seek(0)
                df_temp = pd.read_excel(file, nrows=1)
                df_columns = {}
                for col_name in df_temp.columns:
                    clean_col = str(col_name).split('.')[0].strip()
                    if clean_col not in df_columns:
                        df_columns[clean_col] = col_name
                        
                # Adjust data options based on file columns
                corrected_data_options = {
                    key: df_columns.get(value, value)
                    for key, value in data_options.items()
                    if value in df_columns
                }
                
                # Get max velocity from cell V8 (row index 7, column index 21)
                file.seek(0)
                df_full = pd.read_excel(file, header=None)
                max_velocity = df_full.iloc[7, 21]
                st.markdown(
                    f"<p style='color: red; font-weight: bold;'>Max Velocity: {max_velocity}</p>",
                    unsafe_allow_html=True
                )
                
                # Let user select the measurement type
                selected_data_type = st.selectbox(
                    f"Select Data for {graph_title}",
                    list(corrected_data_options.keys()),
                    key=f"data_type_{index}"
                )
                selected_column = corrected_data_options[selected_data_type]
                
                # Read and process data
                file.seek(0)
                time_column = df_columns.get("Time (s)", "Time (s)")
                selected_column_corrected = df_columns.get(selected_column, selected_column)
                data = pd.read_excel(file, usecols=[time_column, selected_column_corrected])
                data.columns = ['time', 'selected_data']
                data = data.dropna()
                data['time'] = pd.to_numeric(data['time'], errors='coerce')
                data['selected_data'] = pd.to_numeric(data['selected_data'], errors='coerce')
                data = data.dropna()
                
                # User input for time interval and baseline
                min_time, max_time = data['time'].min(), data['time'].max()
                start_time = st.number_input(
                    f"Enter Start Time for {graph_title} (sec)",
                    min_value=float(min_time),
                    max_value=float(max_time),
                    value=float(min_time),
                    step=0.01,
                    key=f"start_time_{index}"
                )
                end_time = st.number_input(
                    f"Enter End Time for {graph_title} (sec)",
                    min_value=float(start_time),
                    max_value=float(max_time),
                    value=float(max_time),
                    step=0.01,
                    key=f"end_time_{index}"
                )
                baseline = st.number_input(
                    f"Enter Baseline Value for {graph_title}",
                    value=0.0,
                    key=f"baseline_{index}"
                )
                
                # Filter data within the selected time interval and calculate area
                mask = (data['time'] >= start_time) & (data['time'] <= end_time)
                selected_data = data[mask]
                area = trapezoid(np.abs(selected_data['selected_data'] - baseline), x=selected_data['time']) if len(selected_data) >= 2 else 0
                delta_t = end_time - start_time
                
                # Create a Bokeh figure with interactive tools (reverting to default background)
                p = figure(
                    title=graph_title,
                    x_axis_label="Time (sec)",
                    y_axis_label=selected_data_type,
                    tools="pan,wheel_zoom,box_zoom,reset,save",
                    width=800, height=500
                )
                
                # Use a ColumnDataSource for hover functionality
                source = ColumnDataSource(data)
                p.line('time', 'selected_data', source=source, line_width=0.5, color='blue', legend_label=selected_data_type)
                
                # Plot the baseline
                p.line(
                    [data['time'].min(), data['time'].max()],
                    [baseline, baseline],
                    line_width=0.5,
                    color='red',
                    line_dash='dashed',
                    legend_label=f'Baseline: {baseline}'
                )
                
                # Add a hover tool to display x and y values from the main line
                hover = HoverTool(tooltips=[("Time", "@time"), ("Value", "@selected_data")])
                p.add_tools(hover)
                
                # Add vertical spans for the start and end boundaries (no legend entries)
                start_span = Span(location=start_time, dimension='height', line_color='red', line_width=0.5)
                end_span = Span(location=end_time, dimension='height', line_color='red', line_width=0.5)
                p.add_layout(start_span)
                p.add_layout(end_span)
                
                st.bokeh_chart(p, use_container_width=True)
                
                # Display calculated results
                st.write(f"**Calculated area:** {area:.4f} ({selected_data_type.split(' ')[-1]}·sec)")
                st.write(f"**Time Interval (Δt):** {delta_t:.2f} sec")
                st.write(f"**Max Velocity:** {max_velocity:.3f} (θ/s)")
else:
    st.warning("Please upload exactly 4 Excel files to proceed.")
