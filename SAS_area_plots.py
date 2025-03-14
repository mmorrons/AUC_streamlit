import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import trapezoid

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
    # Process each file and display its graph in a grid (2 columns x 2 rows)
    for row in range(2):
        cols = st.columns(2)
        for col in range(2):
            index = row * 2 + col
            file = uploaded_files[index]
            with cols[col]:
                file_name = file.name
                base_name = file_name.replace(".xlsx", "").replace(".XLSX", "")

                # Extract metadata from the file name
                condition = "Non Spastic" if "NonSpastic" in base_name else "Spastic"
                measurement = (
                    "Instant Retest" if "InstantReTest" in base_name
                    else "Post" if "PostIntervention" in base_name
                    else "Pre"
                )
                number = base_name.split("_")[-1]
                graph_title = f"{measurement} n°{number} {condition}".strip()
                st.subheader(graph_title)

                # Read first row to determine column names
                file.seek(0)
                df_temp = pd.read_excel(file, nrows=1)
                df_columns = {}
                for col_name in df_temp.columns:
                    clean_col = str(col_name).split('.')[0].strip()
                    if clean_col not in df_columns:
                        df_columns[clean_col] = col_name

                # Adjust the data options based on the file's columns
                corrected_data_options = {
                    key: df_columns.get(value, value)
                    for key, value in data_options.items()
                    if value in df_columns
                }

                # Read the file to get max velocity from cell V8 (row index 7, column index 21)
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

                # Read and process data columns
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

                # Filter data for the selected time interval and calculate area under the curve
                mask = (data['time'] >= start_time) & (data['time'] <= end_time)
                selected_data = data[mask]
                area = (
                    trapezoid(np.abs(selected_data['selected_data'] - baseline), x=selected_data['time'])
                    if len(selected_data) >= 2 else 0
                )
                delta_t = end_time - start_time

                                # Display calculated results
                st.write(f"**Calculated area:** {area:.4f} ({selected_data_type.split(' ')[-1]}·sec)")
                st.write(f"**Time Interval (Δt):** {delta_t:.2f} sec")
                st.write(f"**Max Velocity:** {max_velocity:.3f} (θ/s)")
                
                # Create an interactive Plotly figure
                fig = go.Figure()

                # Main data trace
                fig.add_trace(go.Scatter(
                    x=data['time'],
                    y=data['selected_data'],
                    mode='lines',
                    name=selected_data_type,
                    line=dict(color='blue'),
                    hovertemplate='Time: %{x}<br>Value: %{y}<extra></extra>'
                ))
                # Baseline trace
                fig.add_trace(go.Scatter(
                    x=[data['time'].min(), data['time'].max()],
                    y=[baseline, baseline],
                    mode='lines',
                    name=f'Baseline: {baseline}',
                    line=dict(color='red', dash='dash'),
                    hovertemplate=f'End: Time: %{{x}}<br>{select_data_type}: %{{y}}<extra></extra>'
                ))
                # Fill the area between the curve and baseline with 0.4 transparency
                fig.add_trace(go.Scatter(
                    x=np.concatenate([selected_data['time'], selected_data['time'][::-1]]),
                    y=np.concatenate([selected_data['selected_data'], np.full(len(selected_data), baseline)]),
                    fill='toself',
                    fillcolor='rgba(255, 178, 178, 0.34)',
                    line=dict(color='black'),
                    showlegend=False,
                    name='Area'
                ))
                # Vertical boundary lines at start and end (without legend entries)
                start_y = np.interp(start_time, data['time'], data['selected_data'])
                end_y = np.interp(end_time, data['time'], data['selected_data'])
                fig.add_trace(go.Scatter(
                    x=[start_time, start_time],
                    y=[baseline, start_y],
                    mode='lines',
                    line=dict(color='red'),
                    showlegend=False,
                    hovertemplate=f'Start: Time: %{{x}}<br>{select_data_type}: %{{y}}<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=[end_time, end_time],
                    y=[baseline, end_y],
                    mode='lines',
                    line=dict(color='red'),
                    showlegend=False,
                    hovertemplate=f'End: Time: %{{x}}<br>{select_data_type}: %{{y}}<extra></extra>'
                ))

                # Update layout to set a dark grey background and white font for readability
                fig.update_layout(
                    title=graph_title,
                    xaxis_title="Time (sec)",
                    yaxis_title=selected_data_type,
                    hovermode='closest',
                )

                st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Please upload exactly 4 Excel files to proceed.")
