import streamlit as st
st.set_page_config(
    page_title="SAS - AUC", 
    page_icon="🐼",
    layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# =============================================================================
# Title and Instructions
# =============================================================================
st.title("SAS - AUC v.110325")
st.write(
    "Upload exactly 4 Excel (.xlsx) files containing your acquisition data. "
    "Each file must have the time data in column 'Time (s)' and the selected measurement in its respective column. "
    "Only rows where both time and the selected data type are filled will be processed."
)

# =============================================================================
# File Upload Section
# =============================================================================
uploaded_files = st.file_uploader("Upload Excel files", type=["xlsx"], accept_multiple_files=True)

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
    for row in range(2):
        cols = st.columns(2)
        for col in range(2):
            index = row * 2 + col
            file = uploaded_files[index]
            with cols[col]:
                file_name = file.name
                base_name = file_name.replace(".xlsx", "").replace(".XLSX", "")

                # Extract metadata from file name
                condition = "Non Spastic" if "NonSpastic" in base_name else "Spastic"
                measurement = "Instant Retest" if "InstantReTest" in base_name else "Post" if "PostIntervention" in base_name else "Pre"
                number = base_name.split("_")[-1]
                graph_title = f"{measurement} n°{number} {condition}".strip()
                st.subheader(graph_title)

                # Read the file and dynamically extract column names from the first row
                file.seek(0)
                df_temp = pd.read_excel(file, nrows=1)

                # Ensure only the first occurrence of each variable is used
                df_columns = {}
                for col_name in df_temp.columns:
                    clean_col = str(col_name).split('.')[0].strip()  # Remove .1, .2 suffixes
                    if clean_col not in df_columns:
                        df_columns[clean_col] = col_name  # Store the first occurrence

                # Adjust data options to match dynamically loaded column names
                corrected_data_options = {key: df_columns.get(value, value) for key, value in data_options.items() if value in df_columns}

                # Hard-code reading max velocity from cell V8 (row index 7, column index 21)
                file.seek(0)
                df_full = pd.read_excel(file, header=None)
                max_velocity = df_full.iloc[7, 21]
                st.markdown(
                    f"<p style='color: red; font-weight: bold;'>Max Velocity: {max_velocity}</p>",
                    unsafe_allow_html=True
                )

                # Select data type
                selected_data_type = st.selectbox(
                    f"Select Data for {graph_title}", list(corrected_data_options.keys()), key=f"data_type_{index}"
                )
                selected_column = corrected_data_options[selected_data_type]

                # Read and process data
                file.seek(0)
                time_column = df_columns.get("Time (s)", "Time (s)")  # Ensure first occurrence
                selected_column_corrected = df_columns.get(selected_column, selected_column)

                data = pd.read_excel(file, usecols=[time_column, selected_column_corrected])
                data.columns = ['time', 'selected_data']
                data = data.dropna()

                # Convert data types
                data['time'] = pd.to_numeric(data['time'], errors='coerce')
                data['selected_data'] = pd.to_numeric(data['selected_data'], errors='coerce')
                data = data.dropna()

                # User input for time interval and baseline
                min_time, max_time = data['time'].min(), data['time'].max()
                start_time = st.number_input(
                    f"Enter Start Time for {graph_title} (sec)",
                    min_value=min_time,
                    max_value=max_time,
                    value=min_time,
                    step=0.01,
                    key=f"start_time_{index}"
                )
                end_time = st.number_input(
                    f"Enter End Time for {graph_title} (sec)",
                    min_value=start_time,
                    max_value=max_time,
                    value=max_time,
                    step=0.01,
                    key=f"end_time_{index}"
                )
                baseline = st.number_input(
                    f"Enter Baseline Value for {graph_title}",
                    value=0.0,
                    key=f"baseline_{index}"
                )

                # Filter data within the specified time interval
                mask = (data['time'] >= start_time) & (data['time'] <= end_time)
                selected_data = data[mask]
                area = trapezoid(np.abs(selected_data['selected_data'] - baseline), x=selected_data['time']) if len(selected_data) >= 2 else 0
                delta_t = end_time - start_time

                # Plotting the data
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(data['time'], data['selected_data'], color='blue', label=selected_data_type)
                ax.axhline(baseline, color='red', linestyle='--', label=f'Baseline: {baseline}')
                ax.fill(color=red, alpha=0.5)
                ax.fill_between(
                    selected_data['time'],
                    selected_data['selected_data'],
                    baseline,
                    color='pink',
                    alpha=0.3,
                    label='Area'
                )
                ax.set_xlabel("Time (sec)")
                ax.set_ylabel(selected_data_type)
                ax.set_title(graph_title)
                ax.legend()

                # Display results
                st.write(f"**Calculated area:** {area:.4f} ({selected_data_type.split(' ')[-1]}·sec)")
                st.write(f"**Time Interval (Δt):** {delta_t:.2f} sec")
                st.write(f"**Max Velocity:** {max_velocity:.3f} (θ/s)")

                st.pyplot(fig)
else:
    st.warning("Please upload exactly 4 Excel files to proceed.")
