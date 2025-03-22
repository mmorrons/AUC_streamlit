import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import trapezoid
import os

# Helper function to process the file and compute metrics/figure
def process_file(file_obj, file_name, selected_data_type, data_options, start_time, end_time, baseline):
    # --- Extract column info ---
    file_obj.seek(0)
    try:
        df_temp = pd.read_excel(file_obj, nrows=1)
    except Exception as e:
        return None, f"Error reading file: {e}"
    df_columns = {}
    for col_name in df_temp.columns:
        clean_col = str(col_name).split('.')[0].strip()
        if clean_col not in df_columns:
            df_columns[clean_col] = col_name

    # Adjust data options based on the file's columns
    corrected_data_options = {
        key: df_columns.get(value, value)
        for key, value in data_options.items()
        if value in df_columns
    }
    if not corrected_data_options:
        return None, "No valid data columns found in the selected file based on the mapping."

    # If the submitted selected_data_type isn’t in the corrected list (should not happen), choose the first available
    if selected_data_type not in corrected_data_options:
        selected_data_type = list(corrected_data_options.keys())[0]
    selected_column = corrected_data_options[selected_data_type]

    # --- Build a graph title using file name and (if possible) folder info ---
    base_name = os.path.basename(file_name).replace(".xlsx", "").replace(".XLSX", "")
    condition = "Non Spastic" if "NonSpastic" in base_name else "Spastic"
    measurement = "Instant Retest" if "InstantReTest" in base_name else ("Post" if "PostIntervention" in base_name else "Pre")
    number = base_name.split("_")[-1]
    graph_title = f"{measurement} n°{number} {condition}".strip()

    # --- Get max velocity from a fixed cell (V8: row 7, col 21) ---
    file_obj.seek(0)
    df_full = pd.read_excel(file_obj, header=None)
    try:
        max_velocity = df_full.iloc[7, 21]
    except Exception as e:
        max_velocity = None

    # --- Read the relevant data columns ---
    file_obj.seek(0)
    time_column = df_columns.get("Time (s)", "Time (s)")
    selected_column_corrected = df_columns.get(selected_column, selected_column)
    try:
        data = pd.read_excel(file_obj, usecols=[time_column, selected_column_corrected])
    except Exception as e:
        return None, f"Error reading data columns from {file_name}: {e}"
    data.columns = ['time', 'selected_data']
    data = data.dropna()
    data['time'] = pd.to_numeric(data['time'], errors='coerce')
    data['selected_data'] = pd.to_numeric(data['selected_data'], errors='coerce')
    data = data.dropna()
    if data.empty:
        return None, "No data found after processing."

    # --- Adjust start/end if they are outside the data range ---
    min_time, max_time_value = data['time'].min(), data['time'].max()
    if start_time < min_time:
        start_time = min_time
    if end_time > max_time_value:
        end_time = max_time_value
    if start_time >= end_time:
        return None, "Start time must be less than End time."

    # --- Filter data and compute area under the curve ---
    mask = (data['time'] >= start_time) & (data['time'] <= end_time)
    selected_data_df = data[mask]
    area = (trapezoid(np.abs(selected_data_df['selected_data'] - baseline),
                      x=selected_data_df['time'])
            if len(selected_data_df) >= 2 else 0)
    delta_t = end_time - start_time

    # --- Extra EMG statistics (only for EMG variables) ---
    emg_stats = {}
    if "EMG" in selected_data_type:
        emg_peak = selected_data_df['selected_data'].max()
        emg_avg = selected_data_df['selected_data'].mean()
        emg_rms = np.sqrt(np.mean(selected_data_df['selected_data']**2))
        emg_stats = {
            "emg_peak": emg_peak,
            "emg_avg": emg_avg,
            "emg_rms": emg_rms
        }

    # --- Extract unit from the data type string for the legend ---
    if "(" in selected_data_type and ")" in selected_data_type:
        unit = selected_data_type.split("(")[1].split(")")[0]
    else:
        unit = selected_data_type

    # --- Create the Plotly figure ---
    fig = go.Figure()
    # Main data trace
    fig.add_trace(go.Scatter(
        x=data['time'],
        y=data['selected_data'],
        mode='lines',
        name=selected_data_type,
        line=dict(color='blue'),
        hovertemplate=f"Time: %{{x}}<br>{unit}: %{{y}}<extra></extra>"
    ))
    # Baseline trace
    fig.add_trace(go.Scatter(
        x=[data['time'].min(), data['time'].max()],
        y=[baseline, baseline],
        mode='lines',
        name=f'Baseline: {baseline}',
        line=dict(color='red', dash='dash'),
        hovertemplate=f"Time: %{{x}}<br>{unit}: %{{y}}<extra></extra>"
    ))
    # Shaded area between the curve and the baseline
    fig.add_trace(go.Scatter(
        x=np.concatenate([selected_data_df['time'], selected_data_df['time'][::-1]]),
        y=np.concatenate([selected_data_df['selected_data'],
                           np.full(len(selected_data_df), baseline)]),
        fill='toself',
        fillcolor='rgba(255, 178, 178, 0.34)',
        line=dict(color='red'),
        showlegend=False,
        name='Area'
    ))
    # Vertical lines at start and end times
    start_y = np.interp(start_time, data['time'], data['selected_data'])
    end_y = np.interp(end_time, data['time'], data['selected_data'])
    fig.add_trace(go.Scatter(
        x=[start_time, start_time],
        y=[baseline, start_y],
        mode='lines',
        line=dict(color='red'),
        showlegend=False,
        hovertemplate=f"Start Time: %{{x}}<br>{unit}: %{{y}}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=[end_time, end_time],
        y=[baseline, end_y],
        mode='lines',
        line=dict(color='red'),
        showlegend=False,
        hovertemplate=f"End Time: %{{x}}<br>{unit}: %{{y}}<extra></extra>"
    ))
    fig.update_layout(
        legend=dict(x=0.95, y=0.95, xanchor='right', yanchor='top'),
        title=graph_title,
        xaxis_title="Time (sec)",
        yaxis_title=selected_data_type,
        hovermode='closest'
    )

    # Return all computed metrics
    output = {
        "graph_title": graph_title,
        "max_velocity": max_velocity,
        "area": area,
        "delta_t": delta_t,
        "fig": fig,
        "unit": unit,
        "data_type": selected_data_type,
        "start_time": start_time,
        "end_time": end_time,
        "baseline": baseline,
        "file_name": file_name
    }
    output.update(emg_stats)
    return output, None

# ------------------ Main App Code ------------------

st.set_page_config(page_title="SAS - AUC", page_icon="🦒", layout="wide")
st.title("SAS - AUC v.220325")
st.write(
    "Upload any number of Excel (.xlsx) files containing your acquisition data. "
    "Each file must have the time data in column 'Time (s)' and the selected measurement in its respective column. "
    "Only rows where both time and the selected data type are filled will be processed."
)

# Allow multiple file uploads
uploaded_files = st.file_uploader("Upload Excel files", type=["xlsx"], accept_multiple_files=True)
if not uploaded_files:
    st.warning("Please upload at least one Excel file to proceed.")
    st.stop()

# Create a dictionary mapping file names to file objects.
file_dict = {file.name: file for file in uploaded_files}

st.write("Configure settings for each of the 4 graphs below:")

# Define the mapping for data options.
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

# Create a 2x2 grid for the four graphs.
for row in range(2):
    cols = st.columns(2)
    for col in range(2):
        index = row * 2 + col
        with cols[col]:
            with st.container():
                st.markdown(f"### Graph {index+1}")
                # Use a form so that changes are applied only when OK is pressed.
                with st.form(key=f"form_{index}"):
                    # File selection.
                    selected_file_name = st.selectbox(
                        "Select Excel File", options=list(file_dict.keys()), key=f"file_select_{index}"
                    )
                    file_obj = file_dict[selected_file_name]
                    st.write("File Path:", selected_file_name)
                    # Attempt to extract folder info (subject and session) if available.
                    folder_parts = os.path.normpath(selected_file_name).split(os.sep)
                    if len(folder_parts) >= 3:
                        st.write("Subject:", folder_parts[-3], "| Session:", folder_parts[-2])
                    else:
                        st.write("Subject and Session info not available from the file path.")

                    # Read first row to extract column names.
                    file_obj.seek(0)
                    try:
                        df_temp = pd.read_excel(file_obj, nrows=1)
                    except Exception as e:
                        st.error(f"Error reading file {selected_file_name}: {e}")
                        st.form_submit_button("OK")
                        continue
                    df_columns = {}
                    for col_name in df_temp.columns:
                        clean_col = str(col_name).split('.')[0].strip()
                        if clean_col not in df_columns:
                            df_columns[clean_col] = col_name

                    # Adjust data options based on the file's columns.
                    corrected_data_options = {
                        key: df_columns.get(value, value)
                        for key, value in data_options.items()
                        if value in df_columns
                    }
                    if not corrected_data_options:
                        st.error("No valid data columns found in the selected file based on the mapping.")
                        st.form_submit_button("OK")
                        continue

                    # Data type selection.
                    selected_data_type = st.selectbox(
                        "Select Data", list(corrected_data_options.keys()), key=f"data_type_{index}"
                    )

                    # To determine time range defaults, read the relevant columns.
                    file_obj.seek(0)
                    time_column = df_columns.get("Time (s)", "Time (s)")
                    selected_column_corrected = df_columns.get(corrected_data_options[selected_data_type],
                                                                 corrected_data_options[selected_data_type])
                    try:
                        data = pd.read_excel(file_obj, usecols=[time_column, selected_column_corrected])
                    except Exception as e:
                        st.error(f"Error reading data columns from {selected_file_name}: {e}")
                        st.form_submit_button("OK")
                        continue
                    data.columns = ['time', 'selected_data']
                    data = data.dropna()
                    data['time'] = pd.to_numeric(data['time'], errors='coerce')
                    data['selected_data'] = pd.to_numeric(data['selected_data'], errors='coerce')
                    data = data.dropna()
                    if data.empty:
                        st.error("No data found after processing. Check the file content.")
                        st.form_submit_button("OK")
                        continue
                    min_time, max_time_value = data['time'].min(), data['time'].max()

                    # Independent inputs for start, end, and baseline.
                    start_time_input = st.number_input(
                        "Enter Start Time (sec)",
                        min_value=float(min_time),
                        max_value=float(max_time_value),
                        value=float(min_time),
                        step=0.01,
                        key=f"start_time_{index}"
                    )
                    end_time_input = st.number_input(
                        "Enter End Time (sec)",
                        min_value=float(min_time),
                        max_value=float(max_time_value),
                        value=float(max_time_value),
                        step=0.01,
                        key=f"end_time_{index}"
                    )
                    baseline_input = st.number_input(
                        "Enter Baseline Value", value=0.0, key=f"baseline_{index}"
                    )

                    # The OK button: when pressed, store the parameters and computed output.
                    submitted = st.form_submit_button("OK")
                    if submitted:
                        if start_time_input >= end_time_input:
                            st.error("Start time must be less than End time.")
                            st.session_state[f"graph_{index}_output"] = None
                        else:
                            # Always re-seek the file before processing.
                            file_obj.seek(0)
                            output, error = process_file(
                                file_obj,
                                selected_file_name,
                                selected_data_type,
                                data_options,
                                start_time_input,
                                end_time_input,
                                baseline_input
                            )
                            if error:
                                st.error(error)
                                st.session_state[f"graph_{index}_output"] = None
                            else:
                                st.session_state[f"graph_{index}_output"] = output

                # Outside the form, display the (saved) graph and chosen values.
                output = st.session_state.get(f"graph_{index}_output", None)
                if output:
                    st.subheader(output["graph_title"])
                    if output["max_velocity"] is not None:
                        st.markdown(
                            f"<p style='color: red; font-weight: bold;'>Max Velocity: {output['max_velocity']}</p>",
                            unsafe_allow_html=True
                        )
                    st.write(f"**Calculated area:** {output['area']:.4f} ({output['data_type'].split(' ')[-1]}·sec)")
                    st.write(f"**Time Interval (Δt):** {output['delta_t']:.2f} sec")
                    st.write(
                        f"**Start Time:** {output['start_time']} sec | "
                        f"**End Time:** {output['end_time']} sec | "
                        f"**Baseline:** {output['baseline']}"
                    )
                    # If the selected data type is EMG, display extra EMG statistics.
                    if "EMG" in output["data_type"]:
                        st.write(f"**EMG Peak (max):** {output['emg_peak']:.4f} mV")
                        st.write(f"**EMG Average:** {output['emg_avg']:.4f} mV")
                        st.write(f"**EMG RMS:** {output['emg_rms']:.4f} mV")
                    st.plotly_chart(output["fig"], use_container_width=True)
                else:
                    st.info("Please press OK to generate the graph.")
