import streamlit as st
st.set_page_config(
    page_title="Inertial Motion Unit Data Analysis", 
    page_icon=":rocket:",
    layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# =============================================================================
# Title and Instructions
# =============================================================================
st.title("Inertial Motion Unit Data Analysis v.110325")
st.write(
    "Upload exactly 4 Excel (.xlsx) files containing your acquisition data. "
    "Each file must have the time data in column A (starting at A2, with header in A1) "
    "and the theta data in column M (starting at M2, with header in M1). "
    "Only rows where both time and theta are filled will be processed. \n\n"
    "The file name should follow the format: \n\n"
    "`AnklePlantFlex<Condition><Measurement>_n.xlsx`, \n\n"
    "where `<Condition>` is either `NonSpastic` or `Spastic`, `<Measurement>` is either "
    "absent (indicating Pre), `PostIntervention` (for Post), or `InstantReTest` (for Instant Retest), "
    "and `n` is a number (from 4 to 6). \n\n"
    "For example: \n"
    "- AnklePlantFlexNonSpastic_4.xlsx  → Pre n°4 Non Spastic \n"
    "- AnklePlantFlexNonSpasticInstantReTest_4.xlsx → Instant Retest n°4 Non Spastic \n"
    "- AnklePlantFlexNonSpasticPostIntervention_4.xlsx → Post n°4 Non Spastic \n\n"
    "Also, the Max Velocity value from cell V8 will be displayed in bold red text."
)

# =============================================================================
# File Upload Section
# =============================================================================
uploaded_files = st.file_uploader("Upload Excel files", type="xlsx", accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 4:
    # Arrange graphs in a 2x2 grid layout
    for row in range(2):
        cols = st.columns(2)
        for col in range(2):
            index = row * 2 + col  # Index for current file/graph
            file = uploaded_files[index]
            with cols[col]:
                # -------------------------------
                # Determine Graph Title from File Name
                # -------------------------------
                file_name = file.name  # e.g., "AnklePlantFlexNonSpastic_4.xlsx" or "AnklePlantFlexNonSpasticInstantReTest_4.xlsx"
                base_name = file_name.replace(".xlsx", "").replace(".XLSX", "")
                # Remove the constant part "AnklePlantFlex"
                if base_name.startswith("AnklePlantFlex"):
                    remainder = base_name[len("AnklePlantFlex"):]
                else:
                    remainder = base_name

                # Extract condition: "NonSpastic" or "Spastic"
                if remainder.startswith("NonSpastic"):
                    condition = "NonSpastic"
                    remainder = remainder[len("NonSpastic"):]
                elif remainder.startswith("Spastic"):
                    condition = "Spastic"
                    remainder = remainder[len("Spastic"):]
                else:
                    condition = ""
                # Insert a space for display if condition is NonSpastic
                if condition == "NonSpastic":
                    condition = "Non Spastic"

                # Determine measurement based on remaining string
                if remainder.startswith("InstantReTest"):
                    measurement = "Instant Retest"
                    remainder = remainder[len("InstantReTest"):]
                elif remainder.startswith("PostIntervention"):
                    measurement = "Post"
                    remainder = remainder[len("PostIntervention"):]
                else:
                    measurement = "Pre"

                # The remainder should now be something like "_4" (or with extra spaces)
                number = remainder.lstrip("_").strip()

                # Construct the final graph title
                graph_title = f"{measurement} n°{number} {condition}".strip()

                # Display the graph title as a subheader
                st.subheader(graph_title)

                # -------------------------------
                # Read and Display Max Velocity from Cell V8
                # -------------------------------
                # Reset file pointer to the beginning
                file.seek(0)
                # Read only cell V8: use column V, skip first 7 rows so that row 8 is read.
                max_velocity_df = pd.read_excel(file, header=None, usecols="V", skiprows=7, nrows=1)
                max_velocity = max_velocity_df.iloc[0, 0]
                # Display Max Velocity in bold red text
                st.markdown(
                    f"<p style='color: red; font-weight: bold;'>Max Velocity: {max_velocity}</p>",
                    unsafe_allow_html=True,
                )

                # -------------------------------
                # Read and Clean the Data for Plotting
                # -------------------------------
                file.seek(0)
                # Read only the required columns: column A (index 0) for Time and column M (index 12) for Theta.
                data = pd.read_excel(file, usecols=[0, 12])
                data.columns = ['time', 'theta']
                data = data.dropna(subset=['time', 'theta'])
                # Convert values to float (replace commas with dots if needed)
                if data['theta'].dtype == object:
                    data['theta'] = data['theta'].astype(str).str.replace(',', '.').astype(float)
                if data['time'].dtype == object:
                    data['time'] = data['time'].astype(str).str.replace(',', '.').astype(float)

                # -------------------------------
                # User Inputs for Analysis
                # -------------------------------
                min_time = float(data['time'].min())
                max_time = float(data['time'].max())

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

                # -------------------------------
                # Data Filtering and Area Calculation
                # -------------------------------
                mask = (data['time'] >= start_time) & (data['time'] <= end_time)
                selected_data = data[mask]
                if len(selected_data) < 2:
                    area = 0
                else:
                    area = trapezoid(np.abs(selected_data['theta'] - baseline), x=selected_data['time'])

                delta_t = end_time - start_time  # Calculate Δt

                # -------------------------------
                # Plotting
                # -------------------------------
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(data['time'], data['theta'], color='blue', label='θ')
                ax.axhline(baseline, color='red', linestyle='--', label=f'Baseline: {baseline}')
                ax.fill_between(selected_data['time'], selected_data['theta'], baseline, color='pink', alpha=0.3, label='Area')
                ax.set_xlabel("Time (sec)")
                ax.set_ylabel("θ")
                ax.set_title(graph_title)
                ax.legend()
                # Display Results
                st.write(f"**Calculated area:** {area:.2f} (θ·sec)")
                st.write(f"**Time Interval (Δt):** {delta_t:.2f} sec")
                st.write(f"**Max Velocity:**{max_velocity:.3f} (θ/s")
                #Show Plot
                st.pyplot(fig)
else:
    st.warning("Please upload exactly 4 Excel files to proceed.")
