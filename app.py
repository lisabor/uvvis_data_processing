import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import auswertung_1
import functions
import os
import tempfile
import zipfile
from io import BytesIO

st.title("UV/Vis Data Processing")

# select if you want default calibration data or own
own_calibration = st.checkbox("Use Custom Calibration Data", value=True)

if own_calibration:
    uploaded_calibration = st.file_uploader("Upload Calibration data", accept_multiple_files=True, type=['csv'])
    concentrations_input = st.text_input(
        "Enter Concentrations (separated by commas)",
        placeholder="e.g., 1, 2, 5, 10"
    )

    if concentrations_input:
        try:
            concentrations = [float(x.strip()) for x in concentrations_input.split(',')]
            st.success(f"Concentrations: {concentrations}")
        except ValueError:
            st.error("Please enter valid numbers separated by commas.")
            concentrations = None
    else:
        concentrations = None

    if uploaded_calibration and concentrations:
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_calibration:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            slope = auswertung_1.calibration(temp_dir, concentrations)
            st.success("Calibration completed successfully.")
    else:
        slope = None
        if own_calibration:
            st.info("Please upload calibration files and enter concentrations.")
else:
    conc = [1, 2, 5, 10]
    slope = auswertung_1.calibration('data/MB_Kalibrierung', conc, show=False)
    st.success("Using default calibration data for methylene blue")


# now you can continue with processing the other data
if slope is not None:
    display_option = st.radio(
    "How many experiments should be processed?",
    options=["Only One Experiment", "A Row of Experiments"],
    horizontal=True
)
    if display_option == "Only One Experiment":
        uploaded_files = st.file_uploader("Upload CSV files (name_Zeit.csv)", accept_multiple_files=True, type=['csv'])

        if uploaded_files:
            # Save uploaded files to a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())


                fig, data = auswertung_1.zeitverlauf(temp_dir, slope)
                fig = functions.customize_plot(fig)
                data = data.sort_values(by='time')
                st.dataframe(data)
                st.plotly_chart(fig, use_container_width=True, theme=None)
        
    elif display_option == "A Row of Experiments":
        uploaded_experiments = st.file_uploader("Upload ZIP files for each experiment (each ZIP contains CSV files)",
        accept_multiple_files=True,
        type=['zip']
        )

        experiments_data = {}

        if uploaded_experiments:
            for zip_file in uploaded_experiments:
                experiment_name = os.path.splitext(zip_file.name)[0]  # Use ZIP file name as experiment name
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(BytesIO(zip_file.read())) as z:
                        z.extractall(temp_dir)
                
                    plots = auswertung_1.plot_reihe(temp_dir, slope)

                for plot in plots:
                    st.header(plot['title'])
                    st.plotly_chart(plot['fig'])

# feature inspect the spectra data with plot_spectra show true
