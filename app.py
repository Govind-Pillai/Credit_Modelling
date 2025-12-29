import streamlit as st
import pandas as pd

uploaded_file_1 = st.file_uploader('Upload your file 1st file here', type=['xlsx', 'csv'])
if uploaded_file_1 is not None:
    try:
        if uploaded_file_1.name.endswith('.csv'):
            df_1 = pd.read_csv(uploaded_file_1)
            file_type = "CSV"
        else:
            df_1 = pd.read_excel(uploaded_file_1)
            file_type = "Excel"
            
        st.success(f"{file_type} file successfully uploaded and imported!")

        # Display the DataFrame
        st.write("Data Preview:", df_1.head())
        if st.checkbox("Show full data", key = "df_1"):
            st.dataframe(df_1)

    except Exception as e:
        st.error(f"An error occurred during file processing: {e}")
        st.info("Please ensure the file is a valid CSV or Excel file.")

uploaded_file_2 = st.file_uploader('Upload your file 2nd file here', type=['xlsx', 'csv'])
if uploaded_file_2 is not None:
    try:
        if uploaded_file_2.name.endswith('.csv'):
            df_2 = pd.read_csv(uploaded_file_2)
            file_type = "CSV"
        else:
            df_2 = pd.read_excel(uploaded_file_2)
            file_type = "Excel"
            
        st.success(f"{file_type} file successfully uploaded and imported!")

        # Display the DataFrame
        st.write("Data Preview:", df_2.head())
        if st.checkbox("Show full data", key = "df_2"):
            st.dataframe(df_2)

    except Exception as e:
        st.error(f"An error occurred during file processing: {e}")
        st.info("Please ensure the file is a valid CSV or Excel file.")

# Merge and Validation Logic
if uploaded_file_1 is not None and uploaded_file_2 is not None:
    try:
        # Check if 'PROSPECTID' exists in both dataframes
        if 'PROSPECTID' in df_1.columns and 'PROSPECTID' in df_2.columns:
            df_merged = pd.merge(df_1, df_2, on='PROSPECTID', how='inner')
            st.write("Merged Data Preview:", df_merged.head())
            
            # Validation
            REQUIRED_COLUMNS = []  # Placeholder for required columns
            missing_columns = [col for col in REQUIRED_COLUMNS if col not in df_merged.columns]
            
            if not missing_columns:
                st.success("Validation Successful! All required columns are present.")
                if st.button("Proceed"):
                    st.write("Proceeding...") # Placeholder for next steps
            else:
                st.error(f"Validation Failed. Missing columns: {', '.join(missing_columns)}")
        else:
            st.error("Merge Failed: 'PROSPECTID' column missing in one or both files.")

    except Exception as e:
        st.error(f"An error occurred during merging: {e}")
