import pandas as pd
import io
import pickle
import numpy as np
import streamlit as st


# Page Configuration
st.set_page_config(page_title="Loan Approval System", layout="wide")

st.title("🏦 Smart Loan Approval Dashboard")
st.markdown("---")

# Sidebar Model Selection
st.sidebar.header("⚙️ Model Configuration")

# Initialize session state for locking if it doesn't exist
if 'locked' not in st.session_state:
    st.session_state.locked = False

# The Radio Button is disabled based on the session state
model_choice = st.sidebar.radio(
    "Select Model",
    ["Gradient Boosting", "Random Forest"],
    key="model_choice_radio",
    disabled=st.session_state.locked 
)

st.sidebar.warning("Confirm the model to enable predictions.")

col_left, col_right = st.sidebar.columns(2)
with col_left:    
    if st.sidebar.button("Confirm", type="primary"):
        st.session_state.locked = True
        st.rerun()
with col_right:
    if st.sidebar.button("Reset", type="secondary"):
        st.session_state.locked = False
        st.rerun()

# Load models globally
if st.session_state.locked:
    model_path = "gradient_boosting_model.pkl" if model_choice == "Gradient Boosting" else "random_forest_model.pkl"
    st.sidebar.success(f"Using {model_choice}")
    
    @st.cache_resource
    def load_ml_components(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        return model, le

    try:
        model, le = load_ml_components(model_path)
        st.sidebar.info(f"Loaded: {model_path}")
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        st.stop()
else:
    st.info("👈 Please select and confirm a model in the sidebar to proceed.")
    st.stop()

# Creating Tabs for cleaner UI
tab1, tab2, tab3 = st.tabs(["📊 Batch Predictor", "📋 Real-time Decision", "💾 Template Generator"])
important_features = ['Credit_Score', 'Age_Oldest_TL', 'enq_L3m', 'enq_L6m', 'num_std', 'time_since_recent_enq', 'num_std_12mts', 'pct_PL_enq_L6m_of_L12m', 'num_std_6mts', 'time_since_recent_deliquency']

with tab1:
    st.header("Step 1: Data Integration & Prediction")
    st.info("Upload Bank and CIBIL datasets to generate batch predictions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file_1 = st.file_uploader('Upload Bank Data (case_study1.xlsx)', type=['xlsx', 'csv'])
    with col2:
        uploaded_file_2 = st.file_uploader('Upload CIBIL Data (case_study2.xlsx)', type=['xlsx', 'csv'])

    if uploaded_file_1 and uploaded_file_2:
        # Load files
        df1 = pd.read_csv(uploaded_file_1) if uploaded_file_1.name.endswith('.csv') else pd.read_excel(uploaded_file_1)
        df2 = pd.read_csv(uploaded_file_2) if uploaded_file_2.name.endswith('.csv') else pd.read_excel(uploaded_file_2)
        
        # Merge on PROSPECTID
        if 'PROSPECTID' in df1.columns and 'PROSPECTID' in df2.columns:
            df_merged = pd.merge(df1, df2, on='PROSPECTID', how='inner')
            st.success(f"✅ Successfully merged all records!")
            
            with st.expander("Preview Merged Data"):
                st.dataframe(df_merged.head(10))

            for i in important_features:
                if i not in df_merged.columns:
                    st.error(f"Error: Merged data does not contain {i} column.")
                    st.stop()
            df_merged = df_merged[['PROSPECTID'] + important_features]
            
            if st.button("🚀 Proceed to Prediction"):
                with st.spinner("Analyzing credit profiles..."):
                    # Prepare data for prediction (dropping PROSPECTID as per main.py logic)
                    X = df_merged.drop(columns=['PROSPECTID'], errors='ignore')
                    
                    # Handle internal placeholder -99999
                    X.replace(-99999, np.nan, inplace=True)
                    
                    # Make Predictions
                    try:
                        preds = model.predict(X)
                        pred_labels = le.inverse_transform(preds)
                        
                        # Add results to dataframe
                        df_merged['Predicted_Class'] = pred_labels
                        
                        st.divider()
                        st.subheader("Results")
                        
                        # Show summary metrics
                        c1, c2, c3, c4 = st.columns(4)
                        counts = df_merged['Predicted_Class'].value_counts()
                        for i, label in enumerate(['P1', 'P2', 'P3', 'P4']):
                            count = counts.get(label, 0)
                            st.columns(4)[i].metric(f"{label} (Count)", count)
                        
                        st.dataframe(df_merged[['PROSPECTID', 'Predicted_Class']].head(20))
                        
                        # Download button
                        csv = df_merged.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results with Predictions",
                            data=csv,
                            file_name="loan_predictions.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        st.info("Ensure the uploaded files match the features used during training.")
        else:
            st.error("Error: Both files must contain a 'PROSPECTID' column for merging.")

with tab2:
    st.header("Step 2: Real-time Individual Decision")
    st.info("Enter customer details below to calculate the Approval Category based on the top 10 important features.")
    
    with st.form("prediction_form"):
        # Top 10 important features
        col1, col2 = st.columns(2)
        
        with col1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=750, help="Customer's Credit Score (300-900)")
            age_oldest_tl = st.number_input("Age of Oldest Trade Line (Months)", min_value=0, value=50, help="Age_Oldest_TL")
            enq_l3m = st.number_input("Enquiries in last 3 months", min_value=0, value=0, help="enq_L3m")
            enq_l6m = st.number_input("Enquiries in last 6 months", min_value=0, value=1, help="enq_L6m")
            num_std = st.number_input("Number of Standard Accounts", min_value=0, value=5, help="num_std")
            
        with col2:
            time_since_recent_enq = st.number_input("Time since recent enquiry (Days)", min_value=0, value=100, help="time_since_recent_enq")
            num_std_12mts = st.number_input("Number of Standard Accounts (12m)", min_value=0, value=2, help="num_std_12mts")
            pct_pl_enq_l6m_of_l12m = st.number_input("% PL enquiries L6m of L12m", min_value=0.0, max_value=1.0, value=0.0, step=0.1, help="pct_PL_enq_L6m_of_L12m")
            num_std_6mts = st.number_input("Number of Standard Accounts (6m)", min_value=0, value=1, help="num_std_6mts")
            time_since_recent_deliquency = st.number_input("Time since recent delinquency (Months)", min_value=0, value=12, help="time_since_recent_deliquency")
            
        predict_btn = st.form_submit_button("🚀 Generate Loan Decision")
        
        if predict_btn:
            # Create a dataframe for the input
            input_data = pd.DataFrame([{
                'Credit_Score': credit_score,
                'Age_Oldest_TL': age_oldest_tl,
                'enq_L3m': enq_l3m,
                'enq_L6m': enq_l6m,
                'num_std': num_std,
                'time_since_recent_enq': time_since_recent_enq,
                'num_std_12mts': num_std_12mts,
                'pct_PL_enq_L6m_of_L12m': pct_pl_enq_l6m_of_l12m,
                'num_std_6mts': num_std_6mts,
                'time_since_recent_deliquency': time_since_recent_deliquency
            }])
            
            try:
                # Handle placeholders if necessary (same logic as batch)
                input_data.replace(-99999, np.nan, inplace=True)
                
                # Prediction
                # The model is a pipeline, it will handle the preprocessing
                prediction = model.predict(input_data)
                pred_label = le.inverse_transform(prediction)[0]
                
                # Result Display
                st.markdown("---")
                st.subheader("Decision Result:")
                
                color_map = {
                    'P1': 'green',
                    'P2': 'blue',
                    'P3': 'orange',
                    'P4': 'red'
                }
                label_desc = {
                    'P1': 'Low Risk - Highly Recommended',
                    'P2': 'Moderate Risk - Recommended with caution',
                    'P3': 'High Risk - Review Required',
                    'P4': 'Very High Risk - Not Recommended'
                }
                
                st.markdown(f"Result: <span style='color:{color_map.get(pred_label, 'black')}; font-size:24px; font-weight:bold;'>{pred_label}</span>", unsafe_allow_html=True)
                st.info(f"**Description:** {label_desc.get(pred_label, 'Unknown Status')}")
                
                if pred_label in ['P1', 'P2']:
                    st.success("✅ Application likely to be Approved.")
                else:
                    st.error("❌ Application likely to be Rejected or Requires Manual Review.")
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.info("Please ensure all inputs are valid and the model is correctly loaded.")

with tab3:
    st.header("Step 3: Template Generator")
    st.write("Need a fresh template to collect data? Use the buttons below.")
    
    # Logic to create empty template
    template_cols = ['PROSPECTID', 'Total_TL', 'Age_Oldest_TL', 'NETMONTHLYINCOME', 'GENDER', 'EDUCATION']
    template_df = pd.DataFrame(columns=template_cols)
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        template_df.to_excel(writer, index=False, sheet_name='Template')
        
    st.download_button(
        label="📥 Download Empty CSV Template",
        data=template_df.to_csv(index=False).encode('utf-8'),
        file_name='loan_template.csv',
        mime='text/csv',
    )
    
    st.download_button(
        label="📥 Download Empty Excel Template",
        data=buffer.getvalue(),
        file_name='loan_template.xlsx',
        mime='application/vnd.ms-excel'
    )