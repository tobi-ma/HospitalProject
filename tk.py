import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

import cleaned_code as c
from streamlit_extras.app_logo import add_logo

# Logo Awesome Inc.
st.image('img/Awesome_Inc (1).png', width=300)

# Title of the App
st.title("Hospital Recommendations")

# Load necessary data
hospital_info = c.hospital_info

# State selection
states = hospital_info['State'].unique()
selected_state = st.selectbox('Select State', states)

# City selection based on the selected state
cities = hospital_info[hospital_info['State'] == selected_state]['City'].unique()
selected_city = st.selectbox('Select City', cities)

# Hospital selection based on the selected city
hospitals = hospital_info[(hospital_info['State'] == selected_state) & (hospital_info['City'] == selected_city)]
hospital_names = hospitals['Hospital Name'].unique()
selected_hospital_name = st.selectbox('Select Hospital', hospital_names)

# Get the Provider ID for the selected hospital
selected_hospital = hospitals[hospitals['Hospital Name'] == selected_hospital_name]
selected_hospital_id = selected_hospital['Provider ID'].values[0]

# Auto-fill city and state when hospital is selected
if selected_hospital_name:
    selected_city = selected_hospital['City'].values[0]
    selected_state = selected_hospital['State'].values[0]

# Display current hospital data
st.subheader(f"Current Data for {selected_hospital_name} (Hospital ID {selected_hospital_id})")
hospital_data_view = hospital_info[hospital_info['Provider ID'] == selected_hospital_id][[
    'Provider ID', 'Hospital Name', 'Address', 'City', 'State', 'ZIP Code',
    'County Name', 'Phone Number', 'Hospital Ownership', 'Emergency Services',
    'Hospital overall rating', 'Mortality national comparison',
    'Safety of care national comparison', 'Readmission national comparison',
    'Patient experience national comparison', 'Effectiveness of care national comparison',
    'Timeliness of care national comparison', 'Efficient use of medical imaging national comparison'
]]
st.write(hospital_data_view)

# Ensure hospital_data contains the necessary features for prediction
hospital_data = hospital_info[hospital_info['Provider ID'] == selected_hospital_id]

# Verify required features are present in the DataFrame
missing_features = [feature for feature in c.svm_features if feature not in hospital_data.columns]
if missing_features:
    st.error(f"The following features are missing: {missing_features}")
else:
    # Compute the current prediction for the selected hospital
    current_prediction = c.model.predict(hospital_data[c.svm_features])[0]
    st.write(f"Current predicted rating for Hospital ID {selected_hospital_id}: {current_prediction}")

    # Desired rating selection
    desired_rating = st.slider('Select Desired Rating', 1, 5, 4)

    # Generate and display recommendations
    if st.button('Generate Recommendations'):
        feature_importances = c.feature_importances
        recommendations = c.generate_recommendations(selected_hospital_id, hospital_info, feature_importances, desired_rating)

        st.subheader(f"Recommendations for {selected_hospital_name} (Hospital ID {selected_hospital_id}):")
        if not recommendations:
            st.write("No recommendations available.")
        else:
            for feature, details in recommendations.items():
                st.write(f"- **{feature}**: Increase from {details['current_value']} to {details['recommended_value']} (Difference: {details['difference']:.2f})")

    # Button to accept recommendations
    if st.button("Accept Recommendations"):
        # Ensure feature_importances is defined here
        feature_importances = c.feature_importances
        modified_data, new_version = c.process_hospital_data(selected_hospital_id, hospital_info, feature_importances, c.model, c.version_history, desired_rating)
        st.session_state.modified_data = modified_data
        st.session_state.new_version = new_version
        st.success(f"Recommendations applied and new version {new_version} saved.")

    # Optional: Calculate the new prediction for the modified hospital
    if st.button("Calculate New Prediction"):
        if 'modified_data' in st.session_state:
            modified_data = st.session_state.modified_data
            new_prediction = c.model.predict(modified_data[modified_data['Provider ID'] == selected_hospital_id][c.svm_features])
            st.write(f"New predicted rating for Hospital ID {selected_hospital_id}: {new_prediction[0]}")
        else:
            st.write("No modified data available. Please accept recommendations first.")
