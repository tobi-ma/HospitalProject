import streamlit as st
import cleaned_code as c
from streamlit_extras.app_logo import add_logo

# Logo Awesome Inc.
st.image('img/Awesome_Inc (1).png', width=300)
# Titel der App
st.title("Hospital Data Improvement Recommendations")
# add_logo('img/Awesome_Inc.png')

# Auswahl eines Krankenhauses
hospital_id = st.selectbox("Select Hospital ID", c.not_yet_rated['Provider ID'].unique())

# Anzeige der aktuellen Krankenhausdaten
st.subheader(f"Current Data for Hospital ID {hospital_id}")
hospital_data = c.hospital_info[c.hospital_info['Provider ID'] == hospital_id]
st.write(hospital_data)

# Berechne die aktuelle Bewertung des ausgewählten Krankenhauses
current_prediction = c.model.predict(hospital_data[c.svm_features])[0]
st.write(f"Current predicted rating for Hospital ID {hospital_id}: {current_prediction}")

# Generiere und zeige Empfehlungen an
recommendations = c.generate_recommendations(hospital_id, c.data, c.feature_importances, c.hospital_info)

st.subheader("Improvement Recommendations")
if not recommendations:
    st.write("No recommendations available.")
else:
    for feature, details in recommendations.items():
        st.write(f"- **{feature}**: Increase from {details['current_value']} to {details['recommended_value']} (Difference: {details['difference']:.2f})")

# Button zum Akzeptieren der Empfehlungen
if st.button("Accept Recommendations"):
    modified_data, new_version = c.process_hospital_data(hospital_id, c.data, c.feature_importances, c.model, c.version_history, c.hospital_info)
    st.session_state.modified_data = modified_data
    st.session_state.new_version = new_version
    st.success(f"Recommendations applied and new version {new_version} saved.")

# Optional: Berechne die neue Vorhersage für das modifizierte Krankenhaus
if st.button("Calculate New Prediction"):
    if 'modified_data' in st.session_state:
        modified_data = st.session_state.modified_data
        new_prediction = c.model.predict(modified_data[modified_data['Provider ID'] == hospital_id][c.svm_features])
        st.write(f"New predicted rating for Hospital ID {hospital_id}: {new_prediction[0]}")
    else:
        st.write("No modified data available. Please accept recommendations first.")
