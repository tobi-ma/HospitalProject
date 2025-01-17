import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import cleaned_code as c
import pandas as pd

# Centered and enlarged title
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color: black;'>🏥 CareInnovate</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Logo Awesome Inc.
st.image('img/Awesome_Inc (1).png', width=200)

# Load necessary data
hospital_info = c.hospital_info

# Use columns for the dropdowns
col1, col2, col3 = st.columns(3)

# State selection
with col1:
    states = hospital_info['State'].unique()
    selected_state = st.selectbox('Select State', states)

# City selection based on the selected state
with col2:
    cities = hospital_info[hospital_info['State'] == selected_state]['City'].unique()
    selected_city = st.selectbox('Select City', cities)

# Hospital selection based on the selected city
with col3:
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
st.markdown(
    f"""
    <div style='text-align: center;'>
        <h3 style='font-size: 18px;'>Hospital ID {selected_hospital_id}</h3>
        <h2 style='font-size: 24px;'>{selected_hospital_name}</h2>
    </div>
    """,
    unsafe_allow_html=True
)
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

    # Display current prediction
    st.markdown(
        f"""
        <div style='text-align: center;'>
            <h4>Current Rating: <span style='color: red;'>{current_prediction}</span></h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Exclude the selected hospital from the data to calculate min values
    other_hospitals = hospital_info[hospital_info['Provider ID'] != selected_hospital_id]

    # Calculate deviations from mean values
    deviations_mean = hospital_data[c.svm_features] - c.mean_values

    # Calculate deviations from minimum values excluding the selected hospital
    min_values_excluding_selected = other_hospitals[c.svm_features].min()
    deviations_min = hospital_data[c.svm_features] - min_values_excluding_selected

    # Calculate deviations below minimum values
    deviations_below_min = deviations_min[deviations_min < 0].dropna(axis=1)

    # Get the top 10 and bottom 10 deviations from mean
    deviations_mean_sorted = deviations_mean.T.sort_values(by=0)
    top_10_mean = deviations_mean_sorted.head(10)
    bottom_10_mean = deviations_mean_sorted.tail(10)

    # Create a checkbox for each feature below minimum value
    selected_features = list(deviations_below_min.columns)

    # If no features below minimum, show top 10 and bottom 10 mean deviations
    if deviations_below_min.empty:
        st.markdown("### Features Below Minimum Values")
        selected_features += list(deviations_below_min.columns)

    st.markdown("### Top 10 and Bottom 10 Deviations from Mean Values")
    selected_features += list(top_10_mean.index) + list(bottom_10_mean.index)

    # Remove duplicates from selected_features
    selected_features = list(set(selected_features))

    # Generate recommendations based on selected features
    recommendations = c.generate_recommendations(
        selected_hospital_id, hospital_info, c.feature_importances, current_prediction
    )

    if recommendations:
        st.markdown("### Recommendations")
        for feature, details in recommendations.items():
            st.write(f"**{feature}**: Increase from {details['current_value']} to {details['recommended_value']} (Difference: {details['difference']:.2f})")

    # Multiselect for the selected features
    selected_features = st.multiselect(
        "Select features to improve:",
        options=c.svm_features,
        default=recommendations.items(feature)
    )

    # Button to accept recommendations
    if st.button("Accept Recommendations"):
        feature_importances = c.feature_importances
        modified_data, new_version = c.process_hospital_data(
            selected_hospital_id, hospital_info, feature_importances, c.model, c.version_history, selected_features
        )
        st.session_state.modified_data = modified_data
        st.session_state.new_version = new_version
        st.success(f"Recommendations applied and new version {new_version} saved.")

    # Optional: Calculate the new prediction for the modified hospital
    if st.button("Calculate New Prediction"):
        if 'modified_data' in st.session_state:
            modified_data = st.session_state.modified_data
            new_prediction = c.model.predict(
                modified_data[modified_data['Provider ID'] == selected_hospital_id][c.svm_features]
            )
            st.write(f"New predicted rating for Hospital ID {selected_hospital_id}: {new_prediction[0]}")
        else:
            st.write("No modified data available. Please accept recommendations first.")

# Chatbot integration

# Load the documents
documents = c.load_pdf_files(c.BOOK_DIR)

memory = c.init_memory()

num_docs = 1
temperature = 0.8
length_instruction = "You can provide a detailed answer, but keep it concise."

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message(message["role"], avatar="🧑‍💻").write(message["content"])
    else:
        st.chat_message(message["role"], avatar="🤖").write(message["content"])

# React to user input
if prompt := st.chat_input("Enter your question:"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Retrieving answer..."):
        # Initialize the model with the selected temperature
        llm = c.initialize_llm(temperature)

        # Prompt
        template = f"""You are a nice chatbot having a conversation with a human. Answer the question based on the context and previous conversation, but feel free to provide additional information if needed. {length_instruction}

        Previous conversation:
        {{chat_history}}

        Context to answer question:
        {{context}}

        New human question: {{question}}
        Response:"""

        prompt_template = c.PromptTemplate(template=template, input_variables=["context", "question"])

        # Chain
        chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=c.vector_db.as_retriever(search_kwargs={"k": num_docs}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt_template}
        )

        # Send question to chain to get answer
        answer = chain({"question": prompt})

        # Extract answer from dictionary returned by chain
        response = answer["answer"]

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
            # Add a button to read the response aloud
            if st.button("Read Aloud", key=f"read_aloud_{len(st.session_state.messages)}"):
                c.subprocess.run(["say", response])

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display source documents in the sidebar with expander
        with st.sidebar:
            st.subheader("Context from Retrieved Documents")
            with st.expander("Show/Hide Context", expanded=True):
                for doc in answer['source_documents']:
                    filename = doc.metadata.get("filename", "Unknown file")
                    st.write(f"**Page {doc.metadata['page_num']} from {filename}:**")
                    st.write(doc.page_content[:200] + "...")
