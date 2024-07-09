import streamlit as st
import cleaned_code as c
from streamlit_extras.app_logo import add_logo
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

# Logo Awesome Inc.
st.image('img/Awesome_Inc (1).png', width=300)

# Title of the App
st.title("Hospital Recommendations")

# Load necessary data
hospital_info = c.hospital_info.copy()
print(hospital_info.columns)

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
st.dataframe(hospital_data_view.style.set_properties(**{
    'background-color': 'white',
    'color': 'black',
    'border-color': 'black'
}))

# Ensure hospital_data contains the necessary features for prediction
hospital_data = hospital_info[hospital_info['Provider ID'] == selected_hospital_id]

# Verify required features are present in the DataFrame
missing_features = [feature for feature in c.svm_features if feature not in hospital_data.columns]
if missing_features:
    st.error(f"The following features are missing: {missing_features}")
else:
    # Compute the current prediction for the selected hospital
    current_prediction = c.model.predict(hospital_data[c.svm_features])[0]
    st.metric(label=f"Current predicted rating for Hospital ID {selected_hospital_id}", value=current_prediction)

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
        modified_data, new_version = c.process_hospital_data(selected_hospital_id, hospital_info, feature_importances, c.model, c.version_history, c.hospital_info)
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
