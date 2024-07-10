import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import cleaned_code as c
import json
import os
from streamlit_extras.app_logo import add_logo

# Directory containing PDF files
pdf_dir = c.BOOK_DIR

# Function to load and save file names
def load_saved_filenames():
    try:
        with open('saved_filenames.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_filenames(filenames):
    with open('saved_filenames.json', 'w') as f:
        json.dump(filenames, f)

# Get the current list of PDF file names
current_filenames = sorted([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])

# Load saved file names
saved_filenames = load_saved_filenames()

# Check for changes in the PDF directory
if current_filenames != saved_filenames:
    # Save the current file names
    save_filenames(current_filenames)

    # Rebuild the index
    c.build_index(pdf_dir, c.embeddings, c.INDEX_DIR)

# Logo Awesome Inc.
st.image('img/Awesome_Inc (1).png', width=600)

# Centered and enlarged title
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color: black;'>🏥 CareInnovate</h1>
    </div>
    """,
    unsafe_allow_html=True
)

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
    'Hospital overall rating'
]]
st.write(hospital_data_view)

# Ensure hospital_data contains the necessary features for prediction
hospital_data = hospital_info[hospital_info['Provider ID'] == selected_hospital_id]
print(f"Original df: {hospital_data}")

# Verify required features are present in the DataFrame
missing_features = [feature for feature in c.svm_features if feature not in hospital_data.columns]
if missing_features:
    st.error(f"The following features are missing: {missing_features}")
else:
    # Compute the current prediction for the selected hospital
    current_prediction = c.model.predict(hospital_data[c.svm_features])[0]
    st.markdown(
        f"""
        <div style='text-align: center;'>
            <h4>Current Rating: <span style='color: red;'>{current_prediction}</span></h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Desired rating selection
    desired_rating = st.slider('Select Desired Rating', 1, 5, 4)

    # Generate recommendations automatically when a hospital is selected
    feature_importances = c.feature_importances
    recommendations = c.generate_recommendations(selected_hospital_id, hospital_info, feature_importances, desired_rating)
    st.session_state.recommendations = recommendations
    st.session_state.feature_importances = feature_importances

    if st.session_state.recommendations:
        st.markdown(
            f"""
            <div style='text-align: center;'>
                <h3>Recommendations for {selected_hospital_name} (Hospital ID {selected_hospital_id})</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        recommendations = st.session_state.recommendations
        slider_values = {}
        for feature, details in recommendations.items():
            current_value = round(details['current_value'], 2)
            change = round(details['difference'], 2)
            new_value = round(details['recommended_value'], 2)
            max_value = round(current_value + 2 * change, 2)
            mid_value = round(current_value + change, 2)

            if feature not in st.session_state:
                st.session_state[feature] = mid_value

            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            col1.markdown(f"<div style='font-size: 20px;'>{feature}</div>", unsafe_allow_html=True)
            col2.markdown(f"<div style='text-align: center; font-size: 20px; color: red;'>{current_value}</div>", unsafe_allow_html=True)
            col3.markdown(f"<div style='text-align: center; font-size: 20px;'>{change}</div>", unsafe_allow_html=True)
            col4.markdown(f"<div style='text-align: center; font-size: 20px; color: green;'>{new_value}</div>", unsafe_allow_html=True)

            slider_values[feature] = st.slider(
                feature,
                min_value=current_value,
                max_value=max_value,
                value=st.session_state[feature],
                step=0.01,
                key=f"slider_{feature}"
            )
            st.session_state[feature] = slider_values[feature]

        # Umkehren des Dictionaries
        translated_to_original = {v: k for k, v in c.feature_name_mapping.items()}

        # Apply recommendations directly to the data
        new_data = hospital_data.copy()
        for feature, value in slider_values.items():
            # Übersetzen des Features zurück zum originalen Spaltennamen
            original_feature = translated_to_original.get(feature, feature)
            new_data.at[new_data.index[0], original_feature] = value

        st.session_state.new_data = new_data

        st.write("New data with recommendations applied:")
        st.write(new_data)  # Debugging output

        st.success("New values applied. You can now calculate the new prediction.")

    if st.button('Calculate New Prediction'):
        if 'new_data' in st.session_state:
            new_data = st.session_state.new_data.copy()
            # Drop the 'Hospital overall rating' column for prediction
            if 'Hospital overall rating' in new_data.columns:
                new_data1 = new_data.drop(columns=['Hospital overall rating'])
                new_data1 = new_data1.drop(columns=c.columns_to_drop)
            else:
                new_data1 = new_data.drop(columns=c.columns_to_drop)

            st.write("Data used for prediction:")
            st.write(new_data1[c.svm_features])  # Debugging output

            # Compute the new prediction for the modified hospital
            new_prediction = c.model.predict(new_data1[c.svm_features])[0]
            # Append the new prediction as 'Hospital overall rating'
            st.session_state.new_data['Hospital overall rating'] = new_prediction

            st.write("New data after prediction:")
            st.write(st.session_state.new_data)  # Debugging output

            st.markdown(
                f"""
                <div style='text-align: center;'>
                    <h4>New Rating: <span style='color: red;'>{new_prediction}</span></h4>
                </div>
                """,
                unsafe_allow_html=True
            )
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
