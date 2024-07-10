import streamlit as st
import subprocess
import pandas as pd
import numpy as np
import fitz
import os
import warnings
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from xgboost.testing.data import joblib
from sklearn.pipeline import Pipeline
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pw
import long_texts

# Parameter für das Modell
params = {
    "C": 0.8895813093302738,
    "l1_ratio": 0.9803599632901754,
    "max_iter": 2000,
    "solver": 'saga',
    "penalty": 'elasticnet',
    "random_state": 42,
    "n_jobs": -1
}

# Import CSV files
hospital_info = pd.read_csv('hospital-info.csv')
not_yet_rated = pd.read_csv('not_yet_rated.csv')

# Remove demographic and unnecessary columns
columns_to_drop = [
    'Provider ID',
    'Hospital Name',
    'Address',
    'City',
    'State',
    'ZIP Code',
    'County Name',
    'Phone Number',
    'Hospital Ownership',
    'Emergency Services',
    'rating_group'
]

# Save ID data for later
id_data = hospital_info[['Hospital Name', 'Address']].copy()

# Keep only necessary columns for machine learning
ml_data = hospital_info.drop(columns=columns_to_drop)

# Convert hospital ratings to binary classification
binary_classification = False

if binary_classification:
    ml_data['Hospital overall rating'] = ml_data['Hospital overall rating'].apply(lambda x: 1 if x > 3 else 0)

# Separate predictors and target variable
X = ml_data.drop(columns=['Hospital overall rating'])
y = ml_data['Hospital overall rating']

# Split the data into train and test sets (70-30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Selected Features
svm_features = long_texts.svm_features

# Feature dictionary
feature_name_mapping = long_texts.feature_name_mapping

# Ensure that 'not_yet_rated' contains all the selected features
not_yet_rated_selected = not_yet_rated[svm_features]

# Load the ML model
hospital_pipeline = joblib.load('ml_model.pkl')


# Funktionen zur Versionsverwaltung
def load_version_history(file_path='data_versions.csv'):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=['Version', 'Reference_ID', 'Date', 'Configurations', 'Data'])


def save_version_history(version_history, file_path='data_versions.csv'):
    version_history.to_csv(file_path, index=False)


@st.cache_resource
def load_model(model_path='ml_model.pkl'):
    return joblib.load(model_path)


@st.cache_data
def load_data(data_path='not_yet_rated.csv'):
    return pd.read_csv(data_path)


# Generating recommendations
def generate_recommendations(hospital_id, hospital_info, feature_importances, desired_rating, top_n=5):
    # Calculate correlations
    correlations = hospital_info[svm_features].corrwith(hospital_info['Hospital overall rating'])

    # Select only positively correlated features
    positive_correlated_features = correlations[correlations > 0].index.tolist()

    # Get the top N important features that are positively correlated
    top_features = feature_importances[feature_importances['Feature'].isin(positive_correlated_features)].head(top_n)[
        'Feature'].values

    # Get the data for the specific hospital
    hospital_data = hospital_info[hospital_info['Provider ID'] == hospital_id]

    if hospital_data.empty:
        print(f"No data found for hospital ID {hospital_id}")
        return {}  # No recommendations possible if no data found

    # Compare with hospitals having higher ratings
    higher_rated_hospitals = hospital_info[hospital_info['Hospital overall rating'] >= desired_rating]

    if higher_rated_hospitals.empty:
        print("No higher rated hospitals found")
        return {}  # No recommendations possible if no higher rated hospitals found

    recommendations = {}

    for feature in top_features:
        if feature not in hospital_data.columns:
            print(f"Feature {feature} not found in hospital_data")
            continue

        # Get the mean value of the feature for higher-rated hospitals
        mean_value = higher_rated_hospitals[feature].mean()

        # Get the hospital's current value for the feature
        current_value = hospital_data[feature].values[0]

        # Generate recommendation only if the current value is lower than the mean value
        if current_value < mean_value:
            recommendations[feature_name_mapping.get(feature, feature)] = {
                'current_value': current_value,
                'recommended_value': mean_value,
                'difference': mean_value - current_value
            }

    return recommendations


# Anwenden der Empfehlungen auf den Datensatz
def apply_recommendations(hospital_id, not_yet_rated, recommendations):
    modified_data = not_yet_rated.copy()
    for feature, details in recommendations.items():
        modified_data.loc[modified_data['Provider ID'] == hospital_id, feature] = details['recommended_value']
    return modified_data


# Hauptfunktion zur Verarbeitung der Krankenhausdaten und Speicherung der neuen Version
def process_hospital_data(hospital_id, not_yet_rated, feature_importances, pipeline, version_history, hospital_info):
    recommendations = generate_recommendations(hospital_id, not_yet_rated, feature_importances, hospital_info)
    modified_data = apply_recommendations(hospital_id, not_yet_rated, recommendations)

    new_version = version_history['Version'].max() + 1 if not version_history.empty else 1
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    new_entry = {
        'Version': new_version,
        'Reference_ID': hospital_id,
        'Date': current_date,
        'Configurations': params,
        'Data': modified_data.to_dict()
    }
    version_history.loc[len(version_history)] = new_entry
    save_version_history(version_history)

    return modified_data, new_version


# Initialisierung
version_history = load_version_history()
model = load_model()
data = load_data()

# Berechnung der Feature-Importances
svm_features_length = len(svm_features)
coef_length = len(model.named_steps['regressor'].coef_[0])
if svm_features_length != coef_length:
    raise ValueError(
        f"Length mismatch: svm_features has {svm_features_length} elements, but coefficients array has {coef_length} elements.")

feature_importances = pd.DataFrame({
    'Feature': svm_features,
    'Importance': np.abs(model.named_steps['regressor'].coef_[0])
}).sort_values(by='Importance', ascending=False)

# Calculate mean and min values for hospital features
mean_values = hospital_info[svm_features].mean()
min_values = hospital_info[svm_features].min()

# Chatbot integration
# Constants
BOOK_DIR = './pdf'
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-l6-v2"
EMBEDDINGS_CACHE = './CACHE'
INDEX_DIR = "CACHE/faiss_index"
HF_TOKEN = pw.hf_token

# Suppress future warnings from the HuggingFace Hub
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')

# Set Hugging Face API token environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN


# Load the Hugging Face model and embeddings
def initialize_llm(temperature):
    return HuggingFaceEndpoint(repo_id=HF_MODEL, temperature=temperature)


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, cache_folder=EMBEDDINGS_CACHE)


# PDF Loader class
class PDFLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        documents = []
        with fitz.open(self.file_path) as pdf_document:
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                text = page.get_text("text")
                documents.append(
                    Document(page_content=text, metadata={"page_num": page_num, "filename": self.file_path}))
        return documents


# Load PDF files from a directory
@st.cache_data
def load_pdf_files(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            loader = PDFLoader(file_path)
            documents.extend(loader.load())
            print(f"Loaded file: {filename}")
    return documents

def build_index(directory, embeddings, index_dir):
    documents = load_pdf_files(directory)
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    vector_db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    # Save the index
    vector_db.save_local(index_dir)

    return vector_db


# Check if the index meta file exists
INDEX_META_FILE = os.path.join(INDEX_DIR, "index_meta.json")
if os.path.exists(INDEX_META_FILE):
    with open(INDEX_META_FILE, 'r') as f:
        indexed_files = set(json.load(f))
else:
    indexed_files = set()

# Check if the current files are already indexed
current_files = set(os.listdir(BOOK_DIR))
should_rebuild_index = current_files != indexed_files

if should_rebuild_index:
    vector_db = build_index(BOOK_DIR, embeddings, INDEX_DIR)

    # Update the index meta file
    with open(INDEX_META_FILE, 'w') as f:
        json.dump(list(current_files), f)
else:
    # Load the vector database
    vector_db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


# Memory
@st.cache_resource
def init_memory():
    return ConversationBufferMemory(
        memory_key='chat_history',
        output_key='answer',  # Explicitly set the output key
        return_messages=True
    )
