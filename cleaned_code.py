# %%
import json
import os
import warnings
from datetime import datetime

import fitz
import pandas as pd
import numpy as np
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
import streamlit as st

import pw

# Import CSV files
hospital_info = pd.read_csv('hospital-info.csv')
not_yet_rated = pd.read_csv('not_yet_rated.csv')

# %%
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

# %%
# Save ID data for later
id_data = hospital_info[['Hospital Name', 'Address']].copy()

# %%
# Keep only necessary columns for machine learning
ml_data = hospital_info.drop(columns=columns_to_drop).copy()

# %%
# Convert hospital ratings to binary classification
binary_classification = False

if binary_classification:
    ml_data['Hospital overall rating'] = ml_data['Hospital overall rating'].apply(lambda x: 1 if x > 3 else 0)

# %%
# Separate predictors and target variable
X = ml_data.drop(columns=['Hospital overall rating'])
y = ml_data['Hospital overall rating']

# Ensure 'Hospital overall rating' is still in hospital_info
assert 'Hospital overall rating' in hospital_info.columns, "'Hospital overall rating' not found in hospital_info"

# %%
# Split the data into train and test sets (70-30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# %%
# Selected Features
svm_features = [
    'Mortality national comparison', 'Safety of care national comparison',
    'Readmission national comparison', 'Patient experience national comparison',
    'Effectiveness of care national comparison', 'Timeliness of care national comparison',
    'Efficient use of medical imaging national comparison', 'MORT_30_AMI_Score',
    'MORT_30_CABG_Score', 'MORT_30_COPD_Score', 'MORT_30_HF_Score',
    'MORT_30_PN_Score', 'MORT_30_STK_Score', 'READM_30_AMI_Score',
    'READM_30_CABG_Score', 'READM_30_COPD_Score', 'READM_30_HF_Score',
    'READM_30_HIP_KNEE_Score', 'READM_30_HOSP_WIDE_Score',
    'READM_30_PN_Score', 'READM_30_STK_Score', 'TIME_OP_21_Score',
    'TIME_OP_5_Score', 'EFF_EDV_Score', 'EFF_IMM_2_Score',
    'EFF_OP_20_Score', 'EFF_OP_22_Score', 'EFF_OP_4_Score',
    'EFF_PC_01_Score', 'EFF_STK_1_Score', 'EFF_STK_10_Score',
    'EFF_STK_2_Score', 'EFF_STK_4_Score', 'EFF_STK_5_Score',
    'EFF_STK_6_Score', 'EFF_VTE_1_Score', 'EFF_VTE_2_Score',
    'EFF_VTE_3_Score', 'EFF_VTE_5_Score', 'EFF_VTE_6_Score',
    'EXP_H_CLEAN_STAR_RATING_Score', 'EXP_H_COMP_1_STAR_RATING_Score',
    'EXP_H_COMP_2_STAR_RATING_Score', 'EXP_H_COMP_3_STAR_RATING_Score',
    'EXP_H_COMP_4_STAR_RATING_Score', 'EXP_H_COMP_5_STAR_RATING_Score',
    'EXP_H_COMP_6_STAR_RATING_Score', 'EXP_H_COMP_7_STAR_RATING_Score',
    'EXP_H_HSP_RATING_STAR_RATING_Score', 'EXP_H_QUIET_STAR_RATING_Score',
    'EXP_H_RECMND_STAR_RATING_Score', 'EXP_H_STAR_RATING_Score',
    'SAFETY_COMP_HIP_KNEE_Score', 'SAFETY_PSI_12_POSTOP_PULMEMB_DVT_Score',
    'SAFETY_PSI_13_POST_SEPSIS_Score', 'SAFETY_PSI_14_POSTOP_DEHIS_Score',
    'SAFETY_PSI_15_ACC_LAC_Score', 'SAFETY_PSI_3_ULCER_Score',
    'SAFETY_PSI_6_IAT_PTX_Score', 'SAFETY_PSI_7_CVCBI_Score',
    'SAFETY_PSI_90_SAFETY_Score', 'SAFETY_HAI_1_SIR_Score',
    'SAFETY_HAI_1a_SIR_Score', 'SAFETY_HAI_2_SIR_Score',
    'SAFETY_HAI_2a_SIR_Score', 'SAFETY_HAI_3_SIR_Score',
    'SAFETY_HAI_4_SIR_Score', 'SAFETY_HAI_5_SIR_Score',
    'SAFETY_HAI_6_SIR_Score', 'MED_OP_10_Score', 'MED_OP_11_Score',
    'MED_OP_13_Score', 'MED_OP_14_Score', 'MED_OP_8_Score',
    'MED_OP_9_Score'
]

# Feature dictionary

# Define a mapping dictionary for feature names
feature_name_mapping = {
    'MORT_30_AMI_Score': '30-day mortality rate for Acute Myocardial Infarction (heart attack)',
    'MORT_30_CABG_Score': '30-day mortality rate for Coronary Artery Bypass Graft surgery',
    'MORT_30_COPD_Score': '30-day mortality rate for Chronic Obstructive Pulmonary Disease',
    'MORT_30_HF_Score': '30-day mortality rate for Heart Failure',
    'MORT_30_PN_Score': '30-day mortality rate for Pneumonia',
    'MORT_30_STK_Score': '30-day mortality rate for Stroke',
    'READM_30_AMI_Score': '30-day readmission rate for Acute Myocardial Infarction',
    'READM_30_CABG_Score': '30-day readmission rate for Coronary Artery Bypass Graft surgery',
    'READM_30_COPD_Score': '30-day readmission rate for Chronic Obstructive Pulmonary Disease',
    'READM_30_HF_Score': '30-day readmission rate for Heart Failure',
    'READM_30_HIP_KNEE_Score': '30-day readmission rate for Hip/Knee replacement',
    'READM_30_HOSP_WIDE_Score': '30-day readmission rate hospital-wide',
    'READM_30_PN_Score': '30-day readmission rate for Pneumonia',
    'READM_30_STK_Score': '30-day readmission rate for Stroke',
    'EFF_EDV_Score': 'Efficiency of emergency department volume',
    'EFF_IMM_2_Score': 'Immunization measure',
    'EFF_OP_20_Score': 'Outpatient measure for a specific procedure',
    'EFF_OP_22_Score': 'Outpatient measure for a specific procedure',
    'EFF_OP_4_Score': 'Outpatient measure for a specific procedure',
    'EFF_PC_01_Score': 'Perinatal care measure',
    'EFF_STK_1_Score': 'Stroke care measure',
    'EFF_STK_2_Score': 'Stroke care measure',
    'EFF_STK_4_Score': 'Stroke care measure',
    'EFF_STK_5_Score': 'Stroke care measure',
    'EFF_STK_6_Score': 'Stroke care measure',
    'EFF_VTE_1_Score': 'Venous thromboembolism care measure',
    'EFF_VTE_2_Score': 'Venous thromboembolism care measure',
    'EFF_VTE_6_Score': 'Venous thromboembolism care measure',
    'EXP_H_CLEAN_STAR_RATING_Score': 'Star rating for hospital cleanliness',
    'EXP_H_COMP_1_STAR_RATING_Score': 'Star rating for hospital communications with patients (part 1)',
    'EXP_H_COMP_2_STAR_RATING_Score': 'Star rating for hospital communications with patients (part 2)',
    'EXP_H_COMP_3_STAR_RATING_Score': 'Star rating for hospital communications with patients (part 3)',
    'EXP_H_COMP_4_STAR_RATING_Score': 'Star rating for hospital communications with patients (part 4)',
    'EXP_H_COMP_5_STAR_RATING_Score': 'Star rating for hospital communications with patients (part 5)',
    'EXP_H_COMP_6_STAR_RATING_Score': 'Star rating for hospital communications with patients (part 6)',
    'EXP_H_COMP_7_STAR_RATING_Score': 'Star rating for hospital communications with patients (part 7)',
    'EXP_H_HSP_RATING_STAR_RATING_Score': 'Overall hospital star rating from patients',
    'EXP_H_QUIET_STAR_RATING_Score': 'Star rating for hospital quietness',
    'EXP_H_RECMND_STAR_RATING_Score': 'Star rating for patients’ willingness to recommend the hospital',
    'EXP_H_STAR_RATING_Score': 'Overall star rating for the hospital',
    'SAFETY_PSI_12_POSTOP_PULMEMB_DVT_Score': 'Postoperative pulmonary embolism or deep vein thrombosis rate',
    'SAFETY_PSI_13_POST_SEPSIS_Score': 'Postoperative sepsis rate',
    'SAFETY_PSI_14_POSTOP_DEHIS_Score': 'Postoperative wound dehiscence rate',
    'SAFETY_PSI_15_ACC_LAC_Score': 'Accidental puncture or laceration rate',
    'SAFETY_PSI_3_ULCER_Score': 'Pressure ulcer rate',
    'SAFETY_PSI_6_IAT_PTX_Score': 'Iatrogenic pneumothorax rate',
    'SAFETY_PSI_7_CVCBI_Score': 'Central venous catheter-related bloodstream infection rate',
    'SAFETY_PSI_90_SAFETY_Score': 'Composite patient safety indicator score',
    'SAFETY_HAI_1_SIR_Score': 'Standardized infection ratio for central line-associated bloodstream infections',
    'SAFETY_HAI_1a_SIR_Score': 'Additional standardized infection ratio for central line-associated bloodstream infections',
    'SAFETY_HAI_2_SIR_Score': 'Standardized infection ratio for catheter-associated urinary tract infections',
    'SAFETY_HAI_2a_SIR_Score': 'Additional standardized infection ratio for catheter-associated urinary tract infections',
    'SAFETY_HAI_3_SIR_Score': 'Standardized infection ratio for surgical site infections (colon surgery)',
    'SAFETY_HAI_4_SIR_Score': 'Standardized infection ratio for surgical site infections (abdominal hysterectomy)',
    'SAFETY_HAI_6_SIR_Score': 'Standardized infection ratio for hospital-onset Clostridium difficile infections',
    'MED_OP_10_Score': 'Medical operational score for a specific procedure (part 10)',
    'MED_OP_11_Score': 'Medical operational score for a specific procedure (part 11)',
    'MED_OP_13_Score': 'Medical operational score for a specific procedure (part 13)',
    'MED_OP_14_Score': 'Medical operational score for a specific procedure (part 14)',
    'MED_OP_8_Score': 'Medical operational score for a specific procedure (part 8)',
    'MED_OP_9_Score': 'Medical operational score for a specific procedure (part 9)'
}
# %%
X_train_selected = X_train[svm_features]
X_test_selected = X_test[svm_features]

save_new_model = False

params = {
    "C": 0.8895813093302738,
    "l1_ratio": 0.9803599632901754,
    "max_iter": 2000,
    "solver": 'saga',
    "penalty": 'elasticnet',
    "random_state": 42,
    "n_jobs": -1
}

if save_new_model:
    # Create pipeline
    hospital_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LogisticRegression(**params))
    ])

    # Train the best model on the full training data
    hospital_pipeline.fit(X_train_selected, y_train)

    # Save the trained model
    joblib.dump(hospital_pipeline, 'ml_model.pkl')

    accuracy = hospital_pipeline.score(X_test_selected, y_test)

    print(f"Model accuracy: {round(100 * accuracy, 2)}%")

# %%
# Ensure that 'not_yet_rated' contains all the selected features
not_yet_rated_selected = not_yet_rated[svm_features]

# Load the ML model
hospital_pipeline = joblib.load('ml_model.pkl')


# %%
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


# %%
# Generating recommendations
def generate_recommendations(hospital_id, hospital_info, feature_importances, desired_rating, top_n=5):
    # Get the top N important features
    top_features = feature_importances.head(top_n)['Feature'].values

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

        # Generate recommendation if the current value is lower than the mean value
        if current_value < mean_value:
            recommendations[feature_name_mapping.get(feature, feature)] = {
                'current_value': current_value,
                'recommended_value': mean_value,
                'difference': mean_value - current_value
            }

    return recommendations


# %%
# Anwenden der Empfehlungen auf den Datensatz
def apply_recommendations(hospital_id, not_yet_rated, recommendations):
    modified_data = not_yet_rated.copy()
    for feature, details in recommendations.items():
        modified_data.loc[modified_data['Provider ID'] == hospital_id, feature] = details['recommended_value']
    return modified_data


# %%
# Hauptfunktion zur Verarbeitung der Krankenhausdaten und Speicherung der neuen Version
def process_hospital_data(hospital_id, not_yet_rated, feature_importances, pipeline, version_history, desired_rating):
    recommendations = generate_recommendations(hospital_id, not_yet_rated, feature_importances, desired_rating)
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

# %%
# Berechnung der Feature-Importances
feature_importances = pd.DataFrame({
    'Feature': svm_features,  # Verwende die Namen der ausgewählten Features
    'Importance': np.abs(model.named_steps['regressor'].coef_[0])
}).sort_values(by='Importance', ascending=False)

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
    documents = load_pdf_files(BOOK_DIR)
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    vector_db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    # Save the index
    vector_db.save_local(INDEX_DIR)

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
        return_messages=True)

