import numpy as np
from xgboost.testing.data import joblib

# long_texts.py

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
    'MED_OP_9_Score': 'Medical operational score for a specific procedure (part 9)',
    'Mortality national comparison': 'A comparison of a hospital\'s mortality rate to the national average for similar institutions.',
    'Safety of care national comparison': 'A comparison of a hospital\'s safety of care metrics to the national average for similar institutions.',
    'Readmission national comparison': 'A comparison of a hospital\'s readmission rates to the national average for similar institutions.',
    'Patient experience national comparison': 'A comparison of patient satisfaction scores for a hospital to the national average.',
    'Effectiveness of care national comparison': 'A comparison of the effectiveness of treatments and interventions at a hospital to the national average.',
    'Timeliness of care national comparison': 'A comparison of the speed and efficiency of care provided by a hospital to the national average.',
    'Efficient use of medical imaging national comparison': 'A comparison of a hospital\'s use of medical imaging tests to the national average, evaluating efficiency and necessity.',
    'TIME_OP_21_Score': 'Score related to the timeliness of outpatient surgery procedures.',
    'TIME_OP_5_Score': 'Score related to the timeliness of outpatient diagnostic tests.',
    'EFF_STK_10_Score': 'Stroke care efficiency score, measuring adherence to best practices for stroke treatment.',
    'EFF_VTE_3_Score': 'Venous thromboembolism care efficiency score, measuring adherence to best practices for VTE prevention and treatment.',
    'EFF_VTE_5_Score': 'Venous thromboembolism care efficiency score, measuring outcomes and adherence to treatment protocols.',
    'SAFETY_COMP_HIP_KNEE_Score': 'Composite safety score for hip and knee replacement surgeries.',
    'SAFETY_HAI_5_SIR_Score': 'Standardized infection ratio score for hospital-acquired infections related to surgical site infections.'
}

output = False

if output:
    def load_model(model_path='ml_model.pkl'):
        return joblib.load(model_path)

    model = load_model()
    coef_length = len(model.named_steps['regressor'].coef_[0])
    print(coef_length)
    print(len(svm_features))
    print(len(feature_name_mapping))

    coeffs = np.abs(model.named_steps['regressor'].coef_[0])

    # Anzahl der Elemente in den Listen
    print(f"Anzahl der SVM-Features: {len(svm_features)}")
    print(f"Anzahl der Koeffizienten: {len(coeffs)}")

    # Testen, ob die Längen übereinstimmen
    if len(svm_features) != len(coeffs):
        print("Die Anzahl der Features und Koeffizienten stimmen nicht überein.")
    else:
        print("Die Anzahl der Features und Koeffizienten stimmen überein.")

    # Prüfen, ob alle Features eine Beschreibung haben
    if all(feature in feature_name_mapping for feature in svm_features):
        print("Jedes Feature hat eine Beschreibung.")
    else:
        missing_descriptions = [feature for feature in svm_features if feature not in feature_name_mapping]
        print(f"Folgende Features haben keine Beschreibung: {missing_descriptions}")

    # Erstellen eines DataFrames zur besseren Darstellung
    import pandas as pd

    df = pd.DataFrame({
        'Feature': svm_features,
        'Description': [feature_name_mapping.get(feature) for feature in svm_features],
        'Coefficient': coeffs
    })

    # Anzeigen des DataFrames
    print(df)
