"""Process disease-symptom dataset and create training data."""
import pandas as pd
import numpy as np

# Dataset based on Disease-Symptom Knowledge Database
# Source: https://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html
# and Kaggle datasets

DISEASE_SYMPTOMS = {
    "Common Cold": {
        "common": ["runny_nose", "congestion", "sneezing", "sore_throat", "cough", "fatigue"],
        "occasional": ["fever", "headache", "body_ache"],
        "rare": ["loss_of_smell", "eye_irritation"]
    },
    "Influenza": {
        "common": ["fever", "chills", "fatigue", "body_ache", "headache", "cough"],
        "occasional": ["runny_nose", "sore_throat", "nausea", "loss_of_appetite"],
        "rare": ["vomiting", "diarrhea"]
    },
    "COVID-19": {
        "common": ["fever", "cough", "fatigue", "loss_of_smell", "loss_of_taste"],
        "occasional": ["shortness_of_breath", "body_ache", "headache", "sore_throat"],
        "rare": ["diarrhea", "rash", "confusion"]
    },
    "Pneumonia": {
        "common": ["cough", "fever", "shortness_of_breath", "chest_pain", "fatigue"],
        "occasional": ["confusion", "headache", "muscle_pain", "nausea"],
        "rare": ["joint_pain", "diarrhea"]
    },
    "Tuberculosis": {
        "common": ["cough", "fever", "night_sweats", "weight_loss", "fatigue"],
        "occasional": ["chest_pain", "shortness_of_breath"],
        "rare": ["confusion", "headache"]
    },
    "Bronchial Asthma": {
        "common": ["wheezing", "shortness_of_breath", "chest_pain", "cough"],
        "occasional": ["fatigue", "anxiety"],
        "rare": ["body_ache", "headache"]
    },
    "Hypertension": {
        "common": ["headache", "dizziness", "chest_pain", "anxiety"],
        "occasional": ["fatigue", "confusion"],
        "rare": ["nausea", "vomiting"]
    },
    "Migraine": {
        "common": ["headache", "nausea", "eye_irritation", "dizziness"],
        "occasional": ["fatigue", "anxiety"],
        "rare": ["vomiting", "confusion"]
    },
    "GERD": {
        "common": ["chest_pain", "nausea", "bloating", "abdominal_pain"],
        "occasional": ["cough", "sore_throat"],
        "rare": ["wheezing", "vomiting"]
    },
    "Peptic Ulcer": {
        "common": ["abdominal_pain", "bloating", "nausea", "loss_of_appetite"],
        "occasional": ["vomiting", "weight_loss"],
        "rare": ["chest_pain", "fatigue"]
    },
    "Type 2 Diabetes": {
        "common": ["fatigue", "increased_thirst", "frequent_urination", "weight_loss"],
        "occasional": ["blurred_vision", "slow_healing"],
        "rare": ["headache", "dizziness"]
    },
    "Hypothyroidism": {
        "common": ["fatigue", "weight_gain", "cold_intolerance", "depression"],
        "occasional": ["joint_pain", "muscle_pain"],
        "rare": ["anxiety", "headache"]
    },
    "Major Depression": {
        "common": ["depression", "fatigue", "loss_of_appetite", "anxiety"],
        "occasional": ["weight_loss", "confusion"],
        "rare": ["headache", "body_ache"]
    },
    "Generalized Anxiety": {
        "common": ["anxiety", "fatigue", "muscle_tension", "sleep_problems"],
        "occasional": ["headache", "dizziness"],
        "rare": ["nausea", "chest_pain"]
    },
    "Rheumatoid Arthritis": {
        "common": ["joint_pain", "joint_swelling", "fatigue", "muscle_pain"],
        "occasional": ["fever", "weight_loss"],
        "rare": ["anxiety", "depression"]
    },
    "Chronic Sinusitis": {
        "common": ["congestion", "runny_nose", "facial_pain", "loss_of_smell"],
        "occasional": ["cough", "fatigue"],
        "rare": ["fever", "headache"]
    },
    "Seasonal Allergies": {
        "common": ["sneezing", "runny_nose", "eye_irritation", "congestion"],
        "occasional": ["cough", "fatigue"],
        "rare": ["headache", "sore_throat"]
    },
    "Celiac Disease": {
        "common": ["diarrhea", "abdominal_pain", "bloating", "fatigue"],
        "occasional": ["weight_loss", "nausea"],
        "rare": ["anxiety", "depression"]
    },
    "Irritable Bowel Syndrome": {
        "common": ["abdominal_pain", "bloating", "diarrhea", "constipation"],
        "occasional": ["nausea", "fatigue"],
        "rare": ["anxiety", "depression"]
    },
    "Multiple Sclerosis": {
        "common": ["fatigue", "dizziness", "muscle_weakness", "vision_problems"],
        "occasional": ["muscle_pain", "depression"],
        "rare": ["anxiety", "headache"]
    }
}

# Additional symptoms that can occur in multiple conditions
COMMON_SYMPTOMS = [
    "increased_thirst", "frequent_urination", "blurred_vision", "slow_healing",
    "weight_gain", "cold_intolerance", "depression", "sleep_problems",
    "muscle_tension", "joint_swelling", "facial_pain", "muscle_weakness",
    "vision_problems", "constipation"
]


def create_training_data(num_samples_per_disease=50):
    """Create synthetic training data based on disease-symptom relationships.
    Uses probability distributions to generate realistic symptom combinations."""
    
    # Get all unique symptoms
    all_symptoms = set()
    for disease in DISEASE_SYMPTOMS.values():
        for severity in ['common', 'occasional', 'rare']:
            all_symptoms.update(disease[severity])
    all_symptoms.update(COMMON_SYMPTOMS)
    all_symptoms = sorted(list(all_symptoms))
    
    # Create dataframe
    data = []
    
    for disease, symptoms in DISEASE_SYMPTOMS.items():
        for _ in range(num_samples_per_disease):
            # Initialize symptom vector
            symptom_vector = {s: 0 for s in all_symptoms}
            
            # Add common symptoms (80-100% chance)
            for s in symptoms['common']:
                if s in all_symptoms and np.random.random() > 0.2:
                    symptom_vector[s] = 1
                    
            # Add occasional symptoms (30-60% chance)
            for s in symptoms['occasional']:
                if s in all_symptoms and np.random.random() > 0.6:
                    symptom_vector[s] = 1
                    
            # Add rare symptoms (5-15% chance)
            for s in symptoms['rare']:
                if s in all_symptoms and np.random.random() > 0.85:
                    symptom_vector[s] = 1
                    
            # Add some random noise (1% chance for other symptoms)
            for s in all_symptoms:
                if symptom_vector[s] == 0 and np.random.random() > 0.99:
                    symptom_vector[s] = 1
            
            row = {'disease': disease, **symptom_vector}
            data.append(row)
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Create dataset with 50 samples per disease
    df = create_training_data(num_samples_per_disease=50)
    
    # Save to CSV
    df.to_csv("symptoms_diseases.csv", index=False)
    print(f"Created dataset with {len(df)} samples and {len(df.columns)-1} symptoms")
    print(f"Diseases: {sorted(df.disease.unique())}")