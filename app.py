# app.py
import streamlit as st
import pickle
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

BUNDLE_FILE = "model_bundle.pkl"

# Configure Streamlit page
st.set_page_config(
    page_title="Symptom → Disease Chatbot",
    layout="centered",
)


class FeatureOrderTransformer(BaseEstimator, TransformerMixin):
    """Transformer that records the feature (column) order from a DataFrame and
    converts incoming DataFrames to numpy arrays in that order.
    """
    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = []
        return self

    def transform(self, X):
        if hasattr(X, "loc") and getattr(self, "feature_names_", None):
            return X[self.feature_names_].values
        return X


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a classifier and a LabelEncoder so the estimator fits on raw labels
    and predicts decoded labels.
    """
    def __init__(self, base_clf=None):
        self.base_clf = base_clf

    def fit(self, X, y):
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.base_clf.fit(X, y_enc)
        return self

    def predict(self, X):
        y_enc = self.base_clf.predict(X)
        return self.le_.inverse_transform(y_enc)

    def predict_proba(self, X):
        return self.base_clf.predict_proba(X)


@st.cache_data
def load_pipeline():
    """Load the canonical pipeline bundle. Backwards-compatible with older
    bundle objects that wrapped the pipeline as `.pipeline`.
    """
    with open(BUNDLE_FILE, "rb") as f:
        obj = pickle.load(f)

    if hasattr(obj, "pipeline"):
        pipeline = obj.pipeline
    else:
        pipeline = obj

    # Extract vocab/feature names from the 'order' transformer if present
    try:
        vocab = pipeline.named_steps["order"].feature_names_
    except Exception:
        vocab = None

    # Extract label encoder from classifier (LabelEncodedClassifier stores le_)
    try:
        le = pipeline.named_steps["clf"].le_
    except Exception:
        le = None

    return pipeline, vocab, le


model, SYMPTOMS, label_encoder = load_pipeline()

# synonym mapping to normalize user words to vocab items
SYNONYMS = {
    "fever": [
        "fever", "temperature", "high temperature", "hot", "feverish"
    ],
    "chills": [
        "chills", "feeling cold", "shivering", "rigors", "cold sweats"
    ],
    "cough": [
        "cough", "coughing", "dry cough", "wet cough", "hacking"
    ],
    "sore_throat": [
        "sore throat", "throat pain", "throat hurts",
        "painful throat", "throat irritation"
    ],
    "fatigue": [
        "tired", "fatigue", "exhausted", "no energy", "weakness",
        "lethargic", "exhaustion", "tired all the time"
    ],
    "headache": [
        "headache", "head pain", "head hurts", "head pounding",
        "migraine", "head throbbing"
    ],
    "nausea": [
        "nausea", "nauseous", "queasy", "sick to stomach", "feel sick",
        "want to throw up"
    ],
    "vomiting": [
        "vomit", "vomiting", "throw up", "throwing up", "getting sick",
        "being sick"
    ],
    "diarrhea": [
        "diarrhea", "loose stool", "runny stool", "watery stool",
        "frequent bowel movements"
    ],
    "runny_nose": [
        "runny nose", "drippy nose", "nasal discharge", "nose running"
    ],
    "congestion": [
        "congestion", "stuffy nose", "blocked nose",
        "nasal congestion", "sinus congestion", "stuffed up"
    ],
    "body_ache": [
        "body ache", "body aches", "aching", "body pain", "general pain",
        "everything hurts"
    ],
    "muscle_pain": [
        "muscle pain", "muscular pain", "sore muscles",
        "muscle aches", "myalgia", "muscles hurt"
    ],
    "muscle_weakness": [
        "muscle weakness", "weak muscles", "loss of strength",
        "muscles feel weak", "decreased strength"
    ],
    "joint_pain": [
        "joint pain", "painful joints", "arthralgia", "joints hurt",
        "joint aches", "sore joints"
    ],
    "joint_swelling": [
        "joint swelling", "swollen joints", "puffy joints",
        "inflamed joints"
    ],
    "shortness_of_breath": [
        "shortness of breath", "breathless", "difficulty breathing",
        "hard to breathe", "dyspnea", "can't catch breath"
    ],
    "wheezing": [
        "wheezing", "wheeze", "whistling breath", "noisy breathing",
        "breath sounds"
    ],
    "loss_of_smell": [
        "loss of smell", "can't smell", "no smell", "anosmia",
        "reduced smell", "decreased smell"
    ],
    "loss_of_taste": [
        "loss of taste", "can't taste", "no taste",
        "food tasteless", "ageusia", "decreased taste"
    ],
    "chest_pain": [
        "chest pain", "pain in chest", "chest tightness",
        "chest pressure", "chest discomfort"
    ],
    "facial_pain": [
        "facial pain", "face pain", "face ache",
        "pain in face", "facial discomfort"
    ],
    "vision_problems": [
        "vision problems", "blurry vision", "trouble seeing",
        "visual disturbance", "sight issues"
    ],
    "eye_irritation": [
        "eye irritation", "itchy eyes", "watery eyes",
        "red eyes", "eye redness", "burning eyes"
    ],
    "dizziness": [
        "dizziness", "dizzy", "lightheaded", "vertigo", "unsteady",
        "room spinning"
    ],
    "confusion": [
        "confusion", "confused", "disoriented", "mental fog",
        "brain fog", "can't think clearly"
    ],
    "depression": [
        "depression", "feeling down", "depressed", "hopeless",
        "no interest", "sadness"
    ],
    "anxiety": [
        "anxiety", "anxious", "worried", "nervous", "panic",
        "stress", "on edge"
    ],
    "sleep_problems": [
        "sleep problems", "can't sleep", "insomnia", "trouble sleeping",
        "poor sleep", "restless sleep"
    ],
    "increased_thirst": [
        "increased thirst", "very thirsty", "excessive thirst",
        "drinking lots of water", "polydipsia"
    ],
    "frequent_urination": [
        "frequent urination", "peeing a lot", "urinating often",
        "polyuria", "bathroom frequently"
    ],
    "abdominal_pain": [
        "abdominal pain", "stomach pain", "tummy pain",
        "belly pain", "stomach ache", "gut pain"
    ],
    "bloating": [
        "bloating", "bloated", "swollen abdomen", "distended stomach",
        "stomach swelling"
    ],
    "constipation": [
        "constipation", "can't poop", "hard stools",
        "difficulty passing stool", "irregular bowel movements"
    ],
    "loss_of_appetite": [
        "loss of appetite", "no appetite", "don't want to eat",
        "reduced appetite", "poor appetite", "not hungry"
    ],
    "weight_loss": [
        "weight loss", "losing weight", "dropped weight",
        "unexplained weight loss", "getting thinner"
    ],
    "weight_gain": [
        "weight gain", "gaining weight", "unexplained weight gain",
        "getting heavier"
    ],
    "night_sweats": [
        "night sweats", "sweating at night", "nocturnal sweating",
        "waking up sweating"
    ],
    "slow_healing": [
        "slow healing", "wounds heal slowly", "cuts take long to heal",
        "poor wound healing"
    ],
    "blurred_vision": [
        "blurred vision", "blurry vision", "fuzzy vision",
        "can't see clearly"
    ],
    "muscle_tension": [
        "muscle tension", "tense muscles", "tight muscles",
        "muscle stiffness"
    ],
    "cold_intolerance": [
        "cold intolerance", "always cold", "sensitive to cold",
        "can't get warm"
    ]
}

# Build reverse lookup to map token -> symptom_key
TOKEN_TO_SYMPTOM = {}
for key, synonyms in SYNONYMS.items():
    for s in synonyms:
        TOKEN_TO_SYMPTOM[s] = key


def extract_symptoms(text):
    """
    Very simple symptom extractor: lowercases text and searches for synonyms.
    Returns a set of symptom keys that match.
    """
    text = text.lower()
    found = set()
    # Check multi-word synonyms first
    sorted_syns = sorted(TOKEN_TO_SYMPTOM.keys(), key=lambda x: -len(x))
    for syn in sorted_syns:
        if syn in text:
            found.add(TOKEN_TO_SYMPTOM[syn])
            # remove matched part to avoid double-matching
            text = text.replace(syn, " ")
    # fallback: split into words and match single tokens
    words = re.findall(r"[a-zA-Z']+", text)
    for w in words:
        if w in TOKEN_TO_SYMPTOM:
            found.add(TOKEN_TO_SYMPTOM[w])
    return found


def symptoms_to_vector(found_symptoms, vocab):
    """Return a feature vector in same column order as vocab.

    Returns a numpy array shaped (1, n_features).
    """
    vect = [1 if s in found_symptoms else 0 for s in vocab]
    return np.array(vect).reshape(1, -1)


# Streamlit UI

st.title("Symptom-based Disease Prediction (Demo)")

st.markdown(
    "Type your symptoms (plain text). Example: *I have fever and sore throat,"
    " and I'm very tired.*",
)

with st.form("symptom_form"):
    user_input = st.text_area(
        "Describe your symptoms:",
        height=120,
        placeholder="e.g., fever, cough, tiredness...",
    )

    include_checkboxes = st.checkbox(
        "Or select symptoms manually",
        value=False,
    )
    manual = {}
    if include_checkboxes:
        cols = st.columns(2)
        for i, s in enumerate(SYMPTOMS):
            col = cols[i % 2]
            manual[s] = col.checkbox(s.replace("_", " ").capitalize())
    submitted = st.form_submit_button("Predict")

if submitted:
    found = extract_symptoms(user_input) if user_input.strip() else set()
    # add manual selections
    if include_checkboxes:
        for s, val in manual.items():
            if val:
                found.add(s)

    if not found:
        st.warning(
            "No symptom matched. Try different wording or"
            " select symptoms manually."
        )
    else:
        pretty = [s.replace("_", " ") for s in sorted(found)]
        st.info(f"Detected symptoms: {', '.join(pretty)}")

        x = symptoms_to_vector(found, SYMPTOMS)
        probs = model.predict_proba(x)[0]
        top_idx = np.argsort(probs)[::-1][:5]

        # Label encoder lives inside the pipeline's classifier
        # and stores classes_ in label_encoder.classes_
        le = label_encoder
        st.subheader("Predictions (top suggestions)")
        for i, idx in enumerate(top_idx):
            label = le.inverse_transform([idx])[0]
            st.write(
                f"{i+1}. **{label}** — probability {probs[idx]:.2f}"
            )

        st.markdown("---")
        st.markdown(
            "**Disclaimer:** This is a demo model trained on a small"
            " synthetic dataset."
        )
        st.markdown(
            "Not medical advice. See a healthcare professional for diagnosis."
        )
