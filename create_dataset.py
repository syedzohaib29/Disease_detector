# create_dataset.py
import csv

# Define symptoms and sample diseases with symptom binary vectors
SYMPTOMS = [
    "fever",
    "cough",
    "sore_throat",
    "fatigue",
    "headache",
    "nausea",
    "vomiting",
    "diarrhea",
    "runny_nose",
    "body_ache",
    "shortness_of_breath",
    "loss_of_smell",
    "chest_pain",
    "rash",
]

# Each entry: (disease, [symptom flags in same order as SYMPTOMS])
DATA = [
    ("Common Cold", [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    ("Influenza (Flu)", [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    ("COVID-19", [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]),
    ("Gastroenteritis", [1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0]),
    ("Migraine", [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ("Pneumonia", [1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0]),
    ("Allergic Rhinitis", [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    ("Food Poisoning", [1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0]),
    ("Bronchitis", [1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    ("Chickenpox", [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]),
]

OUTPUT_CSV = "symptoms_diseases.csv"


def write_csv():
    header = ["disease"] + SYMPTOMS
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for disease, flags in DATA:
            writer.writerow([disease] + flags)
    print(f"Wrote {OUTPUT_CSV} with {len(DATA)} rows and {len(SYMPTOMS)} symptoms.")


if __name__ == "__main__":
    write_csv()

# create_dataset.py

# Define symptoms and sample diseases with symptom binary vectors
SYMPTOMS = [
    "fever",
    "cough",
    "sore_throat",
    "fatigue",
    "headache",
    "nausea",
    "vomiting",
    "diarrhea",
    "runny_nose",
    "body_ache",
    "shortness_of_breath",
    "loss_of_smell",
    "chest_pain",
    "rash",
]

# Each entry: (disease, [symptom flags in same order as SYMPTOMS])
DATA = [
    ("Common Cold", [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    ("Influenza (Flu)", [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    ("COVID-19", [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]),
    ("Gastroenteritis", [1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0]),
    ("Migraine", [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ("Pneumonia", [1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0]),
    ("Allergic Rhinitis", [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    ("Food Poisoning", [1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0]),
    ("Bronchitis", [1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    ("Chickenpox", [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]),
]

OUTPUT_CSV = "symptoms_diseases.csv"


def write_csv():
    header = ["disease"] + SYMPTOMS
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for disease, flags in DATA:
            writer.writerow([disease] + flags)
    print(f"Wrote {OUTPUT_CSV} with {len(DATA)} rows and {len(SYMPTOMS)} symptoms.")


if __name__ == "__main__":
    write_csv()
