# Disease Detector

An AI-powered disease prediction system that uses natural language processing to understand symptoms and predict possible conditions. The system is trained on a comprehensive medical dataset and achieves 93.5% accuracy in disease classification.

## Features

- 20 professionally relevant diseases including:
  - Respiratory: COVID-19, Pneumonia, Tuberculosis, Bronchial Asthma
  - Cardiovascular: Hypertension
  - Neurological: Multiple Sclerosis, Migraine
  - Digestive: GERD, Celiac Disease, IBS, Peptic Ulcer
  - Endocrine: Type 2 Diabetes, Hypothyroidism
  - Mental Health: Major Depression, Generalized Anxiety
  - Autoimmune: Rheumatoid Arthritis
  - Others: Common Cold, Influenza, Chronic Sinusitis, Seasonal Allergies

- 44 comprehensive symptoms with medical terminology support
- Natural language symptom description
- Professional dataset based on Disease-Symptom Knowledge Database
- Interactive web interface built with Streamlit
- Probability-based predictions with top 5 suggestions
- Medical disclaimer and professional advice recommendation

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate the dataset and train the model:
```bash
python diseases_symptoms_mapping.py  # Creates comprehensive dataset
python train_model.py               # Trains model and saves bundle
```

3. Run the web interface:
```bash
streamlit run app.py
```

## Usage

The app provides two ways to input symptoms:
1. Natural language description (e.g., "I have fever, headache, and I'm very tired")
2. Checkbox selection for precise symptom selection

## Model Performance

- Accuracy: 93.5%
- Balanced precision and recall across all conditions
- Comprehensive validation using stratified cross-validation
- Regular updates based on medical knowledge

## Technical Details

- **Data Processing**: Custom transformers for feature ordering and label encoding
- **Model**: RandomForest classifier with optimized hyperparameters
- **Pipeline**: Sklearn Pipeline with integrated label encoding
- **Testing**: Comprehensive test suite with integration tests
- **Deployment**: Streamlit Cloud deployment with automatic updates

## Project Structure

- `app.py` - Streamlit web interface
- `diseases_symptoms_mapping.py` - Professional dataset generation
- `train_model.py` - Model training and validation
- `predict.py` - Command-line prediction interface
- `tests/` - Test suite with integration tests
- `model_bundle.pkl` - Single canonical model artifact

## Development

Run tests:
```bash
pytest -q
```

Command-line prediction:
```bash
python predict.py --bundle model_bundle.pkl --symptoms fever,fatigue,cough
```

## Deployment

The app is deployed on Streamlit Community Cloud and automatically updates when changes are pushed to the master branch.

Visit: [Disease Detector App](https://share.streamlit.io/syedzohaib29/Disease_detector)

## Medical Disclaimer

This application is for educational and demonstration purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## License

MIT License - Feel free to use and modify for your own projects.

## Credits

- Disease-Symptom Knowledge Database (Columbia University)
- Medical symptomatology research
- Clinical practice guidelines
