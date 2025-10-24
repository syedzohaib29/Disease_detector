import pickle
import pandas as pd

import train_model


def test_train_creates_artifacts(tmp_path):
    # create a tiny CSV with at least two samples so train can run
    csv_path = tmp_path / "symptoms_diseases.csv"
    df = pd.DataFrame(
        [
            ["DiseaseA", 1, 0, 0],
            ["DiseaseB", 0, 1, 0],
            ["DiseaseA", 1, 1, 0],
            ["DiseaseB", 0, 0, 1],
        ],
        columns=["disease", "fever", "cough", "fatigue"],
    )
    df.to_csv(csv_path, index=False)

    pipeline_file = tmp_path / "pipeline.pkl"
    bundle_file = tmp_path / "model_bundle.pkl"

    # Run train with stratify enabled (dataset has >=2 per class)
    train_model.train(
        csv_path=str(csv_path),
        pipeline_file=str(pipeline_file),
        bundle_file=str(bundle_file),
        test_size=0.5,
        random_state=0,
        stratify_auto=True,
    )

    # bundle artifact should be created
    assert bundle_file.exists(), "bundle file was not created"
    # Load and sanity check bundle
    with open(str(bundle_file), "rb") as f:
        bundle = pickle.load(f)
    assert hasattr(bundle, "predict")
