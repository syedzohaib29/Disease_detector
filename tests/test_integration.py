import pickle

import pandas as pd

import train_model
import predict


def test_train_and_predict_end_to_end(tmp_path):
    # Create a small CSV dataset
    csv_path = tmp_path / "symptoms.csv"
    df = pd.DataFrame(
        [
            ["A", 1, 0, 0],
            ["B", 0, 1, 0],
            ["A", 1, 1, 0],
            ["B", 0, 0, 1],
        ],
        columns=["disease", "s1", "s2", "s3"],
    )
    df.to_csv(csv_path, index=False)

    bundle_file = tmp_path / "model_bundle.pkl"

    # Train and write artifacts
    train_model.train(
        csv_path=str(csv_path),
        bundle_file=str(bundle_file),
        test_size=0.5,
        random_state=0,
    )

    # Bundle file should exist and be loadable
    assert bundle_file.exists()
    with open(str(bundle_file), "rb") as f:
        pipeline = pickle.load(f)

    # Use predictor helper to load the bundle and predict a single sample
    p = predict.Predictor.from_bundle(str(bundle_file))
    label = p.predict_from_list([1, 0, 0])
    assert isinstance(label, str)

    # Also ensure pipeline.predict returns a decoded label for a DataFrame
    df_sample = pd.DataFrame([[1, 0, 0]], columns=["s1", "s2", "s3"])
    pred = pipeline.predict(df_sample)
    assert isinstance(pred[0], str)
