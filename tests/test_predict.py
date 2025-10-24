import pandas as pd
import train_model
import predict


def test_predict_after_train(tmp_path):
    # create CSV with 3 symptom columns
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

    pipeline_file = tmp_path / "pipeline.pkl"
    bundle_file = tmp_path / "model_bundle.pkl"

    train_model.train(
        csv_path=str(csv_path),
        pipeline_file=str(pipeline_file),
        bundle_file=str(bundle_file),
        test_size=0.5,
        random_state=0,
    )

    # Load predictor from bundle
    p = predict.Predictor.from_bundle(str(bundle_file))
    # pass a valid vector
    label = p.predict_from_list([1, 0, 0])
    assert isinstance(label, str)
