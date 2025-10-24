import tempfile
import os
import pickle
import pandas as pd
import train_model

tmp = tempfile.mkdtemp()
csv = os.path.join(tmp, "symptoms.csv")

pd.DataFrame(
    [["A", 1, 0, 0], ["B", 0, 1, 0], ["A", 1, 1, 0], ["B", 0, 0, 1]],
    columns=["disease", "s1", "s2", "s3"],
).to_csv(csv, index=False)

model = os.path.join(tmp, "model.pkl")
vocab = os.path.join(tmp, "vocab.pkl")
label = os.path.join(tmp, "label_enc.pkl")
pipeline = os.path.join(tmp, "pipeline.pkl")
bundle = os.path.join(tmp, "bundle.pkl")

try:
    train_model.train(csv, model, vocab, label, pipeline, 0.5, 0, True)
    print("train finished")
    print("bundle exists?", os.path.exists(bundle))
    if os.path.exists(bundle):
        with open(bundle, "rb") as f:
            b = pickle.load(f)
            print("loaded bundle ok", hasattr(b, "predict"))
except Exception:
    import traceback

    traceback.print_exc()
