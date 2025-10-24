# train_model.py
"""Train a RandomForest model from a symptoms CSV and save artifacts.

Improvements vs previous version:
- CLI args (csv path, output files, test_size, random_state)
- Logging instead of raw prints
- Automatic stratify handling: only stratify when every class has >= 2 samples
- Robust classification_report usage (avoids mismatched target_names)
"""

from typing import Optional
import argparse
import logging
import pickle
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


logger = logging.getLogger(__name__)


CSV_FILE_DEFAULT = "symptoms_diseases.csv"
MODEL_FILE_DEFAULT = "model.pkl"
VOCAB_FILE_DEFAULT = "symptoms_vocab.pkl"
LABEL_ENCODER_FILE_DEFAULT = "label_encoder.pkl"
PIPELINE_FILE_DEFAULT = "pipeline.pkl"
BUNDLE_FILE_DEFAULT = "model_bundle.pkl"


class FeatureOrderTransformer(BaseEstimator, TransformerMixin):
    """Transformer that records the feature (column) order from a DataFrame and
    converts incoming DataFrames to numpy arrays in that order.

    This lets the pipeline remember the column order so we don't need to save a
    separate `vocab` file.
    """

    def fit(self, X, y=None):
        # Expect X to be a pandas DataFrame
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        else:
            # If X is already an array, we can't infer names; set empty list
            self.feature_names_ = []
        return self

    def transform(self, X):
        # If we have feature names, re-order/select them from DataFrame
        if hasattr(X, "loc") and getattr(self, "feature_names_", None):
            return X[self.feature_names_].values
        # Otherwise assume X is already numeric array-like
        return X


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a classifier and a LabelEncoder so the estimator fits on raw labels
    and predicts decoded labels. Implements sklearn estimator API so it can
    be part of a Pipeline.
    """

    def __init__(self, base_clf=None):
        self.base_clf = base_clf

    def fit(self, X, y):
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        # clone underlying classifier if needed (assume provided instance is OK)
        self.base_clf.fit(X, y_enc)
        return self

    def predict(self, X):
        y_enc = self.base_clf.predict(X)
        return self.le_.inverse_transform(y_enc)

    def predict_proba(self, X):
        return self.base_clf.predict_proba(X)



def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    if "disease" not in df.columns:
        raise ValueError("CSV must contain a 'disease' column")
    X = df.drop("disease", axis=1)
    y = df["disease"]
    return X, y


def train(
    csv_path: str = CSV_FILE_DEFAULT,
    model_file: str = MODEL_FILE_DEFAULT,
    vocab_file: str = VOCAB_FILE_DEFAULT,
    label_encoder_file: str = LABEL_ENCODER_FILE_DEFAULT,
    pipeline_file: str = PIPELINE_FILE_DEFAULT,
    bundle_file: str = BUNDLE_FILE_DEFAULT,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_auto: bool = True,
) -> None:
    """Train and save model + artifacts.

    stratify_auto: when True, will enable stratification only if every class
    has at least 2 samples. For tiny toy datasets this avoids sklearn errors.
    """

    X, y = load_data(csv_path)

    stratify_param: Optional[np.ndarray] = None
    if stratify_auto:
        counts = pd.Series(y).value_counts()
        min_count = counts.min()
        if min_count >= 2:
            stratify_param = y
            logger.info(
                "Using stratified split (min class count=%d)", min_count
            )
        else:
            logger.warning(
                "Not stratifying because min class count=%d < 2", min_count
            )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param,
    )

    # Build a pipeline that records feature order and also encodes labels.
    base_clf = RandomForestClassifier(
        n_estimators=200, random_state=random_state
    )
    pipeline = Pipeline(
        [
            ("order", FeatureOrderTransformer()),
            ("clf", LabelEncodedClassifier(base_clf)),
        ]
    )

    # Fit pipeline on raw string labels; LabelEncodedClassifier wraps the
    # label encoding internally.
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Accuracy: %.4f", acc)

    # Use labels present in test set to avoid mismatch with full label list
    present_labels = np.unique(y_test)
    report = classification_report(
        y_test, y_pred, labels=present_labels, zero_division=0
    )
    logger.info("Classification Report:\n%s", report)

    # Save a single canonical artifact (the pipeline): it contains the
    # feature-order transformer and the label-encoding classifier so a single
    # file is sufficient for inference.
    try:
        with open(bundle_file, "wb") as f:
            pickle.dump(pipeline, f)
        logger.info("Saved bundle -> %s", bundle_file)
    except Exception as exc:
        logger.exception(
            "Failed to save model bundle %s: %s", bundle_file, exc
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train symptom->disease model")
    p.add_argument(
        "--csv",
        default=CSV_FILE_DEFAULT,
        help="Path to symptoms CSV",
    )
    p.add_argument(
        "--model",
        default=MODEL_FILE_DEFAULT,
        help="Output model file (pickle)",
    )
    p.add_argument(
        "--vocab",
        default=VOCAB_FILE_DEFAULT,
        help="Output vocab file (pickle)",
    )
    p.add_argument(
        "--label-enc",
        default=LABEL_ENCODER_FILE_DEFAULT,
        help="Output label encoder file (pickle)",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    p.add_argument(
        "--no-stratify",
        dest="stratify_auto",
        action="store_false",
        help="Disable automatic stratification (even when possible)",
    )
    p.add_argument(
        "--pipeline",
        default=PIPELINE_FILE_DEFAULT,
        help="Output pipeline file (pickle)",
    )
    p.add_argument(
        "--bundle",
        default=BUNDLE_FILE_DEFAULT,
        help="Output single model bundle file (pickle)",
    )
    return p


def main(argv: Optional[list] = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    train(
        csv_path=args.csv,
        model_file=args.model,
        vocab_file=args.vocab,
        label_encoder_file=args.label_enc,
        pipeline_file=args.pipeline,
        bundle_file=args.bundle,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify_auto=args.stratify_auto,
    )


if __name__ == "__main__":
    main()
