"""Load trained bundle and predict disease from symptom flags.

Usage (CLI):
    python predict.py --bundle model_bundle.pkl --symptoms 1,0,0,1

Programmatic API:
    from predict import Predictor
    p = Predictor.from_bundle(bundle_path)
    pred = p.predict_from_list([1,0,0,1])
"""

from __future__ import annotations

import argparse
import pickle
from typing import List

import pandas as pd


class Predictor:
    """Load a single bundle (pipeline + vocab + label encoder)."""

    @classmethod
    def from_bundle(cls, bundle_path: str):
        with open(bundle_path, "rb") as f:
            obj = pickle.load(f)

        inst = cls.__new__(cls)
        # Backwards-compatible: older bundles were small containers with
        # attributes pipeline, vocab and le. Newer bundles are the pipeline
        # object itself (which exposes predict). Handle both.
        if hasattr(obj, "pipeline"):
            inst.model = obj.pipeline
            inst.vocab = getattr(obj, "vocab", None)
            inst.le = getattr(obj, "le", None)
        else:
            # assume obj is a pipeline
            inst.model = obj
            inst.vocab = None
            inst.le = None
        return inst

    def predict_from_list(self, flags: List[int]) -> str:
        """flags: list of 0/1 ints in same order as vocab"""
        # Determine vocab/feature names. Prefer stored vocab, else try to
        # extract from the pipeline's feature-order transformer.
        vocab = self.vocab
        if vocab is None:
            try:
                vocab = self.model.named_steps["order"].feature_names_
            except Exception:
                vocab = None

        if vocab is None:
            raise ValueError("Bundle has no recorded feature names (vocab)")

        if len(flags) != len(vocab):
            raise ValueError(
                "Expected {} flags, got {}".format(len(vocab), len(flags))
            )

        df = pd.DataFrame([flags], columns=vocab)
        # The pipeline's final estimator returns decoded labels directly
        # because LabelEncodedClassifier handles inverse transform.
        pred = self.model.predict(df)
        return pred[0]


def _build_parser():
    p = argparse.ArgumentParser(description="Predict disease from bundle")
    p.add_argument(
        "--bundle",
        required=True,
        help="Path to model bundle pickle",
    )
    p.add_argument(
        "--symptoms",
        required=True,
        help="Comma-separated 0/1 flags",
    )
    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()
    flags = [int(x) for x in args.symptoms.split(",")]
    predictor = Predictor.from_bundle(args.bundle)
    print(predictor.predict_from_list(flags))


if __name__ == "__main__":
    main()
