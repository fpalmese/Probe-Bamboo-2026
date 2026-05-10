import math

import numpy as np
import pandas as pd

from . import classifier

def get_confidence(errors: dict, best_filter: str, best_threshold: str) -> tuple:
    min_error = errors[(best_filter, best_threshold)]

    if min_error == 0:
        min_error = 10**-20

    confidence = math.log(
        (1 - min_error) / min_error
    )  # confidence of the weak classifier

    return min_error, confidence


def matrix_error(
    string_pair_df: pd.DataFrame,
    thresholds: list,
    filter: str,
    weights: list,
) -> dict:
    weights = weights.reshape(-1, 1)
    errors_dict = {}

    for threshold in thresholds:
        error = 0

        predictions = classifier.weak_classifier(string_pair_df, threshold, filter)

        predictions = predictions.reshape(-1, 1)

        ground_truth = np.array(string_pair_df["Equality"].to_list())

        ground_truth = ground_truth.reshape(-1, 1)

        errors = np.not_equal(predictions, ground_truth).astype(int)

        error = np.sum(errors * weights)

        errors_dict[(f"{filter}", threshold)] = error

    return errors_dict
