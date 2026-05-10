import math

import numpy as np
import pandas as pd

from . import filters

def weak_classifier(
    string_pair_df: pd.DataFrame, threshold: int, filter_str: str
) -> list:
    filter = filters.filter_to_vector(filter_str)

    items_1 = np.array(string_pair_df["Item 1"].tolist())
    items_2 = np.array(string_pair_df["Item 2"].tolist())

    M_xa = np.multiply(items_1.astype(int), filter)
    M_xb = np.multiply(items_2.astype(int), filter)

    M_f_xa = np.sum(M_xa, axis=1)
    M_f_xb = np.sum(M_xb, axis=1)

    M_f_xa_t_non_bin = M_f_xa - threshold * np.ones(len(M_f_xa))
    M_f_xb_t_non_bin = M_f_xb - threshold * np.ones(len(M_f_xb))

    M_f_xa_t = np.where(M_f_xa_t_non_bin > 0, 1, -1)
    M_f_xb_t = np.where(M_f_xb_t_non_bin > 0, 1, -1)

    # Calculate element-wise product
    predictions = M_f_xa_t * M_f_xb_t

    return predictions


def normalize_weight(ground_truth: list, updated_weights: list) -> list:
    mask = ground_truth == 1

    sum_values_to_normalize = np.sum(updated_weights[mask])

    updated_weights[mask] = updated_weights[mask] / sum_values_to_normalize

    return updated_weights


def weight_update(
    df: pd.DataFrame,
    weights: list,
    best_filter: str,
    best_threshold: int,
    confidence: float,
) -> list:
    # Keep only 1s in the ground truth matrix
    ground_truth = df["Equality"].to_list()
    ground_truth_matrix = np.array(ground_truth).reshape(-1, 1)

    weights = weights.reshape(-1, 1)

    ground_truth_matrix[ground_truth_matrix == -1] = 0

    predictions = weak_classifier(df, best_threshold, best_filter)
    prediction_matrix = np.array(predictions).reshape(-1, 1)

    prediction_matrix[prediction_matrix == 1] = 0

    matching_mispredicted_pairs = (ground_truth_matrix * prediction_matrix).astype(int)

    matching_mispredicted_pairs[matching_mispredicted_pairs == -1] = 1

    confidence_vector = (math.exp(confidence) * np.ones(len(weights))).reshape(-1, 1)

    updated_weights = (
        matching_mispredicted_pairs * weights * confidence_vector
        + np.logical_not(matching_mispredicted_pairs).astype(int) * weights
    )

    normalized_updated_weights = updated_weights / np.sum(updated_weights)

    return normalized_updated_weights
