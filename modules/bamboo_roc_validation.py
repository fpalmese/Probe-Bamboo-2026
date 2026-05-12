import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd
from modules.utils.validation_utils import generateStringPairDf, hamming_distance, parse_bamboo_csv, plot_roc_curves_from_files, calculate_single_fprint, hamming_distance_real


def get_bamboo_validation_data(bin_0_df, validation_pairs_df, bamboo_log_csv, bits_set = [8,16,32,64], hamming=False, roc_save_path=None, show_plot=False):
    string_df = bin_0_df.copy()
    balanced_pairs_df = validation_pairs_df.copy()
    balanced_pairs_df.drop_duplicates(inplace=True)
    balanced_pairs_df.reset_index(drop=True, inplace=True)

    # ensure bamboo log CSV exists
    if not os.path.exists(bamboo_log_csv):
        print(f"Bamboo log CSV not found: {bamboo_log_csv}. Returning empty best-tau DataFrame.")
        return pd.DataFrame(columns=["M", "best_tau", "tpr", "fpr", "dist_to_(0,1)"])

    best_configs_df = parse_bamboo_csv(bamboo_log_csv)

    # Sanitize numeric columns (thresholds, confidence) to avoid strings like 'inf' producing NaNs
    if "confidence" in best_configs_df.columns:
        best_configs_df["confidence"] = pd.to_numeric(best_configs_df["confidence"], errors="coerce")
        # replace infinite with a large finite number to avoid inf/inf operations
        best_configs_df["confidence"].replace([np.inf, -np.inf], 1e6, inplace=True)
        best_configs_df["confidence"].fillna(0.0, inplace=True)
    else:
        best_configs_df["confidence"] = 0.0

    if "best_threshold" in best_configs_df.columns:
        best_configs_df["best_threshold"] = pd.to_numeric(best_configs_df["best_threshold"], errors="coerce").fillna(0).astype(int)
    else:
        best_configs_df["best_threshold"] = 0

    max_bits = max(bits_set)  # default to max if not specified
    if max_bits != 0:
        # take first M filters
        best_configs_df = best_configs_df.head(max_bits)

    matrix_pairs_df = generateStringPairDf(balanced_pairs_df, string_df)
    matrix_pairs_df.reset_index(inplace=True, drop=True)

    # Extracting best filters and thresholds from the main DataFrame (assuming same filters and thresholds for simplicity)
    best_filters = best_configs_df["best_filter"].tolist()
    best_thresholds = best_configs_df["best_threshold"].tolist()
    filter_confidence = best_configs_df["confidence"].tolist()
    
    # Build set of unique items across both columns (by hashable key)
    unique_keys = {}
    for item in pd.concat([matrix_pairs_df["Item 1"], matrix_pairs_df["Item 2"]], ignore_index=True):
        k = tuple(item)
        if k not in unique_keys:
            unique_keys[k] = item  # keep original object for compute
    
    # Compute fingerprint once per unique item
    fprint_cache = {
        k: calculate_single_fprint(orig_item, best_filters, best_thresholds)
        for k, orig_item in unique_keys.items()
    }

    # Assign using the cache (instead of computing twice for duplicates)
    matrix_pairs_df["fprint1"] = matrix_pairs_df["Item 1"].apply(lambda item: fprint_cache[tuple(item)])
    matrix_pairs_df["fprint2"] = matrix_pairs_df["Item 2"].apply(lambda item: fprint_cache[tuple(item)])
    
    best_tau_rows = []
    for M in bits_set:
        matrix_pairs_df[f"h_distance_{M}"] = matrix_pairs_df.apply(
            lambda row: 
                hamming_distance_real(row["fprint1"][:M], row["fprint2"][:M], confidence=filter_confidence[:M]) if hamming else \
                hamming_distance(row["fprint1"][:M], row["fprint2"][:M], confidence=0.0),
            axis=1,
        )
        
        # distances and labels
        dist = matrix_pairs_df[f"h_distance_{M}"].to_numpy()
        y = matrix_pairs_df["Equality"].to_numpy()

        # map {-1,+1} → {0,1}
        y01 = (y == 1).astype(int)

        # score for ROC: higher = more positive
        scores = -dist

        # ROC: handle degenerate cases where only one class is present
        if np.unique(y01).size < 2:
            # fallback ROC: trivial line
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            thr = np.array([np.nan])
        else:
            fpr, tpr, thr = roc_curve(y01, scores)
            # remove first ROC point (threshold = +inf) when available
            if len(fpr) > 1:
                fpr, tpr, thr = fpr[1:], tpr[1:], thr[1:]
            else:
                # keep as-is
                pass

        # ---------- BEST TAU (closest to (0,1)) ----------
        roc_dist = np.sqrt(fpr**2 + (1 - tpr)**2)
        i_best = np.argmin(roc_dist)
        
        # save all data to a file for debug
        best_tau = -thr[i_best]  # convert score threshold → distance τ
        best_tau_rows.append({
            "M": M,
            "best_tau": best_tau,
            "tpr": tpr[i_best],
            "fpr": fpr[i_best],
            "dist_to_(0,1)": roc_dist[i_best],
        })

        # ---------- SAVE FULL ROC DATA ----------
        roc_data = pd.DataFrame({
            "fpr": fpr,
            "tpr": tpr,
            "tau": -thr,   # store τ explicitly (important!)
        })
        
        if roc_save_path is not None:
            roc_data.to_csv(
                f"{roc_save_path}/bamboo_roc_curve_data_{M}.csv" if not hamming else f"{roc_save_path}/bamboo_roc_curve_data_{M}_hamming.csv",
                index=False
            )

    # ---------- SAVE BEST TAU SUMMARY ----------
    best_tau_df = pd.DataFrame(best_tau_rows)

    # show ROC curve
    if roc_save_path is not None:
        roc_figure_name = f"{roc_save_path}/bamboo_roc_curve.pdf" if not hamming else f"{roc_save_path}/bamboo_roc_curve_hamming.pdf"
        plot_roc_curves_from_files(
            [f"{roc_save_path}/bamboo_roc_curve_data_{M}.csv" if not hamming else f"{roc_save_path}/bamboo_roc_curve_data_{M}_hamming.csv" for M in bits_set],
            [f"{M}-bit" for M in bits_set],
            roc_figure_name,
            show_plot=show_plot
        )
    return best_tau_df
