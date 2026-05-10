import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.metrics import (homogeneity_score,completeness_score,v_measure_score,adjusted_rand_score,normalized_mutual_info_score,root_mean_squared_error)


def map_clusters_to_labels(y_true, cluster_labels):
    """
    Map each cluster to the majority true label in that cluster.
    Noise (-1) stays as -1.
    """
    mapping = {}

    for cluster in np.unique(cluster_labels):
        if cluster == -1:
            continue

        mask = cluster_labels == cluster
        true_vals = y_true[mask]

        if len(true_vals) == 0:
            continue

        values, counts = np.unique(true_vals, return_counts=True)
        mapping[cluster] = values[np.argmax(counts)]

    y_pred = np.array([mapping.get(label, -1) for label in cluster_labels])
    return y_pred


def compute_clustering_metrics(y_true, cluster_labels, compute_rmse=True):
    """
    Compute clustering metrics.
    RMSE is optional and only makes sense if y_true is numeric.
    """
    metrics = {}

    metrics["homogeneity"] = homogeneity_score(y_true, cluster_labels)
    metrics["completeness"] = completeness_score(y_true, cluster_labels)
    metrics["v_measure"] = v_measure_score(y_true, cluster_labels)
    metrics["ari"] = adjusted_rand_score(y_true, cluster_labels)
    metrics["nmi"] = normalized_mutual_info_score(y_true, cluster_labels)
    metrics["noise_ratio"] = np.mean(cluster_labels == -1)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    metrics["n_clusters"] = n_clusters

    if compute_rmse:
        y_pred = map_clusters_to_labels(y_true, cluster_labels)
        mask = y_pred != -1

        if np.any(mask):
            y_true_masked = y_true[mask]
            y_pred_masked = y_pred[mask]

            # RMSE needs numeric values; encode categorical labels consistently when needed.
            try:
                y_true_num = np.asarray(y_true_masked, dtype=float)
                y_pred_num = np.asarray(y_pred_masked, dtype=float)
            except (ValueError, TypeError):
                all_labels = np.concatenate([y_true_masked.astype(str), y_pred_masked.astype(str)])
                _, uniques = pd.factorize(all_labels)
                label_to_int = {label: idx for idx, label in enumerate(uniques)}
                y_true_num = np.array([label_to_int[v] for v in y_true_masked.astype(str)], dtype=float)
                y_pred_num = np.array([label_to_int[v] for v in y_pred_masked.astype(str)], dtype=float)

            metrics["rmse"] = root_mean_squared_error(y_true_num, y_pred_num)
        else:
            metrics["rmse"] = np.nan
    else:
        metrics["rmse"] = np.nan

    return metrics

# also considers -1 from the noise points
def compute_clustering_metrics_advanced(y_true, cluster_labels, compute_label_score=True):
    """
    Compute external clustering metrics for DBSCAN-like outputs.
    Reports both:
    - metrics on all points
    - metrics on clustered points only (excluding noise label -1)
    """
    metrics = {}

    y_true = np.asarray(y_true)
    cluster_labels = np.asarray(cluster_labels)

    noise_mask = cluster_labels == -1
    clustered_mask = ~noise_mask

    metrics["noise_ratio"] = np.mean(noise_mask)
    metrics["coverage"] = np.mean(clustered_mask)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    metrics["n_clusters"] = n_clusters

    # Metrics on all points
    metrics["homogeneity_all"] = homogeneity_score(y_true, cluster_labels)
    metrics["completeness_all"] = completeness_score(y_true, cluster_labels)
    metrics["v_measure_all"] = v_measure_score(y_true, cluster_labels)
    metrics["ari_all"] = adjusted_rand_score(y_true, cluster_labels)
    metrics["nmi_all"] = normalized_mutual_info_score(y_true, cluster_labels)

    # Metrics excluding noise
    if np.any(clustered_mask):
        y_true_clustered = y_true[clustered_mask]
        labels_clustered = cluster_labels[clustered_mask]

        # only meaningful if at least 2 predicted clusters remain
        n_clusters_clustered = len(set(labels_clustered))
        if n_clusters_clustered >= 1:
            metrics["homogeneity_clustered"] = homogeneity_score(y_true_clustered, labels_clustered)
            metrics["completeness_clustered"] = completeness_score(y_true_clustered, labels_clustered)
            metrics["v_measure_clustered"] = v_measure_score(y_true_clustered, labels_clustered)
            metrics["ari_clustered"] = adjusted_rand_score(y_true_clustered, labels_clustered)
            metrics["nmi_clustered"] = normalized_mutual_info_score(y_true_clustered, labels_clustered)
        else:
            metrics["homogeneity_clustered"] = np.nan
            metrics["completeness_clustered"] = np.nan
            metrics["v_measure_clustered"] = np.nan
            metrics["ari_clustered"] = np.nan
            metrics["nmi_clustered"] = np.nan
    else:
        metrics["homogeneity_clustered"] = np.nan
        metrics["completeness_clustered"] = np.nan
        metrics["v_measure_clustered"] = np.nan
        metrics["ari_clustered"] = np.nan
        metrics["nmi_clustered"] = np.nan

    # Optional mapped-label classification score
    if compute_label_score:
        y_pred = map_clusters_to_labels(y_true, cluster_labels)
        valid_mask = y_pred != -1

        if np.any(valid_mask):
            metrics["label_accuracy"] = np.mean(y_true[valid_mask] == y_pred[valid_mask])
        else:
            metrics["label_accuracy"] = np.nan
    else:
        metrics["label_accuracy"] = np.nan
    return metrics




def evaluate_dbscan_on_subset(X_subset, y_subset, eps, min_samples, metric="hamming", compute_rmse=True):
    """
    Run DBSCAN on one subset and return clustering metrics.
    """
    # DBSCAN requires a 2D feature matrix; subsets may arrive as object arrays of lists.
    X_subset = np.asarray(X_subset)
    if X_subset.ndim == 1:
        if len(X_subset) > 0 and isinstance(X_subset[0], (list, tuple, np.ndarray)):
            X_subset = np.asarray(list(X_subset))
        else:
            X_subset = X_subset.reshape(-1, 1)

    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
        
    cluster_labels = model.fit_predict(X_subset)

    return compute_clustering_metrics(
        y_true=y_subset,
        cluster_labels=cluster_labels,
        compute_rmse=compute_rmse,
    )


def compute_best_params_for_dbscan_combinations(X_val,y_val,combinations_df,eps_values,min_samples_values,metric="hamming",score_name="v_measure",compute_rmse=True):
    """
    Find best DBSCAN params for each M by averaging over all combinations of that M.

    Parameters
    ----------
    X_val : np.ndarray
        Validation features, shape (n_samples, n_features)
    y_val : np.ndarray
        Ground-truth labels for validation samples
    combinations_df : pd.DataFrame
        Must contain columns:
          - 'combination' : tuple of device IDs
          - 'length' : size of the combination
    eps_values : iterable
    min_samples_values : iterable
    metric : str
    score_name : str
        Metric used to choose best params. Usually 'v_measure'
    compute_rmse : bool
        Only set True if y_val is numeric and RMSE is meaningful

    Returns
    -------
    all_results_df : pd.DataFrame
        One row per (M, eps, min_samples)
    best_results_df : pd.DataFrame
        Best row for each length M, according to score_name
    """
    all_results = []
    best_results = []
    
    # iterate over M values (combination lengths) in the combinations_df
    for M in sorted(combinations_df["length"].unique())[::2]:
    #for M in range(2,8):
        print(f"Evaluating DBSCAN for M={M}...")
        combs_M = combinations_df.loc[combinations_df["length"] == M, "combination"].tolist()

        # perform here the grid search, take each hyperparam combination (eps, min_samples) and average the metrics over all combinations of that M
        for eps in eps_values:
            for min_samples in min_samples_values:
                metrics_list = []

                for comb in combs_M:
                    mask = np.isin(y_val, comb)
                    
                    X_subset = X_val[mask]
                    y_subset = y_val[mask]
                    
                    if len(X_subset) == 0:
                        continue

                    metrics = evaluate_dbscan_on_subset(
                        X_subset=X_subset,
                        y_subset=y_subset,
                        eps=eps,
                        min_samples=min_samples,
                        metric=metric,
                        compute_rmse=compute_rmse,
                    )
                    metrics_list.append(metrics)
                    #print(f"  Comb {comb}: eps={eps}, min_samples={min_samples} → {metrics}")

                if not metrics_list:
                    continue

                avg_metrics = {
                    key: np.nanmean([m[key] for m in metrics_list])
                    for key in metrics_list[0].keys()
                }

                row = {
                    "length": M,
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_combinations": len(metrics_list),
                    **avg_metrics,
                }

                #M_param_results.append(row)
                all_results.append(row)

        #M_param_results_df = pd.DataFrame(M_param_results)
    
    all_results_df = pd.DataFrame(all_results)
    # Select best params globally (across all M values) by averaging metrics
    # over rows that share the same (eps, min_samples).
    if all_results_df.empty:
        best_results_df = pd.DataFrame(best_results)
        return all_results_df, best_results_df

    if score_name not in all_results_df.columns:
        raise ValueError(f"score_name '{score_name}' not found in computed metrics")

    agg_map = {
        "length": "nunique",
        "n_combinations": "sum",
        score_name: "mean",
    }

    best_results_df = (
        all_results_df
        .groupby(["eps", "min_samples"], as_index=False)
        .agg(agg_map)
        .rename(columns={"length": "n_lengths"})
        .sort_values(by=score_name, ascending=False, na_position="last")
        .head(1)
        .reset_index(drop=True)
    )

    return all_results_df, best_results_df

def validate_dbscan_on_bamboo_data(val_df, combinations_df, bits_set = [8,16,32,64],output_folder="."):
    # first of all compute the fingerprint df for each item in the bin_0_df, using the best filters and thresholds from the bamboo log csv
    # check that concatenated column is present in val_df and should be a list
    if "bamboo_fprint" not in val_df.columns:
        raise ValueError("val_df must contain 'bamboo_fprint' column")
    
    for n_bit in bits_set:
        print("Validating DBSCAN for BAMBOO for n_bit =", n_bit)
        # for each n_bit, cut the fingerprint to n_bit
        X_val = val_df["bamboo_fprint"].apply(lambda f: f[:n_bit])
        y_val = val_df["label"].to_numpy()
        
        _ , best_results_df = compute_best_params_for_dbscan_combinations(
            X_val=X_val.to_numpy(),
            y_val=y_val,
            combinations_df=combinations_df,
            eps_values=[1e-5,1e-2],
            min_samples_values=[2, 6],
            metric="hamming",
            score_name="v_measure",
            compute_rmse=True,
        )
        best_results_df.to_csv(f"{output_folder}/best_dbscan_params_{n_bit}_bits.csv", index=False)
        
def validate_dbscan_on_pf_data(val_df, combinations_df, bits_set = [8,16,32,64],output_folder="."):
    # check that all pf_fprint_{n_bit} columns are present in val_df and should be lists
    for n_bit in bits_set:
        col_name = f"pf_fprint_{n_bit}"
        if col_name not in val_df.columns:
            raise ValueError(f"val_df must contain '{col_name}' column")
    
    for n_bit in bits_set:
        print("Validating DBSCAN for PF for n_bit =", n_bit)
        col_name = f"pf_fprint_{n_bit}"
        # for each n_bit, cut the fingerprint to n_bit
        X_val = val_df[col_name]
        y_val = val_df["label"].to_numpy()
        
        _ , best_results_df = compute_best_params_for_dbscan_combinations(
            X_val=X_val.to_numpy(),
            y_val=y_val,
            combinations_df=combinations_df,
            eps_values=[1e-5,1e-2],
            min_samples_values=[2, 6],
            metric="hamming",
            score_name="v_measure",
            compute_rmse=True,
        )
        best_results_df.to_csv(f"{output_folder}/best_dbscan_params_{n_bit}_bits.csv", index=False)
        
    
def validate_dbscan_on_pintor_data(val_df, combinations_df, n_cols = [2,3], output_folder="."):
    columns = ["HT Capabilities", "Extended Capabilities", "Vendor Specific Tags"]
    for n_col in n_cols:
        print("Validating DBSCAN for PINTOR for n_col =", n_col)
        # for each n_col, select the corresponding columns
        selected_columns = columns[-n_col:]
        
        X_val = val_df[selected_columns]
        y_val = val_df["Label"].to_numpy()
        
        _ , best_results_df = compute_best_params_for_dbscan_combinations(
            X_val=X_val.to_numpy(),
            y_val=y_val,
            combinations_df=combinations_df,
            eps_values=[1e-5,1e-2],
            min_samples_values=[2, 6],
            metric="manhattan",
            score_name="v_measure",
            compute_rmse=True,
        )
        best_results_df.to_csv(f"{output_folder}/best_dbscan_params_{n_col}_cols.csv", index=False)
        
                  
def test_dbscan_on_combinations(X, y, combs, eps, min_samples, metric="hamming", compute_rmse=True):
    metrics_list = []
    for M in sorted(combs["length"].unique()):
        combs_M = combs.loc[combs["length"] == M, "combination"].tolist()
        for comb in combs_M:
            mask = np.isin(y, comb)
            X_subset = X[mask]
            y_subset = y[mask]
            if len(X_subset) == 0:
                continue
            metrics = evaluate_dbscan_on_subset(
                X_subset=X_subset,
                y_subset=y_subset,
                eps=eps,
                min_samples=min_samples,
                metric=metric,
                compute_rmse=compute_rmse,
            )
            metrics["length"] = M
            metrics_list.append(metrics)
    
    # create df from metrics_list and return it
    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df

  
def test_dbscan_bamboo(test_df, combinations_df, best_res_df, num_bits):
    # first of all compute the fingerprint df for each item in the bin_0_df, using the best filters and thresholds from the bamboo log csv
    # check that concatenated column is present in val_df and should be a list
    if "bamboo_fprint" not in test_df.columns:
        raise ValueError("val_df must contain 'bamboo_fprint' column")
    
    # take best_res_df, take first row and extract the best eps and min_samples
    best_eps = best_res_df.loc[0, "eps"]
    best_min_samples = best_res_df.loc[0, "min_samples"]
    #
    best_eps = 1e-2
    #
    
    metrics_list = []
    # for each n_bit, cut the fingerprint to n_bit
    X_val = test_df["bamboo_fprint"].apply(lambda f: f[:num_bits])
    y_val = test_df["label"].to_numpy()
    
    metrics_df = test_dbscan_on_combinations(
        X=X_val.to_numpy(),
        y=y_val,
        combs=combinations_df,
        eps=best_eps,
        min_samples=best_min_samples,
        metric="hamming",
        compute_rmse=True,
    )
    return metrics_df

def test_dbscan_pf(test_df, combinations_df, best_res_df, num_bits):
    # check that all pf_fprint_{n_bit} columns are present in val_df and should be lists
    col_name = f"pf_fprint_{num_bits}"
    if col_name not in test_df.columns:
        raise ValueError(f"val_df must contain '{col_name}' column")
    
    # take best_res_df, take first row and extract the best eps and min_samples
    best_eps = best_res_df.loc[0, "eps"]
    best_min_samples = best_res_df.loc[0, "min_samples"]
    
    # for each n_bit, cut the fingerprint to n_bit
    X_val = test_df[col_name]
    y_val = test_df["label"].to_numpy()
    
    metrics_df = test_dbscan_on_combinations(
        X=X_val.to_numpy(),
        y=y_val,
        combs=combinations_df,
        eps=best_eps,
        min_samples=best_min_samples,
        metric="hamming",
        compute_rmse=True,
    )
    return metrics_df

def test_dbscan_pintor(test_df, combinations_df, best_res_df, n_cols):
    columns = ["HT Capabilities", "Extended Capabilities", "Vendor Specific Tags"]
    selected_columns = columns[-n_cols:]
    
    # take best_res_df, take first row and extract the best eps and min_samples
    best_eps = best_res_df.loc[0, "eps"]
    best_min_samples = best_res_df.loc[0, "min_samples"]
    
    X_val = test_df[selected_columns]
    y_val = test_df["Label"].to_numpy()
    
    metrics_df = test_dbscan_on_combinations(
        X=X_val.to_numpy(),
        y=y_val,
        combs=combinations_df,
        eps=best_eps,
        min_samples=best_min_samples,
        metric="manhattan",
        compute_rmse=True,
    )
    return metrics_df

def cluster_oneshot_groupby(X):

    # Make feature values hashable so pandas can group list/array fingerprints.
    for col in X.columns:
        X[col] = X[col].apply(
            lambda v: tuple(v) if isinstance(v, (list, tuple, np.ndarray)) else v
        )
    
    # assign label depending on the groupby of the features, for example if the features are binary and we have 3 features, we can assign a label to each combination of the features (000, 001, 010, 011, 100, 101, 110, 111) and then compute the metrics based on these labels
    cluster_labels = X.groupby(X.columns.tolist()).ngroup()
    X["cluster"] = cluster_labels
    
    return X

def cluster_oneshot_dbscan(X, eps, min_samples, metric="hamming"):


    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    cluster_labels = model.fit_predict(X)
    
    return cluster_labels

def test_groupby_on_combinations(X, y, combs):
    if isinstance(X, pd.Series):
        X = X.to_frame(name=X.name or "feature")

    metrics_list = []
    for M in sorted(combs["length"].unique()):
        combs_M = combs.loc[combs["length"] == M, "combination"].tolist()
        for comb in combs_M:
            mask = np.isin(y, comb)
            X_subset = X[mask].copy()
            y_subset = y[mask]
            if len(X_subset) == 0:
                continue

            # Make feature values hashable so pandas can group list/array fingerprints.
            for col in X_subset.columns:
                X_subset[col] = X_subset[col].apply(
                    lambda v: tuple(v) if isinstance(v, (list, tuple, np.ndarray)) else v
                )
            
            # assign label depending on the groupby of the features, for example if the features are binary and we have 3 features, we can assign a label to each combination of the features (000, 001, 010, 011, 100, 101, 110, 111) and then compute the metrics based on these labels
            cluster_labels = X_subset.groupby(X_subset.columns.tolist()).ngroup()
            
            
            metrics = compute_clustering_metrics(
                y_true=y_subset,
                cluster_labels=cluster_labels,
                compute_rmse=True,
            )
            metrics["length"] = M
            metrics_list.append(metrics)
    
    # create df from metrics_list and return it
    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df

