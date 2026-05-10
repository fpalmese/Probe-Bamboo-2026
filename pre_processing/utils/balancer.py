import pandas as pd
import numpy as np

def _largest_remainder_allocation(proportions: np.ndarray, total: int) -> np.ndarray:
    raw = proportions * total
    base = np.floor(raw).astype(int)
    remainder = raw - base
    missing = total - base.sum()

    if missing > 0:
        idx = np.argsort(-remainder)[:missing]
        base[idx] += 1
    elif missing < 0:
        idx = np.argsort(remainder)  # smallest first
        for i in idx:
            if missing == 0:
                break
            if base[i] > 0:
                base[i] -= 1
                missing += 1
    return base

def balanced_resample_indices_preserve_ratio(
    bin_df: pd.DataFrame,
    n_per_label: int,
    label_col: str = "label",
    bin_col: str = "concatenated",
    random_state: int | None = None,
) -> np.ndarray:
    """
    Returns indices (may contain duplicates) for each label with exactly n_per_label rows,
    preserving within-label bin_col ratios as closely as possible.
    Undersamples without replacement where possible, oversamples with replacement when needed.
    """
    rng = np.random.default_rng(random_state)
    sampled = []

    for label, g in bin_df.groupby(label_col, sort=False):
        counts = g[bin_col].value_counts(dropna=False)
        values = counts.index.to_list()
        props = (counts / counts.sum()).to_numpy()
        alloc = _largest_remainder_allocation(props, n_per_label)

        for v, k in zip(values, alloc):
            if k == 0:
                continue
            bucket_idx = g.index[g[bin_col].eq(v)].to_numpy()
            m = bucket_idx.size
            replace = k > m  # undersample if possible, otherwise oversample
            sampled.append(rng.choice(bucket_idx, size=k, replace=replace))

    sampled_idx = np.concatenate(sampled) if sampled else np.array([], dtype=bin_df.index.dtype)

    # shuffle so output isn't grouped by label/bin
    if sampled_idx.size:
        sampled_idx = sampled_idx[rng.permutation(sampled_idx.size)]
    return sampled_idx

def balanced_resample_propagate(
    bin_df: pd.DataFrame,
    hex_df: pd.DataFrame,
    n_per_label: int,
    label_col: str = "label",
    bin_col: str = "concatenated",
    random_state: int | None = None,
    reset_index: bool = True,
):
    """
    Resample based on bin_df ratios; apply identical row selection to hex_df.
    Works for both undersampling and oversampling.
    """
    if not bin_df.index.equals(hex_df.index):
        raise ValueError("bin_df and hex_df must have identical indices to propagate sampling.")

    idx = balanced_resample_indices_preserve_ratio(
        bin_df=bin_df,
        n_per_label=n_per_label,
        label_col=label_col,
        bin_col=bin_col,
        random_state=random_state,
    )

    bin_out = bin_df.loc[idx]
    hex_out = hex_df.loc[idx]

    if reset_index:
        bin_out = bin_out.reset_index(drop=True)
        hex_out = hex_out.reset_index(drop=True)

    return bin_out, hex_out