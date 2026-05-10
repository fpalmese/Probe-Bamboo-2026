import numpy as np
import pandas as pd

def _largest_remainder_allocation(counts: np.ndarray, total: int) -> np.ndarray:
    s = counts.sum()
    if s == 0 or total == 0:
        return np.zeros_like(counts, dtype=int)

    props = counts / s
    raw = props * total
    base = np.floor(raw).astype(int)
    remainder = raw - base
    missing = total - base.sum()

    if missing > 0:
        idx = np.argsort(-remainder)[:missing]
        base[idx] += 1
    elif missing < 0:
        idx = np.argsort(remainder)
        for i in idx:
            if missing == 0:
                break
            if base[i] > 0:
                base[i] -= 1
                missing += 1
    return base

def _sample_global_indices_preserving_hex(
    g: pd.DataFrame,
    hex_col: str,
    global_index_col: str,
    target_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Samples values from g[global_index_col], approximately preserving hex_col proportions.
    """
    if target_n <= 0:
        return np.array([], dtype=g[global_index_col].dtype)

    vc = g[hex_col].value_counts(dropna=False)
    values = vc.index.to_list()
    counts = vc.to_numpy(dtype=int)

    alloc = _largest_remainder_allocation(counts, target_n)

    chunks = []
    for v, k in zip(values, alloc):
        if k == 0:
            continue

        bucket_vals = g.loc[g[hex_col].eq(v), global_index_col].to_numpy()
        replace = k > bucket_vals.size
        chunks.append(rng.choice(bucket_vals, size=k, replace=replace))

    if not chunks:
        return np.array([], dtype=g[global_index_col].dtype)

    arr = np.concatenate(chunks)
    return arr[rng.permutation(arr.size)]

def generate_balanced_pairs_df(
    df: pd.DataFrame,
    labels: list | None = None,
    pairs_per_label: int = 1000,
    label_col: str = "label",
    hex_col: str = "concatenated",
    global_index_col: str = "global_index",
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Returns a dataframe with columns:
      Item 1, Item 2, Label 1, Label 2, Equality

    Where Item 1 / Item 2 are taken from df[global_index_col] (NOT df.index).

    Properties:
    - For each label L:
        * pairs_per_label same-label pairs (L vs L)
        * pairs_per_label different-label pairs (L vs not-L)
    - Sampling within each label preserves hex_col proportions (approx, integer-rounded).
    - Labels contribute equally.
    """
    if global_index_col not in df.columns:
        raise KeyError(f"'{global_index_col}' column not found in df")

    # Optional but strongly recommended: ensure global_index is unique
    # (remove if you intentionally allow duplicates)
    if df[global_index_col].duplicated().any():
        raise ValueError(f"'{global_index_col}' contains duplicates; expected unique IDs")

    rng = np.random.default_rng(random_state)

    if labels is None:
        labels = df[label_col].unique().tolist()
    else:
        existing = set(df[label_col].unique().tolist())
        labels = [l for l in labels if l in existing]

    if len(labels) == 0:
        return pd.DataFrame(columns=["Item 1", "Item 2", "Label 1", "Label 2", "Equality"])

    groups = {lbl: df[df[label_col].eq(lbl)] for lbl in labels}
    pairs = []

    for lbl in labels:
        g = groups[lbl]

        # --- same-label pairs: sample 2*pairs_per_label IDs from lbl and pair them ---
        ids_same = _sample_global_indices_preserving_hex(
            g, hex_col, global_index_col, 2 * pairs_per_label, rng
        )
        if ids_same.size < 2 * pairs_per_label:
            # defensive (shouldn't happen often; but keeps shape consistent)
            ids_same = np.resize(ids_same, 2 * pairs_per_label)

        left = ids_same[:pairs_per_label]
        right = ids_same[pairs_per_label: 2 * pairs_per_label]
        pairs.extend([(a, b, lbl, lbl, True) for a, b in zip(left.tolist(), right.tolist())])

        # --- different-label pairs: lbl vs other labels, partners distributed evenly ---
        other_labels = [x for x in labels if x != lbl]
        if not other_labels:
            continue

        left_diff = _sample_global_indices_preserving_hex(
            g, hex_col, global_index_col, pairs_per_label, rng
        )

        base = pairs_per_label // len(other_labels)
        rem = pairs_per_label - base * len(other_labels)
        partner_counts = {x: base for x in other_labels}
        if rem > 0:
            chosen = rng.choice(other_labels, size=rem, replace=False)
            for c in chosen:
                partner_counts[c] += 1

        right_chunks = []
        right_labels = []
        for partner_lbl, cnt in partner_counts.items():
            if cnt == 0:
                continue
            gp = groups[partner_lbl]
            ids_partner = _sample_global_indices_preserving_hex(
                gp, hex_col, global_index_col, cnt, rng
            )
            right_chunks.append(ids_partner)
            right_labels.extend([partner_lbl] * ids_partner.size)

        if right_chunks:
            right_diff = np.concatenate(right_chunks)
            right_labels = np.array(right_labels, dtype=object)

            perm = rng.permutation(right_diff.size)
            right_diff = right_diff[perm]
            right_labels = right_labels[perm]

            m = min(left_diff.size, right_diff.size)
            pairs.extend(
                [(a, b, lbl, lab2, False)
                 for a, b, lab2 in zip(left_diff[:m].tolist(),
                                      right_diff[:m].tolist(),
                                      right_labels[:m].tolist())]
            )

    out = pd.DataFrame(pairs, columns=["Item 1", "Item 2", "Label 1", "Label 2", "Equality"])
    out = out.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return out