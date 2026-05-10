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

def _sample_indices_preserving_hex(g: pd.DataFrame, hex_col: str, target_n: int, rng: np.random.Generator) -> np.ndarray:
    if target_n <= 0:
        return np.array([], dtype=g.index.dtype)

    vc = g[hex_col].value_counts(dropna=False)
    values = vc.index.to_list()
    counts = vc.to_numpy(dtype=int)

    alloc = _largest_remainder_allocation(counts, target_n)

    chunks = []
    for v, k in zip(values, alloc):
        if k == 0:
            continue
        bucket_idx = g.index[g[hex_col].eq(v)].to_numpy()
        replace = k > bucket_idx.size
        chunks.append(rng.choice(bucket_idx, size=k, replace=replace))

    if not chunks:
        return np.array([], dtype=g.index.dtype)

    arr = np.concatenate(chunks)
    return arr[rng.permutation(arr.size)]

# main function to import and use
# pass the bin_df, returns the pair indexes
def generate_balanced_pairs_df(
    df: pd.DataFrame,
    labels: list | None = None,
    pairs_per_label: int = 1000,
    label_col: str = "label",
    hex_col: str = "concatenated",
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Returns a dataframe with columns:
      item1_idx, item2_idx, label1, label2, equality

    Properties:
    - For each label L:
        * pairs_per_label same-label pairs (L vs L)
        * pairs_per_label different-label pairs (L vs not-L)
    - Sampling within each label preserves hex_col proportions (approx, integer-rounded).
    - Labels contribute equally.
    """
    rng = np.random.default_rng(random_state)

    if labels is None:
        labels = df[label_col].unique().tolist()
    else:
        existing = set(df[label_col].unique().tolist())
        labels = [l for l in labels if l in existing]

    if len(labels) == 0:
        return pd.DataFrame(columns=["Item 1", "Item 2", "device1", "device2", "Equality"])

    groups = {lbl: df[df[label_col].eq(lbl)] for lbl in labels}
    pairs = []

    for lbl in labels:
        g = groups[lbl]
        
        # --- same-label pairs: sample 2*pairs_per_label indices from lbl and pair them ---
        idx_same = _sample_indices_preserving_hex(g, hex_col, 2 * pairs_per_label, rng)
        if idx_same.size < 2 * pairs_per_label:
            # defensive (shouldn't happen)
            idx_same = np.resize(idx_same, 2 * pairs_per_label)

        left = idx_same[:pairs_per_label]
        right = idx_same[pairs_per_label:2 * pairs_per_label]
        pairs.extend([(a, b, lbl, lbl, 1) for a, b in zip(left.tolist(), right.tolist())])

        # --- different-label pairs: lbl vs other labels, partners distributed evenly ---
        other_labels = [x for x in labels if x != lbl]
        if not other_labels:
            continue

        left_diff = _sample_indices_preserving_hex(g, hex_col, pairs_per_label, rng)

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
            idx_partner = _sample_indices_preserving_hex(gp, hex_col, cnt, rng)
            right_chunks.append(idx_partner)
            right_labels.extend([partner_lbl] * idx_partner.size)

        if right_chunks:
            right_diff = np.concatenate(right_chunks)
            right_labels = np.array(right_labels, dtype=object)

            perm = rng.permutation(right_diff.size)
            right_diff = right_diff[perm]
            right_labels = right_labels[perm]

            m = min(left_diff.size, right_diff.size)
            pairs.extend([(a, b, lbl, lab2, -1)
                          for a, b, lab2 in zip(left_diff[:m].tolist(), right_diff[:m].tolist(), right_labels[:m].tolist())])

    out = pd.DataFrame(pairs, columns=["Item 1", "Item 2", "device1", "device2", "Equality"])
    out = out.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return out