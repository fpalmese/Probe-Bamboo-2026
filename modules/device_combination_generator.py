import numpy as np
import pandas as pd
from math import comb

def balanced_device_combinations(
    labels: list,
    C: int,
    random_state: int | None = None,
    allow_repeats: bool = False,
) -> pd.DataFrame:
    """
    Generate C balanced combinations for each group size k=1..N.

    Output columns:
        - combination  (string formatted tuple)
        - length       (group size)
    """

    rng = np.random.default_rng(random_state)
    labels = list(dict.fromkeys(labels))  # remove duplicates, keep order
    N = len(labels)

    rows = []

    for k in range(1, N + 1):

        max_unique = comb(N, k)
        target = C if allow_repeats else min(C, max_unique)

        counts = {lab: 0 for lab in labels}
        seen = set()
        generated = 0
        attempts = 0
        max_attempts = C * 50  # safety stop

        while generated < target and attempts < max_attempts:
            attempts += 1

            # sort labels by current usage count (balanced participation)
            ordered = sorted(labels, key=lambda x: (counts[x], rng.random()))
            combo = tuple(sorted(ordered[:k]))

            if not allow_repeats:
                if combo in seen:
                    continue
                seen.add(combo)

            # update participation counts
            for lab in combo:
                counts[lab] += 1

            rows.append({
                "combination": str(combo),
                "length": k
            })

            generated += 1

    return pd.DataFrame(rows)