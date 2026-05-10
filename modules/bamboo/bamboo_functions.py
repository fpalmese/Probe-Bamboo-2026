import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import numpy as np
import pandas as pd
from rich import traceback
from rich.console import Console
from rich.progress import Progress

import modules.bamboo.utils as utils

traceback.install()
console = Console()


# ----------------------------- fast core -----------------------------

def _filter_width(filter_str: str) -> int:
    # faster than list comprehension allocation
    return filter_str.count("1") + filter_str.count("N")


def _filter_to_vector_np(filter_str: str) -> np.ndarray:
    """
    Map filter string into int8 vector:
      '1' -> 1
      'N' -> -1
      else -> 0
    """
    # Using a small lookup table on ASCII codes is fast and avoids Python loops.
    b = np.frombuffer(filter_str.encode("ascii"), dtype=np.uint8)
    out = np.zeros(len(b), dtype=np.int8)
    out[b == ord("1")] = 1
    out[b == ord("N")] = -1
    return out


@dataclass(frozen=True)
class PairArrays:
    item1: np.ndarray  # shape (n_pairs, n_bits), int8/ int16
    item2: np.ndarray  # shape (n_pairs, n_bits)
    y: np.ndarray      # shape (n_pairs,), values in {-1, 1}


def _encode_probe_strings_to_bits(probes: np.ndarray) -> np.ndarray:
    """
    Convert an array of probe strings like '01010...' into a 2D int8 array.
    Assumes probes contain only '0' and '1'. If your probes contain other symbols,
    adjust the mapping here.
    """
    # Vectorized byte conversion: join then reshape.
    # This is typically much faster than list(bstr) for each row.
    joined = "".join(probes.tolist()).encode("ascii")
    arr = np.frombuffer(joined, dtype=np.uint8) - ord("0")
    # should reshape from (n_pairs * n_bits,) to (n_pairs, n_bits)
    print(f"Reshaping from {arr.shape} to ({len(probes)}, {arr.shape[0] // len(probes)})")
    return arr.reshape(len(probes), -1).astype(np.int8, copy=False)


def _build_pair_arrays(pairs_df: pd.DataFrame, dataset: pd.DataFrame) -> PairArrays:
    probes = dataset["Probes"].astype(str).to_numpy()
    probe_bits = _encode_probe_strings_to_bits(probes)

    i1 = pairs_df["Item 1"].to_numpy(dtype=np.int64, copy=False)
    i2 = pairs_df["Item 2"].to_numpy(dtype=np.int64, copy=False)

    item1 = probe_bits[i1]
    item2 = probe_bits[i2]
    y = pairs_df["Equality"].to_numpy(copy=False).astype(np.int8, copy=False)

    return PairArrays(item1=item1, item2=item2, y=y)

def _errors_for_filter_thresholds(
    pairs: PairArrays,
    weights: np.ndarray,          # shape (n_pairs,), float64
    filter_str: str,
    thresholds: np.ndarray        # shape (t,), int
) -> np.ndarray:
    """
    Compute weighted classification error for each threshold of this filter.

    Returns:
        errors: shape (t,), float64
    """
    f = _filter_to_vector_np(filter_str).astype(np.int8, copy=False)

    # score = sum(item * f) where item is 0/1 and f is -1/0/1 => int16 is safe
    sa = (pairs.item1 * f).sum(axis=1, dtype=np.int16)
    sb = (pairs.item2 * f).sum(axis=1, dtype=np.int16)

    # For each threshold: sign(score - threshold) in {1, -1}
    # Using broadcasting: (n_pairs, 1) vs (1, t)
    pa = np.where(sa[:, None] - thresholds[None, :] > 0, 1, -1).astype(np.int8, copy=False)
    pb = np.where(sb[:, None] - thresholds[None, :] > 0, 1, -1).astype(np.int8, copy=False)

    pred = pa * pb  # shape (n_pairs, t)

    # mismatches: pred != y
    mism = (pred != pairs.y[:, None])

    # weighted error per threshold
    # weights shape (n_pairs,) => (n_pairs,1) broadcast
    return (mism * weights[:, None]).sum(axis=0, dtype=np.float64)


def _process_chunk(chunk_filters: np.ndarray, chunk_thresholds: list, pairs: PairArrays, weights: np.ndarray):
    """
    Worker: compute all (filter, threshold)->error for a chunk.
    Returns a dict[(filter_str, threshold_int)] = error_float
    """
    out = {}
    for f_str, thr_list in zip(chunk_filters, chunk_thresholds):
        thr = np.asarray(thr_list, dtype=np.int16)
        errs = _errors_for_filter_thresholds(pairs, weights, f_str, thr)
        # Fill dict
        for t, e in zip(thr.tolist(), errs.tolist()):
            out[(f_str, t)] = e
    return out


# ----------------------------- main -----------------------------
# pass as input the training df, the training pairs index, the filters df, the output file name, the number of iterations (M) and the number of filters (F) to be used (if 0, all filters will be used)
def train_bamboo(bin_df, train_pairs_df, advanced_filters_df, bamboo_output_file = "bamboo_output.csv",n_iterations=64, n_filters=0, max_workers=4):
    utils.title.print_title()
    
    custom_columns = utils.progressBarUtil.generateColumns()    

    strings_df = bin_df.copy()
    # if dataset has two columns, use the first as index and the second as probes
    if strings_df.shape[1] == 2:
        strings_df.set_index(strings_df.columns[0], inplace=True) # set the first column as index (if not already) and keep the second as probes (typically label)
    elif strings_df.shape[1] > 2:
        raise ValueError("Dataset has more than 2 columns, please check the input dataframe format.")

    
    dataset = strings_df.rename(columns={strings_df.columns[0]: "Probes"})
    
    pairs_df = train_pairs_df.copy()
    pairs_df.drop_duplicates(inplace=True)
    pairs_df.reset_index(drop=True, inplace=True)


    filters_df = advanced_filters_df.copy()
    # determine number of filters
    if n_filters == 0:
        n_filters = filters_df.shape[0]
        
    filters_df = filters_df.head(n_filters).reset_index()
    filters_bitmask = filters_df["Bitmask"].to_numpy()

    # build thresholds dataframe: columns ['filters','thresholds']
    # (same behavior as your threshold_gen.generate_thresholds_df)
    thresholds_rows = []
    for bitmask in filters_bitmask:
        k = sum(1 for ch in bitmask if ch != "0")
        thresholds_rows.append((bitmask, list(range(1, k + 1))))
    filters = pd.DataFrame(thresholds_rows, columns=["filters", "thresholds"])

    # init log file
    utils.logger.init_csv_file(bamboo_output_file)
    
    # pairs subset
    pairs_index = pairs_df

    # init weights
    weights = np.full(len(pairs_index), 1.0 / len(pairs_index), dtype=np.float64)

    # build fast pair arrays
    pairs = _build_pair_arrays(pairs_index, dataset)

    del dataset, strings_df, pairs_df, pairs_index, filters_df
    gc.collect()

    n_processes = max_workers
    # Chunk by rows (not by index object)
    # pre-split positions for stability across deletions
    def chunk_slices(n, k):
        # near-even contiguous chunks
        size = (n + k - 1) // k
        for start in range(0, n, size):
            yield slice(start, min(start + size, n))

    with Progress(*custom_columns) as progress, ProcessPoolExecutor(max_workers=n_processes) as executor:
        iteration_task = progress.add_task("[cyan]Going through iterations...", total=n_iterations)

        for _ in range(n_iterations):
            filters_task = progress.add_task("[green]Processing filters...", total=n_processes)

            # snapshot current filters table
            f_arr = filters["filters"].to_numpy()
            thr_arr = filters["thresholds"].tolist()

            futures = []
            for sl in chunk_slices(len(filters), n_processes):
                futures.append(
                    executor.submit(
                        _process_chunk,
                        f_arr[sl],
                        thr_arr[sl],
                        pairs,
                        weights,
                    )
                )

            errors = {}
            for fut in as_completed(futures):
                chunk_dict = fut.result()
                errors.update(chunk_dict)
                progress.update(filters_task, advance=1)

            # single-pass best selection (no full sort)
            best_key = None
            best_err = None
            best_width = None
            best_thr = None
            min_count = 0

            for (f_str, thr), err in errors.items():
                if best_err is None or err < best_err:
                    best_key = (f_str, thr)
                    best_err = err
                    best_width = _filter_width(f_str)
                    best_thr = thr
                    min_count = 1
                elif err == best_err:
                    min_count += 1
                    # tie-break: smaller filter width, then smaller threshold
                    w = _filter_width(f_str)
                    if w < best_width or (w == best_width and thr < best_thr):
                        best_key = (f_str, thr)
                        best_err = err
                        best_width = w
                        best_thr = thr

            if min_count > 1:
                utils.logger.log.warning(f"There are {min_count} configurations with the minimum error.")

            best_filter, best_threshold = best_key

            # remove best filter row (matches original behavior: delete by filter string)
            filters = filters[filters["filters"] != best_filter].reset_index(drop=True)
            n_filters -= 1

            # confidence
            min_error = best_err if best_err != 0 else 1e-20
            confidence = np.log((1.0 - min_error) / min_error)

            utils.logger.print_best_config([best_filter, best_threshold, min_error, confidence])
            utils.logger.store_best_config_to_csv([best_filter, best_threshold, min_error, confidence], bamboo_output_file)
            # weight update (same math as your classifier.weight_update but avoids pandas conversions)
            # compute predictions for just the chosen threshold (fast)
            f = _filter_to_vector_np(best_filter).astype(np.int8, copy=False)
            sa = (pairs.item1 * f).sum(axis=1, dtype=np.int16)
            sb = (pairs.item2 * f).sum(axis=1, dtype=np.int16)
            pa = np.where(sa - best_threshold > 0, 1, -1).astype(np.int8, copy=False)
            pb = np.where(sb - best_threshold > 0, 1, -1).astype(np.int8, copy=False)
            pred = pa * pb

            # Asymmetric weight update to match original classifier.weight_update():
            # - treat ground truth -1 as 0 (only positive class matters for boosting)
            # - treat prediction 1 as 0; prediction -1 stays -1
            # - boost weights only for (ground_truth==1 and prediction==-1)
            mispred_pos = ((pairs.y == 1) & (pred == -1)).astype(np.int8)
            weights = weights * np.where(mispred_pos == 1, np.exp(confidence), 1.0)
            weights /= weights.sum()
            gc.collect()
            progress.update(iteration_task, advance=1)
