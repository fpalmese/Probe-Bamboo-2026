import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.metrics import auc
from sklearn.preprocessing import MinMaxScaler

def plot_roc_curves_from_files(csv_files, labels, output_file,show_plot=True):
    if not csv_files or not labels or len(csv_files) != len(labels):
        print("Invalid input: Ensure that csv_files and labels are non-empty and of the same length.")
        return

    plt.figure(figsize=(6.5, 5.5))

    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "d", "*"]
    style_count = 0

    for csv_file, label in zip(csv_files, labels):
        roc_data = pd.read_csv(csv_file)

        fpr = roc_data["fpr"].to_numpy()
        tpr = roc_data["tpr"].to_numpy()

        # sort
        order = np.argsort(fpr)
        fpr = fpr[order]
        tpr = tpr[order]

        # deduplicate FPR (keep max TPR for each FPR)
        df = pd.DataFrame({"fpr": fpr, "tpr": tpr}).groupby("fpr", as_index=False)["tpr"].max()
        fpr_u = df["fpr"].to_numpy()
        tpr_u = df["tpr"].to_numpy()

        # Ensure ROC endpoints exist
        if fpr_u[0] > 0:
            fpr_u = np.insert(fpr_u, 0, 0.0)
            tpr_u = np.insert(tpr_u, 0, 0.0)

        if fpr_u[-1] < 1:
            fpr_u = np.append(fpr_u, 1.0)
            tpr_u = np.append(tpr_u, 1.0)
        # interpolate robustly
        fpr_interpolated = np.linspace(0, 1, 100)
        tpr_interpolated = np.interp(fpr_interpolated, fpr_u, tpr_u)

        roc_auc = auc(fpr_interpolated, tpr_interpolated)

        line_style = line_styles[style_count % len(line_styles)]
        marker = markers[style_count % len(markers)]
        """
        plt.plot(
            fpr_interpolated,
            tpr_interpolated,
            linestyle=line_style,
            marker=marker,
            markevery=5,
            markersize=7,
            lw=2,
            label=f"{label} (AUC = {roc_auc:.2f})"
        )"""

        plt.plot(
            fpr_u,
            tpr_u,
            linestyle=line_style,
            marker=marker,
            markersize=7,
            lw=2,
            label=f"{label} (AUC = {roc_auc:.2f})"
        )
        
        style_count += 1

    plt.plot([0, 1], [0, 1], color="grey", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.minorticks_on()
    plt.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    plt.grid(True, which="minor", color="gray", linestyle=":", linewidth=0.5)

    plt.savefig(output_file, format="pdf")
    if show_plot:
        plt.show()
    plt.close()
    
    

def parse_bamboo_log(filename):
    data = []

    with open(filename, "r") as file:
        lines = file.readlines()

        current_filter = None
        current_threshold = None
        current_min_error = None
        current_confidence = None

        for line in lines:
            if "Best Filter" in line:
                # Extract Best Filter using regex
                filter_match = re.search(r"Best Filter: (.+)", line)
                if filter_match:
                    current_filter = filter_match.group(1).strip()

            elif "Best Threshold" in line:
                # Extract Best Threshold using regex
                threshold_match = re.search(r"Best Threshold: (.+)", line)
                if threshold_match:
                    current_threshold = int(threshold_match.group(1).strip())

            elif "Min error" in line:
                # Extract Min Error using regex
                min_error_match = re.search(r"Min error: (.+)", line)
                if min_error_match:
                    current_min_error = float(min_error_match.group(1).strip())

            elif "Confidence" in line:
                # Extract Confidence using regex
                confidence_match = re.search(r"Confidence: (.+)", line)
                if confidence_match:
                    current_confidence = float(confidence_match.group(1).strip())

                    # Once we have all values, create a tuple and add it to the data list
                    data.append(
                        (
                            current_filter,
                            current_threshold,
                            current_min_error,
                            current_confidence,
                        )
                    )

                    # Reset current values for the next entry
                    current_filter = None
                    current_threshold = None
                    current_min_error = None
                    current_confidence = None

    # Convert the list of tuples into a DataFrame
    df = pd.DataFrame(
        data, columns=["Best Filter", "Best Threshold", "Min Error", "Confidence"]
    )

    return df

def parse_bamboo_csv(filename):
    filter_df = pd.read_csv(filename)
    mapping = {'0': 0, '1': 1, 'N': -1}

    filter_df["best_filter"] = filter_df["best_filter"].apply(
        lambda s: [mapping[c] for c in s]
    )
    return filter_df



def convertColumntoArray(df: pd.DataFrame, column_name: str) -> np.array:
    return np.array([list(bstr) for bstr in df[column_name]])

# turn hex string columns into ascii sum columns, to be used for pintor validation
def sum_ascii_from_hex(df, columns):
    def hex_to_ascii_sum(value):
        # Ensure the value is treated as a string
        hex_string = str(value)
        try:
            # Convert hex string to bytes, then to ASCII characters, and calculate their sum
            return sum(ord(chr(int(hex_string[i:i+2], 16))) for i in range(0, len(hex_string), 2))
        except ValueError:
            # Handle invalid hex strings
            return None
    
    # Apply the function to each column specified in the list, leaving other columns unchanged
    for column in columns:
        if column not in df.columns:
            continue
        df[column] = df[column].apply(hex_to_ascii_sum)
    
    return df

# apply min-max normalization to specified columns of a dataframe. Used for pintor validation to normalize the ascii sum features.
def min_max_normalize(df, columns):
    # take the columns to normalize, leave other columns unchanged
    feature_cols = [col for col in columns if col in df.columns]
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df_normalized

def prepare_pintor_features(df, columns, norm=True):
    # prepare the data using the three columns then compute the cols (sum ascii, normalization) needed for the clustering
    df = sum_ascii_from_hex(df, columns=columns)
    if norm:
        df = min_max_normalize(df, columns=columns)
    for col in columns:
        df[col] = df[col].fillna(0)
    # remove other columns except Label and the ones used for clustering   
    df = df[columns+["Label"]]
    return df
# used for pintor validation
def generateHexPairDf(pairs_df: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    item1_idx = pairs_df["Item 1"].astype(int).to_numpy()
    item2_idx = pairs_df["Item 2"].astype(int).to_numpy()
    left = dataset.iloc[item1_idx].reset_index(drop=True)
    right = dataset.iloc[item2_idx].reset_index(drop=True)
    return_df = pd.concat(
        [left.add_suffix(" 1"), right.add_suffix(" 2")],
        axis=1
    )
    return_df["Equality"] = pairs_df["Equality"].reset_index(drop=True)
    return return_df

def generateStringPairDf(pairs_df: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    # Convert the Probes column to a numpy array
    dataset_array = convertColumntoArray(dataset, "concatenated")

    return_df = pd.DataFrame()

    # Import the Probes into the pairs_df dataframe
    return_df["Item 1"] = pairs_df["Item 1"].apply(lambda index: dataset_array[index])
    return_df["Item 2"] = pairs_df["Item 2"].apply(lambda index: dataset_array[index])
    return_df["Equality"] = pairs_df["Equality"]

    return return_df


def calculate_pf_fprint(item, indexes, num_bits=64):
    return ''.join(item[i] for i in indexes[:num_bits])

def calculate_single_fprint(item, best_filters, best_thresholds):
    fingerprint = []

    for best_filter, best_threshold in zip(best_filters, best_thresholds):
        filtered = np.sum(np.multiply(item.astype(int), best_filter))

        if filtered > best_threshold:
            filtered = 1
        else:
            filtered = -1

        fingerprint.append(filtered)

    return fingerprint


def hamming_distance(array1, array2, confidence=None):
    # Check if arrays have the same length
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length")

    # Initialize distance counter
    distance = 0

    # Iterate through arrays and count differences
    for i in range(len(array1)):
        if array1[i] != array2[i]:
            distance += 1

    return distance

def hamming_distance_pf(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of the same length")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def hamming_distance_real(array1, array2, confidence):
    # Check if arrays have the same length
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length")

    # Initialize distance counter
    distance = 0

    # Iterate through arrays and count differences
    for i in range(len(array1)):
        if array1[i] != array2[i]:
            distance += confidence[i]

    distance = (distance / sum(confidence)) * len(confidence)

    return distance

def compute_auc_from_file(csv_file):
    data = pd.read_csv(csv_file)
    data.sort_values("fpr", inplace=True)
    roc_auc = auc(data["fpr"], data["tpr"])
    return roc_auc


def generate_combinations_df(labels, max_combinations_per_M=200, random_state=42):
    """
    Generate combinations of labels for each M from 2 to len(labels).

    Returns a DataFrame with columns:
        - combination: tuple of labels
        - M: number of labels in the combination
    """
    rng = np.random.default_rng(random_state)
    labels = list(labels)

    rows = []

    for M in range(2, len(labels) + 1):
        all_combs = list(itertools.combinations(labels, M))

        if len(all_combs) > max_combinations_per_M:
            idx = rng.choice(len(all_combs), size=max_combinations_per_M, replace=False)
            selected_combs = [all_combs[i] for i in idx]
        else:
            selected_combs = all_combs

        for comb in selected_combs:
            rows.append({
                "combination": comb,
                "length": M
            })

    return pd.DataFrame(rows)


def get_bamboo_fprint_matrix(bin_0_df, bamboo_log_csv):
    bin_0_df["concatenated"] = bin_0_df["concatenated"].apply(lambda row: np.array(list(row)))

    best_configs_df = parse_bamboo_csv(bamboo_log_csv)
    best_filters = best_configs_df["best_filter"].tolist()
    best_thresholds = best_configs_df["best_threshold"].tolist()
    
    # precompute fingerprints for unique concatenated values to avoid redundant calculations
    unique_keys = {}
    for item in bin_0_df["concatenated"]:
        k = tuple(item)
        if k not in unique_keys:
            unique_keys[k] = item  # keep original object for compute
    
    fprint_cache = {
        k: calculate_single_fprint(orig_item, best_filters, best_thresholds)
        for k, orig_item in unique_keys.items()
    }
    bin_0_df["bamboo_fprint"] = bin_0_df["concatenated"].apply(lambda item: fprint_cache[tuple(item)])
    return bin_0_df


# optimized pf fingerprint matrix generation using precomputation and caching, for faster validation
def get_pf_fprint_matrix(bin_0_df, index_filenames=[], bits_set = [8,16,32,64]):
    bin_0_df["concatenated"] = bin_0_df["concatenated"].apply(lambda row: np.array(list(row)))

    if not index_filenames or len(index_filenames) != len(bits_set):
        raise ValueError("index_filenames must be provided and match the length of bits_set")
    for index_filename, num_bits in zip(index_filenames, bits_set):
        # for each bit, precompute fingerprints for unique concatenated values to avoid redundant calculations
        unique_keys = {}
        for item in bin_0_df["concatenated"]:
            k = tuple(item)
            if k not in unique_keys:
                unique_keys[k] = item  # keep original object for compute
        
        indexes_df = pd.read_csv(index_filename)
        indexes = indexes_df['Index'].tolist()

        fprint_cache = {
            k: calculate_pf_fprint(orig_item, indexes, num_bits=num_bits)
            for k, orig_item in unique_keys.items()
        }
        bin_0_df[f"pf_fprint_{num_bits}"] = bin_0_df["concatenated"].apply(lambda item: fprint_cache[tuple(item)])
        
    return bin_0_df