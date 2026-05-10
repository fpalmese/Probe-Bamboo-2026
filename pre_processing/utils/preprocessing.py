import pandas as pd
import random
import os


def pad_columns(data, symbol='0', exclude=[],length=None):
    max_lengths = data.drop(columns=exclude).apply(lambda col: col.map(lambda x: len(str(x)))).max()
    for col in data.columns:
        if col not in exclude:
            max_length = length if length is not None else max_lengths[col]
            data[col] = data[col].fillna("").astype(str).str.ljust(max_length, symbol)
    return data


def load_and_concat_csv(directory,dtype=None):
    # Initialize an empty dictionary to store DataFrames
    dataframes = {}

    # Traverse the directory structure
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                # Construct the full file path
                file_path = os.path.join(root, file)

                # Read the CSV file into a DataFrame
                if dtype is not None:
                    df_tmp = pd.read_csv(file_path, dtype=dtype)
                else:
                    df_tmp = pd.read_csv(file_path)

                # Store the DataFrame in the dictionary with a unique key (e.g., file name)
                dataframes[file] = df_tmp

    # Concatenate all DataFrames into one
    return pd.concat(dataframes.values(), ignore_index=True)

def generate_random_mac():
    return ":".join(f"{random.randint(0, 255):02x}" for _ in range(6))


def split_non_random_bursts(data: pd.DataFrame, labels: list) -> pd.DataFrame:
    data["dsss_parameter"] = data["dsss_parameter"].replace("nan", "0")
    data["dsss_parameter"] = data["dsss_parameter"].fillna("0")
    for label in labels:
        # Filter rows with the specific label
        label_data = data[data["label"] == label]
        label_data = label_data.sort_index().reset_index()

        # Initialize variables to track the current burst
        random_mac = generate_random_mac()  # Initial random MAC for the first burst
        start_index = 0  # Start index of the current burst

        for i in range(1, len(label_data)):
            # Check for a drop in DS Channel
            if int(label_data.at[i, "dsss_parameter"], 2) < int(
                label_data.at[i - 1, "dsss_parameter"], 2
            ):
                # Update all rows in the current burst with the current random MAC
                for j in range(start_index, i):
                    original_index = label_data.loc[j, "index"]
                    data.loc[original_index, "mac"] = random_mac

                # Generate a new random MAC for the next burst
                random_mac = generate_random_mac()
                # print(f"Channel dropped; assigning new MAC: {random_mac}")

                # Update the start index for the next burst
                start_index = i

        # Update the last burst (from the last drop to the end)
        for j in range(start_index, len(label_data)):
            original_index = label_data.loc[j, "index"]
            data.loc[original_index, "mac"] = random_mac

    return data

def find_non_randomizing_devices(data: pd.DataFrame) -> list:
    non_randomizing_devices = []
    for label in data["label"].unique():
        label_data = data[data["label"] == label]
        if label_data["mac"].nunique() == 1:
            non_randomizing_devices.append(label)
    return non_randomizing_devices


def clean_df(data: pd.DataFrame) -> pd.DataFrame:
    if "mac" in data.columns:
        data = data[data["mac"] != "00:0f:00:6a:68:8b"]
    elif "MAC Address" in data.columns:
        data = data[data["MAC Address"] != "00:0f:00:6a:68:8b"]
    return data


def apply_bin_concatenation(data: pd.DataFrame) -> pd.DataFrame:
    # filter out specific MAC address
    data = clean_df(data)
    
    # split non-random bursts and assign random MAC addresses
    non_randomizing_devices = find_non_randomizing_devices(data)
    data = split_non_random_bursts(data, non_randomizing_devices)
    

    # drop unnecessary columns
    cols_to_drop = ["frame_check_seq", "len_dsss", "ssid"] + [ col for col in data.columns if col.startswith("e_id_") ] + ["len_ssid", "len_sup_rates","len_ext_sup_rates","len_vht_cap","len_ext_tags","supported_rates","ext_sup_rates","vht_cap","ext_tags"]
    data = data.drop(columns=cols_to_drop, errors='ignore')
    
    # replace nan string with U and pad columns
    data = data.replace("nan", "U")
    data = pad_columns(data, symbol="U", exclude=["mac", "label","dsss_parameter"]) # exclude dsss_parameter from padding, we will fix it later
    
    len_vst_fixed = 1336
    new_padded_vst = pad_columns(data[["mac","vst"]],symbol="U",length=len_vst_fixed)
    data["vst"] = new_padded_vst["vst"]
    
    
    # fix the dsss_parameter to its max len, zero cap -> NOT ON BURST BUT PROBES
    if "dsss_parameter" in data.columns:
        dsss_fixed_len = data["dsss_parameter"].str.len().max()
        data["dsss_parameter"] = data["dsss_parameter"].str[:dsss_fixed_len]
        data["dsss_parameter"] = data["dsss_parameter"].str.zfill(dsss_fixed_len)
    else:
        dsss_fixed_len = 128
        data["dsss_parameter"] = "0" * dsss_fixed_len
    
    
    # concat all cols except label and mac into a single string column, then add label
    data["concatenated"] = (
        data.drop(columns=["label", "mac"]).astype(str).apply(lambda x: "".join(x), axis=1)
        )

    #data_result = data_result.sort_values("label")
    
    for col in data.columns:
        print(f"Average length of the column {col}: {data[col].str.len().mean()}")
        
    data_result = data[["label", "concatenated"]]
    return data_result


def generate_filters_dataframe(max_length):
    bitmasks = []
    def generate_for_block(block_size):
        half = block_size // 2

        # iterate over block positions
        for start in range(0, max_length, block_size):
            if start + block_size > max_length:
                break

            prefix = "0" * start
            suffix = "0" * (max_length - start - block_size)

            patterns = [
                "N" * half + "1" * half,
                "1" * half + "N" * half,
                "1" * block_size,
            ]

            for pattern in patterns:
                bitmasks.append(prefix + pattern + suffix)

    # generate filters of size 8 and 16
    generate_for_block(8)
    generate_for_block(16)

    return pd.DataFrame(bitmasks, columns=["Bitmask"])


