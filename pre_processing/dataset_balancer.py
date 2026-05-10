import pandas as pd
from configparser import ConfigParser
import os
from utils.balancer import balanced_resample_propagate
# import the interim

n_entries_per_devices = 500


config = ConfigParser()
config_filename = "config_preprocessing.ini"
full_config_name = os.path.join(os.path.dirname(__file__), config_filename)
config.read(full_config_name)
interim_path = config["DEFAULT"]["output_path"]
interim_bin_file = os.path.join(interim_path, "binary_U_concatenated.csv")
interim_hex_file = os.path.join(interim_path, "hex_full.csv")

# load bin df and hex df
bin_df = pd.read_csv(interim_bin_file)
hex_df = pd.read_csv(interim_hex_file)

# replace the "U" with 0 in the bin_df concatenated column
#bin_df["concatenated"] = bin_df["concatenated"].str.replace("U", "0")

# balance the two dfs at the same time, preserving the ratio of the labels in the bin_df and propagating the balancing to the hex_df based on the label and concatenated column
bin_balanced, hex_balanced = balanced_resample_propagate(bin_df=bin_df, hex_df=hex_df,n_per_label=n_entries_per_devices,label_col="label",bin_col="concatenated")

# add global index column to both dfs, starting from 0 and incrementing by 1 for each row
bin_balanced.to_csv(os.path.join(interim_path, "binary_U_balanced.csv"), index=False)
hex_balanced.to_csv(os.path.join(interim_path, "hex_full_balanced.csv"), index=False)

# create a copy of the balanced bin df and replace the "U" with 0 in the concatenated column, then save it as a new csv file
bin0_balanced = bin_balanced.copy()
bin0_balanced["concatenated"] = bin0_balanced["concatenated"].str.replace("U", "0")
bin0_balanced.to_csv(os.path.join(interim_path, "binary_0_balanced.csv"), index=False)