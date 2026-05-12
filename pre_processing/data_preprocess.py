import os
from configparser import ConfigParser
import sys
sys.path.append(os.path.dirname(__file__))
from pre_processing.utils.preprocessing import apply_bin_concatenation, load_and_concat_csv,generate_filters_dataframe, clean_df

# -------------------
# 1. Create binary concatenated interim used for all the binary operations. Read all the binary files and create the binary_U_concatenated.csv and the hex_full.csv (after cleaning) in the interim folder.
# 2. Generate filters dataframe if config section found
# 
# –------------------

def data_preprocess(config=None, config_file = "config_preprocessing.ini"):
    if config is not None:
        print("Using provided configuration object.")
        
    elif config_file is not None:
        # read configuration
        config = ConfigParser()
        config_filename = "config_preprocessing.ini"
        full_config_name = os.path.join(os.path.dirname(__file__), config_filename)
        config.read(full_config_name)
        
    binary_path = config["DEFAULT"]["binary_path"]
    output_path = config["DEFAULT"]["output_path"]

    # build dataframe from binary CSV files
    bin_df = load_and_concat_csv(binary_path,dtype=str)
    
    print(f"Loaded binary dataframe shape: {bin_df.shape}")
    # apply bin preprocessing steps
    concat_bin_df = apply_bin_concatenation(bin_df)
    
    # save the preprocessed dataframe to a new CSV file
    output_file = os.path.join(output_path, "binary_U_concatenated.csv")
    concat_bin_df.to_csv(output_file, index=False)
    print(f"Saved dataframe with concatenated binary features to: {output_file}. Shape: {concat_bin_df.shape}")
    print(f"Average length of the concatenated column: {concat_bin_df['concatenated'].str.len().mean()}")
    
    # create a copy for the bin file with "U" replaced by "0" in the concatenated column, then save it as a new csv file
    concat_bin_df_0 = concat_bin_df.copy()
    concat_bin_df_0["concatenated"] = concat_bin_df_0["concatenated"].str.replace("U", "0")
    output_file_0 = os.path.join(output_path, "binary_0_concatenated.csv")
    concat_bin_df_0.to_csv(output_file_0, index=False)
    
    # now merge all hex in one path
    hex_path = config["DEFAULT"]["hex_path"]
    hex_df = load_and_concat_csv(hex_path)
    hex_df = clean_df(hex_df)
    # save the preprocessed dataframe to a new CSV file
    hex_output_file = os.path.join(output_path, "hex_full.csv")
    hex_df.to_csv(hex_output_file, index=False)
    print(f"Saved dataframe with concatenated hex features to: {hex_output_file}. Shape: {hex_df.shape}")
    
    
    # generate filters_df if config found
    if "FILTERS" in config.sections():
        print("Generating filters dataframe based on the concatenated binary features...")
        max_length = concat_bin_df['concatenated'].str.len().max()
        filters_df = generate_filters_dataframe(max_length)
        filters_path = config["FILTERS"]["filters_path"]
        # save filters_df
        filter_output_file = os.path.join(filters_path, "bitmask_patterns_sliding_window.csv")
        filters_df.to_csv(filter_output_file, index=False)
        print(f"Saved filter dataframe to: {filter_output_file}. Shape: {filters_df.shape}")
    
        
if __name__ == "__main__":
    data_preprocess()