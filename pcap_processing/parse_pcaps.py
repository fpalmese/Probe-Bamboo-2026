from configparser import ConfigParser
import os
import sys
sys.path.append(os.path.dirname(__file__))
from pcap_processing.utils.hex_extractor import hex_extraction_folder
from pcap_processing.utils.binary_dissector import binary_dissector_folder
from pcap_processing.utils.data_dissector import extract_dissected_folder

# This script processes PCAP files based on the configuration specified in 'config_pcap_processing.ini'.
# input: PCAP folder located in the 'raw_path' specified for each dataset in the configuration file. Each pcap file in the folder will be scanned
# Output: hex data, binary probes and dissected data in CSV format for each dataset specified in the configuration file.
# if multiple datasets need to be processed, they can be added as separate sections in the configuration file with their respective paths.

def parse_pcaps(config_file = "config_pcap_processing.ini"):
    if config_file is not None:
        full_config_name = os.path.join(os.path.dirname(__file__), config_file)
        print("Reading configuration from:", full_config_name)
        config = ConfigParser()
        config.read(full_config_name)
    else:
        print("[ERROR] No configuration file provided.")
        return
        
    
    for i in config.sections():
        print(f"Processing dataset: {i}")
        input_path = config[i]["raw_path"]
        hex_path = config[i]["hex_path"]
        binary_path = config[i]["binary_path"]
        dissected_path = config[i]["dissected_path"]
        hex_extraction_folder(input_path, hex_path) # Extract hex data from PCAP files and save to CSV
        binary_dissector_folder(input_path, binary_path) # Process the hex data to extract binary information and save to CSV
        extract_dissected_folder(input_path, dissected_path) # Extract detailed information from the PCAP files and save to CSV
        
if __name__ == "__main__":
    parse_pcaps()