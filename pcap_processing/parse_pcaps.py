from configparser import ConfigParser
import os
from utils.hex_extractor import hex_extraction_folder
from utils.binary_dissector import binary_dissector_folder
from utils.data_dissector import extract_dissected_folder

def parse_pcaps():
    config = ConfigParser()
    config_filename = "config_pcap_processing.ini"
    full_config_name = os.path.join(os.path.dirname(__file__), config_filename)
    config.read(full_config_name)
    print("Reading configuration from:", full_config_name)
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