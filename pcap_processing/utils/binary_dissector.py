import os
from utils import PCAP_bin_extractor
from utils import fileUtility
from utils.bin_header import HEADER

def binary_dissector_folder(capture_path, output_path):

    # List all pcap files in the directory
    pcap_files = [f for f in os.listdir(capture_path) if f.endswith(".pcap")]

    for i, filename in enumerate(pcap_files):
        print(f"Binary Processing file: {filename} ({i+1}/{len(pcap_files)})")
        # Set output file name to the same as the PCAP file
        label = os.path.splitext(filename)[0]

        # Set the file path for the PCAP file
        file_path = os.path.join(capture_path, filename)

        # Extract information from the PCAP file
        info = PCAP_bin_extractor.extract_pcap_info(file_path, label)

        if info:
            # Check if the output folder exists
            fileUtility.checkCreatePath(output_path)
            fileUtility.csv_writer(HEADER, info, output_path, label)
