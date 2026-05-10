from utils import logger
from scapy.all import rdpcap
from utils import IE_dissected_extractor
from utils import fileUtility


# Create a function to extract information from a PCAP file
def extract_pcap_info(file_path: str, label: str, progress=None) -> list:

    try:
        # Read the PCAP file using Scapy
        packets = rdpcap(file_path)

        output_data = []

        filename = fileUtility.get_substring_after_last_slash(file_path)

        if progress:
            # Create a task for the inner loop
            packet_task = progress.add_task(
                f"[blue]Processing packets: {filename}", total=len(packets)
            )

        for packet in packets:
            # Timestamp
            timestamp = IE_dissected_extractor.extractTimestamp(packet)

            # Source MAC address
            mac_address = IE_dissected_extractor.extractMAC(packet)

            # Channel number
            channel = IE_dissected_extractor.extractChannel(packet)

            # DS Parameter Set channel number
            ds_channel = IE_dissected_extractor.extractDSChannel(packet)

            # HT Capabilities (HEX)
            htcapabilities = IE_dissected_extractor.extractHTCapabilities(packet)

            # Extended Capabilities (HEX)
            extended_capabilities = IE_dissected_extractor.extractExtendedCapabilities(packet)

            # Sequence Number
            seq_number = IE_dissected_extractor.extractSN(packet)

            # Vendor Specific Tags (HEX)
            vendor_specific_tags = IE_dissected_extractor.extractVendorSpecificTags(packet)

            # Additional features

            # SSID
            ssid = IE_dissected_extractor.extractSSID(packet)

            # Supported Rates (HEX)
            supported_rates = IE_dissected_extractor.extractSupportedRates(packet)

            # Extended Supported Rates (HEX)
            extended_supported_rates = IE_dissected_extractor.extractExtendedSupportedRates(packet)

            # VHT Capabilities (HEX)
            vhtcapabilities = IE_dissected_extractor.extractVHTCapabilities(packet)

            # HE Capabilities (HEX)
            hecapabilities = IE_dissected_extractor.extractHECapabilities(packet)

            # Packet size
            packet_length = len(packet)

            combined_list = (
                [
                    timestamp,
                    mac_address,
                    channel,
                    ds_channel,
                    seq_number,
                    vendor_specific_tags,
                    ssid,
                    vhtcapabilities,
                    hecapabilities,
                    packet_length,
                    label,
                ]
                + supported_rates  # add individual Supported Rates
                + extended_supported_rates  # add individual Extended Supported Rates
                + htcapabilities  # add individual HT Capabilities
                + extended_capabilities  # add individual Extended Capabilities
            )

            output_data.append(combined_list)

            if progress:
                # Update the progress for each file
                progress.update(packet_task, advance=1)

        return output_data

    except Exception as e:
        logger.log.critical(f"Error extracting information from {file_path}: {e}")
        return RuntimeError
