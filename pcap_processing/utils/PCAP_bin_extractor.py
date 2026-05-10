from utils import logger
from scapy.all import rdpcap
from utils import IE_bin_extractor as IEextractor
from utils import fileUtility, binUtility


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

            mac = IEextractor.getMAC(packet)

            packet_bits = ""

            packet_bits = binUtility.getMACLayerBits(packet)

            packet_IE = packet_bits[192:]

            index = 0
            packetLength = len(packet_IE)
            elements = []

            while index < packetLength - 32:
                packet_slice = packet_IE[index:]
                elementID = binUtility.readBinElementID(packet_slice)
                convertedID = binUtility.readElementID(packet_slice)

                if convertedID == "unknown":
                    continue

                length = binUtility.readBinLength(packet_slice)
                field = binUtility.readBinField(packet_slice)

                elements.append(
                    (
                        IEextractor.getElementIDText(convertedID),
                        binUtility.convertBinLength(packet_slice),
                        elementID,
                        length,
                        field,
                    )
                )

                index += 16 + binUtility.convertBinLength(packet_slice)

            frame_check_seq = packet_IE[-32:]

            # Init variables
            e_id_ssid = ""
            len_ssid = ""
            ssid = ""
            e_id_sup_rates = ""
            len_sup_rates = ""
            supported_rates = ""
            e_id_ext_sup_rates = ""
            len_ext_sup_rates = ""
            ext_sup_rates = ""
            e_id_dsss = ""
            len_dsss = ""
            dsss_parameter = ""
            e_id_ht_cap = ""
            len_ht_cap = ""
            ht_cap = ""
            e_id_ext_cap = ""
            len_ext_cap = ""
            ext_cap = ""
            e_id_vht_cap = ""
            len_vht_cap = ""
            vht_cap = ""
            e_id_vst = ""
            len_vst = ""
            vst = ""
            e_id_ext_tags = ""
            len_ext_tags = ""
            ext_tags = ""

            for i in range(len(elements)):
                match elements[i][0]:
                    case "ssid":
                        e_id_ssid = elements[i][2]
                        len_ssid = elements[i][3]
                        ssid = elements[i][4]
                    case "supported rates":
                        e_id_sup_rates = elements[i][2]
                        len_sup_rates = elements[i][3]
                        supported_rates = elements[i][4]
                    case "extended supported rates":
                        e_id_ext_sup_rates = elements[i][2]
                        len_ext_sup_rates = elements[i][3]
                        ext_sup_rates = elements[i][4]
                    case "dsss parameters":
                        e_id_dsss = elements[i][2]
                        len_dsss = elements[i][3]
                        dsss_parameter = elements[i][4]
                    case "ht capabilities":
                        e_id_ht_cap = elements[i][2]
                        len_ht_cap = elements[i][3]
                        ht_cap = elements[i][4]
                    case "vht capabilities":
                        e_id_vht_cap = elements[i][2]
                        len_vht_cap = elements[i][3]
                        vht_cap = elements[i][4]
                    case "extended capabilities":
                        e_id_ext_cap = elements[i][2]
                        len_ext_cap = elements[i][3]
                        ext_cap = elements[i][4]
                    case "vendor specific tags":
                        if len(vst) != 0:
                            e_id_vst = elements[i][2]
                            len_vst = elements[i][3]

                            vst += elements[i][2]
                            vst += elements[i][3]
                            vst += elements[i][4]
                        else:
                            e_id_vst = elements[i][2]
                            len_vst = elements[i][3]
                            vst = elements[i][4]
                    case "extended tags":
                        e_id_ext_tags = elements[i][2]
                        len_ext_tags = elements[i][3]
                        ext_tags = elements[i][4]
                    case _:
                        continue

            combined_list = [
                mac,
                e_id_ssid,
                len_ssid,
                ssid,
                e_id_sup_rates,
                len_sup_rates,
                supported_rates,
                e_id_ext_sup_rates,
                len_ext_sup_rates,
                ext_sup_rates,
                e_id_dsss,
                len_dsss,
                dsss_parameter,
                e_id_ht_cap,
                len_ht_cap,
                ht_cap,
                e_id_vht_cap,
                len_vht_cap,
                vht_cap,
                e_id_ext_cap,
                len_ext_cap,
                ext_cap,
                e_id_vst,
                len_vst,
                vst,
                e_id_ext_tags,
                len_ext_tags,
                ext_tags,
                frame_check_seq,
                label,
            ]

            output_data.append(combined_list)

            if progress:
                # Update the progress for each file
                progress.update(packet_task, advance=1)

        return output_data

    except Exception as e:
        logger.log.critical(f"Error extracting information from {file_path}: {e}")
        return RuntimeError
