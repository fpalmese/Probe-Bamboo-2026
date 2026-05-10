from utils import logger
import os
import csv


def checkCreatePath(output_path: str) -> None:
    """checkCreatePath
    Checks if the output folder exists. If not, it creates it.
    """

    if not os.path.exists(output_path):
        logger.log.warning(f"Output folder does not exist. Creating {output_path}")
        try:
            os.makedirs(output_path)
        except Exception as e:
            logger.log.critical(f"Error creating output folder: {e}")
            exit()


def csv_writer(header: list, data: list, output_path: str, label: str) -> None:
    """csv_writer

    Args:
        data (list): list of package network features
        output_path (str): output file path
        label (str): device label from original .pcap file
    """

    # Set the output file path
    output_file = output_path + f"{label}.csv"

    with open(output_file, "w", newline="") as csvfile:
        logger.log.info(f"Writing {label}" + ".csv")

        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        csv_writer.writerows(data)


def get_substring_after_last_slash(s):
    # Find the last occurrence of '/'
    last_slash_index = s.rfind("/")
    # If '/' is found, slice the string from the character after '/' to the end
    if last_slash_index != -1:
        return s[last_slash_index + 1 :]
    else:
        # If '/' is not found, return the entire string
        return s
