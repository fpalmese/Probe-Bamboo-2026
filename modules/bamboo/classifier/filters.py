import numpy as np
from utils import logger

from . import compute_error


def calculate_filter_width(filter_str: str) -> int:
    return len([char for char in filter_str if char != "0"])


def filter_to_vector(filter_str: str) -> np.ndarray:
    # Convert the string to a list of integers
    vector = [1 if char == "1" else -1 if char == "N" else 0 for char in filter_str]

    return vector



def process_filters_chunk(chunk, string_pair_df, weights) -> dict:
    filter_threshold_errors_dict = {}
    print("HERE OK")
    
    if chunk.empty:
        print("Empty chunk error")
        logger.log.critical("The input chunk is empty, cannot process filters.")
        raise ValueError("The input chunk is empty, cannot process filters.")

    for _, row in chunk.iterrows():
        if chunk.empty:
            print("Empty chunk error")
            logger.log.critical("The input chunk is empty, cannot process filters.")
            raise ValueError("The input chunk is empty, cannot process filters.")

        filter = row["filters"]
        thresholds = row["thresholds"]
        current_errors = compute_error.matrix_error(
            string_pair_df, thresholds, filter, weights
        )
        
        # Merge errors
        for key, value in current_errors.items():
            if key in filter_threshold_errors_dict:
                filter_threshold_errors_dict[key].extend(value)
            else:
                filter_threshold_errors_dict[key] = value
        
        print("Row done", filter)
    return filter_threshold_errors_dict
