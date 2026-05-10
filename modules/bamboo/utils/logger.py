import logging

from rich.logging import RichHandler

logging.getLogger("scapy.runtime").setLevel(logging.CRITICAL)

FORMAT = "%(asctime)s %(message)s" 
logging.basicConfig(
    level="INFO",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True), logging.FileHandler("bamboo.log")],
)

log = logging.getLogger("rich")

log.setLevel("INFO")


def format_non_zero_part(filter_str: str) -> str:
    first_non_zero = -1
    for i in range(len(filter_str)):
        if filter_str[i] != "0":
            first_non_zero = i
            break

    last_non_zero = -1
    for i in range(len(filter_str) - 1, -1, -1):
        if filter_str[i] != "0":
            last_non_zero = i
            break

    if first_non_zero == -1:
        return f"0[{len(filter_str)}]"

    zeros_before = first_non_zero

    zeros_after = len(filter_str) - last_non_zero - 1

    non_zero_part = filter_str[first_non_zero : last_non_zero + 1]

    for char in non_zero_part:
        if char == "N":
            #non_zero_part = non_zero_part.replace("N", "🀆")
            non_zero_part = non_zero_part.replace("N", "o")
        if char == "1":
            non_zero_part = non_zero_part.replace("1", "x")
            #non_zero_part = non_zero_part.replace("1", "🀫")

    result = f"0[{zeros_before}] {non_zero_part} 0[{zeros_after}]"
    return result

def print_best_config(best_configs: tuple) -> None:
    log.info(f"Best Filter: {format_non_zero_part(best_configs[0])}")
    log.info(f"Best Threshold: {str(best_configs[1])}")
    if best_configs[2] < 10**-10:
        log.info("Min error: 0")
    else:
        log.info(f"Min error: {best_configs[2]}")
    if best_configs[3] > 40:
        log.info("Confidence: inf")
    else:
        log.info(f"Confidence: {best_configs[3]}")
        
        
def init_csv_file(filename: str) -> None:
    filename = filename if filename.endswith(".csv") else f"{filename}.csv"
    with open(filename, "w") as f:
        f.write("best_filter_fancy,best_filter,best_threshold,min_error,confidence\n")

def store_best_config_to_csv(best_configs: tuple, filename: str) -> None:
    filename = filename if filename.endswith(".csv") else f"{filename}.csv"
    
    error_str = f"{best_configs[2]:.2e}" if best_configs[2] >= 10**-10 else "0"
    confidence_str = f"{best_configs[3]:.2f}" if best_configs[3] <= 40 else "inf"
    
    with open(filename, "a") as f:
        f.write(f"{format_non_zero_part(best_configs[0])},{best_configs[0]},{best_configs[1]},{error_str},{confidence_str}\n")
        