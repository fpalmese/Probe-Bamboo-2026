def fieldPadder(field: list, length: int) -> list:
    """
    Pads a list with None values to a specified length.

    Args:
    field (list): The list to pad.
    length (int): The length to pad the list to.

    Returns:
    list: The padded list.
    """
    if len(field) == length:
        return field
    elif len(field) > length:
        return field[:length]
    return field + [None] * (length - len(field))


def noneList(length: int) -> list:
    """
    Creates a list of None values of a specified length.

    Args:
    length (int): The length of the list.

    Returns:
    list: The list of None values.
    """
    return [None] * length


def hex_string_to_binary(hex_string):
    # Initialize the resulting binary string
    binary_string = ""

    # Iterate every two characters in the hexadecimal string
    for i in range(0, len(hex_string), 2):
        # Extract the pair of characters
        hex_pair = hex_string[i : i + 2]
        # Convert the pair to a byte (decimal) in little endian format
        byte = int(hex_pair, 16)
        # Reverse the byte
        byte = int("{:08b}".format(byte)[::-1], 2)
        # Convert the byte to a binary string of 8 bits and concatenate
        binary_string += f"{byte:08b}"

    return binary_string
