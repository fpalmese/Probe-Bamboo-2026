def hex_string_to_binary(hex_string: str) -> str:
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

def extract_fields_from_binary(ie_dictionary, binary_string):
    # Initialize a list to hold tuples of field names and their corresponding bits
    extracted_fields = []

    # Convert the binary string to a list for easier access by index, and reverse it
    binary_list = list(binary_string)

    # Initialize a set to keep track of fields already added
    added_fields = set()

    # Iterate through each bit index in the extended capabilities dictionary
    for bit_index in range(len(binary_list)):
        # Check if this bit index is in the dictionary (to handle binary strings longer than the dictionary)
        if bit_index in ie_dictionary:
            field_name = ie_dictionary[bit_index]
            # If the field has not been added yet, proceed to extract its bits
            if field_name not in added_fields:
                # Find all bit indexes for this field
                bit_indexes = [
                    index for index, name in ie_dictionary.items() if name == field_name
                ]
                # Extract bits for the current field
                field_bits = "".join(
                    [
                        binary_list[index] if index < len(binary_list) else "0"
                        for index in bit_indexes
                    ]
                )
                # Add the field and its bits to the list
                extracted_fields.append(field_bits)
                # Mark this field as added
                added_fields.add(field_name)

    return extracted_fields
