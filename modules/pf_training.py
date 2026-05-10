import numpy as np
import pandas as pd
import math
from collections import Counter


def calculate_pdf(array):
    # Convert array to a numpy array for easier manipulation
    array = np.array(array, dtype=object)
    # Get the number of rows and columns
    num_rows, num_cols = array.shape
    # Initialize a list to store the PDFs for each column
    pdfs = []
    # Iterate over each column
    for col in range(num_cols):
        # Get the current column
        column_data = array[:, col]
        # Count occurrences of each bit value ('0', '1', 'U')
        counts = Counter(column_data)
        # Calculate probabilities for '0', '1', and 'U'
        prob_0 = counts.get("0", 0) / num_rows
        prob_1 = counts.get("1", 0) / num_rows
        prob_U = counts.get("U", 0) / num_rows
        # Append the PDF for this column
        pdfs.append({"0": prob_0, "1": prob_1, "U": prob_U})
    return pdfs

def shannon_entropy(array):
    # First, calculate the PDF for each column
    pdfs = calculate_pdf(array)
    # Initialize a list to store the entropy values for each column
    entropies = []
    # Iterate over each column's PDF
    for pdf in pdfs:
        # Extract probabilities for '0', '1', and 'U'
        probabilities = [pdf["0"], pdf["1"], pdf["U"]]
        # Calculate Shannon entropy for the current column
        entropy = 0
        for prob in probabilities:
            if prob > 0:  # Avoid log(0)
                entropy -= prob * math.log(prob, 3)
        # Append the entropy value for this column
        entropies.append(entropy)
    return entropies

# find the threshold that allows to select M so to be the minimum number greater or equal to the number of bits specified by num_bits
def find_fingerprint_threshold(u_filtering, num_bits):
    start_threshold = 1
    while True:
        indexes = [i for i, value in enumerate(u_filtering) if value >= start_threshold]
        if len(indexes) >= num_bits:
            return start_threshold
        start_threshold -= 0.001

        if start_threshold < 0:
            return 0
        
def train_pf(bin_df, num_bits=64):
    df_concatenated = bin_df.copy()
    df_concatenated["concatenated_array"] = df_concatenated["concatenated"].apply(lambda x: np.array(list(x)))
    df_concatenated.drop(columns=["concatenated"], inplace=True)

    # Extract the 'concatenated_array' column as a list of lists
    concatenated_array_list = df_concatenated["concatenated_array"].tolist()

    # Convert the list to a NumPy matrix with dtype=object to allow mixed types (e.g., int and str)
    numpy_matrix = np.array(concatenated_array_list, dtype=object)

    def get_matrix(df):
        return np.array(df["concatenated_array"].tolist(), dtype=object)
    
    v = shannon_entropy(numpy_matrix)

    # Group the dataframe by the 'label' column
    grouped = df_concatenated.groupby("label")

    # Create a dictionary to store each DataFrame
    label_dfs = {label: group for label, group in grouped}
    
    # Dictionary to hold s arrays for each label
    s_arrays = {}

    # Iterate through the dictionary
    for label, df in label_dfs.items():
        matrix = get_matrix(df)  # Get the matrix from the DataFrame
        entropy = shannon_entropy(matrix)  # Calculate the Shannon entropy
        sa = [1 - e for e in entropy]  # Compute the s array
        s_arrays[label] = sa  # Store the s array in the dictionary
        
    s = np.mean(list(s_arrays.values()), axis=0)
    
    # Apply thresholding to create u_filtering
    threshold = 0.8
    u_filtering = [v[i] if s[i] >= threshold else 0 for i in range(len(v))]

    fingerprint_threshold = find_fingerprint_threshold(u_filtering, num_bits)
    
    #definisco la soglia che va a impattare sul numero di bit selezionati (prendo la soglia che permette di selezionare almeno num_bits)
    indexes = [i for i, value in enumerate(u_filtering) if value >= fingerprint_threshold]

    # Sort the indexes by the corresponding values in u_filtering in descending order
    sorted_indexes = sorted(indexes, key=lambda i: u_filtering[i], reverse=True)
    # enforce the number of bits by taking only the top num_bits indexes
    sorted_indexes = sorted_indexes[:num_bits]
        
    # Convert the indexes list to a DataFrame
    indexes_df = pd.DataFrame(sorted_indexes, columns=["Index"])
    # indexes_df to be saved with index=False 
    
    # Initialize an array of zeros with the same length as u_filtering
    probabilistic_filter = np.zeros(len(u_filtering), dtype=int)

    # Set the elements at the specified indexes to 1
    for index in sorted_indexes:
        probabilistic_filter[index] = 1
    
    # probabilistic_filter_df to be saved with index=False // not used for now
    return indexes_df
    
def get_train_data_pf(bin_df, num_bits=64):
    df_concatenated = bin_df.copy()
    df_concatenated["concatenated_array"] = df_concatenated["concatenated"].apply(lambda x: np.array(list(x)))
    df_concatenated.drop(columns=["concatenated"], inplace=True)
    concatenated_array_list = df_concatenated["concatenated_array"].tolist()
    numpy_matrix = np.array(concatenated_array_list, dtype=object)
    def get_matrix(df):
        return np.array(df["concatenated_array"].tolist(), dtype=object)    
    v = shannon_entropy(numpy_matrix)
    grouped = df_concatenated.groupby("label")
    label_dfs = {label: group for label, group in grouped}
    s_arrays = {}
    for label, df in label_dfs.items():
        matrix = get_matrix(df)  # Get the matrix from the DataFrame
        entropy = shannon_entropy(matrix)  # Calculate the Shannon entropy
        sa = [1 - e for e in entropy]  # Compute the s array
        s_arrays[label] = sa  # Store the s array in the dictionary
    s = np.mean(list(s_arrays.values()), axis=0)
    threshold = 0.8
    u_filtering = [v[i] if s[i] >= threshold else 0 for i in range(len(v))]
    fingerprint_threshold = find_fingerprint_threshold(u_filtering, num_bits)
    indexes = [i for i, value in enumerate(u_filtering) if value >= fingerprint_threshold]
    sorted_indexes = sorted(indexes, key=lambda i: u_filtering[i], reverse=True)
    sorted_indexes = sorted_indexes[:num_bits]
    indexes_df = pd.DataFrame(sorted_indexes, columns=["Index"])
    return fingerprint_threshold, u_filtering, indexes
    
if __name__ == "__main__":
    df_concatenated = pd.read_csv("C:/Users/fabio/Ricerca/Codice/2026_Bamboo_Journal/data/train_test/bin_train_P+B.csv")
    num_bits = 32
    indexes_df = train_pf(df_concatenated, num_bits=num_bits)
    M = len(indexes_df)
    print(f"N: {num_bits} - Number of bits selected (M): {M}")
