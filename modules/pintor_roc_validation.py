import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd
import sys
sys.path.append("C:/Users/fabio/Ricerca/Codice/2026_Bamboo_Journal/well_structured")
from modules.utils.validation_utils import generateHexPairDf, sum_ascii_from_hex, min_max_normalize, plot_roc_curves_from_files
import os

def get_pintor_validation_data(hex_df, validation_pairs_df, columns = None, roc_save_path=None):
    if columns is None:
        columns = ["HT Capabilities", "Extended Capabilities", "Vendor Specific Tags"]
    n_cols = len(columns)
    # take only three columns from the hex dataframe
    hex_df = hex_df[columns+["Label"]].copy()
    
    balanced_pairs_df = validation_pairs_df.copy()
    balanced_pairs_df.drop_duplicates(inplace=True)
    balanced_pairs_df.reset_index(drop=True, inplace=True)
    
    matrix_pairs_df = generateHexPairDf(balanced_pairs_df, hex_df)
    

    # Apply the sum_ascii_from_hex function to the specified columns (all cols but not Label for the two devices)
    matrix_pairs_df = sum_ascii_from_hex(matrix_pairs_df, columns=[ col + " 1" for col in columns ] + [ col + " 2" for col in columns ])
    
    # apply min-max normalization to the ascii sum features
    matrix_pairs_df = min_max_normalize(matrix_pairs_df, columns=[ col + " 1" for col in columns ] + [ col + " 2" for col in columns ])
    
    # compute manhattan distance between the two devices for each pair and add as a new column
    matrix_pairs_df["Manhattan Distance"] = matrix_pairs_df[[ col + " 1" for col in columns ]].subtract(matrix_pairs_df[[ col + " 2" for col in columns ]].values).abs().sum(axis=1)
    
    # Compute ROC
    dist = matrix_pairs_df[f"Manhattan Distance"].to_numpy()
    y = matrix_pairs_df["Equality"].to_numpy()
    y01 = (y == 1).astype(int) # map {-1,+1} → {0,1}
    scores = -dist
    fpr, tpr, thr = roc_curve(y01, scores)
    fpr, tpr, thr = fpr[1:], tpr[1:], thr[1:] # remove (0,0) point corresponding to τ=∞
    roc_dist = np.sqrt(fpr**2 + (1 - tpr)**2)
    i_best = np.argmin(roc_dist)

    best_tau = -thr[i_best]  # convert score threshold → distance τ
   
    # ---------- SAVE FULL ROC DATA ----------
    roc_data = pd.DataFrame({ "fpr": fpr, "tpr": tpr, "tau": -thr})   # store τ explicitly (important!)
    
    if roc_save_path is not None:
        roc_data.to_csv(f"{roc_save_path}/pintor_roc_curve_data_{n_cols}.csv",index=False)
            
    # show ROC curve
    if roc_save_path is not None:
        roc_figure_name = f"{roc_save_path}/pintor_roc_curve_{n_cols}.pdf"
        plot_roc_curves_from_files([f"{roc_save_path}/pintor_roc_curve_data_{n_cols}.csv"],["Pintor ROC"],roc_figure_name,show_plot=False)
        
    #matrix_pairs_df.to_csv(os.path.join(roc_save_path, "pintor_validation_pairs.csv"), index=False)
    
    return best_tau

# only for testing the function, not to be used in main.py
def collect_csvs_and_concatenate(directory):
    df_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(csv_path)
                    df_list.append(df)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
    
    # Concatenate all dataframes
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

if __name__ == "__main__":
    hex_file = pd.read_csv("C:/Users/fabio/Ricerca/Codice/2026_Bamboo_Journal/well_structured/data/interim/hex_full_balanced.csv", dtype=str)
    validation_pairs_df = pd.read_csv(f"C:/Users/fabio/Ricerca/Codice/2026_Bamboo_Journal/well_structured/results/cycle_0/validation_pairs.csv", index_col=0)
    get_pintor_validation_data(hex_file, validation_pairs_df, roc_save_path=".")