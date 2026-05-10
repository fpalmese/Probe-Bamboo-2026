import pandas as pd
import sys 
sys.path.append("C:/Users/fabio/Ricerca/Codice/2026_Bamboo_Journal/well_structured")
from modules.utils.validation_utils import calculate_pf_fprint, generateStringPairDf, hamming_distance_pf, plot_roc_curves_from_files
from sklearn.metrics import roc_curve
import numpy as np



def get_pf_validation_data(bin_0_df, validation_pairs_df, index_filenames, bits_set = [8,16,32,64], roc_save_path=None):

    best_tau_rows = []
    # read the index file
    for i, num_bits in enumerate(bits_set):
        print(f"Validating PF with {num_bits} bits...")
        index_filename = index_filenames[i]
        indexes_df = pd.read_csv(index_filename)
        indexes = indexes_df['Index'].tolist()
        
        # generate string pairs dataframe
        matrix_pairs_df = generateStringPairDf(validation_pairs_df, bin_0_df)
        
        # Build set of unique items across both columns (by hashable key)
        unique_keys = {}
        for item in pd.concat([matrix_pairs_df["Item 1"], matrix_pairs_df["Item 2"]], ignore_index=True):
            k = tuple(item)
            if k not in unique_keys:
                unique_keys[k] = item  # keep original object for compute
                
        # Compute fingerprint once per unique item
        fprint_cache = {
            k: calculate_pf_fprint(orig_item, indexes, num_bits=num_bits)
            for k, orig_item in unique_keys.items()
        }

        # Assign using the cache (instead of computing twice for duplicates)
        matrix_pairs_df["fprint1"] = matrix_pairs_df["Item 1"].apply(lambda item: fprint_cache[tuple(item)])
        matrix_pairs_df["fprint2"] = matrix_pairs_df["Item 2"].apply(lambda item: fprint_cache[tuple(item)])
        
        # calculate hamming distance for each pair
        matrix_pairs_df[f"h_distance_{num_bits}"] = matrix_pairs_df.apply(lambda row: hamming_distance_pf(row["fprint1"], row["fprint2"]),axis=1,)
        
        # distances and labels
        dist = matrix_pairs_df[f"h_distance_{num_bits}"].to_numpy()
        y = matrix_pairs_df["Equality"].to_numpy()

        # map {-1,+1} → {0,1}
        y01 = (y == 1).astype(int)

        # score for ROC: higher = more positive
        scores = -dist

        # ROC
        fpr, tpr, thr = roc_curve(y01, scores)
        
        # remove first ROC point (threshold = +inf)
        fpr, tpr, thr = fpr[1:], tpr[1:], thr[1:]

        # ---------- BEST TAU (closest to (0,1)) ----------
        roc_dist = np.sqrt(fpr**2 + (1 - tpr)**2)
        i_best = np.argmin(roc_dist)

        best_tau = -thr[i_best]  # convert score threshold → distance τ
        best_tau_rows.append({
            "num_bits": num_bits,
            "best_tau": best_tau,
            "tpr": tpr[i_best],
            "fpr": fpr[i_best],
            "dist_to_(0,1)": roc_dist[i_best],
        })
        
        roc_data = pd.DataFrame({
            "fpr": fpr,
            "tpr": tpr,
            "tau": -thr,   # store τ explicitly (important!)
        })
        
        if roc_save_path is not None:
            roc_data.to_csv(f"{roc_save_path}/pf_roc_curve_data_{num_bits}.csv",index=False)
            
        
    best_tau_df = pd.DataFrame(best_tau_rows)
    print(best_tau_df)
    # show ROC curve
    if roc_save_path is not None:
        plot_roc_curves_from_files(
            [f"{roc_save_path}/pf_roc_curve_data_{num_bits}.csv" for num_bits in bits_set],
            [f"{num_bits}-bit" for num_bits in bits_set],
            f"{roc_save_path}/pf_roc_curve.pdf",show_plot=False
        )
    return best_tau_df

if __name__ == "__main__":
    bin_0_df = pd.read_csv(f"C:/Users/fabio/Ricerca/Codice/2026_Bamboo_Journal/data/train_test/bin_test0_P+B.csv",dtype=str)
    validation_pairs_df = pd.read_csv(f"C:/Users/fabio/Ricerca/Codice/2026_Bamboo_Journal/data/train_test/bin_test_pairs_P+B.csv", index_col=0)
    roc_save_path = "."
    
    index_filenames = [f"C:/Users/fabio/Ricerca/Codice/2026_Bamboo_Journal/data/outputs/PF/P+B/indexes_{m}.csv" for m in [18,18,36,74]]
    best_taus = get_pf_validation_data(bin_0_df, validation_pairs_df, index_filenames=index_filenames, bits_set=[8,16,32,64], roc_save_path=roc_save_path)
    