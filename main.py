from modules.bamboo.bamboo_functions import train_bamboo
from modules.pf_training import train_pf
from modules.pair_generator import generate_balanced_pairs_df
from modules.bamboo_roc_validation import get_bamboo_validation_data
from modules.pf_roc_validation import get_pf_validation_data
from modules.pintor_roc_validation import get_pintor_validation_data
import pandas as pd
import numpy as np
import os
from modules.compare_results import compute_auc_comparison_figures, compute_dbscan_test_comparison_figures
from modules.utils.validation_utils import generate_combinations_df, get_bamboo_fprint_matrix, get_pf_fprint_matrix
from modules.utils.clustering_utils import validate_dbscan_on_bamboo_data, validate_dbscan_on_pf_data, validate_dbscan_on_pintor_data, test_dbscan_bamboo, test_dbscan_pf, test_dbscan_pintor, test_groupby_on_combinations
from modules.utils.validation_utils import prepare_pintor_features
import ast

# define interim data path
data_folder = "/data/interim/"
out_folder = "/data/results/"
bin_0_filename = "binary_0_balanced.csv"
bin_U_filename = "binary_U_balanced.csv"
hex_filename = "hex_full_balanced.csv"
bamboo_filters_filename = "/data/filters/bitmask_patterns_sliding_window.csv"

N_FOLDS = 1 # set to 20 for full cross-val, for now max 10
N_TEST = 20
N_VAL = 20
N_TRAIN = 23

RANDOM_STATE = 42
reset_data = False

N_PAIRS = 1000 # number of pairs to generate per label for training and validation (for both bamboo and pf)

to_train_bamboo = False
to_train_pf = False
to_val_bamboo = False
to_val_pf = False
to_val_pintor = False

to_val_dbscan_bamboo = False
to_val_dbscan_pf = False
to_val_dbscan_pintor = False

to_test_cluster_bamboo = True
to_test_cluster_pf = False
to_test_cluster_pintor = False

auc_comparison = False
dbscan_comparison = False

#clustering_method = "dbscan" 
#clustering_method = "groupby" 
clustering_method = "any"

def rotated_split(labels, t, step=N_TEST):
    """
    labels: list of device IDs in a fixed shuffled order
    t: attempt index
    step: rotation step; using 20 gives good dispersion (gcd(20,63)=1)
    """
    n = len(labels)
    s = (t * step) % n

    test_idx = [(s + i) % n for i in range(N_TEST)]
    val_idx  = [(s + N_TEST + i) % n for i in range(N_VAL)]

    test = [labels[i] for i in test_idx]
    val  = [labels[i] for i in val_idx]

    test_set = set(test)
    val_set = set(val)
    train = [lab for lab in labels if (lab not in test_set and lab not in val_set)]
    
    return train, val, test

def main():
        
    bin_0_df = pd.read_csv(data_folder + bin_0_filename, dtype=str)
    bin_U_df = pd.read_csv(data_folder + bin_U_filename, dtype=str)
    hex_df = pd.read_csv(data_folder + hex_filename, dtype=str)
    bamboo_filters_df = pd.read_csv(bamboo_filters_filename,index_col=0)

    # add label column if not there (take from index)
    if "label" not in bin_U_df.columns:
        bin_U_df["label"] = bin_U_df.index.astype(str)
    if "label" not in bin_0_df.columns:
        bin_0_df["label"] = bin_0_df.index.astype(str)

    # Read all the labels from the binary_U_balanced.csv file
    unique_labels = bin_U_df["label"].unique().tolist()
    # sort labels then shuffle with seed
    unique_labels.sort()
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(unique_labels)

    for t in range(N_FOLDS):
        # create out dir if not exists
        out_dir = out_folder + f"cycle_{t}/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # generate bamboo/pf/pintor directories aswell if needed
        for subdir in ["bamboo/", "pf/", "pintor/"]:
            if (to_val_bamboo and subdir=="bamboo/") or (to_val_pf and subdir=="pf/") or (to_val_pintor and subdir=="pintor/"):
                if not os.path.exists(out_dir + subdir):
                    os.makedirs(out_dir + subdir)
        
        train_labels, val_labels, test_labels = rotated_split(unique_labels, t)
        # print the labels for this cycle in output (labels.txt) for reference
        with open(out_dir + "labels.txt", "w") as f:
            f.write("TRAIN LABELS:\n")
            f.write("\n".join(train_labels) + "\n\n")
            f.write("VAL LABELS:\n")
            f.write("\n".join(val_labels) + "\n\n")
            f.write("TEST LABELS:\n")
            f.write("\n".join(test_labels) + "\n\n")
        
        # Filter dataframe by device(label) set
        
        if to_train_bamboo or to_val_bamboo or to_val_dbscan_bamboo or to_test_cluster_bamboo:
            bamboo_output_file = out_dir + f"bamboo_output.csv"
            
        # load validation pairs if needed 
        if to_val_bamboo or to_val_pf or to_val_pintor:
            # generate validation pairs for both bamboo and pf (same pairs for both, generated from bin_0_df filtered by val_labels)
            # do only if the val pairs file doesn't exist yet (to avoid regenerating pairs for each fold if not needed)
            # otherwise load the existing val pairs file
            if os.path.exists(out_dir + f"validation_pairs.csv") and not reset_data:
                validation_pairs_df = pd.read_csv(out_dir + f"validation_pairs.csv", index_col=0)
            else:
                validation_pairs_df = generate_balanced_pairs_df(bin_0_df[bin_0_df["label"].isin(val_labels)].copy(), pairs_per_label=N_PAIRS, random_state=RANDOM_STATE)
                validation_pairs_df.to_csv(out_dir + f"validation_pairs.csv", index=True)
            
        # train bamboo if needed
        if to_train_bamboo:
            print("Training BAMBOO in cycle", t)
            # generate train pairs for bamboo training
            bamboo_train_df = bin_U_df[bin_U_df["label"].isin(train_labels)].copy() # use this to keep original indexes (in this case you need to pass the full df to training (bin_U_df))
            #bamboo_train_df = bin_U_df[bin_U_df["label"].isin(train_labels)].reset_index(drop=True).copy() # use this if you want to reset the indexes (in this case you should pass only the training df)
            train_pairs = generate_balanced_pairs_df(bamboo_train_df, pairs_per_label=N_PAIRS, random_state=RANDOM_STATE)
            train_pairs.to_csv(out_dir + f"train_pairs.csv", index=True)
            
            #train_bamboo(bamboo_train_df, train_pairs, bamboo_filters_df, bamboo_output_file=bamboo_output_file, n_iterations=64, n_filters=0, max_workers=16)
            train_bamboo(bin_U_df, train_pairs, bamboo_filters_df, bamboo_output_file=bamboo_output_file, n_iterations=64, n_filters=0, max_workers=16)
            # from this point, bamboo is trained and the output is saved (all 64 bits as 64 rows in the bamboo_output_file csv)
        
        # train pf if needed
        if to_train_pf:
            print("Training PF in cycle", t)
            pf_train_df = bin_U_df[bin_U_df["label"].isin(train_labels)].copy()
            for num_bits in [8, 16, 32, 64]:
                pf_output_file = out_dir + f"pf/pf_indexes_{num_bits}bits.csv"
                pf_indexes = train_pf(pf_train_df, num_bits=num_bits)
                pf_indexes.to_csv(pf_output_file, index=False)
            # from this point, pf is trained and the output is saved (the indexes of the selected bits for 8, 16,32,64 bits)
        
        
        # validation for bamboo if needed
        if to_val_bamboo:
            print("Running validation for BAMBOO in cycle", t)
            #bamboo_val_df = bin_0_df[bin_0_df["label"].isin(val_labels)].copy()
            best_taus_bamboo = get_bamboo_validation_data(bin_0_df, validation_pairs_df, bamboo_output_file, bits_set=[8,16,32,64], hamming=False, roc_save_path=out_dir+"bamboo/")
            best_taus_bamboo_hamming = get_bamboo_validation_data(bin_0_df, validation_pairs_df, bamboo_output_file, bits_set=[8,16,32,64], hamming=True, roc_save_path=out_dir+"bamboo/")
            # save best taus in a csv
            best_taus_bamboo.to_csv(out_dir + "bamboo/bamboo_best_taus.csv", index=False)
            best_taus_bamboo_hamming.to_csv(out_dir + "bamboo/bamboo_best_taus_hamming.csv", index=False)
            print("BAMBOO_BEST_TAUS:", best_taus_bamboo)
        
        # validation for pf if needed
        if to_val_pf:
            print("Running validation for PF in cycle", t)
            #pf_val_df = bin_0_df[bin_0_df["label"].isin(val_labels)].copy()
            bits_set = [8,16,32,64]
            best_taus_pf = get_pf_validation_data(bin_0_df, validation_pairs_df, [out_dir + f"pf/pf_indexes_{num_bits}bits.csv" for num_bits in bits_set], bits_set=bits_set, roc_save_path=out_dir+"pf/")
            # save best taus in a csv
            best_taus_pf.to_csv(out_dir + "pf/pf_best_taus.csv", index=False)
            
            print("PF_BEST_TAUS:", best_taus_pf)
            
        # validation for pintor if needed
        if to_val_pintor:
            print("Running validation for PINTOR in cycle", t)
            best_tau_2 = get_pintor_validation_data(hex_df, validation_pairs_df, columns = ["Extended Capabilities", "Vendor Specific Tags"], roc_save_path=out_dir+"pintor/")
            best_tau_3 = get_pintor_validation_data(hex_df, validation_pairs_df, columns = ["HT Capabilities", "Extended Capabilities", "Vendor Specific Tags"], roc_save_path=out_dir+"pintor/")
            
            # save best tau in a txt file
            with open(out_dir + "pintor/pintor_best_tau.txt", "w") as f:
                f.write(f"Best tau: {best_tau_2}\n")
                f.write(f"Best tau: {best_tau_3}\n")
                
                
        # if any of the validation flags for dbscan is true, we need to have the validation combinations ready
        if to_val_dbscan_bamboo or to_val_dbscan_pf or to_val_dbscan_pintor:
            # try to load the validation combinations, or generate them if not present (same pairs for all methods, generated from bin_0_df filtered by val_labels)
            if not os.path.exists(out_dir + f"val_combinations.csv") or reset_data:
                # first of all compute the label combinations for the validation
                val_combinations_df = generate_combinations_df(val_labels, max_combinations_per_M=10, random_state=RANDOM_STATE)
                # save the label combinations in a csv for reference
                val_combinations_df.to_csv(out_dir + "val_combinations.csv", index=False)
            else:
                val_combinations_df = pd.read_csv(out_dir + f"val_combinations.csv")
                val_combinations_df["combination"] = val_combinations_df["combination"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                
        if to_val_dbscan_bamboo:
            print("Running DBSCAN validation for BAMBOO in cycle", t)
            if not os.path.exists(out_dir + "bamboo/bamboo_val_df_with_fprints.csv") or reset_data: 
                print("Computing fingerprint matrix for BAMBOO validation...")
                bamboo_val_df = bin_0_df[bin_0_df["label"].isin(val_labels)].copy()
                bamboo_val_df = get_bamboo_fprint_matrix(bamboo_val_df, bamboo_output_file)
                # save the bamboo_val_df with fingerprints for reference
                bamboo_val_df.to_csv(out_dir + "bamboo/bamboo_val_df_with_fprints.csv", index=False)
            else:
                print("Loading precomputed fingerprint matrix for BAMBOO validation...")
                bamboo_val_df = pd.read_csv(out_dir + "bamboo/bamboo_val_df_with_fprints.csv")
                bamboo_val_df["bamboo_fprint"] = bamboo_val_df["bamboo_fprint"].apply(lambda x: eval(x) if isinstance(x, str) else x)
            
            print("Fingerprint matrix ready for BAMBOO validation")
            validate_dbscan_on_bamboo_data(bamboo_val_df, val_combinations_df, output_folder=out_dir+"bamboo/")
        
        
        if to_val_dbscan_pf:
            print("Running DBSCAN validation for PF in cycle", t)
            if not os.path.exists(out_dir + "pf/pf_val_df_with_fprints.csv") or reset_data: 
                print("Computing fingerprint matrix for PF validation...")
                bits_set = [8,16,32,64]
                pf_val_df = bin_0_df[bin_0_df["label"].isin(val_labels)].copy()
                pf_val_df = get_pf_fprint_matrix(pf_val_df, index_filenames=[out_dir + f"pf/pf_indexes_{num_bits}bits.csv" for num_bits in bits_set], bits_set=bits_set)
                # save the pf_val_df with fingerprints for reference
                pf_val_df.to_csv(out_dir + "pf/pf_val_df_with_fprints.csv", index=False)
            
            else:
                print("Loading precomputed fingerprint matrix for PF validation...")
                pf_val_df = pd.read_csv(out_dir + "pf/pf_val_df_with_fprints.csv")
                for num_bits in [8,16,32,64]:
                    pf_val_df[f"pf_fprint_{num_bits}"] = pf_val_df[f"pf_fprint_{num_bits}"].apply(lambda x: eval(x) if isinstance(x, str) else x)
            
            print("Fingerprint matrix ready for PF validation")
            validate_dbscan_on_pf_data(pf_val_df, val_combinations_df, output_folder=out_dir+"pf/")

        if to_val_dbscan_pintor:
            if not os.path.exists(out_dir + "pintor/pintor_val_df_prepared.csv") or reset_data:
                print("Preparing PINTOR features for DBSCAN validation...")
                columns = ["HT Capabilities", "Extended Capabilities", "Vendor Specific Tags"]
                pintor_val_df = hex_df[hex_df["Label"].isin(val_labels)].copy()
                pintor_val_df = prepare_pintor_features(pintor_val_df, columns = columns)
            
                pintor_val_df.to_csv(out_dir + "pintor/pintor_val_df_prepared.csv", index=False)
            else:
                print("Loading precomputed PINTOR features for DBSCAN validation...")
                pintor_val_df = pd.read_csv(out_dir + "pintor/pintor_val_df_prepared.csv")
                
            print("Running DBSCAN validation for PINTOR in cycle", t)
            validate_dbscan_on_pintor_data(pintor_val_df, val_combinations_df, output_folder=out_dir+"pintor/")

        # if any of the test flags for dbscan is true, we need to have the test combinations ready
        if to_test_cluster_bamboo or to_test_cluster_pf or to_test_cluster_pintor:
            # try to load the test combinations, or generate them if not present (same pairs for all methods, generated from bin_0_df filtered by test_labels)
            if not os.path.exists(out_dir + f"test_combinations.csv") or reset_data:
                # first of all compute the label combinations for the test
                test_combinations_df = generate_combinations_df(test_labels, max_combinations_per_M=20, random_state=RANDOM_STATE)
                # save the label combinations in a csv for reference
                test_combinations_df.to_csv(out_dir + "test_combinations.csv", index=False)
            else:
                test_combinations_df = pd.read_csv(out_dir + f"test_combinations.csv")
                test_combinations_df["combination"] = test_combinations_df["combination"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                
        if to_test_cluster_bamboo:
            print("Running CLUSTER TEST for BAMBOO in cycle", t)
            
            bamboo_test_df = bin_0_df[bin_0_df["label"].isin(test_labels)].copy()
            bamboo_test_df = get_bamboo_fprint_matrix(bamboo_test_df, bamboo_output_file)
            # save the bamboo_test_df with fingerprints for reference
            bamboo_test_df.to_csv(out_dir + "bamboo/bamboo_test_df_with_fprints.csv", index=False)
            print("Fingerprint matrix ready for BAMBOO validation")
            for n_bits in [8,16,32,64]:
                if clustering_method == "groupby" or clustering_method == "any":
                    X = bamboo_test_df[["bamboo_fprint"]].copy()
                    X["bamboo_fprint"] = X["bamboo_fprint"].apply(lambda f: f[:n_bits])
                    res = test_groupby_on_combinations(X, bamboo_test_df["label"], test_combinations_df)
                    res.to_csv(out_dir + f"bamboo/groupby_test_results_{n_bits}_bits.csv", index=False)
                if clustering_method == "dbscan" or clustering_method == "any":
                    best_res_df = pd.read_csv(out_dir + f"bamboo/best_dbscan_params_{n_bits}_bits.csv")
                    res = test_dbscan_bamboo(bamboo_test_df, test_combinations_df, best_res_df, num_bits=n_bits)
                    res.to_csv(out_dir + f"bamboo/dbscan_test_results_{n_bits}_bits.csv", index=False)
                print(f"DBSCAN TEST for BAMBOO with {n_bits} bits completed in cycle {t}, size of results: {len(res)}")        
        
        if to_test_cluster_pf:
            print("Running CLUSTER TEST for PF in cycle", t)
            bits_set = [8,16,32,64]
            pf_test_df = bin_0_df[bin_0_df["label"].isin(test_labels)].copy()
            pf_test_df = get_pf_fprint_matrix(pf_test_df, index_filenames=[out_dir + f"pf/pf_indexes_{num_bits}bits.csv" for num_bits in bits_set], bits_set=bits_set)
            pf_test_df.to_csv(out_dir + "pf/pf_test_df_with_fprints.csv", index=False)
            
            # save the pf_val_df with fingerprints for reference
            for n_bits in [8,16,32,64]:
                best_res_df = pd.read_csv(out_dir + f"pf/best_dbscan_params_{n_bits}_bits.csv")
                if clustering_method == "groupby" or clustering_method == "any":
                    res = test_groupby_on_combinations(pf_test_df[f"pf_fprint_{n_bits}"], pf_test_df["label"], test_combinations_df)
                    res.to_csv(out_dir + f"pf/groupby_test_results_{n_bits}_bits.csv", index=False)
                if clustering_method == "dbscan" or clustering_method == "any":
                    res = test_dbscan_pf(pf_test_df, test_combinations_df, best_res_df, num_bits=n_bits)
                    res.to_csv(out_dir + f"pf/dbscan_test_results_{n_bits}_bits.csv", index=False)
                print(f"DBSCAN TEST for PF with {n_bits} bits completed in cycle {t}, size of results: {len(res)}")
                
        if to_test_cluster_pintor:
            print("Running CLUSTER TEST for PINTOR in cycle", t)
            columns = ["HT Capabilities", "Extended Capabilities", "Vendor Specific Tags"]
            pintor_test_df = hex_df[hex_df["Label"].isin(test_labels)].copy()
            
            if clustering_method == "groupby" or clustering_method == "any":
                pintor_test_df_norm = prepare_pintor_features(pintor_test_df, columns = columns, norm=True)
                pintor_test_df_norm.to_csv(out_dir + "pintor/pintor_test_df_prepared_norm.csv", index=False)
            else:
                pintor_test_df = prepare_pintor_features(pintor_test_df, columns = columns, norm=False)
                pintor_test_df.to_csv(out_dir + "pintor/pintor_test_df_prepared.csv", index=False)
                    
            for n_cols in [2,3]:
                best_res_df = pd.read_csv(out_dir + f"pintor/best_dbscan_params_{n_cols}_cols.csv")
                selected_columns = columns[-n_cols:]
                if clustering_method == "groupby" or clustering_method == "any":
                    res = test_groupby_on_combinations(pintor_test_df_norm[selected_columns], pintor_test_df_norm["Label"], test_combinations_df)
                    res.to_csv(out_dir + f"pintor/groupby_test_results_{n_cols}_cols.csv", index=False)
                if clustering_method == "dbscan" or clustering_method == "any":
                    res = test_dbscan_pintor(pintor_test_df, test_combinations_df, best_res_df=best_res_df, n_cols=n_cols)
                    res.to_csv(out_dir + f"pintor/dbscan_test_results_{n_cols}_cols.csv", index=False)
                print(f"DBSCAN TEST for PINTOR with {n_cols} columns completed in cycle {t}, size of results: {len(res)}")
        

    # compare the results for each method (in terms of AUC)
    if auc_comparison:
        compute_auc_comparison_figures(results_directories = [out_folder+f"cycle_{t}/" for t in range(N_FOLDS)], output_folder = out_folder)
    
    if dbscan_comparison:
        compute_dbscan_test_comparison_figures(results_directories = [out_folder+f"cycle_{t}/" for t in range(N_FOLDS)], output_folder = out_folder)
        
if __name__ == "__main__":
    main()
