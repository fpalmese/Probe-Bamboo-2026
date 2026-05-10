import matplotlib.pyplot as plt
import sys
sys.path.append("C:/Users/fabio/Ricerca/Codice/2026_Bamboo_Journal/well_structured")
from modules.utils.validation_utils import compute_auc_from_file
import pandas as pd
import seaborn as sns
from pandas.errors import EmptyDataError


def _read_csv_or_empty(csv_path, label):
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Missing results for {label}: {csv_path}")
    except EmptyDataError:
        print(f"Empty results for {label}: {csv_path}")
    return pd.DataFrame()

def compute_auc_comparison_figures(results_directories = [], output_folder="."):
        # results bamboo
        bamboo_results_8 = []
        bamboo_results_16 = []
        bamboo_results_32 = []
        bamboo_results_64 = []
        pf_results_8 = []
        pf_results_16 = []
        pf_results_32 = []
        pf_results_64 = []
        pintor_results_2 = []
        pintor_results_3 = []
        try:        
            for dir in results_directories:
                bamboo_dir = dir + f"/bamboo/"
                pf_dir = dir + f"/pf/"
                pintor_dir = dir + f"/pintor/"
                
                bamboo_results_8.append(compute_auc_from_file(bamboo_dir + "bamboo_roc_curve_data_8.csv"))
                bamboo_results_16.append(compute_auc_from_file(bamboo_dir + "bamboo_roc_curve_data_16.csv"))
                bamboo_results_32.append(compute_auc_from_file(bamboo_dir + "bamboo_roc_curve_data_32.csv"))
                bamboo_results_64.append(compute_auc_from_file(bamboo_dir + "bamboo_roc_curve_data_64.csv"))
                
                pf_results_8.append(compute_auc_from_file(pf_dir + "pf_roc_curve_data_8.csv"))
                pf_results_16.append(compute_auc_from_file(pf_dir + "pf_roc_curve_data_16.csv"))
                pf_results_32.append(compute_auc_from_file(pf_dir + "pf_roc_curve_data_32.csv"))
                pf_results_64.append(compute_auc_from_file(pf_dir + "pf_roc_curve_data_64.csv"))
                
                pintor_results_2.append(compute_auc_from_file(pintor_dir + "pintor_roc_curve_data_2.csv"))
                pintor_results_3.append(compute_auc_from_file(pintor_dir + "pintor_roc_curve_data_3.csv"))
        except Exception as e:
            print(f"Error reading results: {e}")
            return
            
        ylim = (0.5, 1.0)
        # create one chart with three plots, one per method. in each plot have the lines for the different bit lengths (8,16,32,64 for bamboo and pf, 2 and 3 for pintor)
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(bamboo_results_8, label="BAMBOO 8 bits")
        plt.plot(bamboo_results_16, label="BAMBOO 16 bits")
        plt.plot(bamboo_results_32, label="BAMBOO 32 bits")
        plt.plot(bamboo_results_64, label="BAMBOO 64 bits")
        plt.title("BAMBOO AUC")
        plt.xlabel("Validation Cycles")
        plt.ylabel("AUC")
        plt.legend()
        plt.ylim(ylim)
        plt.grid()
        
        plt.subplot(1, 3, 2)
        plt.plot(pf_results_8, label="PF 8 bits")
        plt.plot(pf_results_16, label="PF 16 bits")
        plt.plot(pf_results_32, label="PF 32 bits")
        plt.plot(pf_results_64, label="PF 64 bits")
        plt.title("PF AUC")
        plt.xlabel("Validation Cycles")
        plt.ylabel("AUC")
        plt.legend()
        plt.ylim(ylim)
        plt.grid()
        
        plt.subplot(1, 3, 3)
        plt.plot(pintor_results_2, label="PINTOR 2 columns")
        plt.plot(pintor_results_3, label="PINTOR 3 columns")
        plt.title("PINTOR AUC")
        plt.xlabel("Validation Cycles")
        plt.ylabel("AUC")
        plt.legend()
        plt.tight_layout()
        plt.ylim(ylim)
        plt.grid()
        
        #plt.savefig(out_folder + "comparison_auc.png")
        plt.savefig(output_folder + "comparison_auc_lineplot.pdf")

        # now do three charts with 4 box plots each, one per method, comparing the AUC distributions for the different bit lengths (8,16,32,64 for bamboo and pf, 2 and 3 for pintor)
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.boxplot([bamboo_results_8, bamboo_results_16, bamboo_results_32, bamboo_results_64], labels=["8 bits", "16 bits", "32 bits", "64 bits"])
        plt.title("BAMBOO AUC Distribution")
        plt.ylabel("AUC")
        plt.ylim(ylim)
        plt.grid()
        
        plt.subplot(1, 3, 2)
        plt.boxplot([pf_results_8, pf_results_16, pf_results_32, pf_results_64], labels=["8 bits", "16 bits", "32 bits", "64 bits"])
        plt.title("PF AUC Distribution")
        plt.ylabel("AUC")
        plt.ylim(ylim)
        plt.grid()
        
        plt.subplot(1, 3, 3)
        plt.boxplot([pintor_results_2, pintor_results_3],labels=["2 columns", "3 columns"])
        plt.title("PINTOR AUC Distribution")
        plt.ylabel("AUC")
        plt.ylim(ylim)
        
        plt.grid()
        plt.tight_layout()
        #plt.savefig(out_folder + "comparison_auc_boxplots.png")
        plt.savefig(output_folder + "comparison_auc_boxplots.pdf")
        #plt.show()
        
def plot_clustering_metrics(df_list, label_list, output_file):
    """
    Plots clustering metrics (Homogeneity, Completeness, V-Measure, RMSE) from multiple CSV files.
    
    Parameters:
    df_list (list of pd.DataFrame): List of DataFrames containing the clustering metrics.
    label_list (list of str): List of labels for each DataFrame.
    output_file (str): Output filename for the plot.
    """
    if not df_list or not label_list or len(df_list) != len(label_list):
        print("Invalid input: Ensure that df_list and label_list are non-empty and of the same length.")
        return
    
    # Define colors and linestyles
    colors = sns.color_palette("tab10")
    linestyles = ['-', '--', '-.', ':']  # List of line styles

    sns.set(style="whitegrid")
    _, axes = plt.subplots(4, 1, figsize=(8.3, 11.7))

    for i, (df, label) in enumerate(zip(df_list, label_list)):
        # Read the CSV file
        if 'length' not in df.columns or 'homogeneity' not in df.columns or \
           'completeness' not in df.columns or 'v_measure' not in df.columns or 'rmse' not in df.columns:
            print(f"DataFrame for {label} is missing required columns. Skipping.")
            continue
        
        df = df.sort_values('length')
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]

        # Plotting each metric
        sns.lineplot(ax=axes[0],x='length',y='v_measure',data=df,label=f'{label}',color=color,linestyle=linestyle,marker='o',linewidth=2)
        sns.lineplot(ax=axes[1],x='length',y='homogeneity',data=df,label=f'{label}',color=color,linestyle=linestyle,marker='o',linewidth=2)
        sns.lineplot(ax=axes[2],x='length',y='completeness',data=df,label=f'{label}',color=color,linestyle=linestyle,marker='o',linewidth=2)
        sns.lineplot(ax=axes[3],x='length',y='rmse',data=df,label=f'{label}',color=color,linestyle=linestyle,marker='o',linewidth=2)
    
    # Set titles and labels for subplots
    axes[0].set_title('V-Measure Score', fontsize=16)
    axes[0].set_xlabel('', fontsize=14)
    axes[0].set_ylabel('Score', fontsize=14)
    axes[0].tick_params(axis='both', which='major', labelsize=12)

    axes[1].set_title('Homogeneity Score', fontsize=16)
    axes[1].set_xlabel('', fontsize=14)
    axes[1].set_ylabel('Score', fontsize=14)
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    
    axes[2].set_title('Completeness Score', fontsize=16)
    axes[2].set_xlabel('', fontsize=14)
    axes[2].set_ylabel('Score', fontsize=14)
    axes[2].tick_params(axis='both', which='major', labelsize=12)
    
    axes[3].set_title('RMSE', fontsize=16)
    axes[3].set_xlabel('Number of Devices', fontsize=14)
    axes[3].set_ylabel('RMSE', fontsize=14)
    axes[3].tick_params(axis='both', which='major', labelsize=12)
    
    # Adjust spacing between subplots
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot as a PDF file
    if output_file:
        plt.savefig(output_file, format='pdf')
    
    plt.show()
  
def compute_dbscan_test_comparison_figures(results_directories = [], output_folder="."):
    # read the csv of all the bamboo results
    bamboo_results_8 = pd.DataFrame()
    bamboo_results_16 = pd.DataFrame()
    bamboo_results_32 = pd.DataFrame()
    bamboo_results_64 = pd.DataFrame()
    pf_results_8 = pd.DataFrame()
    pf_results_16 = pd.DataFrame()
    pf_results_32 = pd.DataFrame()
    pf_results_64 = pd.DataFrame()
    pintor_results_2 = pd.DataFrame()
    pintor_results_3 = pd.DataFrame()

    for res_dir in results_directories:
        bamboo_dir = res_dir + f"bamboo/"
        pf_dir = res_dir + f"pf/"
        pintor_dir = res_dir + f"pintor/"
        
        bamboo_results_8 = pd.concat([bamboo_results_8, _read_csv_or_empty(bamboo_dir + "dbscan_test_results_8_bits.csv", "BAMBOO 8 bits")], ignore_index=True)
        bamboo_results_16 = pd.concat([bamboo_results_16, _read_csv_or_empty(bamboo_dir + "dbscan_test_results_16_bits.csv", "BAMBOO 16 bits")], ignore_index=True)
        bamboo_results_32 = pd.concat([bamboo_results_32, _read_csv_or_empty(bamboo_dir + "dbscan_test_results_32_bits.csv", "BAMBOO 32 bits")], ignore_index=True)
        bamboo_results_64 = pd.concat([bamboo_results_64, _read_csv_or_empty(bamboo_dir + "dbscan_test_results_64_bits.csv", "BAMBOO 64 bits")], ignore_index=True)
        
        pf_results_8 = pd.concat([pf_results_8, _read_csv_or_empty(pf_dir + "dbscan_test_results_8_bits.csv", "PF 8 bits")], ignore_index=True)
        pf_results_16 = pd.concat([pf_results_16, _read_csv_or_empty(pf_dir + "dbscan_test_results_16_bits.csv", "PF 16 bits")], ignore_index=True)
        pf_results_32 = pd.concat([pf_results_32, _read_csv_or_empty(pf_dir + "dbscan_test_results_32_bits.csv", "PF 32 bits")], ignore_index=True)
        pf_results_64 = pd.concat([pf_results_64, _read_csv_or_empty(pf_dir + "dbscan_test_results_64_bits.csv", "PF 64 bits")], ignore_index=True)
        
        pintor_results_2 = pd.concat([pintor_results_2, _read_csv_or_empty(pintor_dir + "dbscan_test_results_2_cols.csv", "PINTOR 2 columns")], ignore_index=True)
        pintor_results_3 = pd.concat([pintor_results_3, _read_csv_or_empty(pintor_dir + "dbscan_test_results_3_cols.csv", "PINTOR 3 columns")], ignore_index=True)
        
    # plot the results in a figure for all the bamboo
    plot_clustering_metrics([bamboo_results_8, bamboo_results_16, bamboo_results_32, bamboo_results_64], 
                            ["BAMBOO 8 bits", "BAMBOO 16 bits", "BAMBOO 32 bits", "BAMBOO 64 bits"], 
                            output_file = output_folder + "bamboo_dbscan_comparison.pdf")
    # plot the results in a figure for all the pf
    plot_clustering_metrics([pf_results_8, pf_results_16, pf_results_32, pf_results_64], 
                            ["PF 8 bits", "PF 16 bits", "PF 32 bits", "PF 64 bits"], 
                            output_file = output_folder + "pf_dbscan_comparison.pdf")
    # plot the results in a figure for all the pintor
    plot_clustering_metrics([pintor_results_2, pintor_results_3],
                            ["PINTOR 2 columns", "PINTOR 3 columns"], 
                            output_file = output_folder + "pintor_dbscan_comparison.pdf")
    
    # now compare bamboo and pf at 32-64 and pintor at 3
    plot_clustering_metrics([bamboo_results_32, bamboo_results_64, pf_results_32, pf_results_64, pintor_results_3],
                            ["BAMBOO 32 bits", "BAMBOO 64 bits", "PF 32 bits", "PF 64 bits", "PINTOR 3 columns"], 
                            output_file = output_folder + "overall_dbscan_comparison.pdf")
