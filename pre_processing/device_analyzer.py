import pandas as pd
from configparser import ConfigParser
import os
import matplotlib.pyplot as plt

# import the interim


config = ConfigParser()
config_filename = "config_preprocessing.ini"
full_config_name = os.path.join(os.path.dirname(__file__), config_filename)
config.read(full_config_name)
interim_path = config["DEFAULT"]["output_path"]
interim_file = os.path.join(interim_path, "binary_U_balanced.csv")

interim_df = pd.read_csv(interim_file, dtype=str)

# sort by label and then by concatenated column
interim_df = interim_df.sort_values(by=["label", "concatenated"], ignore_index=True)

print(interim_df.shape)

# for each label print the number of entries and the number of unique entries (concatenated column)
# then make a bar plot with the number of unique entries for each label
labels = interim_df["label"].unique()
labels = [l for l in labels if not l.endswith("AR")]  # exclude strings ending with "AR"
n_entries = []
n_uniques = []
for label in labels:
    label_df = interim_df[interim_df["label"] == label]
    num_entries = label_df.shape[0]
    num_unique_entries = label_df["concatenated"].nunique()
    n_entries.append(num_entries)
    n_uniques.append(num_unique_entries)
    print(f"Label: {label} - Number of entries: {num_entries}, Number of unique entries: {num_unique_entries}")

fig, axes = plt.subplots(2, 1, figsize=(25, 12))
plt.sca(axes[0])
plt.bar(labels, n_uniques, color='skyblue')
plt.xlabel('Label')
plt.ylabel('Number of Unique Concatenated Entries')
plt.title('Unique Concatenated Entries per Label')

plt.sca(axes[1])
plt.bar(labels, n_entries, color='salmon')

plt.xlabel('Label')
plt.ylabel('Number of Total Entries')
plt.title('Total Entries per Label')
# save fig
#fig.savefig("label_entry_counts2.png")

plt.show()


