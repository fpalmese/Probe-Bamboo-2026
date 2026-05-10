import os
import sys

PCAPS_ROOT = r"C:/Users/fabio/Ricerca/Codice/2026_Bamboo_Journal/well_structured/data/pcap/Pintor_New/"

# Common magic signatures
PCAP_MAGIC_LE = b'\xd4\xc3\xb2\xa1'   # libpcap little-endian (microsecond)
PCAP_MAGIC_BE = b'\xa1\xb2\xc3\xd4'   # libpcap big-endian (microsecond)
PCAP_MAGIC_LE_NS = b'\x4d\x3c\xb2\xa1'  # nanosecond variants (less common)
PCAP_MAGIC_BE_NS = b'\xa1\xb2\x3c\x4d'
PCAPNG_MAGIC = b'\x0A\x0D\x0D\x0A'    # pcapng Section Header Block type

def detect_file_type(path):
    """Return 'pcap', 'pcapng', or None if unknown."""
    try:
        with open(path, "rb") as f:
            head = f.read(4)
    except Exception:
        return None

    if head in (PCAP_MAGIC_LE, PCAP_MAGIC_BE, PCAP_MAGIC_LE_NS, PCAP_MAGIC_BE_NS):
        return "pcap"
    if head == PCAPNG_MAGIC:
        return "pcapng"
    return None

def concat_files_binary(file_list, output_file, filetype):
    """Concatenate files according to type:
       - pcap: write first file completely; for subsequent files skip first 24 bytes (global header)
       - pcapng: append each file completely (multiple SHBs are allowed)
    """
    if filetype == "pcap":
        global_header_size = 24
        with open(output_file, "wb") as out:
            for i, p in enumerate(file_list):
                with open(p, "rb") as fh:
                    if i == 0:
                        out.write(fh.read())
                    else:
                        fh.seek(global_header_size)
                        out.write(fh.read())
    elif filetype == "pcapng":
        # pcapng may contain multiple Section Header Blocks; appending full files is valid
        with open(output_file, "wb") as out:
            for p in file_list:
                with open(p, "rb") as fh:
                    out.write(fh.read())
    else:
        raise ValueError("Unknown filetype for concatenation")

def process_all_subfolders(root_dir):
    if not os.path.isdir(root_dir):
        print(f"[ERROR] Root path not found: {root_dir}")
        return

    for subdir in sorted(os.listdir(root_dir)):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        # collect pcap and pcapng files
        candidates = []
        for f in os.listdir(subdir_path):
            fl = f.lower()
            if fl.endswith(".pcap") or fl.endswith(".pcapng"):
                candidates.append(os.path.join(subdir_path, f))

        if not candidates:
            print(f"[!] No pcap/pcapng files found in {subdir_path}")
            continue

        # deterministic order
        candidates.sort()

        # detect types for all files in this folder
        types = {}
        for p in candidates:
            t = detect_file_type(p)
            types.setdefault(t, []).append(p)

        # if unknown file type present, warn and skip those files
        if None in types:
            unknowns = types.pop(None)
            print(f"[!] Skipping {len(unknowns)} file(s) with unknown/unsupported format in {subdir_path}:")
            for u in unknowns:
                print("    -", os.path.basename(u))

        # if both pcap and pcapng are present, refuse to merge them into one file
        present_types = sorted(types.keys())
        if not present_types:
            print(f"[!] No supported pcap/pcapng files to process in {subdir_path}")
            continue
        if len(present_types) > 1:
            print(f"[ERROR] Both pcap and pcapng files found in the same folder ({subdir_path}).")
            print("        Merging mixed formats into a single output file is unsafe. Please separate them.")
            print("        Found types:", ", ".join(present_types))
            continue

        filetype = present_types[0]
        pcap_files = types[filetype]

        output_ext = ".pcap" if filetype == "pcap" else ".pcapng"
        output_pcap = os.path.join(root_dir, f"{subdir}{output_ext}")

        try:
            concat_files_binary(pcap_files, output_pcap, filetype)
            print(f"[+] Created {output_pcap} from {len(pcap_files)} {filetype} file(s).")
        except Exception as e:
            print(f"[ERROR] Failed to create {output_pcap}: {e}")

if __name__ == "__main__":
    # optional: allow overriding the root from CLI
    root = PCAPS_ROOT
    if len(sys.argv) > 1:
        root = sys.argv[1]
    process_all_subfolders(root)
