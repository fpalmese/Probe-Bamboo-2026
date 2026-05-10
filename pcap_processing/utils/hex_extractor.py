import csv
from scapy.all import PcapReader
from scapy.layers.dot11 import Dot11, Dot11ProbeReq, Dot11Elt
import os
from utils import fileUtility
# Map IE IDs to your CSV columns
IE_SSID = 0
IE_SUPP_RATES = 1
IE_DS_PARAMS = 3
IE_EXT_CAP = 127
IE_EXT_SUPP_RATES = 50
IE_HT_CAP = 45
IE_VHT_CAP = 191
IE_VENDOR = 221
IE_HE_CAP_EXT = 255  # In 802.11ax, HE Capabilities is carried in an Extension IE (ID=255) with ext_id=35

HE_CAP_EXT_ID = 35   # Extension ID for HE Capabilities

def bytes_to_hex(b: bytes) -> str:
    return b.hex() if b else ""

def elt_chain(pkt):
    """Return list of Dot11Elt layers (information elements) in order."""
    elts = []
    e = pkt.getlayer(Dot11Elt)
    while isinstance(e, Dot11Elt):
        elts.append(e)
        e = e.payload.getlayer(Dot11Elt)
    return elts

def extract_probe_fields(pkt, label=""):
    """
    Returns a dict with the required CSV fields for a probe request frame.
    Assumes 802.11 frames (monitor mode capture preferred).
    """
    # Timestamp (epoch seconds with fraction)
    ts = float(pkt.time)

    dot11 = pkt.getlayer(Dot11)
    if not dot11:
        return None

    # MAC Address: for Probe Request, transmitter is addr2 (TA)
    mac = dot11.addr2 or ""

    # Length: scapy's len(pkt) is the captured frame length (includes radiotap if present)
    length = len(pkt)

    # Channel: try Radiotap if present; else blank
    channel = ""
    if pkt.haslayer("RadioTap"):
        rt = pkt.getlayer("RadioTap")
        # Some captures expose ChannelFrequency/Channel fields differently across versions
        # scapy may have "Channel" or "ChannelFrequency"
        ch = getattr(rt, "Channel", None)
        if ch is not None:
            # Often (freq, flags) tuple or int depending on scapy version
            # We'll try to interpret common patterns
            if isinstance(ch, tuple) and len(ch) > 0:
                # If tuple contains frequency first
                freq = ch[0]
                channel = str(freq)  # fallback if not mapped
            else:
                channel = str(ch)

    # DS Channel: from DS Parameter Set IE (ID=3) -> 1 byte channel number
    ds_channel = ""

    ht_cap = ""
    ext_cap = ""
    vendor_tags = []
    ssid = ""
    supp_rates = ""
    ext_supp_rates = ""
    vht_cap = ""
    he_cap = ""

    elts = elt_chain(pkt)

    # Helpers to keep the first instance (or merge as needed)
    def set_if_empty(varname, value):
        nonlocal ht_cap, ext_cap, ssid, supp_rates, ext_supp_rates, vht_cap, he_cap, ds_channel
        if not value:
            return
        if varname == "ht_cap" and not ht_cap:
            ht_cap = value
        elif varname == "ext_cap" and not ext_cap:
            ext_cap = value
        elif varname == "ssid" and ssid == "":
            ssid = value
        elif varname == "supp_rates" and not supp_rates:
            supp_rates = value
        elif varname == "ext_supp_rates" and not ext_supp_rates:
            ext_supp_rates = value
        elif varname == "vht_cap" and not vht_cap:
            vht_cap = value
        elif varname == "he_cap" and not he_cap:
            he_cap = value
        elif varname == "ds_channel" and not ds_channel:
            ds_channel = value

    for e in elts:
        ie_id = int(e.ID)

        # Raw info bytes for this IE (does not include ID/len bytes)
        try:
            info = bytes(e.info) if e.info else b""
        except:
            info = b""

        if ie_id == IE_SSID:
            # SSID is bytes; empty SSID is allowed (broadcast/wildcard)
            try:
                set_if_empty("ssid", info.decode("utf-8", errors="replace"))
            except Exception:
                set_if_empty("ssid", "")

        elif ie_id == IE_SUPP_RATES:
            set_if_empty("supp_rates", bytes_to_hex(info))

        elif ie_id == IE_EXT_SUPP_RATES:
            set_if_empty("ext_supp_rates", bytes_to_hex(info))

        elif ie_id == IE_DS_PARAMS:
            # DS Params is 1 byte: channel number
            if len(info) >= 1:
                set_if_empty("ds_channel", str(info[0]))

        elif ie_id == IE_HT_CAP:
            set_if_empty("ht_cap", bytes_to_hex(info))

        elif ie_id == IE_EXT_CAP:
            set_if_empty("ext_cap", bytes_to_hex(info))

        elif ie_id == IE_VHT_CAP:
            set_if_empty("vht_cap", bytes_to_hex(info))

        elif ie_id == IE_VENDOR:
            # Vendor Specific: keep hex blob(s); there can be multiple
            if info:
                vendor_tags.append(bytes_to_hex(info))

        elif ie_id == IE_HE_CAP_EXT:
            # Extension IE: first byte is ext_id, rest is extension data
            if len(info) >= 1:
                ext_id = info[0]
                ext_data = info[1:]
                if ext_id == HE_CAP_EXT_ID:
                    set_if_empty("he_cap", bytes_to_hex(ext_data))

    # Vendor Specific Tags: join multiple with ';' (or keep first—your choice)
    vendor_specific = ";".join(vendor_tags) if vendor_tags else ""

    # Channel column: you appear to want actual channel number.
    # If Radiotap gave us a frequency (e.g., 2412), you can map it to channel.
    # We’ll do a minimal mapping for 2.4 GHz common channels; else keep as-is.
    def freq_to_channel(freq_str):
        try:
            f = int(freq_str)
        except Exception:
            return freq_str
        # 2.4 GHz: channel 1 = 2412, increments by 5 MHz
        if 2412 <= f <= 2472 and (f - 2412) % 5 == 0:
            return str(1 + (f - 2412) // 5)
        # channel 14 special
        if f == 2484:
            return "14"
        # 5 GHz: many options; leave as frequency unless you extend mapping
        return freq_str

    if channel:
        channel = freq_to_channel(channel)

    return {
        "Timestamp": f"{ts:.6f}",
        "MAC Address": mac,
        "Channel": channel,
        "DS Channel": ds_channel,
        "HT Capabilities": ht_cap,
        "Extended Capabilities": ext_cap,
        "Vendor Specific Tags": vendor_specific,
        "SSID": ssid,
        "Supported Rates": supp_rates,
        "Extended Supported Rates": ext_supp_rates,
        "VHT Capabilities": vht_cap,
        "HE Capabilities": he_cap,
        "Length": str(length),
        "Label": label  # you can fill this from a map if you have one
    }

def pcap_to_csv(pcap_path, csv_path, label):
    fieldnames = [
        "Timestamp","MAC Address","Channel","DS Channel","HT Capabilities","Extended Capabilities",
        "Vendor Specific Tags","SSID","Supported Rates","Extended Supported Rates",
        "VHT Capabilities","HE Capabilities","Length","Label"
    ]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        with PcapReader(pcap_path) as pr:
            for pkt in pr:
                # Probe Request frames
                if pkt.haslayer(Dot11ProbeReq):
                    row = extract_probe_fields(pkt, label)
                    if row:
                        writer.writerow(row)
                        

def hex_extraction_folder(capture_path, output_path):
    # List all pcap files in the directory
    pcap_files = [f for f in os.listdir(capture_path) if f.endswith(".pcap")]

    for i, filename in enumerate(pcap_files):
        print(f"Hex Processing file: {filename} ({i+1}/{len(pcap_files)})")
        # Set output file name to the same as the PCAP file
        label = os.path.splitext(filename)[0]

        # Set the file path for the PCAP file
        file_path = os.path.join(capture_path, filename)

        fileUtility.checkCreatePath(output_path)
        pcap_to_csv(
            pcap_path=file_path,
            csv_path = os.path.join(output_path, f"{label}.csv"),
            label = label
        )
