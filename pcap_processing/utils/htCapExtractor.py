from scapy.layers.dot11 import Dot11Elt
from utils import logger,fieldUtility
from utils.constants import HT_CAP

# Extract HT capabilities from packet
def extractHTCapabilities(packet: Dot11Elt) -> list:
    try:
        ht_cap = packet.getlayer(Dot11Elt, ID=45)

        # Extract all fields into a list

        fields_list = []

        for field in HT_CAP:
            fields_list.append(getattr(ht_cap, field))

        return fieldUtility.fieldPadder(fields_list, 53)
    except:
        logger.log.debug("No HT capabilities found.")
        return fieldUtility.noneList(53)