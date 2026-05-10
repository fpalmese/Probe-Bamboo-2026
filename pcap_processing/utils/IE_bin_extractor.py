from scapy.layers.dot11 import Dot11Elt

from . import dictionaries

# Extract source MAC address from packet
def getMAC(packet : Dot11Elt) -> str:
    return str(packet.addr2)

def getElementIDText(elementid: int) -> str:
    try:
        return dictionaries.ELEMENT_IDs[elementid]
    except:
        return "unknown"