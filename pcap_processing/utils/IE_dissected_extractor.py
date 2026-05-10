from utils import extendedCapExtractor,htCapExtractor, logger, fieldUtility
from utils.constants import EXTENDED_CAP
from scapy.all import rdpcap
from scapy.layers.dot11 import Dot11Elt, Dot11FCS

import numpy as np


# Convert channel frequency into channel number
def frequencyToChannel(frequency: int) -> int:
    """Convert channel frequency into channel number
    Args:
        frequency (int): Channel frequency
    Returns:
        int: Channel number
    """
    return int((frequency - 2407) / 5)


# Extract timestamp from packet
def extractTimestamp(packet) -> float:
    """Extract timestamp from packet
    Args:
        packet (scapy.layers.dot11.Dot11): Scapy packet
    Returns:
        float: Timestamp
    """
    return packet.time


# Extract source MAC address from packet
def extractMAC(packet) -> str:
    return packet.addr2


# Extract channel number from packet
def extractChannel(packet) -> int:
    return frequencyToChannel(packet.Channel)


# Extract DS channel number from packet
def extractDSChannel(packet):
    try:
        return packet.getlayer(Dot11Elt, ID=3).channel
    except:
        logger.log.debug("No DS channel found.")
        return None


# Extract extended capabilities from packet
def extractExtendedCapabilities(packet) -> list:
    try:
        extendedCapHex = packet.getlayer(Dot11Elt, ID=127).info.hex()
        extendedCapBin = extendedCapExtractor.hex_string_to_binary(extendedCapHex)
        extendedCap = extendedCapExtractor.extract_fields_from_binary(
            EXTENDED_CAP, extendedCapBin
        )
        return fieldUtility.fieldPadder(extendedCap, 72)
    except:
        logger.log.debug("No extended capabilities found.")
        return fieldUtility.noneList(72)


# Extract Sequence Number (SN) from packet


def extractSN(packet) -> int:
    try:
        sn = (packet.SC / 16)
        return sn
    except:
        logger.log.warning("No Sequence Number (SN) found")
        return 0


# Extract vendor specific tags from packet
def extractVendorSpecificTags(packet):
    try:
        return packet.getlayer(Dot11Elt, ID=221).info.hex()
    except:
        logger.log.debug("No vendor specific tags found.")
        return None


# Extract SSID from packet
def extractSSID(packet):
    try:
        return packet.getlayer(Dot11Elt, ID=0).info.decode()
    except:
        logger.log.debug("No SSID found.")
        return None


# Extract supported rates from packet
def extractSupportedRates(packet):
    try:
        supportedRates = []

        rates = packet.getlayer(Dot11Elt, ID=1).rates

        for rate in rates:
            supportedRates.append(rate / 2)

        return fieldUtility.fieldPadder(supportedRates, 8)

    except:
        logger.log.debug("No supported rates found.")
        return fieldUtility.noneList(8)


# Extract extended supported rates from packet
def extractExtendedSupportedRates(packet):
    try:
        extendedSupportedRates = []

        rates = packet.getlayer(Dot11Elt, ID=50).rates

        for rate in rates:
            extendedSupportedRates.append(rate / 2)

        return fieldUtility.fieldPadder(extendedSupportedRates, 8)

    except:
        logger.log.debug("No extended supported rates found.")
        return fieldUtility.noneList(8)


# Extract VHT capabilities from packet
def extractVHTCapabilities(packet):
    try:
        return packet.getlayer(Dot11Elt, ID=191).info.hex()
    except:
        logger.log.debug("No VHT capabilities found.")
        return None


# Extract HE capabilities from packet
def extractHECapabilities(packet):
    try:
        return packet.getlayer(Dot11Elt, ID=255).info.hex()
    except:
        logger.log.debug("No HE capabilities found.")
        return None


def extractHTCapabilities(packet: Dot11Elt):
    return htCapExtractor.extractHTCapabilities(packet)
