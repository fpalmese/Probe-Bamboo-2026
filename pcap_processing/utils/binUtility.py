from bitstring import BitArray
from scapy.all import *
import scapy.layers.dot11


def getMACLayerBits(packet: scapy.layers.dot11.RadioTap) -> BitArray:
    headerLength = packet.getlayer(RadioTap).len
    headerBits = headerLength * 8

    # Convert the packet to bytes
    packet_bytes = bytes(packet)

    # Convert bytes to binary string
    binary_string = "".join(format(byte, "08b") for byte in packet_bytes)

    return binary_string[headerBits:]


def readElementID(packet: str) -> str:
    return int(packet[0:8], 2)


def readBinElementID(packet: str) -> str:
    elementID = packet[0:8]
    return elementID


def readBinLength(packet: str) -> str:
    length = packet[8:16]
    return length


def convertBinLength(packet: str) -> int:
    bitLength = int(packet[8:16], 2) * 8
    return int(bitLength)


def readBinField(packet: str) -> str:
    field = packet[16 : 16 + convertBinLength(packet)]
    return field
