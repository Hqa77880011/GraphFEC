"""Profile-bound datagrams for file export or an application-managed transport."""

import struct
import uuid
import zlib

import numpy as np
import torch

# magic, profile UUID, sample, packet index/count, C/H/W, payload length, CRC32
HEADER = struct.Struct("<4s16sIHHHHHII")


def encode_datagrams(protection, wire, shape, profile_id, sample_id=0):
    """Serialize one sample into equal-size data/parity payloads in uint8 mode."""
    channels, h, w = shape
    datagrams = []
    for index, row in enumerate(wire.detach().cpu().numpy()):
        float_parity = (
            index >= protection.cfg.packets
            and protection.parity is not None
            and protection.cfg.parity_precision == "float32"
        )
        payload = row.astype("<f4" if float_parity else np.uint8).tobytes()
        header = HEADER.pack(
            b"GFC1",
            uuid.UUID(profile_id).bytes,
            sample_id,
            index,
            protection.total_packets,
            channels,
            h,
            w,
            len(payload),
            zlib.crc32(payload),
        )
        datagrams.append(header + payload)
    return datagrams


def decode_datagrams(protection, datagrams, shape, profile_id, sample_id=0):
    """Validate datagrams, construct the mask and restore a tensor of wire symbols."""
    _, h, w = shape
    d = protection.packetizer.capacity * h * w
    wire = torch.zeros(1, protection.total_packets, d)
    mask = torch.zeros(1, protection.total_packets, dtype=torch.bool)
    for datagram in datagrams:
        if len(datagram) < HEADER.size:
            raise ValueError("Truncated packet header")
        magic, profile, sample, index, count, c, ph, pw, size, crc = HEADER.unpack(
            datagram[: HEADER.size]
        )
        if (
            magic != b"GFC1"
            or profile != uuid.UUID(profile_id).bytes
            or sample != sample_id
            or count != protection.total_packets
            or (c, ph, pw) != shape
            or index >= count
        ):
            raise ValueError("Packet profile, sample, shape or index mismatch")
        payload = datagram[HEADER.size :]
        float_parity = (
            index >= protection.cfg.packets
            and protection.parity is not None
            and protection.cfg.parity_precision == "float32"
        )
        dtype = np.dtype("<f4" if float_parity else np.uint8)
        if size != len(payload) or size != d * dtype.itemsize or zlib.crc32(payload) != crc:
            raise ValueError("Packet payload size or checksum mismatch")
        if mask[0, index]:
            raise ValueError("Duplicate packet index")
        wire[0, index] = torch.from_numpy(np.frombuffer(payload, dtype=dtype).copy()).float()
        mask[0, index] = True
    return wire, mask
