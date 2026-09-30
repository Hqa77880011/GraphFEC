"""Systematic Vandermonde erasure coding over GF(2^8), primitive polynomial 0x11D."""

import numpy as np

EXP = np.zeros(510, dtype=np.uint8)
LOG = np.zeros(256, dtype=np.int64)
value = 1
for i in range(255):
    EXP[i] = value
    LOG[value] = i
    value <<= 1
    if value & 256:
        value ^= 0x11D
EXP[255:] = EXP[:255]
MUL = EXP[LOG[:, None] + LOG[None, :]]
MUL[0, :] = 0
MUL[:, 0] = 0


def matmul(a, b):
    a, b = np.asarray(a, dtype=np.uint8), np.asarray(b, dtype=np.uint8)
    result = np.zeros((len(a), b.shape[1]), dtype=np.uint8)
    for j in range(a.shape[1]):
        result ^= MUL[a[:, j, None], b[j][None, :]]
    return result


def row_reduce(matrix, payload):
    a, b = matrix.copy(), payload.copy()
    pivot_row, pivots = 0, []
    for col in range(a.shape[1]):
        candidates = np.flatnonzero(a[pivot_row:, col])
        if len(candidates) == 0:
            continue
        row = pivot_row + candidates[0]
        a[[pivot_row, row]], b[[pivot_row, row]] = a[[row, pivot_row]], b[[row, pivot_row]]
        inverse = EXP[255 - LOG[a[pivot_row, col]]]
        a[pivot_row] = MUL[inverse, a[pivot_row]]
        b[pivot_row] = MUL[inverse, b[pivot_row]]
        factors = a[:, col].copy()
        factors[pivot_row] = 0
        a ^= MUL[factors[:, None], a[pivot_row][None, :]]
        b ^= MUL[factors[:, None], b[pivot_row][None, :]]
        pivots.append(col)
        pivot_row += 1
        if pivot_row == len(a):
            break
    return a, b, pivots


def systematic_matrix(data_packets, parity_packets):
    """Construct an MDS (M+R, M) systematic Reed–Solomon generator."""
    n = data_packets + parity_packets
    if not 1 <= data_packets <= n <= 255:
        raise ValueError("Require 1 <= M <= M+R <= 255")
    x = np.arange(1, n + 1, dtype=np.uint8)
    vandermonde = np.ones((n, data_packets), dtype=np.uint8)
    for j in range(1, data_packets):
        vandermonde[:, j] = MUL[vandermonde[:, j - 1], x]
    _, inverse, _ = row_reduce(vandermonde[:data_packets], np.eye(data_packets, dtype=np.uint8))
    return matmul(vandermonde, inverse)


def unequal_matrix(scores, parity_packets):
    """Allocate parity between high/low-saliency halves, coding each half separately."""
    m = len(scores)
    g = np.zeros((m + parity_packets, m), dtype=np.uint8)
    g[:m] = np.eye(m, dtype=np.uint8)
    if parity_packets == 0:
        return g
    high, low = np.array_split(np.argsort(-np.asarray(scores), kind="stable"), 2)
    share = (np.asarray(scores)[high].sum() + 1e-12) / (np.asarray(scores).sum() + 1e-12)
    r_high = min(parity_packets, max((parity_packets + 1) // 2, round(parity_packets * share)))
    offset = m
    for group, r in ((high, r_high), (low, parity_packets - r_high)):
        if r and len(group):
            g[offset : offset + r, group] = systematic_matrix(len(group), r)[len(group) :]
            offset += r
    return g


def decode(generator, received, mask, fallback):
    """Recover uniquely determined packets and retain means for unresolved variables."""
    result = fallback.copy()
    if not np.any(mask):
        return result
    a, b, pivots = row_reduce(generator[mask], received[mask])
    for row, col in enumerate(pivots):
        if np.count_nonzero(a[row]) == 1:
            result[col] = b[row]
    return result
