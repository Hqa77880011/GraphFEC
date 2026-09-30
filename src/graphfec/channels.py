"""Packet reception masks; True means received, trace value 1 means erased."""

from pathlib import Path

import numpy as np
import torch


def sample_mask(batch, packets, loss, channel, rng):
    if not 0 <= loss <= 1:
        raise ValueError("Loss probability must lie in [0, 1]")
    if channel == "bernoulli":
        received = rng.random((batch, packets)) >= loss
    elif channel == "gilbert":
        if loss > 0.8:
            raise ValueError("Four-packet Gilbert model requires loss <= 0.8")
        bad = rng.random(batch) < loss
        received = np.empty((batch, packets), dtype=bool)
        p_gb = 0.25 * loss / (1 - loss)
        for i in range(packets):
            received[:, i] = ~bad
            transition = rng.random(batch)
            bad = np.where(bad, transition >= 0.25, transition < p_gb)
    else:
        raise ValueError(f"Unknown channel: {channel}")
    return torch.from_numpy(received)


def load_trace(path, examples, packets):
    values = np.loadtxt(Path(path), delimiter="," if str(path).endswith(".csv") else None)
    values = values.reshape(-1)
    if not np.isin(values, (0, 1)).all():
        raise ValueError("Trace must contain only 0 (received) and 1 (erased)")
    if len(values) < examples * packets:
        raise ValueError(f"Trace needs at least {examples * packets} packet events")
    return torch.from_numpy(values[: examples * packets].reshape(examples, packets) == 0)
