"""Channel statistics, sparse graphs, and balanced packet assignments."""

import math

import torch
from torch import nn


class ChannelStatistics:
    def __init__(self, channels, beta=0.95):
        self.beta, self.count = beta, 0
        self.total = torch.zeros(channels, dtype=torch.float64)
        self.cross = torch.zeros(channels, channels, dtype=torch.float64)
        self.ema_mean = torch.zeros_like(self.total)
        self.ema_cross = torch.zeros_like(self.cross)
        self.low = torch.full((channels,), float("inf"))
        self.high = torch.full((channels,), -float("inf"))

    def update(self, activations):
        z = activations.detach().cpu()
        pooled = z.mean((2, 3)).double()
        mean, cross = pooled.mean(0), pooled.T @ pooled / len(pooled)
        beta = self.beta if self.count else 0.0
        self.ema_mean = beta * self.ema_mean + (1 - beta) * mean
        self.ema_cross = beta * self.ema_cross + (1 - beta) * cross
        self.total += pooled.sum(0)
        self.cross += pooled.T @ pooled
        self.count += len(pooled)
        self.low = torch.minimum(self.low, z.amin((0, 2, 3)))
        self.high = torch.maximum(self.high, z.amax((0, 2, 3)))

    def correlation(self, ema=True):
        if self.count < 2:
            raise ValueError("Correlation estimation needs at least two examples")
        mean = self.ema_mean if ema else self.total / self.count
        cross = self.ema_cross if ema else self.cross / self.count
        cov = cross - mean[:, None] * mean[None, :]
        std = cov.diag().clamp_min(0).sqrt()
        corr = (cov / (std[:, None] * std[None, :] + 1e-12)).abs().clamp(0, 1)
        corr.fill_diagonal_(0)
        return corr.float()


def sparsify(correlation, k):
    c = len(correlation)
    _, indices = torch.topk(correlation, min(k, c - 1), dim=1)
    sparse = torch.zeros_like(correlation).scatter(1, indices, correlation.gather(1, indices))
    sparse = torch.maximum(sparse, sparse.T)
    sparse.fill_diagonal_(0)
    return sparse


def assign_channels(graph, packets, balance=0.1, mode="graphfec", seed=17):
    """Apply Algorithm 1; random and cluster modes retain the same capacity."""
    c, capacity = len(graph), math.ceil(len(graph) / packets)
    groups = [[] for _ in range(packets)]
    if mode == "random":
        order = torch.randperm(c, generator=torch.Generator().manual_seed(seed)).tolist()
        for j, channel in enumerate(order):
            groups[j % packets].append(channel)
    else:
        order = sorted(range(c), key=lambda i: (-float(graph[i].sum()), i))
        for channel in order:
            candidates = []
            for r, group in enumerate(groups):
                if len(group) < capacity:
                    affinity = float(graph[channel, group].sum())
                    cost = -affinity if mode == "cluster" else affinity
                    cost += balance * (2 * len(group) + 1 - 2 * c / packets)
                    candidates.append((cost, r))
            groups[min(candidates)[1]].append(channel)
    return groups


def packet_graph(graph, groups):
    membership = torch.zeros(len(groups), len(graph))
    for r, group in enumerate(groups):
        if group:
            membership[r, group] = 1 / len(group)
    result = membership @ graph @ membership.T
    result.fill_diagonal_(0)
    return result


def normalized_graph(graph, self_loops=False):
    a = graph + torch.eye(len(graph), device=graph.device) if self_loops else graph
    degree = a.sum(1)
    inv = torch.where(degree > 0, degree.clamp_min(1e-12).rsqrt(), 0)
    return inv[:, None] * a * inv[None, :]


def laplacian(graph):
    # Isolated vertices have zero Laplacian, rather than artificial shrinkage.
    return torch.diag((graph.sum(1) > 0).to(graph.dtype)) - normalized_graph(graph)


def intra_correlation(graph, groups):
    numerator, count = 0.0, 0
    for group in groups:
        if len(group) > 1:
            numerator += float(graph[group][:, group].sum()) / 2
            count += len(group) * (len(group) - 1) // 2
    return numerator / count if count else 0.0


class Packetizer(nn.Module):
    def __init__(self, groups, channels):
        super().__init__()
        self.channels = channels
        self.packets = len(groups)
        self.capacity = math.ceil(channels / self.packets)
        slots = torch.full((self.packets, self.capacity), channels, dtype=torch.long)
        for r, group in enumerate(groups):
            slots[r, : len(group)] = torch.tensor(group, dtype=torch.long)
        flat = slots.flatten()
        inverse = torch.stack([torch.nonzero(flat == c)[0, 0] for c in range(channels)])
        self.register_buffer("slots", slots)
        self.register_buffer("inverse", inverse)

    def pack(self, z):
        b, _, h, w = z.shape
        padded = torch.cat((z, z.new_zeros(b, 1, h, w)), dim=1)
        return padded[:, self.slots].reshape(b, self.packets, self.capacity * h * w)

    def unpack(self, payload, shape):
        h, w = shape
        return payload.reshape(len(payload), -1, h, w)[:, self.inverse]

    def received_channels(self, mask):
        return mask[:, self.inverse // self.capacity]
