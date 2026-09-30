"""Affine quantization, sparse real parity, and differentiable coarse decoding."""

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


def round_ste(x):
    return x + (x.round() - x).detach()


class Quantizer(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.register_buffer("low", low.reshape(1, -1, 1, 1))
        self.register_buffer("scale", ((high - low) / 255).clamp_min(1e-8).reshape(1, -1, 1, 1))

    def encode(self, z):
        return round_ste(((z - self.low) / self.scale).clamp(0, 255))

    def decode(self, q):
        return q * self.scale + self.low


class SparseParity(nn.Module):
    def __init__(self, graph, rows, sparsity, random=False, seed=17):
        super().__init__()
        m = len(graph)
        support = torch.zeros(rows, m)
        rng = torch.Generator().manual_seed(seed)
        rank = torch.argsort(graph.sum(1), descending=True, stable=True)
        for r in range(rows):
            anchor = int(rank[r % m])
            scores = graph[anchor].clone()
            scores[anchor] = scores.max() + 1
            indices = torch.argsort(scores, descending=True, stable=True)[: min(sparsity, m)]
            support[r, indices] = 1
        if random:
            support.fill_(1)
            initial = torch.randn(rows, m, generator=rng)
        else:
            initial = 0.5 + torch.rand(rows, m, generator=rng)
        self.register_buffer("support", support)
        self.register_buffer("edges", support.nonzero().T)
        self.coefficients = nn.Parameter(initial * support)

    def matrix(self):
        return F.normalize(self.coefficients * self.support, dim=1, eps=1e-8)

    def encode(self, packets, precision="uint8"):
        gamma = self.matrix()
        row, column = self.edges
        contributions = packets[:, column] * gamma[row, column][None, :, None]
        parity = packets.new_zeros(len(packets), len(gamma), packets.shape[2])
        parity = parity.index_add(1, row, contributions)
        if precision == "uint8":
            # Fixed profile-derived bounds cover every possible uint8 data payload.
            low = 255 * gamma.clamp_max(0).sum(1)
            scale = gamma.abs().sum(1).clamp_min(1e-8)
            parity = round_ste(((parity - low[None, :, None]) / scale[None, :, None]).clamp(0, 255))
        return torch.cat((packets, parity), dim=1)

    def dequantize(self, wire, precision):
        m = self.support.shape[1]
        if precision == "float32":
            return wire
        gamma = self.matrix()
        low = 255 * gamma.clamp_max(0).sum(1)
        scale = gamma.abs().sum(1).clamp_min(1e-8)
        parity = wire[:, m:] * scale[None, :, None] + low[None, :, None]
        return torch.cat((wire[:, :m], parity), dim=1)


class CoarseDecoder(nn.Module):
    def __init__(self, laplacian, eta=1e-3, ridge=1e-6, cache_size=256):
        super().__init__()
        self.register_buffer("laplacian", laplacian)
        self.eta, self.ridge, self.cache_size = eta, ridge, cache_size
        self.cache = OrderedDict()

    def train(self, mode=True):
        self.cache.clear()
        return super().train(mode)

    def forward(self, generator, received, mask):
        m = generator.shape[1]
        identity = torch.eye(m, device=generator.device, dtype=generator.dtype)
        patterns, inverse = torch.unique(mask, dim=0, return_inverse=True)
        recovered = torch.empty(
            len(received), m, received.shape[2], device=received.device, dtype=received.dtype
        )
        for i, pattern in enumerate(patterns):
            selected = inverse == i
            key = tuple(pattern.tolist())
            use_cache = not self.training and not torch.is_grad_enabled()
            if use_cache and key in self.cache:
                operator = self.cache.pop(key)
                self.cache[key] = operator
            else:
                gt = generator.T * pattern.to(generator.dtype)[None, :]
                h = gt @ generator + self.eta * self.laplacian + self.ridge * identity
                operator = torch.linalg.solve(h, gt)
                if use_cache and self.cache_size:
                    self.cache[key] = operator
                    if len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
            recovered[selected] = operator @ received[selected]
        return recovered


class GraphRefiner(nn.Module):
    def __init__(self, normalized_adjacency, bottleneck):
        super().__init__()
        c = len(normalized_adjacency)
        edges = normalized_adjacency.nonzero().T
        self.register_buffer("edges", edges)
        self.register_buffer("weights", normalized_adjacency[edges[0], edges[1]])
        hidden = min(bottleneck, c)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c),
                    nn.GELU(),
                    nn.Conv2d(2 * c, hidden, 1),
                    nn.GELU(),
                    nn.Conv2d(hidden, c, 1),
                )
                for _ in range(2)
            ]
        )
        for block in self.blocks:
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)

    def forward(self, z, received_channels):
        mask = received_channels[:, :, None, None].to(z.dtype).expand_as(z)
        for block in self.blocks:
            messages = z[:, self.edges[1]] * self.weights[None, :, None, None]
            aggregate = torch.zeros_like(z).index_add(1, self.edges[0], messages)
            z = z + block(torch.cat((aggregate, mask), dim=1))
        return z
