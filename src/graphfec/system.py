"""End-to-end activation protection and matched-payload baselines."""

import numpy as np
import torch
from torch import nn

from . import gf256
from .codec import CoarseDecoder, GraphRefiner, Quantizer, SparseParity
from .graph import (
    Packetizer,
    assign_channels,
    intra_correlation,
    laplacian,
    normalized_graph,
    packet_graph,
    sparsify,
)

VARIANTS = (
    "graphfec",
    "random",
    "cluster",
    "no_parity",
    "no_refinement",
    "neural_only",
    "task_only",
    "static_graph",
)
BASELINES = ("none", "rs", "rlc", "uep", "duplication")


class Protection(nn.Module):
    def __init__(self, cfg, profile, method="graphfec"):
        super().__init__()
        if method not in VARIANTS + BASELINES:
            raise ValueError(f"Unknown protection method: {method}")
        self.cfg, self.method = cfg, method
        dense = profile["correlation_static" if method == "static_graph" else "correlation"]
        graph = sparsify(dense, cfg.neighbors)
        mode = method if method in ("random", "cluster") else "graphfec"
        self.groups = assign_channels(graph, cfg.packets, cfg.balance, mode, cfg.seed)
        self.intra = intra_correlation(dense, self.groups)
        self.packetizer = Packetizer(self.groups, len(graph))
        self.quantizer = Quantizer(profile["low"], profile["high"])
        self.register_buffer("channel_mean", profile["mean"].reshape(1, -1, 1, 1))
        self.r = 0 if method in ("none", "no_parity") else cfg.parity
        if method == "duplication":
            self.r = cfg.packets
        self.total_packets = cfg.packets + self.r
        self.register_buffer("gf_generator", torch.empty(0, dtype=torch.uint8))
        self.parity = None
        self.coarse = None
        self.refiner = None
        if method in ("rs", "uep"):
            scores = [float(profile["saliency"][g].sum()) for g in self.groups]
            g = (
                gf256.systematic_matrix(cfg.packets, self.r)
                if method == "rs"
                else gf256.unequal_matrix(scores, self.r)
            )
            self.gf_generator = torch.from_numpy(g)
        elif method not in ("none", "duplication"):
            pg = packet_graph(graph, self.groups)
            self.parity = SparseParity(pg, self.r, cfg.sparsity, method == "rlc", cfg.seed)
            if method == "rlc":
                self.parity.requires_grad_(False)
            self.coarse = CoarseDecoder(
                laplacian(pg), 0 if method == "rlc" else cfg.eta, cfg.ridge, cfg.cache_size
            )
            if method not in ("rlc", "no_refinement"):
                self.refiner = GraphRefiner(
                    normalized_graph(graph, self_loops=True), cfg.bottleneck
                )

    def encode(self, z):
        q = self.quantizer.encode(z)
        data = self.packetizer.pack(q)
        if self.method in ("rs", "uep"):
            wire = np.stack(
                [
                    gf256.matmul(self.gf_generator.cpu().numpy(), p)
                    for p in data.detach().cpu().numpy().astype(np.uint8)
                ]
            )
            return torch.from_numpy(wire).to(z.device, dtype=z.dtype)
        if self.method == "duplication":
            return torch.cat((data, data), dim=1)
        if self.method == "none":
            return data
        return self.parity.encode(data, self.cfg.parity_precision)

    def decode(self, wire, mask, shape):
        m = self.cfg.packets
        h, w = shape
        means = self.channel_mean.expand(len(wire), -1, h, w)
        fallback = self.packetizer.pack(self.quantizer.encode(means))
        if self.method in ("rs", "uep"):
            values = wire.detach().cpu().numpy().astype(np.uint8)
            masks = mask.cpu().numpy()
            fills = fallback.detach().cpu().numpy().astype(np.uint8)
            g = self.gf_generator.cpu().numpy()
            decoded = np.stack(
                [gf256.decode(g, y, keep, fill) for y, keep, fill in zip(values, masks, fills)]
            )
            packets = torch.from_numpy(decoded).to(wire.device, dtype=wire.dtype)
        elif self.method == "duplication":
            packets = torch.where(
                mask[:, :m, None],
                wire[:, :m],
                torch.where(mask[:, m:, None], wire[:, m:], fallback),
            )
        elif self.method == "none":
            packets = torch.where(mask[:, :m, None], wire[:, :m], fallback)
        else:
            y = self.parity.dequantize(wire, self.cfg.parity_precision)
            g = torch.cat((torch.eye(m, device=wire.device), self.parity.matrix()), dim=0)
            if self.method == "neural_only":
                numerator = torch.einsum("nm,bnd->bmd", g, y * mask[:, :, None])
                denominator = torch.einsum("nm,bn->bm", g.square(), mask.float())
                packets = numerator / denominator[:, :, None].clamp_min(1e-6)
                packets = torch.where(denominator[:, :, None] > 0, packets, fallback)
            else:
                packets = self.coarse(g, y * mask[:, :, None], mask)
        # Received systematic packets always remain exact after quantization.
        packets = torch.where(mask[:, :m, None], wire[:, :m], packets)
        z = self.quantizer.decode(self.packetizer.unpack(packets, shape))
        channel_mask = self.packetizer.received_channels(mask)
        if self.refiner is not None:
            refined = self.refiner(z, channel_mask)
            z = torch.where(channel_mask[:, :, None, None], z, refined)
        return z

    def forward(self, z, mask):
        return self.decode(self.encode(z), mask, z.shape[-2:])

    def payload_bytes(self, shape):
        h, w = shape
        data_bytes = self.cfg.packets * self.packetizer.capacity * h * w
        parity_width = (
            4 if self.cfg.parity_precision == "float32" and self.parity is not None else 1
        )
        parity_bytes = self.r * self.packetizer.capacity * h * w * parity_width
        return data_bytes + parity_bytes
