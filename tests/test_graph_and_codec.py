import itertools
import uuid

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from graphfec import gf256
from graphfec.codec import CoarseDecoder, Quantizer, SparseParity
from graphfec.config import Config
from graphfec.graph import (
    ChannelStatistics,
    Packetizer,
    assign_channels,
    intra_correlation,
    laplacian,
    sparsify,
)
from graphfec.system import BASELINES, VARIANTS, Protection
from graphfec.transport import HEADER, decode_datagrams, encode_datagrams


def example_profile(channels=8):
    rng = torch.Generator().manual_seed(11)
    a = torch.rand(channels, channels, generator=rng)
    a = (a + a.T) / 2
    a.fill_diagonal_(0)
    return {
        "correlation": a,
        "correlation_static": a,
        "low": torch.zeros(channels),
        "high": torch.ones(channels),
        "mean": torch.full((channels,), 0.5),
        "saliency": torch.arange(channels).float() + 1,
        "shape": [channels, 4, 4],
    }


def test_absolute_correlation_including_constant_and_anticorrelated_channels():
    x = torch.arange(10).float()
    data = torch.stack((x, 3 * x + 2, -x, torch.ones_like(x)), dim=1)[:, :, None, None]
    stats = ChannelStatistics(4)
    stats.update(data[:5])
    stats.update(data[5:])
    for ema in (True, False):
        corr = stats.correlation(ema)
        torch.testing.assert_close(corr[:3, :3], torch.ones(3, 3) - torch.eye(3))
        torch.testing.assert_close(corr[3], torch.zeros(4))
    sparse = sparsify(stats.correlation(), 1)
    torch.testing.assert_close(sparse, sparse.T)
    assert torch.all(sparse.diag() == 0)


def test_interleaving_separates_correlated_pairs_and_roundtrips_padding():
    graph = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
    )
    groups = assign_channels(graph, 2)
    clustered = assign_channels(graph, 2, mode="cluster")
    assert intra_correlation(graph, groups) == 0
    assert intra_correlation(graph, clustered) == 1
    assert groups == assign_channels(graph, 2)
    groups = assign_channels(torch.zeros(5, 5), 3)
    assert max(map(len, groups)) <= 2
    assert sorted(c for group in groups for c in group) == list(range(5))
    packetizer = Packetizer(groups, 5)
    z = torch.arange(2 * 5 * 6).reshape(2, 5, 2, 3).float()
    torch.testing.assert_close(packetizer.unpack(packetizer.pack(z), (2, 3)), z)
    mask = torch.tensor([[True, False, True]])
    for packet, members in enumerate(groups):
        assert torch.all(packetizer.received_channels(mask)[0, members] == mask[0, packet])


def test_real_decoder_exact_full_rank_and_regularized_normal_equations():
    torch.manual_seed(3)
    gamma = torch.randn(2, 4, dtype=torch.float64)
    g = torch.cat((torch.eye(4, dtype=torch.float64), gamma))
    p = torch.randn(2, 4, 7, dtype=torch.float64)
    mask = torch.tensor([[True, False, True, False, True, True]]).expand(2, -1)
    y = g @ p
    decoder = CoarseDecoder(torch.zeros(4, 4, dtype=torch.float64), eta=0, ridge=1e-12)
    recovered = decoder(g, y * mask[:, :, None], mask)
    torch.testing.assert_close(recovered, p, rtol=1e-8, atol=1e-8)
    lp = laplacian(torch.ones(4, 4, dtype=torch.float64) - torch.eye(4, dtype=torch.float64))
    decoder = CoarseDecoder(lp, eta=0.1, ridge=1e-3)
    x = decoder(g, y, mask)
    gt = g.T * mask[0]
    h = gt @ g + 0.1 * lp + 1e-3 * torch.eye(4, dtype=torch.float64)
    torch.testing.assert_close(h @ x, gt @ y)
    torch.testing.assert_close(
        decoder(g, torch.zeros_like(y), torch.zeros_like(mask)), torch.zeros_like(p)
    )


def test_sparse_parity_budget_and_quantization_bounds():
    parity = SparseParity(torch.ones(4, 4) - torch.eye(4), 2, 2)
    gamma = parity.matrix()
    assert torch.all((gamma != 0).sum(1) <= 2)
    torch.testing.assert_close(gamma.norm(dim=1), torch.ones(2))
    packets = torch.randint(256, (3, 4, 16)).float()
    wire = parity.encode(packets)
    assert torch.all((wire >= 0) & (wire <= 255) & (wire == wire.round()))
    y = parity.dequantize(wire, "uint8")
    exact = gamma @ packets
    tolerance = gamma.abs().sum(1)[None, :, None] / 2 + 1e-4
    assert torch.all((y[:, 4:] - exact).abs() <= tolerance)
    q = Quantizer(torch.tensor([-1.0, 2.0]), torch.tensor([1.0, 4.0]))
    z = torch.tensor([[[[-0.7]], [[3.1]]]])
    assert torch.all((q.decode(q.encode(z)) - z).abs() <= q.scale / 2 + 1e-6)


def test_reed_solomon_recovers_every_pattern_within_budget():
    g = gf256.systematic_matrix(4, 2)
    p = np.random.default_rng(7).integers(0, 256, (4, 40), dtype=np.uint8)
    wire = gf256.matmul(g, p)
    np.testing.assert_array_equal(g[:4], np.eye(4, dtype=np.uint8))
    for count in range(3):
        for erased in itertools.combinations(range(6), count):
            mask = np.ones(6, dtype=bool)
            mask[list(erased)] = False
            np.testing.assert_array_equal(gf256.decode(g, wire, mask, np.zeros_like(p)), p)


def test_unequal_protection_recovers_high_group_and_fills_unresolved_low_group():
    g = gf256.unequal_matrix(np.array([10.0, 9.0, 1.0, 1.0]), 2)
    p = np.arange(4 * 10, dtype=np.uint8).reshape(4, 10)
    wire = gf256.matmul(g, p)
    mask = np.array([False, True, False, True, True, True])
    result = gf256.decode(g, wire, mask, np.full_like(p, 99))
    np.testing.assert_array_equal(result[[0, 1, 3]], p[[0, 1, 3]])
    np.testing.assert_array_equal(result[2], np.full(10, 99, dtype=np.uint8))


@pytest.mark.parametrize("method", VARIANTS + BASELINES)
def test_lossless_reception_preserves_quantized_activation(method):
    cfg = Config(packets=4, parity=2, neighbors=3, sparsity=3)
    protection = Protection(cfg, example_profile(), method).eval()
    z = torch.rand(2, 8, 4, 4)
    with torch.no_grad():
        actual = protection(z, torch.ones(2, protection.total_packets, dtype=torch.bool))
    torch.testing.assert_close(actual, protection.quantizer.decode(protection.quantizer.encode(z)))


def test_parity_and_refiner_receive_task_gradients_and_cache_matches():
    torch.manual_seed(4)
    cfg = Config(packets=4, parity=2, neighbors=3, sparsity=3, bottleneck=4)
    protection = Protection(cfg, example_profile())
    z = torch.rand(2, 8, 4, 4)
    mask = torch.tensor(
        [[True, False, True, False, True, True], [True, True, False, True, False, True]]
    )
    loss = F.mse_loss(protection(z, mask), z)
    loss.backward()
    gradient = protection.parity.coefficients.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert protection.refiner.blocks[0][-1].weight.grad.abs().sum() > 0
    protection.eval()
    with torch.no_grad():
        first, second = protection(z, mask), protection(z, mask)
    torch.testing.assert_close(first, second)
    assert len(protection.coarse.cache) == 2


def test_packet_serialization_byte_budget_profile_and_corruption_checks():
    cfg = Config(packets=4, parity=2, neighbors=3, sparsity=3)
    protection = Protection(cfg, example_profile()).eval()
    z = torch.rand(1, 8, 4, 4)
    profile = str(uuid.uuid4())
    wire = protection.encode(z)
    packets = encode_datagrams(protection, wire[0], (8, 4, 4), profile)
    assert sum(map(len, packets)) == protection.payload_bytes((4, 4)) + 6 * HEADER.size
    restored, mask = decode_datagrams(protection, packets[:1] + packets[2:], (8, 4, 4), profile)
    assert not mask[0, 1] and mask.sum() == 5
    torch.testing.assert_close(restored[mask], wire[mask])
    with pytest.raises(ValueError, match="profile"):
        decode_datagrams(protection, packets, (8, 4, 4), str(uuid.uuid4()))
    corrupt = bytearray(packets[0])
    corrupt[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        decode_datagrams(protection, [bytes(corrupt)], (8, 4, 4), profile)
