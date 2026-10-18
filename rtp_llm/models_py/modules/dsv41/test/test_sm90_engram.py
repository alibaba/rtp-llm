"""Small-table SM90 regression of the real UVA -> Humming -> inject chain."""

import hashlib
import math
import os
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv41.engram import (
    Engram,
    EngramLayout,
    HostEngramEmbedding,
    NgramHashState,
    _PinnedSharedTable,
    create_engram_prefetch,
)
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear


class EngramPrefetchFactoryTest(unittest.TestCase):
    def test_sm90_constructs_prefetch_for_registered_layers(self):
        constructor = mock.Mock()
        tables = [(1, types.SimpleNamespace(_uva=object()))]
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.version.hip", None),
            mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            mock.patch.dict(
                os.environ,
                {
                    "DSV41_ENGRAM_PREFETCH": "1",
                    "DSV41_ENGRAM_PREFETCH_MAX_ROWS": "64",
                    "PPU_HOME": "",
                },
            ),
            mock.patch.dict(
                "sys.modules",
                {
                    "rtp_llm.models_py.modules.dsv41.engram_prefetch": types.SimpleNamespace(
                        UvaEngramPrefetch=constructor
                    ),
                },
            ),
        ):
            prefetch, max_rows = create_engram_prefetch("cuda:0", iter(tables))
        self.assertIs(prefetch, constructor.return_value)
        self.assertEqual(max_rows, 64)
        constructor.assert_called_once_with(torch.device("cuda:0"), [1])

    def test_unsupported_cuda_and_hip_do_not_import_prefetch(self):
        tables = [(1, types.SimpleNamespace(_uva=object()))]
        for hip, capability in ((None, (8, 0)), ("6.3.mock", (9, 0))):
            with (
                self.subTest(hip=hip, capability=capability),
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch("torch.version.hip", hip),
                mock.patch(
                    "torch.cuda.get_device_capability", return_value=capability
                ) as get_capability,
                mock.patch.dict(
                    os.environ,
                    {
                        "DSV41_ENGRAM_PREFETCH": "1",
                        "DSV41_ENGRAM_PREFETCH_MAX_ROWS": "invalid",
                        "PPU_HOME": "",
                    },
                ),
                mock.patch.dict(
                    "sys.modules",
                    {
                        "rtp_llm.models_py.modules.dsv41.engram_prefetch": None,
                    },
                ),
            ):
                self.assertEqual(create_engram_prefetch("cuda:0", tables), (None, 0))
                if hip is not None:
                    get_capability.assert_not_called()

    def test_architecture_defaults_and_sm100_opt_in(self):
        for capability, enabled, expected in (
            ((9, 0), None, True),
            ((10, 0), None, False),
            ((10, 0), "0", False),
            ((10, 0), "1", True),
        ):
            constructor = mock.Mock()
            with (
                self.subTest(capability=capability, enabled=enabled),
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch("torch.version.hip", None),
                mock.patch("torch.cuda.get_device_capability", return_value=capability),
                mock.patch.dict(os.environ, {"PPU_HOME": ""}, clear=True),
                mock.patch.dict(
                    "sys.modules",
                    {
                        "rtp_llm.models_py.modules.dsv41.engram_prefetch": types.SimpleNamespace(
                            UvaEngramPrefetch=constructor
                        ),
                    },
                ),
            ):
                if enabled is not None:
                    os.environ["DSV41_ENGRAM_PREFETCH"] = enabled
                actual = create_engram_prefetch(
                    "cuda:0", [(1, types.SimpleNamespace(_uva=object()))]
                )
                if expected:
                    self.assertEqual(actual, (constructor.return_value, 256))
                    constructor.assert_called_once_with(torch.device("cuda:0"), [1])
                else:
                    self.assertEqual(actual, (None, 0))
                    constructor.assert_not_called()

    def test_disabled_or_unregistered_sm90_keeps_per_layer_lookup(self):
        for enabled, uva in (("0", object()), ("1", None)):
            with (
                self.subTest(enabled=enabled, registered=uva is not None),
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch("torch.version.hip", None),
                mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
                mock.patch.dict(
                    os.environ,
                    {
                        "DSV41_ENGRAM_PREFETCH": enabled,
                        "DSV41_ENGRAM_PREFETCH_MAX_ROWS": "invalid",
                        "PPU_HOME": "",
                    },
                ),
                mock.patch.dict(
                    "sys.modules",
                    {
                        "rtp_llm.models_py.modules.dsv41.engram_prefetch": None,
                    },
                ),
            ):
                tables = [(1, types.SimpleNamespace(_uva=uva))]
                self.assertEqual(create_engram_prefetch("cuda:0", tables), (None, 0))

    def test_enabled_sm90_rejects_nonpositive_limit(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.version.hip", None),
            mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            mock.patch.dict(
                os.environ,
                {
                    "DSV41_ENGRAM_PREFETCH": "1",
                    "DSV41_ENGRAM_PREFETCH_MAX_ROWS": "0",
                    "PPU_HOME": "",
                },
            ),
        ):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                create_engram_prefetch(
                    "cuda:0", [(1, types.SimpleNamespace(_uva=object()))]
                )


class Sm90EngramTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            9,
            0,
        ):
            raise RuntimeError("test_sm90_engram requires a CUDA SM90 GPU")

    def setUp(self):
        torch.manual_seed(4193)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.layout = EngramLayout(
            dict(
                engram_layer_ids=[1],
                engram_num_embeddings=[4000],
                engram_max_ngram_size=4,
                engram_n_heads=4,
                engram_head_dim=32,
                engram_compressed_vocab_size=32,
                engram_pad_token_id=2,
                engram_vocab_size=13,
            )
        )
        # Only 132 KiB of host table, but production HC/D geometry ensures
        # Engram.forward selects its real fused in-place inject kernel.
        self.table = torch.randn(4000, 32).to(torch.float8_e4m3fn)
        self.table_scales = torch.randint(124, 130, (4000, 1), dtype=torch.uint8)
        identity = "test-sm90-engram-" + uuid.uuid4().hex
        digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
        self.data_path = Path("/dev/shm") / f"rtp_llm_engram_{os.getuid()}_{digest}"
        self.lock_path = Path("/tmp") / (self.data_path.name + ".lock")
        self.pinned = None
        self.addCleanup(self._release_own_table)
        self.pinned = _PinnedSharedTable(
            self.table, self.table_scales, identity, self.device
        )
        embedding = HostEngramEmbedding(
            self.pinned.weight, self.pinned.scales, pinned=self.pinned
        )
        k, n = self.layout.n_hash_cols * self.layout.head_dim, 5 * 5120
        self.weight = (torch.randn(n, k) * 0.02).to(torch.float8_e4m3fn)
        self.weight_exponents = torch.randint(125, 129, (n // 32, k // 32))
        scales = self.weight_exponents.to(torch.uint8).view(torch.float8_e8m0fnu)
        projection = V41MXFP8Linear(self.weight.to(self.device), scales.to(self.device))
        self.assertIsNotNone(projection._sm90_linear)
        self.q_weight = torch.randn(4, 5120).bfloat16()
        self.k_weight = torch.randn(4, 5120).bfloat16()
        self.model = Engram(
            self.layout,
            0,
            embedding,
            projection,
            self.q_weight.to(self.device),
            self.k_weight.to(self.device),
            1e-6,
        )
        # Independent checkpoint block-scale decoding, not adapter utilities.
        block_scales = torch.exp2(self.weight_exponents.float() - 127)
        self.reference_weight = self.weight.float() * block_scales.repeat_interleave(
            32, 0
        ).repeat_interleave(32, 1)

    def _release_own_table(self):
        torch.cuda.synchronize(self.device)
        if self.pinned is not None:
            # Execute the production finalizers in safe CUDA-before-file order.
            # Their unique UUID lease cannot refer to another test/service.
            self.pinned._registration_finalizer()
            self.pinned._lease_finalizer()
            self.assertFalse(self.data_path.exists())
        self.lock_path.unlink(missing_ok=True)

    def _reference(self, hidden, ids, mask):
        """CPU oracle: byte dequant, g32 activation rounding, FP64 GEMM, gate."""
        ids, hidden, mask = ids.cpu(), hidden.cpu(), mask.cpu()
        flat = ids.clamp_min(0).reshape(-1)
        payload = self.table.view(torch.uint8)[flat].view(torch.float8_e4m3fn)
        scale = torch.exp2(self.table_scales[flat].float() - 127)
        rows = (payload.float() * scale).bfloat16().view(*ids.shape, 32)
        rows.masked_fill_(ids[..., None] < 0, 0)
        rows = rows.flatten(-2)
        groups = rows.float().reshape(rows.shape[0], -1, 32)
        amax = groups.abs().amax(-1, keepdim=True).clamp_min(1e-10)
        scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
        quantized = (groups / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        activation = (quantized.float() * scale).reshape_as(rows)
        kv = (activation.double() @ self.reference_weight.double().T).bfloat16()
        key = kv[:, : 4 * 5120].reshape(-1, 4, 5120).float()
        value = kv[:, 4 * 5120 :].float()[:, None]
        h = hidden.float()
        dot = (h * self.q_weight.float() * self.k_weight.float() * key).sum(-1)
        dot *= torch.rsqrt(h.square().mean(-1) + self.model.eps)
        dot *= torch.rsqrt(key.square().mean(-1) + self.model.eps) / math.sqrt(5120)
        root = dot.abs().clamp_min(1e-6).sqrt()
        gate = torch.sigmoid(torch.where(dot < 0, -root, root))
        gate = torch.where(mask[:, None], gate, 0)
        return (h + gate[..., None] * value).bfloat16(), rows, kv

    def _check(self, actual, original, ids, mask):
        expected, rows, kv = self._reference(original, ids, mask)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.01)
        torch.testing.assert_close(actual[~mask], original[~mask], rtol=0, atol=0)
        # Check the chain's intermediate contracts independently as well: a
        # quiet/masked residual must not conceal a broken lookup or projection.
        looked_up = self.model.embed_tokens(ids, self.device).flatten(-2)
        torch.testing.assert_close(looked_up.cpu(), rows, rtol=0, atol=0)
        torch.testing.assert_close(
            self.model.wkv(looked_up).cpu(), kv, rtol=0.015, atol=0.005
        )

    @torch.inference_mode()
    def test_production_forward_and_graph_replay_match_independent_reference(self):
        self.assertTrue(self.pinned.storage.is_pinned())
        self.assertEqual(self.model.embed_tokens.weight.device.type, "cpu")
        for tokens in (1, 7):
            with self.subTest(tokens=tokens):
                hidden = torch.randn(tokens, 4, 5120, device=self.device).bfloat16()
                ids = torch.randint(
                    0, 4000, (tokens, self.layout.n_hash_cols), device=self.device
                )
                ids[0, 0] = -1
                mask = torch.arange(tokens, device=self.device) % 3 != 1
                original = hidden.clone()
                actual = self.model(hidden, ids, mask)
                self.assertEqual(actual.data_ptr(), hidden.data_ptr())
                self._check(actual, original, ids, mask)

        # Warm the complete production forward on the capture stream. Its
        # in-place output is reset before each replay so iterations do not
        # accumulate a residual from earlier test calls.
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                hidden.copy_(original)
                self.model(hidden, ids, mask)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = self.model(hidden, ids, mask)
        self.assertEqual(captured.data_ptr(), hidden.data_ptr())
        for shift in (31, 197):
            original.normal_()
            hidden.copy_(original)
            ids.copy_((ids + shift) % 4000)
            ids[0, 0] = -1
            mask.logical_not_()
            graph.replay()
            torch.cuda.synchronize(self.device)
            self._check(captured, original, ids, mask)


if __name__ == "__main__":
    unittest.main()
