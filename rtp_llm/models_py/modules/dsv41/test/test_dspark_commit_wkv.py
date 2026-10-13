"""Checkpoint-free contracts for the DSpark group32 projection helper."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41 import _dspark_commit_wkv as commit_wkv
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear


def make_linears():
    torch.manual_seed(1901)
    linears = []
    for _ in range(3):
        weights = torch.randn(512, 5120, device="cuda").to(torch.float8_e4m3fn)
        scales = (2.0 ** torch.randint(-5, 3, (16, 160), device="cuda")).to(
            torch.float8_e8m0fnu
        )
        linears.append(V41MXFP8Linear(weights, scales))
    return linears


def exact(test, actual, expected):
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(actual.dtype, expected.dtype)
    torch.testing.assert_close(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DSparkCommitWKVTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("DeepGEMM MXFP8 requires SM10x")
        cls.linears = make_linears()

    def prepare(self, mode):
        with patch.object(commit_wkv, "sleep_enabled", return_value=False):
            prepared = commit_wkv.prepare_dspark_commit_wkv(self.linears, mode)
        self.assertIsNotNone(prepared)
        return prepared

    def test_shared_stacked_and_scale_layout(self):
        shared, stacked = self.prepare("shared"), self.prepare("stacked")
        self.assertEqual(stacked.stacked.weight_scales.stride(), (1, 1536))
        for i, linear in enumerate(self.linears):
            exact(self, stacked.stacked.weight[i * 512 : (i + 1) * 512], linear.weight)
            exact(
                self,
                stacked.stacked.weight_scales[i * 512 : (i + 1) * 512],
                linear.weight_scales,
            )
        for rows in (1, 6, 65, 192):
            x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
            for factor in (1.0, 1e-12, 0.0):
                source = x * factor
                expected = [linear(source) for linear in self.linears]
                for prepared in (shared, stacked):
                    for actual, wanted in zip(prepared.project(source), expected):
                        exact(self, actual, wanted)
                slices = stacked.project(source)
                self.assertEqual(slices[0].stride(), (1536, 1))

    def test_graph_replay_and_weight_lifetime(self):
        prepared = self.prepare("stacked")
        x = torch.randn(65, 5120, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                prepared.project(x)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = prepared.project(x)
        with torch.cuda.stream(stream):
            for _ in range(3):
                x.normal_()
                graph.replay()
                for actual, linear in zip(output, self.linears):
                    exact(self, actual, linear(x))
            original = self.linears[0].weight
            original_output = output[0].clone()
            address = prepared.stacked.weight.data_ptr()
            try:
                self.linears[0].weight = (-original.float()).to(original.dtype)
                self.assertFalse(prepared.matches(self.linears))
                fresh = self.prepare("stacked")
                self.assertTrue(fresh.matches(self.linears))
                self.assertNotEqual(fresh.stacked.weight.data_ptr(), address)
                graph.replay()
                exact(self, output[0], original_output)
                exact(self, fresh.project(x)[0], self.linears[0](x))
            finally:
                self.linears[0].weight = original
        torch.cuda.current_stream().wait_stream(stream)

    def test_auto_dispatch_boundaries_and_unsupported_shapes(self):
        with patch.object(
            commit_wkv, "sleep_enabled", return_value=False
        ), patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 3)
        ), patch.dict(
            "os.environ", {}, clear=False
        ):
            import os

            os.environ.pop("DSV4_DSPARK_COMMIT_WKV", None)
            automatic = commit_wkv.prepare_dspark_commit_wkv(self.linears)
        self.assertEqual(automatic.mode, "auto")
        self.assertIsNotNone(automatic.stacked)
        for rows in (1, 3072, 3073):
            x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
            actual = automatic.project(x)
            if rows == 3073:
                self.assertIsNone(actual)
            else:
                for got, linear in zip(actual, self.linears):
                    exact(self, got, linear(x))
        for capability in ((9, 0), (10, 0)):
            with patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ):
                unsupported = self.prepare("auto")
            self.assertIsNone(unsupported.stacked)
            self.assertIsNone(unsupported.project(x[:1]))
        for n, k in ((256, 5120), (512, 256)):
            linears = [
                V41MXFP8Linear(
                    torch.zeros(n, k, device="cuda", dtype=torch.float8_e4m3fn),
                    torch.ones(n // 32, k // 32, device="cuda").to(
                        torch.float8_e8m0fnu
                    ),
                )
                for _ in range(3)
            ]
            with patch.object(
                torch.cuda, "get_device_capability", return_value=(10, 3)
            ):
                unsupported = commit_wkv.prepare_dspark_commit_wkv(linears, "auto")
            self.assertIsNone(unsupported.stacked)

    def test_model_routes_each_precomputed_projection_to_its_layer(self):
        import rtp_llm.models_py.model_desc.deepseek_v41_base_dspark_model as module

        with patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)):
            prepared = self.prepare("auto")
        # Exercise SM103 auto dispatch even when the consumer test runs on SM100.
        self.assertIsNotNone(prepared.stacked)
        x = torch.randn(6, 5120, device="cuda", dtype=torch.bfloat16)
        rows = torch.arange(6, device="cuda", dtype=torch.int32)
        received = []
        model = SimpleNamespace(
            _commit_wkv_projection=prepared,
            v4=SimpleNamespace(
                layers=[
                    SimpleNamespace(attn=SimpleNamespace(wkv=w)) for w in self.linears
                ]
            ),
            kv_cache=object(),
            _swa_block_table=lambda *args: torch.ones(
                1, 1, device="cuda", dtype=torch.int32
            ),
            _commit_layer_features=lambda index, *args, **kwargs: received.append(
                (index, kwargs["projected_kv"])
            ),
        )
        with patch.object(
            module, "as_tagged_attention_inputs", side_effect=lambda x: x
        ), patch.object(
            module, "cached_swa_region", return_value="decoder_swa_kv"
        ), patch.object(
            module, "require_pool_tokens_per_block", return_value=256
        ), patch.object(
            module, "create_tagged_cache_store_writer", return_value=None
        ), patch.object(
            module._profiler,
            "make_layer_forward_range",
            return_value=lambda _: nullcontext(),
        ), patch.object(
            prepared, "project", wraps=prepared.project
        ) as project:
            module.DeepSeekV4DSparkModel.commit_feature_rows(
                model,
                x,
                torch.zeros_like(rows),
                rows,
                rows.new_tensor([6]),
                SimpleNamespace(attention_inputs=object()),
            )
            project.assert_called_once_with(x)
        self.assertEqual([index for index, _ in received], [0, 1, 2])
        for (_, actual), linear in zip(received, self.linears):
            exact(self, actual, linear(x))
        self.assertEqual(
            len({value.untyped_storage().data_ptr() for _, value in received}), 1
        )

    def test_shared_storage_reload_updates_commit_and_proposal_graphs(self):
        from rtp_llm.models_py.modules.dsv41.utils import merge_v41_qkv_weights
        from rtp_llm.utils.model_weight import W

        owners, linears, proposals = [], [], []
        for _ in range(3):
            weights = {
                W.v4_attn_wq_a_w: torch.randn(128, 5120, device="cuda").to(
                    torch.float8_e4m3fn
                ),
                W.v4_attn_wq_a_s: torch.ones(4, 160, device="cuda").to(
                    torch.float8_e8m0fnu
                ),
                W.v4_attn_wkv_w: torch.randn(512, 5120, device="cuda").to(
                    torch.float8_e4m3fn
                ),
                W.v4_attn_wkv_s: torch.ones(16, 160, device="cuda").to(
                    torch.float8_e8m0fnu
                ),
            }
            proposal = merge_v41_qkv_weights(
                weights,
                W.v4_attn_wq_a_w,
                W.v4_attn_wq_a_s,
                W.v4_attn_wkv_w,
                W.v4_attn_wkv_s,
            )
            owners.append(weights)
            proposals.append(proposal)
            linears.append(
                V41MXFP8Linear(weights[W.v4_attn_wkv_w], weights[W.v4_attn_wkv_s])
            )
        with patch.object(commit_wkv, "sleep_enabled", return_value=False):
            prepared = commit_wkv.prepare_dspark_commit_wkv(
                linears, "stacked", layer_weights=owners, merged_linears=proposals
            )
        buffers = dict(prepared.named_buffers())
        self.assertIs(buffers["stacked.weight"], prepared.stacked.weight)
        self.assertIs(buffers["stacked.weight_scales"], prepared.stacked.weight_scales)
        self.assertEqual(prepared.state_dict(), {})
        for weights, linear in zip(owners, linears):
            self.assertIs(weights[W.v4_attn_wkv_w], linear.weight)
            self.assertEqual(
                linear.weight.untyped_storage().data_ptr(),
                prepared.stacked.weight.data_ptr(),
            )
        pointers = (
            prepared.stacked.weight.data_ptr(),
            prepared.stacked.weight_scales.data_ptr(),
        )
        x = torch.randn(6, 5120, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            prepared.project(x)
            for proposal in proposals:
                proposal(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            committed = prepared.project(x)
            proposed = [proposal(x) for proposal in proposals]
        original = [owner[W.v4_attn_wkv_w].clone() for owner in owners]
        query = [proposal.weight[:128].clone() for proposal in proposals]
        initial_result = [v.clone() for v in prepared.project(x)]
        for factor in (-1.0, 0.5):
            for weights, initial in zip(owners, original):
                weights[W.v4_attn_wkv_w].copy_(
                    (initial.float() * factor).to(initial.dtype)
                )
                weights[W.v4_attn_wkv_s].copy_(
                    torch.full((16, 160), 2.0, device="cuda").to(torch.float8_e8m0fnu)
                )
            prepared.reload(owners)
            graph.replay()
            self.assertEqual(
                pointers,
                (
                    prepared.stacked.weight.data_ptr(),
                    prepared.stacked.weight_scales.data_ptr(),
                ),
            )
            self.assertTrue(prepared.matches(linears))
            for i, linear in enumerate(linears):
                exact(self, committed[i], linear(x))
                exact(self, proposed[i][..., -512:], linear(x))
                exact(self, proposals[i].weight[:128], query[i])
            if factor == -1.0:
                self.assertFalse(torch.equal(committed[0], initial_result[0]))
        # Tensor replacement without rebinding is rejected before captured storage changes.
        saved = linears[0].weight
        try:
            linears[0].weight = saved.clone()
            with self.assertRaisesRegex(RuntimeError, "rebuild model and graphs"):
                prepared.reload(owners)
        finally:
            linears[0].weight = saved

    def test_model_weight_hook_reuses_owner_without_history_list(self):
        import rtp_llm.models_py.model_desc.deepseek_v41_base_dspark_model as module
        from rtp_llm.utils.model_weight import W

        owners = [
            {
                W.v4_attn_wkv_w: linear.weight,
                W.v4_attn_wkv_s: torch.ones(16, 160, device="cuda").to(
                    torch.float8_e8m0fnu
                ),
            }
            for linear in self.linears
        ]
        weights = SimpleNamespace(
            weights=owners,
            global_weights={
                W.v4_dspark_main_norm: object(),
                W.v4_dspark_main_proj_w: object(),
                W.v4_dspark_main_proj_s: object(),
            },
        )
        model = SimpleNamespace(
            _v4_args=SimpleNamespace(norm_eps=1e-6),
            v4=SimpleNamespace(
                layers=[
                    SimpleNamespace(attn=SimpleNamespace(wkv=w)) for w in self.linears
                ]
            ),
        )
        with patch.object(module, "RMSNorm"), patch.object(
            module, "_v4_fp8_linear"
        ), patch.dict(
            "os.environ", {"DSV4_DSPARK_COMMIT_WKV": "stacked"}
        ), patch.object(
            commit_wkv, "sleep_enabled", return_value=False
        ):
            module.DeepSeekV4DSparkModel._load_extra_weights(model, weights)
            prepared = model._commit_wkv_projection
            for _ in range(2):
                module.DeepSeekV4DSparkModel._load_extra_weights(model, weights)
                self.assertIs(model._commit_wkv_projection, prepared)
        self.assertFalse(hasattr(model, "_commit_wkv_preparations"))
        self.assertTrue(prepared.matches(self.linears))

    def test_fallback_and_capture_preparation(self):
        self.assertIsNone(commit_wkv.prepare_dspark_commit_wkv([]))
        self.assertIsNone(
            commit_wkv.prepare_dspark_commit_wkv([torch.nn.Identity()] * 3)
        )
        prepared = self.prepare("stacked")
        old = self.linears[0].weight_scales
        try:
            self.linears[0].weight_scales = old.contiguous()
            self.assertIsNone(commit_wkv.prepare_dspark_commit_wkv(self.linears))
            self.assertFalse(prepared.matches(self.linears))
        finally:
            self.linears[0].weight_scales = old
        x = torch.randn(6, 5120, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(prepared.project(x.float()))
        self.assertIsNone(prepared.project(x[:0]))
        with patch.object(commit_wkv, "sleep_enabled", return_value=True):
            automatic = commit_wkv.prepare_dspark_commit_wkv(self.linears, "auto")
        self.assertIsNone(automatic.stacked)
        self.assertIsNone(automatic.project(x))
        with self.assertRaises(ValueError):
            commit_wkv.prepare_dspark_commit_wkv(self.linears, "invalid")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            x.add_(1)
            with self.assertRaisesRegex(RuntimeError, "before capture"):
                self.prepare("stacked")
        torch.cuda.current_stream().wait_stream(stream)


if __name__ == "__main__":
    unittest.main()
