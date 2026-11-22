"""Qualify the PPU adapter, carried state and actual downstream consumers."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.dsv41.ppu.mhc import MHCConfig, PPUStepMix
from rtp_llm.models_py.modules.dsv41.ppu.mhc_mega import PPUMegaMHC
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.prefill import _rms_norm
from rtp_llm.models_py.modules.dsv41.ppu.test.test_mhc_mega_probe import (
    OUTPUTS,
    _canaries,
    _make_case,
    _qualification_metric,
    independent_transition_reference,
)


class ValidatedPPUOps(PPUOps):
    """Opt-in standard-driver validation using live model weights/activations.

    The actual model caller and native operation are unchanged. An independent
    CPU IEEE oracle checks all five outputs before returning the native result;
    the actual MoE consumer must receive that exact normalized buffer once.
    This class is for correctness runs, never performance measurements.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validation = []
        self._consumed = {"prefill": 0, "decode": 0}
        self._pending = None

    def try_mhc_ffn_transition(
        self, branch, residual, state, parameters, norm, eps, *, phase
    ):
        assert self._pending is None, "previous normalized output was not consumed"
        inputs = {
            "x": branch.detach().cpu(),
            "residual": residual.detach().cpu(),
            "shifted_prev_mix": state.pre_mix.detach().cpu().unsqueeze(-1),
            "post_mix": state.post_mix.detach().cpu(),
            "comb_res_mix": state.residual_mix.detach().cpu(),
            "fn": parameters["fn"].detach().cpu(),
            "mix_scales": parameters["scale"].detach().cpu(),
            "mix_bases": parameters["base"].detach().cpu(),
            "rmsnorm_weight": norm.detach().cpu(),
        }
        result = super().try_mhc_ffn_transition(
            branch,
            residual,
            state,
            parameters,
            norm,
            eps,
            phase=phase,
        )
        if result is None:
            return None
        cfg = self._mhc.config
        scalars = dict(
            hc_norm_eps=cfg.rms_eps,
            hc_pre_eps=cfg.hc_eps,
            hc_post_scale=cfg.post_mult,
            sinkhorn_eps=cfg.hc_eps,
            num_sinkhorn_iters=cfg.sinkhorn_iters,
            rmsnorm_eps=eps,
            rmsnorm_scale=1.0,
        )
        expected = independent_transition_reference(torch, inputs, scalars)
        outputs = dict(
            zip(
                OUTPUTS,
                (
                    result.residual,
                    result.pre_mix.unsqueeze(-1),
                    result.post_mix,
                    result.residual_mix,
                    result.normalized,
                ),
            )
        )
        metrics = {
            name: _qualification_metric(torch, value, expected[name])
            for name, value in outputs.items()
        }
        assert all(m["qualified_status"] == "PASS" for m in metrics.values()), metrics
        self._validation.append(
            {"phase": phase, "rows": branch.shape[0], "metrics": metrics}
        )
        self._pending = (phase, result.normalized)
        return result

    def moe(self, x, expert_ids, expert_weights, *, layer_id, context=None):
        pending = self._pending
        if pending is not None:
            assert x is pending[1], "normalized output copied or normalized twice"
        output = super().moe(
            x, expert_ids, expert_weights, layer_id=layer_id, context=context
        )
        if pending is not None:
            self._consumed[pending[0]] += 1
            self._pending = None
        return output

    def mhc_execution_stats(self):
        assert self._pending is None, "unconsumed Mega-mHC output"
        return {
            **super().mhc_execution_stats(),
            "validated": self._validation,
            "consumed": self._consumed,
        }


class MegaIntegrationTest(unittest.TestCase):
    def test_selection_and_native_error(self):
        with patch.dict(os.environ):
            os.environ.pop("RTP_V41_PPU_MHC_BACKEND", None)
            self.assertEqual(PPUMegaMHC().backend, "mega")
        with patch.dict(os.environ, {"RTP_V41_PPU_MHC_BACKEND": "invalid"}):
            with self.assertRaises(ValueError):
                PPUMegaMHC()
        with patch.dict(os.environ, {"RTP_V41_PPU_MHC_BACKEND": "legacy"}):
            adapter = PPUMegaMHC()
        self.assertIsNone(
            adapter.try_transition(
                None, None, None, {}, None, 1e-20, MHCConfig(), phase="decode"
            )
        )
        # An error after dispatch must propagate, with no second launch.
        with patch.object(adapter, "_reason", return_value=None), patch(
            "rtp_llm.models_py.modules.dsv41.ppu.mhc_mega._qualified_operation",
            side_effect=RuntimeError("native identity failure"),
        ) as native:
            with self.assertRaisesRegex(RuntimeError, "native identity"):
                adapter.try_transition(
                    None, None, None, {}, None, 1e-20, MHCConfig(), phase="decode"
                )
            self.assertEqual(native.call_count, 1)

    def test_graph_enabled_by_default_with_explicit_opt_out(self):
        with patch.dict(os.environ, {"RTP_V41_PPU_MHC_BACKEND": "mega"}):
            os.environ.pop("RTP_V41_PPU_GRAPH_MEGA_MHC", None)
            self.assertTrue(PPUMegaMHC().graph_enabled)
            os.environ["RTP_V41_PPU_GRAPH_MEGA_MHC"] = "0"
            adapter = PPUMegaMHC()
        self.assertIsNone(
            adapter.try_transition(
                None, None, None, {}, None, 1e-20, MHCConfig(), phase="graph"
            )
        )
        self.assertEqual(adapter.execution_stats()["fallbacks"], {"graph:unqualified_phase": 1})

    @unittest.skipUnless(os.environ.get("RTP_V41_TEST_MHC_DEVICE") == "1", "PPU only")
    def test_device_graph_replay_matches_current_inputs(self):
        with patch.dict(os.environ, {"RTP_V41_PPU_MHC_BACKEND": "mega"}):
            os.environ.pop("RTP_V41_PPU_GRAPH_MEGA_MHC", None)
            adapter = PPUMegaMHC()
        device = torch.device("cuda:0")
        stream = torch.cuda.Stream(device=device)
        config = MHCConfig(rms_eps=1e-20)

        def transition(inputs):
            state = PPUStepMix(
                hidden=None, pre_mix=inputs["shifted_prev_mix"].squeeze(-1),
                post_mix=inputs["post_mix"], residual_mix=inputs["comb_res_mix"],
            )
            return adapter.try_transition(
                inputs["x"], inputs["residual"], state,
                {"fn": inputs["fn"], "base": inputs["mix_bases"], "scale": inputs["mix_scales"]},
                inputs["rmsnorm_weight"], 1e-20, config, phase="graph",
            )

        captures = []
        for rows in (1, 2):
            inputs, _, _, _ = _make_case(torch, rows, device, "normal")
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                self.assertIsNotNone(transition(inputs))
            torch.cuda.current_stream(device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = transition(inputs)
            captures.append((graph, inputs, actual))
        for _ in range(2):
            for graph, inputs, actual in reversed(captures):
                inputs["x"].neg_()
                inputs["residual"].mul_(0.5)
                expected = transition(inputs)
                graph.replay()
                for name, value in vars(actual).items():
                    torch.testing.assert_close(value, getattr(expected, name), rtol=0, atol=0)
        self.assertEqual(adapter.execution_stats()["graph_capture_calls"], 2)
        self.assertFalse(adapter.execution_stats()["fallbacks"])

    @unittest.skipUnless(os.environ.get("RTP_V41_TEST_MHC_DEVICE") == "1", "PPU only")
    @torch.inference_mode()
    def test_graph_verify_aux_boundary(self):
        from types import SimpleNamespace
        from rtp_llm.models_py.modules.dsv41.hc.decode_transition import PendingDecodePost
        from rtp_llm.models_py.modules.dsv41.ppu._mhc_triton import try_verify_mhc_post
        from rtp_llm.models_py.modules.dsv41.ppu.decode_mhc import (
            finish_decode_boundary, prepare_decode_boundary,
        )
        from rtp_llm.models_py.modules.dsv41.ppu.model import SharedPPUTargetBlock
        from rtp_llm.models_py.modules.dsv41.transformer import V4Transformer

        device = torch.device("cuda:0")
        stream = torch.cuda.Stream(device=device)
        ops = PPUOps(num_local_experts=384, mhc_config=MHCConfig(rms_eps=1e-20))
        for batch in (4, 8, 16, 32):
            rows = batch * 4
            inputs, _, _, _ = _make_case(torch, rows, device, "normal")
            shape = (batch, 4)
            residual = inputs["residual"].view(*shape, 4, 5120)
            branch = inputs["x"].view(*shape, 5120)
            mix = PPUStepMix(
                hidden=None,
                pre_mix=inputs["shifted_prev_mix"].view(*shape, 4),
                post_mix=inputs["post_mix"].view(*shape, 4, 1),
                residual_mix=inputs["comb_res_mix"].view(*shape, 4, 4),
            )
            v4 = SimpleNamespace(
                layers=[], capture_aux_hidden_layer_ids=(1, 0),
                _mtp_hidden_buffer=torch.empty((rows, 10240), dtype=torch.bfloat16, device=device),
            )
            state = SimpleNamespace(shared={}, is_graph=True, verify=True, cache=None)
            model = SimpleNamespace(v4=v4, parallelism_config=SimpleNamespace(tp_size=1))
            runtime = SimpleNamespace(current=state, model=model)
            for layer_id in (0, 1):
                layer = SharedPPUTargetBlock.__new__(SharedPPUTargetBlock)
                torch.nn.Module.__init__(layer)
                layer.layer_id = layer_id
                layer.query_only = False
                layer._mhc_phase = "graph_verify"
                layer.ops = ops
                layer.attn = SimpleNamespace(runtime=runtime)
                layer.engram = None
                v4.layers.append(layer)
            src, dst = v4.layers
            src.ffn_hc = SimpleNamespace(state=mix)
            params = dict(fn=inputs["fn"], base=inputs["mix_bases"], scale=inputs["mix_scales"])
            accepted = []
            dst.attn_hc = SimpleNamespace(
                **params, parameters_map=params, _previous_ref=lambda: src.ffn_hc,
                accept_transition=accepted.append,
            )
            dst.attn_norm = SimpleNamespace(weight=inputs["rmsnorm_weight"], variance_epsilon=1e-20)
            metadata = SimpleNamespace(batch_size=batch, q_len_per_req=4)
            ids = torch.zeros(shape, device=device, dtype=torch.int64)
            pending = PendingDecodePost(branch, residual, mix.post_mix, mix.residual_mix, src.ffn_hc)

            def run():
                _, target, scope = prepare_decode_boundary(src, state, residual, metadata, ids, {})
                self.assertIs(target, dst)
                completed = finish_decode_boundary(src, state, pending, target, scope)
                V4Transformer.capture_aux_hidden(v4, dst.layer_id, completed)
                prepared, _, _ = prepare_decode_boundary(dst, state, completed, metadata, ids, {})
                self.assertIs(prepared[0], completed)
                return completed

            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream(device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = run()
            branch.neg_()
            graph.replay()
            expected = try_verify_mhc_post(branch, residual, mix.post_mix, mix.residual_mix, mode="dot")
            aux = v4._mtp_hidden_buffer[:, :5120]
            torch.testing.assert_close(
                aux, actual.mean(dim=-2).view(rows, 5120),
                rtol=0, atol=0,
            )
            for name, observed, reference in (
                ("residual", actual.view(rows, 4, 5120), expected.view(rows, 4, 5120)),
                ("aux", aux, expected.mean(dim=-2).view(rows, 5120)),
            ):
                metric = _qualification_metric(torch, observed, reference)
                self.assertEqual(metric["qualified_status"], "PASS", metric["global"])
                print(f"batch={batch} {name} sse_over_energy={metric['global']['score']:.3g}", flush=True)

    @unittest.skipUnless(os.environ.get("RTP_V41_TEST_MHC_DEVICE") == "1", "PPU only")
    @torch.inference_mode()
    def test_graph_proposal_mhc_post_batch_sizes(self):
        from rtp_llm.models_py.modules.dsv41.ppu._mhc_triton import try_verify_mhc_post
        from rtp_llm.models_py.modules.dsv41.ppu.mhc import mhc_post

        device = torch.device("cuda:0")
        stream = torch.cuda.Stream(device=device)
        for batch in (4, 8, 16, 32):
            with self.subTest(batch=batch):
                rows = batch * 3
                inputs, _, _, _ = _make_case(torch, rows, device, "normal")
                residual = inputs["residual"].view(batch, 3, 4, 5120)
                branch = inputs["x"].view(batch, 3, 5120)
                post = inputs["post_mix"].view(batch, 3, 4, 1)
                comb = inputs["comb_res_mix"].view(batch, 3, 4, 4)
                state = PPUStepMix(hidden=None, post_mix=post, residual_mix=comb)

                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    warm = try_verify_mhc_post(branch, residual, post, comb, mode="dot", proposal_q3=True)
                self.assertIsNotNone(warm)
                torch.cuda.current_stream(device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = try_verify_mhc_post(branch, residual, post, comb, mode="dot", proposal_q3=True)
                branch.neg_()
                graph.replay()
                expected = mhc_post(residual, branch, state=state).hidden
                self.assertFalse(torch.equal(actual, warm))
                metric = _qualification_metric(torch, actual.view(rows, 4, 5120), expected.view(rows, 4, 5120))
                self.assertEqual(metric["qualified_status"], "PASS", metric["global"])
                print(f"proposal batch={batch} post sse_over_energy={metric['global']['score']:.3g}", flush=True)

    @unittest.skipUnless(os.environ.get("RTP_V41_TEST_MHC_DEVICE") == "1", "PPU only")
    def test_device_transition_state_and_consumers(self):
        reports, first_rows = [], {}
        with patch.dict(os.environ):
            os.environ.pop("RTP_V41_PPU_MHC_BACKEND", None)
            ops = PPUOps(num_local_experts=384, mhc_config=MHCConfig(rms_eps=1e-20))
        saved_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for rows in (1, 3, 4, 5, 6, 64, 65, 129):
                inputs, unused, buffers, _ = _make_case(torch, rows, "cuda:0", "normal")
                originals = {name: value.clone() for name, value in inputs.items()}
                state = PPUStepMix(
                    hidden=torch.empty_like(inputs["x"]),
                    pre_mix=inputs["shifted_prev_mix"].squeeze(-1),
                    post_mix=inputs["post_mix"],
                    residual_mix=inputs["comb_res_mix"],
                )
                parameters = {
                    "fn": inputs["fn"],
                    "base": inputs["mix_bases"],
                    "scale": inputs["mix_scales"],
                }
                scalars = dict(
                    hc_norm_eps=1e-20,
                    hc_pre_eps=1e-6,
                    hc_post_scale=2.0,
                    sinkhorn_eps=1e-6,
                    num_sinkhorn_iters=20,
                    rmsnorm_eps=1e-20,
                    rmsnorm_scale=1.0,
                )
                expected = independent_transition_reference(torch, inputs, scalars)
                results = []
                for phase in ("prefill", "decode"):
                    result = ops.try_mhc_ffn_transition(
                        inputs["x"],
                        inputs["residual"],
                        state,
                        parameters,
                        inputs["rmsnorm_weight"],
                        1e-20,
                        phase=phase,
                    )
                    self.assertIsNotNone(result, ops.mhc_execution_stats())
                    self.assertFalse(hasattr(result, "hidden"))
                    outputs = dict(
                        zip(
                            OUTPUTS,
                            (
                                result.residual,
                                result.pre_mix.unsqueeze(-1),
                                result.post_mix,
                                result.residual_mix,
                                result.normalized,
                            ),
                        )
                    )
                    metrics = {}
                    for name, value in outputs.items():
                        self.assertEqual(value.shape, expected[name].shape)
                        self.assertEqual(value.dtype, expected[name].dtype)
                        self.assertTrue(bool(torch.isfinite(value).all()))
                        metric = _qualification_metric(torch, value, expected[name])
                        self.assertEqual(
                            metric["qualified_status"], "PASS", (name, metric)
                        )
                        metrics[name] = metric
                        # Row zero and all weights are identical across M.
                        if name in first_rows:
                            cross = _qualification_metric(
                                torch, value[:1], first_rows[name]
                            )
                            self.assertEqual(cross["qualified_status"], "PASS", cross)
                        else:
                            first_rows[name] = value[:1].clone()
                    all_values = list(inputs.values()) + list(outputs.values())
                    for name, output in outputs.items():
                        lo, hi = (
                            output.data_ptr(),
                            output.data_ptr() + output.numel() * output.element_size(),
                        )
                        for other in all_values:
                            if other is output:
                                continue
                            a, b = (
                                other.data_ptr(),
                                other.data_ptr() + other.numel() * other.element_size(),
                            )
                            self.assertFalse(lo < b and a < hi, name)
                    results.append(outputs)
                    reports.append({"rows": rows, "phase": phase, "metrics": metrics})
                    # Use a deterministic stand-in branch to isolate the FFN
                    # post consumer from router/MoE errors; model tests use MoE.
                    branch = result.normalized * 0.125
                    actual_post = ops.mhc(
                        result.residual, branch, phase="post", state=result
                    )
                    post_reference = (
                        expected["new_post_mix"] * branch.float().unsqueeze(1)
                        + torch.einsum(
                            "tij,tih->tjh",
                            expected["new_comb_res_mix"],
                            expected["new_residual"].float(),
                        )
                    ).to(torch.bfloat16)
                    metric = _qualification_metric(
                        torch, actual_post.hidden, post_reference
                    )
                    self.assertEqual(metric["qualified_status"], "PASS", metric)
                    # Next attention pre/head consume the NEW FFN pre, while
                    # FFN normalized input above used the carried ATTENTION pre.
                    head = ops.mhc(
                        actual_post.hidden, None, phase="head", state=result
                    ).hidden
                    expected_head = (
                        (actual_post.hidden.float() * expected["new_prev_mix"])
                        .sum(1)
                        .to(torch.bfloat16)
                    )
                    self.assertEqual(
                        _qualification_metric(torch, head, expected_head)[
                            "qualified_status"
                        ],
                        "PASS",
                    )
                    next_pre = ops.mhc(
                        actual_post.hidden,
                        None,
                        phase="pre",
                        parameters=parameters,
                        state=result,
                    )
                    self.assertTrue(torch.equal(head, next_pre.hidden))
                    # Independent unfused chain checks once-only RMSNorm.
                    legacy_post = ops.mhc(
                        inputs["residual"], inputs["x"], phase="post", state=state
                    )
                    legacy_pre = ops.mhc(
                        legacy_post.hidden,
                        None,
                        phase="pre",
                        parameters=parameters,
                        state=state,
                    )
                    norm = _rms_norm(legacy_pre.hidden, inputs["rmsnorm_weight"], 1e-20)
                    self.assertEqual(
                        _qualification_metric(torch, result.normalized, norm)[
                            "qualified_status"
                        ],
                        "PASS",
                    )
                for name in OUTPUTS:
                    self.assertTrue(
                        torch.equal(results[0][name], results[1][name]), name
                    )
                for name, value in inputs.items():
                    self.assertTrue(torch.equal(value, originals[name]), name)
                self.assertTrue(
                    all(
                        all(item.values())
                        for item in _canaries(torch, buffers).values()
                    )
                )
            self.assertEqual(
                ops.mhc_execution_stats()["calls"], {"prefill": 8, "decode": 8}
            )
            target = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
            (target / "mhc-integration.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "stats": ops.mhc_execution_stats(),
                        "cases": reports,
                    },
                    indent=2,
                )
                + "\n"
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = saved_tf32


if __name__ == "__main__":
    unittest.main()
