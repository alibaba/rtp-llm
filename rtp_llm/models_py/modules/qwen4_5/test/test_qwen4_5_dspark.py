#!/usr/bin/env python3
"""Unit & GPU tests for Qwen4.5-Flash 5-layer DFlash2 (DFlashQwenDraftModel / SP_TYPE_DSPARK)."""

import glob
import json
import tempfile
import types
import os
import sys
import unittest
from typing import Dict, Optional, Sequence, Tuple, Any, Dict, List, Optional


def _ensure_runfiles_on_sys_path() -> None:
    try:
        import rtp_llm.ops  # noqa: F401
        import safetensors  # noqa: F401
        return
    except ImportError:
        pass
    wt = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
    rf = os.path.join(
        wt,
        "bazel-bin/rtp_llm/models_py/modules/qwen4_5/test/qwen4_5_pd_serving_test.runfiles",
    )
    if os.path.isdir(rf):
        extra = [wt, os.path.join(rf, "rtp_llm")]
        for d in glob.glob(os.path.join(rf, "pip_gpu_cuda13_*")):
            sp = os.path.join(d, "site-packages")
            extra.append(sp if os.path.isdir(sp) else d)
        for p in reversed(extra):
            if p not in sys.path:
                sys.path.insert(0, p)


_ensure_runfiles_on_sys_path()

from safetensors import safe_open
import torch

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.qwen4_5.qwen4_5_dspark import (
    DFLASH2_W,
    DFlash2ArchSpec,
    Qwen4_5DSpark,
    Qwen4_5DSparkWeight,
    dspark_offset_d2t_to_absolute,
    dspark_parse_target_layer_ids,
)
from rtp_llm.models_py.model_desc.qwen4_5_dspark_model import (
    CandidateSelector,
    DFlashGroupedConv,
    QWEN4_5_OPAQUE_STATE_TAGS,
    Qwen4_5DSparkModel,
    Qwen4_5OpaqueStateTransaction,
    _DFlash2PagedFMHADispatcher,
    compute_qsa_indexer_state_lengths,
    extract_qwen4_5_aux_hidden_states,
    greedy_select_path_ref,
    grouped_conv_ref,
    rollback_qwen4_5_opaque_states,
    scatter_kv_to_paged_cache,
    snapshot_qwen4_5_opaque_states,
)
from rtp_llm.ops import ParallelismConfig, SpeculativeType
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, PyModelInputs
from rtp_llm.utils.model_weight import W

_SANDBOX_MOCK_TMP = None


def _resolve_mock_dflash_dirs():
    global _SANDBOX_MOCK_TMP
    p8 = "/data0/luohaocheng.lhc/models/Qwen4.5-Flash-8L-dflash-mock"
    p40 = "/data0/luohaocheng.lhc/models/Qwen4.5-Flash-1109-dflash-mock"
    if os.path.isdir(p8) and os.path.isdir(p40):
        return p8, p40

    if _SANDBOX_MOCK_TMP is None:
        import atexit
        from collections import OrderedDict
        from pathlib import Path
        from safetensors.torch import save_file
        from rtp_llm.models_py.modules.qwen4_5.tools.build_dflash_mock_checkpoint import (
            DEFAULT_CONFIGS,
            _make_tensor,
            _shard_plans,
            _tensor_specs,
            build_dflash2_config,
        )

        _SANDBOX_MOCK_TMP = tempfile.TemporaryDirectory(prefix="dflash_sandbox_mock_")
        atexit.register(_SANDBOX_MOCK_TMP.cleanup)
        root = Path(_SANDBOX_MOCK_TMP.name)

        # Build 8L full mock weights cleanly inside sandbox
        d8 = root / "8l"
        d8.mkdir(parents=True, exist_ok=True)
        base8 = {"vocab_size": 2048, "num_hidden_layers": 8, "partial_rotary_factor": 0.25, "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "max_position_embeddings": 512}
        cfg8 = build_dflash2_config(base8, dict(DEFAULT_CONFIGS["synthetic-8l"]), seed=42)
        (d8 / "config.json").write_text(json.dumps(cfg8), encoding="utf-8")
        gen = torch.Generator(device="cpu").manual_seed(42)
        specs8 = _tensor_specs(cfg8)
        wmap8 = OrderedDict()
        tensors8 = OrderedDict()
        total8 = 0
        for name, shape, dt_tag, init_k in specs8:
            t = _make_tensor(gen, shape, dt_tag, init_k)
            tensors8[name] = t
            wmap8[name] = "model-00001-of-00001.safetensors"
            total8 += int(t.numel() * t.element_size())
        save_file(tensors8, str(d8 / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
        (d8 / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total8}, "weight_map": wmap8}),
            encoding="utf-8",
        )

        # Build 40L config + weight_map index only (zero weight allocation!)
        d40 = root / "40l"
        d40.mkdir(parents=True, exist_ok=True)
        base40 = {"vocab_size": 248320, "num_hidden_layers": 40, "partial_rotary_factor": 0.25, "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "max_position_embeddings": 262144}
        cfg40 = build_dflash2_config(base40, dict(DEFAULT_CONFIGS["40l"]), seed=42)
        (d40 / "config.json").write_text(json.dumps(cfg40), encoding="utf-8")
        specs40 = _tensor_specs(cfg40)
        plans40 = _shard_plans(specs40, 1_900_000_000)
        wmap40 = OrderedDict()
        for idx, shard_items in enumerate(plans40, start=1):
            sname = f"model-{idx:05d}-of-{len(plans40):05d}.safetensors"
            for name, _s, _d, _k in shard_items:
                wmap40[name] = sname
        (d40 / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 0}, "weight_map": wmap40}),
            encoding="utf-8",
        )

    root = Path(_SANDBOX_MOCK_TMP.name)
    return str(root / "8l"), str(root / "40l")


MOCK_8L_DFLASH, MOCK_40L_DFLASH = _resolve_mock_dflash_dirs()


def _load_dspark_model_weights(
    ckpt_dir: str, device: str = "cpu"
) -> tuple[Any, ModelWeights]:
    cfg = Qwen4_5DSpark._create_config(ckpt_dir)
    with open(
        os.path.join(ckpt_dir, "model.safetensors.index.json"), "r", encoding="utf-8"
    ) as f:
        weight_map: Dict[str, str] = json.load(f)["weight_map"]

    raw_tensors: Dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(weight_map.values())):
        with safe_open(
            os.path.join(ckpt_dir, shard_name), framework="pt", device="cpu"
        ) as sf:
            for k in sf.keys():
                raw_tensors[k] = sf.get_tensor(k)

    weight_builder = Qwen4_5DSparkWeight(cfg)
    weight_builder._process_meta({}, list(raw_tensors.keys()))
    winfo = weight_builder._get_weight_info()

    mw = ModelWeights(cfg.num_layers, device=device, dtype=torch.bfloat16)

    def _eval_atomic(atomic_w) -> torch.Tensor:
        target_dt = getattr(atomic_w, "data_type", None) or torch.bfloat16
        loaded = [raw_tensors[ck.name].to(dtype=target_dt) for ck in atomic_w.weights]
        t = atomic_w.process_fun(loaded)
        return t.to(device=device, dtype=target_dt).contiguous()

    for gw in winfo.weights:
        mw.set_global_weight(gw.name, _eval_atomic(gw))

    for lid, layer_ws in enumerate(winfo.layer_weights):
        for lw in layer_ws:
            for comp in lw.get_components():
                mw.set_layer_weight(lid, comp.name, _eval_atomic(comp))

    return cfg, mw


class _MultiLayerDictKVCache:
    """5-layer LayerKVCache container matching GptModelBase.kv_cache calls."""

    def __init__(self, layer_kv_list: List[LayerKVCache]) -> None:
        self._layers = {i: kv for i, kv in enumerate(layer_kv_list)}

    def get_layer_cache(
        self, layer_idx: int, tag: str = "default"
    ) -> Optional[LayerKVCache]:
        del tag
        return self._layers.get(layer_idx)

    def get_kernel_seq_size_per_block(self, tag: str = "full") -> int:
        del tag
        return int(self._layers[0].seq_size_per_block)


def _make_dflash2_inputs(
    *,
    batch_size: int,
    prefix_lens: List[int],
    query_width: int,
    anchor_ids: List[int],
    block_table: torch.Tensor,
    noise_token_id: int = 3,
    input_hiddens: Optional[torch.Tensor] = None,
    device: str = "cpu",
    use_device_mirrors: bool = True,
    is_target_verify: bool = True,
    active_query_lens: Optional[List[int]] = None,
) -> PyModelInputs:
    q_lens = active_query_lens or [query_width] * batch_size
    pref_cpu = torch.tensor(prefix_lens, dtype=torch.int32)
    inp_cpu = torch.tensor(q_lens, dtype=torch.int32)
    empty_decode_seq = torch.empty(0, dtype=torch.int32)
    cu_vals = [0]
    for ql in q_lens:
        cu_vals.append(cu_vals[-1] + ql)
    cu_cpu = torch.tensor(cu_vals, dtype=torch.int32)

    attn = PyAttentionInputs()
    attn.is_prefill = True
    attn.is_target_verify = is_target_verify
    attn.total_tokens = int(sum(q_lens))

    if device.startswith("cuda"):
        attn.prefix_lengths = pref_cpu.to(device)
        attn.input_lengths = inp_cpu.to(device)
        attn.sequence_lengths = empty_decode_seq
        attn.cu_seqlens = cu_cpu.to(device)
        attn.kv_cache_kernel_block_id = block_table.cpu().to(torch.int32)
        attn.kv_cache_block_id = attn.kv_cache_kernel_block_id
    else:
        attn.prefix_lengths = pref_cpu
        attn.input_lengths = inp_cpu
        attn.sequence_lengths = empty_decode_seq
        attn.cu_seqlens = cu_cpu
        attn.kv_cache_kernel_block_id = block_table.to(torch.int32)
        attn.kv_cache_block_id = attn.kv_cache_kernel_block_id

    if use_device_mirrors:
        attn.cu_seqlens_device = cu_cpu.to(device)
        attn.kv_cache_kernel_block_id_device = block_table.to(
            device=device, dtype=torch.int32
        )
        attn.kv_cache_block_id_device = attn.kv_cache_kernel_block_id_device

    inputs = PyModelInputs()
    inputs.attention_inputs = attn
    toks: List[int] = []
    if is_target_verify:
        for b in range(batch_size):
            n_tok = max(0, q_lens[b])
            if n_tok > 0:
                toks.extend([anchor_ids[b]] + [noise_token_id] * (n_tok - 1))
    else:
        for b in range(batch_size):
            toks.extend([anchor_ids[b]] + [noise_token_id] * (query_width - 1))
    inputs.input_ids = torch.tensor(toks, dtype=torch.int32, device=device)
    if input_hiddens is not None:
        inputs.input_hiddens = input_hiddens.to(device)
    return inputs


def _build_dummy_target_opaque_state(
    num_blocks: int, device: str
) -> Dict[str, torch.Tensor]:
    return {
        "gdn_ssm_state": torch.zeros((num_blocks, 4, 16, 16), dtype=torch.float32, device=device),
        "gdn_conv_state": torch.zeros((num_blocks, 48, 3), dtype=torch.bfloat16, device=device),
        "gdn_consumer_q_conv": torch.zeros((num_blocks, 16, 3), dtype=torch.bfloat16, device=device),
        "full_attn_kv_sconv": torch.zeros((num_blocks, 2, 16, 3), dtype=torch.bfloat16, device=device),
        "full_attn_output_sconv": torch.zeros((num_blocks, 32, 3), dtype=torch.bfloat16, device=device),
        "linear_attn_output_sconv": torch.zeros((num_blocks, 32, 3), dtype=torch.bfloat16, device=device),
        "ffn_output_sconv": torch.zeros((num_blocks, 32, 3), dtype=torch.bfloat16, device=device),
        "ple_conv_state": torch.zeros((num_blocks, 64, 3), dtype=torch.bfloat16, device=device),
        "ple_ngram_ctx": torch.zeros((num_blocks, 3), dtype=torch.int64, device=device),
        "linear": torch.zeros((num_blocks, 4, 8, 8), dtype=torch.float32, device=device),
        "indexer_state": torch.zeros((num_blocks, 3, 16), dtype=torch.bfloat16, device=device),
        "indexer_kv": torch.zeros((num_blocks, 8, 16), dtype=torch.bfloat16, device=device),
    }


def _step_opaque_state_deterministic(
    state: Dict[str, torch.Tensor],
    block_ids: torch.Tensor,
    token_vals: torch.Tensor,
    active_mask: Optional[torch.Tensor] = None,
) -> None:
    B = int(block_ids.numel())
    for b in range(B):
        if active_mask is not None and not bool(active_mask[b].item()):
            continue
        bid = int(block_ids[b].item())
        tok = float(token_vals[b].item())
        tok_int = int(token_vals[b].item())
        state["gdn_ssm_state"][bid].mul_(0.9).add_(tok * 0.05)
        state["linear"][bid].mul_(0.85).add_(tok * 0.03)
        for conv_tag in (
            "gdn_conv_state",
            "gdn_consumer_q_conv",
            "full_attn_kv_sconv",
            "full_attn_output_sconv",
            "linear_attn_output_sconv",
            "ffn_output_sconv",
            "ple_conv_state",
            "indexer_state",
        ):
            buf = state[conv_tag][bid]
            shifted = buf.clone()
            shifted[..., :-1] = buf[..., 1:]
            shifted[..., -1] = tok * 0.01
            state[conv_tag][bid] = shifted
        ng = state["ple_ngram_ctx"][bid].clone()
        ng[:-1] = state["ple_ngram_ctx"][bid, 1:]
        ng[-1] = tok_int
        state["ple_ngram_ctx"][bid] = ng
        idx_slot = tok_int % state["indexer_kv"].size(1)
        state["indexer_kv"][bid, idx_slot, :] = tok * 0.02


class TestDFlash2ConfigAndWeights(unittest.TestCase):
    def test_synthetic_8l_config_and_full_weight_load(self):
        self.assertTrue(os.path.isdir(MOCK_8L_DFLASH))
        cfg, mw = _load_dspark_model_weights(MOCK_8L_DFLASH, device="cpu")
        spec = cfg._dflash2_spec
        self.assertIsInstance(spec, DFlash2ArchSpec)
        self.assertEqual(cfg.num_layers, 5)
        self.assertEqual(
            spec.layer_types,
            (
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ),
        )
        self.assertEqual(spec.full_attention_layer_ids, (4,))
        self.assertEqual(cfg.dspark_target_layer_ids, [0, 1, 2, 3, 4])
        self.assertEqual(spec.fc_in_features, 2560)
        self.assertTrue(cfg.dspark_sample_from_anchor)
        self.assertEqual(cfg.gen_num_per_cycle, 8)
        self.assertEqual(spec.query_width, 8)
        self.assertEqual(cfg.dspark_noise_token_id, 3)
        self.assertEqual(cfg.dspark_markov_rank, 64)
        self.assertEqual(cfg.attn_config.rope_config.dim, 64)
        self.assertFalse(cfg.attn_config.is_causal)

        self.assertEqual(tuple(mw.get_global_weight(W.dspark_fc_w).shape), (2560, 512))
        self.assertEqual(tuple(mw.get_global_weight(W.dspark_markov_w1).shape), (2048, 64))
        self.assertEqual(tuple(mw.get_global_weight(W.dspark_markov_w2).shape), (2048, 64))
        self.assertEqual(
            tuple(mw.get_global_weight(DFLASH2_W.selector_hidden_proj).shape), (64, 512)
        )
        self.assertEqual(
            tuple(mw.weights[0][DFLASH2_W.attn_output_gate].shape), (8 * 256, 512)
        )
        self.assertEqual(
            tuple(mw.weights[0][DFLASH2_W.attn_hc_up].shape), (512, 32)
        )
        self.assertEqual(
            tuple(mw.weights[1][DFLASH2_W.attn_hc_up].shape), (4 * 512, 32)
        )
        self.assertIn(DFLASH2_W.mlp_hc_down, mw.weights[3])
        self.assertNotIn(DFLASH2_W.mlp_hc_down, mw.weights[4])

    def test_40l_erke_config_and_weight_map(self):
        self.assertEqual(SpeculativeType.DSPARK.name, "DSPARK")
        self.assertTrue(os.path.isdir(MOCK_40L_DFLASH))
        cfg = Qwen4_5DSpark._create_config(MOCK_40L_DFLASH)
        spec = cfg._dflash2_spec
        self.assertEqual(cfg.num_layers, 5)
        self.assertEqual(cfg.hidden_size, 5120)
        self.assertEqual(cfg.vocab_size, 248320)
        self.assertEqual(cfg.attn_config.head_num, 32)
        self.assertEqual(cfg.attn_config.kv_head_num, 1)
        self.assertEqual(cfg.attn_config.size_per_head, 256)
        self.assertEqual(cfg.attn_config.rope_config.dim, 64)
        self.assertEqual(cfg.attn_config.sliding_window, 2048)
        self.assertFalse(cfg.attn_config.is_causal)
        self.assertEqual(cfg.dspark_target_layer_ids, [0, 5, 10, 15, 20])
        self.assertEqual(spec.fc_in_features, 25600)
        self.assertEqual(spec.target_hidden_source, "post_next_hc_mix")
        self.assertEqual(cfg.dspark_markov_rank, 256)
        self.assertEqual(spec.selector_top_k, 16)
        self.assertEqual(cfg.dspark_noise_token_id, 248280)
        self.assertEqual(cfg.gen_num_per_cycle, 8)
        self.assertTrue(cfg.dspark_sample_from_anchor)
        self.assertEqual(spec.num_experts, 640)
        self.assertEqual(spec.num_experts_per_tok, 10)
        self.assertEqual(spec.routed_expert_hidden_size, 2560)
        self.assertEqual(spec.moe_intermediate_size, 2560)
        self.assertEqual(spec.shared_expert_intermediate_size, 1280)

        with open(
            os.path.join(MOCK_40L_DFLASH, "model.safetensors.index.json"),
            "r",
            encoding="utf-8",
        ) as f:
            weight_map = json.load(f)["weight_map"]

        wb = Qwen4_5DSparkWeight(cfg)
        wb._process_meta({}, list(weight_map.keys()))
        winfo = wb._get_weight_info()
        w1 = next(w for w in winfo.weights if w.name == W.dspark_markov_w1)
        w2 = next(w for w in winfo.weights if w.name == W.dspark_markov_w2)
        self.assertEqual(w1.weights[0].name, "candidate_selector.predecessor_codebook")
        self.assertEqual(w2.weights[0].name, "candidate_selector.successor_codebook")
        self.assertEqual(len(winfo.layer_weights), 5)

        d2t_abs = dspark_offset_d2t_to_absolute(
            [torch.tensor([5, 10, -1], dtype=torch.int64)]
        )
        self.assertTrue(torch.equal(d2t_abs, torch.tensor([5, 11, 1], dtype=torch.int32)))
        self.assertEqual(
            dspark_parse_target_layer_ids({"dflash_config": {"target_layer_ids": [0, 5, 10, 15, 20]}}),
            [0, 5, 10, 15, 20],
        )

    def test_dflash2_mxfp4_build_model_config_and_precision(self):
        from rtp_llm.config.model_args import ModelArgs
        from rtp_llm.config.model_config import build_model_config
        from rtp_llm.config.quant_config import MXFp4QuarkQuantConfig
        from rtp_llm.ops import (
            KVCacheConfig,
            KVCacheSpecType,
            KvCacheDataType,
            ProfilingDebugLoggingConfig,
            TpCacheTransferMode,
        )

        for ckpt_dir in (MOCK_8L_DFLASH, MOCK_40L_DFLASH):
            cfg = Qwen4_5DSpark._create_config(ckpt_dir)
            cfg.is_mtp = True
            model_args = ModelArgs()
            model_args.ckpt_path = ckpt_dir
            model_args.model_type = "qwen4_5_dspark"
            model_args.act_type = "bfloat16"

            kv_cfg = KVCacheConfig()
            kv_cfg.seq_size_per_block = 16
            kv_cfg.kernel_seq_size_per_block = 16

            prof_cfg = ProfilingDebugLoggingConfig()
            build_model_config(
                model_config=cfg,
                model_args=model_args,
                kv_cache_config=kv_cfg,
                profiling_debug_logging_config=prof_cfg,
                embedding_config=None,
            )
            Qwen4_5DSpark._apply_kv_cache_config(cfg, kv_cfg)
            Qwen4_5DSpark._post_build_model_config(cfg)

            self.assertIsInstance(cfg.quant_config, MXFp4QuarkQuantConfig)
            self.assertEqual(cfg.quant_algo.getWeightBits(), 4)
            self.assertEqual(cfg.quant_algo.getGroupSize(), 32)
            self.assertEqual(cfg.expert_weight_dtype, "fp4")
            self.assertEqual(cfg.dense_weight_dtype, "bf16")
            from rtp_llm.ops import DataType
            self.assertEqual(cfg.data_type, DataType.TYPE_BF16)
            self.assertEqual(cfg.attn_config.kv_cache_dtype, KvCacheDataType.BASE)
            self.assertEqual(cfg.moe_layer_index, [0, 1, 2, 3, 4])
            self.assertGreater(cfg.eval_model_weight_size(), 0.0)
            self.assertEqual(len(cfg.kv_cache_spec_descs), 5)
            self.assertEqual(cfg.kv_cache_spec_descs[0][0].cache_type, KVCacheSpecType.MHA)
            self.assertEqual(
                cfg.kv_cache_spec_descs[0][0].tp_transfer_mode,
                TpCacheTransferMode.MHA_HEADS,
            )

            from rtp_llm.utils.model_weight import merge_qkv_hf

            wb = Qwen4_5DSparkWeight(cfg)
            winfo = wb.get_weight_info()
            qkv_desc = next(
                w for w in winfo.layer_weights[0] if w.name == W.attn_qkv_w
            )
            self.assertIs(qkv_desc.process_fun, merge_qkv_hf)
            self.assertEqual(len(qkv_desc.weights), 3)
            exp_gate = next(
                w
                for w in winfo.layer_weights[0]
                if w.name == DFLASH2_W.moe_expert_gate_w.format(e=0)
            )
            self.assertIs(exp_gate.data_type, torch.uint8)
            exp_bias = next(
                w for w in winfo.layer_weights[0] if w.name == DFLASH2_W.moe_expert_bias
            )
            self.assertIs(exp_bias.data_type, torch.float32)


class TestDFlash2ModulesAndParity(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_grouped_conv_triton_vs_pytorch_parity_and_request_isolation(self):
        conv = DFlashGroupedConv(
            hidden_size=512,
            taps=2,
            group_size=16,
            block_size=8,
            identity_folded=True,
            dtype=torch.bfloat16,
            device=torch.device("cuda:0"),
        )
        with torch.no_grad():
            conv.base_kernel.normal_(0.0, 0.5)
            conv.kernel_projection.weight.normal_(0.0, 0.05)

        x = torch.randn((16, 512), dtype=torch.bfloat16, device="cuda:0")
        out_triton, coeff_triton = conv.prepare(x)
        coeff_all = conv.kernel_projection(x).reshape(16, 2, 2, 32)
        out_ref = grouped_conv_ref(
            x, coeff_all[:, 0], conv.base_kernel[0], 8, 32, 16, 2
        )
        self.assertLess((out_triton.float() - out_ref.float()).abs().max().item(), 2e-2)

        fin_triton = conv.finish(out_triton, coeff_triton)
        fin_ref = grouped_conv_ref(
            out_triton, coeff_all[:, 1], conv.base_kernel[1], 8, 32, 16, 2
        )
        self.assertLess((fin_triton.float() - fin_ref.float()).abs().max().item(), 2e-2)

        x_pert = x.clone()
        x_pert[7].add_(5.0)
        out_pert, _ = conv.prepare(x_pert)
        self.assertTrue(torch.equal(out_pert[8], out_triton[8]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_candidate_selector_triton_vs_ref_and_markov_alias(self):
        sel = CandidateSelector(
            hidden_size=512,
            vocab_size=2048,
            rank=64,
            top_k=16,
            dtype=torch.bfloat16,
            device=torch.device("cuda:0"),
        )
        with torch.no_grad():
            sel.hidden_projection.weight.normal_(0.0, 0.05)
            sel.predecessor_codebook.normal_(0.0, 0.1)
            sel.successor_codebook.normal_(0.0, 0.1)

        logits = torch.randn((2 * 8, 2048), dtype=torch.float32, device="cuda:0")
        h = torch.randn((2 * 8, 512), dtype=torch.bfloat16, device="cuda:0")
        anchors = torch.tensor([11, 22], dtype=torch.int32, device="cuda:0")

        tokens_tri, cand_ids, edge_scores = sel.select_candidates(logits, h, anchors)
        self.assertEqual(tuple(tokens_tri.shape), (2, 8))
        self.assertEqual(tuple(cand_ids.shape), (2, 8, 16))
        self.assertEqual(tuple(edge_scores.shape), (2, 8, 16, 16))

        tokens_ref = greedy_select_path_ref(cand_ids, edge_scores)
        self.assertTrue(torch.equal(tokens_tri, tokens_ref))


class TestQwen4_5DSpark5LayerExecution(unittest.TestCase):
    def _run_5layer_e2e(self, device: str) -> None:
        cfg, mw = _load_dspark_model_weights(MOCK_8L_DFLASH, device=device)
        model = Qwen4_5DSparkModel(
            cfg, ParallelismConfig(), mw, max_generate_batch_size=4
        )
        self.assertEqual(model.cuda_graph_input_hidden_size(), 2560)
        self.assertEqual(model._dspark_query_width, 8)
        self.assertEqual(len(model.layers), 5)
        self.assertEqual(
            model.candidate_selector.predecessor_codebook.data_ptr(),
            mw.get_global_weight(W.dspark_markov_w1).data_ptr(),
        )
        self.assertEqual(
            model.candidate_selector.successor_codebook.data_ptr(),
            mw.get_global_weight(W.dspark_markov_w2).data_ptr(),
        )

        page_size = 16
        kv_bufs = [
            torch.zeros(
                (8, 2, 1, page_size, 256), dtype=torch.bfloat16, device=device
            )
            for _ in range(5)
        ]
        model.kv_cache = _MultiLayerDictKVCache(
            [LayerKVCache(kv_bufs[i], page_size, i, "default") for i in range(5)]
        )

        bt = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        layer_post_next_hc_mix = [
            torch.randn((5, 512), dtype=torch.bfloat16, device=device) * 0.05
            for _ in range(8)
        ]
        aux_h = extract_qwen4_5_aux_hidden_states(
            layer_post_next_hc_mix,
            cfg.dspark_target_layer_ids,
            source="post_next_hc_mix",
        )
        self.assertEqual(tuple(aux_h.shape), (5, 2560))

        commit_inputs = _make_dflash2_inputs(
            batch_size=2,
            prefix_lens=[0, 0],
            query_width=8,
            anchor_ids=[15, 25],
            block_table=bt,
            noise_token_id=cfg.dspark_noise_token_id,
            input_hiddens=aux_h,
            device=device,
            use_device_mirrors=device.startswith("cuda"),
            is_target_verify=True,
            active_query_lens=[3, 2],
        )
        commit_out = model.forward_commit(commit_inputs)
        self.assertEqual(tuple(commit_out.hidden_states.shape), (5, 512))
        for i in range(5):
            self.assertGreater(kv_bufs[i][0, 1, 0, :3, :].abs().sum().item(), 0.0)
            self.assertGreater(kv_bufs[i][2, 1, 0, :2, :].abs().sum().item(), 0.0)
            self.assertEqual(kv_bufs[i][1].abs().sum().item(), 0.0)

        kv_snapshot = [b.clone() for b in kv_bufs]
        propose_inputs = _make_dflash2_inputs(
            batch_size=2,
            prefix_lens=[3, 2],
            query_width=8,
            anchor_ids=[15, 25],
            block_table=bt,
            noise_token_id=cfg.dspark_noise_token_id,
            device=device,
            use_device_mirrors=device.startswith("cuda"),
            is_target_verify=False,
        )

        fmha_impl = model.prepare_fmha_impl(propose_inputs)
        if device.startswith("cuda"):
            self.assertTrue(fmha_impl.use_triton_paged)
            self.assertTrue(fmha_impl.support_cuda_graph())

        out = model.forward_propose(propose_inputs, fmha_impl=fmha_impl).hidden_states
        self.assertEqual(tuple(out.shape), (16, 512))
        self.assertTrue(torch.isfinite(out).all().item())

        anchors = torch.tensor([15, 25], dtype=torch.int32, device=device)
        draft_toks, cand_ids, edge_scores = model.propose_with_candidate_selector(
            out, anchors
        )
        self.assertEqual(tuple(draft_toks.shape), (2, 8))
        self.assertEqual(tuple(cand_ids.shape), (2, 8, 16))
        self.assertEqual(tuple(edge_scores.shape), (2, 8, 16, 16))

        if device.startswith("cuda"):
            # Break near-ties in random mock MoE routers so 1-ULP BF16 differences
            # between FlashInfer and PyTorch SDPA do not flip discrete top-k experts.
            saved_biases = [lyr.mlp.expert_bias.detach().clone() for lyr in model.layers]
            saved_gates = [lyr.mlp.gate.weight.detach().clone() for lyr in model.layers]
            try:
                for lyr in model.layers:
                    lyr.mlp.gate.weight.data.zero_()
                    num_e = lyr.mlp.expert_bias.numel()
                    lyr.mlp.expert_bias.data.copy_(
                        torch.linspace(8.0, -8.0, num_e, device=device)
                    )
                for i in range(5):
                    kv_bufs[i].copy_(kv_snapshot[i])
                out_fi = model.forward_propose(
                    propose_inputs, fmha_impl=fmha_impl
                ).hidden_states
                for i in range(5):
                    kv_bufs[i].copy_(kv_snapshot[i])
                sdpa_impl = _DFlash2PagedFMHADispatcher(
                    cfg.attn_config, propose_inputs.attention_inputs, use_triton_paged=False
                )
                out_ref = model.forward_propose(
                    propose_inputs, fmha_impl=sdpa_impl
                ).hidden_states
                cos = (
                    torch.nn.functional.cosine_similarity(
                        out_fi.float(), out_ref.float(), dim=-1
                    )
                    .min()
                    .item()
                )
                self.assertGreater(cos, 0.995)
            finally:
                for lyr, bias, gw in zip(model.layers, saved_biases, saved_gates):
                    lyr.mlp.expert_bias.data.copy_(bias)
                    lyr.mlp.gate.weight.data.copy_(gw)

        for i in range(5):
            kv_bufs[i].copy_(kv_snapshot[i])
        saved_mask_emb = model.embed_weight[cfg.dspark_noise_token_id].clone()
        try:
            model.embed_weight[cfg.dspark_noise_token_id].add_(2.0)
            out_pert = model.forward_propose(
                propose_inputs, fmha_impl=fmha_impl
            ).hidden_states
            self.assertGreater((out_pert[0] - out[0]).abs().max().item(), 1e-4)
        finally:
            model.embed_weight[cfg.dspark_noise_token_id].copy_(saved_mask_emb)

    def test_5layer_e2e_cpu(self):
        self._run_5layer_e2e("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_5layer_e2e_gpu(self):
        self._run_5layer_e2e("cuda:0")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gpu_cuda_graph_kv_scatter_and_grouped_conv(self):
        page_size = 16
        g_k = torch.randn((4, 1, 256), dtype=torch.bfloat16, device="cuda:0")
        g_v = torch.randn((4, 1, 256), dtype=torch.bfloat16, device="cuda:0")
        g_req = torch.tensor([0, 1, -1, 0], dtype=torch.int32, device="cuda:0")
        g_pos = torch.tensor([4, 5, 6, -1], dtype=torch.int32, device="cuda:0")
        g_bt = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cuda:0")
        g_kv = torch.zeros((6, 2, 1, page_size, 256), dtype=torch.bfloat16, device="cuda:0")

        conv = DFlashGroupedConv(
            512, taps=2, group_size=16, block_size=8, device=torch.device("cuda:0")
        )
        g_x = torch.randn((16, 512), dtype=torch.bfloat16, device="cuda:0")

        scatter_kv_to_paged_cache(g_k, g_v, g_req, g_pos, g_bt, g_kv, page_size)
        y_warm, coeff_warm = conv.prepare(g_x)
        _ = conv.finish(y_warm, coeff_warm)
        g_kv.zero_()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            scatter_kv_to_paged_cache(g_k, g_v, g_req, g_pos, g_bt, g_kv, page_size)
            y_cg, coeff_cg = conv.prepare(g_x)
            y_fin = conv.finish(y_cg, coeff_cg)
        graph.replay()
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(g_kv[0, 0, :, 4, :], g_k[0]))
        self.assertTrue(torch.equal(g_kv[2, 1, :, 5, :], g_v[1]))
        self.assertEqual(g_kv[0, :, :, 6, :].abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(y_fin).all().item())


class TestQwen4_5OpaqueStateVerifyRollback(unittest.TestCase):
    def _run_step_rollback_test(self, device: str) -> None:
        num_blocks = 8
        init_state = _build_dummy_target_opaque_state(num_blocks, device)
        active_blocks = torch.tensor([1, 3, 6], dtype=torch.int64, device=device)
        for buf in init_state.values():
            if buf.dtype.is_floating_point:
                buf[active_blocks] = torch.randn_like(buf[active_blocks]) * 0.1
            else:
                buf[active_blocks] = 7

        tokens_per_step = torch.arange(10, 34, device=device).reshape(3, 8)
        accepted_lens = torch.tensor([1, 4, 8], dtype=torch.int32, device=device)
        query_lens = torch.tensor([8, 8, 8], dtype=torch.int32, device=device)

        gt_state = {k: v.clone() for k, v in init_state.items()}
        for s in range(8):
            active_b = accepted_lens > s
            _step_opaque_state_deterministic(
                gt_state, active_blocks, tokens_per_step[:, s], active_mask=active_b
            )

        spec_state = {k: v.clone() for k, v in init_state.items()}
        tx = Qwen4_5OpaqueStateTransaction.from_container(spec_state)
        self.assertEqual(
            sorted(h.key for h in tx.pools), sorted(QWEN4_5_OPAQUE_STATE_TAGS)
        )
        tx.begin_verify(active_blocks)
        for s in range(8):
            _step_opaque_state_deterministic(
                spec_state, active_blocks, tokens_per_step[:, s]
            )
            tx.capture_step(s, active_blocks)

        rollback_qwen4_5_opaque_states(
            tx,
            accepted_lens=accepted_lens,
            query_lens=query_lens,
            final_block_ids=active_blocks,
        )
        for tag in QWEN4_5_OPAQUE_STATE_TAGS:
            self.assertTrue(
                torch.equal(spec_state[tag], gt_state[tag]),
                msg=f"Mismatch after partial-acceptance rollback on {tag} ({device})",
            )

    def test_step_indexed_rollback_cpu(self):
        self._run_step_rollback_test("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_step_indexed_rollback_gpu(self):
        self._run_step_rollback_test("cuda:0")

    def test_pre_verify_snapshot_restore_and_qsa_indexer_lengths(self):
        state = _build_dummy_target_opaque_state(4, "cpu")
        blocks = torch.tensor([0, 2], dtype=torch.int64)
        state["gdn_ssm_state"][blocks] = 3.5
        state["indexer_state"][blocks] = 1.25

        tx = snapshot_qwen4_5_opaque_states(state, blocks)
        state["gdn_ssm_state"][blocks] = -9.0
        state["indexer_state"][blocks] = -4.0

        rollback_qwen4_5_opaque_states(
            tx, batch_mask=torch.tensor([True, False], dtype=torch.bool)
        )
        self.assertTrue(torch.all(state["gdn_ssm_state"][0] == 3.5).item())
        self.assertTrue(torch.all(state["indexer_state"][0] == 1.25).item())
        self.assertTrue(torch.all(state["gdn_ssm_state"][2] == -9.0).item())

        comp_kv, tail_ring = compute_qsa_indexer_state_lengths(
            prefix_lens=torch.tensor([13, 16], dtype=torch.int32),
            accepted_lens=torch.tensor([1, 4], dtype=torch.int32),
            compress_ratio=4,
        )
        self.assertTrue(torch.equal(comp_kv, torch.tensor([3, 5], dtype=torch.int64)))
        self.assertTrue(torch.equal(tail_ring, torch.tensor([2, 0], dtype=torch.int64)))

    def test_kv_cache_spec_and_single_file_alias_contract(self):
        from safetensors.torch import save_file
        from rtp_llm.models.qwen4_5.qwen4_5_dspark import Qwen4_5DSpark
        from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import FULL_ATTN_KV_TAG
        from rtp_llm.ops import KVCacheConfig, KVCacheSpecType, TpCacheTransferMode

        cfg = Qwen4_5DSpark._create_config_from_json(
            "",
            {
                "architectures": ["DFlashQwenDraftModel"],
                "model_type": "qwen4_5_dspark",
                "num_hidden_layers": 5,
                "hidden_size": 512,
                "vocab_size": 1024,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "partial_rotary_factor": 0.25,
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "routed_expert_hidden_size": 256,
                "shared_expert_intermediate_size": 128,
                "hc_count": 4,
                "hc_lowrank": 32,
                "dflash_config": {
                    "target_layer_ids": [0, 1],
                    "mask_token_id": 1000,
                    "block_size": 8,
                },
            },
        )
        kv_cfg = KVCacheConfig()
        kv_cfg.seq_size_per_block = 32
        Qwen4_5DSpark._apply_kv_cache_config(cfg, kv_cfg)
        self.assertFalse(cfg.attn_config.is_causal)
        self.assertEqual(cfg.attn_config.tokens_per_block, 32)
        self.assertEqual(cfg.attn_config.kernel_tokens_per_block, 32)
        self.assertEqual(len(cfg.kv_cache_spec_descs), 5)
        for group in cfg.kv_cache_spec_descs:
            self.assertEqual(len(group), 1)
            self.assertEqual(group[0].tag, FULL_ATTN_KV_TAG)
            self.assertEqual(group[0].cache_type, KVCacheSpecType.MHA)
            self.assertEqual(group[0].tp_transfer_mode, TpCacheTransferMode.MHA_HEADS)

        with tempfile.TemporaryDirectory() as tmpdir:
            target = types.SimpleNamespace(
                model_config=types.SimpleNamespace(vocab_size=1024, hidden_size=512)
            )
            save_file({"fc.weight": torch.zeros((512, 1024))}, os.path.join(tmpdir, "model.safetensors"))
            cfg.ckpt_path = tmpdir
            self.assertEqual(
                Qwen4_5DSpark.speculative_weight_alias_names(target, cfg),
                (W.embedding, W.lm_head),
            )
            save_file(
                {
                    "fc.weight": torch.zeros((512, 1024)),
                    "model.embed_tokens.weight": torch.zeros((1024, 512)),
                    "lm_head.weight": torch.zeros((1024, 512)),
                },
                os.path.join(tmpdir, "model.safetensors"),
            )
            self.assertEqual(
                Qwen4_5DSpark.speculative_weight_alias_names(target, cfg),
                (),
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_5layer_dflash2_cuda_graph_capture_and_replay_gpu(self):
        from unittest import mock

        device = "cuda:0"
        cfg, mw = _load_dspark_model_weights(MOCK_8L_DFLASH, device=device)
        model = Qwen4_5DSparkModel(
            cfg, ParallelismConfig(), mw, max_generate_batch_size=4
        )
        # Keep routing deterministic and avoid top-k ties under BF16.
        for lyr in model.layers:
            num_e = lyr.mlp.expert_bias.numel()
            lyr.mlp.expert_bias.data.copy_(torch.linspace(0.5, -0.5, num_e, device=device))

        page_size = 16
        # Intentionally pass 2D raw hybrid-packed KV buffers to match C++ KVCacheManager.
        packed_stride = 2 * 1 * page_size * 256 + 64
        kv_bufs = [
            torch.zeros((8, packed_stride), dtype=torch.bfloat16, device=device)
            for _ in range(5)
        ]

        class _TaggedKVCache:
            def get_kernel_seq_size_per_block(self, tag: str = "full"):
                if tag != "full":
                    raise KeyError(tag)
                return page_size

            def get_layer_cache(self, layer_idx: int):
                return LayerKVCache(kv_bufs[layer_idx], page_size, layer_idx, "full_attn_kv")

            def get_layer_cache_groups(self, layer_idx: int):
                return [LayerKVCache(kv_bufs[layer_idx], page_size, layer_idx, "full_attn_kv")]

        model.kv_cache = _TaggedKVCache()

        # 1. Verify commit_feature_rows triggers CacheStore writer when present.
        bt = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device=device)
        commit_inputs = _make_dflash2_inputs(
            batch_size=2,
            prefix_lens=[0, 0],
            query_width=8,
            anchor_ids=[11, 22],
            block_table=bt,
            noise_token_id=cfg.dspark_noise_token_id,
            device=device,
            use_device_mirrors=True,
            is_target_verify=False,
            active_query_lens=[3, 2],
        )
        commit_inputs.input_ids = torch.tensor([11, 12, 13, 22, 23], dtype=torch.int32, device=device)
        commit_inputs.input_hiddens = torch.randn((5, 2560), dtype=torch.bfloat16, device=device) * 0.05

        writer_calls = []
        with mock.patch(
            "rtp_llm.models_py.model_desc.qwen4_5_dspark_model.create_write_cache_store_impl",
            return_value=lambda cache: writer_calls.append(cache),
        ):
            model.forward_commit(commit_inputs)
        self.assertEqual(len(writer_calls), 5)

        # 2. Capture PROPOSE CUDA graph with tagged attention inputs & replay with changed buffers.
        propose_inputs = _make_dflash2_inputs(
            batch_size=2,
            prefix_lens=[3, 2],
            query_width=8,
            anchor_ids=[15, 25],
            block_table=bt,
            noise_token_id=cfg.dspark_noise_token_id,
            device=device,
            use_device_mirrors=True,
            is_target_verify=False,
        )
        tagged_propose = types.SimpleNamespace(
            input_ids=propose_inputs.input_ids,
            input_hiddens=propose_inputs.input_hiddens,
            attention_inputs=propose_inputs.attention_inputs,
            attention_inputs_by_tag={"full_attn_kv": propose_inputs.attention_inputs},
        )
        fmha_impl = model.prepare_fmha_impl(tagged_propose, is_cuda_graph=True)
        self.assertTrue(fmha_impl.support_cuda_graph())

        kv_snap = [b.clone() for b in kv_bufs]
        # Warmup eager before capture.
        warmup_out = model.forward_propose(tagged_propose, fmha_impl=fmha_impl).hidden_states.clone()
        for b, s in zip(kv_bufs, kv_snap):
            b.copy_(s)

        graph = torch.cuda.CUDAGraph()
        fmha_impl.prepare_cuda_graph(tagged_propose.attention_inputs_by_tag)
        with torch.cuda.graph(graph):
            cg_out = model.forward_propose(tagged_propose, fmha_impl=fmha_impl).hidden_states
        torch.cuda.synchronize()

        # Mutate in-place for replay: deactivate request 1 (padding row) and shift anchor/prefix.
        for b, s in zip(kv_bufs, kv_snap):
            b.copy_(s)
        noise = int(cfg.dspark_noise_token_id)
        tagged_propose.input_ids.copy_(
            torch.tensor([77] + [noise] * 7 + [88] + [noise] * 7, dtype=torch.int32, device=device)
        )
        propose_inputs.attention_inputs.prefix_lengths.copy_(
            torch.tensor([2, 0], dtype=torch.int32, device=device)
        )
        fmha_impl.prepare_cuda_graph(tagged_propose.attention_inputs_by_tag)
        graph.replay()
        torch.cuda.synchronize()
        replay_out = cg_out.clone()

        for b, s in zip(kv_bufs, kv_snap):
            b.copy_(s)
        eager_out = model.forward_propose(tagged_propose, fmha_impl=fmha_impl).hidden_states
        self.assertLess((replay_out - eager_out).abs().max().item(), 1e-3)
        self.assertGreater((replay_out[:8] - warmup_out[:8]).abs().max().item(), 1e-3)



class TestQwen4_5TargetVerifyRuntime(unittest.TestCase):
    @staticmethod
    def _build_8l_runtime(device: str, *, with_qsa: bool, capture_layers: Sequence[int]):
        from rtp_llm.config.qwen4_5_config import Qwen4_5ModelConfig
        from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import build_qwen4_5_kv_cache_plan
        from rtp_llm.models_py.model_desc.qwen4_5 import Qwen4_5ForCausalLM
        from rtp_llm.models_py.model_desc.qwen4_5_runtime import (
            LocalKVCache,
            Qwen4_5RTPModel,
        )
        from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig

        arch = Qwen4_5ArchConfig(
            vocab_size=256,
            hidden_size=512,
            num_hidden_layers=8,
            intermediate_size=1024,
            first_k_dense_replace=1,
            rms_norm_eps=1e-6,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            max_position_embeddings=512,
            full_attention_interval=4,
            num_attention_heads=32,
            num_key_value_heads=1,
            head_dim=256,
            attn_output_gate=True,
            partial_rotary_factor=0.25,
            rope_theta=10000.0,
            use_full_attn_kv_sconv=True,
            full_attn_kv_sconv_residual=True,
            full_attn_kv_sconv_use_activation=False,
            linear_conv_kernel_dim=4,
            linear_num_key_heads=4,
            linear_key_head_dim=64,
            linear_num_value_heads=16,
            linear_value_head_dim=64,
            gdn_conv_residual=False,
            gdn_no_q_conv=False,
            norm_before_gate=True,
            output_gate_type="sigmoid",
            use_linear_attn_output_sconv=True,
            use_full_attn_output_sconv=True,
            use_ffn_output_sconv=True,
            output_sconv_use_activation=False,
            hc_count=4,
            hc_lowrank=32,
            hc_prev_mix=True,
            hc_per_branch_norm=True,
            hc_rmsnorm_bias=True,
            num_experts=16,
            num_experts_per_tok=4,
            routed_expert_hidden_size=256,
            moe_intermediate_size=256,
            shared_expert_intermediate_size=128,
            shared_expert_gate=True,
            router_precision="fp32_logits",
            router_score_function="sigmoid",
            moe_router_pre_softmax=True,
            router_normalize_topk_scores=True,
            moe_router_enable_expert_bias=True,
            hidden_act="situ_glu",
            hidden_act_gate_beta=4.0,
            hidden_act_up_beta=25.0,
            clip_mlp_output=1000.0,
            ple_layer_ids=(2,),
            ple_embed_dim=256,
            ple_conv_kernel_size=4,
            ngram_size=3,
            heads_per_ngram=4,
            ngram_vocab_size_base=256,
            make_ngram_vocab_size_divisible_by=64,
            split_ngram_parts=4,
            ple_quant_scheme="bf16",
            share_map="4->8,5->6,5->7",
            enable_yoco_decode=True,
            quant_method="mxfp4",
            indexer_head_dim=128 if with_qsa else 0,
            indexer_n_heads=8 if with_qsa else 0,
            indexer_kv_heads=1 if with_qsa else 0,
            indexer_compress_ratio=4,
            indexer_budget=2048 if with_qsa else 0,
        )
        torch.manual_seed(17)
        causal_lm = Qwen4_5ForCausalLM(
            arch,
            use_triton_mxfp4=True,
            compute_mode="bf16",
            dtype=torch.bfloat16,
            device=torch.device(device),
        )
        causal_lm.init_synthetic_weights(seed=17, ple_mode="gpu_full", ple_fp8=False)
        with torch.no_grad():
            for layer in causal_lm.model.layers:
                if layer.is_moe and layer.mlp.expert_bias is not None:
                    layer.mlp.expert_bias.zero_()
                    shard = layer.mlp.num_experts // 4
                    for r in range(4):
                        layer.mlp.expert_bias[r * shard + (layer.layer_idx % shard)] = 4.0
        causal_lm.prepare_weights()
        plan = build_qwen4_5_kv_cache_plan(arch)
        kv_cache = LocalKVCache(
            plan,
            tokens_per_block=16,
            device=torch.device(device),
            dtype=torch.bfloat16,
        )
        mcfg = Qwen4_5ModelConfig(arch_config=arch)
        mcfg.capture_aux_hidden_layer_ids = list(capture_layers)
        rtp_model = Qwen4_5RTPModel.from_causal_lm(
            causal_lm, model_config=mcfg, kv_cache=kv_cache
        )
        return arch, kv_cache, rtp_model

    @staticmethod
    def _clone_kv_snapshot(kv_cache) -> Dict[Tuple[int, str], torch.Tensor]:
        snap: Dict[Tuple[int, str], torch.Tensor] = {}
        for lid, tag_map in enumerate(kv_cache.layer_caches):
            for tag, lc in tag_map.items():
                snap[(lid, tag)] = lc.kv_cache_base.detach().clone()
        return snap

    @staticmethod
    def _restore_kv_snapshot(kv_cache, snap: Dict[Tuple[int, str], torch.Tensor]) -> None:
        for lid, tag_map in enumerate(kv_cache.layer_caches):
            for tag, lc in tag_map.items():
                lc.kv_cache_base.copy_(snap[(lid, tag)])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_target_verify_qsa_block_boundary_rollback_and_aux_gpu(self):
        from rtp_llm.models_py.model_desc.qwen4_5_runtime import (
            LocalPyModelInputs,
            build_block_tables_for_requests,
            make_tagged_attention_inputs,
        )

        device = torch.device("cuda:0")
        cap_layers = [0, 1, 2, 3, 4]
        for with_qsa in (True, False):
            arch, kv_cache, rt = self._build_8l_runtime(
                "cuda:0", with_qsa=with_qsa, capture_layers=cap_layers
            )
            self.assertTrue(rt.has_mtp_hidden_buffer())
            self.assertTrue(rt.requires_speculative_target_commit())

            # Prefill two requests where req0 sits near page boundary (13, tpb=16)
            # so a verify width of 6 crosses boundary 15->16.
            p0 = torch.randint(5, 250, (13,), dtype=torch.int32, device=device)
            p1 = torch.randint(5, 250, (6,), dtype=torch.int32, device=device)
            tbls = build_block_tables_for_requests(kv_cache, [13 + 6, 6 + 6])
            prefill_attn = make_tagged_attention_inputs(
                tbls,
                is_prefill=True,
                input_lengths=[13, 6],
                sequence_lengths=[13, 6],
                prefix_lengths=[0, 0],
                device=device,
            )
            prefill_in = LocalPyModelInputs(
                input_ids=torch.cat([p0, p1], dim=0),
                attention_inputs=prefill_attn,
            )
            pref_out = rt.forward(prefill_in)
            self.assertEqual(
                tuple(pref_out.mtp_target_hidden_states.shape), (19, 5 * arch.hidden_size)
            )
            self.assertIs(rt.get_mtp_target_hidden_states(19), pref_out.mtp_target_hidden_states)

            pre_verify_snap = self._clone_kv_snapshot(kv_cache)

            # Verify width V=6 -> tokens [B, 6]
            v_toks = torch.randint(5, 250, (2, 6), dtype=torch.int32, device=device)
            verify_attn = make_tagged_attention_inputs(
                tbls,
                is_prefill=True,
                input_lengths=[6, 6],
                sequence_lengths=[],
                prefix_lengths=[13, 6],
                device=device,
            )
            for grp in verify_attn.values():
                grp.is_target_verify = True
            verify_in = LocalPyModelInputs(
                input_ids=v_toks.reshape(-1),
                attention_inputs=verify_attn,
            )

            # 1) Test transactional rollback before commit / after tentative commit.
            ver_out = rt.forward(verify_in)
            self.assertEqual(tuple(ver_out.hidden_states.shape), (12, arch.hidden_size))
            self.assertEqual(
                tuple(ver_out.mtp_target_hidden_states.shape), (12, 5 * arch.hidden_size)
            )
            # Persistent OPAQUE_STATE pools must remain completely untouched before commit!
            for lid, tag_map in enumerate(kv_cache.layer_caches):
                for tag, lc in tag_map.items():
                    if tag not in ("full", "full_attn_kv", "indexer_kv"):
                        self.assertTrue(
                            torch.equal(lc.kv_cache_base, pre_verify_snap[(lid, tag)]),
                            f"Persistent {tag} at layer {lid} mutated during forward before commit!",
                        )

            # Prepare & tentative commit & rollback (finish(False)) -> must undo cleanly!
            acc = torch.tensor([5, 2], dtype=torch.int32, device=device)
            rt.prepare_speculative_target_commit(acc)
            rt.finish_speculative_target_commit(True)
            rt.finish_speculative_target_commit(False)
            rt.finalize_speculative_target_commit()
            for lid, tag_map in enumerate(kv_cache.layer_caches):
                for tag, lc in tag_map.items():
                    if tag not in ("full", "full_attn_kv", "indexer_kv"):
                        self.assertTrue(
                            torch.equal(lc.kv_cache_base, pre_verify_snap[(lid, tag)]),
                            f"Rollback failed to restore {tag} at layer {lid}!",
                        )

            # 2) Run full verify + accept([5, 2]) + commit(True), then compare against
            # sequential 1-by-1 decode from the exact same pre-verify snapshot!
            self._restore_kv_snapshot(kv_cache, pre_verify_snap)
            ver_out = rt.forward(verify_in)
            rt.prepare_speculative_target_commit(acc)
            rt.finish_speculative_target_commit(True)
            rt.finalize_speculative_target_commit()
            ver_after_commit_snap = self._clone_kv_snapshot(kv_cache)

            # Sequential reference on fresh pre-verify snapshot
            self._restore_kv_snapshot(kv_cache, pre_verify_snap)
            seq_hiddens_0 = []
            seq_hiddens_1 = []
            seq_aux_0 = []
            seq_aux_1 = []
            for step in range(6):
                dec_attn = make_tagged_attention_inputs(
                    tbls,
                    is_prefill=False,
                    input_lengths=[1, 1],
                    sequence_lengths=[13 + step, 6 + step],
                    prefix_lengths=[0, 0],
                    device=device,
                )
                dec_in = LocalPyModelInputs(
                    input_ids=v_toks[:, step].reshape(-1),
                    attention_inputs=dec_attn,
                )
                d_out = rt.forward(dec_in)
                seq_hiddens_0.append(d_out.hidden_states[0])
                seq_hiddens_1.append(d_out.hidden_states[1])
                seq_aux_0.append(d_out.mtp_target_hidden_states[0])
                seq_aux_1.append(d_out.mtp_target_hidden_states[1])

            ref_h = torch.stack(seq_hiddens_0 + seq_hiddens_1, dim=0)
            h_diff = (ver_out.hidden_states.float() - ref_h.float()).abs()
            h_cos = torch.nn.functional.cosine_similarity(
                ver_out.hidden_states.float(), ref_h.float(), dim=-1
            ).min().item()
            self.assertGreater(h_cos, 0.997)
            self.assertLess(h_diff.mean().item(), 0.025)
            self.assertLess(h_diff.max().item(), 0.12)
            ref_aux = torch.stack(seq_aux_0 + seq_aux_1, dim=0)
            a_diff = (ver_out.mtp_target_hidden_states.float() - ref_aux.float()).abs()
            a_cos = torch.nn.functional.cosine_similarity(
                ver_out.mtp_target_hidden_states.float(), ref_aux.float(), dim=-1
            ).min().item()
            self.assertGreater(a_cos, 0.997)
            self.assertLess(a_diff.mean().item(), 0.025)
            self.assertLess(a_diff.max().item(), 0.12)

            # 3) Continue 1 more accepted decode step from committed state [13+5=18, 6+2=8]
            # and verify exact continuation match against an independent sequential prefix+accept!
            self._restore_kv_snapshot(kv_cache, ver_after_commit_snap)
            next_toks = torch.tensor([42, 73], dtype=torch.int32, device=device)
            next_attn_a = make_tagged_attention_inputs(
                tbls,
                is_prefill=False,
                input_lengths=[1, 1],
                sequence_lengths=[18, 8],
                prefix_lengths=[0, 0],
                device=device,
            )
            cont_verify_out = rt.forward(
                LocalPyModelInputs(input_ids=next_toks, attention_inputs=next_attn_a)
            ).hidden_states

            # Fresh sequential replay up to exact accepted prefix lengths (5 for r0, 2 for r1)
            self._restore_kv_snapshot(kv_cache, pre_verify_snap)
            for s in range(5):
                # step r0 only up to s<5, r1 only up to s<2 using single-row sequential decode
                if s < 2:
                    da = make_tagged_attention_inputs(
                        tbls,
                        is_prefill=False,
                        input_lengths=[1, 1],
                        sequence_lengths=[13 + s, 6 + s],
                        prefix_lengths=[0, 0],
                        device=device,
                    )
                    rt.forward(LocalPyModelInputs(input_ids=v_toks[:, s], attention_inputs=da))
                else:
                    # Only advance r0 at index 0 cleanly; keep r1 at pos 8 by replaying on req0 table
                    pass
            # Verify r1 state at accepted length=2 independently using 1-request table!
            r1_tbls = {tag: [rows[1]] for tag, rows in tbls.items()}
            # Restore pre-verify and advance r0 individually 5 steps, then step 18
            self._restore_kv_snapshot(kv_cache, pre_verify_snap)
            r0_tbls = {tag: [rows[0]] for tag, rows in tbls.items()}
            for s in range(5):
                da0 = make_tagged_attention_inputs(
                    r0_tbls,
                    is_prefill=False,
                    input_lengths=[1],
                    sequence_lengths=[13 + s],
                    prefix_lengths=[0],
                    device=device,
                )
                rt.forward(LocalPyModelInputs(input_ids=v_toks[0 : 1, s], attention_inputs=da0))
            da0_fin = make_tagged_attention_inputs(
                r0_tbls,
                is_prefill=False,
                input_lengths=[1],
                sequence_lengths=[18],
                prefix_lengths=[0],
                device=device,
            )
            seq_r0_next = rt.forward(
                LocalPyModelInputs(input_ids=next_toks[0:1], attention_inputs=da0_fin)
            ).hidden_states
            r0_diff = (cont_verify_out[0].float() - seq_r0_next[0].float()).abs()
            self.assertGreater(
                torch.nn.functional.cosine_similarity(
                    cont_verify_out[0].float().unsqueeze(0),
                    seq_r0_next[0].float().unsqueeze(0),
                    dim=-1,
                ).item(),
                0.997,
            )
            self.assertLess(r0_diff.max().item(), 0.12)

            self._restore_kv_snapshot(kv_cache, pre_verify_snap)
            for s in range(2):
                da1 = make_tagged_attention_inputs(
                    r1_tbls,
                    is_prefill=False,
                    input_lengths=[1],
                    sequence_lengths=[6 + s],
                    prefix_lengths=[0],
                    device=device,
                )
                rt.forward(LocalPyModelInputs(input_ids=v_toks[1 : 2, s], attention_inputs=da1))
            da1_fin = make_tagged_attention_inputs(
                r1_tbls,
                is_prefill=False,
                input_lengths=[1],
                sequence_lengths=[8],
                prefix_lengths=[0],
                device=device,
            )
            seq_r1_next = rt.forward(
                LocalPyModelInputs(input_ids=next_toks[1:2], attention_inputs=da1_fin)
            ).hidden_states
            r1_diff = (cont_verify_out[1].float() - seq_r1_next[0].float()).abs()
            self.assertGreater(
                torch.nn.functional.cosine_similarity(
                    cont_verify_out[1].float().unsqueeze(0),
                    seq_r1_next[0].float().unsqueeze(0),
                    dim=-1,
                ).item(),
                0.997,
            )
            self.assertLess(r1_diff.max().item(), 0.12)

            # 4) Verify non-speculative on_cuda_graph_replay does not fail.
            rt.on_cuda_graph_replay(
                1,
                LocalPyModelInputs(
                    input_ids=next_toks[0:1], attention_inputs=da0_fin
                ),
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_target_verify_cuda_graph_capture_and_replay_gpu(self):
        from rtp_llm.models_py.model_desc.qwen4_5_runtime import (
            LocalPyModelInputs,
            build_block_tables_for_requests,
            make_tagged_attention_inputs,
        )

        device = torch.device("cuda:0")
        arch, kv_cache, rt = self._build_8l_runtime(
            "cuda:0", with_qsa=True, capture_layers=[0, 2, 4]
        )
        tbls = build_block_tables_for_requests(kv_cache, [20, 20])
        snap = self._clone_kv_snapshot(kv_cache)

        def _make_cg_verify_inputs(toks_2d: torch.Tensor, pfx_list: List[int], *, zero_second: bool = False):
            tbls_used = {
                tag: [list(rows[0]), [0] * len(rows[1])] if zero_second else [list(rows[0]), list(rows[1])]
                for tag, rows in tbls.items()
            }
            attn = make_tagged_attention_inputs(
                tbls_used,
                is_prefill=True,
                input_lengths=[4, 4],
                sequence_lengths=[],
                prefix_lengths=pfx_list,
                device=device,
            )
            for grp in attn.values():
                grp.is_target_verify = True
                grp.prefix_lengths = grp.prefix_lengths.to(device=device, dtype=torch.int32)
            return LocalPyModelInputs(input_ids=toks_2d.reshape(-1), attention_inputs=attn)

        t_a = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32, device=device)
        vin = _make_cg_verify_inputs(t_a, [5, 7])
        fmha = rt.prepare_fmha_impl(vin, is_cuda_graph=True)
        self.assertTrue(fmha.support_cuda_graph())

        # Warmup eager + callback
        _ = rt.forward(vin, fmha_impl=fmha)
        rt.on_cuda_graph_warmup(vin)
        self._restore_kv_snapshot(kv_cache, snap)

        graph = torch.cuda.CUDAGraph()
        fmha.prepare_cuda_graph(vin.attention_inputs)
        with torch.cuda.graph(graph):
            cg_res = rt.forward(vin, fmha_impl=fmha)
        torch.cuda.synchronize()
        rt.on_cuda_graph_capture(2, vin)

        # Replay with modified tokens, modified prefix lengths, and dummy row 1 (padded batch!).
        self._restore_kv_snapshot(kv_cache, snap)
        t_b = torch.tensor([[30, 31, 32, 33], [99, 99, 99, 99]], dtype=torch.int32, device=device)
        vin_rep = _make_cg_verify_inputs(t_b, [6, 0], zero_second=True)
        vin.input_ids.copy_(vin_rep.input_ids)
        for tag, grp in vin.attention_inputs.items():
            rep_grp = vin_rep.attention_inputs[tag]
            grp.prefix_lengths.copy_(rep_grp.prefix_lengths)
            grp.kv_cache_block_id_device.copy_(rep_grp.kv_cache_block_id_device)
            grp.kv_cache_kernel_block_id_device.copy_(rep_grp.kv_cache_kernel_block_id_device)
        fmha.prepare_cuda_graph(vin.attention_inputs)
        graph.replay()
        torch.cuda.synchronize()
        rt.on_cuda_graph_replay(2, vin)
        cg_hidden = cg_res.hidden_states.clone()
        cg_aux = cg_res.mtp_target_hidden_states.clone()

        rt.prepare_speculative_target_commit(torch.tensor([3, 0], dtype=torch.int32, device=device))
        rt.finish_speculative_target_commit(True)
        rt.finalize_speculative_target_commit()
        cg_committed_snap = self._clone_kv_snapshot(kv_cache)

        # Compare against fresh eager verify + commit
        self._restore_kv_snapshot(kv_cache, snap)
        eager_res = rt.forward(vin_rep)
        rt.prepare_speculative_target_commit(torch.tensor([3, 0], dtype=torch.int32, device=device))
        rt.finish_speculative_target_commit(True)
        rt.finalize_speculative_target_commit()

        self.assertLess((cg_hidden[:4] - eager_res.hidden_states[:4]).abs().max().item(), 1e-3)
        self.assertLess((cg_aux[:4] - eager_res.mtp_target_hidden_states[:4]).abs().max().item(), 1e-3)
        for lid, tag_map in enumerate(kv_cache.layer_caches):
            for tag, lc in tag_map.items():
                if tag not in ("full", "full_attn_kv", "indexer_kv"):
                    self.assertTrue(
                        torch.equal(lc.kv_cache_base, cg_committed_snap[(lid, tag)]),
                        f"Graph commit diverged from eager commit for {tag} layer {lid}!",
                    )

if __name__ == "__main__":
    unittest.main(verbosity=2)
