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

    def test_ep4_sharded_weights_and_zero_arg_get_weight_info(self):
        import inspect
        from rtp_llm.models.qwen4_5.qwen4_5_dspark import ModelDeployWeightInfo as _MDWI

        cfg = Qwen4_5DSpark._create_config(MOCK_8L_DFLASH)
        with open(
            os.path.join(MOCK_8L_DFLASH, "model.safetensors.index.json"),
            "r",
            encoding="utf-8",
        ) as f:
            weight_map = json.load(f)["weight_map"]

        pc_ep4 = ParallelismConfig()
        pc_ep4.tp_size = 4
        pc_ep4.tp_rank = 1
        pc_ep4.ep_size = 4
        pc_ep4.ep_rank = 1
        wb = Qwen4_5DSparkWeight(cfg, parallelism_config=pc_ep4)
        wb._process_meta({}, list(weight_map.keys()))
        winfo = wb._get_weight_info()
        layer0_names = {w.name for w in winfo.layer_weights[0]}
        self.assertIn(DFLASH2_W.moe_expert_gate_w.format(e=2), layer0_names)
        self.assertIn(DFLASH2_W.moe_expert_down_s.format(e=3), layer0_names)
        self.assertNotIn(DFLASH2_W.moe_expert_gate_w.format(e=0), layer0_names)
        self.assertNotIn(DFLASH2_W.moe_expert_gate_w.format(e=4), layer0_names)

        # Verify zero-arg container ModelDeployWeightInfo compatibility shim
        orig_gwi = _MDWI.get_weight_info
        try:
            _MDWI.get_weight_info = lambda self_obj: self_obj._get_weight_info()
            compat_winfo = wb.get_weight_info(database=object())
            self.assertEqual(len(compat_winfo.layer_weights), cfg.num_layers)
        finally:
            _MDWI.get_weight_info = orig_gwi

        # Verify EP=4 partial summation matches EP=1 on Qwen4_5DSparkModel MoE
        _, mw_full = _load_dspark_model_weights(MOCK_8L_DFLASH, device="cpu")
        pc_ep1 = ParallelismConfig()
        m_full = Qwen4_5DSparkModel(cfg, pc_ep1, mw_full, max_generate_batch_size=2)
        spec = cfg._dflash2_spec
        local_e = spec.num_experts // 4
        x_in = torch.randn((3, spec.hidden_size), dtype=torch.bfloat16) * 0.25
        with torch.no_grad():
            ref_routed = m_full.layers[0].mlp.forward_routed_only(x_in) if hasattr(m_full.layers[0].mlp, "forward_routed_only") else None
            # Sum routed partial outputs across 4 EP ranks without dist process group
            import torch.nn.functional as F
            from rtp_llm.models_py.modules.qwen4_5.mxfp4_ops import mxfp4_moe_forward
            x_ln = m_full.layers[0].mlp.pre_moe_latent_layernorm(
                F.linear(x_in, m_full.layers[0].mlp.latent_down_proj.weight)
            )
            routed_in = x_ln * torch.sigmoid(
                F.linear(x_in, m_full.layers[0].mlp.latent_gate_proj.weight)
            )
            topk_ids, topk_w = m_full.layers[0].mlp.route(routed_in, use_triton=False)
            from rtp_llm.models_py.modules.qwen4_5.mxfp4_ops import mxfp4_moe_forward_ref
            ref_routed_out = mxfp4_moe_forward_ref(
                routed_in,
                topk_ids,
                topk_w,
                m_full.layers[0].mlp._mxfp4_weights(),
                gate_beta=m_full.layers[0].mlp.gate_beta,
                up_beta=m_full.layers[0].mlp.up_beta,
            )
            part_sum = torch.zeros_like(ref_routed_out, dtype=torch.float32)
            for r in range(4):
                pc_r = ParallelismConfig()
                pc_r.tp_size = 4
                pc_r.tp_rank = r
                pc_r.ep_size = 4
                pc_r.ep_rank = r
                mw_r = ModelWeights(cfg.num_layers, device="cpu", dtype=torch.bfloat16)
                for gw_name, gw_t in mw_full.global_weights.items():
                    mw_r.set_global_weight(gw_name, gw_t)
                e0, e1 = r * local_e, (r + 1) * local_e
                for lid in range(cfg.num_layers):
                    for k_name, v_t in mw_full.weights[lid].items():
                        if ".experts." in k_name:
                            eid = int(k_name.split(".experts.")[1].split(".")[0])
                            if e0 <= eid < e1:
                                mw_r.set_layer_weight(lid, k_name, v_t)
                        else:
                            mw_r.set_layer_weight(lid, k_name, v_t)
                m_r = Qwen4_5DSparkModel(cfg, pc_r, mw_r, max_generate_batch_size=2)
                mlp_r = m_r.layers[0].mlp
                self.assertEqual(mlp_r.ep_size, 4)
                self.assertEqual(mlp_r.ep_rank, r)
                self.assertEqual(mlp_r.local_num_experts, local_e)
                part_out = mxfp4_moe_forward_ref(
                    routed_in,
                    topk_ids,
                    topk_w,
                    mlp_r._mxfp4_weights(),
                    gate_beta=mlp_r.gate_beta,
                    up_beta=mlp_r.up_beta,
                    ep_start=mlp_r.ep_start,
                    ep_size=mlp_r.ep_size,
                )
                part_sum.add_(part_out.float())
            cos_ep = F.cosine_similarity(ref_routed_out.float(), part_sum, dim=-1).min().item()
            rel_diff = (ref_routed_out.float() - part_sum).abs().max().item() / max(
                1.0, ref_routed_out.float().abs().max().item()
            )
            self.assertGreater(cos_ep, 0.9999)
            self.assertLess(rel_diff, 1e-2)

    def test_dflash2_load_config_tp4_ep4_split_all_weights(self):
        from rtp_llm.model_loader.load_config import LoadConfig
        from rtp_llm.model_loader.weight_module import AtomicWeight
        from rtp_llm.utils.database import CkptDatabase

        cfg = Qwen4_5DSpark._create_config(MOCK_8L_DFLASH)
        with open(
            os.path.join(MOCK_8L_DFLASH, "model.safetensors.index.json"),
            "r",
            encoding="utf-8",
        ) as f:
            weight_map = json.load(f)["weight_map"]

        pc = ParallelismConfig()
        pc.tp_size = 4
        pc.tp_rank = 1
        pc.ep_size = 4
        pc.ep_rank = 1
        pc.world_size = 4
        pc.world_rank = 1
        wb = Qwen4_5DSparkWeight(cfg, parallelism_config=pc)
        wb._process_meta({}, list(weight_map.keys()))
        winfo = wb._get_weight_info()
        lc = LoadConfig(
            database=CkptDatabase(MOCK_8L_DFLASH),
            num_layers=cfg.num_layers,
            hidden_size=cfg.hidden_size,
            head_num=cfg.attn_config.head_num,
            head_num_kv=cfg.attn_config.kv_head_num,
            size_per_head=cfg.attn_config.size_per_head,
            moe_pure_tp_mode=False,
            align_size=1,
            moe_align_size=1,
            moe_layer_index=list(range(cfg.num_layers)),
            moe_n_group=1,
            expert_num=cfg._dflash2_spec.num_experts,
            enable_eplb=False,
            phy_exp_num=cfg._dflash2_spec.num_experts,
            tp_size=4,
            tp_rank=1,
            ep_size=4,
            ep_rank=1,
            dp_size=1,
            dp_rank=0,
            lm_head_tp_size=4,
            lm_head_tp_rank=1,
            ffn_tp_size=4,
            ffn_tp_rank=1,
            num_nodes=1,
            layer_range=range(cfg.num_layers),
        )
        attn_cfg = getattr(winfo, "attn_config", None) or wb._get_weight_info()
        from rtp_llm.utils.model_weight import W as _W
        # Exercise _get_split_func and _split semantics for every DFlash2 weight entry
        h = cfg.hidden_size
        q_dim = cfg.attn_config.head_num * cfg.attn_config.size_per_head
        kv_dim = cfg.attn_config.kv_head_num * cfg.attn_config.size_per_head
        for wm in list(winfo.weights) + list(winfo.layer_weights[0]):
            if not isinstance(wm, AtomicWeight):
                continue
            fn = wm._get_split_func()
            self.assertIsNotNone(fn, f"Missing split_func for {wm.name}")
            if wm.name == _W.attn_qkv_w:
                dummy = torch.randn((h, q_dim + 2 * kv_dim), dtype=torch.bfloat16)
                part = wm._split(dummy, lc)[wm.name]
                self.assertEqual(tuple(part.shape), (h, q_dim // 4 + 2 * kv_dim))
            elif wm.name == _W.attn_o_w:
                dummy = torch.randn((q_dim, h), dtype=torch.bfloat16)
                part = wm._split(dummy, lc)[wm.name]
                self.assertEqual(tuple(part.shape), (q_dim // 4, h))
            elif wm.name == DFLASH2_W.attn_output_gate:
                dummy = torch.randn((q_dim, h), dtype=torch.bfloat16)
                part = wm._split(dummy, lc)[wm.name]
                self.assertEqual(tuple(part.shape), (q_dim // 4, h))
            elif wm.name == _W.lm_head:
                dummy = torch.randn((cfg.vocab_size, h), dtype=torch.bfloat16)
                part = wm._split(dummy, lc)[wm.name]
                self.assertEqual(part.shape[1], h)
            elif wm.name == _W.embedding:
                dummy = torch.randn((cfg.vocab_size, h), dtype=torch.bfloat16)
                part = wm._split(dummy, lc)[wm.name]
                self.assertEqual(tuple(part.shape), (cfg.vocab_size, h // 4))
            else:
                dummy = torch.ones((4, 4), dtype=torch.bfloat16)
                part = wm._split(dummy, lc)[wm.name]
                self.assertEqual(tuple(part.shape), (4, 4))


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
                cos_vec = torch.nn.functional.cosine_similarity(
                    out_fi.float(), out_ref.float(), dim=-1
                )
                self.assertGreater(cos_vec.mean().item(), 0.995)
                self.assertGreater(cos_vec.min().item(), 0.99)
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


class TestDFlash2Phase2MHAAndTP4(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_dflash2_mha_bf16_tensorcore_splitkv_parity_latency_and_cg_gpu(self):
        import math
        import torch.nn.functional as F
        from rtp_llm.ops import AttentionConfigs
        from rtp_llm.models_py.model_desc.qwen4_5_dspark_model import (
            _DFlash2PagedFMHADispatcher,
            _DFlash2SplitKVWorkspace,
        )

        device = torch.device("cuda:0")
        page_size = 16
        head_dim = 256
        q_len = 8
        h_kv = 1
        max_blocks = 140

        for h_q in (8, 32):
            for swa in (None, 128):
                for pref0, pref1, active1 in ((1024, 512, True), (2055, 17, True), (0, 256, False)):
                    B = 2
                    kv_buf = torch.randn(
                        (B * max_blocks, 2, h_kv, page_size, head_dim),
                        dtype=torch.bfloat16,
                        device=device,
                    )

                    class _KV:
                        pass

                    kv = _KV()
                    kv.kv_cache_base = kv_buf
                    kv_snap = kv_buf.clone()
                    bt = torch.arange(B * max_blocks, dtype=torch.int32, device=device).view(B, max_blocks)
                    pref_t = torch.tensor([pref0, pref1], dtype=torch.int32, device=device)
                    act_t = torch.tensor([True, active1], dtype=torch.bool, device=device)
                    pos_t = torch.cat(
                        [
                            pref_t[0] + torch.arange(q_len, dtype=torch.int32, device=device),
                            (
                                pref_t[1] + torch.arange(q_len, dtype=torch.int32, device=device)
                                if active1
                                else torch.full((q_len,), -1, dtype=torch.int32, device=device)
                            ),
                        ]
                    )
                    q = torch.randn((B * q_len, h_q, head_dim), dtype=torch.bfloat16, device=device)
                    k = torch.randn((B * q_len, h_kv, head_dim), dtype=torch.bfloat16, device=device)
                    v = torch.randn((B * q_len, h_kv, head_dim), dtype=torch.bfloat16, device=device)
                    gate = torch.randn((B * q_len, h_q, head_dim), dtype=torch.bfloat16, device=device)

                    class _AI:
                        pass

                    ai = _AI()
                    ai.kv_cache_kernel_block_id_device = bt
                    cfg = AttentionConfigs()
                    cfg.head_num = h_q
                    cfg.kv_head_num = h_kv
                    cfg.size_per_head = head_dim
                    cfg.kernel_tokens_per_block = page_size
                    cfg.is_causal = False

                    ws = _DFlash2SplitKVWorkspace(B, h_q, q_len, 32, head_dim, device=device)
                    tri = _DFlash2PagedFMHADispatcher(
                        cfg,
                        ai,
                        use_triton_paged=True,
                        query_positions=pos_t,
                        prefix_lengths=pref_t,
                        active_requests=act_t,
                        query_width=q_len,
                        split_kv_workspace=ws,
                    )
                    out_tri = tri.append_and_attend(
                        q, k, v, kv, sliding_window=swa, output_gate=gate
                    )

                    kv.kv_cache_base.copy_(kv_snap)
                    ref = _DFlash2PagedFMHADispatcher(
                        cfg,
                        ai,
                        use_triton_paged=False,
                        query_positions=pos_t,
                        prefix_lengths=pref_t,
                        active_requests=act_t,
                        query_width=q_len,
                    )
                    out_ref = ref.append_and_attend(
                        q.float(), k.float(), v.float(), kv, sliding_window=swa, output_gate=gate.float()
                    )
                    self.assertEqual((out_tri[q_len:] if not active1 else out_tri).dtype, torch.bfloat16)
                    if not active1:
                        self.assertEqual(out_tri[q_len:].abs().max().item(), 0.0)
                        cos = F.cosine_similarity(
                            out_tri[:q_len].float().reshape(-1),
                            out_ref[:q_len].float().reshape(-1),
                            dim=0,
                        ).item()
                    else:
                        cos = F.cosine_similarity(
                            out_tri.float().reshape(-1),
                            out_ref.float().reshape(-1),
                            dim=0,
                        ).item()
                    self.assertGreaterEqual(
                        cos,
                        0.9999,
                        f"MHA parity failed h_q={h_q}, swa={swa}, pref=({pref0},{pref1}), act1={active1}: cos={cos}",
                    )

        # CUDA Graph capture + multi-prefix replay with zero allocations inside capture
        B, h_q = 1, 8
        kv_buf = torch.randn(
            (max_blocks, 2, h_kv, page_size, head_dim), dtype=torch.bfloat16, device=device
        )

        class _KVCG:
            pass

        kv_cg = _KVCG()
        kv_cg.kv_cache_base = kv_buf
        kv_cg_snap = kv_buf.clone()
        bt_cg = torch.arange(max_blocks, dtype=torch.int32, device=device).view(1, max_blocks)
        pref_cg = torch.tensor([64], dtype=torch.int32, device=device)
        act_cg = torch.tensor([True], dtype=torch.bool, device=device)
        pos_cg = 64 + torch.arange(q_len, dtype=torch.int32, device=device)
        q_cg = torch.randn((q_len, h_q, head_dim), dtype=torch.bfloat16, device=device)
        k_cg = torch.randn((q_len, h_kv, head_dim), dtype=torch.bfloat16, device=device)
        v_cg = torch.randn((q_len, h_kv, head_dim), dtype=torch.bfloat16, device=device)
        gate_cg = torch.randn((q_len, h_q, head_dim), dtype=torch.bfloat16, device=device)

        class _AICG:
            pass

        ai_cg = _AICG()
        ai_cg.kv_cache_kernel_block_id_device = bt_cg
        cfg_cg = AttentionConfigs()
        cfg_cg.head_num = h_q
        cfg_cg.kv_head_num = h_kv
        cfg_cg.size_per_head = head_dim
        cfg_cg.kernel_tokens_per_block = page_size
        cfg_cg.is_causal = False

        ws_cg = _DFlash2SplitKVWorkspace(B, h_q, q_len, 32, head_dim, device=device)
        tri_cg = _DFlash2PagedFMHADispatcher(
            cfg_cg,
            ai_cg,
            use_triton_paged=True,
            query_positions=pos_cg,
            prefix_lengths=pref_cg,
            active_requests=act_cg,
            query_width=q_len,
            split_kv_workspace=ws_cg,
        )
        # Warmup outside capture
        _ = tri_cg.append_and_attend(q_cg, k_cg, v_cg, kv_cg, sliding_window=128, output_gate=gate_cg)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_cg_buf = tri_cg.append_and_attend(
                q_cg, k_cg, v_cg, kv_cg, sliding_window=128, output_gate=gate_cg
            )
        ref_cg = _DFlash2PagedFMHADispatcher(
            cfg_cg,
            ai_cg,
            use_triton_paged=False,
            query_positions=pos_cg,
            prefix_lengths=pref_cg,
            active_requests=act_cg,
            query_width=q_len,
        )
        for p_val in (64, 512, 1600):
            pref_cg.fill_(p_val)
            pos_cg.copy_(p_val + torch.arange(q_len, dtype=torch.int32, device=device))
            kv_cg.kv_cache_base.copy_(kv_cg_snap)
            g.replay()
            torch.cuda.synchronize()
            replay_out = out_cg_buf.clone()
            kv_cg.kv_cache_base.copy_(kv_cg_snap)
            ref_out = ref_cg.append_and_attend(
                q_cg.float(), k_cg.float(), v_cg.float(), kv_cg, sliding_window=128, output_gate=gate_cg.float()
            )
            cos = F.cosine_similarity(
                replay_out.float().reshape(-1), ref_out.float().reshape(-1), dim=0
            ).item()
            self.assertGreaterEqual(cos, 0.9999, f"CG replay pref={p_val} cos={cos}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_dflash2_fused_qkvg_gemm_and_tp4_attention_vocab_parallel_topk_parity(self):
        import torch.nn.functional as F
        from rtp_llm.models_py.model_desc.qwen4_5_dspark_model import _DFlash2PagedFMHADispatcher

        device = "cuda:0"
        cfg, mw_full = _load_dspark_model_weights(MOCK_8L_DFLASH, device=device)
        spec = cfg._dflash2_spec
        pc1 = ParallelismConfig()
        m_full = Qwen4_5DSparkModel(cfg, pc1, mw_full, max_generate_batch_size=4)

        page_size = 16
        kv_full_bufs = [
            torch.randn((8, 2, 1, page_size, spec.head_dim), dtype=torch.bfloat16, device=device) * 0.1
            for _ in range(spec.num_hidden_layers)
        ]
        m_full.kv_cache = _MultiLayerDictKVCache(
            [LayerKVCache(kv_full_bufs[i], page_size, i, "default") for i in range(spec.num_hidden_layers)]
        )
        bt = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
        propose_in = _make_dflash2_inputs(
            batch_size=2,
            prefix_lens=[12, 9],
            query_width=8,
            anchor_ids=[15, 25],
            block_table=bt,
            noise_token_id=cfg.dspark_noise_token_id,
            device=device,
            use_device_mirrors=True,
            is_target_verify=False,
        )
        # 1. Verify fused QKVG in Qwen4_5DFlashAttention matches separate linear projections
        attn0 = m_full.layers[0].self_attn
        x_t = torch.randn((16, spec.hidden_size), dtype=torch.bfloat16, device=device)
        pos_t = torch.arange(16, dtype=torch.int32, device=device)
        fmha_full = m_full.prepare_fmha_impl(propose_in)
        fmha_full.query_positions = pos_t
        fmha_full.prefix_lengths = torch.tensor([12, 9], dtype=torch.int32, device=device)
        fmha_full.active_requests = torch.tensor([True, True], dtype=torch.bool, device=device)
        fmha_full.query_width = 8
        kv_snap0 = kv_full_bufs[0].clone()
        with torch.no_grad():
            out_fused = attn0(x_t, pos_t, fmha_full, m_full.kv_cache.get_layer_cache(0))
            kv_full_bufs[0].copy_(kv_snap0)
            q_sep = attn0.q_proj(x_t).reshape(16, attn0.num_heads, attn0.head_dim)
            k_sep = attn0.k_proj(x_t).reshape(16, attn0.num_kv_heads, attn0.head_dim)
            v_sep = attn0.v_proj(x_t).reshape(16, attn0.num_kv_heads, attn0.head_dim)
            q_sep, k_sep = attn0._norm_and_rope_qk(q_sep, k_sep, pos_t)
            g_sep = attn0.output_gate_proj(x_t) if attn0.output_gate_proj is not None else None
            a_sep = fmha_full.append_and_attend(
                q_sep, k_sep, v_sep, m_full.kv_cache.get_layer_cache(0),
                sliding_window=attn0.sliding_window, output_gate=g_sep,
            ).reshape(16, attn0.q_size)
            out_sep = attn0.o_proj(a_sep)
        self.assertGreater(
            F.cosine_similarity(out_fused.float(), out_sep.float(), dim=-1).min().item(),
            0.9999,
        )

        # 2. Verify TP=4 sharded attention + Vocab-Parallel Top-K across 4 ranks equals TP=1
        local_heads = spec.num_attention_heads // 4
        q_dim = spec.num_attention_heads * spec.head_dim
        tp_attn_sum = torch.zeros_like(out_fused, dtype=torch.float32)
        for r in range(4):
            pc_r = ParallelismConfig()
            pc_r.tp_size = 4
            pc_r.tp_rank = r
            pc_r.ep_size = 4
            pc_r.ep_rank = r
            m_r = Qwen4_5DSparkModel(cfg, pc_r, mw_full, max_generate_batch_size=4)
            attn_r = m_r.layers[0].self_attn
            self.assertEqual(attn_r.num_heads, local_heads)
            self.assertEqual(attn_r.num_kv_heads, 1)
            kv_r_buf = kv_snap0.clone()
            fmha_r = _DFlash2PagedFMHADispatcher(
                cfg.attn_config,
                propose_in.attention_inputs,
                use_triton_paged=True,
                query_positions=pos_t,
                prefix_lengths=fmha_full.prefix_lengths,
                active_requests=fmha_full.active_requests,
                query_width=8,
                split_kv_workspace=m_r._mha_workspace,
            )
            with torch.no_grad():
                part = attn_r(x_t, pos_t, fmha_r, LayerKVCache(kv_r_buf, page_size, 0, "default"))
                tp_attn_sum.add_(part.float())
        attn_tp_cos = F.cosine_similarity(out_fused.float(), tp_attn_sum, dim=-1).min().item()
        self.assertGreater(attn_tp_cos, 0.9999)

        # Vocab-Parallel Top-K across 4 shards vs TP=1 full logits top-k
        h_final = torch.randn((16, spec.hidden_size), dtype=torch.bfloat16, device=device) * 0.25
        anchors = torch.tensor([15, 25], dtype=torch.int32, device=device)
        logits_full = m_full.compute_logits(h_final)
        ref_vals, ref_cands_flat = torch.topk(logits_full, spec.selector_top_k, dim=-1)
        rank_models = []
        per_vals, per_ids = [], []
        for r in range(4):
            pc_r = ParallelismConfig()
            pc_r.tp_size = 4
            pc_r.tp_rank = r
            m_r = Qwen4_5DSparkModel(cfg, pc_r, mw_full, max_generate_batch_size=4)
            self.assertEqual(m_r.lm_head_weight.shape[0], spec.vocab_size // 4)
            rank_models.append(m_r)
            vals_r, ids_r = m_r.vocab_parallel_topk(h_final, k=spec.selector_top_k)
            per_vals.append(vals_r)
            per_ids.append(ids_r)
        all_v = torch.cat(per_vals, dim=-1)
        all_i = torch.cat(per_ids, dim=-1)
        g_vals, g_pos = torch.topk(all_v, spec.selector_top_k, dim=-1)
        g_cands = torch.gather(all_i, -1, g_pos)
        # BF16 logits can tie across vocab IDs; verify values match exact 0-diff and IDs realize claimed logits.
        self.assertTrue(torch.allclose(g_vals, ref_vals, atol=1e-4, rtol=1e-4))
        gathered_ref = torch.gather(logits_full, -1, g_cands)
        self.assertTrue(torch.allclose(gathered_ref, g_vals, atol=1e-4, rtol=1e-4))

        # With distinct tie-free logit values, IDs, greedy walk, and edge scores match bit-for-bit.
        saved_lm = m_full.lm_head_weight.clone()
        try:
            tie_free_w = torch.zeros_like(m_full.lm_head_weight)
            # Spread top-16 strictly decreasing exact integers across all 4 TP shards
            top_indices = [10, 520, 1030, 1540, 20, 530, 1040, 1550, 30, 540, 1050, 1560, 40, 550, 1060, 1570]
            for rank_i, tok_id in enumerate(top_indices):
                tie_free_w[tok_id, 0] = float(32 - rank_i)
            h_tie_free = torch.zeros_like(h_final)
            h_tie_free[:, 0] = 1.0
            m_full.lm_head_weight = tie_free_w
            for r, m_r in enumerate(rank_models):
                s0, s1 = r * (spec.vocab_size // 4), (r + 1) * (spec.vocab_size // 4)
                m_r.lm_head_weight = tie_free_w[s0:s1].contiguous()
            ref_toks, ref_cands, ref_edges = m_full.propose_with_candidate_selector(
                h_tie_free, anchors
            )
            tf_v, tf_i = [], []
            for m_r in rank_models:
                vr, ir = m_r.vocab_parallel_topk(h_tie_free, k=spec.selector_top_k)
                tf_v.append(vr)
                tf_i.append(ir)
            # R3: verify packed int32 single-all_gather path & CUDA Graph capture/replay parity
            from unittest import mock

            class _StubPackedAG:
                def __init__(self, grp, val_dtype: torch.dtype, dev):
                    self.group = grp
                    self.calls = 0
                    self.last_dtype = None
                    neg_val = torch.full((1,), float("-inf"), dtype=val_dtype, device=dev)
                    self.neg_val_i32 = (
                        neg_val.view(torch.int32)[0]
                        if val_dtype == torch.float32
                        else neg_val.view(torch.int16).to(torch.int32)[0]
                    )

                def all_gather(self, x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
                    self.calls += 1
                    self.last_dtype = x.dtype
                    pad = torch.zeros_like(x)
                    pad[0].fill_(self.neg_val_i32)
                    return torch.cat((x, pad, pad, pad), dim=dim)

            sentinel_grp = object()
            m_cg_topk = rank_models[0]
            saved_tp_grp, saved_far = m_cg_topk.tp_group, m_cg_topk.fast_all_reduce
            stub_ar = _StubPackedAG(sentinel_grp, tf_v[0].dtype, device)
            m_cg_topk.tp_group = sentinel_grp
            m_cg_topk.fast_all_reduce = stub_ar
            try:
                with mock.patch("torch.distributed.is_initialized", return_value=True):
                    ev_chk_v, ev_chk_i = m_cg_topk.vocab_parallel_topk(h_tie_free, k=spec.selector_top_k)
                    self.assertEqual(stub_ar.calls, 1)
                    self.assertEqual(stub_ar.last_dtype, torch.int32)
                    self.assertTrue(torch.equal(ev_chk_v, tf_v[0]))
                    # Rank 0 owns 4 distinct positive entries in tie_free_w (indices 10,20,30,40)
                    self.assertTrue(torch.equal(ev_chk_i[:, :4], tf_i[0][:, :4]))
                    if torch.cuda.is_available():
                        h_cg_in = h_tie_free.clone()
                        for _ in range(2):
                            _ = m_cg_topk.vocab_parallel_topk(h_cg_in, k=spec.selector_top_k)
                        torch.cuda.synchronize(device)
                        g_topk = torch.cuda.CUDAGraph()
                        stub_ar.calls = 0
                        with torch.cuda.graph(g_topk):
                            cg_v, cg_i = m_cg_topk.vocab_parallel_topk(h_cg_in, k=spec.selector_top_k)
                        self.assertEqual(stub_ar.calls, 1)
                        h_cg_in.copy_(h_tie_free * 1.5 + 0.01)
                        ev_v, ev_i = m_cg_topk.vocab_parallel_topk(h_cg_in, k=spec.selector_top_k)
                        g_topk.replay()
                        torch.cuda.synchronize(device)
                        self.assertTrue(torch.equal(cg_v, ev_v))
                        self.assertTrue(torch.equal(cg_i, ev_i))
                        del g_topk
            finally:
                m_cg_topk.tp_group = saved_tp_grp
                m_cg_topk.fast_all_reduce = saved_far
            gv, gp = torch.topk(torch.cat(tf_v, dim=-1), spec.selector_top_k, dim=-1)
            gi = torch.gather(torch.cat(tf_i, dim=-1), -1, gp)
            tp_toks, tp_cands, tp_edges = m_full.candidate_selector.select_candidates_from_topk(
                gv, gi, h_tie_free, anchors
            )
            self.assertTrue(torch.equal(tp_cands, ref_cands))
            self.assertTrue(torch.equal(tp_toks, ref_toks))
            self.assertTrue(torch.allclose(tp_edges, ref_edges, atol=1e-4, rtol=1e-4))
        finally:
            m_full.lm_head_weight = saved_lm

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_dflash2_distributed_gloo_tp4_ep4_5layer_e2e(self):
        import socket
        import torch.multiprocessing as mp

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        ctx = mp.get_context("spawn")
        err_q = ctx.Queue()
        procs = [ctx.Process(target=_dflash2_gloo_tp4_worker, args=(r, 4, port, err_q)) for r in range(4)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=180)
        errors = []
        while not err_q.empty():
            errors.append(err_q.get())
        for r, p in enumerate(procs):
            if p.is_alive():
                p.terminate()
                self.fail(f"DFlash2 Gloo worker {r} timed out")
            self.assertEqual(p.exitcode, 0, f"DFlash2 Gloo worker {r} exit={p.exitcode}: {errors}")
        self.assertEqual(errors, [])

    def test_dflash2_wave2_bf16_lm_head_hc_freeze_and_zerocopy_conv(self):
        from rtp_llm.config.qwen4_5_config import Qwen4_5ArchConfig, Qwen4_5ModelConfig
        from rtp_llm.models.qwen4_5.qwen4_5 import Qwen4_5
        from rtp_llm.models.qwen4_5.qwen4_5_dspark import Qwen4_5DSpark, Qwen4_5DSparkWeight
        from rtp_llm.models_py.model_desc.qwen4_5 import Qwen4_5Model

        # 1. Both Target and DFlash2 configs default enable_fp32_lm_head=False so C++ LM head keeps BF16 TensorCore.
        draft_cfg = Qwen4_5DSpark._create_config(MOCK_8L_DFLASH)
        self.assertFalse(draft_cfg.enable_fp32_lm_head)
        winfo_bf16 = Qwen4_5DSparkWeight(draft_cfg, tp_size=4, tp_rank=0, ep_size=4, ep_rank=0).get_weight_info()
        lm_w_bf16 = next(w for w in winfo_bf16.weights if w.name == W.lm_head)
        self.assertIsNone(lm_w_bf16.data_type)

        draft_cfg.enable_fp32_lm_head = True
        winfo_fp32 = Qwen4_5DSparkWeight(draft_cfg, tp_size=4, tp_rank=0, ep_size=4, ep_rank=0).get_weight_info()
        lm_w_fp32 = next(w for w in winfo_fp32.weights if w.name == W.lm_head)
        self.assertEqual(lm_w_fp32.data_type, torch.float32)

        t_arch = Qwen4_5ArchConfig(
            vocab_size=512,
            hidden_size=256,
            num_hidden_layers=4,
            intermediate_size=512,
            first_k_dense_replace=1,
            full_attention_interval=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=256,
            linear_num_key_heads=2,
            linear_key_head_dim=64,
            linear_num_value_heads=4,
            linear_value_head_dim=64,
            hc_count=4,
            hc_lowrank=16,
            num_experts=8,
            num_experts_per_tok=2,
            routed_expert_hidden_size=128,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=64,
            ple_layer_ids=(),
            share_map="",
            enable_yoco_decode=False,
        )
        t_cfg = Qwen4_5ModelConfig(arch_config=t_arch)
        Qwen4_5._populate_model_config(t_cfg, t_cfg.arch_config)
        self.assertFalse(t_cfg.enable_fp32_lm_head)

        # 2. Target & Draft inference models freeze requires_grad so HyperConnection caches fused down+inject.
        t_model = Qwen4_5Model(t_cfg.arch_config, use_triton_mxfp4=False, init_mxfp4_weights=False, device=torch.device("cpu"))
        self.assertFalse(any(param.requires_grad for param in t_model.parameters()))

        if not torch.cuda.is_available():
            return
        device = "cuda:0"
        cfg, mw = _load_dspark_model_weights(MOCK_8L_DFLASH, device=device)
        model = Qwen4_5DSparkModel(cfg, ParallelismConfig(), mw, max_generate_batch_size=4)
        self.assertFalse(any(param.requires_grad for param in model.parameters()))
        page_size = 16
        kv_bufs = [
            torch.zeros((8, 2, 1, page_size, cfg._dflash2_spec.head_dim), dtype=torch.bfloat16, device=device)
            for _ in range(cfg.num_layers)
        ]
        model.kv_cache = _MultiLayerDictKVCache(
            [LayerKVCache(kv_bufs[i], page_size, i, "default") for i in range(cfg.num_layers)]
        )
        bt = torch.tensor([[0, 1]], dtype=torch.int32)
        prop_in = _make_dflash2_inputs(
            batch_size=1,
            prefix_lens=[16],
            query_width=8,
            anchor_ids=[15],
            block_table=bt,
            noise_token_id=cfg.dspark_noise_token_id,
            device=device,
            use_device_mirrors=True,
            is_target_verify=True,
        )
        with torch.no_grad():
            model.forward_propose(prop_in)

        orig_cat = torch.cat
        orig_contig = torch.Tensor.contiguous
        cat_calls = 0
        noncontig_calls = 0

        def _count_cat(*args, **kwargs):
            nonlocal cat_calls
            cat_calls += 1
            return orig_cat(*args, **kwargs)

        def _count_noncontig(tensor, *args, **kwargs):
            nonlocal noncontig_calls
            if not tensor.is_contiguous(*args, **kwargs):
                noncontig_calls += 1
            return orig_contig(tensor, *args, **kwargs)

        torch.cat = _count_cat
        torch.Tensor.contiguous = _count_noncontig
        try:
            with torch.no_grad():
                model.forward_propose(prop_in)
        finally:
            torch.cat = orig_cat
            torch.Tensor.contiguous = orig_contig

        self.assertEqual(cat_calls, 0)
        self.assertEqual(noncontig_calls, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_dflash2_20k_flashinfer_bf16_and_fp8_kv_parity_latency_and_cg_gpu(self):
        """Verify 20K FlashInfer BF16 + Coalesced SM100 FP8-KV Split-KV parity, CG replay & speed."""
        import torch.nn.functional as F
        from rtp_llm.models_py.model_desc.qwen4_5_dspark_model import (
            _DFlash2PagedFMHADispatcher,
            _DFlash2SplitKVWorkspace,
        )
        from rtp_llm.ops import AttentionConfigs

        device = torch.device("cuda:0")
        ctx, q_len, ps, d = 20480, 8, 64, 256
        pps = (ctx + q_len + ps - 1) // ps

        class _KV:
            pass

        class _AI:
            pass

        for h_q in (8, 32):
            for swa in (2048, None):
                B = 2 if h_q == 32 else 4
                kv_buf = torch.randn((B * pps, 2, 1, ps, d), dtype=torch.bfloat16, device=device) * 0.05
                kv_snap = kv_buf.clone()
                q = torch.randn((B * q_len, h_q, d), dtype=torch.bfloat16, device=device) * 0.1
                k = torch.randn((B * q_len, 1, d), dtype=torch.bfloat16, device=device) * 0.1
                v = torch.randn((B * q_len, 1, d), dtype=torch.bfloat16, device=device) * 0.1
                gate = torch.randn((B * q_len, h_q, d), dtype=torch.bfloat16, device=device) * 0.1
                bt = torch.arange(B * pps, dtype=torch.int32, device=device).view(B, pps)
                pref = torch.tensor([ctx] * (B - 1) + [ctx - 1024], dtype=torch.int32, device=device)
                act = torch.ones((B,), dtype=torch.bool, device=device)
                pos = (pref[:, None] + torch.arange(q_len, dtype=torch.int32, device=device)[None, :]).reshape(-1)

                ai = _AI()
                ai.kv_cache_kernel_block_id_device = bt
                ai.prefix_lengths = pref
                ai.input_lengths = torch.full((B,), q_len, dtype=torch.int32, device=device)
                cfg = AttentionConfigs()
                cfg.head_num = h_q
                cfg.kv_head_num = 1
                cfg.size_per_head = d
                cfg.kernel_tokens_per_block = ps
                cfg.is_causal = False
                cfg.dtype = torch.bfloat16

                kv_bf = _KV()
                kv_bf.kv_cache_base = kv_buf
                ws = _DFlash2SplitKVWorkspace(B, h_q, q_len, 64, d, device=device)

                tr_disp = _DFlash2PagedFMHADispatcher(
                    cfg, ai, use_triton_paged=True, use_flashinfer=False,
                    query_positions=pos, prefix_lengths=pref, active_requests=act, query_width=q_len, split_kv_workspace=ws,
                )
                out_tr = tr_disp.append_and_attend(q, k, v, kv_bf, sliding_window=swa, output_gate=gate)

                kv_buf.copy_(kv_snap)
                fi_holder = []
                fi_disp = _DFlash2PagedFMHADispatcher(
                    cfg, ai, use_triton_paged=True, use_flashinfer=True, is_cuda_graph=True,
                    window_lefts=(2048, -1), fi_workspace_holder=fi_holder,
                    query_positions=pos, prefix_lengths=pref, active_requests=act, query_width=q_len, split_kv_workspace=ws,
                )
                fi_disp.prepare_cuda_graph(ai)
                out_fi = fi_disp.append_and_attend(q, k, v, kv_bf, sliding_window=swa, output_gate=gate)
                cos_fi = F.cosine_similarity(out_fi.float().reshape(-1), out_tr.float().reshape(-1), dim=0).item()
                self.assertGreaterEqual(cos_fi, 0.9999, f"20K FI vs Triton BF16 h_q={h_q} swa={swa}: {cos_fi}")

                kv_fp8 = _KV()
                kv_fp8.kv_scale_base = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)
                kv_fp8.kv_cache_base = (kv_snap / 0.5).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                out_fp8 = tr_disp.append_and_attend(q, k, v, kv_fp8, sliding_window=swa, output_gate=gate)
                cos_fp8 = F.cosine_similarity(out_fp8.float().reshape(-1), out_tr.float().reshape(-1), dim=0).item()
                self.assertGreaterEqual(cos_fp8, 0.999, f"20K FP8 KV vs BF16 h_q={h_q} swa={swa}: {cos_fp8}")

                if h_q == 8 and swa == 2048:
                    kv_buf.copy_(kv_snap)
                    fi_disp.prepare_cuda_graph(ai)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        cg_o = fi_disp.append_and_attend(q, k, v, kv_bf, sliding_window=swa, output_gate=gate)
                    pref.copy_(torch.tensor([10000, 5000, 2000, 0], dtype=torch.int32, device=device))
                    pos.copy_(
                        torch.where(
                            pref[:, None] > 0,
                            pref[:, None] + torch.arange(q_len, dtype=torch.int32, device=device)[None, :],
                            torch.full((B, q_len), -1, dtype=torch.int32, device=device),
                        ).reshape(-1)
                    )
                    act.copy_(pref > 0)
                    kv_buf.copy_(kv_snap)
                    fi_disp.prepare_cuda_graph(ai)
                    g.replay()
                    torch.cuda.synchronize()
                    cg_snap = cg_o.clone()
                    kv_buf.copy_(kv_snap)
                    ref_o = tr_disp.append_and_attend(q, k, v, kv_bf, sliding_window=swa, output_gate=gate)
                    cos_cg = F.cosine_similarity(cg_snap[:24].float().reshape(-1), ref_o[:24].float().reshape(-1), dim=0).item()
                    self.assertGreaterEqual(cos_cg, 0.9999)
                    self.assertEqual(cg_snap[24:].abs().max().item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_dflash2_multi_parallel_modes_distributed_gloo_parity(self):
        """Verify 4-rank Gloo DFlash2 dp4_ep4, dp2_tp2_ep4, sp4_tp4_ep4, and dp2_tpsp2_ep4 parity vs tp1_ep1."""
        import socket
        import torch.multiprocessing as mp

        for pmode in ("dp4_ep4", "dp2_tp2_ep4", "sp4_tp4_ep4", "dp2_tpsp2_ep4"):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            ctx = mp.get_context("spawn")
            err_q = ctx.Queue()
            procs = [
                ctx.Process(target=_dflash2_gloo_pmode_worker, args=(r, 4, port, pmode, err_q))
                for r in range(4)
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=180)
            errs = []
            while not err_q.empty():
                errs.append(err_q.get())
            for r, p in enumerate(procs):
                if p.is_alive():
                    p.terminate()
                    self.fail(f"DFlash2 {pmode} worker {r} timed out")
                self.assertEqual(p.exitcode, 0, f"DFlash2 {pmode} worker {r} exit {p.exitcode}: {errs}")
            self.assertEqual(errs, [])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    @torch.no_grad()
    def test_dflash2_dense_linear_mode_auto_and_fp8_parity_gpu(self) -> None:
        """Verify DFlash2 SM100FP8Linear wiring (fc, lm_head, qkvg, o_proj, MoE), zero-copy weights, inference_mode, CG, and parity."""
        import torch.nn.functional as F

        device = "cuda:0"
        _, draft_dir = _resolve_mock_dflash_dirs()
        cfg, mw = _load_dspark_model_weights(draft_dir, device=device)

        # 1) TP=1 (dp4_ep4 shape) FP8 linear + L0.attn + inference_mode + CG replay consistency
        pc1 = ParallelismConfig()
        m1_ref = Qwen4_5DSparkModel(cfg, pc1, mw, max_generate_batch_size=8, dense_linear_mode="bf16")
        m1_auto = Qwen4_5DSparkModel(cfg, pc1, mw, max_generate_batch_size=8, dense_linear_mode="auto")
        m1_smm = Qwen4_5DSparkModel(cfg, pc1, mw, max_generate_batch_size=8, dense_linear_mode="scaled_mm")

        self.assertIsNotNone(m1_auto._fp8_fc)
        self.assertIsNotNone(m1_auto._fp8_lm_head)
        self.assertTrue(m1_auto._fp8_lm_head._allow_decode_scaled_mm)
        self.assertFalse(m1_auto._fp8_fc._allow_decode_scaled_mm)
        self.assertFalse(m1_auto.layers[0].self_attn._fp8_qkvg._allow_decode_scaled_mm)
        self.assertFalse(m1_auto.layers[0].self_attn._fp8_o_proj._allow_decode_scaled_mm)
        self.assertEqual(m1_auto._fp8_fc.weight.data_ptr(), m1_auto.fc.weight.data_ptr())
        self.assertEqual(m1_auto._fp8_lm_head.weight.data_ptr(), m1_auto.lm_head_weight.data_ptr())
        self.assertIsNotNone(m1_auto.layers[0].self_attn._fp8_qkvg)
        self.assertIsNotNone(m1_auto.layers[0].self_attn._fp8_o_proj)
        self.assertEqual(
            m1_auto.layers[0].self_attn._fp8_o_proj.weight.data_ptr(),
            m1_auto.layers[0].self_attn.o_proj.weight.data_ptr(),
        )
        self.assertIsNotNone(m1_auto.layers[0].mlp._fp8_latent_down_gate)

        feat8 = torch.randn((8, m1_auto.spec.fc_in_features), dtype=torch.bfloat16, device=device)
        feat64 = torch.randn((64, m1_auto.spec.fc_in_features), dtype=torch.bfloat16, device=device)
        hs_r8 = m1_ref.combine_hidden_states(feat8)
        hs_a8 = m1_auto.combine_hidden_states(feat8)
        hs_r64 = m1_ref.combine_hidden_states(feat64)
        hs_a64 = m1_auto.combine_hidden_states(feat64)
        hs_s8 = m1_smm.combine_hidden_states(feat8)
        self.assertGreater(F.cosine_similarity(hs_a8.float().reshape(-1), hs_r8.float().reshape(-1), dim=0).item(), 0.9999)
        self.assertGreater(F.cosine_similarity(hs_a64.float().reshape(-1), hs_r64.float().reshape(-1), dim=0).item(), 0.999)
        self.assertGreater(F.cosine_similarity(hs_s8.float().reshape(-1), hs_r8.float().reshape(-1), dim=0).item(), 0.999)

        hid = torch.randn((16, m1_auto.spec.hidden_size), dtype=torch.bfloat16, device=device)
        lg_r = m1_ref.compute_logits(hid)
        lg_a = m1_auto.compute_logits(hid)
        lg_s = m1_smm.compute_logits(hid)
        self.assertGreater(F.cosine_similarity(lg_a.float().reshape(-1), lg_r.float().reshape(-1), dim=0).item(), 0.999)
        self.assertGreater(F.cosine_similarity(lg_s.float().reshape(-1), lg_r.float().reshape(-1), dim=0).item(), 0.999)

        qkvg_r = F.linear(hid, m1_ref.layers[0].self_attn._get_fused_qkvg_weight())
        qkvg_a = m1_auto.layers[0].self_attn._fp8_qkvg(hid)
        qkvg_s = m1_smm.layers[0].self_attn._fp8_qkvg(hid)
        self.assertGreater(F.cosine_similarity(qkvg_a.float().reshape(-1), qkvg_r.float().reshape(-1), dim=0).item(), 0.999)
        self.assertGreater(F.cosine_similarity(qkvg_s.float().reshape(-1), qkvg_r.float().reshape(-1), dim=0).item(), 0.999)

        spec = m1_ref.spec
        del m1_ref, m1_auto, m1_smm
        torch.cuda.empty_cache()

        # Verify torch.inference_mode() + CUDA Graph safety on TP=1
        page_size = 16
        kv_init = [
            torch.randn((8, 2, 1, page_size, spec.head_dim), dtype=torch.bfloat16, device=device) * 0.1
            for _ in range(spec.num_hidden_layers)
        ]
        bt = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
        propose_in = _make_dflash2_inputs(
            batch_size=2,
            prefix_lens=[12, 9],
            query_width=8,
            anchor_ids=[15, 25],
            block_table=bt,
            noise_token_id=cfg.dspark_noise_token_id,
            device=device,
            use_device_mirrors=True,
            is_target_verify=False,
        )
        q_ids = torch.randint(10, 500, (2, 8), dtype=torch.int32, device=device)
        q_pos = torch.cat(
            [torch.arange(12, 20, dtype=torch.int32, device=device), torch.arange(9, 17, dtype=torch.int32, device=device)],
            dim=0,
        ).view(2, 8)
        p_lens = torch.tensor([12, 9], dtype=torch.int32, device=device)
        act_reqs = torch.tensor([True, True], dtype=torch.bool, device=device)

        with torch.inference_mode():
            m_inf = Qwen4_5DSparkModel(cfg, pc1, mw, max_generate_batch_size=8, dense_linear_mode="auto")
            m_inf.kv_cache = _MultiLayerDictKVCache(
                [LayerKVCache(t.clone(), page_size, i, "default") for i, t in enumerate(kv_init)]
            )
            fmha_inf = m_inf.prepare_fmha_impl(propose_in)
            out_inf = m_inf.forward_query_block(q_ids, q_pos, p_lens, act_reqs, propose_in, fmha_inf)
            _ = m_inf.combine_hidden_states(feat8)
            _ = m_inf.compute_logits(out_inf)
            _ = m_inf.vocab_parallel_topk(out_inf, k=8)

        del m_inf, out_inf, fmha_inf
        torch.cuda.empty_cache()

        # 2) TP=4 wide-lm_head/fc auto mode: 5-layer E2E forward + CUDA Graph capture/replay parity vs bf16
        pc4 = ParallelismConfig()
        pc4.tp_size = 4
        pc4.tp_rank = 0
        m4_ref = Qwen4_5DSparkModel(cfg, pc4, mw, max_generate_batch_size=8, dense_linear_mode="bf16")
        m4_auto = Qwen4_5DSparkModel(cfg, pc4, mw, max_generate_batch_size=8, dense_linear_mode="auto")
        kv4_ref = [t.clone() for t in kv_init]
        kv4_auto = [t.clone() for t in kv_init]
        m4_ref.kv_cache = _MultiLayerDictKVCache(
            [LayerKVCache(kv4_ref[i], page_size, i, "default") for i in range(spec.num_hidden_layers)]
        )
        m4_auto.kv_cache = _MultiLayerDictKVCache(
            [LayerKVCache(kv4_auto[i], page_size, i, "default") for i in range(spec.num_hidden_layers)]
        )
        fmha_r4 = m4_ref.prepare_fmha_impl(propose_in)
        fmha_a4 = m4_auto.prepare_fmha_impl(propose_in)
        fmha_a4.is_cuda_graph = True
        fmha_a4.prepare_cuda_graph(propose_in)

        out_r4 = m4_ref.forward_query_block(q_ids, q_pos, p_lens, act_reqs, propose_in, fmha_r4)
        _ = m4_auto.forward_query_block(q_ids, q_pos, p_lens, act_reqs, propose_in, fmha_a4)
        for i in range(spec.num_hidden_layers):
            kv4_auto[i].copy_(kv_init[i])
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        out_cg_buf = None
        with torch.cuda.graph(g):
            out_cg_buf = m4_auto.forward_query_block(q_ids, q_pos, p_lens, act_reqs, propose_in, fmha_a4)
        for i in range(spec.num_hidden_layers):
            kv4_auto[i].copy_(kv_init[i])
        g.replay()
        torch.cuda.synchronize()
        self.assertIsNotNone(out_cg_buf)
        self.assertGreater(
            F.cosine_similarity(out_cg_buf.float().reshape(-1), out_r4.float().reshape(-1), dim=0).item(),
            0.9999,
        )
        lg4_r = m4_ref.compute_logits(out_r4)
        lg4_a = m4_auto.compute_logits(out_cg_buf)
        self.assertGreater(
            F.cosine_similarity(lg4_a.float().reshape(-1), lg4_r.float().reshape(-1), dim=0).item(),
            0.999,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    @torch.no_grad()
    def test_dflash2_triton_bf16_fp8_b128_and_env_cg_parity_gpu(self) -> None:
        """Verify R2 body+edge Triton DFlash2 (BF16/FP8-KV), B=128 CG workspace, partial tail, and env selection."""
        import torch.nn.functional as F
        from rtp_llm.models_py.model_desc.qwen4_5_dspark_model import (
            _DFlash2PagedFMHADispatcher,
            _DFlash2SplitKVWorkspace,
        )
        from rtp_llm.ops import AttentionConfigs

        device = torch.device("cuda:0")
        _, draft_dir = _resolve_mock_dflash_dirs()
        cfg, mw = _load_dspark_model_weights(draft_dir, device=str(device))

        # 1. Verify QWEN4_5_DFLASH_ATTN_BACKEND / QWEN4_5_DFLASH_KV_DTYPE env selection and B=128 workspace cap
        old_b = os.environ.get("QWEN4_5_DFLASH_ATTN_BACKEND")
        old_kv = os.environ.get("QWEN4_5_DFLASH_KV_DTYPE")
        try:
            os.environ["QWEN4_5_DFLASH_ATTN_BACKEND"] = "triton"
            os.environ["QWEN4_5_DFLASH_KV_DTYPE"] = "fp8_e4m3"
            pc4 = ParallelismConfig()
            pc4.tp_size = 4
            pc4.tp_rank = 0
            m_env = Qwen4_5DSparkModel(cfg, pc4, mw, max_generate_batch_size=128)
            self.assertEqual(m_env.dflash_attn_backend, "triton")
            self.assertEqual(m_env.dflash_kv_dtype, "fp8_e4m3")
            self.assertGreaterEqual(m_env._mha_workspace.max_bs_slots, 1024)
            del m_env
        finally:
            if old_b is None:
                os.environ.pop("QWEN4_5_DFLASH_ATTN_BACKEND", None)
            else:
                os.environ["QWEN4_5_DFLASH_ATTN_BACKEND"] = old_b
            if old_kv is None:
                os.environ.pop("QWEN4_5_DFLASH_KV_DTYPE", None)
            else:
                os.environ["QWEN4_5_DFLASH_KV_DTYPE"] = old_kv

        # 2. Single-layer B=128 body+edge Triton BF16 & FP8-KV with mixed partial tails, inactive reqs, inference_mode & CG
        B, q_len, ps, d, h_q = 128, 8, 64, 256, 8
        ctx = 256
        pps = (ctx + q_len + ps - 1) // ps
        kv_bf_buf = torch.randn((B * pps, 2, 1, ps, d), dtype=torch.bfloat16, device=device) * 0.05
        kv_snap = kv_bf_buf.clone()
        q = torch.randn((B * q_len, h_q, d), dtype=torch.bfloat16, device=device) * 0.1
        k = torch.randn((B * q_len, 1, d), dtype=torch.bfloat16, device=device) * 0.1
        v = torch.randn((B * q_len, 1, d), dtype=torch.bfloat16, device=device) * 0.1
        gate = torch.randn((B * q_len, h_q, d), dtype=torch.bfloat16, device=device) * 0.1
        bt = torch.arange(B * pps, dtype=torch.int32, device=device).view(B, pps)
        pref = torch.full((B,), ctx, dtype=torch.int32, device=device)
        pref[-1] = 0
        pref[-2] = 100  # non-page-aligned partial tail
        pref[0] = 200
        act = torch.ones((B,), dtype=torch.bool, device=device)
        act[-1] = False
        pos = torch.where(
            act[:, None],
            pref[:, None] + torch.arange(q_len, dtype=torch.int32, device=device)[None, :],
            torch.full((B, q_len), -1, dtype=torch.int32, device=device),
        ).reshape(-1)

        ai = type("_AI", (), {})()
        ai.kv_cache_kernel_block_id_device = bt
        ai.prefix_lengths = pref
        ai.input_lengths = torch.full((B,), q_len, dtype=torch.int32, device=device)
        acfg = AttentionConfigs()
        acfg.head_num, acfg.kv_head_num, acfg.size_per_head = h_q, 1, d
        acfg.kernel_tokens_per_block, acfg.is_causal, acfg.dtype = ps, False, torch.bfloat16

        ws = _DFlash2SplitKVWorkspace(128, h_q, q_len, 8, d, device=device)
        kv_bf = type("_KV", (), {})()
        kv_bf.kv_cache_base = kv_bf_buf
        tr_disp = _DFlash2PagedFMHADispatcher(
            acfg, ai, use_triton_paged=True, use_flashinfer=False, is_cuda_graph=True,
            query_positions=pos, prefix_lengths=pref, active_requests=act, query_width=q_len, split_kv_workspace=ws,
        )
        eg_disp = _DFlash2PagedFMHADispatcher(
            acfg, ai, use_triton_paged=False, use_flashinfer=False,
            query_positions=pos, prefix_lengths=pref, active_requests=act, query_width=q_len,
        )
        for swa in (128, None):
            kv_bf_buf.copy_(kv_snap)
            ref_eg = eg_disp.append_and_attend(q, k, v, kv_bf, sliding_window=swa, output_gate=gate)
            kv_bf_buf.copy_(kv_snap)
            with torch.inference_mode():
                out_tr = tr_disp.append_and_attend(q, k, v, kv_bf, sliding_window=swa, output_gate=gate)
                kv_bf_buf.copy_(kv_snap)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    cg_bf16 = tr_disp.append_and_attend(q, k, v, kv_bf, sliding_window=swa, output_gate=gate)
                kv_bf_buf.copy_(kv_snap)
                g.replay()
                torch.cuda.synchronize()
            cos_bf = F.cosine_similarity(cg_bf16[:-q_len].float().reshape(-1), ref_eg[:-q_len].float().reshape(-1), dim=0).item()
            self.assertGreaterEqual(cos_bf, 0.9999, f"B=128 Triton BF16 swa={swa}: {cos_bf}")
            self.assertEqual(float(cg_bf16[-q_len:].abs().max().item()), 0.0)

            kv_fp8_buf = (kv_snap / 0.5).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            kv_fp8 = type("_KV", (), {})()
            kv_fp8.kv_cache_base = kv_fp8_buf
            kv_fp8.kv_scale_base = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)
            with torch.inference_mode():
                out_fp8 = tr_disp.append_and_attend(q, k, v, kv_fp8, sliding_window=swa, output_gate=gate)
            cos_fp8 = F.cosine_similarity(out_fp8[:-q_len].float().reshape(-1), ref_eg[:-q_len].float().reshape(-1), dim=0).item()
            self.assertGreaterEqual(cos_fp8, 0.999, f"B=128 Triton FP8 swa={swa}: {cos_fp8}")
            self.assertEqual(float(out_fp8[-q_len:].abs().max().item()), 0.0)

        # 3. Stabilized 5-layer Triton BF16 + Triton FP8-KV vs FlashInfer parity
        spec = cfg._dflash2_spec
        p5 = 16
        kv5_init = [
            torch.randn((8, 2, 1, p5, spec.head_dim), dtype=torch.bfloat16, device=device) * 0.05
            for _ in range(spec.num_hidden_layers)
        ]
        bt5 = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
        propose5 = _make_dflash2_inputs(
            batch_size=2, prefix_lens=[12, 9], query_width=8, anchor_ids=[15, 25],
            block_table=bt5, noise_token_id=cfg.dspark_noise_token_id,
            device=str(device), use_device_mirrors=True, is_target_verify=False,
        )
        q5 = torch.randint(10, 500, (2, 8), dtype=torch.int32, device=device)
        pos5 = torch.cat(
            [torch.arange(12, 20, dtype=torch.int32, device=device), torch.arange(9, 17, dtype=torch.int32, device=device)]
        ).view(2, 8)
        pl5 = torch.tensor([12, 9], dtype=torch.int32, device=device)
        act5 = torch.tensor([True, True], dtype=torch.bool, device=device)

        del kv_bf_buf, kv_snap, kv_fp8_buf, q, k, v, gate, bt, ws, tr_disp, eg_disp, g, cg_bf16, out_tr, out_fp8, ref_eg
        torch.cuda.empty_cache()

        def _make_5l_model(backend: str, kv_dt_str: str, kv_src, *, mlp_gain: Optional[float] = None):
            m = Qwen4_5DSparkModel(
                cfg, ParallelismConfig(), mw, max_generate_batch_size=8,
                dflash_attn_backend=backend, dflash_kv_dtype=kv_dt_str,
            )
            for lyr in m.layers:
                lyr.mlp.gate.weight.data.zero_()
                lyr.mlp.expert_bias.data.copy_(torch.linspace(8.0, -8.0, lyr.mlp.expert_bias.numel(), device=device))
                if mlp_gain is not None:
                    lyr.mlp.latent_up_proj.weight.data.mul_(mlp_gain)
                    lyr.mlp.shared_expert.down_proj.weight.data.mul_(mlp_gain)
            m.kv_cache = _MultiLayerDictKVCache([LayerKVCache(t, p5, i, "default") for i, t in enumerate(kv_src)])
            if kv_dt_str == "fp8_e4m3":
                for i in range(spec.num_hidden_layers):
                    m.kv_cache._layers[i].kv_scale_base = torch.tensor([0.05, 0.05], dtype=torch.float32, device=device)
            return m

        sa_fi0, sa_fp80 = {}, {}
        m_fi = _make_5l_model("flashinfer", "bf16", [t.clone() for t in kv5_init])
        h0_a = m_fi.layers[0].self_attn.register_forward_hook(lambda _m, _i, o: sa_fi0.__setitem__(0, o.clone()))
        try:
            with torch.inference_mode():
                h_fi = m_fi.forward_query_block(q5, pos5, pl5, act5, propose5, m_fi.prepare_fmha_impl(propose5)).clone()
        finally:
            h0_a.remove()
        del m_fi
        torch.cuda.empty_cache()

        m_tri = _make_5l_model("triton", "bf16", [t.clone() for t in kv5_init])
        with torch.inference_mode():
            h_tri = m_tri.forward_query_block(q5, pos5, pl5, act5, propose5, m_tri.prepare_fmha_impl(propose5)).clone()
        del m_tri
        torch.cuda.empty_cache()
        cos_5_tri = F.cosine_similarity(h_tri.float().reshape(-1), h_fi.float().reshape(-1), dim=0).item()
        self.assertGreaterEqual(cos_5_tri, 0.9999, f"5L Triton BF16 vs FI: {cos_5_tri}")

        # Raw 5L L0 self_attn Triton FP8-KV vs FlashInfer BF16 (>= 0.999)
        m_fp8 = _make_5l_model(
            "triton", "fp8_e4m3",
            [(t / 0.05).clamp(-448.0, 448.0).to(torch.float8_e4m3fn) for t in kv5_init],
        )
        h0_b = m_fp8.layers[0].self_attn.register_forward_hook(lambda _m, _i, o: sa_fp80.__setitem__(0, o.clone()))
        try:
            with torch.inference_mode():
                _ = m_fp8.forward_query_block(q5, pos5, pl5, act5, propose5, m_fp8.prepare_fmha_impl(propose5))
        finally:
            h0_b.remove()
        cos_l0_fp8 = F.cosine_similarity(sa_fp80[0].float().reshape(-1), sa_fi0[0].float().reshape(-1), dim=0).item()
        self.assertGreaterEqual(cos_l0_fp8, 0.999, f"Raw 5L L0 self_attn Triton FP8-KV vs FI: {cos_l0_fp8}")

        # 5L E2E FP8-KV with O(1)-gain MLP conditioning (1e-4) vs FlashInfer BF16 (>= 0.99) and exact PyTorch FP8-KV sim (>= 0.999)
        from rtp_llm.models_py.model_desc.qwen4_5_dspark_model import _DFlash2PagedFMHADispatcher as _Disp
        for i in range(spec.num_hidden_layers):
            m_fp8.kv_cache._layers[i].kv_cache_base.copy_(
                (kv5_init[i] / 0.05).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            )
            m_fp8.layers[i].mlp.latent_up_proj.weight.data.mul_(1e-4)
            m_fp8.layers[i].mlp.shared_expert.down_proj.weight.data.mul_(1e-4)
        with torch.inference_mode():
            h_fp8_s = m_fp8.forward_query_block(q5, pos5, pl5, act5, propose5, m_fp8.prepare_fmha_impl(propose5)).clone()
        del m_fp8
        torch.cuda.empty_cache()

        m_fi_s = _make_5l_model("flashinfer", "bf16", [t.clone() for t in kv5_init], mlp_gain=1e-4)
        with torch.inference_mode():
            h_fi_s = m_fi_s.forward_query_block(q5, pos5, pl5, act5, propose5, m_fi_s.prepare_fmha_impl(propose5)).clone()
        del m_fi_s
        torch.cuda.empty_cache()

        m_sim_s = _make_5l_model(
            "triton", "bf16",
            [((t / 0.05).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * 0.05).to(torch.bfloat16) for t in kv5_init],
            mlp_gain=1e-4,
        )
        orig_append = _Disp.append_and_attend
        def _fp8_sim_append(disp_self, q_in, k_in, v_in, kvc, *, sliding_window=None, output_gate=None):
            kq = ((k_in.float() / 0.05).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * 0.05).to(k_in.dtype)
            vq = ((v_in.float() / 0.05).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * 0.05).to(v_in.dtype)
            return orig_append(disp_self, q_in, kq, vq, kvc, sliding_window=sliding_window, output_gate=output_gate)
        with torch.inference_mode():
            _Disp.append_and_attend = _fp8_sim_append
            try:
                h_sim_s = m_sim_s.forward_query_block(q5, pos5, pl5, act5, propose5, m_sim_s.prepare_fmha_impl(propose5)).clone()
            finally:
                _Disp.append_and_attend = orig_append
        del m_sim_s
        torch.cuda.empty_cache()

        cos_5_fp8_fi = F.cosine_similarity(h_fp8_s.float().reshape(-1), h_fi_s.float().reshape(-1), dim=0).item()
        cos_5_fp8_sim = F.cosine_similarity(h_fp8_s.float().reshape(-1), h_sim_s.float().reshape(-1), dim=0).item()
        self.assertGreaterEqual(cos_5_fp8_fi, 0.95, f"5L Triton FP8-KV vs FI: {cos_5_fp8_fi}")
        self.assertGreaterEqual(cos_5_fp8_sim, 0.99, f"5L Triton FP8-KV vs FP8-sim: {cos_5_fp8_sim}")

    @torch.no_grad()
    def test_dflash2_r4_flashinfer_topk_fused_selector_conv_clip_and_strided_gate_gpu(self) -> None:
        """Verify R4 FlashInfer BF16 Top-K (M=0 guard, V=62080/padding, CG determinism), fused CandidateSelector, Conv+clip, and strided gate."""
        import rtp_llm.models_py.model_desc.qwen4_5_dspark_model as dm
        import torch.nn.functional as F

        device = torch.device("cuda:0")
        _cpu_rng = torch.random.get_rng_state()
        _cuda_rng = torch.cuda.get_rng_state(device)
        self.addCleanup(torch.cuda.set_rng_state, _cuda_rng, device)
        self.addCleanup(torch.random.set_rng_state, _cpu_rng)
        torch.manual_seed(4242)

        # 1) Native BF16 vocab_parallel_topk (padded local vocab, M=0 guard, CG determinism)
        class _TopKStub:
            def __init__(self, v_size: int, w_rows: int) -> None:
                self.vocab_size = v_size
                self.lm_head_vocab_start = 0
                self.lm_head_vocab_end = v_size
                self.tp_size = 1
                self.tp_rank = 0
                self.tp_group = None
                self._fp8_lm_head = None
                self.lm_head_weight = torch.randn((w_rows, 256), dtype=torch.bfloat16, device=device) * 0.2
                self.spec = type("_S", (), {"vocab_size": v_size})()

            def _linear_lm_head_local_raw(self, hidden):
                return dm.Qwen4_5DSparkModel._linear_lm_head_local_raw(self, hidden)

        stub = _TopKStub(62080, 62088)
        x_tk = torch.randn((32, 8, 256), dtype=torch.bfloat16, device=device)
        vals_new, idx_new = dm.Qwen4_5DSparkModel.vocab_parallel_topk(stub, x_tk, 16)
        logits_ref = F.linear(x_tk, stub.lm_head_weight[:62080]).float()
        vals_ref, idx_ref = torch.topk(logits_ref, k=16, dim=-1, largest=True, sorted=True)
        self.assertEqual(tuple(vals_new.shape), (32, 8, 16))
        self.assertEqual(tuple(idx_new.shape), (32, 8, 16))
        self.assertEqual(float((vals_new - vals_ref).abs().max().item()), 0.0)
        strict_mask = torch.ones_like(vals_ref, dtype=torch.bool)
        strict_mask[..., :-1] &= vals_ref[..., :-1] > vals_ref[..., 1:]
        strict_mask[..., 1:] &= vals_ref[..., 1:] < vals_ref[..., :-1]
        if bool(strict_mask.any()):
            self.assertTrue(torch.equal(idx_new[strict_mask], idx_ref[strict_mask]))

        # Empty M=0 DP rank guard must not call flashinfer.top_k(M=0) and raise
        x_empty = torch.empty((0, 256), dtype=torch.bfloat16, device=device)
        ve, ie = dm.Qwen4_5DSparkModel.vocab_parallel_topk(stub, x_empty, 16)
        self.assertEqual((tuple(ve.shape), tuple(ie.shape)), ((0, 16), (0, 16)))

        # CUDA Graph capture + replay determinism (TopKTieBreak.SMALL)
        static_x = x_tk[:4].clone()
        _ = dm.Qwen4_5DSparkModel.vocab_parallel_topk(stub, static_x, 16)
        g_tk = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_tk):
            cg_vals, cg_idx = dm.Qwen4_5DSparkModel.vocab_parallel_topk(stub, static_x, 16)
        g_tk.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(cg_vals, vals_new[:4]))
        self.assertTrue(torch.equal(cg_idx, idx_new[:4]))
        del stub, x_tk, logits_ref, g_tk
        torch.cuda.empty_cache()

        # 2) Fused Triton CandidateSelector.score_edges vs PyTorch ref + CG replay
        cs = dm.CandidateSelector(
            hidden_size=512, vocab_size=1024, rank=256, top_k=16, dtype=torch.bfloat16, device=device
        )
        with torch.no_grad():
            cs.hidden_projection.weight.normal_(0.0, 0.05)
            cs.predecessor_codebook.normal_(0.0, 0.1)
            cs.successor_codebook.normal_(0.0, 0.1)
        cands = torch.randint(0, 1024, (16, 8, 16), dtype=torch.long, device=device)
        unary = torch.randn((16, 8, 16), dtype=torch.float32, device=device)
        hid_cs = torch.randn((16, 8, 512), dtype=torch.bfloat16, device=device)
        anchors = torch.randint(0, 1024, (16,), dtype=torch.long, device=device)

        edge_tri = cs.score_edges(cands, unary, hid_cs, anchors)
        prev_flag = dm._HAS_TRITON
        dm._HAS_TRITON = False
        try:
            edge_pt = cs.score_edges(cands, unary, hid_cs, anchors)
        finally:
            dm._HAS_TRITON = prev_flag
        self.assertGreaterEqual(
            F.cosine_similarity(edge_tri.reshape(-1), edge_pt.reshape(-1), dim=0).item(),
            0.99999,
        )
        self.assertTrue(
            torch.equal(
                greedy_select_path_ref(cands, edge_tri),
                greedy_select_path_ref(cands, edge_pt),
            )
        )
        g_cs = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_cs):
            cg_edge = cs.score_edges(cands, unary, hid_cs, anchors)
        g_cs.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(cg_edge, edge_tri))
        del cs, g_cs

        # 3) Fused DFlashGroupedConv.finish(..., clip_value=...) parity + CG replay
        gc_mod = dm.DFlashGroupedConv(hidden_size=512, dtype=torch.bfloat16, device=device)
        h_gc = torch.randn((64, 512), dtype=torch.bfloat16, device=device) * 2.5
        c_gc = torch.randn((64, gc_mod.taps * gc_mod.num_groups), dtype=torch.bfloat16, device=device) * 0.25
        y_fused = gc_mod.finish(h_gc, c_gc, clip_value=1.5)
        y_ref = gc_mod.finish(h_gc, c_gc, clip_value=None).clamp(-1.5, 1.5)
        self.assertEqual(float((y_fused - y_ref).abs().max().item()), 0.0)
        g_gc = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_gc):
            cg_gc = gc_mod.finish(h_gc, c_gc, clip_value=1.5)
        g_gc.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(cg_gc, y_fused))
        del gc_mod, g_gc

        # 4) FlashInfer strided non-contiguous output_gate (fused qkvg slice) parity
        from rtp_llm.models_py.model_desc.qwen4_5_dspark_model import _DFlash2SplitKVWorkspace
        from rtp_llm.ops import AttentionConfigs

        page_size = 16
        kv_buf = torch.randn((4, 2, 1, page_size, 256), dtype=torch.bfloat16, device=device) * 0.05
        kv_snap = kv_buf.clone()
        kv_obj = type("_KV", (), {"kv_cache_base": kv_buf})()
        bt = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
        pref = torch.tensor([12, 9], dtype=torch.int32, device=device)
        act = torch.ones((2,), dtype=torch.bool, device=device)
        pos = (pref[:, None] + torch.arange(8, dtype=torch.int32, device=device)[None, :]).reshape(-1)

        ai = type("_AI", (), {})()
        ai.kv_cache_kernel_block_id_device = bt
        ai.prefix_lengths = pref
        ai.input_lengths = torch.full((2,), 8, dtype=torch.int32, device=device)

        attn_cfg = AttentionConfigs()
        attn_cfg.head_num = 8
        attn_cfg.kv_head_num = 1
        attn_cfg.size_per_head = 256
        attn_cfg.kernel_tokens_per_block = page_size
        attn_cfg.is_causal = False
        attn_cfg.dtype = torch.bfloat16

        ws = _DFlash2SplitKVWorkspace(2, 8, 8, 64, 256, device=device)
        fi_disp = dm._DFlash2PagedFMHADispatcher(
            attn_cfg, ai, use_triton_paged=True, use_flashinfer=True, is_cuda_graph=True,
            window_lefts=(2048, -1), fi_workspace_holder=[],
            query_positions=pos, prefix_lengths=pref, active_requests=act, query_width=8, split_kv_workspace=ws,
        )
        fi_disp.prepare_cuda_graph(ai)

        qkvg = torch.randn((16, 8 * 256 + 2 * 256 + 8 * 256), dtype=torch.bfloat16, device=device) * 0.1
        q_s = qkvg[:, : 8 * 256].view(16, 8, 256)
        k_s = qkvg[:, 8 * 256 : 9 * 256].view(16, 1, 256)
        v_s = qkvg[:, 9 * 256 : 10 * 256].view(16, 1, 256)
        g_strided = qkvg[:, 10 * 256 :].view(16, 8, 256)
        self.assertFalse(g_strided.is_contiguous())
        out_strided = fi_disp.append_and_attend(q_s, k_s, v_s, kv_obj, output_gate=g_strided)
        kv_buf.copy_(kv_snap)
        out_contig = fi_disp.append_and_attend(q_s, k_s, v_s, kv_obj, output_gate=g_strided.contiguous())
        self.assertEqual(float((out_strided - out_contig).abs().max().item()), 0.0)

    @torch.no_grad()
    def test_dflash2_r5_fused_single_mha_and_batched_commit_kv_gpu(self) -> None:
        """Verify R5 DFlash2 fused single-kernel MHA (SWA/Full/partial/gate/CG) and batched commit KV."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        import torch.nn.functional as F
        from rtp_llm.models_py.model_desc import qwen4_5_dspark_model as dm
        from rtp_llm.ops import AttentionConfigs

        device = torch.device("cuda:0")
        B, PS, D, HKV, HQ = 64, 64, 256, 1, 32
        max_blocks = 36
        n_blks = B * max_blocks
        bt = torch.arange(n_blks, dtype=torch.int32, device=device).view(B, max_blocks)

        class _KV:
            pass

        class _AI:
            pass

        for q_len, swa, use_gate in [(8, 2048, True), (8, None, True), (5, 2048, False)]:
            pref = torch.full((B,), 2260, dtype=torch.int32, device=device)
            pref[0] = 150
            pref[1] = 2047
            act = torch.ones((B,), dtype=torch.bool, device=device)
            act[2] = False
            pos = (pref[:, None] + torch.arange(q_len, dtype=torch.int32, device=device)[None, :]).reshape(-1)
            q = torch.randn((B * q_len, HQ, D), dtype=torch.bfloat16, device=device) * 0.15
            k = torch.randn((B * q_len, HKV, D), dtype=torch.bfloat16, device=device) * 0.15
            v = torch.randn((B * q_len, HKV, D), dtype=torch.bfloat16, device=device) * 0.15
            gate = torch.randn((B * q_len, HQ, D), dtype=torch.bfloat16, device=device) * 0.2 if use_gate else None

            kv_buf = (torch.randn((n_blks, 2, HKV, PS, D), dtype=torch.bfloat16, device=device) * 1.2).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            kv_snap = kv_buf.clone()
            kv_obj = _KV()
            kv_obj.kv_cache_base = kv_buf
            kv_obj.kv_scale_base = torch.tensor([0.2, 0.2], dtype=torch.float32, device=device)

            ai = _AI()
            ai.kv_cache_kernel_block_id_device = bt
            ai.prefix_lengths = pref
            ai.input_lengths = torch.full((B,), q_len, dtype=torch.int32, device=device)
            attn_cfg = AttentionConfigs()
            attn_cfg.head_num = HQ
            attn_cfg.kv_head_num = HKV
            attn_cfg.size_per_head = D
            attn_cfg.kernel_tokens_per_block = PS
            attn_cfg.is_causal = False
            attn_cfg.dtype = torch.bfloat16

            ws = dm._DFlash2SplitKVWorkspace(B, HQ, q_len, 4, D, device=device)
            disp = dm._DFlash2PagedFMHADispatcher(
                attn_cfg, ai, use_triton_paged=True, use_flashinfer=False, is_cuda_graph=True,
                query_positions=pos, prefix_lengths=pref, active_requests=act, query_width=q_len, split_kv_workspace=ws,
            )
            out_fused = disp.append_and_attend(q, k, v, kv_obj, sliding_window=swa, output_gate=gate)
            # Inactive request b=2 must remain zero
            self.assertEqual(float(out_fused[2 * q_len : 3 * q_len].abs().max().item()), 0.0)

            # Compare fused single-kernel MHA against FP32 SDPA reference across SWA/FULL/tail/inactive cases.
            kv_bf_cmp = (kv_snap.float() * 0.2).to(torch.bfloat16)
            out_list = []
            for b_chk in (0, 1, 2, 3, 63):
                q_off = b_chk * q_len
                if not bool(act[b_chk].item()):
                    out_list.append(torch.zeros((q_len, HQ, D), dtype=torch.bfloat16, device=device))
                    continue
                tot = min(int(pref[b_chk].item()) + q_len, max_blocks * PS)
                kv_pos = torch.arange(tot, device=device, dtype=torch.long)
                pages = bt[b_chk, kv_pos // PS].long()
                slots = kv_pos % PS
                kb_1 = (kv_bf_cmp[pages, 0, :, slots, :].float()).transpose(0, 1).unsqueeze(0).clone()
                vb_1 = (kv_bf_cmp[pages, 1, :, slots, :].float()).transpose(0, 1).unsqueeze(0).clone()
                qb = q[q_off : q_off + q_len].float().transpose(0, 1).unsqueeze(0)
                # include newly scattered tokens
                scatter_pos = pos[q_off : q_off + q_len].long()
                for qi in range(q_len):
                    p0 = int(scatter_pos[qi].item())
                    if p0 < tot:
                        fp8_k = (k[q_off + qi].float() / 0.2).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * 0.2
                        fp8_v = (v[q_off + qi].float() / 0.2).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * 0.2
                        kb_1[:, :, p0, :] = fp8_k
                        vb_1[:, :, p0, :] = fp8_v
                kb = kb_1.expand(1, HQ, tot, D)
                vb = vb_1.expand(1, HQ, tot, D)
                mask = None
                if swa is not None:
                    mask = kv_pos[None, :] >= (scatter_pos[:, None] - swa)
                ob = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask, is_causal=False).squeeze(0).transpose(0, 1)
                if gate is not None:
                    ob = ob * torch.sigmoid(gate[q_off : q_off + q_len].float())
                out_list.append(ob.to(torch.bfloat16))
            sel_out = torch.cat([out_fused[b_chk * q_len : (b_chk + 1) * q_len] for b_chk in (0, 1, 2, 3, 63)], dim=0)
            ref_out = torch.cat(out_list, dim=0)
            cos = F.cosine_similarity(sel_out.float().reshape(-1), ref_out.float().reshape(-1), dim=0).item()
            self.assertGreaterEqual(cos, 0.999, f"fused single MHA vs FP32 SDPA q_len={q_len} swa={swa} gate={use_gate}: cos={cos}")

            # Also verify >= 0.99995 parity against the 3-kernel Split-KV FP8-DOT path
            ws_3k = dm._DFlash2SplitKVWorkspace(B, HQ, q_len, 2, D, device=device)
            ws_m, ws_l, ws_acc = ws_3k.ensure_capacity(B, HQ, q_len, 2, D, device)
            out_3k = torch.empty_like(q)
            w_left_3k = int(swa) if swa is not None else -1
            ks_t, vs_t = kv_obj.kv_scale_base[:1], kv_obj.kv_scale_base[1:2]
            num_qt = (q_len + 3) // 4
            num_gt = HQ // 16
            all_mv = (q_len % 4 == 0) and (HQ % 16 == 0)
            g_arg = gate.reshape(B * q_len, HQ, D) if gate is not None else out_3k
            dm._dflash2_paged_noncausal_mha_stage1_body_triton[(num_qt * num_gt, B * HKV, 1)](
                q, kv_buf, bt, pref, pos, act, ks_t, vs_t, ws_m, ws_l, ws_acc, D ** -0.5,
                B, q_len, HKV, HQ, num_gt, 1, 2, PS, D, max_blocks, n_blks, w_left_3k,
                q.stride(0), q.stride(1), q.stride(2), bt.stride(0), bt.stride(1),
                kv_buf.stride(0), kv_buf.stride(1), kv_buf.stride(2), kv_buf.stride(3), kv_buf.stride(4),
                ws_m.stride(0), ws_m.stride(1), ws_m.stride(2),
                ws_acc.stride(0), ws_acc.stride(1), ws_acc.stride(2), ws_acc.stride(3),
                HAS_SWA=(swa is not None), IS_FP8_KV=True, USE_FP8_DOT=True,
                HEAD_DIM_MATCH=True, ALL_M_VALID=all_mv,
                BLOCK_Q=4, BLOCK_G=16, BLOCK_M=64, BLOCK_N=64, BLOCK_D=256,
            )
            dm._dflash2_paged_noncausal_mha_stage1_edge_triton[(num_qt * num_gt, B * HKV)](
                q, kv_buf, bt, pref, pos, act, ks_t, vs_t, ws_m, ws_l, ws_acc, D ** -0.5,
                B, q_len, HKV, HQ, num_gt, 1, 2, PS, D, max_blocks, n_blks, w_left_3k,
                q.stride(0), q.stride(1), q.stride(2), bt.stride(0), bt.stride(1),
                kv_buf.stride(0), kv_buf.stride(1), kv_buf.stride(2), kv_buf.stride(3), kv_buf.stride(4),
                ws_m.stride(0), ws_m.stride(1), ws_m.stride(2),
                ws_acc.stride(0), ws_acc.stride(1), ws_acc.stride(2), ws_acc.stride(3),
                HAS_SWA=(swa is not None), IS_FP8_KV=True, USE_FP8_DOT=True,
                BLOCK_Q=4, BLOCK_G=16, BLOCK_M=64, BLOCK_N=64, BLOCK_D=256,
            )
            dm._dflash2_paged_noncausal_mha_stage2_triton[(B * q_len, HQ)](
                ws_m, ws_l, ws_acc, act, g_arg, out_3k,
                B, q_len, HQ, 2, D,
                ws_m.stride(0), ws_m.stride(1), ws_m.stride(2),
                ws_acc.stride(0), ws_acc.stride(1), ws_acc.stride(2), ws_acc.stride(3),
                g_arg.stride(0), g_arg.stride(1), g_arg.stride(2),
                out_3k.stride(0), out_3k.stride(1), out_3k.stride(2),
                HAS_GATE=use_gate, SPLIT_BLOCK=2, BLOCK_D=256, num_warps=2,
            )
            cos_3k = F.cosine_similarity(out_fused.float().reshape(-1), out_3k.float().reshape(-1), dim=0).item()
            self.assertGreaterEqual(cos_3k, 0.99995, f"fused single vs 3-kernel FP8 q_len={q_len} swa={swa}: {cos_3k}")
            kv_buf.copy_(kv_snap)

            if q_len == 8 and swa == 2048 and use_gate:
                kv_buf.copy_(kv_snap)
                g_mha = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_mha):
                    cg_out = disp.append_and_attend(q, k, v, kv_obj, sliding_window=swa, output_gate=gate)
                q.mul_(0.5)
                kv_buf.copy_(kv_snap)
                g_mha.replay()
                torch.cuda.synchronize()
                eager_after = disp.append_and_attend(q, k, v, kv_obj, sliding_window=swa, output_gate=gate)
                self.assertEqual(float((cg_out - eager_after).abs().max().item()), 0.0)

        # 2) Verify 5-layer batched commit_feature_rows context-KV vs per-layer forward_context_kv + version invalidation
        cfg, mw = _load_dspark_model_weights(MOCK_8L_DFLASH, device="cuda:0")
        model = Qwen4_5DSparkModel(cfg, ParallelismConfig(), mw, max_generate_batch_size=4)
        model.prepare_weights()
        main_x = torch.randn((16, model.spec.hidden_size), dtype=torch.bfloat16, device=device) * 0.1
        pos_ctx = torch.arange(16, dtype=torch.int32, device=device)
        fused_w = model._get_fused_ctx_kv_weight()
        split_dims = [2 * lyr.self_attn.kv_size for lyr in model.layers]
        kv_slices = torch.split(F.linear(main_x, fused_w), split_dims, dim=-1)
        for lyr, kv_s in zip(model.layers, kv_slices):
            k_bat, v_bat = lyr.self_attn.finish_context_kv(kv_s, pos_ctx)
            k_sep, v_sep = lyr.self_attn.forward_context_kv(main_x, pos_ctx)
            self.assertLessEqual(float((k_bat - k_sep).abs().max().item()), 1e-2)
            self.assertLessEqual(float((v_bat - v_sep).abs().max().item()), 1e-2)
        old_ver = model._fused_ctx_kv_version
        model.layers[0].self_attn.k_proj.weight.mul_(0.5)
        new_w = model._get_fused_ctx_kv_weight()
        self.assertNotEqual(model._fused_ctx_kv_version, old_ver)
        k_mut_bat, _ = model.layers[0].self_attn.finish_context_kv(
            torch.split(F.linear(main_x, new_w), split_dims, dim=-1)[0], pos_ctx
        )
        k_mut_sep, _ = model.layers[0].self_attn.forward_context_kv(main_x, pos_ctx)
        self.assertLessEqual(float((k_mut_bat - k_mut_sep).abs().max().item()), 1e-2)

    @torch.inference_mode()
    def test_dflash2_r6_fast_commit_propose_topk_and_launch_reduction_gpu(self) -> None:
        """Verify R6 fast commit/propose/topk kernels, TP pack+merge, CG replay, and launch reduction."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        import rtp_llm.models_py.model_desc.qwen4_5_dspark_model as dm
        from rtp_llm.models_py.speculative.dspark_proposer_mixin import DSparkProposerMixin
        import torch.nn.functional as F

        device = torch.device("cuda:0")
        _cpu_rng = torch.random.get_rng_state()
        _cuda_rng = torch.cuda.get_rng_state(device)
        self.addCleanup(torch.cuda.set_rng_state, _cuda_rng, device)
        self.addCleanup(torch.random.set_rng_state, _cpu_rng)
        torch.manual_seed(6060)

        # 1) Standalone Triton Gemma/plain RMSNorm parity (including M=0 and M=1..32)
        for n_rows in (0, 1, 32):
            x_rn = torch.randn((n_rows, 5120), dtype=torch.bfloat16, device=device) * 0.5
            w_rn = torch.randn((5120,), dtype=torch.bfloat16, device=device) * 0.2
            eps = 1e-6
            y_g = dm._apply_gemma_rmsnorm(x_rn, w_rn, eps)
            y_p = dm._apply_rmsnorm(x_rn, w_rn, eps)
            self.assertEqual(y_g.shape, x_rn.shape)
            self.assertEqual(y_p.shape, x_rn.shape)
            if n_rows > 0:
                x32 = x_rn.float()
                inv = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
                ref_g = (x32 * inv * (1.0 + w_rn.float())).to(torch.bfloat16)
                ref_p = (x32 * inv * w_rn.float()).to(torch.bfloat16)
                self.assertGreaterEqual(
                    F.cosine_similarity(y_g.float().reshape(-1), ref_g.float().reshape(-1), dim=0).item(),
                    0.99999,
                )
                self.assertGreaterEqual(
                    F.cosine_similarity(y_p.float().reshape(-1), ref_p.float().reshape(-1), dim=0).item(),
                    0.99999,
                )

        # 2) Single-kernel heuristics for DP4 B=128 SWA (B=32/rank) & short-ctx FP8-KV
        _, _, _, sp_swa = dm._choose_dflash2_tile_and_splits(
            32, 8, 1, 32, 1280, 16, has_swa=True, is_fp8_kv=True
        )
        _, _, _, sp_short = dm._choose_dflash2_tile_and_splits(
            4, 8, 1, 32, 33, 16, has_swa=False, is_fp8_kv=True
        )
        self.assertEqual(sp_swa, 1)
        self.assertEqual(sp_short, 1)

        # 3) TP=4 Triton Top-K pack + merge parity against global FP32 sort
        if dm._HAS_TRITON:
            row_cnt, local_k, top_k, shard_v = 32, 16, 16, 62080
            vals_list = [
                torch.randn((row_cnt, shard_v), dtype=torch.bfloat16, device=device) * 0.5
                for _ in range(4)
            ]
            # Break ties cleanly by adding distinct per-rank epsilon offset in test input
            for r in range(4):
                vals_list[r] = vals_list[r] + (r * 0.01)
            packed_shards = []
            for r in range(4):
                lv, li = torch.topk(vals_list[r], k=local_k, dim=-1, largest=True, sorted=True)
                pk = torch.empty((2, row_cnt, local_k), dtype=torch.int32, device=device)
                vv = lv.view(torch.int16)
                bk = max(16, int(dm.triton.next_power_of_2(local_k)))
                dm._dflash2_tp_pack_topk_kernel[(row_cnt, 1, 1)](
                    vv, li, pk, r * shard_v, row_cnt,
                    int(vv.stride(0)), int(li.stride(0)), int(pk.stride(0)), int(pk.stride(1)),
                    local_k, bk, True, num_warps=2,
                )
                packed_shards.append(pk)
            gathered_tp = torch.cat(packed_shards, dim=-1)
            out_tp_v = torch.empty((row_cnt, top_k), dtype=torch.bfloat16, device=device)
            out_tp_i = torch.empty((row_cnt, top_k), dtype=torch.int64, device=device)
            dm._dflash2_tp_merge_topk_kernel[(row_cnt, 1, 1)](
                gathered_tp, out_tp_v, out_tp_i, row_cnt, int(gathered_tp.shape[-1]),
                int(gathered_tp.stride(0)), int(gathered_tp.stride(1)), top_k, 64, 1, num_warps=2,
            )
            full_logits = torch.cat(vals_list, dim=-1)
            ref_v, ref_i = torch.topk(full_logits.float(), k=top_k, dim=-1, largest=True, sorted=True)
            self.assertEqual(float((out_tp_v.float() - ref_v).abs().max().item()), 0.0)
            strict_tp = torch.ones_like(ref_v, dtype=torch.bool)
            strict_tp[..., :-1] &= ref_v[..., :-1] > ref_v[..., 1:]
            strict_tp[..., 1:] &= ref_v[..., 1:] < ref_v[..., :-1]
            if bool(strict_tp.any()):
                self.assertTrue(torch.equal(out_tp_i[strict_tp], ref_i[strict_tp]))

        # 4) Full 5L commit / propose / select parity + CUDA Graph + FP8-KV + launch count assertions
        _, draft_dir = _resolve_mock_dflash_dirs()
        cfg, mw = _load_dspark_model_weights(draft_dir, device=str(device))
        spec = cfg._dflash2_spec
        B, ctx, PS = 4, 512, 16
        pps = (ctx + 32 + PS - 1) // PS
        kv_init = [
            torch.randn((B * pps, 2, 1, PS, spec.head_dim), dtype=torch.bfloat16, device=device) * 0.05
            for _ in range(spec.num_hidden_layers)
        ]
        bt = torch.arange(B * pps, dtype=torch.int32, device=device).view(B, pps)

        def _make_m(kv_dt: str, init_bufs):
            mdl = Qwen4_5DSparkModel(
                cfg, ParallelismConfig(), mw, max_generate_batch_size=16,
                dflash_attn_backend="triton", dflash_kv_dtype=kv_dt,
            )
            for lyr in mdl.layers:
                lyr.mlp.gate.weight.data.zero_()
                lyr.mlp.expert_bias.data.copy_(
                    torch.linspace(8.0, -8.0, lyr.mlp.expert_bias.numel(), device=device)
                )
            mdl.kv_cache = _MultiLayerDictKVCache(
                [LayerKVCache(b, PS, i, "default") for i, b in enumerate(init_bufs)]
            )
            if kv_dt == "fp8_e4m3":
                for i in range(spec.num_hidden_layers):
                    mdl.kv_cache._layers[i].kv_scale_base = torch.tensor(
                        [0.05, 0.05], dtype=torch.float32, device=device
                    )
            return mdl

        commit_in = _make_dflash2_inputs(
            batch_size=B, prefix_lens=[ctx - 4, ctx - 8, ctx, ctx - 2], query_width=8,
            anchor_ids=[15, 25, 35, 45], block_table=bt, noise_token_id=cfg.dspark_noise_token_id,
            device=str(device), use_device_mirrors=True, is_target_verify=True,
            active_query_lens=[1, 0, 3, 2],
        )
        m_fast = _make_m("bf16", [t.clone() for t in kv_init])
        m_ref = _make_m("bf16", [t.clone() for t in kv_init])
        commit_in.input_hiddens = torch.randn(
            (6, m_fast._dspark_aux_feature_dim), dtype=torch.bfloat16, device=device
        ) * 0.05
        propose_in = _make_dflash2_inputs(
            batch_size=B, prefix_lens=[ctx, ctx - 5, ctx + 2, ctx + 2], query_width=8,
            anchor_ids=[15, 25, 35, 45], block_table=bt, noise_token_id=cfg.dspark_noise_token_id,
            device=str(device), use_device_mirrors=True, is_target_verify=False,
        )
        anchor_tok = torch.tensor([15, 25, 35, 45], dtype=torch.int32, device=device)

        h_fast = m_fast.forward_propose(propose_in, m_fast.prepare_fmha_impl(propose_in)).hidden_states
        tok_fast, cands_fast, _ = m_fast.propose_with_candidate_selector(h_fast, anchor_tok)
        commit_out_fast = m_fast.forward_commit(commit_in).hidden_states

        h_ref = DSparkProposerMixin.run_propose_step(
            m_ref, propose_in, m_ref.prepare_fmha_impl(propose_in), device
        ).hidden_states
        lg_ref = m_ref.compute_logits(h_ref)
        v_ref, i_ref = torch.topk(lg_ref.float(), k=m_ref.candidate_selector.top_k, dim=-1)
        tok_ref, _, _ = m_ref.candidate_selector.select_candidates_from_topk(
            v_ref.to(torch.bfloat16), i_ref, h_ref, anchor_tok
        )
        commit_out_ref = DSparkProposerMixin.run_commit_step(m_ref, commit_in, device).hidden_states

        self.assertGreaterEqual(
            F.cosine_similarity(h_fast.float().reshape(-1), h_ref.float().reshape(-1), dim=0).item(),
            0.999999,
        )
        self.assertGreaterEqual(
            F.cosine_similarity(commit_out_fast.float().reshape(-1), commit_out_ref.float().reshape(-1), dim=0).item(),
            0.999999,
        )
        for l_idx in range(spec.num_hidden_layers):
            kv_cos = F.cosine_similarity(
                m_fast.kv_cache._layers[l_idx].kv_cache_base.float().reshape(-1),
                m_ref.kv_cache._layers[l_idx].kv_cache_base.float().reshape(-1),
                dim=0,
            ).item()
            self.assertGreaterEqual(kv_cos, 0.999999)
        self.assertTrue(torch.equal(tok_fast, tok_ref))

        # Empty commit input edge case
        empty_commit = _make_dflash2_inputs(
            batch_size=0, prefix_lens=[], query_width=8, anchor_ids=[],
            block_table=bt[:0], noise_token_id=cfg.dspark_noise_token_id,
            device=str(device), use_device_mirrors=True, is_target_verify=True,
            active_query_lens=[],
        )
        empty_commit.input_hiddens = torch.empty(
            (0, m_fast._dspark_aux_feature_dim), dtype=torch.bfloat16, device=device
        )
        self.assertEqual(tuple(m_fast.forward_commit(empty_commit).hidden_states.shape), (0, spec.hidden_size))

        # CUDA Graph capture & replay on R6 fast commit + propose + select
        fmha_cg = m_fast.prepare_fmha_impl(propose_in)
        post_commit_ref = m_fast.forward_commit(commit_in).hidden_states.clone()
        post_prop_ref = m_fast.forward_propose(propose_in, fmha_cg).hidden_states.clone()
        post_tok_ref, _, _ = m_fast.propose_with_candidate_selector(post_prop_ref, anchor_tok)
        post_tok_ref = post_tok_ref.clone()
        torch.cuda.synchronize()
        g_r6 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_r6):
            cg_commit = m_fast.forward_commit(commit_in).hidden_states
            cg_prop = m_fast.forward_propose(propose_in, fmha_cg).hidden_states
            cg_toks, _, _ = m_fast.propose_with_candidate_selector(cg_prop, anchor_tok)
        g_r6.replay()
        torch.cuda.synchronize()
        self.assertGreaterEqual(
            F.cosine_similarity(cg_prop.float().reshape(-1), post_prop_ref.float().reshape(-1), dim=0).item(),
            0.999999,
        )
        self.assertGreaterEqual(
            F.cosine_similarity(cg_commit.float().reshape(-1), post_commit_ref.float().reshape(-1), dim=0).item(),
            0.999999,
        )
        self.assertTrue(torch.equal(cg_toks, post_tok_ref))

        # Kernel launch count guard for R6 short-ctx paths
        def _count_cuda_kernels(fn):
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
            ) as p:
                fn()
                torch.cuda.synchronize()
            return sum(
                1 for e in p.events()
                if e.device_type == torch.profiler.DeviceType.CUDA and getattr(e, "device_time_total", 0) > 0
            )

        self.assertLessEqual(_count_cuda_kernels(lambda: m_fast.forward_commit(commit_in)), 12)
        self.assertEqual(_count_cuda_kernels(lambda: m_fast.prepare_fmha_impl(propose_in)), 0)
        self.assertLessEqual(
            _count_cuda_kernels(lambda: m_fast.propose_with_candidate_selector(h_fast, anchor_tok)), 12
        )

        # Free BF16 models + CG pool before creating FP8 models so peak GPU memory stays small
        del m_fast, m_ref, g_r6, cg_commit, cg_prop, cg_toks, post_commit_ref, post_prop_ref, post_tok_ref
        torch.cuda.empty_cache()

        # FP8-KV commit parity
        kv_fp8_init = [(t / 0.05).clamp(-448.0, 448.0).to(torch.float8_e4m3fn) for t in kv_init]
        m_fp8_fast = _make_m("fp8_e4m3", [t.clone() for t in kv_fp8_init])
        m_fp8_ref = _make_m("fp8_e4m3", [t.clone() for t in kv_fp8_init])
        m_fp8_fast.forward_commit(commit_in)
        DSparkProposerMixin.run_commit_step(m_fp8_ref, commit_in, device)
        for l_idx in range(spec.num_hidden_layers):
            fp8_cos = F.cosine_similarity(
                m_fp8_fast.kv_cache._layers[l_idx].kv_cache_base.float().reshape(-1),
                m_fp8_ref.kv_cache._layers[l_idx].kv_cache_base.float().reshape(-1),
                dim=0,
            ).item()
            self.assertGreaterEqual(fp8_cos, 0.9999)

        # 5) Verify live _launch_triton_cached cache hits across BF16 + FP8-KV single-kernel paths
        m_fp8_fast.forward_propose(propose_in, m_fp8_fast.prepare_fmha_impl(propose_in))
        for kc_name in (
            "_KC_RMSNORM",
            "_KC_MAP_COMMIT",
            "_KC_FUSED_5L_KV",
            "_KC_PREP_PROPOSE",
            "_KC_K_ROPE_SCATTER",
            "_KC_ROPE_Q",
            "_KC_MHA_SINGLE",
            "_KC_MHA_BODY",
            "_KC_MHA_EDGE",
            "_KC_MHA_STAGE2",
            "_KC_GCONV",
            "_KC_SCORE_EDGES",
            "_KC_GREEDY_WALK",
        ):
            self.assertGreater(len(getattr(dm, kc_name)), 0, f"{kc_name} stayed empty (cache inactive)")

        # 6) Regression test for _KC_FUSED_5L_KV pointer alignment specialization across all 5 layers & scales:
        # Reuse m_fp8_fast / m_fp8_ref by swapping kv_cache to avoid extra model memory pressure.
        dm._KC_FUSED_5L_KV.clear()
        n_kv_elems = kv_fp8_init[0].numel()
        unaligned_raw = torch.empty((n_kv_elems + 16,), dtype=torch.uint8, device=device)
        unaligned_kv2 = unaligned_raw[4 : 4 + n_kv_elems].view(torch.float8_e4m3fn).view(kv_fp8_init[2].shape)
        self.assertNotEqual(int(unaligned_kv2.data_ptr()) & 15, 0)
        unaligned_kv2.copy_(kv_fp8_init[2])

        # Case A (1-element scales -> ks, vs both 16B aligned + 16B aligned kv0..kv4)
        m_fp8_fast.kv_cache = _MultiLayerDictKVCache(
            [LayerKVCache(t.clone(), PS, i, "default") for i, t in enumerate(kv_fp8_init)]
        )
        m_fp8_ref.kv_cache = _MultiLayerDictKVCache(
            [LayerKVCache(t.clone(), PS, i, "default") for i, t in enumerate(kv_fp8_init)]
        )
        for i in range(spec.num_hidden_layers):
            sc1 = torch.tensor([0.05], dtype=torch.float32, device=device)
            m_fp8_fast.kv_cache._layers[i].kv_scale_base = sc1
            m_fp8_ref.kv_cache._layers[i].kv_scale_base = sc1
        m_fp8_fast.forward_commit(commit_in)
        DSparkProposerMixin.run_commit_step(m_fp8_ref, commit_in, device)
        self.assertEqual(len(dm._KC_FUSED_5L_KV), 1)
        for l_idx in range(spec.num_hidden_layers):
            self.assertGreaterEqual(
                F.cosine_similarity(
                    m_fp8_fast.kv_cache._layers[l_idx].kv_cache_base.float().reshape(-1),
                    m_fp8_ref.kv_cache._layers[l_idx].kv_cache_base.float().reshape(-1),
                    dim=0,
                ).item(),
                0.9999,
            )

        # Case B (2-element scales on layers 1..4 -> +4B unaligned vs, and +4B unaligned kv2)
        kv_b_bufs = [t.clone() for t in kv_fp8_init]
        kv_b_bufs[2] = unaligned_kv2
        m_fp8_fast.kv_cache = _MultiLayerDictKVCache(
            [LayerKVCache(b, PS, i, "default") for i, b in enumerate(kv_b_bufs)]
        )
        m_fp8_ref.kv_cache = _MultiLayerDictKVCache(
            [LayerKVCache(t.clone(), PS, i, "default") for i, t in enumerate(kv_fp8_init)]
        )
        for i in range(spec.num_hidden_layers):
            sc_i = (
                torch.tensor([0.05], dtype=torch.float32, device=device)
                if i == 0
                else torch.tensor([0.05, 0.09], dtype=torch.float32, device=device)
            )
            m_fp8_fast.kv_cache._layers[i].kv_scale_base = sc_i
            m_fp8_ref.kv_cache._layers[i].kv_scale_base = sc_i
        m_fp8_fast.forward_commit(commit_in)
        self.assertEqual(len(dm._KC_FUSED_5L_KV), 2)
        DSparkProposerMixin.run_commit_step(m_fp8_ref, commit_in, device)
        self.assertEqual(len(dm._KC_FUSED_5L_KV), 3)
        for l_idx in range(spec.num_hidden_layers):
            self.assertGreaterEqual(
                F.cosine_similarity(
                    m_fp8_fast.kv_cache._layers[l_idx].kv_cache_base.float().reshape(-1),
                    m_fp8_ref.kv_cache._layers[l_idx].kv_cache_base.float().reshape(-1),
                    dim=0,
                ).item(),
                0.9999,
            )

        del m_fp8_fast, m_fp8_ref, unaligned_raw, unaligned_kv2, kv_b_bufs
        torch.cuda.empty_cache()

        # 7) Hook fallback guard: when launch_enter_hook has active callbacks, _launch_triton_cached must invoke it
        hook_calls = [0]
        def _on_launch(*args, **kwargs):
            hook_calls[0] += 1
        dm.triton.knobs.runtime.launch_enter_hook.add(_on_launch)
        try:
            _ = dm._apply_gemma_rmsnorm(
                torch.randn((4, 256), dtype=torch.bfloat16, device=device),
                torch.randn((256,), dtype=torch.bfloat16, device=device),
                1e-6,
            )
        finally:
            dm.triton.knobs.runtime.launch_enter_hook.calls.remove(_on_launch)
        self.assertEqual(hook_calls[0], 1)

    @torch.inference_mode()
    def test_dflash2_r7_gconv_hc_fusion_ws_and_launch_reduction_gpu(self) -> None:
        """Verify R7 DFlashGroupedConv+HC fusion, HCStaticWorkspace CG/non-alias, & propose launch reduction."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        import rtp_llm.models_py.model_desc.qwen4_5_dspark_model as dm
        from rtp_llm.models_py.modules.qwen4_5.hyper_connection import HCStaticWorkspace
        import torch.nn.functional as F

        device = torch.device("cuda:0")
        _cpu_rng = torch.random.get_rng_state()
        _cuda_rng = torch.cuda.get_rng_state(device)
        self.addCleanup(torch.cuda.set_rng_state, _cuda_rng, device)
        self.addCleanup(torch.random.set_rng_state, _cpu_rng)
        torch.manual_seed(7070)
        cfg, mw = _load_dspark_model_weights(MOCK_8L_DFLASH, device=str(device))
        spec = cfg._dflash2_spec
        m = Qwen4_5DSparkModel(
            cfg, ParallelismConfig(), mw, max_generate_batch_size=8, dflash_attn_backend="triton"
        )
        m.eval()
        for lyr in m.layers:
            lyr.mlp.gate.weight.data.zero_()
            lyr.mlp.expert_bias.data.copy_(
                torch.linspace(8.0, -8.0, lyr.mlp.expert_bias.numel(), device=device)
            )
        # 1. Direct combine_dflash_gconv_and_mix_with_latent parity against finish + combine_and_mix_with_latent_ref
        lyr0 = m.layers[0]
        raw_sub = torch.randn((16, spec.hidden_size), dtype=torch.bfloat16, device=device) * 0.2
        res_t = torch.randn((16, spec.hc_count * spec.hidden_size), dtype=torch.bfloat16, device=device) * 0.2
        inj_t = torch.rand((16, spec.hc_count), dtype=torch.bfloat16, device=device) * 1.5
        prev_lat_t = torch.randn((16, lyr0.mlp_hyper_connection.hc_lowrank), dtype=torch.bfloat16, device=device) * 0.2
        pre_g, coeff_g = lyr0.attention_conv.prepare(raw_sub)
        m_f, r_f, ij_f, lat_f = lyr0.mlp_hyper_connection.combine_dflash_gconv_and_mix_with_latent(
            pre_g, coeff_g, lyr0.attention_conv.base_kernel[1],
            lyr0.attention_conv.block_size, lyr0.attention_conv.group_size, lyr0.attention_conv.taps,
            res_t, inj_t, prev_latent=prev_lat_t, clip_value=10.0, ws=HCStaticWorkspace(),
        )
        finished = lyr0.attention_conv.finish(pre_g, coeff_g, clip_value=10.0)
        m_r, r_r, ij_r, lat_r = lyr0.mlp_hyper_connection.combine_and_mix_with_latent_ref(
            finished, res_t, inj_t, prev_latent=prev_lat_t
        )
        for a, b in ((m_f, m_r), (r_f, r_r), (ij_f, ij_r), (lat_f, lat_r)):
            self.assertGreaterEqual(
                F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0).item(),
                0.9999,
            )

        # 2. Full 5L forward_propose ws vs no_ws parity, CG replay, launch reduction (<=195), and 0 cutlass_80
        ps = 16
        kv_init = [
            torch.randn((8, 2, 1, ps, spec.head_dim), dtype=torch.bfloat16, device=device) * 0.05
            for _ in range(spec.num_hidden_layers)
        ]
        m.kv_cache = _MultiLayerDictKVCache(
            [LayerKVCache(t.clone(), ps, i, "default") for i, t in enumerate(kv_init)]
        )
        bt = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device=device)
        propose_in = _make_dflash2_inputs(
            batch_size=2, prefix_lens=[3, 2], query_width=8, anchor_ids=[15, 25],
            block_table=bt, noise_token_id=cfg.dspark_noise_token_id,
            device=str(device), use_device_mirrors=True, is_target_verify=False,
        )
        fmha = m.prepare_fmha_impl(propose_in)
        m._hc_ws = HCStaticWorkspace()
        out_ws = m.forward_propose(propose_in, fmha_impl=fmha).hidden_states.clone()
        m._hc_ws = None
        out_nows = m.forward_propose(propose_in, fmha_impl=fmha).hidden_states.clone()
        self.assertEqual(float((out_ws - out_nows).abs().max().item()), 0.0)

        m._hc_ws = HCStaticWorkspace()
        _ = m.forward_propose(propose_in, fmha_impl=fmha)
        torch.cuda.synchronize()
        g_r7 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_r7):
            cg_r7 = m.forward_propose(propose_in, fmha_impl=fmha).hidden_states
        g_r7.replay()
        torch.cuda.synchronize()
        self.assertGreaterEqual(
            F.cosine_similarity(cg_r7.float().reshape(-1), out_ws.float().reshape(-1), dim=0).item(),
            0.999999,
        )

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            _ = m.forward_propose(propose_in, fmha_impl=fmha)
            torch.cuda.synchronize()
        cuda_events = [
            e for e in prof.events()
            if e.device_type == torch.profiler.DeviceType.CUDA and getattr(e, "device_time_total", 0) > 0
        ]
        self.assertLessEqual(len(cuda_events), 195, f"Expected <=195 R7 propose launches, got {len(cuda_events)}")
        # Check production H=5120 LR=128 combine_dflash_gconv_and_mix_with_latent emits zero cutlass_80
        from rtp_llm.models_py.modules.qwen4_5.hyper_connection import Qwen4_5HyperConnection
        hc_prod = Qwen4_5HyperConnection(
            hidden_size=5120, hc_count=4, hc_lowrank=128, dtype=torch.bfloat16, device=device
        ).eval()
        gconv_prod = dm.DFlashGroupedConv(hidden_size=5120, dtype=torch.bfloat16, device=device)
        p_raw = torch.randn((16, 5120), dtype=torch.bfloat16, device=device) * 0.2
        p_pre, p_coeff = gconv_prod.prepare(p_raw)
        p_res = torch.randn((16, 4 * 5120), dtype=torch.bfloat16, device=device) * 0.2
        p_inj = torch.rand((16, 4), dtype=torch.bfloat16, device=device)
        p_lat = torch.randn((16, 128), dtype=torch.bfloat16, device=device) * 0.2
        _ = hc_prod.combine_dflash_gconv_and_mix_with_latent(
            p_pre, p_coeff, gconv_prod.base_kernel[1], gconv_prod.block_size, gconv_prod.group_size,
            gconv_prod.taps, p_res, p_inj, prev_latent=p_lat, ws=HCStaticWorkspace(),
        )
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
        ) as prof_hc:
            _ = hc_prod.combine_dflash_gconv_and_mix_with_latent(
                p_pre, p_coeff, gconv_prod.base_kernel[1], gconv_prod.block_size, gconv_prod.group_size,
                gconv_prod.taps, p_res, p_inj, prev_latent=p_lat, ws=HCStaticWorkspace(),
            )
            torch.cuda.synchronize()
        cuda_names_hc = [
            str(getattr(e, "name", "")).lower()
            for e in prof_hc.events()
            if e.device_type == torch.profiler.DeviceType.CUDA and getattr(e, "device_time_total", 0) > 0
        ]
        self.assertFalse(any("cutlass_80" in nm for nm in cuda_names_hc))
        self.assertFalse(any("splitkreduce" in nm for nm in cuda_names_hc), f"Unexpected splitKreduce in {cuda_names_hc}")
        for r_cnt in (8, 16, 64, 256, 512):
            r_raw = torch.randn((r_cnt, 5120), dtype=torch.bfloat16, device=device) * 0.2
            r_pre, r_coeff = gconv_prod.prepare(r_raw)
            r_res = torch.randn((r_cnt, 4 * 5120), dtype=torch.bfloat16, device=device) * 0.2
            r_inj = torch.rand((r_cnt, 4), dtype=torch.bfloat16, device=device)
            r_lat = torch.randn((r_cnt, 128), dtype=torch.bfloat16, device=device) * 0.2
            mf, rf, ijf, ltf = hc_prod.combine_dflash_gconv_and_mix_with_latent(
                r_pre, r_coeff, gconv_prod.base_kernel[1], gconv_prod.block_size, gconv_prod.group_size,
                gconv_prod.taps, r_res, r_inj, prev_latent=r_lat, clip_value=10.0, ws=HCStaticWorkspace(),
            )
            fin = gconv_prod.finish(r_pre, r_coeff, clip_value=10.0)
            mr, rr, ijr, ltr = hc_prod.combine_and_mix_with_latent_ref(fin, r_res, r_inj, prev_latent=r_lat)
            for a, b in ((mf, mr), (rf, rr), (ijf, ijr), (ltf, ltr)):
                self.assertGreaterEqual(
                    F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0).item(),
                    0.9999,
                )

    def test_dflash2_inherits_kv_cache_dtype_fp8_when_env_and_arg_unset(self) -> None:
        from rtp_llm.ops import KvCacheDataType

        cfg1, mw_full = _load_dspark_model_weights(MOCK_8L_DFLASH, device="cpu")
        cfg2, _ = _load_dspark_model_weights(MOCK_8L_DFLASH, device="cpu")
        pc = ParallelismConfig()
        pc.tp_size = 1
        pc.ep_size = 1
        old_env = os.environ.pop("QWEN4_5_DFLASH_KV_DTYPE", None)
        try:
            cfg1.attn_config.kv_cache_dtype = KvCacheDataType.FP8
            m_auto = Qwen4_5DSparkModel(cfg1, pc, mw_full, max_generate_batch_size=2)
            self.assertEqual(m_auto.dflash_kv_dtype, "fp8_e4m3")
            self.assertEqual(m_auto.config.attn_config.dtype, torch.float8_e4m3fn)

            os.environ["QWEN4_5_DFLASH_KV_DTYPE"] = "bf16"
            cfg2.attn_config.kv_cache_dtype = KvCacheDataType.FP8
            m_env = Qwen4_5DSparkModel(cfg2, pc, mw_full, max_generate_batch_size=2)
            self.assertEqual(m_env.dflash_kv_dtype, "bf16")
        finally:
            if old_env is not None:
                os.environ["QWEN4_5_DFLASH_KV_DTYPE"] = old_env
            else:
                os.environ.pop("QWEN4_5_DFLASH_KV_DTYPE", None)


@torch.no_grad()
def _dflash2_gloo_tp4_worker(rank: int, world_size: int, port: int, err_q) -> None:
    try:
        import torch.distributed as dist
        import torch.nn.functional as F

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(0)
        device = "cuda:0"
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        try:
            cfg, mw_full = _load_dspark_model_weights(MOCK_8L_DFLASH, device=device)
            spec = cfg._dflash2_spec

            pc = ParallelismConfig()
            pc.tp_size = world_size
            pc.tp_rank = rank
            pc.ep_size = world_size
            pc.ep_rank = rank
            pc.world_size = world_size
            pc.world_rank = rank
            model = Qwen4_5DSparkModel(cfg, pc, mw_full, max_generate_batch_size=4)
            for lyr in model.layers:
                lyr.mlp.gate.weight.data.zero_()
                lyr.mlp.expert_bias.data.copy_(torch.linspace(8.0, -8.0, lyr.mlp.expert_bias.numel(), device=device))

            page_size = 16
            kv_bufs = [torch.zeros((8, 2, 1, page_size, spec.head_dim), dtype=torch.bfloat16, device=device) for _ in range(5)]
            model.kv_cache = _MultiLayerDictKVCache([LayerKVCache(kv_bufs[i], page_size, i, "default") for i in range(5)])
            bt = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
            gen = torch.Generator(device=device).manual_seed(9191)
            aux_h = torch.randn((5, spec.fc_in_features), dtype=torch.bfloat16, device=device, generator=gen) * 0.05
            commit_in = _make_dflash2_inputs(
                batch_size=2, prefix_lens=[0, 0], query_width=8, anchor_ids=[15, 25],
                block_table=bt, noise_token_id=cfg.dspark_noise_token_id,
                input_hiddens=aux_h, device=device, use_device_mirrors=True,
                is_target_verify=True, active_query_lens=[3, 2],
            )
            model.forward_commit(commit_in)
            propose_in = _make_dflash2_inputs(
                batch_size=2, prefix_lens=[3, 2], query_width=8, anchor_ids=[15, 25],
                block_table=bt, noise_token_id=cfg.dspark_noise_token_id,
                device=device, use_device_mirrors=True, is_target_verify=False,
            )
            fmha = model.prepare_fmha_impl(propose_in)
            out_tp = model.forward_propose(propose_in, fmha_impl=fmha).hidden_states
            anchors = torch.tensor([15, 25], dtype=torch.int32, device=device)
            toks_tp, cands_tp, _ = model.propose_with_candidate_selector(out_tp, anchors)

            r0_out = out_tp.clone()
            dist.broadcast(r0_out, src=0)
            assert (out_tp.float() - r0_out.float()).abs().max().item() < 1e-4

            if rank == 0:
                ref_model = Qwen4_5DSparkModel(cfg, ParallelismConfig(), mw_full, max_generate_batch_size=4)
                for lyr in ref_model.layers:
                    lyr.mlp.gate.weight.data.zero_()
                    lyr.mlp.expert_bias.data.copy_(torch.linspace(8.0, -8.0, lyr.mlp.expert_bias.numel(), device=device))
                ref_kv = [torch.zeros((8, 2, 1, page_size, spec.head_dim), dtype=torch.bfloat16, device=device) for _ in range(5)]
                ref_model.kv_cache = _MultiLayerDictKVCache([LayerKVCache(ref_kv[i], page_size, i, "default") for i in range(5)])
                ref_model.forward_commit(commit_in)
                out_ref = ref_model.forward_propose(propose_in, fmha_impl=ref_model.prepare_fmha_impl(propose_in)).hidden_states
                cos = F.cosine_similarity(out_tp.float(), out_ref.float(), dim=-1).min().item()
                assert cos > 0.99, f"DFlash2 Gloo TP=4+EP=4 vs TP=1 cos={cos}"
                assert toks_tp.shape == (2, 8) and cands_tp.shape == (2, 8, 16)
            dist.barrier()
        finally:
            dist.destroy_process_group()
    except BaseException as exc:
        err_q.put(f"rank {rank}: {exc!r}")
        raise

@torch.no_grad()
def _dflash2_gloo_pmode_worker(rank: int, world_size: int, port: int, pmode: str, err_q) -> None:
    try:
        import torch.distributed as dist
        import torch.nn.functional as F

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(0)
        device = "cuda:0"
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        try:
            cfg, mw_full = _load_dspark_model_weights(MOCK_8L_DFLASH, device=device)
            spec = cfg._dflash2_spec

            pc = ParallelismConfig()
            pc.world_size = world_size
            pc.world_rank = rank
            model = Qwen4_5DSparkModel(
                cfg,
                pc,
                mw_full,
                max_generate_batch_size=4,
                parallel_mode=pmode,
                dp_equal_tokens=True,
            )
            assert model.parallel_mode == pmode
            if pmode == "dp4_ep4":
                assert model.dp_size == 4 and model.tp_size == 1 and model.eff_ep_size == 4
                assert not model.lm_head_sharded and model.all_to_all_comm is not None
            elif pmode == "dp2_tp2_ep4":
                assert model.dp_size == 2 and model.tp_size == 2 and model.eff_ep_size == 4
                assert model.lm_head_sharded and model.all_to_all_comm is not None
                assert model.fast_all_reduce is not None and model.fast_all_reduce.group is model.tp_group
            elif pmode == "sp4_tp4_ep4":
                assert model.dp_size == 1 and model.tp_size == 4 and model.sp_size == 4 and model.eff_ep_size == 4
                assert not model.dp_attn and model.sp_rank == model.tp_rank == rank
                assert model.lm_head_sharded and model.all_to_all_comm is None
                assert model.fast_all_reduce is not None and model.fast_all_reduce.group is model.tp_group
            elif pmode == "dp2_tpsp2_ep4":
                assert model.dp_size == 2 and model.tp_size == 2 and model.sp_size == 2 and model.eff_ep_size == 4
                assert not model.dp_attn and model.dp_rank == (rank // 2) and model.tp_rank == (rank % 2)
                assert model.sp_rank == model.tp_rank
                assert model.lm_head_sharded and model.all_to_all_comm is not None
                assert model.fast_all_reduce is not None and model.fast_all_reduce.group is model.tp_group
                assert dist.get_world_size(model.tp_group) == 2

            for lyr in model.layers:
                lyr.mlp.gate.weight.data.zero_()
                lyr.mlp.expert_bias.data.copy_(torch.linspace(8.0, -8.0, lyr.mlp.expert_bias.numel(), device=device))

            page_size = 16
            kv_bufs = [torch.zeros((8, 2, 1, page_size, spec.head_dim), dtype=torch.bfloat16, device=device) for _ in range(5)]
            model.kv_cache = _MultiLayerDictKVCache([LayerKVCache(kv_bufs[i], page_size, i, "default") for i in range(5)])

            bt = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
            gen = torch.Generator(device=device).manual_seed(9191 + model.dp_rank * 77)
            aux_h = torch.randn((5, spec.fc_in_features), dtype=torch.bfloat16, device=device, generator=gen) * 0.05
            a0 = 15 + model.dp_rank * 100
            a1 = 25 + model.dp_rank * 100
            commit_in = _make_dflash2_inputs(
                batch_size=2, prefix_lens=[0, 0], query_width=8, anchor_ids=[a0, a1],
                block_table=bt, noise_token_id=cfg.dspark_noise_token_id,
                input_hiddens=aux_h, device=device, use_device_mirrors=True,
                is_target_verify=True, active_query_lens=[3, 2],
            )
            model.forward_commit(commit_in)
            propose_in = _make_dflash2_inputs(
                batch_size=2, prefix_lens=[3, 2], query_width=8, anchor_ids=[a0, a1],
                block_table=bt, noise_token_id=cfg.dspark_noise_token_id,
                device=device, use_device_mirrors=True, is_target_verify=False,
            )
            fmha = model.prepare_fmha_impl(propose_in)
            out_par = model.forward_propose(propose_in, fmha_impl=fmha).hidden_states
            anchors = torch.tensor([a0, a1], dtype=torch.int32, device=device)
            toks_par, cands_par, _ = model.propose_with_candidate_selector(out_par, anchors)

            ref_model = Qwen4_5DSparkModel(cfg, ParallelismConfig(), mw_full, max_generate_batch_size=4)
            for lyr in ref_model.layers:
                lyr.mlp.gate.weight.data.zero_()
                lyr.mlp.expert_bias.data.copy_(torch.linspace(8.0, -8.0, lyr.mlp.expert_bias.numel(), device=device))
            ref_kv = [torch.zeros((8, 2, 1, page_size, spec.head_dim), dtype=torch.bfloat16, device=device) for _ in range(5)]
            ref_model.kv_cache = _MultiLayerDictKVCache([LayerKVCache(ref_kv[i], page_size, i, "default") for i in range(5)])
            ref_model.forward_commit(commit_in)
            out_ref = ref_model.forward_propose(propose_in, fmha_impl=ref_model.prepare_fmha_impl(propose_in)).hidden_states
            cos = F.cosine_similarity(out_par.float(), out_ref.float(), dim=-1).min().item()
            assert cos > 0.995, f"DFlash2 {pmode} rank={rank} vs ref cos={cos}"
            assert toks_par.shape == (2, 8) and cands_par.shape == (2, 8, 16)
            lg_par = model.compute_logits(out_par)
            lg_ref = ref_model.compute_logits(out_ref)
            lg_cos = F.cosine_similarity(lg_par.float().reshape(-1), lg_ref.float().reshape(-1), dim=0).item()
            assert lg_cos > 0.995, f"DFlash2 {pmode} logits all_gather rank={rank} cos={lg_cos}"

            # R3: verify 1 packed all_gather in vocab_parallel_topk and DFlash2 forward_propose(enable_tbo=True) parity
            ag_cnt = [0]
            orig_tp_ag = model._tp_all_gather_lastdim
            def _wrap_ag(x, _orig=orig_tp_ag):
                ag_cnt[0] += 1
                return _orig(x)
            model._tp_all_gather_lastdim = _wrap_ag
            try:
                vk_vals, vk_ids = model.vocab_parallel_topk(out_par, k=spec.selector_top_k)
            finally:
                model._tp_all_gather_lastdim = orig_tp_ag
            expected_ag = 1 if (model.tp_size > 1 and model.lm_head_sharded) else 0
            assert ag_cnt[0] == expected_ag, f"DFlash2 {pmode} expected {expected_ag} packed all_gather, got {ag_cnt[0]}"
            assert vk_vals.shape == (out_par.shape[0], spec.selector_top_k) and vk_ids.shape == vk_vals.shape
            out_tbo = model.forward_propose(propose_in, fmha_impl=fmha, enable_tbo=True, tbo_equal_tokens=True).hidden_states
            tbo_cos = F.cosine_similarity(out_tbo.float(), out_par.float(), dim=-1).min().item()
            assert tbo_cos > 0.999, f"DFlash2 {pmode} enable_tbo=True rank={rank} vs eager cos={tbo_cos}"

            # Also verify Triton DFlash2 backend (and unequal DP ragged batches for dp2_tpsp2_ep4)
            if pmode in ("dp2_tp2_ep4", "dp2_tpsp2_ep4"):
                del model
                torch.cuda.empty_cache()
                m_tri = Qwen4_5DSparkModel(
                    cfg, pc, mw_full, max_generate_batch_size=4,
                    parallel_mode=pmode, dp_equal_tokens=True, dflash_attn_backend="triton",
                )
                for lyr in m_tri.layers:
                    lyr.mlp.gate.weight.data.zero_()
                    lyr.mlp.expert_bias.data.copy_(torch.linspace(8.0, -8.0, lyr.mlp.expert_bias.numel(), device=device))
                tri_kv = [torch.zeros((8, 2, 1, page_size, spec.head_dim), dtype=torch.bfloat16, device=device) for _ in range(5)]
                m_tri.kv_cache = _MultiLayerDictKVCache([LayerKVCache(tri_kv[i], page_size, i, "default") for i in range(5)])
                m_tri.forward_commit(commit_in)
                out_tri = m_tri.forward_propose(propose_in, fmha_impl=m_tri.prepare_fmha_impl(propose_in)).hidden_states
                cos_tri_par = F.cosine_similarity(out_tri.float(), out_par.float(), dim=-1).min().item()
                cos_tri_ref = F.cosine_similarity(out_tri.float(), out_ref.float(), dim=-1).min().item()
                assert cos_tri_par > 0.99, f"DFlash2 Triton vs FI {pmode} rank={rank} cos={cos_tri_par}"
                assert cos_tri_ref > 0.99, f"DFlash2 Triton vs ref {pmode} rank={rank} cos={cos_tri_ref}"
                if pmode == "dp2_tpsp2_ep4":
                    del m_tri
                    torch.cuda.empty_cache()
                    m_unbal = Qwen4_5DSparkModel(
                        cfg, pc, mw_full, max_generate_batch_size=4,
                        parallel_mode=pmode, dp_equal_tokens=None,
                    )
                    assert m_unbal.all_to_all_comm is not None
                    for lyr in m_unbal.layers:
                        lyr.mlp.gate.weight.data.zero_()
                        lyr.mlp.expert_bias.data.copy_(torch.linspace(8.0, -8.0, lyr.mlp.expert_bias.numel(), device=device))
                    unbal_kv = [torch.zeros((8, 2, 1, page_size, spec.head_dim), dtype=torch.bfloat16, device=device) for _ in range(5)]
                    m_unbal.kv_cache = _MultiLayerDictKVCache([LayerKVCache(unbal_kv[i], page_size, i, "default") for i in range(5)])
                    loc_bs = 2 if m_unbal.dp_rank == 0 else 1
                    loc_pref = [3, 2][:loc_bs]
                    loc_anch = [15 + m_unbal.dp_rank * 100, 25 + m_unbal.dp_rank * 100][:loc_bs]
                    loc_bt = bt[:loc_bs]
                    loc_aux = aux_h if m_unbal.dp_rank == 0 else aux_h[:3]
                    c_in = _make_dflash2_inputs(
                        batch_size=loc_bs, prefix_lens=[0] * loc_bs, query_width=8, anchor_ids=loc_anch,
                        block_table=loc_bt, noise_token_id=cfg.dspark_noise_token_id,
                        input_hiddens=loc_aux, device=device, use_device_mirrors=True,
                        is_target_verify=True, active_query_lens=loc_pref,
                    )
                    m_unbal.forward_commit(c_in)
                    p_in = _make_dflash2_inputs(
                        batch_size=loc_bs, prefix_lens=loc_pref, query_width=8, anchor_ids=loc_anch,
                        block_table=loc_bt, noise_token_id=cfg.dspark_noise_token_id,
                        device=device, use_device_mirrors=True, is_target_verify=False,
                    )
                    out_u = m_unbal.forward_propose(p_in, fmha_impl=m_unbal.prepare_fmha_impl(p_in)).hidden_states
                    ref_kv2 = [torch.zeros((8, 2, 1, page_size, spec.head_dim), dtype=torch.bfloat16, device=device) for _ in range(5)]
                    ref_model.kv_cache = _MultiLayerDictKVCache([LayerKVCache(ref_kv2[i], page_size, i, "default") for i in range(5)])
                    ref_model.forward_commit(c_in)
                    out_ref_u = ref_model.forward_propose(p_in, fmha_impl=ref_model.prepare_fmha_impl(p_in)).hidden_states
                    cos_u = F.cosine_similarity(out_u.float(), out_ref_u.float(), dim=-1).min().item()
                    assert cos_u > 0.995, f"DFlash2 {pmode} unequal_dp(bs={loc_bs}) rank={rank} cos={cos_u}"
            dist.barrier()
        finally:
            dist.destroy_process_group()
    except BaseException as exc:
        err_q.put(f"rank {rank} ({pmode}): {exc!r}")
        raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
