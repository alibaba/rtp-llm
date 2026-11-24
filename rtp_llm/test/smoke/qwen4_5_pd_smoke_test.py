"""Pure PD-disaggregated (PREFILL -> CacheStore RDMA -> DECODE) Bazel smoke test for Qwen 4.5-Flash.

Supports two modes via ``QWEN4_5_PD_SMOKE_MODE``:
  - ``synthetic_8layer`` (default, 2 GPUs: 1P + 1D, TP=1): builds a deterministic 8-layer
    Qwen 4.5-Flash safetensors checkpoint + HF Qwen2Tokenizer in a TemporaryDirectory,
    launches pure-PD servers with RDMA transport + Prefill Device/Memory Cache, and verifies
    prefix branching, exact page boundaries, barrier-synchronized mixed-reuse Prefill batches,
    staggered continuous-batching Decode, and (when ``QWEN4_5_FORCE_HOST_DEMOTION=1``)
    Device-to-Host eviction + Host-to-Device promotion + mixed Host/Cold batches.
  - ``real_3fs_40layer`` (8 GPUs: 4P + 4D, TP=4/EP=4): reads the real 40-layer 3FS
    checkpoint strictly read-only and verifies RDMA, Prefill Device/Memory Cache, exact
    page boundaries, prefix branching, barrier-synchronized mixed-reuse Prefill batch,
    and staggered continuous-batching Decode with grounded coherent outputs.

Pure PD only: unified / PDFUSION serving mode is intentionally not included.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import logging
import os
import re
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import unittest

import requests
import torch
from transformers import AutoTokenizer

from rtp_llm.test.smoke.qwen4_5_pd_server import (
    DEFAULT_3FS_CKPT,
    build_pure_pd_managers,
    select_single_layer_rdma_hcas,
    start_pure_pd_and_wait_rpc,
)
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


@dataclass(frozen=True)
class RequestSpec:
    """Single PD generation request specification."""

    name: str
    prompt: str
    max_new_tokens: int
    reuse_cache: bool = True
    expected_substr: Optional[str] = None
    expected_input_len: Optional[int] = None
    return_logits: bool = False


def _write_synthetic_qwen2_tokenizer(ckpt_dir: Path, vocab_size: int) -> None:
    """Write a minimal valid HF Qwen2Tokenizer directory matching ``vocab_size``."""
    vocab: Dict[str, int] = {
        "<|pad|>": 0,
        "<|im_start|>": 1,
        "<|endoftext|>": 2,
        "<|im_end|>": 3,
    }
    next_id = 4
    for cp in range(32, 127):
        ch = chr(cp)
        tok = "\u0120" if ch == " " else ch
        if tok not in vocab and next_id < vocab_size:
            vocab[tok] = next_id
            next_id += 1
    while next_id < vocab_size:
        vocab[f"tok_{next_id}"] = next_id
        next_id += 1

    (ckpt_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
    )
    (ckpt_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (ckpt_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "Qwen2Tokenizer",
                "bos_token": "<|im_start|>",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|pad|>",
                "unk_token": "<|endoftext|>",
                "model_max_length": 4096,
                "clean_up_tokenization_spaces": False,
                "added_tokens_decoder": {
                    "0": {"content": "<|pad|>", "special": True},
                    "1": {"content": "<|im_start|>", "special": True},
                    "2": {"content": "<|endoftext|>", "special": True},
                    "3": {"content": "<|im_end|>", "special": True},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (ckpt_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": "<|im_start|>",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|pad|>",
                "unk_token": "<|endoftext|>",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _build_synthetic_8layer_checkpoint(ckpt_dir: Path) -> Path:
    """Build a deterministic 8-layer Qwen 4.5-Flash (MXFP4 + FP8 PLE + YoCo) checkpoint."""
    from rtp_llm.models_py.model_desc.qwen4_5 import Qwen4_5ForCausalLM
    from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import (
        save_qwen4_5_safetensors_checkpoint,
    )
    from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig

    cfg = Qwen4_5ArchConfig(
        vocab_size=2048,
        hidden_size=512,
        num_hidden_layers=8,
        intermediate_size=1024,
        first_k_dense_replace=1,
        rms_norm_eps=1e-6,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        max_position_embeddings=256,
        full_attention_interval=4,
        num_attention_heads=8,
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
        num_experts=32,
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
        ple_embed_dim=512,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=4,
        ngram_vocab_size_base=512,
        make_ngram_vocab_size_divisible_by=64,
        split_ngram_parts=8,
        ple_quant_scheme="fp8_pertensor",
        share_map="4->8,5->6,5->7",
        enable_yoco_decode=True,
        quant_method="mxfp4",
    )
    with torch.no_grad():
        model = Qwen4_5ForCausalLM(
            cfg,
            use_triton_mxfp4=False,
            compute_mode="bf16",
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        model.init_synthetic_weights(seed=2027, ple_mode="gpu_full")
        # Match test_e2e_qwen4_5_model._stabilize_router_margins so isotropic random
        # synthetic router logits do not flip top-k experts on 1-ULP BF16 B=1 vs B>1 GEMM rounding.
        for layer in model.model.layers:
            if layer.is_moe and layer.mlp.expert_bias is not None:
                layer.mlp.expert_bias.zero_()
                n_exp = layer.mlp.num_experts
                shard = n_exp // 4
                for r in range(4):
                    e_id = r * shard + (layer.layer_idx % shard)
                    layer.mlp.expert_bias[e_id] = 4.0
        # Suppress special tokens 0..3 and background vocab rows (0.02x), while
        # setting 8 lossless single-letter ASCII tokens ('A','D','G','J','M','P','S','V',
        # IDs 37..58 in _write_synthetic_qwen2_tokenizer) to mutually orthogonal QR
        # rows so isotropic random lm_head ties do not flip under normal BF16
        # B=1 vs B>1 rounding and Decode outputs round-trip losslessly.
        gen = torch.Generator().manual_seed(2028)
        q_mat, _ = torch.linalg.qr(
            torch.randn((cfg.hidden_size, 8), generator=gen, dtype=torch.float32)
        )
        active_ids = [37, 40, 43, 46, 49, 52, 55, 58]
        model.lm_head.weight.mul_(0.02)
        model.lm_head.weight[:4].mul_(0.05)
        model.lm_head.weight[active_ids] = (q_mat.T * 2.5).to(torch.bfloat16)

    save_qwen4_5_safetensors_checkpoint(model, ckpt_dir)
    _write_synthetic_qwen2_tokenizer(ckpt_dir, cfg.vocab_size)
    return ckpt_dir


def _prefill_cache_tiers(aux: Dict[str, Any]) -> Dict[str, int]:
    """Validate and decompose Prefill cache-tier reuse counters (from feat/k3.1_dev)."""
    keys = (
        "prefill_total_reuse_len",
        "prefill_local_reuse_len",
        "prefill_memory_reuse_len",
        "prefill_disk_reuse_len",
    )
    missing = [key for key in keys if key not in aux]
    if missing:
        raise AssertionError(
            f"Response aux_info missing Prefill cache-tier keys {missing}: {aux}"
        )
    total, local, memory, disk = (int(aux[key]) for key in keys)
    device = local - memory - disk
    if min(total, local, memory, disk, device) < 0 or local > total:
        raise AssertionError(
            f"Inconsistent Prefill cache-tier counters: "
            f"total={total} local={local} device={device} memory={memory} disk={disk}"
        )
    return {
        "total": total,
        "local": local,
        "device": device,
        "memory": memory,
        "disk": disk,
    }


def _fit_exact_token_prompt(
    tokenizer: Any,
    prefix: str,
    suffix: str,
    target_tokens: int,
) -> Tuple[str, List[int]]:
    """Construct ``prefix + padding + suffix`` with exact ``target_tokens`` IDs."""

    def _encode(text: str) -> List[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    base_ids = _encode(prefix + suffix)
    if len(base_ids) > target_tokens:
        raise ValueError(
            f"prefix+suffix already exceeds target_tokens={target_tokens}: len={len(base_ids)}"
        )
    if len(base_ids) == target_tokens:
        return prefix + suffix, base_ids

    # Pick a bulk filler whose length advances smoothly under either char-BPE or Qwen BPE.
    bulk_unit = " x"
    for cand in (" x", " y", "a", " b"):
        l1 = len(_encode(prefix + cand + suffix))
        l8 = len(_encode(prefix + (cand * 8) + suffix))
        if l8 > l1 >= len(base_ids):
            bulk_unit = cand
            break

    lo, hi = 0, target_tokens + 16
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(_encode(prefix + (bulk_unit * mid) + suffix)) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1

    pad = bulk_unit * lo
    ids = _encode(prefix + pad + suffix)
    fine_tokens = (
        " a",
        " b",
        " 1",
        " .",
        " ,",
        " x",
        " y",
        " z",
        "!",
        "?",
        "0",
        "9",
        "|",
        "a",
        "b",
        "c",
    )
    while len(ids) < target_tokens:
        best_pad: Optional[str] = None
        best_len = len(ids)
        for step in fine_tokens:
            cand_pad = pad + step
            cand_len = len(_encode(prefix + cand_pad + suffix))
            if cand_len == target_tokens:
                cand_text = prefix + cand_pad + suffix
                return cand_text, _encode(cand_text)
            if best_len < cand_len < target_tokens:
                best_pad = cand_pad
                best_len = cand_len
        if best_pad is None:
            raise AssertionError(
                f"Failed to advance token length cleanly from {len(ids)} toward {target_tokens}"
            )
        pad = best_pad
        ids = _encode(prefix + pad + suffix)

    prompt = prefix + pad + suffix
    ids = _encode(prompt)
    if len(ids) != target_tokens:
        raise AssertionError(f"Expected {target_tokens} tokens, got {len(ids)}")
    return prompt, ids


def _dump_manager_logs(managers: Sequence[MagaServerManager]) -> None:
    for mgr in managers:
        log_path = getattr(mgr, "log_file_path", None)
        if log_path and Path(log_path).exists():
            tail = Path(log_path).read_text(errors="replace")[-6000:]
            logging.error("=== Tail of %s ===\n%s", log_path, tail)


def _emit(msg: str, *args: Any) -> None:
    formatted = msg % args if args else msg
    print(formatted, flush=True)
    logging.info("%s", formatted)


def _output_ids(resp: Dict[str, Any]) -> List[int]:
    ids = resp.get("output_ids") or []
    return list(ids[0]) if ids and isinstance(ids[0], list) else list(ids)


def _input_ids(resp: Dict[str, Any]) -> List[int]:
    ids = resp.get("input_ids") or []
    return list(ids[0]) if ids and isinstance(ids[0], list) else list(ids)


def _final_logits(resp: Dict[str, Any]) -> torch.Tensor:
    logits = resp.get("logits")
    if not logits:
        raise AssertionError(f"Expected non-empty logits in response, got keys={list(resp.keys())}")
    vec = logits[0] if isinstance(logits[0], list) else logits
    return torch.tensor(vec, dtype=torch.float32).reshape(-1)


class Qwen4_5PurePDSmokeTest(unittest.TestCase):
    """Pure PD-disaggregated smoke test (ROLE_TYPE=PREFILL + ROLE_TYPE=DECODE only)."""

    def setUp(self) -> None:
        self.mode = os.environ.get("QWEN4_5_PD_SMOKE_MODE", "synthetic_8layer").strip()
        self.force_host_demotion = (
            os.environ.get("QWEN4_5_FORCE_HOST_DEMOTION", "0").strip() == "1"
        )
        self.enable_decode_cuda_graph = (
            os.environ.get("QWEN4_5_ENABLE_DECODE_CUDA_GRAPH", "1").strip() == "1"
        )
        self.max_context_batch_size = 8
        self.concurrency_limit = 16
        self._tmpdir: Optional[tempfile.TemporaryDirectory[str]] = None

        if self.mode == "synthetic_8layer":
            self.role_size = 1
            self.tp_size = 1
            self.ep_size = 1
            self.seq_size_per_block = 16
            self.max_seq_len = 256
            self.reserve_mem_mb = 1024
            self.startup_timeout_s = 240
            self.memory_cache_size_mb = int(
                os.environ.get("QWEN4_5_MEMORY_CACHE_SIZE_MB", "1024")
            )
            self._tmpdir = tempfile.TemporaryDirectory(prefix="qwen45_pd_8l_")
            self.addCleanup(self._tmpdir.cleanup)
            self.ckpt_path = str(
                _build_synthetic_8layer_checkpoint(Path(self._tmpdir.name))
            )
        elif self.mode == "real_3fs_40layer":
            self.role_size = 4
            self.tp_size = 4
            self.ep_size = 4
            self.seq_size_per_block = 64
            self.max_seq_len = 2048
            self.reserve_mem_mb = 16384
            self.startup_timeout_s = 1800
            self.memory_cache_size_mb = int(
                os.environ.get("QWEN4_5_MEMORY_CACHE_SIZE_MB", "32768")
            )
            self.ckpt_path = os.environ.get("QWEN4_5_CKPT_PATH", DEFAULT_3FS_CKPT)
            if not Path(self.ckpt_path).is_dir():
                self.fail(f"Real 3FS checkpoint not found: {self.ckpt_path}")
        else:
            self.fail(f"Unsupported QWEN4_5_PD_SMOKE_MODE={self.mode!r}")

        raw_kv_mem = os.environ.get("QWEN4_5_KV_CACHE_MEM_MB", "").strip()
        self.kv_cache_mem_mb = int(raw_kv_mem) if raw_kv_mem else None

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.ckpt_path,
            trust_remote_code=True,
        )
        self.rdma_hcas, self.rdma_link_layer = select_single_layer_rdma_hcas()
        _emit(
            "Selected RDMA HCAs (%s): %s | host_demotion=%s | decode_cuda_graph=%s | mem_cache_mb=%d",
            self.rdma_link_layer,
            self.rdma_hcas,
            self.force_host_demotion,
            self.enable_decode_cuda_graph,
            self.memory_cache_size_mb,
        )
        self.prefill_mgr, self.decode_mgr = self._start_pd_servers()

    def _start_pd_servers(self) -> Tuple[MagaServerManager, MagaServerManager]:
        cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cuda_env:
            gpu_ids = [g.strip() for g in cuda_env.split(",") if g.strip()]
        else:
            gpu_ids = [str(g) for g in get_gpu_ids()]
        need_gpus = 2 * self.role_size
        self.assertGreaterEqual(
            len(gpu_ids),
            need_gpus,
            f"Pure PD mode {self.mode!r} needs {need_gpus} GPUs, found {gpu_ids}",
        )

        prefill_port = int(MagaServerManager.get_free_port())
        decode_port = int(MagaServerManager.get_free_port())
        prefill_mgr, decode_mgr = build_pure_pd_managers(
            service_id=f"qwen4_5_pd_{self.mode}",
            prefill_gpus=gpu_ids[: self.role_size],
            decode_gpus=gpu_ids[self.role_size : 2 * self.role_size],
            prefill_port=prefill_port,
            decode_port=decode_port,
            tp_size=self.tp_size,
            ep_size=self.ep_size,
            seq_size_per_block=self.seq_size_per_block,
            max_seq_len=self.max_seq_len,
            reserver_runtime_mem_mb=self.reserve_mem_mb,
            load_workers=int(os.environ.get("QWEN4_5_LOAD_WORKERS", "8")),
            rdma_hcas=self.rdma_hcas,
            memory_cache_size_mb=self.memory_cache_size_mb,
            kv_cache_mem_mb=self.kv_cache_mem_mb,
            max_context_batch_size=self.max_context_batch_size,
            concurrency_limit=self.concurrency_limit,
            enable_decode_cuda_graph=self.enable_decode_cuda_graph,
            force_host_demotion=self.force_host_demotion,
            role_name_suffix=self.mode,
            extra_common_env={"FT_SERVER_TEST": "1"},
        )
        self.addCleanup(prefill_mgr.stop_server)
        self.addCleanup(decode_mgr.stop_server)

        try:
            start_pure_pd_and_wait_rpc(
                prefill_mgr,
                decode_mgr,
                ckpt_path=self.ckpt_path,
                role_size=self.role_size,
                startup_timeout_s=self.startup_timeout_s,
            )
        except Exception:
            _dump_manager_logs((prefill_mgr, decode_mgr))
            raise

        return prefill_mgr, decode_mgr

    def _post_generate(
        self,
        spec: RequestSpec,
        *,
        barrier: Optional[threading.Barrier] = None,
    ) -> Dict[str, Any]:
        payload = {
            "prompt": spec.prompt,
            "generate_config": {
                "max_new_tokens": spec.max_new_tokens,
                "top_k": 1,
                "top_p": 1.0,
                "temperature": 0.0,
                "random_seed": 42,
                "reuse_cache": spec.reuse_cache,
                "can_use_pd_separation": True,
                "return_input_ids": True,
                "return_output_ids": True,
                "return_logits": spec.return_logits,
                "aux_info": True,
            },
        }
        # Per-call Session prevents urllib3 pool locking across synchronized concurrent bursts.
        with requests.Session() as sess:
            sess.trust_env = False
            if barrier is not None:
                barrier.wait(timeout=30.0)
            try:
                resp = sess.post(
                    f"http://127.0.0.1:{self.prefill_mgr.port}/",
                    json=payload,
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                _dump_manager_logs((self.prefill_mgr, self.decode_mgr))
                raise

        self.assertNotIn("error_code", data, f"[{spec.name}] Unexpected PD error: {data}")
        self.assertIs(data.get("finished"), True, f"[{spec.name}] Did not finish: {data}")
        out_ids = _output_ids(data)
        self.assertGreater(len(out_ids), 0, f"[{spec.name}] Empty output_ids: {data}")
        in_ids = _input_ids(data)
        if spec.expected_input_len is not None:
            self.assertEqual(
                len(in_ids),
                spec.expected_input_len,
                f"[{spec.name}] Expected input_len={spec.expected_input_len}, got {len(in_ids)}",
            )
        if spec.expected_substr is not None:
            text = data.get("response", "")
            self.assertIn(
                spec.expected_substr,
                text,
                f"[{spec.name}] Expected {spec.expected_substr!r} in response {text!r}",
            )
        return data

    def _run_concurrent_barrier(self, specs: Sequence[RequestSpec]) -> List[Dict[str, Any]]:
        """Fire ``specs`` behind a shared Barrier so they pack into the same batch."""
        barrier = threading.Barrier(len(specs))
        with ThreadPoolExecutor(max_workers=len(specs)) as executor:
            futs = [
                executor.submit(self._post_generate, spec, barrier=barrier)
                for spec in specs
            ]
            return [fut.result() for fut in futs]

    def _run_staggered_decode_batch(
        self,
        early_specs: Sequence[RequestSpec],
        late_specs: Sequence[RequestSpec],
        stagger_s: float,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Overlap two early long-decode requests with late shorter requests."""
        barrier_early = threading.Barrier(len(early_specs))
        barrier_late = threading.Barrier(len(late_specs))
        with ThreadPoolExecutor(max_workers=len(early_specs) + len(late_specs)) as executor:
            early_futs = [
                executor.submit(self._post_generate, spec, barrier=barrier_early)
                for spec in early_specs
            ]
            time.sleep(stagger_s)
            late_futs = [
                executor.submit(self._post_generate, spec, barrier=barrier_late)
                for spec in late_specs
            ]
            return (
                [fut.result() for fut in early_futs],
                [fut.result() for fut in late_futs],
            )

    def _logs_root_dir(self) -> Optional[Path]:
        if self.prefill_mgr.log_file_path:
            candidate = Path(self.prefill_mgr.log_file_path).parent.parent / "logs"
            if candidate.is_dir():
                return candidate
        return None

    def _snapshot_batch_schedule_counts(self) -> Dict[Path, int]:
        logs_dir = self._logs_root_dir()
        if logs_dir is None:
            return {}
        # Wait briefly so ExecutionRecorder / async alog flushes prior steps.
        time.sleep(0.35)
        counts: Dict[Path, int] = {}
        for path in sorted(logs_dir.glob("batch_schedule*.log")):
            try:
                counts[path] = len(path.read_text(errors="replace").splitlines())
            except OSError:
                counts[path] = 0
        return counts

    def _collect_new_batch_sizes(
        self,
        start_counts: Dict[Path, int],
        phase: str,
        *,
        expected_min_bs: int = 1,
        timeout_s: float = 1.5,
    ) -> List[int]:
        logs_dir = self._logs_root_dir()
        if logs_dir is None:
            return []

        def _read_sizes() -> List[int]:
            sizes: List[int] = []
            for path in sorted(logs_dir.glob("batch_schedule*.log")):
                lines = path.read_text(errors="replace").splitlines()
                start = start_counts.get(path, 0)
                for line in lines[start:]:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    reqs = rec.get("requests") or []
                    if reqs and all(r.get("phase") == phase for r in reqs):
                        sizes.append(len(reqs))
            return sizes

        deadline = time.monotonic() + timeout_s
        last: List[int] = []
        stable_hits = 0
        while True:
            time.sleep(0.15)
            cur = _read_sizes()
            if cur == last:
                stable_hits += 1
            else:
                last = cur
                stable_hits = 0
            if (cur and max(cur) >= expected_min_bs) or (
                cur and stable_hits >= 2
            ) or time.monotonic() >= deadline:
                return cur

    def _assert_rdma_active_in_logs(self) -> None:
        """Verify RDMA CacheStore is active in both roles and never fell back to TCP."""
        role_logs = []
        for role_name, mgr in (("prefill", self.prefill_mgr), ("decode", self.decode_mgr)):
            path = Path(mgr.log_file_path) if mgr.log_file_path else None
            self.assertIsNotNone(path, f"{role_name} log_file_path missing")
            assert path is not None
            self.assertTrue(path.exists(), f"{role_name} process.log missing: {path}")
            role_logs.append((role_name, path.read_text(errors="replace")))

        logs_dir = self._logs_root_dir()
        engine_texts: List[str] = []
        if logs_dir is not None:
            for p in sorted(logs_dir.glob("engine*.log")):
                engine_texts.append(p.read_text(errors="replace"))

        combined_text = "\n".join([t for _, t in role_logs] + engine_texts)
        forbidden_markers = (
            "cache_store_rdma_mode: 0",
            "TransferBackend=kTcp",
            "NoRdmaMemoryUtilImpl",
            "tcp mode, no memory",
        )
        for marker in forbidden_markers:
            self.assertNotIn(
                marker,
                combined_text,
                f"Detected forbidden TCP / non-RDMA fallback marker {marker!r}",
            )

        for role_name, text in role_logs:
            self.assertIn(
                "cache_store_rdma_mode: 1",
                text,
                f"{role_name} process.log lacks cache_store_rdma_mode: 1",
            )

        positive_cpp_markers = (
            "rdma client init success",
            "reg user mr success",
            "rdma connection to",
            "TransferBackend=kBarexRdma",
            "[RKEY_TRACE]",
        )
        matched = [m for m in positive_cpp_markers if m in combined_text]
        self.assertTrue(
            matched,
            f"No C++ RDMA initialization/transfer marker found (expected one of {positive_cpp_markers})",
        )
        _emit(
            "[RDMA verified] HCAs=%s (%s) matched_markers=%s",
            self.rdma_hcas,
            self.rdma_link_layer,
            matched,
        )

    def _assert_decode_cuda_graph_active_in_logs(self) -> None:
        """Verify Decode captured and enabled CUDA Graph cleanly without errors."""
        if not self.enable_decode_cuda_graph:
            return
        path = Path(self.decode_mgr.log_file_path) if self.decode_mgr.log_file_path else None
        self.assertIsNotNone(path, "decode log_file_path missing")
        assert path is not None
        self.assertTrue(path.exists(), f"decode process.log missing: {path}")
        decode_text = path.read_text(errors="replace")
        logs_dir = self._logs_root_dir()
        engine_texts: List[str] = []
        if logs_dir is not None:
            for ep in sorted(logs_dir.glob("engine*.log")):
                engine_texts.append(ep.read_text(errors="replace"))
        combined_decode = "\n".join([decode_text] + engine_texts)

        for bad in (
            "initCapture prepare_fmha_impl failed",
            "initCapture forward for output datatype failed",
            "WarmUp forward failed",
        ):
            self.assertNotIn(
                bad,
                combined_decode,
                f"Detected Decode CUDA Graph failure marker {bad!r}",
            )
        self.assertIn(
            "CUDA graph capture is enabled",
            decode_text,
            "Decode process.log lacks CUDA graph capture enablement log",
        )
        self.assertIn(
            "Capture Decode End",
            decode_text,
            "Decode process.log did not complete Capture Decode End",
        )
        captured = sorted(
            {
                int(m)
                for m in re.findall(
                    r"(?:\[CudaGraph Memory\] captured batch size|capture success for batch size:)\s*(\d+)",
                    combined_decode,
                )
            }
        )
        self.assertTrue(
            captured,
            "No captured Decode CUDA Graph batch sizes found in Decode logs",
        )
        _emit("[Decode CUDA Graph verified] captured_batches=%s", captured)

    def _build_branch_pair(self, ns: str) -> Tuple[str, List[int], str, List[int]]:
        """Build prompts A and B sharing >= 1 full block while A spans >= 2 blocks."""
        block = self.seq_size_per_block
        if self.mode == "synthetic_8layer":
            head = "BR:"
            block2_prose = ""
            tail_a = "alpha_payload_det"
            tail_b = "beta_diverged_tail_second_block_z9"
        else:
            head = (
                f"城市档案记录 {ns}：北京位于华北平原，上海位于长江入海口，广州地处珠江三角洲，"
                "杭州以西湖景色著称，重庆是山城，西安为十三朝古都，武汉横跨长江，青岛濒临黄海，"
                "南京文化底蕴深厚，苏州园林驰名。注："
            )
            block2_prose = (
                "次级档案核对：天山南北地形多变，川西高原风光壮丽，东北平原物产丰饶，"
                "江南水乡古镇密布，东南沿海海运发达，渤海湾港口众多，河西走廊丝路绵延，"
                "珠江水系航运便利，长三角制造业密集。核对："
            )
            tail_a = (
                "Row 1: ALPHA-01 => Tag: ALPHA-01\n"
                "Row 2: OMEGA-09 => Tag: OMEGA-09\n"
                "Row 3: CEDAR-AMBER => Tag:"
            )
            tail_b = (
                "Branch B override.\n"
                "Row 1: BETA-02 => Tag: BETA-02\n"
                "Row 2: DELTA-04 => Tag: DELTA-04\n"
                "Row 3: MAPLE-VIOLET => Tag:"
            )

        shared_1block, _ = _fit_exact_token_prompt(
            self.tokenizer, head, "\n", block
        )
        trunk_prefix, trunk_ids = _fit_exact_token_prompt(
            self.tokenizer, shared_1block + block2_prose, "\n", 2 * block
        )
        prompt_a = trunk_prefix + tail_a
        prompt_b = shared_1block + tail_b

        ids_a = list(self.tokenizer.encode(prompt_a, add_special_tokens=False))
        ids_b = list(self.tokenizer.encode(prompt_b, add_special_tokens=False))
        self.assertGreater(len(ids_a), 2 * block)
        self.assertGreater(len(ids_b), block)
        self.assertEqual(
            ids_a[:block],
            ids_b[:block],
            f"Branch pair must share at least 1 full block ({block} tokens)",
        )
        self.assertNotEqual(ids_a[block : block + 4], ids_b[block : block + 4])
        self.assertEqual(ids_a[: 2 * block], trunk_ids)
        return prompt_a, ids_a, prompt_b, ids_b

    def _assert_logits_close(
        self,
        label: str,
        got_resp: Dict[str, Any],
        ref_resp: Dict[str, Any],
        *,
        min_cos: float = 0.998,
        max_abs: Optional[float] = 0.15,
    ) -> float:
        got = _final_logits(got_resp)
        ref = _final_logits(ref_resp)
        self.assertEqual(got.shape, ref.shape, f"[{label}] Logits shape mismatch: {got.shape} vs {ref.shape}")
        cos = float(torch.nn.functional.cosine_similarity(got.unsqueeze(0), ref.unsqueeze(0)).item())
        abs_diff = float((got - ref).abs().max().item())
        _emit("[Logit Check] %s cos=%.6f max_abs=%.6f", label, cos, abs_diff)
        self.assertGreaterEqual(cos, min_cos, f"[{label}] Logit cosine too low: {cos:.6f}")
        if max_abs is not None:
            self.assertLessEqual(abs_diff, max_abs, f"[{label}] Logit max_abs too high: {abs_diff:.6f}")
        return cos

    def _run_host_demotion_8layer(self) -> None:
        """Verify Device->Host eviction, Host->Device promotion, and mixed Host/Cold batch."""
        block = self.seq_size_per_block
        prompt_a, ids_a, prompt_b, ids_b = self._build_branch_pair("host_demote")
        prompt_c, ids_c = _fit_exact_token_prompt(
            self.tokenizer, "CH:", "\n", block + 5
        )
        spec_a = RequestSpec(
            "host_A", prompt_a, max_new_tokens=8, reuse_cache=False, expected_input_len=len(ids_a), return_logits=True
        )
        spec_b = RequestSpec(
            "host_B", prompt_b, max_new_tokens=8, reuse_cache=False, expected_input_len=len(ids_b), return_logits=True
        )
        spec_c = RequestSpec(
            "host_C", prompt_c, max_new_tokens=8, reuse_cache=False, expected_input_len=len(ids_c), return_logits=True
        )

        cold_resp_a = self._post_generate(spec_a)
        cold_resp_b = self._post_generate(spec_b)
        cold_resp_c = self._post_generate(spec_c)
        cold_a = _output_ids(cold_resp_a)
        cold_b = _output_ids(cold_resp_b)
        cold_c = _output_ids(cold_resp_c)

        # Seed A and B (reuse_cache=True); watermark ratio 1e-12/2e-12 demotes idle Device->Host.
        seed_a = self._post_generate(
            RequestSpec("host_seed_A", prompt_a, 8, True, expected_input_len=len(ids_a))
        )
        self.assertEqual(_output_ids(seed_a), cold_a)
        seed_b = self._post_generate(
            RequestSpec("host_seed_B", prompt_b, 8, True, expected_input_len=len(ids_b))
        )
        self.assertEqual(_output_ids(seed_b), cold_b)

        time.sleep(0.5)
        promo_a = self._post_generate(
            RequestSpec("host_promote_A", prompt_a, 8, True, expected_input_len=len(ids_a), return_logits=True)
        )
        tiers_a = _prefill_cache_tiers(promo_a["aux_info"])
        _emit("[8L Host Promote A] tiers=%s output_ids=%s", tiers_a, _output_ids(promo_a))
        self.assertEqual(_output_ids(promo_a), cold_a)
        self._assert_logits_close("host_promote_A", promo_a, cold_resp_a)
        self.assertEqual(tiers_a["device"], 0, f"Expected Device evicted to 0, got {tiers_a}")
        self.assertGreaterEqual(
            tiers_a["memory"],
            2 * block,
            f"Expected Host memory reuse >= {2 * block}, got {tiers_a}",
        )

        # Wait again so A & B sit in Host, then fire mixed B=3 [A_host, B_host, C_cold] together.
        time.sleep(0.5)
        snap = self._snapshot_batch_schedule_counts()
        mixed = self._run_concurrent_barrier(
            [
                RequestSpec("host_mix_A", prompt_a, 8, True, expected_input_len=len(ids_a), return_logits=True),
                RequestSpec("host_mix_B", prompt_b, 8, True, expected_input_len=len(ids_b), return_logits=True),
                RequestSpec("host_mix_C", prompt_c, 8, False, expected_input_len=len(ids_c), return_logits=True),
            ]
        )
        prefill_bs = self._collect_new_batch_sizes(snap, "prefill", expected_min_bs=2)
        decode_bs = self._collect_new_batch_sizes(snap, "decode")
        tiers_mix_a = _prefill_cache_tiers(mixed[0]["aux_info"])
        tiers_mix_b = _prefill_cache_tiers(mixed[1]["aux_info"])
        tiers_mix_c = _prefill_cache_tiers(mixed[2]["aux_info"])
        _emit(
            "[8L Host Mixed B=3] prefill_bs=%s decode_bs=%s tiers_A=%s tiers_B=%s tiers_C=%s",
            prefill_bs,
            decode_bs,
            tiers_mix_a,
            tiers_mix_b,
            tiers_mix_c,
        )
        self.assertEqual(_output_ids(mixed[0]), cold_a)
        self.assertEqual(_output_ids(mixed[1]), cold_b)
        self.assertEqual(_output_ids(mixed[2]), cold_c)
        self._assert_logits_close("host_mix_A", mixed[0], cold_resp_a)
        self._assert_logits_close("host_mix_B", mixed[1], cold_resp_b)
        self._assert_logits_close("host_mix_C", mixed[2], cold_resp_c)
        self.assertGreaterEqual(tiers_mix_a["memory"], 2 * block)
        self.assertGreaterEqual(tiers_mix_b["memory"], block)
        self.assertEqual(tiers_mix_c["total"], 0)
        self._assert_rdma_active_in_logs()
        self._assert_decode_cuda_graph_active_in_logs()

    def _run_synthetic_8layer(self) -> None:
        block = self.seq_size_per_block
        prompt_a, ids_a, prompt_b, ids_b = self._build_branch_pair("8l_reg")
        p_m1, ids_m1 = _fit_exact_token_prompt(self.tokenizer, "PM1:", "\n", block - 1)
        p_0, ids_0 = _fit_exact_token_prompt(self.tokenizer, "P0:", "\n", block)
        p_p1, ids_p1 = _fit_exact_token_prompt(self.tokenizer, "PP1:", "\n", block + 1)
        # Prefixes chosen for >= 0.15 early margin, >= 0.06 full-horizon margin, and >= 0.20 active-logit margin across all 32-48 steps.
        p_d1, ids_d1 = _fit_exact_token_prompt(self.tokenizer, "D1_24:", "\n", block - 1)
        p_d2, ids_d2 = _fit_exact_token_prompt(self.tokenizer, "D2_60:", "\n", 2 * block + 3)
        p_d3, ids_d3 = _fit_exact_token_prompt(self.tokenizer, "D3_28:", "\n", block)
        p_d4, ids_d4 = _fit_exact_token_prompt(self.tokenizer, "D4_15:", "\n", block + 1)

        # 1. Cold B=1 ground-truth baselines + exact page boundaries (15, 16, 17).
        cold_resp_a = self._post_generate(RequestSpec("cold_A", prompt_a, 8, False, expected_input_len=len(ids_a), return_logits=True))
        cold_resp_b = self._post_generate(RequestSpec("cold_B", prompt_b, 8, False, expected_input_len=len(ids_b), return_logits=True))
        cold_resp_m1 = self._post_generate(RequestSpec("page_m1", p_m1, 8, False, expected_input_len=len(ids_m1)))
        cold_resp_0 = self._post_generate(RequestSpec("page_0", p_0, 8, False, expected_input_len=len(ids_0), return_logits=True))
        cold_resp_p1 = self._post_generate(RequestSpec("page_p1", p_p1, 8, False, expected_input_len=len(ids_p1), return_logits=True))
        cold_a = _output_ids(cold_resp_a)
        cold_b = _output_ids(cold_resp_b)
        cold_0 = _output_ids(cold_resp_0)
        cold_p1 = _output_ids(cold_resp_p1)
        self.assertGreater(len(_output_ids(cold_resp_m1)), 0)

        # 2. Prefix branching + Device cache verification (B=1 bit-for-bit).
        seed_a = self._post_generate(
            RequestSpec("seed_A", prompt_a, 8, True, expected_input_len=len(ids_a))
        )
        self.assertEqual(_prefill_cache_tiers(seed_a["aux_info"])["total"], 0)
        self.assertEqual(_output_ids(seed_a), cold_a)

        partial_b = self._post_generate(
            RequestSpec("partial_B", prompt_b, 8, True, expected_input_len=len(ids_b), return_logits=True)
        )
        tiers_b = _prefill_cache_tiers(partial_b["aux_info"])
        self.assertGreaterEqual(tiers_b["device"], block)
        self.assertEqual(_output_ids(partial_b), cold_b)
        self._assert_logits_close("partial_B", partial_b, cold_resp_b)

        return_a = self._post_generate(
            RequestSpec("return_A", prompt_a, 8, True, expected_input_len=len(ids_a), return_logits=True)
        )
        tiers_a = _prefill_cache_tiers(return_a["aux_info"])
        self.assertGreaterEqual(tiers_a["device"], 2 * block)
        self.assertEqual(_output_ids(return_a), cold_a)
        self._assert_logits_close("return_A", return_a, cold_resp_a)

        # 3. Composed Batch 1: Barrier-synchronized Mixed-Reuse Prefill Batch (B=4).
        snap_prefill = self._snapshot_batch_schedule_counts()
        mix_specs = [
            RequestSpec("mix_A_hit", prompt_a, 8, True, expected_input_len=len(ids_a), return_logits=True),
            RequestSpec("mix_B_branch", prompt_b, 8, True, expected_input_len=len(ids_b), return_logits=True),
            RequestSpec("mix_page0_cold", p_0, 8, False, expected_input_len=len(ids_0), return_logits=True),
            RequestSpec("mix_pagep1_cold", p_p1, 8, False, expected_input_len=len(ids_p1), return_logits=True),
        ]
        mix_res = self._run_concurrent_barrier(mix_specs)
        mix_prefill_bs = self._collect_new_batch_sizes(snap_prefill, "prefill", expected_min_bs=2)
        mix_decode_bs = self._collect_new_batch_sizes(snap_prefill, "decode")
        _emit(
            "[8L Composed Mixed Prefill B=4] prefill_bs=%s decode_bs=%s",
            mix_prefill_bs,
            mix_decode_bs,
        )
        self.assertGreaterEqual(
            max(mix_prefill_bs or [0]),
            2,
            f"Expected multi-request Prefill batch packing, observed prefill_bs={mix_prefill_bs}",
        )
        for spec, res, expected_ids, cold_ref_resp, expect_device_min in zip(
            mix_specs,
            mix_res,
            (cold_a, cold_b, cold_0, cold_p1),
            (cold_resp_a, cold_resp_b, cold_resp_0, cold_resp_p1),
            (2 * block, block, 0, 0),
        ):
            out_ids = _output_ids(res)
            exact_match = out_ids == expected_ids
            tiers = _prefill_cache_tiers(res["aux_info"])
            _emit(
                "[8L Composed Mixed Prefill B=4] %s tiers=%s exact_vs_B1=%s",
                spec.name,
                tiers,
                exact_match,
            )
            self.assertEqual(out_ids, expected_ids)
            self._assert_logits_close(spec.name, res, cold_ref_resp)
            if expect_device_min > 0:
                self.assertGreaterEqual(tiers["device"], expect_device_min)
            else:
                self.assertEqual(tiers["total"], 0)

        # 4. Composed Batch 2: Staggered Continuous-Batching Decode (2 -> 4 -> 2 -> 0).
        cold_resp_d1 = self._post_generate(RequestSpec("cold_d1", p_d1, 48, False, expected_input_len=len(ids_d1), return_logits=True))
        cold_resp_d2 = self._post_generate(RequestSpec("cold_d2", p_d2, 48, False, expected_input_len=len(ids_d2), return_logits=True))
        cold_resp_d3 = self._post_generate(RequestSpec("cold_d3", p_d3, 32, False, expected_input_len=len(ids_d3), return_logits=True))
        cold_resp_d4 = self._post_generate(RequestSpec("cold_d4", p_d4, 40, False, expected_input_len=len(ids_d4), return_logits=True))

        snap_decode = self._snapshot_batch_schedule_counts()
        early_res, late_res = self._run_staggered_decode_batch(
            [
                RequestSpec("stag_d1", p_d1, 48, False, expected_input_len=len(ids_d1), return_logits=True),
                RequestSpec("stag_d2", p_d2, 48, False, expected_input_len=len(ids_d2), return_logits=True),
            ],
            [
                RequestSpec("stag_d3", p_d3, 32, False, expected_input_len=len(ids_d3), return_logits=True),
                RequestSpec("stag_d4", p_d4, 40, False, expected_input_len=len(ids_d4), return_logits=True),
            ],
            stagger_s=0.005,
        )
        stag_decode_bs = self._collect_new_batch_sizes(snap_decode, "decode", expected_min_bs=3)
        _emit("[8L Staggered Decode] decode_bs=%s", stag_decode_bs)
        self.assertGreaterEqual(
            max(stag_decode_bs or [0]),
            3,
            f"Expected >= 3 overlapping active Decode requests, observed {stag_decode_bs}",
        )
        for label, prompt, prompt_ids, got_resp, cold_ref_resp in zip(
            ("stag_d1", "stag_d2", "stag_d3", "stag_d4"),
            (p_d1, p_d2, p_d3, p_d4),
            (ids_d1, ids_d2, ids_d3, ids_d4),
            early_res + late_res,
            (cold_resp_d1, cold_resp_d2, cold_resp_d3, cold_resp_d4),
        ):
            got_ids = _output_ids(got_resp)
            ref_ids = _output_ids(cold_ref_resp)
            self.assertEqual(
                len(got_ids),
                len(ref_ids),
                f"[{label}] output_ids length mismatch vs B=1",
            )
            prefix_match = 0
            for a, b in zip(got_ids, ref_ids):
                if a != b:
                    break
                prefix_match += 1
            self.assertGreaterEqual(
                prefix_match,
                10,
                f"[{label}] prefix_match={prefix_match} < 10 vs B=1 (got={got_ids[:16]}, ref={ref_ids[:16]})",
            )
            prefix_text = self.tokenizer.decode(got_ids[:-1], skip_special_tokens=False)
            cold_prefix_prompt = prompt + prefix_text
            expected_input_ids = prompt_ids + got_ids[:-1]
            self.assertEqual(
                list(self.tokenizer.encode(cold_prefix_prompt, add_special_tokens=False)),
                expected_input_ids,
                f"[{label}] local lossless tokenizer round-trip failed for got_ids[:-1]",
            )
            cold_prefill_resp = self._post_generate(
                RequestSpec(
                    f"{label}_cold_prefill_ref",
                    cold_prefix_prompt,
                    1,
                    False,
                    expected_input_len=len(expected_input_ids),
                    return_logits=True,
                )
            )
            self.assertEqual(
                _input_ids(cold_prefill_resp),
                expected_input_ids,
                f"[{label}] server _input_ids round-trip mismatch",
            )
            self.assertEqual(
                _prefill_cache_tiers(cold_prefill_resp["aux_info"])["total"],
                0,
                f"[{label}] cold_prefill_ref unexpectedly reused cache",
            )
            cos = self._assert_logits_close(
                label, got_resp, cold_prefill_resp, min_cos=0.995, max_abs=None
            )
            _emit(
                "[8L Staggered Decode] %s prefix_match=%d/%d final_cos=%.6f",
                label,
                prefix_match,
                len(ref_ids),
                cos,
            )

        self._assert_rdma_active_in_logs()
        self._assert_decode_cuda_graph_active_in_logs()

    def _run_real_3fs_40layer(self) -> None:
        block = self.seq_size_per_block

        # 1. Core factual B=1 cold baselines + telemetry.
        core_prompts = [
            RequestSpec("zh_capital", "中国的首都是", 24, False, expected_substr="北京"),
            RequestSpec("en_capital", "The capital of China is", 24, False, expected_substr="Beijing"),
            RequestSpec("math_mul", "17 * 24 = ", 16, False, expected_substr="408"),
            RequestSpec("jp_capital", "日本的首都是", 16, False, expected_substr="东京"),
        ]
        core_cold: Dict[str, List[int]] = {}
        for spec in core_prompts:
            res = self._post_generate(spec)
            core_cold[spec.name] = _output_ids(res)
            _emit(
                "[40L 3FS Core B=1] %s -> %r aux=%s",
                spec.name,
                res.get("response"),
                res.get("aux_info"),
            )

        # 2. Prefix branching (A=CEDAR-AMBER, B=MAPLE-VIOLET) with shared block prefix.
        prompt_a, ids_a, prompt_b, ids_b = self._build_branch_pair("40l_3fs")
        cold_a = _output_ids(
            self._post_generate(
                RequestSpec("cold_A", prompt_a, 16, False, "CEDAR-AMBER", len(ids_a))
            )
        )
        cold_b = _output_ids(
            self._post_generate(
                RequestSpec("cold_B", prompt_b, 16, False, "MAPLE-VIOLET", len(ids_b))
            )
        )

        seed_a = self._post_generate(
            RequestSpec("seed_A", prompt_a, 16, True, "CEDAR-AMBER", len(ids_a))
        )
        self.assertEqual(_prefill_cache_tiers(seed_a["aux_info"])["total"], 0)
        self.assertEqual(_output_ids(seed_a), cold_a)

        partial_b = self._post_generate(
            RequestSpec("partial_B", prompt_b, 16, True, "MAPLE-VIOLET", len(ids_b))
        )
        self.assertGreaterEqual(_prefill_cache_tiers(partial_b["aux_info"])["device"], block)
        self.assertEqual(_output_ids(partial_b), cold_b)

        return_a = self._post_generate(
            RequestSpec("return_A", prompt_a, 16, True, "CEDAR-AMBER", len(ids_a))
        )
        self.assertGreaterEqual(_prefill_cache_tiers(return_a["aux_info"])["device"], 2 * block)
        self.assertEqual(_output_ids(return_a), cold_a)

        # 3. Exact page boundaries (63, 64, 65 tokens) cold determinism.
        page_specs: Dict[int, Tuple[str, List[int], List[int]]] = {}
        for target in (block - 1, block, block + 1):
            tag = f"PAGE-{target}"
            p, pids = _fit_exact_token_prompt(
                self.tokenizer,
                f"档案页 {target}：北京、上海、杭州、成都、广州、南京、西安、武汉地理与交通代码记录。",
                f"\nRow 1: PAGE-00 => Tag: PAGE-00\nRow 2: {tag} => Tag:",
                target,
            )
            r_page = self._post_generate(
                RequestSpec(f"page_{target}", p, 12, False, tag, len(pids))
            )
            page_specs[target] = (p, pids, _output_ids(r_page))
            _emit("[40L Page=%d] resp=%r", target, r_page.get("response"))

        # 4. Composed Batch 1: Barrier-synchronized Mixed-Reuse Prefill Batch (B=4).
        snap_prefill = self._snapshot_batch_schedule_counts()
        p64, ids64, cold_64 = page_specs[block]
        p65, ids65, cold_65 = page_specs[block + 1]
        mix_specs = [
            RequestSpec("mix_A_hit", prompt_a, 16, True, "CEDAR-AMBER", len(ids_a)),
            RequestSpec("mix_B_branch", prompt_b, 16, True, "MAPLE-VIOLET", len(ids_b)),
            RequestSpec("mix_page64_cold", p64, 12, False, f"PAGE-{block}", len(ids64)),
            RequestSpec("mix_page65_cold", p65, 12, False, f"PAGE-{block + 1}", len(ids65)),
        ]
        mix_res = self._run_concurrent_barrier(mix_specs)
        mix_prefill_bs = self._collect_new_batch_sizes(snap_prefill, "prefill", expected_min_bs=2)
        mix_decode_bs = self._collect_new_batch_sizes(snap_prefill, "decode")
        for res, spec, cold_ref, min_dev in zip(
            mix_res, mix_specs, (cold_a, cold_b, cold_64, cold_65), (2 * block, block, 0, 0)
        ):
            tiers = _prefill_cache_tiers(res["aux_info"])
            if min_dev > 0:
                self.assertGreaterEqual(tiers["device"], min_dev)
            else:
                self.assertEqual(tiers["total"], 0)
            out_ids = _output_ids(res)
            exact_match = out_ids == cold_ref
            _emit(
                "[40L Composed Mixed Prefill] %s tiers=%s exact_vs_B1=%s",
                spec.name,
                tiers,
                exact_match,
            )
            self.assertEqual(out_ids, cold_ref, f"[{spec.name}] Mixed batch output diverged from B=1")
        _emit(
            "[40L Mixed Prefill summary] prefill_bs=%s decode_bs=%s",
            mix_prefill_bs,
            mix_decode_bs,
        )

        # 5. Composed Batch 2: Staggered Continuous-Batching Decode (B=4, asymmetric lengths).
        snap_decode = self._snapshot_batch_schedule_counts()
        early_specs = [core_prompts[0], core_prompts[1]]
        late_specs = [
            RequestSpec("math_mul", "17 * 24 = ", 12, False, expected_substr="408"),
            core_prompts[3],
        ]
        early_res, late_res = self._run_staggered_decode_batch(
            early_specs, late_specs, stagger_s=0.15
        )
        stag_decode_bs = self._collect_new_batch_sizes(snap_decode, "decode", expected_min_bs=3)
        _emit("[40L Staggered Decode] decode_bs=%s", stag_decode_bs)
        self.assertGreaterEqual(
            max(stag_decode_bs or [0]),
            3,
            f"Expected >= 3 overlapping active Decode requests, observed {stag_decode_bs}",
        )
        for spec, res in zip(early_specs + late_specs, early_res + late_res):
            ref = core_cold[spec.name]
            got = _output_ids(res)
            prefix_match = sum(1 for a, b in zip(got, ref) if a == b)
            _emit(
                "[40L Staggered Decode] %s token_match=%d/%d resp=%r",
                spec.name,
                prefix_match,
                min(len(got), len(ref)),
                res.get("response"),
            )

        self._assert_rdma_active_in_logs()
        self._assert_decode_cuda_graph_active_in_logs()

    def test_pure_pd_generation(self) -> None:
        if self.force_host_demotion:
            self._run_host_demotion_8layer()
        elif self.mode == "synthetic_8layer":
            self._run_synthetic_8layer()
        else:
            self._run_real_3fs_40layer()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
