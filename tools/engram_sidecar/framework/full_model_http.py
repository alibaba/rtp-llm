#!/usr/bin/env python3
"""Explicit TP4 full-checkpoint experiment with opt-in decode Graph, separate from component tests.

runtime/preflight never start a model or allocate EGM. launch starts the actual
RTP server in this process; its lifetime and the Owner's STOP belong to the
operator. No fixture, fake compute module, synthetic projection, or HTTP mock.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
NATIVE_LIBS = ("libth_transformer_config.so", "librtp_compute_ops.so",
               "libth_transformer.so", "libth_grammar_tokenizer_info.so")
PYTHON_PACKAGES = ("torch", "transformers", "tokenizers", "safetensors", "fastsafetensors",
                   "triton", "deep_gemm", "flashinfer", "flash_mla", "tilelang", "fastapi", "uvicorn",
                   "pydantic", "grpc", "requests", "PIL", "engram_native_dev_492ecf45")
TOP_LEVEL_FIELDS = ("text_config", "vision_config", "quantization_config", "dtype",
                    "bos_token_id", "eos_token_id", "pad_token_id", "image_token_id")
TEXT_FIELDS = ("num_hidden_layers", "hidden_size", "vocab_size", "num_attention_heads",
               "head_dim", "qk_rope_head_dim", "compress_ratios", "o_groups", "o_lora_rank",
               "index_head_dim", "index_n_heads", "index_topk", "routed_scaling_factor",
               "num_experts_per_tok", "n_routed_experts", "n_shared_experts", "moe_intermediate_size",
               "max_position_embeddings", "kv_source_layer_ids", "index_source_layer_ids",
               "engram_layer_ids", "engram_num_embeddings", "engram_max_ngram_size",
               "engram_n_heads", "engram_head_dim", "engram_compressed_vocab_size",
               "engram_pad_token_id", "engram_vocab_size")
STORAGE_MODES = ("external-egm", "local-pinned")
EXTERNAL_ENV = ("RTP_ENGRAM_COHORT", "RTP_ENGRAM_INSTANCE_INDEX", "RTP_ENGRAM_MODEL_REVISION",
                "RTP_ENGRAM_LOCAL_DESCRIPTORS", "RTP_ENGRAM_COHORT_TIMEOUT",
                "RTP_ENGRAM_TLS_DIR", "RTP_ENGRAM_CHECKPOINT", "ENGRAM_SHARD_COUNT",
                "RTP_ENGRAM_GRAPH_MODE",
                "RTP_ENGRAM_MAX_ALLOCATION_BYTES", "RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES")


def sha256_file(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def checkpoint_preflight(checkpoint, storage_mode="external-egm"):
    """Read config, index, and file metadata only; never materialize weights.

    This is a fail-fast screen, not a proof all model descriptors are satisfiable.
    Real loading/forward/HTTP still must pass. Both modes check every shard
    header against its index; only local-pinned may contain the exact tables.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.is_absolute():
        raise ValueError("absolute checkpoint path required")
    if storage_mode not in STORAGE_MODES:
        raise ValueError("invalid storage mode")
    from tools.engram_sidecar.framework.stage_checkpoint import _canonical, _small_file, _header, _snapshot, _decode
    checkpoint = _canonical(checkpoint)
    raw = _decode(_small_file(checkpoint / "config.json")[0])
    if raw.get("fixture_component_only"):
        raise ValueError("component fixture is not a full LLM checkpoint")
    missing = [key for key in TOP_LEVEL_FIELDS if key not in raw]
    if missing:
        raise ValueError("missing V4.1 top-level fields: " + ",".join(missing))
    text = raw["text_config"]
    if not isinstance(text, dict) or not isinstance(raw["vision_config"], dict):
        raise ValueError("text_config and vision_config must be objects")
    missing = [key for key in TEXT_FIELDS if key not in text]
    if missing:
        raise ValueError("missing V4.1 text fields: " + ",".join(missing))
    architectures = raw.get("architectures", [])
    if architectures and "DeepseekV41ForCausalLM" not in architectures:
        raise ValueError("checkpoint architecture does not identify DeepseekV41ForCausalLM")
    if (not text["engram_layer_ids"]
            or len(text["engram_layer_ids"]) != len(text["engram_num_embeddings"])):
        raise ValueError("full Engram layer geometry required")
    layers, rows, dim = text["engram_layer_ids"], text["engram_num_embeddings"], text["engram_head_dim"]
    if (type(text["num_hidden_layers"]) is not int or text["num_hidden_layers"] <= 0
            or any(type(x) is not int or not 0 <= x < text["num_hidden_layers"] for x in layers)
            or len(set(layers)) != len(layers)
            or any(type(x) is not int or x <= 0 for x in rows)
            or type(dim) is not int or dim <= 0 or dim % 32):
        raise ValueError("invalid Engram layer geometry")
    for required in ("tokenizer.json", "tokenizer_config.json", "encoding/encoding.py"):
        if not (checkpoint / required).is_file():
            raise ValueError("missing HTTP tokenizer/encoding asset: " + required)
    index = _decode(_small_file(checkpoint / "model.safetensors.index.json")[0])
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("nonempty safetensors weight_map required")
    actual_tables = {name for name in weights if ".engram.embed." in name}
    expected_tables = {f"layers.{layer}.engram.embed.{role}": (count, width, dtypes)
                       for layer, count in zip(layers, rows)
                       for role, width, dtypes in (("weight", dim, {"F8_E4M3", "F8_E4M3FN"}),
                                                   ("scale", dim // 32, {"F8_E8M0", "U8"}))}
    if storage_mode == "external-egm" and actual_tables:
        raise ValueError("consumer checkpoint must omit Engram embedding table entries")
    if storage_mode == "local-pinned" and actual_tables != set(expected_tables):
        raise ValueError("local-pinned requires exactly every configured Engram weight and scale")
    for layer in text["engram_layer_ids"]:
        for suffix in ("wkv.weight", "wkv.scale", "q_weight", "k_weight"):
            if f"layers.{layer}.engram.{suffix}" not in weights:
                raise ValueError(f"missing local Engram projection: layer={layer}, suffix={suffix}")
    # Genuine full backbone and local ViT loading are mandatory in this launch.
    for layer in range(text["num_hidden_layers"]):
        prefix = f"layers.{layer}."
        if not any(key.startswith(prefix + "attn.") for key in weights):
            raise ValueError(f"missing attention weights for full layer {layer}")
        if not any(key.startswith(prefix + "ffn.") for key in weights):
            raise ValueError(f"missing MoE weights for full layer {layer}")
    if not any(key.startswith("vision.") for key in weights) or not any(key.startswith("aligner.") for key in weights):
        raise ValueError("missing local ViT weights")
    for name in ("embed.weight", "head.weight", "norm.weight", "image_start", "image_newline", "image_end"):
        if name not in weights:
            raise ValueError("missing global model weight: " + name)
    files = set()
    for relative in weights.values():
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or Path(relative).name != relative or "\\" in relative
                or not relative.endswith(".safetensors")):
            raise ValueError("unsafe checkpoint index path")
        path = checkpoint / relative
        if not path.is_file():
            raise ValueError("checkpoint shard unavailable: " + relative)
        files.add(path)
    if {path.name for path in checkpoint.glob("*.safetensors")} != {path.name for path in files}:
        raise ValueError("unindexed checkpoint tensor files")
    header_hashes = {}
    for path in sorted(files):
        _canonical(path)
        header, _, header_hash = _header(path, _snapshot(path))
        names = set(header) - {"__metadata__"}
        if names != {name for name, shard in weights.items() if shard == path.name}:
            raise ValueError("checkpoint header/index mismatch")
        header_hashes[path.name] = header_hash
        for name in names & actual_tables:
            count, width, dtypes = expected_tables[name]
            if header[name]["shape"] != [count, width] or header[name]["dtype"] not in dtypes:
                raise ValueError("Engram table dtype/shape mismatch")
    engram_bytes = sum(text["engram_num_embeddings"]) * (text["engram_head_dim"] + text["engram_head_dim"] // 32)
    return {"model_type": "deepseek_v41", "full_layer_count": text["num_hidden_layers"],
            "engram_layers": text["engram_layer_ids"], "engram_logical_bytes": engram_bytes,
            "consumer_tensor_count": len(weights), "consumer_shard_count": len(files),
            "consumer_shard_file_bytes": sum(path.stat().st_size for path in files),
            "consumer_has_table_entries": bool(actual_tables), "storage_mode": storage_mode,
            "shard_header_sha256": header_hashes,
            "tokenizer_sha256": sha256_file(checkpoint / "tokenizer.json"),
            "config_sha256": sha256_file(checkpoint / "config.json"),
            "index_sha256": sha256_file(checkpoint / "model.safetensors.index.json")}



def runtime_preflight(repository=REPO, *, load_native=False):
    """Require bounded same-checkout library paths; no recursive .so fallback."""
    libs = Path(repository) / "rtp_llm" / "libs"
    missing = [name for name in NATIVE_LIBS if not (libs / name).is_file()]
    if missing:
        raise RuntimeError("missing exact-main runtime libraries: " + ",".join(missing))
    missing_packages = [name for name in PYTHON_PACKAGES if importlib.util.find_spec(name) is None]
    if missing_packages:
        raise RuntimeError("missing runtime Python packages: " + ",".join(missing_packages))
    hashes = {name: sha256_file(libs / name) for name in NATIVE_LIBS}
    if sys.platform.startswith("linux"):
        for name in NATIVE_LIBS:
            checked = subprocess.run(["ldd", str(libs / name)], text=True, capture_output=True, timeout=60)
            if checked.returncode or "not found" in checked.stdout:
                raise RuntimeError("unresolved ELF dependency: " + name + "\n" + checked.stdout + checked.stderr)
    if load_native:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(repository) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        # Respect required main initialization order. This is only a load/import
        # probe, with GPUs hidden; no model is constructed and no weights read.
        code = (
            "import torch; import rtp_llm.ops as ops; ops.ensure_engine_ops_loaded(); "
            "import libth_grammar_tokenizer_info; import rtp_llm.start_server; "
            "import rtp_llm.models.deepseek_v41; "
            "import rtp_llm.openai.renderers.deepseekv41_renderer; "
            "from rtp_llm.dash_sc.app import DashScApp; "
            "from rtp_llm.frontend.frontend_app import FrontendApp; "
            "from rtp_llm.server.backend_manager import BackendManager; "
            "from rtp_llm.async_decoder_engine.rpc_engine import LanguageCppEngine; "
            "from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model; "
            "import rtp_llm.start_backend_server; "
            # These are lazy in the RTP wrappers; load the actual wheel and
            # its CUDA bindings before any complete-checkpoint allocation.
            "from flash_mla import flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata; "
            "from flash_mla import cuda as flash_mla_cuda; "
            "assert all(callable(f) for f in (flash_mla_sparse_fwd, flash_mla_with_kvcache, "
            "get_mla_metadata, flash_mla_cuda.sparse_prefill_fwd, flash_mla_cuda.sparse_decode_fwd)), "
            "'missing V4.1 attention entrypoints'; "
            "assert not torch.cuda.is_initialized(), 'preflight initialized CUDA'; "
            "print('FULL_HTTP_NATIVE_IMPORT_OK')"
        )
        subprocess.run([sys.executable, "-c", code], cwd=repository, env=env, check=True, timeout=180)
    return {"native_sha256": hashes, "native_import_checked": load_native,
            "note": "Library provenance must match the recorded same-main build; names alone do not prove ABI."}


def launch_environment(args, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    cuda_graph = getattr(args, "cuda_graph", False)
    if type(cuda_graph) is not bool:
        raise ValueError("cuda_graph must be an explicit boolean")
    storage_mode = getattr(args, "storage_mode", "external-egm")
    if storage_mode not in STORAGE_MODES:
        raise ValueError("invalid storage mode")
    if storage_mode == "local-pinned":
        # A local run must never inherit a static cohort contract by accident.
        for key in EXTERNAL_ENV:
            env.pop(key, None)
    # Dedicated test slice: no inherited service discovery, drafts, truncation,
    # or model overrides. Driver/library/device visibility remains operator set.
    for key in ("MODEL_SERVICE_CONFIG", "REMOTE_RPC_SERVER_IP", "REMOTE_VIT_SERVER_IP",
                "RTP_LLM_DECODE_CM2_CONFIG", "RTP_LLM_MULTIMODAL_PART_CM2_CONFIG",
                "SP_CHECKPOINT_PATH", "PTUNING_PATH", "LORA_INFO",
                "GENERATION_PREFILL_CAPTURE_CONFIG", "GENERATION_PREFILL_CUDA_GRAPH_MAX_REQUESTS",
                "PREFILL_CAPTURE_CONFIG", "DECODE_CAPTURE_CONFIG"):
        env.pop(key, None)
    env.update({
        "MODEL_TYPE": "deepseek_v41", "CHECKPOINT_PATH": str(args.checkpoint),
        "TOKENIZER_PATH": str(args.checkpoint),
        "LOAD_PYTHON_MODEL": "1", "ROLE_TYPE": "PDFUSION", "USE_LOCAL": "1", "THINK_MODE": "0", "ACT_TYPE": "BF16",
        "TP_SIZE": "4", "EP_SIZE": "4", "DP_SIZE": "1", "WORLD_SIZE": "4", "WORLD_RANK": "0",
        "ENABLE_CUDA_GRAPH": "1" if cuda_graph else "0",
        "ENABLE_NATIVE_CUDA_GRAPH": "0", "DECODE_CAPTURE_CONFIG": "1" if cuda_graph else "",
        "SP_TYPE": "", "SP_MODEL_TYPE": "", "HACK_LAYER_NUM": "0",
        "JSON_MODEL_OVERRIDE_ARGS": "", "LOAD_METHOD": "fastsafetensors", "FASTSAFETENSORS_NOGDS": "1",
        "VIT_SEPARATION": "0",
        "DSV4_MOE_STRATEGY": "mega", "DSV4_USE_MEGA_MOE_SE": "0", "DSV4_USE_MEGA_MOE_FUSED": "0",
        "DSV41_SWA_BOUNDED_REPLAY": "0", "DSV41_CED": "0", "FP8_KV_CACHE": "1",
        # Compare the same direct GPU lookup path on both storage backends.
        # Release defaults to CPU prefill lookup for local tables, unavailable
        # to the device-only external EGM view. Default-prefill speed is a
        # separate experiment, not part of this storage-cost comparison.
        "DSV41_CPU_ENGRAM_LOOKUP": "0", "DSV41_ASYNC_ENGRAM_LOOKUP": "0",
        "SEQ_SIZE_PER_BLOCK": "128", "KERNEL_SEQ_SIZE_PER_BLOCK": "128",
        "MAX_SEQ_LEN": "4096", "MAX_CONTEXT_BATCH_SIZE": "1", "CONCURRENCY_LIMIT": "1",
        "REUSE_CACHE": "1", "RESERVER_RUNTIME_MEM_MB": "32768", "KV_CACHE_MEM_MB": "8192",
        "FRONTEND_SERVER_COUNT": "1", "START_PORT": str(args.port), "WARM_UP": "1", "MODEL_WARM_UP": "1",
        "PYTHONPATH": str(REPO) + os.pathsep + env.get("PYTHONPATH", ""),
    })
    if storage_mode == "external-egm":
        # The cohort contract comes from RTP_ENGRAM_* in the caller's environment;
        # serving no longer consumes bundle files.
        env.update({"RTP_ENGRAM_MAX_ALLOCATION_BYTES": str(args.max_allocation_bytes),
                    "RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES": str(args.budget_bytes),
                    "RTP_ENGRAM_GRAPH_MODE": "process_lifetime" if cuda_graph else "eager"})
    return env


def cohort_preflight(args):
    """Validate the static cohort inputs that the serving path consumes."""
    from rtp_llm.models_py.modules.dsv41.engram_startup import build_group

    config = build_group(os.environ, args.world_size if hasattr(args, "world_size") else 1)
    if config is None:
        raise ValueError("external-egm requires RTP_ENGRAM_COHORT in the environment")
    return {key: config["group"][key] for key in ("generation", "model_revision")}


def chat_payload(*, max_tokens=32):
    # V4.1 resolve_thinking gives this explicit schema precedence over defaults.
    # A short content smoke must not spend its whole token budget on reasoning.
    return {"model": "deepseek_v41", "messages": [{"role": "user", "content": "你好，请用一句话介绍自己。"}],
            "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "thinking": {"type": "disabled"}}


def chat_request(port, *, max_tokens=32):
    payload = chat_payload(max_tokens=max_tokens)
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=180) as response:
        result = json.load(response)
    if "error" in result or "error_code" in result:
        raise RuntimeError("HTTP returned an inference error: " + json.dumps(result, ensure_ascii=False))
    content = result["choices"][0]["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("chat response has no user-visible text; not a successful human-readable output")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("runtime", "preflight", "launch", "chat"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--model-revision")
    parser.add_argument("--storage-mode", choices=STORAGE_MODES, default="external-egm")
    parser.add_argument("--max-allocation-bytes", type=int, default=1 << 30)
    parser.add_argument("--budget-bytes", type=int, default=8 << 30)
    parser.add_argument("--port", type=int, default=18630)
    parser.add_argument("--cuda-graph", action="store_true",
                        help="Experimental decode-only batch-1 Graph; mappings remain pinned until worker exit")
    parser.add_argument("--load-native", action="store_true", help="Probe exact native + start_server imports in a GPU-hidden child")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65000:
        parser.error("port must be in [1024,65000]")
    if args.command == "chat":
        print(json.dumps(chat_request(args.port), ensure_ascii=False, indent=2))
        return 0
    report = {"runtime": runtime_preflight(load_native=args.load_native or args.command == "launch")}
    if args.command == "runtime":
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0
    if args.checkpoint is None or not args.model_revision:
        parser.error("checkpoint and model revision required")
    report["checkpoint"] = checkpoint_preflight(args.checkpoint, args.storage_mode)
    report["storage_mode"] = args.storage_mode
    report["model_revision"] = args.model_revision
    if args.storage_mode == "external-egm":
        from rtp_llm.models_py.modules.dsv41.engram_capacity import validate_capacity
        validate_capacity(args.max_allocation_bytes, args.budget_bytes)
        os.environ["RTP_ENGRAM_MAX_ALLOCATION_BYTES"] = str(args.max_allocation_bytes)
        os.environ["RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES"] = str(args.budget_bytes)
        report["cohort"] = cohort_preflight(args)
    elif args.bundle is not None:
        parser.error("local-pinned must not supply an external bundle")
    # Public reproducibility record: never include capabilities or inherited
    # secrets. The normal main server receives the full prepared environment.
    fixed = launch_environment(args, {})
    report["experiment_environment"] = {k: v for k, v in fixed.items()
                                        if k not in (*EXTERNAL_ENV, "CHECKPOINT_PATH", "TOKENIZER_PATH", "PYTHONPATH")}
    print(json.dumps(report, sort_keys=True, indent=2), flush=True)
    if args.command == "launch":
        # Preflight made no CUDA allocation. exec starts the real main server;
        # it sets rank-local devices in its normal worker initialization.
        os.chdir(REPO)
        os.execve(sys.executable, [sys.executable, "-m", "rtp_llm.start_server"], launch_environment(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
