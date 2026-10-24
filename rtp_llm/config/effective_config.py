"""Portable engine behavior snapshot, exported after real configuration finalization.

This module deliberately imports neither Torch nor rtp_llm.ops. Capacity is a
separate observation: a configured memory budget is not an allocator result.
"""

import argparse
import json
from pathlib import Path

SCHEMA_VERSION = 1


def snapshot(engine, model, *, revision, capacity=None):
    runtime = engine.runtime_config
    fifo = runtime.fifo_scheduler_config
    kv = engine.kv_cache_config
    parallel = engine.parallelism_config
    role = engine.pd_sep_config.role_type.name.removeprefix("ROLE_TYPE_").lower()
    cp_enabled = parallel.prefill_cp_config.is_enabled()
    values = {
        "pd_sep": {
            **{
                name: getattr(engine.pd_sep_config, name)
                for name in (
                    "prefill_retry_times",
                    "prefill_retry_timeout_ms",
                    "prefill_max_wait_timeout_ms",
                    "decode_retry_interval_ms",
                    "load_cache_timeout_ms",
                    "max_rpc_timeout_ms",
                    "prefill_stop_stream_wait_timeout_ms",
                    "decode_retry_times",
                    "decode_retry_timeout_ms",
                )
            },
            "prefill_prepare_resource_pool_size": max(
                128,
                engine.pd_sep_config.prefill_prepare_resource_pool_size
                if engine.pd_sep_config.prefill_prepare_resource_pool_size > 0
                else max(1, engine.concurrency_config.concurrency_limit) * 2,
            ),
        },
        "scheduler": {
            "type": "batch_decode"
            if runtime.use_batch_decode_scheduler
            else "ratio"
            if role == "pdfusion"
            and fifo.pdfusion_scheduler_mode.strip().lower() == "ratio"
            else "fifo",
            "max_generate_batch_size": runtime.max_generate_batch_size,
            "max_batch_tokens_size": fifo.max_batch_tokens_size,
            "max_batch_tokens_without_cache": fifo.max_batch_tokens_without_cache,
            "max_inited_kv_cache_streams": fifo.max_inited_kv_cache_streams,
            "cp_force_single_prefill": fifo.cp_force_single_prefill,
        },
        "model": {
            "max_seq_len": model.max_seq_len,
            "cp_enabled": cp_enabled,
            "cp_size": parallel.tp_size if cp_enabled else 1,
            "tp_size": parallel.tp_size,
            "dp_size": parallel.dp_size,
        },
        "kv_cache": {
            name: getattr(kv, name)
            for name in (
                "seq_size_per_block",
                "reuse_cache",
                "enable_device_cache",
                "enable_memory_cache",
                "enable_disk_cache",
                "enable_remote_cache",
                "reserve_block_ratio",
                "device_eviction_policy",
                "memory_eviction_policy",
                "block_tree_transfer_worker_count",
                "block_tree_business_queue_max_size",
                "block_tree_transfer_queue_max_size",
                "memory_cache_max_descriptors_per_transfer_batch",
                "memory_cache_sync_timeout_ms",
                "block_tree_device_evict_low_watermark_ratio",
                "block_tree_device_evict_high_watermark_ratio",
                "block_tree_memory_evict_low_watermark_ratio",
                "block_tree_memory_evict_high_watermark_ratio",
            )
        },
    }
    provenance = getattr(engine, "config_provenance", {})
    values["provenance"] = {}
    for section in ("scheduler", "model", "kv_cache", "pd_sep"):
        for key, value in values[section].items():
            parsed = provenance.get(key)
            values["provenance"][section + "." + key] = (
                parsed["source"] if parsed and parsed["value"] == value else "derived"
            )
    if capacity is not None:
        values["capacity"] = capacity
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {"kind": "real", "revision": revision},
        "roles": {role: values},
    }


def vit_snapshot(vit, *, revision, work_budget=None):
    scheduler = vit.embedding_scheduler_args()
    names = (
        "vit_concurrency",
        "vit_max_queue_size",
        "gpu_max_batch_size",
        "gpu_max_batch_images",
        "gpu_batch_wait_ms",
        "mm_max_queue_size",
        "mm_cache_gpu_max_bytes",
        "mm_cache_cpu_max_bytes",
        "mm_hash_key_cache_max_bytes",
        "mm_remote_cache_enable",
        "mm_timeout_ms",
        "mm_image_max_file_size_kb",
        "use_local_preprocess",
        "mm_preprocess_max_workers",
        "download_headers",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {"kind": "real", "revision": revision},
        "roles": {
            "vit": {
                **{name: getattr(vit, name) for name in names},
                "mm_timeout_ms": vit.mm_timeout_ms if vit.mm_timeout_ms > 0 else 120000,
                "work_budget": None
                if work_budget is None
                else {
                    name: getattr(work_budget, name)
                    for name in (
                        "input_patches",
                        "output_tokens",
                        "estimated_workspace_bytes",
                        "max_attention_segment",
                        "attention_work",
                    )
                },
                "mm_transport_mode": vit.output_transport.mode,
                "vit_concurrency": max(1, int(vit.vit_concurrency)),
                "vit_max_queue_size": max(0, int(vit.vit_max_queue_size)),
                "gpu_max_batch_size": scheduler["max_batch_size"],
                "gpu_max_batch_images": scheduler["max_batch_images"],
                "gpu_batch_wait_ms": scheduler["batch_wait_ms"],
                "mm_max_queue_size": scheduler["max_queue_size"],
            }
        },
    }


def write_snapshot(path, document):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def merge(documents):
    if not documents:
        raise ValueError("at least one snapshot is required")
    result = {
        "schema_version": SCHEMA_VERSION,
        "source": documents[0]["source"],
        "roles": {},
    }
    for document in documents:
        if (
            document["schema_version"] != SCHEMA_VERSION
            or document["source"] != result["source"]
        ):
            raise ValueError("snapshots must use the same schema and source revision")
        for role, values in document["roles"].items():
            if role in result["roles"]:
                raise ValueError(f"duplicate role: {role}")
            result["roles"][role] = values
    return result


def export_requested(document_factory, *, rank=0, dp_rank=0):
    import os

    output = os.environ.get("RTP_LLM_EFFECTIVE_CONFIG_OUTPUT")
    if output:
        revision = os.environ.get("RTP_LLM_SOURCE_REVISION")
        if not revision:
            raise ValueError(
                "RTP_LLM_SOURCE_REVISION is required when exporting effective config"
            )
        document = document_factory(revision)
        role = next(iter(document["roles"]))
        write_snapshot(f"{output}.{role}.rank{rank}.dp{dp_rank}.json", document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="append", required=True, type=Path)
    parser.add_argument(
        "--capacity",
        action="append",
        type=Path,
        default=[],
        help="role -> allocator observation; explicit kind=real or synthetic",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    document = merge([json.loads(path.read_text()) for path in args.snapshot])
    for capacity_path in args.capacity:
        for role, capacity in json.loads(capacity_path.read_text()).items():
            if capacity.get("kind") not in {"real", "synthetic"} or not capacity.get(
                "source"
            ):
                raise ValueError("capacity requires kind and observation source")
            if (
                capacity["kind"] == "real"
                and capacity.get("source_revision") != document["source"]["revision"]
            ):
                raise ValueError(
                    "allocator observation and effective config use different source revisions"
                )
            if "capacity" in document["roles"][role]:
                raise ValueError(f"capacity already provided for {role}")
            document["roles"][role]["capacity"] = capacity
    write_snapshot(args.output, document)


if __name__ == "__main__":
    main()
