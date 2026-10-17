"""Opt-in forward/copy accounting and separately armed Indexer diagnostics.

Timing runs only read existing CPU lengths and count submitted copy bytes.
The expensive Indexer bitmap pass is explicitly armed by the smoke driver
after performance/timeline collection. Request IDs namespace logical token
positions; equal positions in different requests are never merged.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch


class PrefillCacheDiagnostics:
    def __init__(self, rank):
        self.rank = rank
        self.root = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.seen_captures = set()
        self.active = False

    def begin(self, fmha):
        self.step += 1
        op = fmha.fmha_impl
        inputs = op.attn_inputs
        cp_info = getattr(op, "cp_info", None)
        lengths = (
            (
                cp_info.prefill_actual_input_lengths_cpu
                if cp_info is not None
                else inputs.input_lengths
            )
            .reshape(-1)
            .tolist()
        )
        prefixes = inputs.prefix_lengths.reshape(-1).tolist()
        self.lengths = [int(q + p) for q, p in zip(lengths, prefixes)]
        self.row = {
            "rank": self.rank,
            "step": self.step,
            "begin_us": time.time_ns() // 1000,
            "batch": len(lengths),
            "exec_tokens": sum(lengths),
            "kv_tokens": sum(self.lengths),
            "sequence_lengths": self.lengths,
        }
        stream = getattr(fmha, "_kv_overlap_stream", None)
        self.row["kv_overlap_stream"] = (
            stream.cuda_stream if stream is not None else None
        )
        self.row["kv_cross_layer"] = (
            getattr(fmha, "_cross_layer_prefetch", None) is not None
        )
        pipeline = getattr(fmha, "_cross_layer_prefetch", None)
        self.row["kv_registered"] = bool(getattr(pipeline, "registered", False))
        self.row["kv_index_opt"] = os.environ.get("RTP_LLM_CP_KV_INDEX_OPT", "0") == "1"
        self.row["async_prepare_cache"] = (
            os.environ.get("RTP_LLM_ASYNC_PREPARE_CACHE", "0") == "1"
        )
        self.op = op
        self.capture = None
        control = self.root / "prefill_indexer_capture.json"
        if control.exists():
            config = json.loads(control.read_text())
            if (
                config["id"] not in self.seen_captures
                and len(lengths) >= config["min_batch"]
            ):
                self.capture = config["id"]
        self.bitmaps = {}
        self.indexer_rows = []
        self.active = True

    def indexer(self, layer, topk):
        if self.capture is None or topk is None:
            return
        values = topk.reshape(topk.shape[0], -1).long()
        request_ids = self.op.precomputed_req_ids.long().reshape(-1, 1)
        if request_ids.shape[0] != values.shape[0]:
            raise ValueError("Indexer rows must match CP request IDs")
        width = max(self.lengths)
        limits = torch.tensor(self.lengths, device=values.device)[request_ids]
        valid = (values >= 0) & (values < limits)
        keys = (request_ids * width + values)[valid]
        counts = torch.bincount(keys, minlength=len(self.lengths) * width)
        bitmap = (counts > 0).cpu().numpy()
        self.bitmaps[f"layer_{layer}"] = np.packbits(bitmap)
        self.indexer_rows.append(
            {
                "layer": layer,
                "query_rows": values.shape[0],
                "topk": values.shape[1],
                "selection_count": keys.numel(),
                "unique_request_tokens": int(bitmap.sum()),
                "bitmap_width": width,
            }
        )

    def finish(self, layered, failed):
        if not self.active:
            return
        self.active = False
        self.row["end_us"] = time.time_ns() // 1000
        self.row["failed"] = failed
        from rtp_llm.models_py.distributed.pynccl_cp import _INDEXER_BUFFERS

        self.row["indexer_registered"] = bool(_INDEXER_BUFFERS)
        self.row["indexer_capture"] = self.capture
        stats = torch.cuda.memory_stats()
        self.row["cuda_allocator"] = {
            key: stats.get(key, 0)
            for key in (
                "num_alloc_retries",
                "num_sync_all_streams",
                "num_device_alloc",
                "num_device_free",
                "allocated_bytes.all.current",
                "reserved_bytes.all.current",
                "allocated_bytes.all.peak",
                "reserved_bytes.all.peak",
            )
        }
        if layered is not None:
            self.row.update(layered.copy_stats)
            self.row["unique_fetched_pages"] = sum(n for _, _, n in layered.fetch_runs)
            self.row["dirty_pages"] = sum(n for _, _, n in layered.write_runs)
            self.row["page_bytes"] = layered.page_bytes
        if self.capture is not None:
            self.row["indexer"] = self.indexer_rows
            name = f"prefill_indexer_rank{self.rank}_step{self.step}.npz"
            np.savez_compressed(self.root / name, **self.bitmaps)
            self.row["indexer_bitmaps"] = name
            self.seen_captures.add(self.capture)
        with (self.root / f"prefill_forwards_rank{self.rank}.jsonl").open("a") as out:
            out.write(json.dumps(self.row) + "\n")
        self.bitmaps.clear()
