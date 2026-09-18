"""Real-checkpoint TP8 validation for the Qwen3.8-Flash-Next PLE table.

This test is deliberately opt-in: the released PLE table is 102.4 GB and the
smallest safe production layout is eight H20-class GPUs.  It loads only the 16
logical table shards owned by each TP rank plus the three tiny integer hash
metadata tensors; no other model tensor is read.

Set ``QWEN4_EXP_CKPT_DIR`` to the complete checkpoint directory.  The test
skips when the directory or eight CUDA devices are unavailable.
"""

import json
import os
import unittest
from unittest import mock

import torch
import torch.multiprocessing as mp

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.tensor_source import DatabaseTensorSource, TensorSource
from rtp_llm.models.qwen4_exp.qwen4_exp import Qwen4Exp
from rtp_llm.models.qwen4_exp.qwen4_exp_weight import (
    Qwen4ExpPleNgramWeight,
    Qwen4ExpWeight,
)
from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.modules.qwen4_exp.ple import Qwen4ExpNGramEmbedding
from rtp_llm.ops import HWKernelConfig, NcclCommConfig, ParallelismConfig
from rtp_llm.test.utils.port_util import PortsContext
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W


_CKPT_DIR = os.environ.get("QWEN4_EXP_CKPT_DIR", "")
_INDEX_NAME = "model.safetensors.index.json"
_WORLD_SIZE = 8
_SHARDS_PER_RANK = 16
_TOTAL_SHARDS = 128
_PLE_LAYER_INDEX = 1
_EXPECTED_SHARD_BYTES = 800_003_840
_EXPECTED_RANK_TABLE_BYTES = _SHARDS_PER_RANK * _EXPECTED_SHARD_BYTES
_METADATA_NAMES = (
    W.qwen4_ple_multipliers,
    W.qwen4_ple_ngram_offsets,
    W.qwen4_ple_ngram_vocab_sizes,
)


def _checkpoint_available() -> bool:
    if not _CKPT_DIR:
        return False
    index = os.path.join(_CKPT_DIR, _INDEX_NAME)
    if not os.path.isfile(index):
        return False
    with open(index) as reader:
        shard_names = set(json.load(reader).get("weight_map", {}).values())
    return len(shard_names) == 131 and all(
        os.path.isfile(os.path.join(_CKPT_DIR, name)) for name in shard_names
    )


class _RecordingTensorSource(TensorSource):
    """Record the payload boundary while delegating to the production source."""

    def __init__(self, database: CkptDatabase):
        self._base = DatabaseTensorSource(database)
        self.requests: list[str] = []
        self.bytes_by_key: dict[str, int] = {}

    def load_tensor(self, name, data_type=torch.float16):
        tensors = self._base.load_tensor(name, data_type)
        self.requests.append(name)
        self.bytes_by_key[name] = sum(
            tensor.numel() * tensor.element_size() for tensor in tensors
        )
        return tensors

    def has_tensor(self, name: str) -> bool:
        return self._base.has_tensor(name)

    def get_database(self):
        return self._base.get_database()


def _parallelism_config(rank: int) -> ParallelismConfig:
    config = ParallelismConfig()
    config.world_rank = rank
    config.world_size = _WORLD_SIZE
    config.local_rank = rank
    config.local_world_size = _WORLD_SIZE
    config.tp_size = _WORLD_SIZE
    config.tp_rank = rank
    config.dp_size = 1
    config.dp_rank = 0
    config.ep_size = 1
    config.ep_rank = 0
    return config


class _IdentityExportedDevice:
    def maybe_rewrite_weight_by_key(self, _name: str, tensor: torch.Tensor):
        return tensor


def _ple_descriptors(
    database: CkptDatabase, parallelism_config: ParallelismConfig
) -> tuple[Qwen4ExpPleNgramWeight, dict, LoadConfig]:
    os.environ["RTP_LLM_ENABLE_QWEN4_EXP_PLE"] = "true"
    os.environ["RTP_LLM_ENABLE_QWEN4_EXP_QSA"] = "false"
    model_config = Qwen4Exp.create_config(_CKPT_DIR)
    weight = Qwen4ExpWeight(
        model_config=model_config,
        parallelism_config=parallelism_config,
        hw_kernel_config=HWKernelConfig(),
        kv_cache_config=KVCacheConfig(),
    )
    weight._process_meta({}, database.get_pretrain_tensor_names())
    layer = weight._get_weight_info().layer_weights[_PLE_LAYER_INDEX]
    ngram = [item for item in layer if isinstance(item, Qwen4ExpPleNgramWeight)]
    if len(ngram) != 1:
        raise AssertionError(f"expected one PLE n-gram descriptor, got {len(ngram)}")
    descriptors = {item.name: item for item in layer}
    missing = set(_METADATA_NAMES) - descriptors.keys()
    if missing:
        raise AssertionError(
            f"PLE hash metadata descriptors missing: {sorted(missing)}"
        )
    load_config = weight.create_load_config(
        compute_dtype=torch.bfloat16,
        database=database,
        exported_device=_IdentityExportedDevice(),
    )
    return ngram[0], descriptors, load_config


def _manual_hash_ids(
    token_history: list[int],
    multipliers: list[int],
    offsets: list[int],
    vocab_sizes: list[int],
    ngram_size: int,
) -> list[int]:
    """Independent reference for the single-token lookup used by this test."""
    heads_per_ngram = len(vocab_sizes) // (ngram_size - 1)
    result = []
    for ngram in range(2, ngram_size + 1):
        mixed = token_history[-1] * multipliers[0]
        for position in range(1, ngram):
            mixed ^= token_history[-1 - position] * multipliers[position]
        start = (ngram - 2) * heads_per_ngram
        for head in range(start, start + heads_per_ngram):
            result.append(mixed % vocab_sizes[head] + offsets[head])
    return result


def _init_nccl(rank: int, port: int, parallelism_config: ParallelismConfig) -> None:
    base_port = port + 11
    init_distributed_environment(
        parallelism_config,
        nccl_comm_config=NcclCommConfig(
            nccl_ip="127.0.0.1",
            tp_nccl_port=base_port - 2,
            dp_tp_nccl_port=base_port - 10,
            ffn_tp_nccl_port=base_port - 5,
        ),
        nccl_init_port=port,
        backend="nccl",
        timeout=600,
    )


def _worker(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    parallelism_config = _parallelism_config(rank)
    database = CkptDatabase(_CKPT_DIR)
    source = _RecordingTensorSource(database)
    ngram_descriptor, descriptors, load_config = _ple_descriptors(
        database, parallelism_config
    )

    local_indices = tuple(range(rank * _SHARDS_PER_RANK, (rank + 1) * _SHARDS_PER_RANK))
    expected_table_keys = [
        ngram_descriptor.weights[index].tensor_name(_PLE_LAYER_INDEX)
        for index in local_indices
    ]
    if ngram_descriptor.local_shard_indices(load_config) != local_indices:
        raise AssertionError("descriptor ownership does not match the TP8 partition")

    # Any generic aggregation during table loading is a correctness failure: it
    # would require a second table-sized allocation, and a global aggregation
    # would materialize the full 102.4 GB table on every rank.
    aggregation_error = AssertionError("PLE table loading must not aggregate shards")
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    with mock.patch.object(
        torch, "stack", side_effect=aggregation_error
    ), mock.patch.object(
        torch, "cat", side_effect=aggregation_error
    ), mock.patch.object(
        torch, "concat", side_effect=aggregation_error
    ), mock.patch.object(
        torch, "concatenate", side_effect=aggregation_error
    ):
        loaded = ngram_descriptor.load(
            source, _PLE_LAYER_INDEX, str(device), load_config
        )
    torch.cuda.synchronize(device)
    peak_delta = torch.cuda.max_memory_allocated(device) - allocated_before

    if source.requests != expected_table_keys:
        raise AssertionError(
            f"rank {rank} requested unexpected table keys: {source.requests}"
        )
    table_bytes = sum(source.bytes_by_key[key] for key in expected_table_keys)
    if len(expected_table_keys) != _SHARDS_PER_RANK:
        raise AssertionError(f"rank {rank} did not request exactly 16 table shards")
    if set(source.bytes_by_key.values()) != {_EXPECTED_SHARD_BYTES}:
        raise AssertionError(
            f"rank {rank} table tensor sizes differ: {source.bytes_by_key}"
        )
    if table_bytes != _EXPECTED_RANK_TABLE_BYTES:
        raise AssertionError(
            f"rank {rank} loaded {table_bytes} table bytes, "
            f"expected {_EXPECTED_RANK_TABLE_BYTES}"
        )
    # Permit one in-flight shard-sized transfer on top of resident tensors.
    if peak_delta > table_bytes + _EXPECTED_SHARD_BYTES:
        raise AssertionError(
            f"rank {rank} PLE load peak grew by {peak_delta} bytes for "
            f"{table_bytes} resident bytes"
        )

    local_shards = [
        loaded[f"{W.qwen4_ple_ngram_shards}.{index}"] for index in local_indices
    ]
    metadata = {}
    for name in _METADATA_NAMES:
        result = descriptors[name].load(
            source, _PLE_LAYER_INDEX, str(device), load_config
        )
        metadata[name] = result[name]

    expected_all_keys = expected_table_keys + [
        descriptors[name].weights[0].tensor_name(_PLE_LAYER_INDEX)
        for name in _METADATA_NAMES
    ]
    if source.requests != expected_all_keys:
        raise AssertionError(
            f"rank {rank} loaded tensors outside the PLE table/hash metadata: "
            f"{source.requests}"
        )
    if any(metadata[name].dtype != torch.int64 for name in _METADATA_NAMES):
        raise AssertionError("PLE hash metadata was not preserved as int64")

    with open(os.path.join(_CKPT_DIR, _INDEX_NAME)) as reader:
        weight_map = json.load(reader)["weight_map"]
    physical_files = sorted({weight_map[key] for key in expected_table_keys})
    if len(physical_files) != 5:
        raise AssertionError(
            f"rank {rank} table tensors unexpectedly span {physical_files}"
        )

    _init_nccl(rank, port, parallelism_config)
    try:
        embedding = Qwen4ExpNGramEmbedding(
            local_shards,
            metadata[W.qwen4_ple_ngram_vocab_sizes],
            metadata[W.qwen4_ple_ngram_offsets],
            metadata[W.qwen4_ple_multipliers],
            ngram_size=3,
            eos_token_id=248044,
            shard_indices=list(local_indices),
            total_shards=_TOTAL_SHARDS,
            distributed_reduce=True,
        )

        # With the released hash metadata, [1, 2, 3] sends exactly two of the
        # 16 n-gram heads to every TP rank.  This makes every rank contribute to
        # the result and catches both incorrect ownership and a missing reduce.
        token_history = torch.tensor([[1, 2, 3]], dtype=torch.int64, device=device)
        reference_ids = _manual_hash_ids(
            [1, 2, 3],
            metadata[W.qwen4_ple_multipliers].cpu().tolist(),
            metadata[W.qwen4_ple_ngram_offsets].cpu().tolist(),
            metadata[W.qwen4_ple_ngram_vocab_sizes].cpu().tolist(),
            ngram_size=3,
        )
        reference_ids_tensor = torch.tensor(
            reference_ids, dtype=torch.int64, device=device
        ).view(1, 1, -1)
        torch.testing.assert_close(
            embedding.hashed_ids(token_history, seq_len=1),
            reference_ids_tensor,
            rtol=0,
            atol=0,
        )

        shard_rows = local_shards[0].shape[0]
        owned = [
            (position, value)
            for position, value in enumerate(reference_ids)
            if value // shard_rows in local_indices
        ]
        if len(owned) != 2:
            raise AssertionError(f"rank {rank} owns {len(owned)} lookup heads, not 2")

        manual_local = torch.zeros(
            len(reference_ids),
            local_shards[0].shape[1],
            dtype=local_shards[0].dtype,
            device=device,
        )
        for position, value in owned:
            global_shard = value // shard_rows
            row = value % shard_rows
            manual_local[position].copy_(
                local_shards[global_shard - local_indices[0]][row]
            )

        actual = embedding(token_history, seq_len=1)
        torch.distributed.all_reduce(manual_local, op=torch.distributed.ReduceOp.SUM)
        expected = manual_local.reshape(1, 1, -1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        ownership = torch.tensor(
            [
                local_indices[0],
                local_indices[-1],
                len(expected_table_keys),
                table_bytes,
            ],
            dtype=torch.int64,
            device=device,
        )
        gathered = [torch.empty_like(ownership) for _ in range(_WORLD_SIZE)]
        torch.distributed.all_gather(gathered, ownership)
        expected_ownership = [
            [
                other_rank * _SHARDS_PER_RANK,
                (other_rank + 1) * _SHARDS_PER_RANK - 1,
                _SHARDS_PER_RANK,
                _EXPECTED_RANK_TABLE_BYTES,
            ]
            for other_rank in range(_WORLD_SIZE)
        ]
        if [item.cpu().tolist() for item in gathered] != expected_ownership:
            raise AssertionError(
                "TP8 ownership reports are not a full disjoint partition"
            )

        print(
            json.dumps(
                {
                    "rank": rank,
                    "table_keys": expected_table_keys,
                    "table_bytes": table_bytes,
                    "physical_files": physical_files,
                    "peak_cuda_bytes_over_baseline": peak_delta,
                    "metadata_bytes": {
                        key: source.bytes_by_key[tensor_key]
                        for key, tensor_key in zip(
                            _METADATA_NAMES, expected_all_keys[-len(_METADATA_NAMES) :]
                        )
                    },
                    "lookup_shards": [value // shard_rows for _, value in owned],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        torch.distributed.barrier()
        torch.cuda.synchronize(device)
    finally:
        if torch.distributed.is_initialized():
            destroy_distributed_environment()


@unittest.skipUnless(
    _checkpoint_available(),
    "QWEN4_EXP_CKPT_DIR must contain the complete 131-shard checkpoint",
)
class Qwen4ExpRealPleTP8LoadingTest(unittest.TestCase):
    def test_rank_local_load_and_distributed_lookup(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE:
            self.skipTest("Qwen4Exp real PLE validation requires eight CUDA devices")
        mp.set_start_method("spawn", force=True)
        with PortsContext(num_ports=1, ttl=3600) as ports:
            mp.spawn(
                _worker,
                args=(ports[0],),
                nprocs=_WORLD_SIZE,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
