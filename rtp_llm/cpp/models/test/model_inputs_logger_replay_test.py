import io
import os
import shlex
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

DEFAULT_DEEPSEEK_V4_FLASH_MODEL_PATH = "/data3/DeepSeekV4-Flash"


def _iter_model_inputs_dumps(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        [*path.rglob("*.pt"), *path.rglob("*.ptlog")],
        key=lambda p: str(p.relative_to(path)),
    )


def _load_model_inputs_dump_records(path: Path) -> list[dict]:
    def load_framed_records() -> list[dict]:
        records = []
        with path.open("rb") as f:
            while True:
                length_bytes = f.read(8)
                if not length_bytes:
                    break
                if len(length_bytes) != 8:
                    raise ValueError(f"incomplete record length in {path}")
                (record_size,) = struct.unpack("<Q", length_bytes)
                record = f.read(record_size)
                if len(record) != record_size:
                    raise ValueError(f"incomplete record payload in {path}")
                records.append(
                    torch.load(
                        io.BytesIO(record), map_location="cpu", weights_only=False
                    )
                )
        return records

    try:
        records = load_framed_records()
        if records:
            return records
    except Exception:
        if path.suffix == ".ptlog":
            raise

    return [torch.load(path, map_location="cpu", weights_only=False)]


def _runfile_path(relative_path: str) -> str:
    test_srcdir = Path(os.environ["TEST_SRCDIR"])
    workspace = os.environ.get("TEST_WORKSPACE", "rtp_llm")
    candidates = [
        test_srcdir / workspace / relative_path,
        test_srcdir / relative_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"runfile not found: {relative_path}")


def _is_tensor(value) -> bool:
    return isinstance(value, torch.Tensor)


def _is_replayable_payload(payload: dict) -> bool:
    return (
        not bool(payload.get("warmup", False))
        and not bool(payload.get("skip_run", False))
        and _is_tensor(payload.get("kv_cache_block_id"))
        and _is_tensor(payload.get("combo_tokens"))
        and _is_tensor(payload.get("input_lengths"))
        and _is_tensor(payload.get("sequence_lengths"))
    )


def _to_cpu_i32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(device="cpu", dtype=torch.int32).contiguous()


def _to_cuda_i32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(device="cuda", dtype=torch.int32).contiguous()


def _maybe_cuda(tensor) -> torch.Tensor:
    if not _is_tensor(tensor):
        return torch.empty(0, device="cuda")
    return tensor.to(device="cuda").contiguous()


def _build_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    lengths = _to_cpu_i32(lengths)
    return torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(lengths, dim=0)]
    ).to("cuda")


def _expand_kernel_block_ids(
    block_ids: torch.Tensor, kernel_blocks_per_kv_block: int
) -> torch.Tensor:
    if kernel_blocks_per_kv_block <= 1:
        return block_ids.clone()
    offsets = torch.arange(kernel_blocks_per_kv_block, dtype=torch.int32).view(
        *([1] * block_ids.dim()), kernel_blocks_per_kv_block
    )
    expanded = block_ids.unsqueeze(-1) * kernel_blocks_per_kv_block + offsets
    expanded = torch.where(
        block_ids.unsqueeze(-1) == -1, torch.full_like(expanded, -1), expanded
    )
    return expanded.reshape(
        *block_ids.shape[:-1], block_ids.shape[-1] * kernel_blocks_per_kv_block
    )


def _pad_last_dim(tensor: torch.Tensor, size: int) -> torch.Tensor:
    if tensor.size(-1) == size:
        return tensor
    padded = torch.zeros(*tensor.shape[:-1], size, dtype=tensor.dtype)
    padded[..., : tensor.size(-1)] = tensor
    return padded


def _derive_kv_cache_kernel_block_id(payload: dict) -> torch.Tensor:
    block_id = payload["kv_cache_block_id"]
    if not _is_tensor(block_id) or block_id.numel() == 0:
        return torch.empty(0, dtype=torch.int32)

    block_id = _to_cpu_i32(block_id)
    seq_size_per_block = int(payload["seq_size_per_block"])
    kernel_seq_size_per_block = int(payload["kernel_seq_size_per_block"])
    full_group_bpk = 1
    if kernel_seq_size_per_block > 0:
        full_group_bpk = max(1, seq_size_per_block // kernel_seq_size_per_block)

    group_types = payload.get("kv_cache_group_types")
    group_types = (
        _to_cpu_i32(group_types).flatten()
        if _is_tensor(group_types)
        else torch.empty(0, dtype=torch.int32)
    )

    if block_id.dim() != 3:
        group_type = int(group_types[0].item()) if group_types.numel() else 1
        return _expand_kernel_block_ids(
            block_id, full_group_bpk if group_type == 1 else 1
        )

    groups = []
    kernel_block_count = block_id.size(-1) * full_group_bpk
    for group_idx in range(block_id.size(0)):
        group_type = (
            int(group_types[group_idx].item()) if group_idx < group_types.numel() else 1
        )
        group_kernel_blocks = _expand_kernel_block_ids(
            block_id[group_idx], full_group_bpk if group_type == 1 else 1
        )
        groups.append(_pad_last_dim(group_kernel_blocks, kernel_block_count))
    return torch.stack(groups, dim=0)


def _replay_one_dump_with_auto_model(auto_model, payload: dict) -> None:
    from rtp_llm.ops.compute_ops import (
        BertEmbeddingInputs,
        PyAttentionInputs,
        PyModelInputs,
        get_typemeta,
    )

    input_lengths = _to_cpu_i32(payload["input_lengths"])
    sequence_lengths = _to_cpu_i32(payload["sequence_lengths"])
    prefix_lengths = _to_cpu_i32(payload["prefix_lengths"])
    combo_tokens = _to_cuda_i32(payload["combo_tokens"])

    attn = PyAttentionInputs()
    attn.input_lengths = _to_cuda_i32(input_lengths)
    attn.sequence_lengths = _to_cuda_i32(sequence_lengths)
    attn.prefix_lengths = _to_cuda_i32(prefix_lengths)
    attn.is_prefill = sequence_lengths.numel() == 0
    attn.is_target_verify = bool(payload["is_target_verify"])
    attn.dtype = get_typemeta(auto_model.kv_cache.kv_cache_base_by_layer[0])

    if attn.is_prefill:
        context_lengths = input_lengths - prefix_lengths
        attn.total_tokens = int(combo_tokens.numel())
        attn.context_total_kv_length = int(input_lengths.sum().item())
        attn.cu_seqlens = _build_cu_seqlens(context_lengths)
        attn.cu_kv_seqlens = _build_cu_seqlens(input_lengths)
        attn.padding_offset = torch.zeros(
            attn.total_tokens, dtype=torch.int32, device="cuda"
        )
    else:
        batch_size = int(input_lengths.numel())
        attn.total_tokens = 0
        attn.cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
        attn.cu_kv_seqlens = torch.zeros(
            batch_size + 1, dtype=torch.int32, device="cuda"
        )
        attn.decode_cu_seqlens_d = torch.arange(
            batch_size + 1, dtype=torch.int32, device="cuda"
        )
        attn.padding_offset = torch.zeros(
            int(combo_tokens.numel()), dtype=torch.int32, device="cuda"
        )

    block_id = payload["kv_cache_block_id"]
    if _is_tensor(block_id) and block_id.numel():
        block_id_host = _to_cpu_i32(block_id).pin_memory()
        attn.kv_cache_block_id_host = block_id_host
        attn.kv_cache_block_id_device = block_id_host.to("cuda", non_blocking=True)

    kernel_block_id = payload.get("kv_cache_kernel_block_id")
    if not _is_tensor(kernel_block_id) or kernel_block_id.numel() == 0:
        kernel_block_id = _derive_kv_cache_kernel_block_id(payload)
    if _is_tensor(kernel_block_id) and kernel_block_id.numel():
        kernel_host = _to_cpu_i32(kernel_block_id).pin_memory()
        if kernel_host.dim() == 3:
            attn.kv_cache_kernel_block_id_host = kernel_host[0]
            attn.kv_cache_kernel_block_id_device_by_group = [
                kernel_host[group].to("cuda", non_blocking=True)
                for group in range(kernel_host.size(0))
            ]
            attn.kv_cache_kernel_block_id_device = (
                attn.kv_cache_kernel_block_id_device_by_group[0]
            )
        else:
            attn.kv_cache_kernel_block_id_host = kernel_host
            attn.kv_cache_kernel_block_id_device = kernel_host.to(
                "cuda", non_blocking=True
            )

    if _is_tensor(payload.get("kv_cache_layer_to_group")):
        attn.kv_cache_layer_to_group = _to_cuda_i32(payload["kv_cache_layer_to_group"])
    if _is_tensor(payload.get("sequence_lengths_plus_1")):
        attn.sequence_lengths_plus_1_d = _to_cuda_i32(
            payload["sequence_lengths_plus_1"]
        )

    bert_inputs = BertEmbeddingInputs()
    if _is_tensor(payload.get("combo_position_ids")):
        bert_inputs.combo_position_ids = _to_cuda_i32(payload["combo_position_ids"])
    if _is_tensor(payload.get("combo_tokens_type_ids")):
        bert_inputs.combo_tokens_type_ids = _to_cuda_i32(
            payload["combo_tokens_type_ids"]
        )

    input_hiddens = _maybe_cuda(payload.get("last_hidden_states"))
    model_inputs = PyModelInputs(combo_tokens, input_hiddens, attn, bert_inputs)
    fmha_impl = None
    if hasattr(auto_model.model, "prepare_fmha_impl"):
        fmha_impl = auto_model.model.prepare_fmha_impl(model_inputs, False)
    auto_model.model.forward(model_inputs, fmha_impl)


def _replay_dumps_with_auto_model(path: Path) -> None:
    from rtp_llm.models_py.standalone.auto_model import AutoModel

    limit = int(os.environ.get("MODEL_INPUTS_REPLAY_LIMIT", "0"))
    payloads = []
    for dump in _iter_model_inputs_dumps(path):
        for payload in _load_model_inputs_dump_records(dump):
            if not _is_replayable_payload(payload):
                continue
            payloads.append(payload)
            if limit > 0 and len(payloads) >= limit:
                break
        if limit > 0 and len(payloads) >= limit:
            break
    if not payloads:
        raise AssertionError(f"no replayable model inputs dumps found under {path}")

    max_total_tokens = int(os.environ.get("MODEL_INPUTS_REPLAY_MAX_TOTAL_TOKENS", "0"))
    if max_total_tokens <= 0:
        max_total_tokens = 0
        for payload in payloads:
            if _is_tensor(payload["input_lengths"]):
                max_total_tokens = max(
                    max_total_tokens, int(payload["input_lengths"].max().item())
                )
            if (
                _is_tensor(payload["sequence_lengths"])
                and payload["sequence_lengths"].numel()
            ):
                max_total_tokens = max(
                    max_total_tokens, int(payload["sequence_lengths"].max().item()) + 1
                )
        max_total_tokens = max(max_total_tokens, 1)

    model_path = os.environ.get(
        "MODEL_INPUTS_REPLAY_MODEL_PATH", DEFAULT_DEEPSEEK_V4_FLASH_MODEL_PATH
    )
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"model path not found: {model_path}; set MODEL_INPUTS_REPLAY_MODEL_PATH"
        )
    hack_layer_num = int(os.environ.get("MODEL_INPUTS_REPLAY_HACK_LAYER_NUM", "6"))
    first_payload = payloads[0]
    auto_model = AutoModel.from_pretrained(
        model_path,
        max_total_tokens=max_total_tokens,
        tokens_per_block=int(
            os.environ.get(
                "MODEL_INPUTS_REPLAY_TOKENS_PER_BLOCK",
                str(int(first_payload["seq_size_per_block"])),
            )
        ),
        kernel_tokens_per_block=int(
            os.environ.get(
                "MODEL_INPUTS_REPLAY_KERNEL_TOKENS_PER_BLOCK",
                str(int(first_payload["kernel_seq_size_per_block"])),
            )
        ),
        hack_layer_num=hack_layer_num,
        fp8_kv_cache=os.environ.get("MODEL_INPUTS_REPLAY_FP8_KV_CACHE", "1") != "0",
        init_tokenizer_and_lm_head=False,
    )
    for payload in payloads:
        _replay_one_dump_with_auto_model(auto_model, payload)


def _validate_model_inputs_dump(path: Path) -> dict:
    records = _load_model_inputs_dump_records(path)
    assert len(records) > 0
    payload = records[0]
    expected_keys = {
        "trace_ids",
        "combo_tokens",
        "input_lengths",
        "sequence_lengths",
        "lm_output_indexes",
        "prefix_lengths",
        "combo_tokens_type_ids",
        "combo_position_ids",
        "last_hidden_states",
        "attention_mask",
        "kv_cache_block_id",
        "kv_cache_layer_to_group",
        "kv_cache_group_types",
        "kv_cache_update_mapping",
        "request_id",
        "request_pd_separation",
        "kv_block_stride_bytes",
        "kv_scale_stride_bytes",
        "seq_size_per_block",
        "kernel_seq_size_per_block",
        "pd_separation",
        "decode_entrance",
        "need_all_logits",
        "need_moe_gating",
        "warmup",
        "skip_run",
        "is_fake_stream",
        "is_target_verify",
    }
    assert set(payload.keys()) == expected_keys
    assert payload["combo_tokens"].tolist() == [11, 12, 13]
    assert payload["last_hidden_states"] is None
    assert payload["attention_mask"] is None
    assert tuple(payload["kv_cache_block_id"].shape) == (1, 1, 2)
    kernel_block_id = _derive_kv_cache_kernel_block_id(payload)
    assert tuple(kernel_block_id.shape) == (1, 1, 4)
    assert kernel_block_id.tolist() == [[[14, 15, 16, 17]]]
    return payload


class ModelInputsLoggerReplayTest(unittest.TestCase):
    def test_generated_dump_can_be_loaded_by_torch(self) -> None:
        tool = _runfile_path("rtp_llm/cpp/models/test/model_inputs_logger_dump_tool")
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.check_call([tool, temp_dir, "2"])
            dumps = sorted(Path(temp_dir).glob("model_inputs_r2_s3*.pt"))
            self.assertEqual(1, len(dumps))
            records = _load_model_inputs_dump_records(dumps[0])
            self.assertEqual(2, len(records))
            payload = _validate_model_inputs_dump(dumps[0])
            self.assertEqual(["trace-a", "trace-b"], payload["trace_ids"])

    def test_optional_external_dump_forward_replay(self) -> None:
        replay_path = os.environ.get("MODEL_INPUTS_DUMP_PATH")
        if not replay_path:
            self.skipTest("MODEL_INPUTS_DUMP_PATH is not set")

        path = Path(replay_path)
        dumps = _iter_model_inputs_dumps(path)
        self.assertGreater(len(dumps), 0)
        replayable_count = 0
        for dump in dumps:
            for payload in _load_model_inputs_dump_records(dump):
                if not _is_replayable_payload(payload):
                    continue
                replayable_count += 1
                self.assertTrue(_is_tensor(_derive_kv_cache_kernel_block_id(payload)))
        self.assertGreater(replayable_count, 0)

        replay_cmd = os.environ.get("MODEL_INPUTS_FORWARD_REPLAY_CMD")
        if replay_cmd:
            env = os.environ.copy()
            env["MODEL_INPUTS_DUMP_PATH"] = str(path)
            subprocess.check_call(shlex.split(replay_cmd), env=env)
        elif (
            os.environ.get("MODEL_INPUTS_REPLAY_MODEL_PATH")
            or Path(DEFAULT_DEEPSEEK_V4_FLASH_MODEL_PATH).exists()
        ):
            _replay_dumps_with_auto_model(path)
        else:
            self.fail(
                "MODEL_INPUTS_FORWARD_REPLAY_CMD or MODEL_INPUTS_REPLAY_MODEL_PATH is not set, "
                f"and default model path {DEFAULT_DEEPSEEK_V4_FLASH_MODEL_PATH} does not exist; "
                "set one of them to exercise real model forward replay"
            )


if __name__ == "__main__":
    unittest.main()
