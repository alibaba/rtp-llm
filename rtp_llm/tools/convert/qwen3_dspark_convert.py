"""Convert a TorchSpec Qwen3 DSpark DCP checkpoint to RTP-LLM safetensors.

The converter deliberately uses only PyTorch's public distributed-checkpoint
reader.  TorchSpec is a training dependency and is neither imported nor needed
to read its CPU DCP artifact.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file

_DCP_PREFIX = "model_state.model.draft_model."
_BF16 = torch.bfloat16


def _load_json(path: Path) -> Dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"config {path} must contain a JSON object")
    return value


def _positive_int(config: Mapping, name: str) -> int:
    value = config.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"config field {name!r} must be a positive integer")
    return value


def _target_layer_ids(config: Mapping) -> Tuple[int, ...]:
    values = config.get("target_layer_ids")
    if not isinstance(values, list) or not values:
        raise ValueError("config field 'target_layer_ids' must be a non-empty list")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ):
        raise ValueError(
            "config field 'target_layer_ids' must contain non-negative integers"
        )
    if list(values) != sorted(set(values)):
        raise ValueError("config field 'target_layer_ids' must be strictly increasing")
    return tuple(values)


def expected_tensor_shapes(config: Mapping) -> Dict[str, Tuple[int, ...]]:
    """Return the complete, strict TorchSpec DSpark parameter contract."""
    hidden = _positive_int(config, "hidden_size")
    intermediate = _positive_int(config, "intermediate_size")
    layers = _positive_int(config, "num_hidden_layers")
    heads = _positive_int(config, "num_attention_heads")
    kv_heads = _positive_int(config, "num_key_value_heads")
    head_dim = _positive_int(config, "head_dim")
    vocab = _positive_int(config, "vocab_size")
    markov_rank = _positive_int(config, "markov_rank")
    target_ids = _target_layer_ids(config)
    target_hidden = _positive_int(config, "target_hidden_size")
    target_layers = _positive_int(config, "target_num_hidden_layers")
    declared_target_layers = _positive_int(config, "num_target_layers")
    if config.get("model_type") != "dspark":
        raise ValueError("TorchSpec config model_type must be 'dspark'")
    if config.get("architectures") != ["DSparkDraftModel"]:
        raise ValueError("TorchSpec config architectures must be ['DSparkDraftModel']")
    if (
        not isinstance(config.get("rms_norm_eps"), (int, float))
        or isinstance(config["rms_norm_eps"], bool)
        or config["rms_norm_eps"] <= 0
    ):
        raise ValueError("config field 'rms_norm_eps' must be positive")
    mask_token_id = config.get("mask_token_id")
    if (
        not isinstance(mask_token_id, int)
        or isinstance(mask_token_id, bool)
        or not 0 <= mask_token_id < vocab
    ):
        raise ValueError("config field 'mask_token_id' must be a valid vocabulary id")
    if heads * head_dim != hidden:
        raise ValueError("num_attention_heads * head_dim must equal hidden_size")
    if heads % kv_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    _positive_int(config, "block_size")
    if target_hidden != hidden:
        raise ValueError("target_hidden_size must equal hidden_size")
    if declared_target_layers != target_layers:
        raise ValueError("num_target_layers must equal target_num_hidden_layers")
    if target_ids[-1] >= target_layers:
        raise ValueError("target_layer_ids must be within target_num_hidden_layers")
    layer_types = config.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != layers:
        raise ValueError("layer_types must list one entry per draft layer")
    if any(layer_type != "full_attention" for layer_type in layer_types):
        raise ValueError("TorchSpec Qwen3 DSpark requires full_attention draft layers")
    if (
        config.get("use_sliding_window") is not False
        or config.get("sliding_window") is not None
    ):
        raise ValueError("TorchSpec Qwen3 DSpark must disable sliding-window attention")
    if config.get("hidden_act") != "silu" or config.get("attention_bias") is not False:
        raise ValueError("TorchSpec Qwen3 DSpark requires SiLU without attention bias")
    if config.get("markov_head_type") != "vanilla":
        raise ValueError("TorchSpec Qwen3 DSpark requires vanilla Markov head")
    if config.get("tie_word_embeddings") is not False:
        raise ValueError("TorchSpec Qwen3 DSpark must not tie word embeddings")
    if config.get("enable_confidence_head") is not True:
        raise ValueError("TorchSpec DSpark checkpoint must enable the confidence head")
    if config.get("confidence_head_with_markov") is not True:
        raise ValueError(
            "TorchSpec DSpark confidence head must include Markov features"
        )

    shapes = {
        "context_proj.weight": (hidden, len(target_ids) * hidden),
        "context_norm.weight": (hidden,),
        "embed_tokens.weight": (vocab, hidden),
        "final_norm.weight": (hidden,),
        "markov_head.markov_w1.weight": (vocab, markov_rank),
        "markov_head.markov_w2.weight": (vocab, markov_rank),
        "confidence_head.proj.weight": (1, hidden + markov_rank),
        "confidence_head.proj.bias": (1,),
    }
    for layer in range(layers):
        prefix = f"layers.{layer}."
        shapes.update(
            {
                prefix + "self_attn.q_proj.weight": (heads * head_dim, hidden),
                prefix + "self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
                prefix + "self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
                prefix + "self_attn.o_proj.weight": (hidden, heads * head_dim),
                prefix + "self_attn.q_norm.weight": (head_dim,),
                prefix + "self_attn.k_norm.weight": (head_dim,),
                prefix + "mlp.gate_proj.weight": (intermediate, hidden),
                prefix + "mlp.up_proj.weight": (intermediate, hidden),
                prefix + "mlp.down_proj.weight": (hidden, intermediate),
                prefix + "input_layernorm.weight": (hidden,),
                prefix + "post_attention_layernorm.weight": (hidden,),
            }
        )
    return shapes


def _map_key(key: str) -> str:
    if key == "context_proj.weight":
        return "fc.weight"
    if key == "context_norm.weight":
        return "hidden_norm.weight"
    if key == "final_norm.weight":
        return "model.norm.weight"
    if key == "embed_tokens.weight" or key.startswith("layers."):
        return "model." + key
    return key


def _read_dcp(source_dir: Path, config: Mapping) -> Dict[str, torch.Tensor]:
    expected = expected_tensor_shapes(config)
    reader = dcp.FileSystemReader(str(source_dir))
    metadata = reader.read_metadata()
    entries = metadata.state_dict_metadata
    actual_fqns = set(entries)
    expected_fqns = {_DCP_PREFIX + key for key in expected}
    if actual_fqns != expected_fqns:
        missing = sorted(expected_fqns - actual_fqns)
        unknown = sorted(actual_fqns - expected_fqns)
        raise ValueError(f"DCP keys mismatch: missing={missing}, unknown={unknown}")

    state: Dict = {"model_state": {"model": {"draft_model": {}}}}
    draft = state["model_state"]["model"]["draft_model"]
    for key, shape in expected.items():
        item = entries[_DCP_PREFIX + key]
        if getattr(getattr(item, "properties", None), "dtype", None) != _BF16:
            raise ValueError(f"DCP tensor {_DCP_PREFIX + key} must be bfloat16")
        item_shape = tuple(int(dim) for dim in item.size)
        if item_shape != shape:
            raise ValueError(
                f"DCP tensor {_DCP_PREFIX + key} shape mismatch: "
                f"expected={shape}, actual={item_shape}"
            )
        # TorchSpec writes draft_model as a state-dict whose keys contain dots;
        # DCP's FQN is nevertheless model_state.model.draft_model.<key>.
        draft[key] = torch.empty(shape, dtype=_BF16, device="cpu")
    dcp.load(state_dict=state, storage_reader=reader)

    loaded: Dict[str, torch.Tensor] = {}
    for key in expected:
        node = draft[key]
        if node.dtype != _BF16 or tuple(node.shape) != expected[key]:
            raise ValueError(f"DCP loader returned invalid tensor for {key}")
        loaded[_map_key(key)] = node.contiguous()
    return loaded


def _output_config(source: Mapping) -> Dict:
    result = dict(source)
    # sample_from_anchor changes the draft query position semantics at
    # inference time, so it must be a contract of the conversion rather than a
    # silently forced default: reject source configs that train without anchor
    # sampling instead of silently flipping them.
    if source.get("sample_from_anchor") not in (None, True):
        raise ValueError(
            "TorchSpec DSpark conversion requires anchor sampling, got "
            f"sample_from_anchor={source.get('sample_from_anchor')!r}"
        )
    result.update(
        {
            "architectures": ["Qwen3DSparkForCausalLM"],
            "model_type": "qwen_3_dspark",
            "aux_hidden_state_layer_ids": list(_target_layer_ids(source)),
            "sample_from_anchor": True,
            "lm_head_source": "target",
            "dtype": "bfloat16",
            "torch_dtype": "bfloat16",
        }
    )
    # This field would make Qwen3DSpark subtract one from already zero-based ids.
    result.pop("speculators_model_type", None)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_safetensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    with safe_open(str(path), framework="pt", device="cpu") as saved:
        if set(saved.keys()) != set(tensors):
            raise ValueError("safetensors key set differs from converted DCP tensors")
        for key, tensor in tensors.items():
            reread = saved.get_tensor(key)
            if (
                reread.dtype != tensor.dtype
                or tuple(reread.shape) != tuple(tensor.shape)
                or not torch.equal(
                    reread.contiguous().view(torch.uint8),
                    tensor.contiguous().view(torch.uint8),
                )
            ):
                raise ValueError(f"bitwise safetensors roundtrip failed for {key}")


def convert(source_dir: str, draft_config: str, output_dir: str) -> Path:
    """Convert one DCP directory.  The destination must not already exist."""
    source = Path(source_dir)
    config_path = Path(draft_config)
    destination = Path(output_dir)
    if not source.is_dir():
        raise ValueError(f"DCP source directory does not exist: {source}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"output parent directory does not exist: {destination.parent}"
        )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    config = _load_json(config_path)
    # Reject non-anchor sampling before writing any output artifact.
    if config.get("sample_from_anchor") not in (None, True):
        raise ValueError(
            "TorchSpec DSpark conversion requires anchor sampling, got "
            f"sample_from_anchor={config.get('sample_from_anchor')!r}"
        )
    tensors = _read_dcp(source, config)

    temp = Path(
        tempfile.mkdtemp(prefix=destination.name + ".tmp-", dir=destination.parent)
    )
    try:
        weights_path = temp / "model.safetensors"
        save_file(tensors, str(weights_path))
        _verify_safetensors(weights_path, tensors)
        (temp / "config.json").write_text(
            json.dumps(_output_config(config), indent=2, sort_keys=True) + "\n"
        )
        index = {
            "metadata": {
                "total_size": sum(
                    t.numel() * t.element_size() for t in tensors.values()
                )
            },
            "weight_map": {key: "model.safetensors" for key in sorted(tensors)},
        }
        (temp / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n"
        )
        confidence = sorted(
            key for key in tensors if key.startswith("confidence_head.")
        )
        report = {
            "source_dcp": str(source.resolve()),
            "source_config": str(config_path.resolve()),
            "verification": {
                "method": "dtype/shape plus torch.uint8 byte equality",
                "bitwise": True,
                "passed": True,
            },
            "weights_sha256": _sha256(weights_path),
            "tensors": [
                {
                    "source": _DCP_PREFIX + source_key,
                    "destination": dest_key,
                    "shape": list(tensors[dest_key].shape),
                    "dtype": str(tensors[dest_key].dtype),
                    "numel": tensors[dest_key].numel(),
                }
                for source_key, dest_key in sorted(
                    (key, _map_key(key)) for key in expected_tensor_shapes(config)
                )
            ],
            "unused_by_current_runtime": {
                "confidence_keys": confidence,
                "reason": "preserved checkpoint tensors; Qwen3DSparkWeight does not currently consume confidence_head",
            },
        }
        (temp / "conversion-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite existing output: {destination}"
            )
        os.replace(temp, destination)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dcp_dir")
    parser.add_argument("draft_config")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    convert(args.dcp_dir, args.draft_config, args.output_dir)


if __name__ == "__main__":
    main()
