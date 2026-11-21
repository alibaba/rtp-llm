"""Checkpoint-authoritative input table dimensions for BERT-family encoders.

Legacy encoders loaded the complete input table, independently of the advertised
vocabulary size. Keep that contract without padding missing rows or resizing a
live Parameter. All later ReadSpec and tensor copies retain exact shape checks.
"""

import logging

import torch

from rtp_llm.models_py.weight_mapper import (
    _unwrap_pytorch_state_dict,
    discover_ckpt_files,
    is_rank_local_checkpoint,
)


def static_int8_input_major(config, prefix, layer_ids, *, tp_rank, tp_size):
    """Resolve the checkpoint convention once, before reading square Q/K/V.

    Legacy calibrated BERT stores qweight as [in,out]; HF-shaped producers
    store [out,in]. A square attention projection cannot distinguish them,
    but the intermediate and output FFN matrices have opposite shapes.
    Read only headers (or PyTorch fake tensors), never checkpoint payloads.
    """
    h, f = config.hidden_size, config.inter_size
    if h == f:
        return True  # No rectangular witness: use the legacy format convention.
    expected = {}
    for layer in layer_ids:
        for proj, oi in (("intermediate", (f, h)), ("output", (h, f))):
            name = f"encoder.layer.{layer}.{proj}.dense.qweight"
            expected[name] = expected[prefix + "." + name] = oi
    layouts = set()

    def observe(name, shape):
        oi = expected[name]
        if tuple(shape) not in (oi, oi[::-1]):
            raise ValueError(
                f"Invalid static INT8 {name}: {tuple(shape)}; expected {oi} or {oi[::-1]}"
            )
        layouts.add(tuple(shape) == oi[::-1])
        if len(layouts) > 1:
            raise ValueError(
                "Inconsistent input/output-major static INT8 BERT qweights"
            )

    files = discover_ckpt_files(config.ckpt_path, tp_rank=tp_rank, tp_size=tp_size)
    if not is_rank_local_checkpoint(files):
        for path in files:
            if path.endswith(".safetensors"):
                from safetensors import safe_open

                with safe_open(path, framework="pt", device="cpu") as handle:
                    for name in expected.keys() & set(handle.keys()):
                        observe(name, handle.get_slice(name).get_shape())
            else:
                from torch._subclasses.fake_tensor import FakeTensorMode

                with FakeTensorMode():
                    state = _unwrap_pytorch_state_dict(
                        torch.load(path, map_location="cpu", weights_only=True)
                    )
                    for name in expected.keys() & state.keys():
                        observe(name, state[name].shape)
    return next(iter(layouts), True)


def checkpoint_input_vocab_size(config, prefix, *, tp_rank=0, tp_size=1):
    declared = int(config.vocab_size)
    explicit = int(getattr(config, "input_vocab_size", 0) or 0)
    names = {
        "embeddings.word_embeddings.weight",
        prefix + ".embeddings.word_embeddings.weight",
    }
    files = discover_ckpt_files(config.ckpt_path, tp_rank=tp_rank, tp_size=tp_size)
    # Rank-local/FT manifests own their parameter dimensions and import route.
    # Do not reinterpret a local table shape as a global HF table.
    if is_rank_local_checkpoint(files):
        return explicit or declared
    found = []
    for path in files:
        if path.endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in names.intersection(handle.keys()):
                    tensor = handle.get_slice(name)
                    if tensor.get_dtype() not in ("F16", "BF16", "F32", "F64"):
                        raise ValueError("BERT input embedding must be floating point")
                    found.append(tuple(tensor.get_shape()))
        else:
            # FakeTensorMode reads metadata, not a second model-sized CPU copy.
            # weights_only keeps the same restricted deserialization as loading.
            from torch._subclasses.fake_tensor import FakeTensorMode

            with FakeTensorMode():
                state = _unwrap_pytorch_state_dict(
                    torch.load(path, map_location="cpu", weights_only=True)
                )
                for name in names.intersection(state):
                    tensor = state[name]
                    if (
                        not isinstance(tensor, torch.Tensor)
                        or not tensor.is_floating_point()
                    ):
                        raise ValueError(
                            "BERT input embedding must be a floating point tensor"
                        )
                    found.append(tuple(tensor.shape))
    if not found:
        # Preserve completeness errors (and manifest-only format imports).
        return explicit or declared
    if len(found) != 1:
        raise ValueError("Ambiguous duplicate BERT input embedding sources")
    shape = found[0]
    if len(shape) != 2 or shape[0] <= 0 or shape[1] != config.hidden_size:
        raise ValueError(
            f"Invalid BERT input embedding shape {shape}; hidden size must be {config.hidden_size}"
        )
    rows = shape[0]
    if explicit and explicit != rows:
        raise ValueError(
            f"Explicit BERT input_vocab_size {explicit} disagrees with checkpoint rows {rows}"
        )
    if declared != rows:
        logging.warning(
            "BERT declared vocab_size=%d differs from checkpoint input rows=%d; "
            "using exact checkpoint input table without padding. Output heads "
            "are unchanged; input IDs must satisfy 0 <= id < %d.",
            declared,
            rows,
            rows,
        )
    return rows


def publish_input_vocab(config, tokenizer, rows):
    """Publish request bounds and diagnose tokenizer/config disagreements.

    Tokenizers may expose IDs absent from a legacy checkpoint. Valid requests remain
    valid; the host embedding request boundary rejects actually used invalid IDs.
    Never drop added/special token IDs when computing the diagnostic range.
    """
    real = tokenizer.get_real_tokenizer() if tokenizer is not None else None
    ids = []
    if real is not None:
        get_vocab = getattr(real, "get_vocab", None)
        if callable(get_vocab):
            ids.extend(get_vocab().values())
        ids.extend(getattr(real, "all_special_ids", ()) or ())
    if ids and (min(ids) < 0 or max(ids) >= rows):
        logging.warning(
            "BERT tokenizer reachable IDs [%d, %d] exceed checkpoint input range "
            "[0, %d); out-of-range IDs (including special tokens) are rejected "
            "at the host request boundary, not remapped or padded.",
            min(ids),
            max(ids),
            rows,
        )
    config.input_vocab_size = rows
