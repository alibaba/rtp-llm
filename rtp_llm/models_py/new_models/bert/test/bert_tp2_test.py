"""Checkpoint-to-forward TP arithmetic with two real CPU/Gloo ranks.

This isolates sharding and reductions from NCCL/RCCL and GPU attention. It
does not replace the two-GPU HTTP embedding deployment smoke.
"""

import os
import tempfile
import time
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from safetensors.torch import save_file

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.bert.test.bert_newloader_test import _config, _weights


def _distinct_weights(model_type):
    generator = torch.Generator().manual_seed(5821)
    weights = {}
    for name, template in _weights(model_type).items():
        value = torch.randn(template.shape, generator=generator) * 0.3
        if "LayerNorm.weight" in name:
            value += 1.0
        weights[name] = value
    return weights


def _reference(weights, prefix, ids, positions, token_types):
    """Full, unsharded encoder using only original checkpoint tensors."""
    embeddings = prefix + ".embeddings."
    layer = prefix + ".encoder.layer.0."

    def linear(x, name):
        return F.linear(x, weights[name + ".weight"], weights[name + ".bias"])

    def norm(x, name):
        return F.layer_norm(
            x, (4,), weights[name + ".weight"], weights[name + ".bias"], 1e-5
        )

    hidden = norm(
        F.embedding(ids, weights[embeddings + "word_embeddings.weight"])
        + F.embedding(positions, weights[embeddings + "position_embeddings.weight"])
        + F.embedding(
            token_types, weights[embeddings + "token_type_embeddings.weight"]
        ),
        embeddings + "LayerNorm",
    )
    q, k, v = [
        linear(hidden, layer + "attention.self." + name)
        .reshape(-1, 2, 2)
        .transpose(0, 1)
        for name in ("query", "key", "value")
    ]
    scores = torch.softmax((q @ k.transpose(-2, -1)) / (2**0.5), dim=-1)
    attention = (scores @ v).transpose(0, 1).reshape(-1, 4)
    hidden = norm(
        hidden + linear(attention, layer + "attention.output.dense"),
        layer + "attention.output.LayerNorm",
    )
    output = linear(
        F.gelu(linear(hidden, layer + "intermediate.dense")), layer + "output.dense"
    )
    return norm(hidden + output, layer + "output.LayerNorm")


class _LocalHeadAttention:
    def forward(self, qkv, kv_cache, layer_idx):
        # One head per TP rank; all sequence tokens stay local in this test.
        q, k, v = qkv.split(2, dim=-1)
        result = F.scaled_dot_product_attention(
            q[None, None], k[None, None], v[None, None], is_causal=False
        )
        return result.reshape(-1, 2)


def _run_rank(rank, rendezvous, checkpoint_root):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        for model_type in ("bert", "roberta"):
            model = NewModelLoader(
                _config(model_type),
                NewLoaderConfig(
                    device="cpu",
                    compute_dtype=torch.float32,
                    tp_size=2,
                    tp_rank=rank,
                    ep_size=2,
                    ep_rank=rank,
                ),
                model_path=os.path.join(checkpoint_root, model_type),
            ).load()
            reductions = []

            def real_all_reduce(tensor, group, *, inplace=True):
                if group != Group.TP:
                    raise AssertionError(f"unexpected collective group: {group}")
                reductions.append(tuple(tensor.shape))
                result = tensor if inplace else tensor.clone()
                dist.all_reduce(result, op=dist.ReduceOp.SUM)
                return result

            for ids in ([1, 3, 6], [6, 0]):
                ids = torch.tensor(ids)
                positions = torch.arange(ids.numel())
                token_types = positions % 2
                inputs = SimpleNamespace(
                    input_ids=ids,
                    bert_embedding_inputs=SimpleNamespace(
                        combo_position_ids=positions,
                        combo_tokens_type_ids=token_types,
                        input_embedding_scalar=1.0,
                    ),
                )
                # Only transport and attention are adapted to CPU. The actual
                # loader, public forward, projections and row-bias logic run.
                with patch(
                    "rtp_llm.models_py.layers.linear.all_reduce", real_all_reduce
                ), torch.inference_mode():
                    output = model(inputs, _LocalHeadAttention()).hidden_states
                expected = _reference(
                    _distinct_weights(model_type),
                    model_type,
                    ids,
                    positions,
                    token_types,
                )
                torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
            if len(reductions) != 4:
                raise AssertionError(
                    f"expected attention + FFN reduction per call: {reductions}"
                )
    finally:
        dist.destroy_process_group()


class BertTp2Test(unittest.TestCase):
    def test_bert_and_roberta_tp2_forward_matches_unsharded_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            for model_type in ("bert", "roberta"):
                directory = os.path.join(root, model_type)
                os.mkdir(directory)
                save_file(
                    _distinct_weights(model_type),
                    os.path.join(directory, "model.safetensors"),
                )
            context = mp.spawn(
                _run_rank,
                args=(os.path.join(root, "rendezvous"), root),
                nprocs=2,
                join=False,
            )
            try:
                deadline = time.monotonic() + 120
                while not context.join(timeout=1):
                    if time.monotonic() >= deadline:
                        self.fail("TP2 Gloo regression exceeded 120 seconds")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in context.processes:
                    process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                        process.join()


if __name__ == "__main__":
    unittest.main()
