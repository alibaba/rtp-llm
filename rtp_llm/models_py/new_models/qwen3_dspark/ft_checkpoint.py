"""Self-contained Qwen3 DSpARK FT drafts; the dense decoder is shared."""

import torch

from rtp_llm.models_py.ft_checkpoint import RuntimeTensorBinding
from rtp_llm.models_py.new_models.qwen2.ft_checkpoint import load_gqa_ffn_ft_checkpoint
from rtp_llm.models_py.speculative.qwen3_dspark_checkpoint import absolute_d2t


def load_dspark_ft_checkpoint(model, checkpoint):
    def markov(owner):
        # The sampler uses bf16 independently of decoder compute dtype. Apply
        # exactly its HF row/padding contract to these replicated FT tables.
        return lambda values: owner.logical_source(values[0]).to(torch.bfloat16)

    extras = []
    for source, target, convert in (
        ("dspark_fc.weight", "fc.weight", lambda v: v[0].T),
        ("dspark_hidden_norm.gamma", "hidden_norm.weight", lambda v: v[0]),
        (
            "dspark_markov_w1.weight",
            "markov_head.markov_w1.weight",
            markov(model.markov_head.markov_w1),
        ),
        (
            "dspark_markov_w2.weight",
            "markov_head.markov_w2.weight",
            markov(model.markov_head.markov_w2),
        ),
    ):
        key = "global." + source
        if key not in checkpoint.entries:
            raise ValueError(
                f"DSpARK FT archive omitted {key}; re-export from the original HF draft"
            )
        extras.append(RuntimeTensorBinding((key,), target, convert))
    key = "global.multi_tokens_predict_d2t_map"
    if key in checkpoint.entries:
        model.d2t = torch.empty(model.vocab_size, dtype=torch.int64)
        # FT stores absolute IDs for both source conventions.
        extras.append(
            RuntimeTensorBinding(
                (key,),
                "d2t",
                lambda v: absolute_d2t(
                    "model.d2t", v[0], model.vocab_size, model._input_vocab_size
                ),
            )
        )
    elif model.vocab_size != model._input_vocab_size:
        raise ValueError(
            "DSpARK FT archive omitted required d2t; re-export from the original HF draft"
        )
    load_gqa_ffn_ft_checkpoint(model, checkpoint, extra_bindings=extras)
