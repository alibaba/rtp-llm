"""Drive the actual draft loader, retaining the tensors read from safetensors."""

from unittest.mock import patch

import torch

from rtp_llm.models_py import weight_mapper
from rtp_llm.models_py.borrowed_parameters import BorrowedParameter
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.ops import ParallelismConfig, TaskType


def load_draft(config, tp, rank, roles=(), *, method="scratch"):
    # The service sets the task in build_model_config, after _create_config.
    config.task_type = TaskType.LANGUAGE_MODEL
    parallel = ParallelismConfig()
    parallel.tp_size = parallel.world_size = parallel.local_world_size = tp
    parallel.tp_rank = parallel.world_rank = parallel.local_rank = rank
    parallel.ffn_tp_size, parallel.ffn_tp_rank = tp, rank
    load = NewLoaderConfig.from_runtime(
        parallel, device="cpu", compute_dtype=torch.bfloat16, load_method=method
    )
    target, aliases = RtpModule(), {}
    for slot, role in enumerate(roles):
        leaf = torch.nn.Module()
        leaf.tp_size, leaf.tp_rank = tp, rank
        leaf.weight = torch.nn.Parameter(
            torch.full(
                (config.vocab_size // tp, config.hidden_size),
                2 * rank + slot + 1,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        target.add_module(role, leaf)
        path = role + ".weight"
        target.load_weights([(path, leaf.weight.detach())])
        aliases[path] = BorrowedParameter(target, path)

    actual_reader = weight_mapper._load_safetensors
    read = set()

    def record(*args, **kwargs):
        for key, value in actual_reader(*args, **kwargs):
            read.add(key)
            yield key, value

    with patch.object(weight_mapper, "_load_safetensors", record):
        loader = NewModelLoader(
            config, load, model_path=config.ckpt_path, parameter_aliases=aliases
        )
        model = loader.load()
        if loader.loaded_method is None or loader.loaded_method.value != method:
            raise AssertionError(f"requested {method}, got {loader.loaded_method}")
    return model, target, read
