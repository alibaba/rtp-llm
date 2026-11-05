# BSD 3-Clause License
#
# Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Project-scoped generic padding helpers; no FlashAttention kernel or package.

This module is safe in Ray's pre-device-assignment setup hook: all imports at
module scope are from the standard library. Tensor libraries load only when
the already-imported verl attention module actually asks for its helpers.

The index autograd and five-value unpad API follow flash-attention v2.8.3's
bert_padding.py. See ../references/flash-attention-v2.8.3/LICENSE and provenance.
No installed package is edited and Transformers' FlashAttention detection is
left unchanged.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import socket
import time


@lru_cache(maxsize=1)
def _index_functions():
    import torch

    class IndexFirstAxis(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, indices):
            assert input.ndim >= 2
            ctx.save_for_backward(indices)
            ctx.input_shape = input.shape
            return torch.index_select(input, 0, indices)

        @staticmethod
        def backward(ctx, grad_output):
            (indices,) = ctx.saved_tensors
            assert grad_output.ndim >= 2
            grad_input = grad_output.new_zeros(ctx.input_shape)
            # Match upstream's scatter (including duplicate-index semantics).
            expanded = indices.reshape(-1, *([1] * (grad_output.ndim - 1))).expand_as(grad_output)
            grad_input.scatter_(0, expanded, grad_output)
            return grad_input, None

    class IndexPutFirstAxis(torch.autograd.Function):
        @staticmethod
        def forward(ctx, values, indices, first_axis_dim):
            assert indices.ndim == 1
            assert values.ndim >= 2
            ctx.save_for_backward(indices)
            output = values.new_zeros((first_axis_dim, *values.shape[1:]))
            output[indices] = values
            return output

        @staticmethod
        def backward(ctx, grad_output):
            (indices,) = ctx.saved_tensors
            return torch.index_select(grad_output, 0, indices), None, None

    return IndexFirstAxis.apply, IndexPutFirstAxis.apply


def index_first_axis(input, indices):
    return _index_functions()[0](input, indices)


def pad_input(hidden_states, indices, batch, seqlen):
    output = _index_functions()[1](hidden_states, indices, batch * seqlen)
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """Return values, indices, int32 offsets, max length and used lengths.

    unused_mask selects allocated-but-unused entries exactly as the upstream
    API: selection/offsets include both masks; the last return uses only the
    original attention_mask. Masks are not silently modified or normalized.
    """
    import torch
    import torch.nn.functional as F

    all_masks = attention_mask + unused_mask if unused_mask is not None else attention_mask
    lengths = all_masks.sum(dim=-1, dtype=torch.int32)
    used_lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_length = lengths.max().item()
    offsets = F.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))
    flat = hidden_states.flatten(0, 1)
    return index_first_axis(flat, indices), indices, offsets, max_length, used_lengths


def _source(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _proof_filename(hostname, pid):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", hostname)
    if not safe or safe != hostname or len(safe) > 128:
        safe = safe[:96] + "-" + hashlib.sha256(hostname.encode()).hexdigest()[:12]
    return f"first-use-{safe}-{pid}.json"


def install(attention_module):
    """Fallback only for the exact missing optional FlashAttention package.

    Existing attention wrappers look up the getter at call time, so references
    captured by verl.workers.utils.padding before this install also work.
    Other import errors and the NPU/original available backend are preserved.
    """
    original = attention_module._get_attention_functions
    if getattr(original, "_q397_torch_padding", False):
        return
    emitted = False

    def getter():
        nonlocal emitted
        try:
            return original()
        except ModuleNotFoundError as exc:
            if exc.name not in {"flash_attn", "flash_attn.bert_padding"}:
                raise
            from einops import rearrange

            if not emitted:
                run_root = os.environ.get("Q397_RUN_ROOT")
                if run_root:
                    getter_text = inspect.getsource(original).encode()
                    proof = {
                        "schema": "q397-generic-torch-padding-first-use-v1",
                        "pid": os.getpid(), "hostname": socket.gethostname(), "unix": time.time(),
                        "implementation": "torch_generic_padding_only", "missing_module": exc.name,
                        "source": _source(__file__),
                        "original_attention_source": _source(attention_module.__file__),
                        "original_getter_sha256": hashlib.sha256(getter_text).hexdigest(),
                        "flash_attention_kernel_used": False,
                    }
                    directory = Path(run_root) / "validation" / "padding"
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / _proof_filename(proof["hostname"], proof["pid"])
                    # Once per process; never replace another process's proof.
                    with path.open("x") as out:
                        out.write(json.dumps(proof, indent=2) + "\n")
                emitted = True
            return index_first_axis, pad_input, rearrange, unpad_input

    getter._q397_torch_padding = True
    getter._q397_original_getter = original
    attention_module._get_attention_functions = getter
