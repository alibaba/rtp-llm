"""Direct owners for Qwen4's two MTP input projections, never the target globals.

Attach this subtree at `mtp` (or map the source prefix to it). Other draft
children such as layers/mixer are owned by their respective runtime modules.
"""

import torch
import torch.nn.functional as F

from rtp_llm.models_py.model_desc.qwen4_exp_runtime.mtp_input_projection import (
    Qwen4ExpMTPInputProjection,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen4_exp.hyper_connection import (
    Qwen4ProjectionWeight,
    Qwen4TensorWeight,
)


class _DraftFC(Qwen4ProjectionWeight):
    """HF FC is already [output, input], including in FP8 compute layout."""

    def __init__(self, hidden, *, dtype, fp8_storage, role):
        super().__init__((hidden, hidden), dtype=dtype, fp8_storage=fp8_storage)
        self.role = role
        object.__setattr__(self, "_fp8", None)

    def process_weights_after_loading(self):
        super().process_weights_after_loading()
        if self.storage_scale is not None and self._fp8 is None:
            from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear

            runtime = Fp8Linear(self.weight, self.storage_scale, role=self.role)
            object.__setattr__(self, "_fp8", runtime)

    def forward(self, value):
        if not self._storage_ready:
            raise RuntimeError("MTP projection weights have not been finalized")
        if self._fp8 is not None:
            return self._fp8(value)
        return F.linear(value, self.weight)


class Qwen4MTPInputOwner(RtpModule):
    """Own the four replicated tensors of the released draft's input mixer."""

    def __init__(
        self,
        hidden,
        hc_mult,
        *,
        norm_eps,
        dtype=torch.bfloat16,
        fp8_storage=False,
        role="PDFUSION",
    ):
        super().__init__()
        if any(type(n) is not int or n <= 0 for n in (hidden, hc_mult)):
            raise ValueError("MTP hidden size and HC multiplier must be positive")
        self.hidden = hidden
        self.norm_eps = norm_eps
        self.pre_fc_norm_embedding = Qwen4TensorWeight((hidden,), dtype=dtype)
        self.pre_fc_norm_hidden = Qwen4TensorWeight((hidden * hc_mult,), dtype=dtype)
        kwargs = dict(dtype=dtype, fp8_storage=fp8_storage, role=role)
        self.fc_embedding = _DraftFC(hidden, **kwargs)
        self.fc_hidden = _DraftFC(hidden, **kwargs)
        object.__setattr__(self, "_runtime", None)

    def process_weights_after_loading(self):
        if self._runtime is not None:
            return
        self.fc_embedding.process_weights_after_loading()
        self.fc_hidden.process_weights_after_loading()
        runtime = Qwen4ExpMTPInputProjection(
            self.pre_fc_norm_embedding.weight,
            self.pre_fc_norm_hidden.weight,
            self.fc_embedding,
            self.fc_hidden,
            self.hidden,
            self.norm_eps,
        )
        # Runtime borrows the active owner parameters; registering it here would
        # make the loader expect a duplicate checkpoint subtree.
        object.__setattr__(self, "_runtime", runtime)

    def forward(self, inputs_embeds, input_hiddens):
        if self._runtime is None:
            raise RuntimeError("MTP input owner is not finalized")
        return self._runtime(inputs_embeds, input_hiddens)
