"""ROCm facade for BERT profile attention; metadata is request-local."""
import torch

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.modules.factory.attention.rocm_impl.bert_uqi_two_pass_core import (
    prepare_two_pass, run_two_pass,
)
from rtp_llm.ops.compute_ops import ParamsBase


class BertUqiParams(ParamsBase):
    def __init__(self, plan):
        super().__init__()
        self.plan = plan

    def check_recycle(self):
        return True


class BertUqiTwoPassAttnOp:
    def __init__(self, attn_configs):
        self.heads = attn_configs.head_num
        self.kv_heads = attn_configs.kv_head_num
        self.dim = attn_configs.size_per_head

    def forward(self, qkv, plan):
        qkv = qkv.reshape(qkv.shape[0], (self.heads + 2 * self.kv_heads) * self.dim)
        q, k, v = torch.split(
            qkv, [self.heads * self.dim, self.kv_heads * self.dim, self.kv_heads * self.dim],
            dim=-1,
        )
        n = qkv.shape[0]
        q = q.reshape(n, self.heads, self.dim)
        k = k.reshape(n, self.kv_heads, self.dim)
        v = v.reshape(n, self.kv_heads, self.dim)
        return run_two_pass(plan, q, k, v)


class BertUqiTwoPassImpl(FMHAImplBase):
    def __init__(self, op, attn_inputs, schedule):
        self.fmha_impl = op
        self.attn_inputs = attn_inputs
        # ROCm tensors use the cuda device namespace. The current device is
        # selected by the engine, including when host-only metadata is supplied.
        device = schedule.perm.device if schedule.perm is not None else torch.device("cuda", torch.cuda.current_device())
        self.fmha_params = BertUqiParams(prepare_two_pass(schedule, device))
        self.uqi_perm = schedule.perm
        self.uqi_inv_perm = schedule.inv_perm

    def forward(self, qkv, kv_cache, layer_idx=0):
        return self.fmha_impl.forward(qkv, self.fmha_params.plan)

    @staticmethod
    def support(attn_configs, attn_inputs):
        return attn_inputs.is_prefill

    def support_cuda_graph(self):
        return False
