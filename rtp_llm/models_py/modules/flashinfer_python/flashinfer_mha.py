from typing import Optional

import torch

from rtp_llm.models_py.modules.flashinfer_python import flashinfer_python
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters

# from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs

class FlashInferPythonParams(object):
    def __init__(
            self,
            batch_size: int,
            max_seq_len: int,
            seq_lens: Optional[torch.Tensor] = None,
            block_tables: Optional[torch.Tensor] = None,
            cu_seqlens: Optional[torch.Tensor] = None,
            cu_kv_seqlens: Optional[torch.Tensor] = None,
    ):

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.seq_lens = seq_lens
        self.block_tables = block_tables
        self.cu_seqlens = cu_seqlens
        self.cu_kv_seqlens = cu_kv_seqlens
    
# Constants
DEFAULT_WORKSPACE_SIZE_MB = (
    512  # Memory workspace size in MB, todo(Yingyi): read from config
)

# Reuse this workspace buffer across all TRTLLM MHA wrappers
g_zero_workspace_buffer = None

def create_g_workspace_buffer(device: str='cuda:0'):
    global g_zero_workspace_buffer, g_empty_workspace_buffer
    if g_zero_workspace_buffer is None:        
        g_zero_workspace_buffer = torch.zeros(
            DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
    # if g_empty_workspace_buffer is None:
    #     g_empty_workspace_buffer = torch.empty(
    #         DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024,
    #         dtype=torch.uint8,
    #         device=device,
    #     )
    return g_zero_workspace_buffer

class FlashInferPythonPrefillOp(object):
    def __init__(
        self,
        config: GptInitModelParameters,
    ):
        self.config = config
        self.head_dim = config.hidden_size // config.head_num
        self.head_num = config.head_num
        self.scaling = self.head_dim**-0.5        
        self.local_head_num = config.head_num // config.tp_size
        self.workspace_buffer = create_g_workspace_buffer()

    def support(self, attention_inputs: PyAttentionInputs):
        return attention_inputs.is_prefill

    def prepare(self, attention_inputs: PyAttentionInputs) -> FlashInferPythonParams:
        return FlashInferPythonParams(
            batch_size=attention_inputs.input_lengths.size(0),
            max_seq_len=attention_inputs.input_lengths.max().item(),
            seq_lens=attention_inputs.input_lengths,
            block_tables=attention_inputs.kv_cache_block_id_device,
            cu_seqlens=attention_inputs.cu_seqlens,
            cu_kv_seqlens=attention_inputs.cu_kv_seqlens
        )

    def forward(self, q: torch.Tensor, kv_cache: Optional[KVCache], fmha_params: FlashInferPythonParams) -> torch.Tensor:
        q_type = q.dtype        
        q = q.to(torch.float8_e4m3fn)
        o_type = torch.float8_e4m3fn
        q = q.contiguous().view(-1, self.local_head_num, self.head_dim)
        q_scale = 1.0
        k_scale = 1.0
        bmm1_scale = q_scale * k_scale * self.scaling
        bmm2_scale = 1.0

        o = flashinfer_python.prefill.trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=(kv_cache.k_cache_base, kv_cache.v_cache_base),
            workspace_buffer=self.workspace_buffer,
            block_tables=fmha_params.block_tables,
            seq_lens=fmha_params.seq_lens,
            max_q_len=fmha_params.max_seq_len,
            max_kv_len=fmha_params.max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            batch_size=fmha_params.batch_size,
            cum_seq_lens_q=fmha_params.cu_seqlens,
            cum_seq_lens_kv=fmha_params.cu_kv_seqlens,
            window_left=-1,
            # TODO: add attention_sink operation or nvfp4 scale factor if needed
            sinks=None,
            out_dtype=o_type,  # model_runner.dtype
        )

        return o.view(-1, self.local_head_num * self.head_dim).to(q_type)
        

class FlashInferPythonDecodeOp(object):
    def __init__(
        self,
        config: GptInitModelParameters,
    ):
        self.config = config
        self.head_dim = config.hidden_size // config.head_num
        self.head_num = config.head_num
        self.scaling = self.head_dim**-0.5        
        self.local_head_num = config.head_num // config.tp_size
        self.workspace_buffer = create_g_workspace_buffer()

    def support(self, attention_inputs: PyAttentionInputs):
        return not attention_inputs.is_prefill

    def prepare(self, attention_inputs: PyAttentionInputs) -> FlashInferPythonParams:
        return FlashInferPythonParams(
            batch_size=attention_inputs.sequence_lengths.size(0),
            max_seq_len=attention_inputs.sequence_lengths.max().item(),
            seq_lens=attention_inputs.sequence_lengths,
            block_tables=attention_inputs.kv_cache_block_id_device,
            cu_seqlens=attention_inputs.cu_seqlens,
            cu_kv_seqlens=attention_inputs.cu_kv_seqlens
        )

    def forward(self, q: torch.Tensor, kv_cache: Optional[KVCache], fmha_params: FlashInferPythonParams) -> torch.Tensor:
        q_type = q.dtype
        q = q.to(torch.float8_e4m3fn)
        o_type = torch.float8_e4m3fn
        
        q = q.contiguous().view(-1, self.local_head_num, self.head_dim)
        q_scale = 1.0
        k_scale = 1.0
        bmm1_scale = q_scale * k_scale * self.scaling
        bmm2_scale = 1.0
        # sink: additional value per head in the denominator of the softmax.

        # Call TRT-LLM kernel
        # raw_out: like q, [bs, acc_q_len, num_q_heads, head_dim] but with output dtype
        o = flashinfer_python.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            block_tables=fmha_params.block_tables,
            seq_lens=fmha_params.seq_lens,
            max_seq_len=fmha_params.max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=-1,
            # TODO: add attention_sink operation or nvfp4 scale factor if needed
            sinks=None,
            out_dtype=o_type,  # model_runner.dtype
        )

        return o.view(-1, self.local_head_num * self.head_dim).to(q_type)
