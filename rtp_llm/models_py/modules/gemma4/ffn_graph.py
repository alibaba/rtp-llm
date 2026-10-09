"""Experimental FFN subgraph replay; native GPU operations remain visible.

Each module/device/stream/shape owns fixed input/output buffers. The returned
tensor owns a separate buffer, so later replay cannot overwrite an earlier
result. This is not complete model Graph/async/MTP or native-zero acceptance.
"""

import torch
import threading
import os
import triton
import triton.language as tl


@triton.jit
def _copy_bf16_kernel(X, Y, COUNT, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0,BLOCK)
    value = tl.load(X+index,mask=index<COUNT,other=0)
    tl.store(Y+index,value,mask=index<COUNT)


def copy_into(source,destination):
    if (
        source.dtype!=torch.bfloat16 or destination.dtype!=source.dtype
        or not source.is_cuda or source.device!=destination.device
        or source.shape!=destination.shape or not source.is_contiguous()
        or not destination.is_contiguous() or source.numel()==0
    ):
        raise ValueError("Gemma4 FFN graph copy requires matching contiguous CUDA BF16 buffers")
    _copy_bf16_kernel[(triton.cdiv(source.numel(),1024),)](
        source,destination,source.numel(),1024,num_warps=4,
    )


class FFNGraph:
    def __init__(self):
        self.states={}
        self.replays=0
        self._lock=threading.RLock()

    def forward(self,value,eager):
        if (
            not value.is_cuda or value.dtype!=torch.bfloat16 or value.dim()!=2
            or not value.is_contiguous() or not 1<=value.shape[0]<=64
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        # Serialize host enqueues through shared state. Same-stream callers
        # finish scheduling the output copy before the next input is written.
        with self._lock,torch.cuda.device(value.device):
            return self._forward(value,eager)

    def _forward(self,value,eager):
        stream=torch.cuda.current_stream(value.device)
        capture_policy=tuple(os.environ.get(name,"0") for name in (
            "GEMMA4_FUSED_NORM","GEMMA4_FUSED_GEGLU","GEMMA4_FUSED_RESIDUAL",
            "GEMMA4_FUSED_MOE_DISPATCH","GEMMA4_FUSED_MOE_COMBINE",
            "GEMMA4_FUSED_ROUTER_SCALES",
            "GEMMA4_FUSED_MOE_SORT",
            "GEMMA4_FUSED_ROUTER_SOFTMAX",
            "GEMMA4_FUSED_ROUTER_TOPK",
            "GEMMA4_FUSED_GROUPED_METADATA",
        ))
        key=(value.device.index,stream.cuda_stream,tuple(value.shape),capture_policy)
        state=self.states.get(key)
        if state is None:
            fixed=torch.empty_like(value)
            copy_into(value,fixed)
            eager(fixed)
            # Materialize the copy specialization before capture. Preparation
            # is paid by the first real caller and excluded only by warmup.
            warm_output=eager(fixed)
            warm_copy=torch.empty_like(warm_output)
            copy_into(warm_output,warm_copy)
            torch.cuda.synchronize(value.device)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured=eager(fixed)
            state=(fixed,captured,graph)
            self.states[key]=state
        fixed,captured,graph=state
        output=torch.empty_like(captured)
        copy_into(value,fixed)
        graph.replay()
        copy_into(captured,output)
        self.replays+=1
        return output
