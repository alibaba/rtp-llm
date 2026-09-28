# SPDX-License-Identifier: Apache-2.0

# Portions of the mathematical reference are adapted from z-lab/dflash.
# MIT License
#
# Copyright (c) 2026 Z Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""DFlash2 proposal distribution, independent of target sampling filters.

Mathematical reference: z-lab/dflash, commit
07ebd93db9f472af339b644bb70221ad8428328a, dflash/model.py:CandidateSelector.
The sparse q returned here is conditional on the *selected* previous token;
using the unary top-k softmax in rejection sampling would be incorrect.
"""

import logging
import threading
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

SelectorOutput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
logger = logging.getLogger(__name__)


def selector_reference(
    hidden_projection: torch.Tensor,
    predecessor_codebook: torch.Tensor,
    successor_codebook: torch.Tensor,
    hidden: torch.Tensor,
    logits: torch.Tensor,
    anchors: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    uniforms: torch.Tensor,
    top_k: int,
) -> SelectorOutput:
    """Independent PyTorch oracle using only the actual predecessor per step.

    Projection uses the checkpoint dtype; edge products, reduction and q use
    FP32. This reference deliberately does not construct the GPU K x K lattice.
    It is also the CPU implementation for small contract/probability tests.
    """
    vocab = predecessor_codebook.shape[0]
    unary, candidates = torch.topk(logits[..., :vocab], top_k, sorted=True)
    projected = F.linear(hidden, hidden_projection)
    previous = anchors.to(torch.int64)
    path, probabilities = [], []
    for slot in range(hidden.shape[1]):
        score = unary[:, slot].float() + (
            predecessor_codebook[previous].float()[:, None]
            * projected[:, slot].float()[:, None]
            * successor_codebook[candidates[:, slot]].float()
        ).sum(-1)
        best, argmax = score.max(-1, keepdim=True)
        shifted = torch.where(score == best, 0.0, score - best)
        weight = (
            shifted
            / temperatures.float().clamp_min(torch.finfo(torch.float32).tiny)[:, None]
        ).exp()
        sampled_q = weight / weight.sum(-1, keepdim=True)
        greedy_q = torch.zeros_like(sampled_q).scatter_(-1, argmax, 1.0)
        q = torch.where(greedy_mask.bool()[:, None], greedy_q, sampled_q)
        index = (
            (uniforms[:, slot, None].float() >= q.cumsum(-1))
            .sum(-1)
            .clamp_max(top_k - 1)
        )
        index = torch.where(greedy_mask.bool(), argmax.squeeze(-1), index)
        previous = candidates[:, slot].gather(-1, index[:, None]).squeeze(-1)
        path.append(previous.to(torch.int32))
        probabilities.append(q)
    if not path:
        return (
            torch.empty(hidden.shape[:2], dtype=torch.int32, device=hidden.device),
            candidates,
            torch.empty_like(unary, dtype=torch.float32),
        )
    return torch.stack(path, 1), candidates, torch.stack(probabilities, 1)


def dense_probabilities(
    candidate_ids: torch.Tensor,
    q: torch.Tensor,
    vocab_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scatter actual sparse q for the existing full-vocabulary rejection API.

    A reused output is cleared in full, including any padded vocabulary tail.
    candidate_ids must come from this selector (unique, valid top-k token IDs).
    """
    if candidate_ids.shape != q.shape or q.ndim != 3:
        raise ValueError(
            "DFlash2 candidates and q must have equal [B, gamma, K] shapes"
        )
    if candidate_ids.dtype != torch.int64 or q.dtype != torch.float32:
        raise ValueError("DFlash2 candidates must be int64 and q must be float32")
    if candidate_ids.device != q.device:
        raise ValueError("DFlash2 candidates and q must be on the same device")
    if vocab_size < q.shape[-1]:
        raise ValueError("DFlash2 vocabulary must contain at least K candidates")
    shape = (*q.shape[:2], vocab_size)
    if out is None:
        out = torch.empty(shape, device=q.device, dtype=torch.float32)
    if out.shape != shape or out.dtype != torch.float32 or out.device != q.device:
        raise ValueError("DFlash2 dense-q scratch must be FP32 [B, gamma, vocab_size]")
    out.zero_()
    out.scatter_(-1, candidate_ids, q)
    return out


class DFlash2CandidateSelector(torch.nn.Module):
    """Replicated, rank-local selector for already gathered target-head logits.

    Inputs are mask rows only: hidden [B,gamma,H], logits [B,gamma,V or padded V],
    anchors [B], temperatures [B], greedy_mask [B], uniforms [B,gamma]. Active
    anchors must be valid vocabulary IDs, stochastic temperatures finite > 0,
    and uniforms in [0,1). The engine validates these request contracts; the GPU
    hot path does not read tensor values back to the host. Greedy rows ignore
    their temperature/uniform values. Weights and hidden share FP32/FP16/BF16.

    Outputs: tokens int32 [B,gamma], candidate IDs int64 [B,gamma,K], conditional
    q FP32 [B,gamma,K]. Greedy rows produce one-hot q. No request top-k/top-p is
    applied: this checkpoint's selector_top_k defines the proposal support.
    """

    def __init__(
        self,
        hidden_projection: torch.Tensor,
        predecessor_codebook: torch.Tensor,
        successor_codebook: torch.Tensor,
        top_k: int,
    ):
        super().__init__()
        if hidden_projection.ndim != 2 or predecessor_codebook.ndim != 2:
            raise ValueError("DFlash2 projection/codebooks must be matrices")
        if predecessor_codebook.shape != successor_codebook.shape:
            raise ValueError("DFlash2 predecessor/successor codebooks must match")
        vocab, rank = predecessor_codebook.shape
        if rank == 0 or hidden_projection.shape[0] != rank:
            raise ValueError("DFlash2 projection and codebook rank must match")
        if top_k < 1 or top_k > vocab:
            raise ValueError("DFlash2 selector_top_k must be within [1, vocab_size]")
        weights = (hidden_projection, predecessor_codebook, successor_codebook)
        if any(w.device != hidden_projection.device for w in weights):
            raise ValueError("DFlash2 selector weights must share a device")
        if hidden_projection.dtype not in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ):
            raise ValueError("DFlash2 selector supports FP32, FP16 or BF16 weights")
        if any(
            w.dtype != hidden_projection.dtype or not w.is_contiguous() for w in weights
        ):
            raise ValueError(
                "DFlash2 selector weights must be contiguous and share a dtype"
            )
        self.register_buffer("hidden_projection", hidden_projection, persistent=False)
        self.register_buffer(
            "predecessor_codebook", predecessor_codebook, persistent=False
        )
        self.register_buffer("successor_codebook", successor_codebook, persistent=False)
        self.top_k = int(top_k)
        self.vocab_size = vocab
        self._graph_enabled = False
        self._graph_max_cached_shapes = 8
        self._graph_cache = {}
        self._graph_disabled_shapes = {}
        self._graph_lock = threading.Lock()
        self._graph_captures = 0
        self._graph_replays = 0
        self._graph_eager_cache_limit = 0
        self._graph_capture_failures = 0
        self._graph_eager_capture_failure = 0

    def configure_graph(self, enabled: bool, max_cached_shapes: int = 8) -> None:
        """Enable lazy rank-local CUDA/HIP capture after weights are loaded.

        This method only records policy; it does not allocate device memory or
        capture a graph. Existing entries are retained when disabling graph so
        in-flight replays keep their graph-owned storage alive. An entry is
        never evicted while serving; new shapes beyond the limit use eager and
        are explicitly counted/logged. Capture failures disable only that
        signature and use eager thereafter; failed signatures also count toward
        the cap so an unbounded sequence of shapes cannot grow host metadata.
        The cap includes input dtype variants.
        """
        if max_cached_shapes < 1:
            raise ValueError("DFlash2 graph cache must allow at least one shape")
        with self._graph_lock:
            self._graph_enabled = bool(enabled)
            self._graph_max_cached_shapes = int(max_cached_shapes)

    @property
    def graph_stats(self):
        with self._graph_lock:
            return {
                "enabled": self._graph_enabled,
                "cached_shapes": len(self._graph_cache),
                "max_cached_shapes": self._graph_max_cached_shapes,
                "captures": self._graph_captures,
                "replays": self._graph_replays,
                "eager_cache_limit": self._graph_eager_cache_limit,
                "capture_failures": self._graph_capture_failures,
                "disabled_shapes": len(self._graph_disabled_shapes),
                "eager_capture_failure": self._graph_eager_capture_failure,
            }

    def _forward_graph(self, *inputs: torch.Tensor) -> SelectorOutput:
        # Each graph owns contiguous static inputs. Original input strides may
        # vary between rounds because copy_ handles them outside capture.
        key = tuple((str(t.device), t.dtype, tuple(t.shape)) for t in inputs)
        weights = (
            self.hidden_projection,
            self.predecessor_codebook,
            self.successor_codebook,
        )
        key += tuple(t.data_ptr() for t in weights)
        # Host serialization protects the graph's shared inputs even when two
        # callers select this same shape on different CUDA streams/threads.
        with self._graph_lock, torch.cuda.device(inputs[0].device):
            current = torch.cuda.current_stream(inputs[0].device)
            # Caller-owned inputs may have been allocated on another stream
            # and can be released as soon as this method returns. Record our
            # asynchronous clone/copy reads with the caching allocator.
            for source in inputs:
                source.record_stream(current)
            if key in self._graph_disabled_shapes:
                self._graph_eager_capture_failure += 1
                return self._forward_gpu(*inputs)
            entry = self._graph_cache.get(key)
            if (
                entry is None
                and len(self._graph_cache) + len(self._graph_disabled_shapes)
                >= self._graph_max_cached_shapes
            ):
                self._graph_eager_cache_limit += 1
                if self._graph_eager_cache_limit == 1:
                    logger.warning(
                        "DFlash2 selector graph cache limit=%s reached; uncached shapes "
                        "execute eager (first hidden=%s logits=%s device=%s). "
                        "See graph_stats.eager_cache_limit for the fallback count.",
                        self._graph_max_cached_shapes,
                        tuple(inputs[0].shape),
                        tuple(inputs[1].shape),
                        inputs[0].device,
                    )
                return self._forward_gpu(*inputs)
            if entry is None:
                capture_stream = None
                static_inputs = None
                static_outputs = None
                graph = None
                try:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "DFlash2 selector lazy graph capture must run outside "
                            "the draft-forward graph"
                        )
                    # Clone/copy on the caller's stream first, then make the
                    # warmup stream wait for all input-producing work. Do not
                    # assume producer data is ready merely because tensors exist.
                    static_inputs = tuple(t.contiguous().clone() for t in inputs)
                    capture_stream = torch.cuda.Stream(device=inputs[0].device)
                    capture_stream.wait_stream(current)
                    for static in static_inputs:
                        static.record_stream(capture_stream)
                    with torch.cuda.stream(capture_stream):
                        for _ in range(3):
                            self._forward_gpu(*static_inputs)
                    graph = torch.cuda.CUDAGraph()
                    # The outer stream context restores the caller's stream
                    # even if graph.__exit__/capture_end itself raises after an
                    # unsupported CUDA/HIP operation invalidates capture.
                    with torch.cuda.stream(capture_stream):
                        with torch.cuda.graph(
                            graph,
                            stream=capture_stream,
                            capture_error_mode="thread_local",
                        ):
                            static_outputs = self._forward_gpu(*static_inputs)
                    current.wait_stream(capture_stream)
                    entry = {
                        "inputs": static_inputs,
                        "outputs": static_outputs,
                        "graph": graph,
                        "done": torch.cuda.Event(),
                        # Keep externally referenced weights alive if the
                        # module is later moved/reloaded with new tensor storage.
                        "weights": weights,
                        "has_replayed": False,
                    }
                    self._graph_cache[key] = entry
                    self._graph_captures += 1
                    logger.info(
                        "DFlash2 selector graph captured: hidden=%s logits=%s "
                        "device=%s dtype=%s static_input_bytes=%s cached_shapes=%s",
                        tuple(inputs[0].shape),
                        tuple(inputs[1].shape),
                        inputs[0].device,
                        inputs[0].dtype,
                        sum(t.numel() * t.element_size() for t in static_inputs),
                        len(self._graph_cache),
                    )
                except Exception as error:
                    # Capture is rank-local: raising only on TP0 could leave
                    # the other ranks waiting in the next broadcast. A graph
                    # runtime limitation must preserve the eager model path.
                    if capture_stream is not None:
                        current.wait_stream(capture_stream)
                    self._graph_capture_failures += 1
                    self._graph_eager_capture_failure += 1
                    self._graph_disabled_shapes[key] = str(error)
                    logger.warning(
                        "DFlash2 selector graph capture failed; disabling this shape "
                        "and falling back to eager: hidden=%s logits=%s device=%s "
                        "reason=%s. See graph_stats.eager_capture_failure.",
                        tuple(inputs[0].shape),
                        tuple(inputs[1].shape),
                        inputs[0].device,
                        error,
                    )
                    # Release partial graph-pool/input allocations before eager
                    # retries, especially when capture failed for lack of memory.
                    # Leaving this except block also releases its traceback.
                    graph = static_inputs = static_outputs = None
                if entry is None:
                    return self._forward_gpu(*inputs)
            else:
                # This event is recorded after the preceding replay *and* its
                # output clones. It also orders calls arriving on another stream.
                current.wait_event(entry["done"])
                for static, source in zip(entry["inputs"], inputs):
                    static.copy_(source, non_blocking=True)
            # Cache ownership protects these allocations between rounds;
            # record replay/clone uses as well for teardown or future stream
            # changes, since the graph pool was allocated on the capture stream.
            for tensor in (*entry["inputs"], *entry["outputs"]):
                tensor.record_stream(current)
            entry["graph"].replay()
            # Both tokens and q can outlive this proposal round while target
            # verification runs asynchronously. Never expose graph-owned outputs.
            result = tuple(t.clone() for t in entry["outputs"])
            entry["done"].record(current)
            self._graph_replays += 1
            if not entry["has_replayed"]:
                logger.info(
                    "DFlash2 selector graph first replay: hidden=%s logits=%s device=%s",
                    tuple(inputs[0].shape),
                    tuple(inputs[1].shape),
                    inputs[0].device,
                )
                entry["has_replayed"] = True
            return result

    def forward(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        anchors: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        uniforms: torch.Tensor,
    ) -> SelectorOutput:
        if hidden.ndim != 3 or logits.ndim != 3:
            raise ValueError("DFlash2 hidden/logits must have [B, gamma, H/V] shapes")
        shape = hidden.shape[:2]
        if logits.shape[:2] != shape or logits.shape[-1] < self.vocab_size:
            raise ValueError(
                "DFlash2 logits must contain the complete valid vocabulary"
            )
        if hidden.shape[-1] != self.hidden_projection.shape[1]:
            raise ValueError("DFlash2 hidden size does not match selector projection")
        if (
            anchors.shape != shape[:1]
            or temperatures.shape != shape[:1]
            or greedy_mask.shape != shape[:1]
        ):
            raise ValueError(
                "DFlash2 anchors, temperatures and greedy_mask must have shape [B]"
            )
        if uniforms.shape != shape:
            raise ValueError("DFlash2 uniforms must have shape [B, gamma]")
        if anchors.dtype not in (torch.int32, torch.int64):
            raise ValueError("DFlash2 anchors must be integer token IDs")
        if greedy_mask.dtype not in (torch.bool, torch.int32, torch.int64):
            raise ValueError("DFlash2 greedy_mask must be boolean or integer")
        if hidden.dtype != self.hidden_projection.dtype:
            raise ValueError("DFlash2 hidden and projection must share a dtype")
        if any(not x.is_floating_point() for x in (logits, temperatures, uniforms)):
            raise ValueError(
                "DFlash2 logits, temperatures and uniforms must be floating point"
            )
        if any(
            x.device != hidden.device
            for x in (
                logits,
                anchors,
                temperatures,
                greedy_mask,
                uniforms,
                self.hidden_projection,
            )
        ):
            raise ValueError("DFlash2 selector inputs and weights must share a device")
        if not hidden.is_cuda:
            return selector_reference(
                self.hidden_projection,
                self.predecessor_codebook,
                self.successor_codebook,
                hidden,
                logits,
                anchors,
                temperatures,
                greedy_mask,
                uniforms,
                self.top_k,
            )
        inputs = (hidden, logits, anchors, temperatures, greedy_mask, uniforms)
        if self._graph_enabled and hidden.shape[0] > 0 and hidden.shape[1] > 0:
            return self._forward_graph(*inputs)
        return self._forward_gpu(*inputs)

    def _forward_gpu(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        anchors: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        uniforms: torch.Tensor,
    ) -> SelectorOutput:
        # Lazy import keeps CPU configuration/reference tests independent of a
        # Triton installation. HIP tensors also have is_cuda=True.
        from rtp_llm.models_py.triton_kernels.common.dflash2_selector import (
            select_candidates,
        )

        unary, candidate_ids = torch.topk(
            logits[..., : self.vocab_size], self.top_k, sorted=True
        )
        projected = F.linear(hidden, self.hidden_projection)
        tokens, q = select_candidates(
            projected.contiguous(),
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary,
            anchors.contiguous(),
            temperatures.contiguous(),
            greedy_mask.contiguous(),
            uniforms.contiguous(),
        )
        return tokens, candidate_ids, q
