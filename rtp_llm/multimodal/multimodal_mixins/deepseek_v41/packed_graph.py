"""One exact-shape packed encoder graph, with owned per-request outputs.

Only a full image/patch-budget-limited batch is captured. Tails and other geometries
remain eager; there is no padding, eviction, numerical change, or error retry.
The cache belongs to one frozen inference model, after its weights are loaded.

For homogeneous inputs the effective cap is min(image cap, patch cap / patches
per image). No padding is introduced. Inspect stats to confirm actual replay.
"""

import threading

import torch

from rtp_llm.utils.cuda_graph_gate import cuda_graph_gate


class PackedVisionGraph:
    def __init__(self, full_batch_images, max_patches=None):
        if full_batch_images < 1 or (max_patches is not None and max_patches < 1):
            raise ValueError("graph image and patch budgets must be positive")
        self.full_batch_images = full_batch_images
        self.max_patches = max_patches
        self._lock = threading.Lock()
        self._key = None
        self._graph = None
        self._inputs = None
        self._outputs = None
        self._done = None
        self.captures = 0
        self.replays = 0
        self.eager = 0

    def stats(self):
        with self._lock:
            return {
                "captures": self.captures,
                "replays": self.replays,
                "eager": self.eager,
                "cached_grids": self._key[0] if self._key else None,
            }

    def effective_batch_images(self, grid):
        patches_per_image = grid[0] * grid[1]
        if patches_per_image <= 0:
            return 0
        if self.max_patches is None:
            return self.full_batch_images
        return min(self.full_batch_images, self.max_patches // patches_per_image)

    def run(self, patches, grids, forward, consume):
        eligible = (
            bool(patches)
            and len(grids) == len(patches)
            and len(patches) == self.effective_batch_images(grids[0])
            and patches[0].is_cuda
            and not torch.is_grad_enabled()
            and all(grid == grids[0] for grid in grids)
        )
        key = (
            tuple(grids),
            tuple((tuple(p.shape), p.dtype, p.device) for p in patches),
        )
        # Gate before model mutex: never upgrade an operation to capture and
        # never wait on a graph mutex while blocking another capture owner.
        scope = (
            cuda_graph_gate.capture
            if eligible and self._key is None
            else cuda_graph_gate.operation
        )
        with scope(), self._lock:
            if not eligible or (self._key is not None and self._key != key):
                self.eager += 1
                return consume(forward(patches, grids))
            stream = torch.cuda.current_stream(patches[0].device)
            if self._graph is None:
                static = [p.clone() for p in patches]
                warmup = torch.cuda.Stream(device=patches[0].device)
                warmup.wait_stream(stream)
                with torch.cuda.stream(warmup):
                    for _ in range(2):
                        forward(static, grids)
                warmup.synchronize()
                # No live operation may allocate while global capture runs.
                torch.cuda.synchronize(patches[0].device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=warmup):
                    aligned = forward(static, grids)
                stream.wait_stream(warmup)
                self._inputs, self._outputs, self._graph = static, aligned, graph
                self._done = torch.cuda.Event()
                self._key = key
                self.captures += 1
            else:
                # Also protect direct embedding callers on different streams.
                stream.wait_event(self._done)
                for destination, source in zip(self._inputs, patches):
                    destination.copy_(source)
            self._graph.replay()
            self.replays += 1
            # Span assembly must finish reading the private pool before the
            # next replay; callers only receive independently owned tensors.
            result = consume(self._outputs)
            self._done.record(stream)
            return result
