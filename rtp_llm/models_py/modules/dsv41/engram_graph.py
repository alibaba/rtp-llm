"""CPU Engram prefetch outside a decode CUDA graph.

The graph owns only small, fixed-address selected-row buffers. The existing
host tables are neither duplicated nor CUDA-registered. This correctness-first
bridge intentionally retains CPU work and blocking H2D copies before replay;
it is not a zero-copy or fully asynchronous Engram implementation.
"""

import torch


class CpuEngramGraphState:
    """One graph bucket's selected rows, held alive by its attention impl.

    ``prepare`` must run outside capture on the same stream that will replay
    the graph. Blocking copies complete before temporary CPU tensors are
    released; writes to the fixed GPU buffers are stream-ordered after any
    earlier replay. Concurrent preparation/replay on another stream is not
    supported by this state.
    """

    @torch.inference_mode()
    def __init__(self, model, inputs):
        self.device = model.v4.embed.weight.device
        if self.device.type != "cuda":
            raise ValueError("CPU Engram graph state requires CUDA output buffers")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Create CPU Engram graph state outside capture")
        self.capacity = int(inputs.input_ids.numel())
        if self.capacity <= 0:
            raise ValueError("CPU Engram graph capacity must be positive")
        self._hash_state = model._engram_hash_state
        if self._hash_state is None or not model._engram_layers:
            raise ValueError("CPU Engram graph state requires initialized Engram")
        self.window_size = int(self._hash_state.layout.max_ngram_size)
        self._image_token = int(model._v4_args.v41_config.get("image_token_id", 129264))
        self._layers = tuple(
            (layer_id, hash_index, model.v4.layers[layer_id].engram.embed_tokens)
            for layer_id, hash_index in model._engram_layers
        )
        if len({layer_id for layer_id, _, _ in self._layers}) != len(self._layers):
            raise ValueError("Duplicate Engram layers in graph state")
        layout = self._hash_state.layout
        self.rows = {
            layer_id: torch.empty(
                (self.capacity, layout.n_hash_cols, layout.head_dim),
                dtype=torch.bfloat16,
                device=self.device,
            )
            for layer_id, _, _ in self._layers
        }
        self.token_mask = torch.empty(self.capacity, dtype=torch.bool, device=self.device)
        self.prepare(inputs.engram_token_windows)

    @torch.inference_mode()
    def prepare(self, windows):
        """Refresh all rows for current history, explicitly clearing bucket tail.

        Accepts CPU or CUDA integer windows. A CUDA input entails a synchronous
        D2H history transfer, followed by CPU hashing once for all Engram layers.
        Selected-row H2D is blocking, so no reused pinned staging lifetime is
        hidden behind an asynchronous copy.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CPU Engram prefetch must run outside CUDA Graph capture")
        if not isinstance(windows, torch.Tensor):
            raise TypeError("Engram history must be a tensor")
        if windows.dtype not in (torch.int32, torch.int64):
            raise TypeError("Engram history must use int32 or int64 token IDs")
        if windows.ndim != 2 or windows.shape[1] != self.window_size:
            raise ValueError(f"Expected Engram history [N, {self.window_size}]")
        count = int(windows.shape[0])
        if count > self.capacity:
            raise ValueError("Engram history exceeds selected graph capacity")
        history = windows.detach().to(device="cpu", dtype=torch.int64)
        if bool((history >= self._hash_state.token_map.numel()).any()):
            raise ValueError("Engram token ID exceeds checkpoint tokenizer vocabulary")
        dead = history == self._image_token
        hashes = self._hash_state.hash_token_windows(history, dead_mask=dead)
        mask = torch.zeros(self.capacity, dtype=torch.bool, device="cpu")
        mask[:count] = (history[:, 0] >= 0) & ~dead[:, 0]
        for layer_id, hash_index, embedding in self._layers:
            # CPU indices force the original CPU lookup even if the embedding
            # also happens to have a UVA view. No table ownership is changed.
            selected = embedding(hashes[:, hash_index], device="cpu")
            full = torch.zeros(
                self.rows[layer_id].shape, dtype=torch.bfloat16, device="cpu"
            )
            full[:count].copy_(selected)
            self.rows[layer_id].copy_(full, non_blocking=False)
        self.token_mask.copy_(mask, non_blocking=False)
