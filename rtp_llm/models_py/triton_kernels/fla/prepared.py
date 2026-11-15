"""Explicit fixed-capacity GDN metadata; refresh only outside graph capture."""

import operator

import torch

from rtp_llm.models_py.triton_kernels.causal_conv1d.causal_conv1d import (
    BLOCK_M,
    CausalConv1dMetadata,
)


class GdnPrefillPrepared:
    """One owner per captured token/request capacity and GDN geometry.

    All device addresses remain stable across refreshes. Copies are synchronous
    until the graph runner supplies an explicit staging-buffer lease. Keep this
    owner alive with its graph; release it only after outstanding replays finish.
    """

    def __init__(
        self, token_capacity, batch_capacity, device, *, output_live_only=False
    ):
        if not 64 <= token_capacity < 2**31 or not 0 < batch_capacity < 2**31:
            raise ValueError(
                "prepared GDN requires token capacity >= 64 and positive batch capacity"
            )
        self.token_capacity = operator.index(token_capacity)
        self.batch_capacity = operator.index(batch_capacity)
        self.device = torch.device(device)
        self.cu_seqlens = torch.zeros(
            batch_capacity + 1, device=device, dtype=torch.int32
        )
        self.device = self.cu_seqlens.device
        self.prefix_lengths = torch.zeros(
            batch_capacity, device=device, dtype=torch.int32
        )
        self.request_live = torch.zeros(batch_capacity, device=device, dtype=torch.bool)
        self.num_tokens = torch.zeros((), device=device, dtype=torch.int32)
        self.chunk_indices = {}
        self.live_counts = {}
        self.chunk_offsets = torch.zeros(
            batch_capacity + 1, device=device, dtype=torch.int64
        )
        for size in (16, 64):
            capacity = (token_capacity + size - 1) // size + batch_capacity
            self.chunk_indices[size] = torch.full(
                (capacity, 2), -1, device=device, dtype=torch.int32
            )
            self.live_counts[size] = torch.zeros((), device=device, dtype=torch.int32)
        conv_capacity = (token_capacity + BLOCK_M - 1) // BLOCK_M + batch_capacity
        self.conv_metadata = CausalConv1dMetadata(
            torch.full((conv_capacity,), -1, device=device, dtype=torch.int32),
            torch.full((conv_capacity,), -1, device=device, dtype=torch.int32),
            conv_capacity,
            torch.zeros((), device=device, dtype=torch.int32),
        )
        self.checkpoint_starts = torch.zeros(
            batch_capacity + 1, device=device, dtype=torch.int64
        )
        self.workspace = {}
        # Opt in only when the caller slices live rows before any consumer.
        self.output_live_only = bool(output_live_only)
        self._prefix_alignment = None
        self._prefixes = ()
        self._host_lengths = ()
        self._sm120_cp_config = None
        self.sm120_cp_chunk_len = None
        self.sm120_min_cp_chunk_len = 64

    @staticmethod
    def _host_values(values):
        if isinstance(values, torch.Tensor):
            if values.device.type != "cpu" or values.ndim != 1:
                raise ValueError(
                    "prepared GDN refresh requires one-dimensional host metadata"
                )
            values = values.tolist()
        return tuple(operator.index(value) for value in values)

    @staticmethod
    def _copy(destination, values):
        destination.copy_(
            torch.as_tensor(values, dtype=destination.dtype), non_blocking=False
        )

    def refresh(self, host_lengths, host_prefixes):
        """Validate a live partition, then refresh existing buffers in place."""
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "prepared GDN metadata must be refreshed outside capture"
            )
        lengths = self._host_values(host_lengths)
        prefixes = self._host_values(host_prefixes)
        if len(lengths) != len(prefixes) or len(lengths) > self.batch_capacity:
            raise ValueError("prepared GDN request capacity or prefix count exceeded")
        if (
            any(value < 0 for value in lengths + prefixes)
            or sum(lengths) > self.token_capacity
        ):
            raise ValueError(
                "prepared GDN lengths must be non-negative and fit token capacity"
            )
        if any(prefix + length >= 2**31 for prefix, length in zip(prefixes, lengths)):
            raise ValueError("prepared GDN metadata must fit int32")
        prefixes = tuple(
            prefix if length else 0 for prefix, length in zip(prefixes, lengths)
        )
        if self._prefix_alignment and any(
            value % self._prefix_alignment for value in prefixes
        ):
            raise ValueError("prepared SM120 GDN requires page-aligned prefixes")
        # Retain the eager dispatch inputs before appending inactive requests.
        self._host_lengths = lengths
        self._refresh_sm120_cp()
        lengths += (0,) * (self.batch_capacity - len(lengths))
        prefixes += (0,) * (self.batch_capacity - len(prefixes))
        cu = [0]
        checkpoints = [0]
        for length in lengths:
            cu.append(cu[-1] + length)
            checkpoints.append(checkpoints[-1] + length // 128)
        self._copy(self.cu_seqlens, cu)
        self._copy(self.prefix_lengths, prefixes)
        self._copy(self.request_live, [length > 0 for length in lengths])
        self._copy(self.num_tokens, cu[-1])
        self._copy(self.checkpoint_starts, checkpoints)
        for size, table in self.chunk_indices.items():
            indices = [
                (row, chunk)
                for row, length in enumerate(lengths)
                for chunk in range((length + size - 1) // size)
            ]
            host = torch.full(table.shape, -1, dtype=table.dtype)
            if indices:
                host[: len(indices)] = torch.tensor(indices, dtype=table.dtype)
            table.copy_(host, non_blocking=False)
            self._copy(self.live_counts[size], len(indices))
            if size == 64:
                offsets = [0]
                for length in lengths:
                    offsets.append(offsets[-1] + (length + size - 1) // size)
                self._copy(self.chunk_offsets, offsets)
        conv = [
            (row, chunk)
            for row, length in enumerate(lengths)
            for chunk in range((length + BLOCK_M - 1) // BLOCK_M)
        ]
        host = torch.full((2, self.conv_metadata.total), -1, dtype=torch.int32)
        if conv:
            host[:, : len(conv)] = torch.tensor(conv, dtype=torch.int32).T
        self.conv_metadata.batch_ptr.copy_(host[0], non_blocking=False)
        self.conv_metadata.token_chunk_offset_ptr.copy_(host[1], non_blocking=False)
        self._copy(self.conv_metadata.live_count, len(conv))
        self._prefixes = prefixes

    def bind_sm120_cp(self, heads):
        """Bind once per plan; refresh retains the original live CP policy."""
        if self._sm120_cp_config is not None:
            if self._sm120_cp_config[0] != heads:
                raise ValueError("prepared SM120 GDN head geometry changed")
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepared SM120 CP must be bound before capture")
        from flashinfer.utils import get_device_sm_count

        from rtp_llm.models_py.kernels.cuda.gdn_sm120.device_utils import (
            get_compute_capability,
            get_device_name,
        )

        capability = get_compute_capability(self.device)
        self._sm120_cp_config = (
            heads,
            get_device_sm_count(self.device),
            capability,
            get_device_name(self.device),
        )
        # On SM120 the short-sequence heuristic is disabled above 16 heads;
        # the remaining chooser always returns a multiple of 512. Other head
        # geometries may choose any multiple of the kernel's 64-token block.
        self.sm120_min_cp_chunk_len = 512 if capability[0] == 12 and heads > 16 else 64
        self.sm120_cp_chunk_len = torch.empty(
            (1,), device=self.device, dtype=torch.int32
        )
        self._refresh_sm120_cp()

    def _refresh_sm120_cp(self):
        if self._sm120_cp_config is None:
            return
        from rtp_llm.models_py.kernels.cuda.gdn_sm120.varlen_helper import (
            choose_cp_chunk_len_host,
        )

        heads, sms, capability, name = self._sm120_cp_config
        lengths = self._host_lengths
        chunk = self.sm120_min_cp_chunk_len
        if lengths and any(lengths):
            chunk = choose_cp_chunk_len_host(
                max(lengths),
                heads,
                sms,
                device_capability=capability,
                total_seqlen=sum(lengths),
                num_seqs=len(lengths),
                device_name=name,
            )
        self._copy(self.sm120_cp_chunk_len, [chunk])

    def require_prefix_alignment(self, page_size):
        if any(value % page_size for value in self._prefixes):
            raise ValueError("prepared SM120 GDN requires page-aligned prefixes")
        self._prefix_alignment = page_size

    def get_workspace(self, name, shape, dtype, device):
        """Allocate during warmup, then enforce one stable geometry per owner."""
        if torch.device(device) != self.device:
            raise ValueError("prepared GDN workspace must stay on its owner's device")
        tensor = self.workspace.get(name)
        if tensor is None:
            if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "prepared GDN workspace must be warmed before capture"
                )
            tensor = torch.empty(shape, device=device, dtype=dtype)
            self.workspace[name] = tensor
        elif (
            tensor.shape != tuple(shape)
            or tensor.dtype != dtype
            or tensor.device != torch.device(device)
        ):
            raise ValueError(
                "prepared GDN workspace geometry changed; create a separate plan"
            )
        return tensor
