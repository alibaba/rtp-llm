"""Views of the shared ZigzagProcessor permutation, prepared once per batch."""

import torch


class AttentionInputView:
    def __init__(self, original, **values):
        self.original = original
        self.__dict__.update(values)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __copy__(self):
        return type(self)(
            self.original,
            **{key: value for key, value in self.__dict__.items() if key != "original"}
        )


class QueryLayout:
    """Keep outer padded rank-major rows and true natural temporal rows."""

    def __init__(self, info, size, rank, device, valid_mask=None, buffers=None):
        self.size, self.rank, self.device = size, rank, torch.device(device)
        lengths = info.prefill_actual_input_lengths_cpu
        if lengths.device.type != "cpu" or lengths.ndim != 1:
            raise ValueError("zigzag requires original host request lengths")
        self.lengths = tuple(int(x) for x in lengths.tolist())
        self.prefixes = tuple(int(x) for x in info.prefill_prefix_lengths_cpu.tolist())
        if len(self.lengths) != len(self.prefixes):
            raise ValueError("zigzag request/prefix metadata disagree")
        self.restore = info.prefill_qkv_restore_indice.long()
        self.padding_mask = info.prefill_qkv_padding_mask.bool()
        self.physical_rows = self.restore.numel()
        if self.physical_rows % size:
            raise ValueError("zigzag rank-major rows must divide the TP group")
        self.local_rows = self.physical_rows // size
        self.true_rows = sum(self.lengths)
        # An explicit mask may contain holes even after removing layout padding.
        self.natural_all_valid = valid_mask is None
        self.all_valid = self.natural_all_valid and self.true_rows == self.physical_rows
        self.natural_to_rankmajor = self.restore[self.padding_mask]
        if self.natural_to_rankmajor.numel() != self.true_rows:
            raise ValueError("zigzag padding mask does not match real lengths")
        self.rankmajor_to_natural = torch.full(
            (self.physical_rows,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self.rankmajor_to_natural.index_copy_(
            0,
            self.natural_to_rankmajor,
            torch.arange(self.true_rows, dtype=torch.long, device=self.device),
        )
        self.natural_valid = (
            torch.ones(self.true_rows, dtype=torch.bool, device=self.device)
            if valid_mask is None
            else valid_mask.bool()
        )
        if self.natural_valid.numel() != self.true_rows:
            raise ValueError("original valid mask must describe real natural rows")
        rankmajor_valid = torch.zeros_like(self.padding_mask)
        rankmajor_valid.index_copy_(0, self.natural_to_rankmajor, self.natural_valid)
        start = rank * self.local_rows
        self.local_valid = rankmajor_valid[start : start + self.local_rows]
        offsets = [0]
        for length in self.lengths:
            offsets.append(offsets[-1] + length)
        self.cu_host = torch.tensor(offsets, dtype=torch.int32)
        self.cu_device = self.cu_host.to(device=self.device, non_blocking=True)
        self.lengths_host = lengths
        self.lengths_device = lengths.to(device=self.device, non_blocking=True)
        requests, positions = [], []
        for request, (length, prefix) in enumerate(zip(self.lengths, self.prefixes)):
            requests.append(
                torch.full((length,), request, dtype=torch.int32, device=self.device)
            )
            positions.append(
                torch.arange(
                    prefix, prefix + length, dtype=torch.int32, device=self.device
                )
            )
        req = torch.cat(requests)
        pos = torch.cat(positions)
        self.rankmajor_requests = torch.zeros(
            self.physical_rows, dtype=torch.int32, device=self.device
        )
        self.rankmajor_positions = torch.full_like(self.rankmajor_requests, -1)
        self.rankmajor_requests.index_copy_(0, self.natural_to_rankmajor, req)
        self.rankmajor_positions.index_copy_(
            0, self.natural_to_rankmajor, torch.where(self.natural_valid, pos, -1)
        )
        self.local_requests = self.rankmajor_requests[start : start + self.local_rows]
        self.local_positions = self.rankmajor_positions[start : start + self.local_rows]
        self._buffers = {} if buffers is None else buffers
        # cuLA's round-local metadata cache uses attention-input identity.
        # Recreating this view at every layer repeats planning and blocking
        # metadata uploads. Views live only for this layout/forward round.
        self._natural_attention_views = {}

    def prepare_last_hidden_rows(self):
        # cu_host describes true per-request rows; map those final rows through
        # the shared padded zigzag permutation rather than using local row IDs.
        last_natural = (self.cu_host[1:] - 1).to(device=self.device, dtype=torch.long)
        last_rankmajor = self.natural_to_rankmajor.index_select(0, last_natural)
        self.last_local_offsets = last_rankmajor.remainder(self.local_rows)
        self.last_is_mine = (
            last_rankmajor.div(self.local_rows, rounding_mode="floor") == self.rank
        ) & self.natural_valid.index_select(0, last_natural)

    def local_last_hidden(self, hidden):
        if hidden.ndim != 2 or hidden.shape[0] != self.local_rows:
            raise ValueError("CP last-hidden selection requires complete local rows")
        selected = hidden.index_select(0, self.last_local_offsets)
        return torch.where(self.last_is_mine[:, None], selected, 0)

    @property
    def resident_bytes(self):
        storages = {}
        for x in (*self._buffers.values(), *self.__dict__.values()):
            if isinstance(x, torch.Tensor) and x.device == self.device:
                storage = x.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
        return sum(storages.values())

    def reserve(self, name, rows, columns, dtype, *, column_major=False):
        key = (name, dtype, column_major)
        old = self._buffers.get(key)
        if old is None or old.shape[0] < rows or old.shape[1] < columns:
            self._buffers[key] = (
                torch.empty_strided(
                    (rows, columns),
                    (1, (rows + 31) // 32 * 32),
                    dtype=dtype,
                    device=self.device,
                )
                if column_major
                else torch.empty((rows, columns), dtype=dtype, device=self.device)
            )
        return self._buffers[key][:rows, :columns]

    def prepare_kda_buffers(self, projection_width, output_width, *, quantized=False):
        self.reserve("projection", self.true_rows, projection_width, torch.bfloat16)
        self.reserve(
            "output",
            self.physical_rows,
            output_width,
            torch.float8_e4m3fn if quantized else torch.bfloat16,
        )
        if quantized:
            self.reserve(
                "scales",
                self.physical_rows,
                (output_width + 511) // 512,
                torch.int32,
                column_major=True,
            )

    def gather_natural(self, projected):
        out = self.reserve(
            "projection", self.true_rows, projected.shape[1], projected.dtype
        )
        torch.index_select(projected, 0, self.natural_to_rankmajor, out=out)
        return out

    def scatter_rankmajor(self, values, scales=None):
        out = self.reserve("output", self.physical_rows, values.shape[1], values.dtype)
        out.zero_()
        # FP8 index_copy support varies with Torch; byte transport is exact.
        wire = torch.uint8 if values.dtype == torch.float8_e4m3fn else values.dtype
        out.view(wire).index_copy_(0, self.natural_to_rankmajor, values.view(wire))
        if scales is None:
            return out, None
        packed = scales.dtype == torch.int32
        restored = self.reserve(
            "scales",
            self.physical_rows,
            scales.shape[1],
            scales.dtype,
            column_major=scales.stride(0) == 1,
        )
        restored.fill_(0x7F7F7F7F if packed else 1.0)
        restored.index_copy_(0, self.natural_to_rankmajor, scales)
        return out, restored

    def natural_attention(self, original):
        view = self._natural_attention_views.get(original)
        if view is not None:
            return view
        view = AttentionInputView(
            original,
            input_lengths=self.lengths_host,
            input_lengths_device=self.lengths_device,
            cu_seqlens=self.cu_host,
            cu_seqlens_device=self.cu_device,
            valid_token_mask=self.natural_valid,
            logical_token_count=self.true_rows,
            physical_token_count=self.true_rows,
            logical_request_count=len(self.lengths),
        )
        self._natural_attention_views[original] = view
        return view
