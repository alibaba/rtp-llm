"""Module-owned canonicalization of legacy per-tensor FP8 FT projections.

This restores checkpoint-local parameters, not a finalized accelerator state.
The existing FP8 method still selects its backend and performs device dtype
conversion once after migration. Historical tensor names are supplied by the
model's format adapter; they are not a runtime registry here.
"""

import torch


def _cat_fp8(values, dim):
    return torch.cat(
        [value.contiguous().view(torch.uint8) for value in values], dim=dim
    ).view(values[0].dtype)


class PerTensorFP8FTProjection:
    def __init__(
        self,
        layer,
        checkpoint,
        kernel_name,
        scale_name,
        projection,
        *,
        swizzled=False,
        owner_partition=None,
    ):
        self.layer, self.checkpoint = layer, checkpoint
        self.swizzled = swizzled
        self.kernel_name, self.scale_name, self.projection = (
            kernel_name,
            scale_name,
            projection,
        )
        self.tp, self.rank = checkpoint.topology.tp_size, checkpoint.topology.tp_rank
        # Hybrid attention can partition Q/K/V/Z in its parent before passing
        # the local matrix to a nominally TP=1 linear child.
        partition = (
            (layer.tp_size, layer.tp_rank)
            if owner_partition is None
            else owner_partition
        )
        if partition != (self.tp, self.rank):
            raise ValueError("FP8 FT projection topology does not match the layer")
        self.shape = tuple(layer.weight.shape)
        self.local_width = (
            self.shape[0] // 2 if projection == "gate_up" else self.shape[1]
        )
        self.hidden = self.shape[1] if projection == "gate_up" else self.shape[0]

    def _read(self, name, peer):
        shape = self.checkpoint.read_rank_slice(name, peer)
        return self.checkpoint.read_rank_slice(name, peer, (slice(None),) * len(shape))

    def _kernel_shape(self, peer):
        return self.checkpoint.read_rank_slice(self.kernel_name, peer)

    def _undo_rocm_dense_layout(self, value, raw_dtype):
        from rtp_llm.models_py.quant_methods.block_fp8_ft import (
            block_ft_source_is_rocm,
            block_ft_weight,
        )

        if not block_ft_source_is_rocm(
            self.checkpoint.index_metadata.get("ft_export_backend"), raw_dtype, False
        ):
            return value
        if self.swizzled:
            from rtp_llm.models_py.quant_methods.dense_layout import decode_dense_weight

            return decode_dense_weight(value)
        return block_ft_weight(
            value.unsqueeze(0), (1, *value.shape), False, rocm_shuffle=True
        )[0]

    def _read_kernel(self, peer):
        value = self._read(self.kernel_name, peer)
        return self._undo_rocm_dense_layout(value, value.dtype)

    def _scale(self, peer, part=0):
        value = self._read(self.scale_name, peer).float().reshape(-1)
        allowed = (1, 2) if self.projection == "gate_up" else (1,)
        if (
            value.numel() not in allowed
            or not bool(torch.isfinite(value).all())
            or bool((value <= 0).any())
        ):
            raise ValueError("FP8 FT requires finite positive per-projection scales")
        return value[part if value.numel() == 2 else 0].item()

    def _pieces(self):
        if self.projection not in ("gate_up", "down"):
            if self._kernel_shape(self.rank) != self.shape:
                raise ValueError("FP8 FT projection shape mismatch")
            return [(self.rank, 0, 0, self.shape[0], self.shape[0])]
        expected_widths = {self.local_width, ((self.local_width + 63) // 64) * 64}
        begin, end = self.rank * self.local_width, (self.rank + 1) * self.local_width
        pieces, source_width = [], None
        for peer in range(self.tp):
            shape = self._kernel_shape(peer)
            if len(shape) != 2:
                raise ValueError("FP8 FT FFN kernel must be a matrix")
            width = shape[0] // 2 if self.projection == "gate_up" else shape[1]
            expected = (
                (2 * width, self.hidden)
                if self.projection == "gate_up"
                else (self.hidden, width)
            )
            if (
                shape != expected
                or width not in expected_widths
                or source_width not in (None, width)
            ):
                raise ValueError("FP8 FT FFN padded source layout mismatch")
            source_width = width
            left, right = max(begin, peer * width), min(end, (peer + 1) * width)
            if left < right:
                for part in range(2 if self.projection == "gate_up" else 1):
                    pieces.append(
                        (peer, part, left - peer * width, right - peer * width, width)
                    )
        if not pieces:
            raise ValueError("FP8 FT source does not cover destination FFN shard")
        return pieces

    def scale(self):
        return torch.tensor(
            [max(self._scale(peer, part) for peer, part, *_ in self._pieces())],
            dtype=torch.float32,
        )

    def weight(self):
        pieces = self._pieces()
        maximum = self.scale().item()
        results = [[], []] if self.projection == "gate_up" else [[]]
        cache = {}
        for peer, part, begin, end, width in pieces:
            if peer not in cache:
                cache[peer] = self._read_kernel(peer)
            source = cache[peer]
            if source.dtype != torch.float8_e4m3fn:
                raise TypeError(
                    "Legacy per-tensor FP8 FT requires declared e4m3fn checkpoint storage"
                )
            if self.projection == "gate_up":
                value = source[part * width + begin : part * width + end]
            elif self.projection == "down":
                value = source[:, begin:end]
            else:
                value = source
            scale = self._scale(peer, part)
            if scale != maximum:
                value = (value.float() * (scale / maximum)).to(source.dtype)
            results[part].append(value)
        if self.projection in ("gate_up", "down"):
            # A different source rank may contain only padding. Validate those
            # rows too before discarding them, without retaining other layers.
            width = pieces[0][-1]
            total = self.local_width * self.tp
            axis = 0 if self.projection == "gate_up" else 1
            for peer in range(self.tp):
                padding = max(0, total - peer * width)
                if padding >= width:
                    continue
                source = cache.get(peer)
                if source is None:
                    source = self._read_kernel(peer)
                for part in range(len(results)):
                    offset = part * width + padding if axis == 0 else padding
                    if torch.count_nonzero(
                        source.narrow(axis, offset, width - padding).float()
                    ).item():
                        raise ValueError("FP8 FT FFN padding contains nonzero values")
        axis = 1 if self.projection == "down" else 0
        merged = [_cat_fp8(parts, axis) for parts in results]
        value = _cat_fp8(merged, 0) if len(merged) > 1 else merged[0]
        if tuple(value.shape) != self.shape:
            raise ValueError("FP8 FT restored projection shape mismatch")
        return value

    def input_scale(self, actual_name, companion_name, companion_is_reciprocal):
        if actual_name is None:
            return torch.ones(1, dtype=torch.float32)
        values = []
        for peer in sorted({piece[0] for piece in self._pieces()}):
            actual = self._read(actual_name, peer).float().reshape(-1)
            companion = self._read(companion_name, peer).float().reshape(-1)
            if (
                actual.numel() != 1
                or companion.numel() != 1
                or not bool(torch.isfinite(actual).all())
                or bool((actual <= 0).any())
            ):
                raise ValueError(
                    "FP8 FT static activation scale must be finite positive scalar"
                )
            expected = actual.reciprocal() if companion_is_reciprocal else actual
            if not torch.allclose(companion, expected):
                raise ValueError("FP8 FT activation scale pair is inconsistent")
            values.append(actual)
        return torch.stack(values).max().reshape(1)


class PerChannelFP8FTProjection(PerTensorFP8FTProjection):
    """Undo main's flat-storage reshape, retaining each output-channel scale.

    A source dtype of e4m3fnuz records ROCm's reversible dtype conversion: same
    bits and doubled scales. Decode that conversion before the existing runtime
    performs its one-time platform preparation. No source platform is guessed.
    """

    def _kernel_shape(self, peer):
        shape = super()._kernel_shape(peer)
        if len(shape) != 2:
            raise ValueError("Per-channel FP8 FT requires flat matrix storage")
        return shape[1], shape[0]

    def _dtype(self, peer):
        value = self.checkpoint.read_rank_slice(
            self.kernel_name, peer, (slice(0, 1), slice(0, 1))
        )
        if value.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            raise TypeError("Per-channel FP8 FT requires an FP8 source dtype")
        return value.dtype

    def _read(self, name, peer):
        value = super()._read(name, peer)
        if name == self.kernel_name:
            source_dtype = value.dtype
            value = value.reshape(self._kernel_shape(peer))
            if not bool(torch.isfinite(value.float()).all()):
                raise ValueError("Per-channel FP8 FT contains nonfinite weights")
            if value.dtype == torch.float8_e4m3fnuz:
                value = value.view(torch.float8_e4m3fn)
            elif value.dtype != torch.float8_e4m3fn:
                raise TypeError("Per-channel FP8 FT requires an FP8 source dtype")
            value = self._undo_rocm_dense_layout(value, source_dtype)
        return value

    def _selected_scale(self, piece):
        peer, part, begin, end, width = piece
        shape = self._kernel_shape(peer)
        scale = super()._read(self.scale_name, peer).float()
        if tuple(scale.shape) != (1, shape[0]):
            raise ValueError("Per-channel FP8 FT scale flat-storage shape mismatch")
        scale = scale.reshape(-1, 1)
        if self._dtype(peer) == torch.float8_e4m3fnuz:
            scale = scale * 0.5
        if self.projection == "gate_up":
            scale = scale[part * width + begin : part * width + end]
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("Per-channel FP8 FT requires positive finite used scales")
        return scale

    def scale(self):
        pieces = self._pieces()
        if self.projection == "gate_up":
            return torch.cat(
                [
                    self._selected_scale(p)
                    for part in (0, 1)
                    for p in pieces
                    if p[1] == part
                ],
                dim=0,
            )
        values = [self._selected_scale(p) for p in pieces]
        return torch.stack(values).amax(dim=0)

    def weight(self):
        pieces, destination_scale = self._pieces(), self.scale()
        results = [[], []] if self.projection == "gate_up" else [[]]
        cache = {}
        for piece in pieces:
            peer, part, begin, end, width = piece
            if peer not in cache:
                cache[peer] = self._read(self.kernel_name, peer)
            value = cache[peer]
            if self.projection == "gate_up":
                value = value[part * width + begin : part * width + end]
            elif self.projection == "down":
                value = value[:, begin:end]
                scale = self._selected_scale(piece)
                if not torch.equal(scale, destination_scale):
                    value = (value.float() * (scale / destination_scale)).to(
                        value.dtype
                    )
            results[part].append(value)
        if self.projection in ("gate_up", "down"):
            width = pieces[0][-1]
            total = self.local_width * self.tp
            axis = 0 if self.projection == "gate_up" else 1
            for peer in range(self.tp):
                padding = max(0, total - peer * width)
                if padding >= width:
                    continue
                value = cache.get(peer)
                if value is None:
                    value = self._read(self.kernel_name, peer)
                for part in range(len(results)):
                    offset = part * width + padding if axis == 0 else padding
                    if torch.count_nonzero(
                        value.narrow(axis, offset, width - padding).float()
                    ).item():
                        raise ValueError(
                            "Per-channel FP8 FT padding contains nonzero values"
                        )
        axis = 1 if self.projection == "down" else 0
        merged = [_cat_fp8(values, axis) for values in results]
        result = _cat_fp8(merged, 0) if len(merged) == 2 else merged[0]
        if tuple(result.shape) != self.shape:
            raise ValueError("Per-channel FP8 FT restored projection shape mismatch")
        return result
