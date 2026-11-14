"""Online NVFP4 packing after native projection fusion and TP slicing."""

import torch


def nvfp4_scale_keys(name):
    return name + ".nvfp4.local", name + ".nvfp4.global"


def nvfp4_global_scale(tensor):
    # Match the repository's online NVFP4 convention, without a full FP32 copy.
    return tensor.abs().amax().float().clamp_min(1e-8).reshape(1) / (6.0 * 448.0)


def quantize_nvfp4_weight(matrix, *, device=None):
    from flashinfer import fp4_quantize

    outer = nvfp4_global_scale(matrix)
    if not torch.isfinite(outer).all().item():
        raise ValueError("NVFP4 source weights must be finite")
    if matrix.is_cuda:
        codes, scales = fp4_quantize(matrix.contiguous(), outer.reciprocal())
        return codes, scales, outer
    # CPU conversion must not stage the complete dense weight on the GPU.
    # Each 128-row scale tile is independent under one shared global scale.
    rows, columns = matrix.shape
    device = torch.device("cuda") if device is None else device
    scale_columns = (columns // 16 + 3) // 4 * 4
    codes = torch.empty((rows, columns // 2), dtype=torch.uint8)
    scales = torch.empty(((rows + 127) // 128 * 128, scale_columns), dtype=torch.uint8)
    inverse = outer.to(device=device).reciprocal()
    for begin in range(0, rows, 1024):
        end = min(begin + 1024, rows)
        chunk = matrix[begin:end].to(device=device)
        chunk_codes, chunk_scales = fp4_quantize(chunk, inverse)
        codes[begin:end].copy_(chunk_codes)
        scales[begin : begin + chunk_scales.shape[0]].copy_(chunk_scales)
        del chunk, chunk_codes, chunk_scales
    return codes, scales, outer


def unswizzle_nvfp4_scales(scales, rows, columns):
    """Recover logical scale rows from CUTLASS's padded 128x4 layout."""
    padded_rows, padded_columns = (rows + 127) // 128 * 128, (columns + 3) // 4 * 4
    return (
        scales.contiguous()
        .view(torch.uint8)
        .reshape(padded_rows // 128, padded_columns // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(padded_rows, padded_columns)[:rows, :columns]
    )


def swizzle_nvfp4_scales(scales):
    rows, columns = scales.shape
    padded_rows, padded_columns = (rows + 127) // 128 * 128, (columns + 3) // 4 * 4
    padded = torch.zeros(
        (padded_rows, padded_columns), dtype=torch.uint8, device=scales.device
    )
    padded[:rows, :columns] = scales.contiguous().view(torch.uint8)
    return (
        padded.reshape(padded_rows // 128, 4, 32, padded_columns // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(padded_rows, padded_columns)
    )


def repack_nvfp4_projections(parts):
    """Slice/concatenate N rows of one packed GEMM without requantization.

    Each item is (projection, start, end). All fragments retain the same
    weight global scale. Reinterleave only local scales; KV/gate boundaries
    need not coincide with the original 128-row scale tile.
    """
    first = parts[0][0]
    codes, scales = [], []
    for projection, start, end in parts:
        if (
            not projection.dynamic_input_scale
            or projection.bias is not None
            or projection.weight.dtype != torch.uint8
            or projection.hidden_size != first.hidden_size
            or projection.weight_scale_2 is not first.weight_scale_2
            or not (0 <= start < end <= projection.output_size)
            or start % 32
            or end % 32
        ):
            raise ValueError(
                "natural NVFP4 slices require aligned rows and one shared global scale"
            )
        local = unswizzle_nvfp4_scales(
            projection.weight_scales,
            projection.output_size,
            projection.hidden_size // 16,
        )
        codes.append(projection.weight[start:end])
        scales.append(local[start:end])
    result = type(first)(
        weight=torch.cat(codes, 0).contiguous(),
        weight_scales=swizzle_nvfp4_scales(torch.cat(scales, 0)),
        weight_scale_2=first.weight_scale_2,
        quant_config=first.quant_config,
    )
    result.logical_output_size = result.output_size
    return result


class OnlineNvfp4Projection:
    def __init__(self, source, *, module):
        self.source, self.module, self.name = source, module, source.name
        self.local_key, self.global_key = nvfp4_scale_keys(source.name)
        self.logical_n_key = source.name + ".nvfp4.logical_n"

    def get_tensor_names(self, layer_id, load_config):
        return self.source.get_tensor_names(layer_id, load_config)

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        if load_config.merge_lora:
            raise ValueError("K3.1 NVFP4 does not support load-time LoRA merging")
        return self.source._load_raw_tensor(
            tensor_source, layer_id, device, load_config
        )

    def _split(self, tensor, load_config):
        # Native KDA/DSA loaders own component order, head slicing, replicated
        # F_a/Q/KV and preserved-layout query CP. Never slice packed FP4 bytes.
        return self.source._split(tensor, load_config)

    def _postprocess(self, tensor, device, load_config):
        matrix = tensor[self.name].T.contiguous()  # native [K,N] -> FP4 [N,K]
        if matrix.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"{self.name}: online NVFP4 requires original BF16/FP16 weights"
            )
        requested_device = torch.device(device)
        quant_device = matrix.device if matrix.is_cuda else requested_device
        if quant_device.type != "cuda":
            quant_device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(quant_device) not in ((10, 0), (10, 3)):
            raise ValueError("K3.1 NVFP4 requires CUDA SM100/SM103")
        if torch.version.cuda is None or int(torch.version.cuda.split(".")[0]) < 13:
            raise ValueError("K3.1 NVFP4 requires CUDA13")
        logical_n, k = matrix.shape
        if k % 32:
            raise ValueError(
                f"{self.name}: rank-local NVFP4 K must align to 32, got {k}"
            )
        padding = (-logical_n) % 32
        if padding:
            if self.module != "kda.input":
                raise ValueError(
                    f"{self.name}: NVFP4 N must align to 32, got {logical_n}"
                )
            matrix = torch.nn.functional.pad(matrix, (0, 0, 0, padding))
        codes, scales, outer = quantize_nvfp4_weight(matrix, device=quant_device)
        result = {
            self.name: codes,
            self.local_key: scales,
            self.global_key: outer,
            self.logical_n_key: torch.tensor(
                logical_n, dtype=torch.int64, device=matrix.device
            ),
        }
        return {
            name: value.to(device=requested_device) for name, value in result.items()
        }
