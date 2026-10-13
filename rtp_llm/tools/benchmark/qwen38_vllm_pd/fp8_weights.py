"""Online FP8 storage methods used by the matched Qwen P/D baseline."""

import torch
from torch import nn
from torch.nn import functional as F


def hard_fp8(weight):
    scale = weight.float().abs().amax().clamp_min(1e-12).reshape(1) / 448.0
    return (weight.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


def hard_block_fp8(weight):
    n, k = weight.shape
    padded = weight.new_zeros((((n + 127) // 128) * 128, ((k + 127) // 128) * 128))
    padded[:n, :k] = weight
    blocks = padded.view(-1, 128, padded.shape[1] // 128, 128)
    scale = blocks.abs().float().amax((1, 3), keepdim=True).clamp_min(1e-4) / 448
    quantized = (blocks * (1 / scale)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized.view_as(padded)[:n, :k].contiguous(), scale.reshape(
        blocks.shape[0], blocks.shape[2]
    )


def scaled_projection(x, weight, scale, compute_weight=None, bias=None):
    if compute_weight is not None:
        return F.linear(x, compute_weight, bias)
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    if rows.shape[0] == 0:
        return x.new_empty((*x.shape[:-1], weight.shape[0]))
    from vllm import _custom_ops as ops

    quantized, activation_scale = ops.scaled_fp8_quant(
        rows, num_token_padding=max(16, ((rows.shape[0] + 15) // 16) * 16)
    )
    output = torch._scaled_mm(
        quantized,
        weight.t(),
        scale_a=activation_scale,
        scale_b=scale,
        out_dtype=torch.bfloat16,
    )
    if isinstance(output, tuple):
        output = output[0]
    output = output[: rows.shape[0]].reshape(*x.shape[:-1], weight.shape[0])
    return (output if bias is None else output + bias).to(x.dtype)


def process_tensor_weight(layer):
    if not layer.weight.is_cuda:
        raise ValueError("Main vocabulary and dense projections must remain on GPU")
    weight, scale = hard_fp8(layer.weight.data)
    layer.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
    layer.register_buffer("qwen_fp8_scale", scale.float())
    aligned = weight.is_cuda and weight.shape[0] % 16 == 0 and weight.shape[1] % 16 == 0
    layer.qwen_fp8_compute_weight = (
        None if aligned else (weight.float() * scale).to(torch.bfloat16)
    )


def install():
    """Install only in the isolated benchmark interpreter, before model creation."""
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
        VocabParallelEmbedding,
    )

    class OnlineFp8Linear(UnquantizedLinearMethod):
        def __init__(self, block_quant=False):
            super().__init__()
            self.block_quant = block_quant
            self.native_method = None

        def create_weights(self, layer, *args, **kwargs):
            self.create_args = args
            self.create_kwargs = kwargs.copy()
            super().create_weights(layer, *args, **kwargs)

        def process_weights_after_loading(self, layer):
            if not self.block_quant:
                process_tensor_weight(layer)
                return
            from vllm.model_executor.layers.quantization.fp8 import (
                Fp8Config,
                Fp8LinearMethod,
            )

            original = layer.weight.detach()
            weight, scale = hard_block_fp8(original)
            method = Fp8LinearMethod(Fp8Config(weight_block_size=[128, 128]))
            del layer._parameters["weight"]
            with torch.device(original.device):
                method.create_weights(layer, *self.create_args, **self.create_kwargs)
            layer.weight.data.copy_(weight)
            layer.weight_scale_inv.data.copy_(scale)
            method.process_weights_after_loading(layer)
            self.native_method = method

        def apply(self, layer, x, bias=None):
            if self.native_method is not None:
                return self.native_method.apply(layer, x, bias)
            return scaled_projection(
                x,
                layer.weight,
                layer.qwen_fp8_scale,
                layer.qwen_fp8_compute_weight,
                bias,
            )

    class OnlineFp8Embedding(UnquantizedEmbeddingMethod):
        def process_weights_after_loading(self, layer):
            process_tensor_weight(layer)

        def embedding(self, layer, input_):
            if not layer.weight.is_cuda:
                raise ValueError("Only the separate PLE big table may be on CPU")
            # Main vocabulary lookup stays on the GPU.
            values = F.embedding(input_, layer.weight.view(torch.uint8))
            values = values.view(torch.float8_e4m3fn).to(torch.bfloat16)
            return values * layer.qwen_fp8_scale.to(values.device)

        def apply(self, layer, x, bias=None):
            return scaled_projection(
                x,
                layer.weight,
                layer.qwen_fp8_scale,
                layer.qwen_fp8_compute_weight,
                bias,
            )

    original_linear_init = LinearBase.__init__

    def linear_init(self, *args, **kwargs):
        original_linear_init(self, *args, **kwargs)
        # Expert matrices are managed by the serialized NVFP4 MoE method.
        if (
            self.prefix.startswith(("model.", "mtp.", "language_model."))
            and ".experts" not in self.prefix
        ):
            block_quant = "attn" in self.prefix and self.prefix.endswith(
                (
                    ".qkv_proj",
                    ".o_proj",
                    ".in_proj_qkvz",
                    ".in_proj_qkv",
                    ".in_proj_z",
                    ".out_proj",
                )
            )
            self.quant_method = OnlineFp8Linear(block_quant)

    LinearBase.__init__ = linear_init
    original_embedding_init = VocabParallelEmbedding.__init__

    def embedding_init(self, *args, **kwargs):
        # PLE has its own FP8 pinned-host method and global checkpoint scale.
        prefix = kwargs.get("prefix", "")
        if "ngram_embedding" not in prefix and kwargs.get("quant_method") is None:
            kwargs["quant_method"] = OnlineFp8Embedding()
        original_embedding_init(self, *args, **kwargs)

    VocabParallelEmbedding.__init__ = embedding_init


def quantize_compute_views(model):
    """Keep FP8 norm/conv storage and explicit BF16 compute views, as in RTP."""
    for module in model.modules():
        for name, parameter in list(module.named_parameters(recurse=False)):
            if (
                parameter.ndim == 2
                or parameter.dtype not in (torch.bfloat16, torch.float16, torch.float32)
                or "scale" in name
            ):
                continue
            weight, scale = hard_fp8(parameter.data)
            module.register_buffer(name + "_qwen_fp8_storage", weight)
            module.register_buffer(name + "_qwen_fp8_storage_scale", scale)
            parameter.data = (weight.float() * scale).to(parameter.dtype)
