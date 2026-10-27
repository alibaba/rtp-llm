"""D3 code-readout prefill-only last-token readout.

The backbone supplies final-normalized activations, including expanded media
positions. This module projects only the final text token of each complete
row with the checkpoint's fp32 ``[255, hidden]`` readout weight and returns
raw logits for every answer code in sidecar order. Candidate masking,
temperature scaling and softmax belong to the frontend and run exactly once;
the engine never normalizes.

Rows are complete prompts: the handler supports no prefix reuse and no
chunked prefill (``supports_prefix_cache`` is False, so the engine rejects
cached or chunked rows before the readout). The runtime temporary KV/GDN
storage a prefill needs is unaffected by that. Importing this module loads
neither PyTorch nor the native RTP-LLM runtime (see d3_config for the
sidecar contract).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from rtp_llm.models.downstream_modules.d3_config import D3Config

if TYPE_CHECKING:
    import torch


class D3Handler:
    prefill_only = True
    supports_prefix_cache = False
    need_post_process = False

    def __init__(self, config: D3Config):
        self.config = config
        self._weight: torch.Tensor | None = None
        self._device: torch.device | None = None

    def custom_weight_info(self) -> list[Any]:
        # The generic weight loader casts backbone weights; the readout stays FP32.
        return []

    def init_cpp_handler(self) -> None:
        pass

    def init(self, tensors: Mapping[str, torch.Tensor]) -> None:
        import torch

        if set(tensors) != {self.config.readout_key}:
            raise ValueError(
                f"D3 readout must contain exactly {self.config.readout_key!r}"
            )
        weight = tensors[self.config.readout_key]
        shape = (len(self.config.codes), self.config.hidden_size)
        if weight.dtype != torch.float32 or tuple(weight.shape) != shape:
            raise ValueError(
                f"D3 {self.config.readout_key} must be fp32 with shape {shape}"
            )
        if not torch.isfinite(weight).all().item():
            raise ValueError(f"D3 {self.config.readout_key} contains nonfinite weights")
        self._weight = weight.detach()
        self._device = None

    def extend_forward_args(self) -> list[str]:
        return ["hidden_states", "input_ids", "input_lengths"]

    def select_token_position(
        self,
        input_ids: torch.Tensor,
        text_tokens_mask: torch.Tensor | None = None,
    ) -> int:
        """Select after multimodal expansion; media tokens may precede readout."""
        import torch

        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.device.type != "cpu"
            or input_ids.ndim != 1
            or input_ids.dtype not in (torch.int32, torch.int64)
            or input_ids.numel() == 0
        ):
            raise ValueError(
                "D3 admission requires a nonempty one-dimensional CPU token row"
            )
        if (
            self.config.max_length is not None
            and input_ids.numel() > self.config.max_length
        ):
            raise ValueError(
                "D3 expanded row exceeds the checkpoint max_length; "
                "nothing is truncated"
            )
        if text_tokens_mask is not None:
            if (
                not isinstance(text_tokens_mask, torch.Tensor)
                or text_tokens_mask.device.type != "cpu"
                or text_tokens_mask.ndim != 1
                or text_tokens_mask.numel() != input_ids.numel()
                or text_tokens_mask.dtype not in (torch.bool, torch.int32, torch.int64)
                or not bool(text_tokens_mask[-1].item())
            ):
                raise ValueError(
                    "D3 text_tokens_mask must align with expanded tokens "
                    "and end in text"
                )
        return input_ids.numel() - 1

    def extend_forward(
        self,
        *,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        import torch
        import torch.nn.functional as functional

        if self._weight is None:
            raise RuntimeError("D3 readout has not been initialized")
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[1] != self.config.hidden_size
            or not hidden_states.is_floating_point()
        ):
            raise ValueError(
                "D3 hidden_states must have shape [tokens, hidden_size] "
                "and floating dtype"
            )
        for name, vector in (
            ("input_ids", input_ids),
            ("input_lengths", input_lengths),
        ):
            if (
                not isinstance(vector, torch.Tensor)
                or vector.ndim != 1
                or vector.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError(f"D3 {name} must be a one-dimensional integer tensor")
        if not input_lengths.numel() or input_ids.numel() != hidden_states.shape[0]:
            raise ValueError("D3 requires complete, unpadded prefill rows")

        def require(condition: torch.Tensor, message: str) -> None:
            if condition.device.type == "cpu":
                if not condition.item():
                    raise ValueError(message)
            else:
                # Do not synchronize or copy GPU IDs/lengths back to the CPU.
                torch._assert_async(condition, message)

        require((input_lengths > 0).all(), "D3 row lengths must be positive")
        require(
            input_lengths.sum() == hidden_states.shape[0],
            "D3 requires complete, unpadded prefill rows",
        )
        if self.config.max_length is not None:
            require(
                (input_lengths <= self.config.max_length).all(),
                "D3 expanded row exceeds the checkpoint max_length",
            )
        device = hidden_states.device
        if self._device != device:
            self._weight = self._weight.to(device=device)
            self._device = device
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            # Lengths describe expanded, executed rows, never raw prompt
            # tokenizer lengths. The readout is the last token's hidden state
            # times the fp32 weight: raw logits, no mask, temperature or softmax.
            lengths = input_lengths.to(device=device, dtype=torch.int64)
            last = hidden_states.index_select(0, lengths.cumsum(0) - 1).float()
            return functional.linear(last, self._weight)


class D3Module:
    """The CustomModule interface, without importing model-loader/native code."""

    def __init__(self, config: Any, tokenizer: Any):
        import torch

        if config.compute_dtype != torch.bfloat16:
            raise ValueError(
                "D3 requires BF16 backbone activations; set --act_type bf16"
            )
        quant_config = getattr(config, "quant_config", None)
        quant_algo = config.quant_algo
        requested_quantization = getattr(config, "quantization", None)
        has_quantization = (
            quant_algo.isQuant()
            or quant_config is not None
            or str(requested_quantization or "").lower() not in ("", "none")
        )
        # Use the resolved checkpoint configuration, not just the CLI string:
        # checkpoint quantization takes precedence in ModelConfig. The NVIDIA
        # target is FP8 per-block weights (8-bit, group_size=128).
        fp8_per_block = (
            not torch.version.hip
            and quant_config is not None
            and getattr(quant_config, "get_method", lambda: None)() == "FP8_PER_BLOCK"
            and quant_algo.isQuant()
            and getattr(quant_algo, "isFp8", lambda: False)()
            and getattr(quant_algo, "getWeightBits", lambda: None)() == 8
            and getattr(quant_algo, "getGroupSize", lambda: None)() == 128
        )
        # PTPC is a distinct native quantization method (isFp8() is false);
        # the ROCm target is per-channel weights with per-token activations.
        fp8_per_channel = (
            bool(torch.version.hip)
            and quant_config is not None
            and getattr(quant_config, "get_method", lambda: None)()
            == "FP8_PER_CHANNEL_COMPRESSED"
            and getattr(quant_config, "bits", None) == 8
            and getattr(quant_config, "group_size", lambda: None)() == 0
            and quant_algo.isQuant()
            and getattr(quant_algo, "isFp8PTPC", lambda: False)()
            and getattr(quant_algo, "getWeightBits", lambda: None)() == 8
            and getattr(quant_algo, "getGroupSize", lambda: None)() == 0
        )
        layered_quantization = (
            getattr(config, "enable_nvfp4_dense_ffn", False)
            or getattr(config, "enable_w4a16_sm120_dense_ffn", False)
            or getattr(quant_config, "enable_nvfp4_dense_ffn", False)
        )
        if (
            has_quantization and not (fp8_per_block or fp8_per_channel)
        ) or layered_quantization:
            raise ValueError(
                "D3 supports only BF16 or FP8_PER_BLOCK backbone weights "
                "(8-bit, group_size=128) on CUDA; ROCm also supports "
                "FP8_PER_CHANNEL_COMPRESSED (8-bit, group_size=0), "
                "without layered quantization"
            )
        self.config_ = config
        self.tokenizer_ = tokenizer
        self.d3_config = D3Config.load(config.ckpt_path)
        if config.hidden_size != self.d3_config.hidden_size:
            raise ValueError("D3 readout hidden_size does not match the backbone")
        if self.d3_config.readout_dtype != "float32":
            raise ValueError("this engine computes the D3 readout in float32")
        # The noncausal hookup lives in Qwen35Dense._create_config; refusing a
        # mismatched wiring here keeps a sidecar on a wrong model class from
        # silently running causal (or bidirectional) attention.
        attn_config = getattr(config, "attn_config", None)
        if attn_config is None:
            is_causal = True
        else:
            is_causal = bool(getattr(attn_config, "is_causal", True))
        if self.d3_config.attention_mode == "noncausal_full_attention":
            if is_causal:
                raise ValueError(
                    "D3 noncausal_full_attention requires the Qwen35Dense config "
                    "hook that sets attn_config.is_causal=False"
                )
        elif not is_causal:
            raise ValueError(
                "a causal D3 checkpoint must keep attn_config.is_causal=True"
            )
        if tokenizer is not None:
            if hasattr(tokenizer, "get_real_tokenizer"):
                tokenizer = tokenizer.get_real_tokenizer()
            for code, token_id in zip(self.d3_config.codes, self.d3_config.token_ids):
                if list(tokenizer.encode(code, add_special_tokens=False)) != [token_id]:
                    raise ValueError(
                        f"D3 answer code {code!r} does not match the "
                        "checkpoint tokenizer"
                    )
        self.handler = D3Handler(self.d3_config)

    def get_handler(self) -> D3Handler:
        return self.handler

    def get_custom_weight_info(self) -> list[Any]:
        return []

    def init(self, weight: Any) -> None:
        from safetensors.torch import load_file

        tensors = load_file(
            str(Path(self.config_.ckpt_path) / self.d3_config.readout_file)
        )
        self.handler.init(tensors)
