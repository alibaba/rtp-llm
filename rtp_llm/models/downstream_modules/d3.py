"""D3 code-readout prefill-only readout (last token, or joint markers).

The backbone supplies final-normalized activations, including expanded media
positions. A single-question (version 1) checkpoint projects only the final
text token of each complete row with the checkpoint's fp32 ``[255, hidden]``
readout weight; a joint (version 2) checkpoint projects the hidden state at
every ``<|fim_suffix|>`` readout marker of a row instead and packs each
row's question readouts side by side into that row's single engine output
row. Both return raw logits for every answer code in sidecar order;
candidate masking, temperature scaling and softmax belong to the frontend
and run exactly once; the engine never normalizes.

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

    def __init__(self, config: D3Config, marker_id: int | None = None):
        self.config = config
        self.joint = config.joint is not None
        if self.joint:
            if type(marker_id) is not int or marker_id < 0:
                raise ValueError(
                    "a joint D3 handler requires the marker token id"
                )
        elif marker_id is not None:
            raise ValueError("only joint D3 checkpoints take a marker token id")
        self.marker_id = marker_id
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
        allowed = (torch.float32, torch.bfloat16) if self.joint else (torch.float32,)
        if weight.dtype not in allowed or tuple(weight.shape) != shape:
            raise ValueError(
                f"D3 {self.config.readout_key} must be "
                f"{'fp32 or bf16' if self.joint else 'fp32'} with shape {shape}"
            )
        if not torch.isfinite(weight.float()).all().item():
            raise ValueError(f"D3 {self.config.readout_key} contains nonfinite weights")
        # The joint reference checkpoints ship a bf16 readout and the engine
        # computes the projection in fp32 either way, so upcast once here.
        self._weight = weight.detach().to(dtype=torch.float32)
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
        if self.joint:
            markers = (input_ids == self.marker_id).nonzero().flatten()
            if markers.numel() < 1:
                raise ValueError(
                    "a joint row requires at least one readout marker token"
                )
            return int(markers[-1].item())
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
            # tokenizer lengths. The readout is raw logits, no mask,
            # temperature or softmax.
            lengths = input_lengths.to(device=device, dtype=torch.int64)
            if self.joint:
                return self._joint_readout(hidden_states, input_ids, lengths)
            # The single-question readout is the last token's hidden state
            # times the fp32 weight.
            last = hidden_states.index_select(0, lengths.cumsum(0) - 1).float()
            return functional.linear(last, self._weight)

    def _joint_readout(
        self,
        hidden_states: "torch.Tensor",
        input_ids: "torch.Tensor",
        lengths: "torch.Tensor",
    ) -> "torch.Tensor":
        """Read out at every joint marker and pack one row per sequence.

        The engine contract returns exactly one row per sequence, so a joint
        row of ``N`` questions packs its ``N`` readouts side by side: the row
        is ``[255 * max_question_count]`` wide, each question's 255 logits at
        its own block offset, zero padded. The frontend knows each row's own
        question count and slices its blocks; rows never share markers.
        """
        import torch
        import torch.nn.functional as functional

        rows = int(lengths.numel())
        ids = input_ids.to(device=lengths.device)
        markers = (ids == self.marker_id).nonzero().flatten()
        if markers.numel() < rows:
            raise ValueError(
                "every joint row requires at least one readout marker token"
            )
        row_of_marker = torch.searchsorted(lengths.cumsum(0), markers, right=True)
        counts = torch.bincount(row_of_marker, minlength=rows)
        if not bool((counts > 0).all().item()):
            raise ValueError(
                "every joint row requires at least one readout marker token"
            )
        gathered = hidden_states.index_select(0, markers).float()
        logits = functional.linear(gathered, self._weight)
        width = int(counts.max().item()) * len(self.config.codes)
        starts = counts.cumsum(0) - counts
        within = (
            torch.arange(markers.numel(), device=lengths.device) - starts[row_of_marker]
        )
        packed = torch.zeros(
            (rows, width), device=lengths.device, dtype=torch.float32
        )
        packed.view(rows, -1, len(self.config.codes))[row_of_marker, within] = logits
        return packed


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
        if self.d3_config.readout_dtype != "float32" and self.d3_config.joint is None:
            raise ValueError("this engine computes the D3 readout in float32")
        marker_id: int | None = None
        if self.d3_config.joint is not None:
            if tokenizer is None:
                raise ValueError(
                    "a joint D3 checkpoint requires a tokenizer to resolve "
                    "its readout marker"
                )
            real = tokenizer
            if hasattr(real, "get_real_tokenizer"):
                real = real.get_real_tokenizer()
            marker = self.d3_config.joint.marker
            ids = real.encode(marker, add_special_tokens=False)
            if len(ids) != 1 or real.convert_ids_to_tokens(ids[0]) != marker:
                raise ValueError(
                    f"the joint readout marker {marker!r} must be a single "
                    "existing token"
                )
            marker_id = ids[0]
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
        self.handler = D3Handler(self.d3_config, marker_id=marker_id)

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
