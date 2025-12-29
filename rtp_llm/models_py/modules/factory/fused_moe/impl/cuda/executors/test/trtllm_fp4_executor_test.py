import pytest
from abc import ABC, abstractmethod
from typing import Dict
import torch
from cuda.bindings import runtime
from torch.nn import functional as F

from flashinfer import (
    RoutingMethodType,
    GatedActType,
    e2m1_and_ufp8sf_scale_to_float,
    fp4_quantize,
    mxfp8_dequantize_host,
    mxfp8_quantize,
    reorder_rows_for_gated_act_gemm,
    shuffle_matrix_a,
)
from flashinfer.autotuner import autotune
from flashinfer.fp4_quantization import block_scale_interleave
from flashinfer.fused_moe import (
    WeightLayout,
    convert_to_block_layout,
    trtllm_fp4_block_scale_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_per_tensor_scale_moe,
    trtllm_bf16_moe,
)
from flashinfer.fused_moe.core import (
    get_w2_permute_indices_with_cache,
    _maybe_get_cached_w3_w1_permute_indices,
)

from enum import IntEnum
from flashinfer.utils import get_compute_capability

# Import RTP-LLM specific modules for Fp4Executor backend
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.ops import ParallelismConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import MoEConfigAdapter
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.trtllm_fp4_executor import (
    TrtllmFp4Executor,
)
from rtp_llm.utils.model_weight import W

class QuantMode(IntEnum):
    """Supported quantization modes for MoE testing."""

    FP4_NVFP4_NVFP4 = 1
    BF16 = 2

def check_cuda(err):
    """Unified CUDA error checking function used throughout the file."""
    if err != runtime.cudaError_t.cudaSuccess:
        error_name = runtime.cudaGetErrorName(err)
        error_string = runtime.cudaGetErrorString(err)
        raise RuntimeError(f"CUDA error: {error_name[1]}: {error_string[1]}")


class CUDAGraphMoE:
    """
    Simple CUDA Graph wrapper for MoE operations.

    The graph captures tensor references and automatically updates them during execution.

    Three core methods: capture(), launch(), cleanup()

    Usage:
        cuda_graph = CUDAGraphMoE(moe_impl, static_data, **config)
        cuda_graph.capture(hidden_states_sample, expert_logits=logits, routing_bias=bias)
        output = cuda_graph.launch(new_hidden_states)  # Repeat as needed
        cuda_graph.cleanup()
    """

    def __init__(self, moe_impl, static_data, **config):
        self.moe_impl = moe_impl
        self.static_data = static_data
        self.config = config
        self.enable_autotune = config.get("enable_autotune", True)
        self.graph = None
        self.graph_exec = None
        self.stream = None
        self.input_tensor = None
        self.output_tensor = None
        self.is_captured = False

    def capture(self, hidden_states_sample, **runtime_args):
        """Capture CUDA graph with the given sample input."""
        if self.is_captured:
            raise RuntimeError(
                "Graph already captured. Call cleanup() first to re-capture."
            )
        if not isinstance(self.moe_impl, FP4Moe):
            raise NotImplementedError(
                f"CUDA graph capture not yet implemented for {type(self.moe_impl)}"
            )

        # Create stream
        err, self.stream = runtime.cudaStreamCreate()
        check_cuda(err)

        # Get the raw stream pointer for PyTorch
        stream_ptr = int(self.stream)
        torch_stream = torch.cuda.ExternalStream(stream_ptr)

        # Store input tensor reference (will be updated in place during launch)
        self.input_tensor = hidden_states_sample.clone()

        # Warmup
        with torch.cuda.stream(torch_stream), autotune(self.enable_autotune):
            for _ in range(1):
                self._run_moe_computation(runtime_args)

        # Synchronize our stream after warmup
        err = runtime.cudaStreamSynchronize(self.stream)[0]
        check_cuda(err)

        # Begin capture
        err, self.graph = runtime.cudaGraphCreate(0)
        check_cuda(err)
        err = runtime.cudaStreamBeginCapture(
            self.stream, runtime.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
        )[0]
        check_cuda(err)

        try:
            # Capture computation on our stream
            with torch.cuda.stream(torch_stream):
                self.output_tensor = self._run_moe_computation(runtime_args)
            err, self.graph = runtime.cudaStreamEndCapture(self.stream)
            check_cuda(err)
            err, self.graph_exec = runtime.cudaGraphInstantiate(self.graph, 0)
            check_cuda(err)
            self.is_captured = True
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"CUDA graph capture failed: {e}") from e

    def launch(self, hidden_states_new):
        """Launch captured CUDA graph with new input."""
        if not self.is_captured:
            raise RuntimeError("Graph not captured. Call capture() first.")

        # Update input tensor in place
        self.input_tensor.copy_(hidden_states_new)

        # Launch graph
        err = runtime.cudaGraphLaunch(self.graph_exec, self.stream)[0]
        check_cuda(err)
        err = runtime.cudaStreamSynchronize(self.stream)[0]
        check_cuda(err)

        # Return output tensor (automatically updated by graph execution)
        return self.output_tensor

    def cleanup(self):
        """Clean up all CUDA graph resources."""
        if self.graph_exec is not None:
            err = runtime.cudaGraphExecDestroy(self.graph_exec)[0]
            check_cuda(err)
            self.graph_exec = None
        if self.graph is not None:
            err = runtime.cudaGraphDestroy(self.graph)[0]
            check_cuda(err)
            self.graph = None
        if self.stream is not None:
            err = runtime.cudaStreamDestroy(self.stream)[0]
            check_cuda(err)
            self.stream = None
        self.input_tensor = None
        self.output_tensor = None
        self.is_captured = False

    def _run_moe_computation(self, runtime_args):
        """Run the MoE computation."""
        input_quantized = self.moe_impl.quantize_inputs(
            self.input_tensor,
            self.config["hidden_states_scale_global"],
            is_swizzling=False,
        )

        output = trtllm_fp4_block_scale_moe(
            routing_logits=runtime_args["expert_logits"],
            routing_bias=runtime_args["routing_bias"],
            hidden_states=input_quantized["hidden_states"],
            hidden_states_scale=input_quantized["hidden_states_scale"],
            gemm1_weights=self.static_data["gemm1_weights_fp4_shuffled"],
            gemm1_weights_scale=self.static_data["gemm1_scales_fp4_shuffled"],
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=self.static_data["gemm2_weights_fp4_shuffled"],
            gemm2_weights_scale=self.static_data["gemm2_scales_fp4_shuffled"],
            gemm2_bias=None,
            output1_scale_scalar=self.static_data["scale_c_fc1"],
            output1_scale_gate_scalar=self.static_data["scale_gate_fc1"],
            output2_scale_scalar=self.static_data["scale_c_fc2"],
            num_experts=self.config["num_experts"],
            top_k=self.config["top_k"],
            n_group=self.config["n_groups"],
            topk_group=self.config["top_k_groups"],
            intermediate_size=self.config["intermediate_size"],
            local_expert_offset=0,
            local_num_experts=self.config["num_experts"],
            routed_scaling_factor=self.config["routed_scaling"],
            tile_tokens_dim=None,
            routing_method_type=self.config["routing_method_type"],
            gated_act_type=self.config["gated_act_type"],
            do_finalize=True,
        )
        return output  # Extract tensor from tuple


# ====================================================================================
# Abstract Base Class for MoE Implementations
# ====================================================================================


class Moe(ABC):
    """Abstract base class for MoE implementations."""

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def quantize_weights(self, gemm1_weights, gemm2_weights, hidden_states_sample):
        """Quantize static weights and compute global scale factors (done offline)."""
        pass

    @abstractmethod
    def quantize_inputs(self, hidden_states, hidden_states_scale_global):
        """Quantize dynamic inputs/hidden states using pre-computed global scale (done at runtime)."""
        pass

    @abstractmethod
    def prepare_static_weights_for_kernel(
        self,
        args_dequant,
        args,
        gemm1_weights_orig,
        gemm2_weights_orig,
        hidden_size,
        intermediate_size,
        num_experts,
        weight_processing,
    ):
        """
        Prepare quantized weights for kernel (done offline with weights).

        Args:
            args_dequant: Contains c_global_sf and other dequantization parameters
            args: Contains already quantized weights (gemm1_weights, gemm2_weights) and scales
            gemm1_weights_orig: Original unquantized FC1 weights (used by FP4 for re-quantization)
            gemm2_weights_orig: Original unquantized FC2 weights (used by FP4 for re-quantization)

        Note:
            - FP4 implementations use both original weights (for linear layout quantization)
              and args.gemm*_weights (for swizzled layout)
            - FP8 implementations typically only use args.gemm*_weights (already quantized)
        """
        pass

    @abstractmethod
    def call_moe(
        self, static_data, hidden_states_orig, hidden_states_scale_global, **kwargs
    ):
        """Call MoE with runtime input quantization + kernel execution (done at runtime)."""
        pass

    @abstractmethod
    def compute_reference(self, args):
        """Compute reference output using dequantized operations."""
        pass

    def compute_production(self, args_dequant, args, **kwargs):
        """Unified actual computation that delegates to implementation-specific methods."""
        return _compute_moe_actual_unified(self, args_dequant, args, **kwargs)

    @abstractmethod
    def get_tolerances(self):
        """Get accuracy tolerances for this quantization mode."""
        pass

    def __str__(self):
        return self.name


# ====================================================================================
# FP4 Quantization Implementation
# ====================================================================================


class FP4Moe(Moe):
    """
    FP4 NvFP4 / MxFP4 MoE implementation with block scaling.
    Args:
        is_mxfp4: Whether to use MxFP4 or NvFP4 weight quantization
            If True, the activation is quantized to MxFP8, else the activation is quantized to NvFP4
    """

    def __init__(self, quant_mode: QuantMode):
        super().__init__()
        self.quant_mode = quant_mode
        self.is_mxfp4 = (
            quant_mode == QuantMode.FP4_MXFP4_MXFP8
            or quant_mode == QuantMode.FP4_MXFP4_Bf16
        )
        self.sf_vec_size = 32 if self.is_mxfp4 else 16

    def quantize_weights(self, gemm1_weights, gemm2_weights, hidden_states_sample):
        """Quantize weights to FP4 format and compute global scale factors."""
        num_experts = gemm1_weights.shape[0]
        # Compute global scale factor for hidden states (offline calibration)
        if self.quant_mode == QuantMode.FP4_NVFP4_NVFP4:
            # nvfp4 hidden states
            hidden_states_scale_global = calculate_fp4_global_scale_factor(
                hidden_states_sample,
                False,
            )
        else:
            # mxfp8 / bf16 hidden states
            hidden_states_scale_global = 1.0

        # Quantize the weights for FC1
        gemm1_weights_fp4_bytes, gemm1_scales_fp4_bytes, gemm1_scales_global = (
            quant_fp4_batches(gemm1_weights, num_experts, self.is_mxfp4, True)
        )

        # Quantize the weights for FC2
        gemm2_weights_fp4_bytes, gemm2_scales_fp4_bytes, gemm2_scales_global = (
            quant_fp4_batches(gemm2_weights, num_experts, self.is_mxfp4, True)
        )

        return {
            "hidden_states_scale_global": hidden_states_scale_global,
            "gemm1_weights": gemm1_weights_fp4_bytes,
            "gemm1_scales": gemm1_scales_fp4_bytes,
            "gemm1_scales_global": gemm1_scales_global,
            "gemm2_weights": gemm2_weights_fp4_bytes,
            "gemm2_scales": gemm2_scales_fp4_bytes,
            "gemm2_scales_global": gemm2_scales_global,
        }

    def quantize_inputs(
        self, hidden_states, hidden_states_scale_global, is_swizzling=True
    ):
        if self.quant_mode == QuantMode.FP4_MXFP4_MXFP8:
            """Quantize hidden states to MxFP8 format."""
            hidden_states_quant, hidden_states_scale = mxfp8_quantize(
                hidden_states, is_swizzling
            )
            hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(
                *hidden_states.shape[:-1], -1
            )
            return {
                "hidden_states": hidden_states_quant,
                "hidden_states_scale": hidden_states_scale,
            }
        elif self.quant_mode == QuantMode.FP4_NVFP4_NVFP4:
            """Quantize hidden states to NvFP4 format using pre-computed global scale."""
            (
                hidden_states_fp4_bytes,
                hidden_states_scale_fp4_bytes,
                _,
            ) = quant_fp4(
                hidden_states, hidden_states_scale_global, False, is_swizzling
            )
            hidden_states_scale_fp4_bytes = hidden_states_scale_fp4_bytes.view(
                torch.float8_e4m3fn
            ).reshape(*hidden_states.shape[:-1], -1)

            return {
                "hidden_states": hidden_states_fp4_bytes,
                "hidden_states_scale": hidden_states_scale_fp4_bytes,
            }
        else:  # bf16
            return {
                "hidden_states": hidden_states.to(torch.bfloat16),
                "hidden_states_scale": None,
            }

    def prepare_static_weights_for_kernel(
        self,
        args_dequant,
        args,
        gemm1_weights_orig,
        gemm2_weights_orig,
        hidden_size,
        intermediate_size,
        num_experts,
        weight_processing,
    ):
        """Prepare quantized weights for kernel (done offline with weights)."""
        use_ue8m0 = self.is_mxfp4
        epilogue_tile_m = 128  # FIXME: this depends on the kernel internals

        # Quantize weights with linear layout for kernels
        _, gemm1_scales_linear_fp4_bytes, _ = quant_fp4_batches(
            gemm1_weights_orig, num_experts, use_ue8m0, False
        )
        _, gemm2_scales_linear_fp4_bytes, _ = quant_fp4_batches(
            gemm2_weights_orig, num_experts, use_ue8m0, False
        )

        # Convert quantized weights to proper formats
        gemm1_weights_fp4 = args.gemm1_weights.view(torch.float8_e4m3fn).reshape(
            num_experts, 2 * intermediate_size, hidden_size // 2
        )  # packed fp4
        gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(
            num_experts, 2 * intermediate_size, hidden_size // self.sf_vec_size
        )  # fp8 scaling factors

        gemm2_weights_fp4 = args.gemm2_weights.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, intermediate_size // 2
        )  # packed fp4
        gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(
            num_experts, hidden_size, intermediate_size // self.sf_vec_size
        )  # fp8 scaling factors

        # Using cached permute index calculation can speed up weights preprocessing
        gemm1_weights_fp4_shuffled = []
        gemm1_scales_fp4_shuffled = []
        gemm2_weights_fp4_shuffled = []
        gemm2_scales_fp4_shuffled = []
        for i in range(num_experts):
            # Calculate the permute indices for the following:
            # 1. Reorder rows of W1 and scales for fused gated activation
            # 2. Shuffle weights and scaling factors for transposed mma output
            # for both w3_w1 and w2 weights and scale factors
            permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                self._cache_permute_indices,
                gemm1_weights_fp4[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm1_weights_fp4_shuffled.append(
                gemm1_weights_fp4[i]
                .view(torch.uint8)[permute_indices.to(gemm1_weights_fp4.device)]
                .contiguous()
            )

            permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
                self._cache_permute_indices,
                gemm1_scales_linear_fp4[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm1_scales_fp4_shuffled.append(
                block_scale_interleave(
                    gemm1_scales_linear_fp4[i]
                    .view(torch.uint8)[
                        permute_sf_indices.to(gemm1_scales_linear_fp4.device)
                    ]
                    .contiguous()
                )
            )

            permute_indices = get_w2_permute_indices_with_cache(
                self._cache_permute_indices,
                gemm2_weights_fp4[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm2_weights_fp4_shuffled.append(
                gemm2_weights_fp4[i]
                .view(torch.uint8)[permute_indices.to(gemm2_weights_fp4.device)]
                .contiguous()
            )

            permute_sf_indices = get_w2_permute_indices_with_cache(
                self._cache_permute_indices,
                gemm2_scales_linear_fp4[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm2_scales_fp4_shuffled.append(
                block_scale_interleave(
                    gemm2_scales_linear_fp4[i]
                    .view(torch.uint8)[
                        permute_sf_indices.to(gemm2_scales_linear_fp4.device)
                    ]
                    .contiguous()
                )
            )

        # Stack weights for all experts
        gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
        gemm1_scales_fp4_shuffled = (
            torch.stack(gemm1_scales_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(
                num_experts, 2 * intermediate_size, hidden_size // self.sf_vec_size
            )
        )

        gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
        gemm2_scales_fp4_shuffled = (
            torch.stack(gemm2_scales_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, hidden_size, intermediate_size // self.sf_vec_size)
        )

        # Calculate scaling factors that depend on weights
        scale_c_fc1 = (
            args_dequant.c_global_sf
            * (1.0 / args.gemm1_scales_global)
            * (1.0 / args.hidden_states_scale_global)
        )
        scale_gate_fc1 = (1.0 / args.gemm1_scales_global) * (
            1.0 / args.hidden_states_scale_global
        )
        scale_c_fc2 = (1.0 / args_dequant.c_global_sf) * (
            1.0 / args.gemm2_scales_global
        )

        return {
            "gemm1_weights_fp4_shuffled": gemm1_weights_fp4_shuffled,
            "gemm1_scales_fp4_shuffled": gemm1_scales_fp4_shuffled,
            "gemm2_weights_fp4_shuffled": gemm2_weights_fp4_shuffled,
            "gemm2_scales_fp4_shuffled": gemm2_scales_fp4_shuffled,
            "scale_c_fc1": scale_c_fc1,
            "scale_gate_fc1": scale_gate_fc1,
            "scale_c_fc2": scale_c_fc2,
        }

    def call_moe(
        self, static_data, hidden_states_orig, hidden_states_scale_global, **kwargs
    ):
        """Call MoE using CUDA graph for maximum performance (create, capture, launch)."""
        # Extract runtime arguments
        expert_logits = kwargs["expert_logits"]
        routing_bias = kwargs["routing_bias"]
        num_experts = kwargs["num_experts"]
        top_k = kwargs["top_k"]
        n_groups = kwargs["n_groups"]
        top_k_groups = kwargs["top_k_groups"]
        intermediate_size = kwargs["intermediate_size"]
        routed_scaling = kwargs["routed_scaling"]
        gated_act_type = kwargs["gated_act_type"]
        routing_method_type = kwargs["routing_method_type"]
        enable_autotune = kwargs.get("enable_autotune", True)

        # Create CUDA graph configuration
        config = {
            "hidden_states_scale_global": hidden_states_scale_global,
            "num_experts": num_experts,
            "top_k": top_k,
            "n_groups": n_groups,
            "top_k_groups": top_k_groups,
            "intermediate_size": intermediate_size,
            "routed_scaling": routed_scaling,
            "gated_act_type": gated_act_type,
            "routing_method_type": routing_method_type,
            "enable_autotune": enable_autotune,
        }

        runtime_args = {
            "expert_logits": expert_logits,
            "routing_bias": routing_bias,
        }

        # Create, capture and launch CUDA graph in one shot
        cuda_graph = CUDAGraphMoE(self, static_data, **config)
        try:
            cuda_graph.capture(hidden_states_orig, **runtime_args)
            output = cuda_graph.launch(hidden_states_orig)
            return output[0].to(torch.float)
        finally:
            cuda_graph.cleanup()

    def compute_reference(self, args):
        return run_moe_reference_fp4(args, self.quant_mode)

    def get_tolerances(self):
        """Get FP4-specific accuracy tolerances."""
        return {"atol": 0.1, "rtol": 0.85, "percent": 0.925}

class FP4MoeExecutor(FP4Moe):
    def call_moe(
        self, static_data, hidden_states_orig, hidden_states_scale_global, **kwargs
    ):
        model_config = ModelConfig()
        model_config.expert_num = kwargs["num_experts"]
        model_config.hidden_size = kwargs["hidden_size"]
        model_config.moe_inter_size = kwargs["intermediate_size"]
        model_config.moe_k = kwargs["top_k"]
        parallelism_config = ParallelismConfig()
        parallelism_config.dp_size = 1
        parallelism_config.tp_size = 1
        parallelism_config.ep_size = 1
        config = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
        )
        payload = ExpertForwardPayload(
            expert_x=hidden_states_orig,
            expert_x_origin_dtype=torch.bfloat16,
            expert_topk_ids=kwargs["topk_ids"],
            expert_topk_weights=kwargs["topk_weights"],
        )
        weights = {
            W.moe_w1: static_data["gemm1_weights_fp4_shuffled"],
            W.moe_w2: static_data["gemm2_weights_fp4_shuffled"],
            W.moe_s1: static_data["gemm1_scales_fp4_shuffled"],
            W.moe_s2: static_data["gemm2_scales_fp4_shuffled"],
            "w13_input_scale": kwargs["w13_input_scale"],
            "w13_weight_scale_2": kwargs["w13_weight_scale_2"],
            "w2_input_scale": kwargs["w2_input_scale"],
            "w2_weight_scale_2": kwargs["w2_weight_scale_2"],
        }

        executor = TrtllmFp4Executor(config, weights, FusedMoEQuantConfig())
        output = executor.execute(payload, "silu", None, None, False, None)
        return output.to(torch.float)

# ====================================================================================
# BF16 Implementation
# ====================================================================================


class BF16Moe(Moe):
    """BF16 MoE implementation."""

    def quantize_weights(self, gemm1_weights, gemm2_weights, hidden_states_sample):
        """No scaling for weights."""
        return {
            "hidden_states_scale_global": None,
            "gemm1_weights": gemm1_weights.to(torch.bfloat16),
            "gemm1_scales": None,
            "gemm1_scales_global": None,
            "gemm2_weights": gemm2_weights.to(torch.bfloat16),
            "gemm2_scales": None,
            "gemm2_scales_global": None,
        }

    def quantize_inputs(self, hidden_states, *unused_args):
        """No scaling for hidden states."""
        return {
            "hidden_states": hidden_states.to(torch.bfloat16),
            "hidden_states_scale": None,
        }

    def prepare_static_weights_for_kernel(
        self,
        args_dequant,
        args,
        gemm1_weights_orig,
        gemm2_weights_orig,
        hidden_size,
        intermediate_size,
        num_experts,
        weight_processing,
    ):
        """Prepare quantized weights for kernel (done offline with weights)."""

        # Use shuffled weights with BlockMajorK layout for better performance
        use_shuffled_weight = weight_processing["use_shuffled_weight"]
        weight_layout = weight_processing["layout"]

        if use_shuffled_weight:
            # FIXME: this depends on the kernel internals
            epilogue_tile_m = 128

            # Reorder rows of W1 for fused gated activation and shuffle for both W1 and W2
            # Using cached permute index calculation can speed up weights preprocessing
            gemm1_weights_bf16_shuffled = []
            gemm2_weights_bf16_shuffled = []
            for i in range(num_experts):
                permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                    self._cache_permute_indices,
                    args.gemm1_weights[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                tmp_weights1 = (
                    args.gemm1_weights[i]
                    .view(torch.uint8)[permute_indices.to(args.gemm1_weights.device)]
                    .contiguous()
                )

                permute_indices = get_w2_permute_indices_with_cache(
                    self._cache_permute_indices,
                    args.gemm2_weights[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                tmp_weights2 = (
                    args.gemm2_weights[i]
                    .view(torch.uint8)[permute_indices.to(args.gemm2_weights.device)]
                    .contiguous()
                )

                if weight_layout == WeightLayout.BlockMajorK:
                    block_k = 128
                    tmp_weights1 = convert_to_block_layout(
                        tmp_weights1.view(torch.uint8), block_k
                    )
                    tmp_weights2 = convert_to_block_layout(
                        tmp_weights2.view(torch.uint8), block_k
                    )

                gemm1_weights_bf16_shuffled.append(tmp_weights1.view(torch.bfloat16))
                gemm2_weights_bf16_shuffled.append(tmp_weights2.view(torch.bfloat16))

            # Stack weights for all experts
            gemm1_weights_bf16_shuffled = (
                torch.stack(gemm1_weights_bf16_shuffled)
                .view(torch.bfloat16)
                .contiguous()
            )
            gemm2_weights_bf16_shuffled = (
                torch.stack(gemm2_weights_bf16_shuffled)
                .view(torch.bfloat16)
                .contiguous()
            )

            return {
                "gemm1_weights": gemm1_weights_bf16_shuffled,
                "gemm2_weights": gemm2_weights_bf16_shuffled,
                "use_shuffled_weight": use_shuffled_weight,
                "weight_layout": weight_layout,
            }

    def call_moe(
        self, static_data, hidden_states_orig, hidden_states_scale_global, **kwargs
    ):
        """Call MoE with runtime input quantization + kernel execution (done at runtime)."""
        expert_logits = kwargs["expert_logits"]
        routing_bias = kwargs["routing_bias"]
        num_experts = kwargs["num_experts"]
        top_k = kwargs["top_k"]
        n_groups = kwargs["n_groups"]
        top_k_groups = kwargs["top_k_groups"]
        intermediate_size = kwargs["intermediate_size"]
        routing_method_type = kwargs["routing_method_type"]
        enable_autotune = kwargs.get("enable_autotune", True)

        # Use autotuner for optimal kernel selection
        with autotune(enable_autotune):
            output = trtllm_bf16_moe(
                expert_logits,  # float
                routing_bias,
                hidden_states_orig,
                static_data["gemm1_weights"],
                static_data["gemm2_weights"],
                num_experts,
                top_k,
                n_groups,
                top_k_groups,
                intermediate_size,
                0,
                num_experts,
                use_shuffled_weight=static_data["use_shuffled_weight"],
                weight_layout=static_data["weight_layout"],
                routing_method_type=routing_method_type,
            )
        return output.to(torch.float)

    def compute_reference(self, args):
        """BF16 reference implementation."""
        return run_moe_reference_bf16(args)

    def get_tolerances(self):
        """Get BF16 accuracy tolerances."""
        return {"atol": 0.1, "rtol": 0.85, "percent": 0.925}


# ====================================================================================
# Fp4Executor Implementation (using TrtllmFp4Executor as backend)
# ====================================================================================

class moe_args:
    """Arguments container for MoE operations."""

    def __init__(
        self,
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        hidden_states,
        hidden_states_scale,
        hidden_states_scale_global,
        expert_logits,
        gemm1_weights,
        gemm1_scales,
        gemm1_scales_global,
        gemm2_weights,
        gemm2_scales,
        gemm2_scales_global,
        permute_info,
        use_routing_scales_on_input,
        gated_act_type,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.padding = padding
        self.hidden_states = hidden_states
        self.hidden_states_scale = hidden_states_scale
        self.hidden_states_scale_global = hidden_states_scale_global
        self.expert_logits = expert_logits
        self.gemm1_weights = gemm1_weights
        self.gemm1_scales = gemm1_scales
        self.gemm1_scales_global = gemm1_scales_global
        self.gemm2_weights = gemm2_weights
        self.gemm2_scales = gemm2_scales
        self.gemm2_scales_global = gemm2_scales_global
        self.permute_info = permute_info
        self.use_routing_scales_on_input = use_routing_scales_on_input
        self.gated_act_type = gated_act_type


class moe_args_dequant:
    """Arguments container for dequantized MoE operations."""

    def __init__(
        self,
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        hidden_states,
        expert_logits,
        gemm1_weights,
        gemm2_weights,
        permute_info,
        use_routing_scales_on_input,
        gated_act_type,
        hidden_states_scale=None,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.padding = padding
        self.hidden_states = hidden_states
        self.expert_logits = expert_logits
        self.gemm1_weights = gemm1_weights
        self.gemm2_weights = gemm2_weights
        self.permute_info = permute_info
        self.use_routing_scales_on_input = use_routing_scales_on_input
        self.gated_act_type = gated_act_type
        self.hidden_states_scale = hidden_states_scale


def routing_reference(expertLogits, topK, padding):
    """Reference routing implementation for permutation calculation."""
    originalDevice = expertLogits.device
    expertLogits = expertLogits.cpu()
    numTokens, numExperts = expertLogits.shape
    assert topK <= numExperts

    numTokensPerExpert = torch.zeros(numExperts, dtype=torch.int64)
    expandedTokenIdxToExpert = -torch.ones(numTokens * topK, dtype=torch.int64)
    expandedTokenIdxToIdxInExpert = -torch.ones(numTokens * topK, dtype=torch.int64)

    topKLogits, topKIndices = torch.topk(expertLogits, topK, dim=1)
    for tokenIdx in range(numTokens):
        for k in range(topK):
            expandedIdx = tokenIdx * topK + k
            expertIndex = topKIndices[tokenIdx, k]
            expandedTokenIdxToExpert[expandedIdx] = expertIndex
            expandedTokenIdxToIdxInExpert[expandedIdx] = numTokensPerExpert[expertIndex]
            numTokensPerExpert[expertIndex] += 1

    paddedTokensPerExpertPrefixSum = torch.zeros(numExperts + 1, dtype=torch.int64)
    for ii in range(numExperts):

        def divUpMul(a, b):
            return (a + b - 1) // b * b

        paddedTokensPerExpertPrefixSum[ii + 1] = paddedTokensPerExpertPrefixSum[
            ii
        ] + divUpMul(numTokensPerExpert[ii], padding)
    permutedBufferSize = paddedTokensPerExpertPrefixSum[numExperts]

    expandedTokenIdxToPermutedIdx = -torch.ones(numTokens * topK, dtype=torch.int64)
    permutedIdxToExpandedIdx = -torch.ones(permutedBufferSize, dtype=torch.int64)
    permutedIdxToTokenIdx = -torch.ones(permutedBufferSize, dtype=torch.int64)
    for tokenIdx in range(numTokens):
        for k in range(topK):
            expandedIdx = tokenIdx * topK + k
            expert = expandedTokenIdxToExpert[expandedIdx]
            offsetWithinExpert = expandedTokenIdxToIdxInExpert[expandedIdx]
            offsetForExpert = paddedTokensPerExpertPrefixSum[expert]
            permutedIdx = offsetForExpert + offsetWithinExpert

            expandedTokenIdxToPermutedIdx[expandedIdx] = permutedIdx
            permutedIdxToExpandedIdx[permutedIdx] = expandedIdx
            permutedIdxToTokenIdx[permutedIdx] = tokenIdx
    return {
        "paddedTokensPerExpertPrefixSum": paddedTokensPerExpertPrefixSum.to(
            originalDevice
        ),
        "permutedBufferSize": permutedBufferSize.item(),
        "expandedTokenIdxToPermutedIdx": expandedTokenIdxToPermutedIdx.to(
            originalDevice
        ),
        "permutedIdxToExpandedIdx": permutedIdxToExpandedIdx.to(originalDevice),
        "numTokensPerExpert": numTokensPerExpert.to(originalDevice),
        "expandedTokenIdxToExpert": expandedTokenIdxToExpert.to(originalDevice),
        "topKLogits": topKLogits.to(originalDevice),
        "permutedIdxToTokenIdx": permutedIdxToTokenIdx.to(originalDevice),
        "topKIndices": topKIndices.to(originalDevice),
    }


def noaux_tc_ref(logits, bias, n_group, topk_group, top_k, routed_scaling_factor):
    """DeepSeek-style no-aux routing reference implementation."""
    scores = F.sigmoid(logits)
    scores_with_bias = scores + bias
    if n_group > 1:
        scores_shape = list(scores_with_bias.shape)
        group_scores = torch.sum(
            torch.topk(
                scores_with_bias.view(
                    scores_shape[:-1] + [n_group, scores_shape[-1] // n_group]
                ),
                k=2,
                dim=-1,
                largest=True,
                sorted=True,
            )[0],
            dim=-1,
        )
        _, group_idx = torch.topk(
            group_scores, k=topk_group, dim=-1, largest=True, sorted=True
        )
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(-1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(scores_shape[:-1] + [n_group, scores_shape[-1] // n_group])
            .reshape(scores_shape)
        )
        scores_with_bias = scores_with_bias * score_mask

    _, topk_idx = torch.topk(
        scores_with_bias, k=top_k, dim=-1, largest=True, sorted=True
    )
    new_mask = torch.zeros_like(scores)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = scores * new_mask
    score_sum = torch.sum(scores, dim=-1, keepdim=True) + 1e-20
    scores = scores / score_sum * routed_scaling_factor
    return scores


def routing_reference_no_aux(
    expert_logits,
    routing_bias,
    top_k,
    n_groups,
    top_k_groups,
    routed_scaling,
    padding,
    use_routing_scales_on_input=False,
):
    """Tiered TopK routing used by DeepSeek."""
    routing_logits = expert_logits.to(dtype=torch.float, device="cuda")
    if use_routing_scales_on_input:
        # if using routing scales on input, topK == 1 and the score is a plain sigmoid
        scores = F.sigmoid(routing_logits)
    else:
        scores = noaux_tc_ref(
            routing_logits, routing_bias, n_groups, top_k_groups, top_k, routed_scaling
        )
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


def routing_reference_renormalize(expert_logits, top_k, num_experts, padding):
    """TopK -> Softmax routing reference."""
    topk_values, topk_idx = torch.topk(expert_logits, k=top_k, dim=-1)
    topk_values = torch.nn.functional.softmax(topk_values.float(), dim=-1)

    new_mask = torch.zeros_like(expert_logits)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = expert_logits * new_mask

    for i in range(topk_idx.shape[0]):
        for j in range(topk_idx.shape[1]):
            scores[i, topk_idx[i, j]] = topk_values[i, j]
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


def routing_reference_renormalize_naive(expert_logits, top_k, num_experts, padding):
    """Softmax->TopK -> Normalize routing reference."""
    norm_topk_prob = True
    scores = torch.nn.functional.softmax(expert_logits.float(), dim=-1)
    topk_values, topk_idx = torch.topk(scores, k=top_k, dim=-1)

    if norm_topk_prob:  # only diff with mixtral sparse moe block!
        topk_values /= topk_values.sum(dim=-1, keepdim=True)
    topk_values = topk_values.to(expert_logits.dtype)
    scores = scores.to(expert_logits.dtype)

    new_mask = torch.zeros_like(expert_logits)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = expert_logits * new_mask

    for i in range(topk_idx.shape[0]):
        for j in range(topk_idx.shape[1]):
            scores[i, topk_idx[i, j]] = topk_values[i, j]
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


def routing_reference_topk(expert_logits, top_k, num_experts, padding):
    """TopK only (no softmax) routing reference."""
    topk_values, topk_idx = torch.topk(expert_logits, k=top_k, dim=-1)

    new_mask = torch.zeros_like(expert_logits)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = expert_logits * new_mask

    for i in range(topk_idx.shape[0]):
        for j in range(topk_idx.shape[1]):
            scores[i, topk_idx[i, j]] = topk_values[i, j]
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


def check_accuracy(a, b, atol, rtol, percent):
    """Unified accuracy checking function with detailed error reporting."""
    if not torch.isfinite(a).all():
        raise Exception("Non-finite values in reference output")
    if not torch.isfinite(b).all():
        raise Exception("Non-finite values in actual output")
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    close = torch.isclose(a, b, atol=atol, rtol=rtol)
    match_ratio = close.float().mean()
    if match_ratio >= percent:
        return

    mismatch_percent = 1.0 - match_ratio.item()
    if mismatch_percent > 1 - percent:
        raise Exception(
            f"Mismatch percentage is {mismatch_percent:.4f} for rtol {rtol} "
            f"(threshold: {1 - percent:.4f})"
        )


# ====================================================================================
# FP4 Quantization Functions
# ====================================================================================


def calculate_fp4_global_scale_factor(tensor, use_ue8m0=False):
    """
    Calculate FP4 global scale factor for a tensor.

    NOTE: In production, global scale factors are typically obtained offline during:
    - Post-Training Quantization (PTQ) calibration process
    - Quantization-Aware Training (QAT) process

    This function is used here for testing/reference purposes.
    Formula: (448 * 6) represents max representable value in FP4 format.
    """
    if use_ue8m0:
        return torch.tensor(1.0, dtype=torch.float32)
    else:
        return (448 * 6) / tensor.float().abs().nan_to_num().max()


def e2m1_and_ufp8_scale_batches(
    mat_fp4: torch.Tensor,
    scale_tensor: torch.Tensor,
    global_scale_tensor: torch.Tensor,
    sf_vec_size: int,
    ufp8_type: int = 1,
):
    """Batch FP4 dequantization helper."""
    num_batches = mat_fp4.size(0)
    scale_tensor = scale_tensor.view(num_batches, -1)

    tensors = [
        e2m1_and_ufp8sf_scale_to_float(
            mat_fp4[b, :, :].cpu(),
            scale_tensor[b, :].cpu().reshape(-1),
            global_scale_tensor[b].cpu(),
            sf_vec_size,
            ufp8_type,
            True,  # is_sf_swizzled_layout
        )
        for b in range(num_batches)
    ]

    result = torch.stack(tensors)
    return result


def quant_fp4(a, a_global_sf, use_ue8m0=False, is_sf_swizzled_layout=True):
    """
    Quantize FP4 with pre-computed global scale factor.

    This function expects global scale factors that have been pre-computed offline
    during PTQ/QAT calibration process. The global scale factor should NOT be
    computed at runtime to avoid performance overhead.

    Pure function - same inputs always produce same outputs.
    """
    sf_vec_size = 32 if use_ue8m0 else 16

    a_fp4, a_sf = fp4_quantize(
        a.cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0, is_sf_swizzled_layout
    )

    return a_fp4, a_sf, a_global_sf


def quant_fp4_batches(a, num_experts, use_ue8m0=False, is_sf_swizzled_layout=True):
    """FP4 batch quantization function with centralized global scale factor calculation."""
    quant_a = []
    sfs = []
    global_sfs = []
    for i in range(num_experts):
        # Use centralized global scale factor calculation
        a_global_sf = calculate_fp4_global_scale_factor(a[i], use_ue8m0)
        a_fp4, a_sf, _ = quant_fp4(a[i], a_global_sf, use_ue8m0, is_sf_swizzled_layout)
        quant_a.append(a_fp4)
        sfs.append(a_sf)
        global_sfs.append(a_global_sf)

    result_quant_a = torch.stack(quant_a)
    result_sfs = torch.stack(sfs)
    result_global_sfs = torch.stack(global_sfs)

    return result_quant_a, result_sfs, result_global_sfs


def quant_dequant_fp4(a, use_ue8m0=False, is_sf_swizzled_layout=True):
    """FP4 quantize-dequantize roundtrip function with centralized global scale factor calculation."""
    # Use centralized global scale factor calculation
    a_global_sf = calculate_fp4_global_scale_factor(a, use_ue8m0)
    sf_vec_size = 32 if use_ue8m0 else 16

    a_fp4, a_sf = fp4_quantize(
        a.cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0, is_sf_swizzled_layout
    )

    a_pt = e2m1_and_ufp8sf_scale_to_float(
        a_fp4.cpu(),
        a_sf.cpu().reshape(-1),
        (1 / a_global_sf).cpu(),
        sf_vec_size,
        1 if not use_ue8m0 else 0,  # ufp8_type
        is_sf_swizzled_layout,
    )

    return a_pt.cuda(), a_global_sf


# ====================================================================================
# Common MoE Reference Implementation
# ====================================================================================


def run_moe_dequant(args, quant_mode: QuantMode):
    """Common dequantized MoE reference implementation."""
    # Permute
    total_num_padded_tokens = args.permute_info["permutedBufferSize"]
    expanded_idx_to_permuted_idx = args.permute_info[
        "expandedTokenIdxToPermutedIdx"
    ].cpu()
    num_tokens_per_expert = args.permute_info["numTokensPerExpert"].cpu()
    permute_output = torch.full(
        (total_num_padded_tokens, args.hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    for i in range(args.num_tokens):
        for j in range(args.top_k):
            permuted_idx = expanded_idx_to_permuted_idx[i * args.top_k + j]
            permute_output[permuted_idx] = args.hidden_states[i]

    # Gemm1
    gemm1_output = torch.full(
        (total_num_padded_tokens, 2 * args.intermediate_size),
        float("nan"),
        device="cuda",
    ).to(torch.float)
    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = permute_output[i : i + my_num_tokens]
        my_b = args.gemm1_weights[expert_idx]
        my_c = my_a @ my_b.t()
        gemm1_output[i : i + my_num_tokens] = my_c
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding

    if args.use_routing_scales_on_input:
        assert args.top_k == 1
        # For each token and its top_k experts
        for token_idx in range(args.num_tokens):
            for k in range(args.top_k):
                # Get the permuted index for this token's k-th expert
                expanded_idx = token_idx * args.top_k + k
                permuted_idx = expanded_idx_to_permuted_idx[expanded_idx]
                expert_weight = args.permute_info["topKLogits"].to(torch.float)
                # Get the expert weight for this token and expert
                weight = expert_weight[token_idx, k]
                # Scale the corresponding row in gemm1_output
                gemm1_output[permuted_idx] *= weight

    # Activation
    activation_output = torch.full(
        (total_num_padded_tokens, args.intermediate_size), float("nan"), device="cuda"
    ).to(torch.float)

    gated_act_type = args.gated_act_type
    gated_act_type_to_func = {
        0: F.silu,
        1: F.gelu,
    }
    gated_act_func = gated_act_type_to_func[gated_act_type]

    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = gemm1_output[i : i + my_num_tokens]
        my_x1 = my_a[:, : args.intermediate_size]
        my_x2 = my_a[:, args.intermediate_size :]
        activation_output[i : i + my_num_tokens] = gated_act_func(my_x2) * my_x1
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding

    if quant_mode == QuantMode.FP4_NVFP4_NVFP4:
        # Use centralized function for activation quantization
        activation_output, c_global_sf = quant_dequant_fp4(
            activation_output.to(torch.bfloat16), False, True
        )
        activation_output = activation_output.to(torch.float)
        args.c_global_sf = c_global_sf
    elif quant_mode == QuantMode.FP8_PER_TENSOR:
        activation_output, c_global_sf = quant_dequant_per_tensor_fp8(
            activation_output.to(torch.bfloat16)
        )
        activation_output = activation_output.to(torch.float)
        args.c_global_sf = c_global_sf
    elif quant_mode == QuantMode.FP4_MXFP4_MXFP8:
        activation_output, scale_bytes = mxfp8_quantize(
            activation_output.to(torch.bfloat16), True
        )
        scale_bytes = scale_bytes.view(torch.uint8).reshape(-1).cpu()
        activation_output = (
            mxfp8_dequantize_host(
                activation_output.cpu().view(torch.uint8), scale_bytes
            )
            .cuda()
            .to(torch.float)
        )
        args.c_global_sf = 1.0
    elif quant_mode == QuantMode.BF16:
        activation_output = activation_output.to(torch.bfloat16).to(torch.float)
        args.c_global_sf = 1.0
    else:  # mxfp4Bf16
        activation_output = activation_output.to(torch.bfloat16).to(torch.float)
        args.c_global_sf = 1.0

    # Gemm2
    gemm2_output = torch.full(
        (total_num_padded_tokens, args.hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = activation_output[i : i + my_num_tokens]
        my_b = args.gemm2_weights[expert_idx]
        my_c = my_a @ my_b.t()
        gemm2_output[i : i + my_num_tokens] = my_c
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding

    # Finalize
    expert_weight = args.permute_info["topKLogits"].to(torch.float)
    finalize_output = torch.full(
        (args.num_tokens, args.hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    for i in range(args.num_tokens):
        acc = torch.zeros(args.hidden_size, dtype=torch.float, device="cuda")
        for top_k_idx in range(args.top_k):
            expanded_idx = i * args.top_k + top_k_idx
            permuted_idx = expanded_idx_to_permuted_idx[expanded_idx]
            original_vector = gemm2_output[permuted_idx]
            weight = (
                expert_weight[i, top_k_idx]
                if not args.use_routing_scales_on_input
                else 1.0
            )
            acc += original_vector * weight
        finalize_output[i] = acc
    return finalize_output


# ====================================================================================
# Quantization-Specific Reference Implementations
# ====================================================================================


def run_moe_reference_fp4(args, quant_mode: QuantMode):
    sf_vec_size = 16 if quant_mode == QuantMode.FP4_NVFP4_NVFP4 else 32
    ufp8_type_weights = 1 if quant_mode == QuantMode.FP4_NVFP4_NVFP4 else 0

    if quant_mode == QuantMode.FP4_NVFP4_NVFP4:
        hidden_states_dequant = e2m1_and_ufp8sf_scale_to_float(
            args.hidden_states.cpu(),
            args.hidden_states_scale.cpu().view(torch.uint8).reshape(-1),
            (1 / args.hidden_states_scale_global).cpu(),
            sf_vec_size,
            ufp8_type_weights,
            True,  # is_sf_swizzled_layout
        ).cuda()
    elif quant_mode == QuantMode.FP4_MXFP4_MXFP8:
        hidden_states_dequant = mxfp8_dequantize_host(
            args.hidden_states.cpu().view(torch.uint8),
            args.hidden_states_scale.cpu().view(torch.uint8).reshape(-1),
            True,  # is_sf_swizzled_layout
        ).cuda()
    else:
        hidden_states_dequant = args.hidden_states.to(torch.bfloat16).to(torch.float)

    gemm1_weights_dequant = e2m1_and_ufp8_scale_batches(
        args.gemm1_weights,
        args.gemm1_scales,
        1 / args.gemm1_scales_global,
        sf_vec_size,
        ufp8_type_weights,
    ).cuda()

    gemm2_weights_dequant = e2m1_and_ufp8_scale_batches(
        args.gemm2_weights,
        args.gemm2_scales,
        1 / args.gemm2_scales_global,
        sf_vec_size,
        ufp8_type_weights,
    ).cuda()

    args_dequant = moe_args_dequant(
        args.num_tokens,
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        args.top_k,
        args.padding,
        hidden_states_dequant,
        args.expert_logits,
        gemm1_weights_dequant,
        gemm2_weights_dequant,
        args.permute_info,
        args.use_routing_scales_on_input,
        args.gated_act_type,
    )

    return run_moe_dequant(args_dequant, quant_mode), args_dequant

def run_moe_reference_bf16(args):
    """BF16 reference implementation."""

    # no scaling for hidden states and weights
    hidden_states_dequant = args.hidden_states.to(torch.float)
    gemm1_weights_dequant = {}
    for i in range(args.num_experts):
        gemm1_weights_dequant[i] = args.gemm1_weights[i].to(torch.float)
    gemm2_weights_dequant = {}
    for i in range(args.num_experts):
        gemm2_weights_dequant[i] = args.gemm2_weights[i].to(torch.float)

    args_dequant = moe_args_dequant(
        args.num_tokens,
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        args.top_k,
        args.padding,
        hidden_states_dequant,
        args.expert_logits,
        gemm1_weights_dequant,
        gemm2_weights_dequant,
        args.permute_info,
        args.use_routing_scales_on_input,
        GatedActType.SwiGlu.value,  # gated_act_type
    )

    return run_moe_dequant(args_dequant, QuantMode.BF16), args_dequant

def _compute_moe_actual_unified(moe_impl, args_dequant, args, **kwargs):
    """Unified actual computation that delegates to implementation-specific methods."""
    # 1. Prepare static weights for the kernel (offline processing)
    static_data = moe_impl.prepare_static_weights_for_kernel(
        args_dequant,
        args,
        kwargs["gemm1_weights_orig"],
        kwargs["gemm2_weights_orig"],
        args.hidden_size,
        args.intermediate_size,
        args.num_experts,
        kwargs["weight_processing"],
    )

    topk_ids = args.permute_info["topKIndices"].to(torch.int32)
    # 2. Call MoE with runtime input quantization + kernel execution
    kernel_kwargs = {
        "expert_logits": kwargs["expert_logits"],
        "routing_bias": kwargs["routing_bias"],
        "num_experts": args.num_experts,
        "num_tokens": args.num_tokens,
        "hidden_size": args.hidden_size,
        "top_k": args.top_k,
        "n_groups": kwargs["n_groups"],
        "top_k_groups": kwargs["top_k_groups"],
        "intermediate_size": args.intermediate_size,
        "routed_scaling": kwargs["routed_scaling"],
        "routing_method_type": kwargs["routing_method_type"],
        "do_finalize": True,
        "gated_act_type": args.gated_act_type,
        "hidden_states_scale": args.hidden_states_scale,
        "hidden_states_quant": kwargs["hidden_states_quant"],
        "enable_autotune": kwargs.get("enable_autotune", True),
        "topk_weights": args.expert_logits.view(args.num_tokens, args.num_experts)[
            torch.arange(args.num_tokens).unsqueeze(1), topk_ids
        ].to(torch.bfloat16),
        "topk_ids": topk_ids,
        "w13_weight_scale_2": 1.0 / args.gemm1_scales_global,
        "w2_weight_scale_2": 1.0 / args.gemm2_scales_global,
        "w13_input_scale": 1.0 / args.hidden_states_scale_global,
        "w2_input_scale": 1.0 / args.hidden_states_scale_global,
    }

    return moe_impl.call_moe(
        static_data,
        kwargs["hidden_states_orig"],
        args.hidden_states_scale_global,
        **kernel_kwargs,
    )


def run_moe_test(
    num_tokens,
    hidden_size,
    intermediate_size,
    moe_impl,
    routing_config,
    weight_processing,
    gated_act_type,
    cache_permute_indices,
):
    """Common test logic for all routing methods."""

    torch.cuda.synchronize()

    moe_impl._cache_permute_indices = cache_permute_indices

    seed = 0
    torch.random.manual_seed(seed)

    # Extract routing configuration
    top_k = routing_config["top_k"]
    padding = routing_config["padding"]
    n_groups = routing_config["n_groups"]
    top_k_groups = routing_config["top_k_groups"]
    routed_scaling = routing_config["routed_scaling"]
    num_experts = routing_config["num_experts"]
    routing_method_type = routing_config["routing_method_type"]

    # Validation checks
    assert top_k <= num_experts
    assert top_k <= 10
    if (top_k_groups is not None) and (n_groups is not None) and (n_groups > 0):
        assert top_k_groups <= 4
        assert num_experts > n_groups
        assert num_experts % n_groups == 0
        assert num_experts % 4 == 0
        assert top_k < (top_k_groups * num_experts / n_groups)

    # Create test data based on routing method
    if routing_method_type == RoutingMethodType.DeepSeekV3:
        expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
            torch.float
        )
    else:
        expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
            torch.bfloat16
        )

    if routing_config["has_routing_bias"]:
        routing_bias = torch.randn(num_experts, device="cuda", dtype=torch.bfloat16)
    else:
        routing_bias = None

    hidden_states = 2 * torch.randn(
        (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    gemm1_weights = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    gemm2_weights = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device="cuda",
        dtype=torch.bfloat16,
    )

    if routing_method_type == RoutingMethodType.DeepSeekV3:
        permute_info, scores = routing_reference_no_aux(
            expert_logits,
            routing_bias,
            top_k,
            n_groups,
            top_k_groups,
            routed_scaling,
            padding,
            False,
        )
    elif routing_method_type == RoutingMethodType.Renormalize:
        permute_info, scores = routing_reference_renormalize(
            expert_logits, top_k, num_experts, padding
        )
    elif routing_method_type == RoutingMethodType.RenormalizeNaive:
        permute_info, scores = routing_reference_renormalize_naive(
            expert_logits, top_k, num_experts, padding
        )
    elif routing_method_type == RoutingMethodType.TopK:
        permute_info, scores = routing_reference_topk(
            expert_logits, top_k, num_experts, padding
        )
    elif routing_method_type == RoutingMethodType.Llama4:
        permute_info, scores = routing_reference_no_aux(
            expert_logits,
            routing_bias,
            top_k,
            n_groups,
            top_k_groups,
            routed_scaling,
            padding,
            use_routing_scales_on_input=True,
        )
    else:
        raise NotImplementedError(
            f"Routing method {routing_method_type} not implemented"
        )

    # 1. Quantize weights offline
    weights_data = moe_impl.quantize_weights(
        gemm1_weights, gemm2_weights, hidden_states
    )

    # 2. Quantize inputs at runtime
    inputs_data = moe_impl.quantize_inputs(
        hidden_states, weights_data["hidden_states_scale_global"]
    )

    # 3. Combine quantized data
    quant_data = {**weights_data, **inputs_data}

    # Create arguments for reference computation
    args = moe_args(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        quant_data["hidden_states"],
        quant_data["hidden_states_scale"],
        quant_data["hidden_states_scale_global"],
        scores,
        quant_data["gemm1_weights"],
        quant_data["gemm1_scales"],
        quant_data["gemm1_scales_global"],
        quant_data["gemm2_weights"],
        quant_data["gemm2_scales"],
        quant_data["gemm2_scales_global"],
        permute_info,
        use_routing_scales_on_input,
        gated_act_type,
    )

    # Compute reference output
    output_dequant_reference, args_dequant = moe_impl.compute_reference(args)

    if output_dequant_reference is None:
        pytest.fail("Reference computation failed to produce output")

    # Compute actual output
    enable_autotune = routing_config.get("enable_autotune", True)

    output_dequant_actual = moe_impl.compute_production(
        args_dequant,
        args,
        expert_logits=expert_logits,
        routing_bias=routing_bias,
        hidden_states_orig=hidden_states,
        gemm1_weights_orig=gemm1_weights,
        gemm2_weights_orig=gemm2_weights,
        n_groups=n_groups,
        top_k_groups=top_k_groups,
        routed_scaling=routed_scaling,
        routing_method_type=routing_method_type,
        weight_processing=weight_processing,
        enable_pdl=True,
        hidden_states_quant=inputs_data["hidden_states"],
        enable_autotune=enable_autotune,
    )

    # Compare outputs
    tolerances = moe_impl.get_tolerances()
    print(output_dequant_reference)
    print(output_dequant_actual)
    check_accuracy(
        output_dequant_reference,
        output_dequant_actual,
        atol=tolerances["atol"],
        rtol=tolerances["rtol"],
        percent=tolerances["percent"],
    )


_cache_permute_indices = {}

if __name__ == "__main__":
    run_moe_test(
        num_tokens=3072,
        hidden_size=1024,
        intermediate_size=768,
        moe_impl=FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4),
        routing_config={
                "num_experts": 128,
                "top_k": 8,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.Renormalize,
                "compatible_moe_impls": [
                    FP4Moe,
                    BF16Moe,
                ],
                "compatible_intermediate_size": [384, 768, 1024],
                "enable_autotune": False,
            },
        weight_processing={
                "use_shuffled_weight": True,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP4Moe],
            },
        gated_act_type=GatedActType.SwiGlu,
        cache_permute_indices=_cache_permute_indices,
    )

# import random
# from typing import Dict, Tuple

# import torch

# from rtp_llm.config.model_config import ModelConfig
# from rtp_llm.ops import ParallelismConfig
# from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import MoEConfigAdapter
# from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
#     ExpertForwardPayload,
# )
# from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
#     FusedMoEQuantConfig,
# )
# from rtp_llm.utils.model_weight import W

# from flashinfer import (
#     fp4_quantize,
#     e2m1_and_ufp8sf_scale_to_float,
# )
# from flashinfer.fused_moe import (
#     RoutingMethodType,
#     GatedActType,
#     trtllm_fp4_block_scale_moe,
# )
# from flashinfer.utils import (
#     device_support_pdl,
# )
# from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.trtllm_fp4_executor import (
#     TrtllmFp4Executor,
# )

# DP_SIZE = 1
# TP_SIZE = 1
# EP_SIZE = 1
# NUM_EXPERTS = 128
# SEQ_LEN = 908
# HIDDEN_SIZE = 2048
# MOE_INTERMEDIATE_SIZE = 768
# TOP_K = 8

# NVFP4_BLOCK_SIZE = 16

# def routing_reference(expertLogits, topK, padding):
#     """Reference routing implementation for permutation calculation."""
#     originalDevice = expertLogits.device
#     expertLogits = expertLogits.cpu()
#     numTokens, numExperts = expertLogits.shape
#     assert topK <= numExperts

#     numTokensPerExpert = torch.zeros(numExperts, dtype=torch.int64)
#     expandedTokenIdxToExpert = -torch.ones(numTokens * topK, dtype=torch.int64)
#     expandedTokenIdxToIdxInExpert = -torch.ones(numTokens * topK, dtype=torch.int64)

#     topKLogits, topKIndices = torch.topk(expertLogits, topK, dim=1)
#     for tokenIdx in range(numTokens):
#         for k in range(topK):
#             expandedIdx = tokenIdx * topK + k
#             expertIndex = topKIndices[tokenIdx, k]
#             expandedTokenIdxToExpert[expandedIdx] = expertIndex
#             expandedTokenIdxToIdxInExpert[expandedIdx] = numTokensPerExpert[expertIndex]
#             numTokensPerExpert[expertIndex] += 1

#     paddedTokensPerExpertPrefixSum = torch.zeros(numExperts + 1, dtype=torch.int64)
#     for ii in range(numExperts):

#         def divUpMul(a, b):
#             return (a + b - 1) // b * b

#         paddedTokensPerExpertPrefixSum[ii + 1] = paddedTokensPerExpertPrefixSum[
#             ii
#         ] + divUpMul(numTokensPerExpert[ii], padding)
#     permutedBufferSize = paddedTokensPerExpertPrefixSum[numExperts]

#     expandedTokenIdxToPermutedIdx = -torch.ones(numTokens * topK, dtype=torch.int64)
#     permutedIdxToExpandedIdx = -torch.ones(permutedBufferSize, dtype=torch.int64)
#     permutedIdxToTokenIdx = -torch.ones(permutedBufferSize, dtype=torch.int64)
#     for tokenIdx in range(numTokens):
#         for k in range(topK):
#             expandedIdx = tokenIdx * topK + k
#             expert = expandedTokenIdxToExpert[expandedIdx]
#             offsetWithinExpert = expandedTokenIdxToIdxInExpert[expandedIdx]
#             offsetForExpert = paddedTokensPerExpertPrefixSum[expert]
#             permutedIdx = offsetForExpert + offsetWithinExpert

#             expandedTokenIdxToPermutedIdx[expandedIdx] = permutedIdx
#             permutedIdxToExpandedIdx[permutedIdx] = expandedIdx
#             permutedIdxToTokenIdx[permutedIdx] = tokenIdx
#     return {
#         "paddedTokensPerExpertPrefixSum": paddedTokensPerExpertPrefixSum.to(
#             originalDevice
#         ),
#         "permutedBufferSize": permutedBufferSize.item(),
#         "expandedTokenIdxToPermutedIdx": expandedTokenIdxToPermutedIdx.to(
#             originalDevice
#         ),
#         "permutedIdxToExpandedIdx": permutedIdxToExpandedIdx.to(originalDevice),
#         "numTokensPerExpert": numTokensPerExpert.to(originalDevice),
#         "expandedTokenIdxToExpert": expandedTokenIdxToExpert.to(originalDevice),
#         "topKLogits": topKLogits.to(originalDevice),
#         "permutedIdxToTokenIdx": permutedIdxToTokenIdx.to(originalDevice),
#         "topKIndices": topKIndices.to(originalDevice),
#     }

# def routing_reference_renormalize(expert_logits, top_k, num_experts, padding):
#     """TopK -> Softmax routing reference."""
#     topk_values, topk_idx = torch.topk(expert_logits, k=top_k, dim=-1)
#     topk_values = torch.nn.functional.softmax(topk_values.float(), dim=-1)

#     new_mask = torch.zeros_like(expert_logits)
#     new_mask.scatter_(-1, topk_idx, 1)
#     scores = expert_logits * new_mask

#     for i in range(topk_idx.shape[0]):
#         for j in range(topk_idx.shape[1]):
#             scores[i, topk_idx[i, j]] = topk_values[i, j]
#     permute_info = routing_reference(scores, top_k, padding)
#     return permute_info, scores

# def _generate_config() -> MoEConfigAdapter:
#     model_config = ModelConfig()
#     model_config.expert_num = NUM_EXPERTS
#     model_config.hidden_size = HIDDEN_SIZE
#     model_config.moe_inter_size = MOE_INTERMEDIATE_SIZE
#     model_config.moe_k = TOP_K
#     parallelism_config = ParallelismConfig()
#     parallelism_config.dp_size = DP_SIZE
#     parallelism_config.tp_size = TP_SIZE
#     parallelism_config.ep_size = EP_SIZE
#     return MoEConfigAdapter(
#         model_config=model_config,
#         parallelism_config=parallelism_config,
#     )

# def _generate_payload_and_weights(
#     config: MoEConfigAdapter,
# ) -> Tuple[ExpertForwardPayload, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
#     hidden_states = torch.empty(
#         (SEQ_LEN, HIDDEN_SIZE),
#         dtype=torch.bfloat16,
#         device='cuda:0',
#     ).normal_(-0.003, 0.15).clamp_(-2.9, 2.2)
#     routing_logits = torch.empty(
#         (SEQ_LEN, config.expert_num),
#         dtype=torch.bfloat16,
#         device='cuda:0',
#     ).normal_(-4.8, 0.86).clamp_(-9.6, -1.3)
#     w13_input_scale = torch.empty(
#         (config.expert_num,),
#         dtype=torch.float32,
#         device='cuda:0',
#     ).fill_(0.0014)
#     w13 = torch.empty(
#         (config.expert_num, config.model_config.moe_inter_size * 2, config.hidden_size // 2),
#         dtype=torch.float32,
#         device='cuda:0',
#     ).normal_(130, 72).clamp_(0, 255).round().to(torch.uint8)
#     w13_scale = torch.empty(
#         (config.expert_num, config.model_config.moe_inter_size * 2, config.hidden_size // NVFP4_BLOCK_SIZE),
#         dtype=torch.float32,
#         device='cuda:0',
#     ).normal_(76.1, 36.3).clamp_(0.1, 448.0).to(torch.float8_e4m3fn)
#     w13_scale_2 = torch.empty(
#         (config.expert_num,),
#         dtype=torch.float32,
#         device='cuda:0',
#     ).normal_(9e-05, 3.6e-05).clamp_(4.8e-05, 0.0002)
#     w2_input_scale = torch.empty(
#         (config.expert_num,),
#         dtype=torch.float32,
#         device='cuda:0',
#     ).fill_(0.0028)
#     w2 = torch.empty(
#         (config.expert_num, config.hidden_size, config.model_config.moe_inter_size // 2),
#         dtype=torch.float32,
#         device='cuda:0',
#     ).normal_(130, 72).clamp_(0, 255).round().to(torch.uint8)
#     w2_scale = torch.empty(
#         (config.expert_num, config.hidden_size, config.model_config.moe_inter_size // NVFP4_BLOCK_SIZE),
#         dtype=torch.float32,
#         device='cuda:0',
#     ).normal_(53.2, 20.7).clamp_(4.5, 448.0).to(torch.float8_e4m3fn)
#     w2_scale_2 = torch.empty(
#         (config.expert_num,),
#         dtype=torch.float32,
#         device='cuda:0',
#     ).normal_(0.0001, 4.2e-05).clamp_(8.5e-05, 0.0003)

#     permute_info, topk_weights = routing_reference_renormalize(
#         routing_logits, config.moe_k, config.expert_num, 8
#     )
#     topk_ids = permute_info["topKIndices"].to(torch.int32)
#     topk_weights = topk_weights.view(SEQ_LEN, config.expert_num)[
#         torch.arange(SEQ_LEN).unsqueeze(1), topk_ids
#     ].to(torch.bfloat16)
#     payload = ExpertForwardPayload(
#         expert_x=hidden_states,
#         expert_x_origin_dtype=torch.bfloat16,
#         expert_topk_ids=topk_ids,
#         expert_topk_weights=topk_weights,
#     )
#     weights = {
#         W.moe_w1: w13,
#         W.moe_w2: w2,
#         W.moe_s1: w13_scale,
#         W.moe_s2: w2_scale,
#         "w13_input_scale": w13_input_scale,
#         "w13_weight_scale_2": w13_scale_2,
#         "w2_input_scale": w2_input_scale,
#         "w2_weight_scale_2": w2_scale_2,
#     }
#     extra_kwargs = {
#         "routing_logits": routing_logits,
#     }
#     return payload, weights, extra_kwargs

# def _generate_ref_output(
#     config: MoEConfigAdapter,
#     payload: ExpertForwardPayload,
#     weights: Dict[str, torch.Tensor],
#     extra_kwargs: Dict[str, torch.Tensor],
# ) -> torch.Tensor:
#     # g1_alphas = weights["w13_input_scale"] * weights["w13_weight_scale_2"]
#     # g2_alphas = weights["w2_input_scale"] * weights["w2_weight_scale_2"]
#     # g1_scale_c = g1_alphas / weights["w2_input_scale"]
#     # hidden_states, hidden_states_scale = fp4_quantize(
#     #     payload.expert_x, 1 / weights["w13_input_scale"], is_sf_swizzled_layout=False)
#     # ref_output = trtllm_fp4_block_scale_moe(
#     #     extra_kwargs["routing_logits"],
#     #     None,  # routing_bias
#     #     hidden_states,
#     #     hidden_states_scale.view(torch.float8_e4m3fn),
#     #     weights[W.moe_w1],
#     #     weights[W.moe_s1],
#     #     None,  # w13_bias
#     #     None,  # gemm1_alpha
#     #     None,  # gemm1_beta
#     #     None,  # gemm1_clamp_limit
#     #     weights[W.moe_w2],
#     #     weights[W.moe_s2],
#     #     None,  # w2_bias
#     #     g1_scale_c,
#     #     g1_alphas,
#     #     g2_alphas,
#     #     config.expert_num,
#     #     config.moe_k,
#     #     None,  # n_group
#     #     None,  # topk_group
#     #     config.model_config.moe_inter_size,
#     #     0,  # local_expert_offset
#     #     config.expert_num,
#     #     None,  # routed_scaling_factor
#     #     None,  # tile_tokens_dim
#     #     RoutingMethodType.Renormalize.value,
#     #     True,  # do_finalize
#     #     device_support_pdl(payload.expert_x.device),
#     #     GatedActType.SwiGlu.value,  # gated_act_type
#     #     None,
#     # )[0]

#     # return ref_output

#     hidden_states = payload.expert_x
#     topk_ids = payload.expert_topk_ids
#     topk_weights = payload.expert_topk_weights

#     device = hidden_states.device
#     dtype = hidden_states.dtype

#     w13_global_scale = weights["w13_weight_scale_2"]
#     w13_float_list = []
#     for expert_id in range(config.expert_num):
#         expert_w13 = weights[W.moe_w1][expert_id]
#         expert_w13_scale = weights[W.moe_s1][expert_id]
#         expert_global_scale = w13_global_scale[expert_id]

#         expert_w13_float = e2m1_and_ufp8sf_scale_to_float(
#             expert_w13.view(torch.uint8),
#             expert_w13_scale.view(torch.uint8),
#             expert_global_scale,
#             sf_vec_size=NVFP4_BLOCK_SIZE,
#             ufp8_type=1,
#             is_sf_swizzled_layout=True,
#         )
#         w13_float_list.append(expert_w13_float)
#     w13_float = torch.stack(w13_float_list, dim=0)

#     w2_global_scale = weights["w2_weight_scale_2"]
#     w2_float_list = []
#     for expert_id in range(config.expert_num):
#         expert_w2 = weights[W.moe_w2][expert_id]
#         expert_w2_scale = weights[W.moe_s2][expert_id]
#         expert_global_scale = w2_global_scale[expert_id]

#         expert_w2_float = e2m1_and_ufp8sf_scale_to_float(
#             expert_w2.view(torch.uint8),
#             expert_w2_scale.view(torch.uint8),
#             expert_global_scale,
#             sf_vec_size=NVFP4_BLOCK_SIZE,
#             ufp8_type=1,
#             is_sf_swizzled_layout=True,
#         )
#         w2_float_list.append(expert_w2_float)
#     w2_float = torch.stack(w2_float_list, dim=0)

#     w13_float = w13_float.to(device).to(dtype)
#     w2_float = w2_float.to(device).to(dtype)

#     ref_output = torch.zeros((SEQ_LEN, config.hidden_size), dtype=dtype, device=device)

#     for token_idx in range(SEQ_LEN):
#         token_hidden = hidden_states[token_idx:token_idx+1]
#         token_output = torch.zeros((1, config.hidden_size), dtype=dtype, device=device)

#         for k in range(config.moe_k):
#             expert_id = topk_ids[token_idx, k].item()
#             expert_weight = topk_weights[token_idx, k]
#             w13_expert = w13_float[expert_id]
#             w2_expert = w2_float[expert_id]
#             workspace1 = torch.matmul(token_hidden, w13_expert.transpose(0, 1))
#             N = workspace1.shape[-1]
#             gate = workspace1[..., N // 2:].to(torch.float32)
#             value = workspace1[..., :N // 2].to(torch.float32)
#             gate = gate * torch.sigmoid(gate)
#             workspace2 = (gate * value).to(dtype)
#             expert_output = torch.matmul(workspace2, w2_expert.transpose(0, 1))
#             token_output += expert_output / expert_weight
#         ref_output[token_idx] = token_output[0]

#     return ref_output

# def test_trtllm_fp4_executor():
#     torch.manual_seed(42)
#     torch.cuda.manual_seed(42)
#     random.seed(42)

#     config = _generate_config()
#     payload, weights, extra_kwargs = _generate_payload_and_weights(config)
#     ref_output = _generate_ref_output(config, payload, weights, extra_kwargs)

#     executor = TrtllmFp4Executor(config, weights, FusedMoEQuantConfig())

#     output = executor.execute(payload, "silu", None, None, False, None)

#     print(output)
#     print(ref_output)
#     mask = torch.isclose(output, ref_output, rtol=1e-3, atol=1e-3)
#     mismatch_pct = (~mask).float().mean().item() * 100
#     assert mismatch_pct < 6, f"Mismatch percentage is {mismatch_pct:.2f}"

# if __name__ == "__main__":
#     test_trtllm_fp4_executor()

