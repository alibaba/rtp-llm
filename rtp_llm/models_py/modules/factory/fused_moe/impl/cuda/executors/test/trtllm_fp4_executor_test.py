from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict
import torch
from cuda.bindings import runtime
from torch.nn import functional as F
import os
import json
from pathlib import Path

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


def dump_trtllm_fp4_params(**kwargs):
    """Dump trtllm_fp4_block_scale_moe function parameters to JSON and save tensors as .pt files.
    
    If DUMP_FP4 environment variable is set, this function will:
    1. Save torch tensors as .pt files and record their absolute paths
    2. Convert other types to strings
    3. Save all information to a JSON file specified by DUMP_FP4
    
    Args:
        **kwargs: All parameters passed to trtllm_fp4_block_scale_moe
    """
    dump_file = os.getenv("DUMP_FP4")
    if not dump_file:
        return
    
    # Use pathlib for path operations
    dump_path = Path(dump_file).resolve()
    dump_dir = dump_path.parent
    
    # Create directory for JSON file (with proper error handling)
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Create a directory for tensor files (same directory as JSON file, with _tensors suffix)
    tensor_dir = dump_dir / f"{dump_path.stem}_tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    
    dumped_params = {}
    
    for param_name, param_value in kwargs.items():
        if isinstance(param_value, torch.Tensor):
            # Save tensor as .pt file
            tensor_filename = f"{param_name}.pt"
            tensor_path = tensor_dir / tensor_filename
            tensor_abs_path = tensor_path.resolve()
            
            try:
                torch.save(param_value, str(tensor_abs_path))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to save tensor {param_name} to {tensor_abs_path}: {e}"
                ) from e
            
            # Record tensor info in JSON
            dumped_params[param_name] = {
                "type": "torch.Tensor",
                "file_path": str(tensor_abs_path),
                "shape": list(param_value.shape),
                "dtype": str(param_value.dtype),
                "device": str(param_value.device),
            }
        elif param_value is None:
            dumped_params[param_name] = None
        else:
            # Convert other types to string
            try:
                # Try to convert to JSON-serializable format
                if isinstance(param_value, (int, float, bool, str)):
                    dumped_params[param_name] = param_value
                elif hasattr(param_value, "value"):  # Enum types
                    dumped_params[param_name] = str(param_value)
                else:
                    dumped_params[param_name] = str(param_value)
            except Exception:
                dumped_params[param_name] = str(param_value)
    
    # Save to JSON file
    try:
        with open(dump_path, "w") as f:
            json.dump(dumped_params, f, indent=2, ensure_ascii=False)
    except PermissionError as e:
        raise PermissionError(
            f"Cannot write to {dump_path}: {e}. "
            f"Please ensure you have write permissions or use a different path."
        ) from e


class QuantMode(IntEnum):
    """Supported quantization modes for MoE testing."""

    FP4_NVFP4_NVFP4 = 1

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
        # err, self.graph = runtime.cudaGraphCreate(0)
        # check_cuda(err)
        # err = runtime.cudaStreamBeginCapture(
        #     self.stream, runtime.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
        # )[0]
        check_cuda(err)

        try:
            # Capture computation on our stream
            with torch.cuda.stream(torch_stream):
                self.output_tensor = self._run_moe_computation(runtime_args)
            # err, self.graph = runtime.cudaStreamEndCapture(self.stream)
            # check_cuda(err)
            return self.output_tensor
            err, self.graph_exec = runtime.cudaGraphInstantiate(self.graph, 0)
            check_cuda(err)
            self.is_captured = True
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"CUDA graph capture failed: {e}") from e

    def launch(self, hidden_states_new):
        """Launch captured CUDA graph with new input."""
        return self.output_tensor
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

        # Prepare all parameters for the function call
        func_params = {
            "routing_logits": runtime_args["expert_logits"],
            "routing_bias": runtime_args["routing_bias"],
            "hidden_states": input_quantized["hidden_states"],
            "hidden_states_scale": input_quantized["hidden_states_scale"],
            "gemm1_weights": self.static_data["gemm1_weights_fp4_shuffled"],
            "gemm1_weights_scale": self.static_data["gemm1_scales_fp4_shuffled"],
            "gemm1_bias": None,
            "gemm1_alpha": None,
            "gemm1_beta": None,
            "gemm1_clamp_limit": None,
            "gemm2_weights": self.static_data["gemm2_weights_fp4_shuffled"],
            "gemm2_weights_scale": self.static_data["gemm2_scales_fp4_shuffled"],
            "gemm2_bias": None,
            "output1_scale_scalar": self.static_data["scale_c_fc1"],
            "output1_scale_gate_scalar": self.static_data["scale_gate_fc1"],
            "output2_scale_scalar": self.static_data["scale_c_fc2"],
            "num_experts": self.config["num_experts"],
            "top_k": self.config["top_k"],
            "n_group": self.config["n_groups"],
            "topk_group": self.config["top_k_groups"],
            "intermediate_size": self.config["intermediate_size"],
            "local_expert_offset": 0,
            "local_num_experts": self.config["num_experts"],
            "routed_scaling_factor": self.config["routed_scaling"],
            "tile_tokens_dim": None,
            "routing_method_type": self.config["routing_method_type"],
            "gated_act_type": self.config["gated_act_type"],
            "do_finalize": True,
        }
        
        # Dump parameters if DUMP_FP4 environment variable is set
        dump_trtllm_fp4_params(**func_params)
        
        # Call the function with all parameters
        output = trtllm_fp4_block_scale_moe(**func_params)
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
    def prepare_static_weights_for_kernel(self, args):
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
    def call_moe(self, args):
        """Call MoE with runtime input quantization + kernel execution (done at runtime)."""
        pass

    @abstractmethod
    def compute_reference(self, args):
        """Compute reference output using dequantized operations."""
        pass

    def compute_production(self, args):
        self.prepare_static_weights_for_kernel(args)

        topk_ids = args.permute_info["topKIndices"].to(torch.int32)
        args.topk_weights = args.expert_logits.view(args.num_tokens, args.num_experts)[
            torch.arange(args.num_tokens).unsqueeze(1), topk_ids
        ].to(torch.bfloat16)
        args.topk_ids = topk_ids
        args.w13_weight_scale_2 = 1.0 / args.gemm1_scales_global
        args.w2_weight_scale_2 = 1.0 / args.gemm2_scales_global
        args.w13_input_scale = 1.0 / args.hidden_states_scale_global
        args.w2_input_scale = 1.0 / args_dequant.c_global_sf

        return self.call_moe(args)

    @abstractmethod
    def get_tolerances(self):
        """Get accuracy tolerances for this quantization mode."""
        pass

    def __str__(self):
        return self.name

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
        self.sf_vec_size = 16

    def quantize_weights(self, gemm1_weights, gemm2_weights, hidden_states_sample):
        """Quantize weights to FP4 format and compute global scale factors."""
        num_experts = gemm1_weights.shape[0]
        hidden_states_scale_global = calculate_fp4_global_scale_factor(
            hidden_states_sample,
            False,
        )

        gemm1_weights_fp4_bytes, gemm1_scales_fp4_bytes, gemm1_scales_global = (
            quant_nvfp4_batches(gemm1_weights, num_experts, True)
        )
        gemm2_weights_fp4_bytes, gemm2_scales_fp4_bytes, gemm2_scales_global = (
            quant_nvfp4_batches(gemm2_weights, num_experts, True)
        )

        _, gemm1_scales_linear_fp4_bytes, _ = (
            quant_nvfp4_batches(gemm1_weights, num_experts, False)
        )
        _, gemm2_scales_linear_fp4_bytes, _ = (
            quant_nvfp4_batches(gemm2_weights, num_experts, False)
        )

        return {
            "hidden_states_scale_global": hidden_states_scale_global,
            "gemm1_weights": gemm1_weights_fp4_bytes,
            "gemm1_scales": gemm1_scales_fp4_bytes,
            "gemm1_scales_global": gemm1_scales_global,
            "gemm1_scales_linear": gemm1_scales_linear_fp4_bytes,
            "gemm1_weights_orig": gemm1_weights,
            "gemm2_weights": gemm2_weights_fp4_bytes,
            "gemm2_scales": gemm2_scales_fp4_bytes,
            "gemm2_scales_global": gemm2_scales_global,
            "gemm2_scales_linear": gemm2_scales_linear_fp4_bytes,
            "gemm2_weights_orig": gemm2_weights,
        }

    def quantize_inputs(
        self, hidden_states, hidden_states_scale_global, is_swizzling=True
    ):
        """Quantize hidden states to NvFP4 format using pre-computed global scale."""
        (
            hidden_states_fp4_bytes,
            hidden_states_scale_fp4_bytes,
        ) = fp4_quantize(hidden_states.cuda(), hidden_states_scale_global.cuda(), 16, False, is_swizzling)
        hidden_states_scale_fp4_bytes = hidden_states_scale_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(*hidden_states.shape[:-1], -1)

        return {
            "hidden_states": hidden_states_fp4_bytes,
            "hidden_states_scale": hidden_states_scale_fp4_bytes,
        }

    def prepare_static_weights_for_kernel(self, args):
        """Prepare quantized weights for kernel (done offline with weights)."""
        hidden_size = args.hidden_size
        intermediate_size = args.intermediate_size
        num_experts = args.num_experts

        epilogue_tile_m = 128  # FIXME: this depends on the kernel internals

        gemm1_scales_linear_fp4_bytes = args.gemm1_scales_linear
        gemm2_scales_linear_fp4_bytes = args.gemm2_scales_linear

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
            args.c_global_sf
            * (1.0 / args.gemm1_scales_global)
            * (1.0 / args.hidden_states_scale_global)
        )
        scale_gate_fc1 = (1.0 / args.gemm1_scales_global) * (
            1.0 / args.hidden_states_scale_global
        )
        scale_c_fc2 = (1.0 / args.c_global_sf) * (
            1.0 / args.gemm2_scales_global
        )

        args.gemm1_weights_fp4_shuffled = gemm1_weights_fp4_shuffled
        args.gemm1_scales_fp4_shuffled = gemm1_scales_fp4_shuffled
        args.gemm2_weights_fp4_shuffled = gemm2_weights_fp4_shuffled
        args.gemm2_scales_fp4_shuffled = gemm2_scales_fp4_shuffled
        args.scale_c_fc1 = scale_c_fc1
        args.scale_gate_fc1 = scale_gate_fc1
        args.scale_c_fc2 = scale_c_fc2

    def call_moe(self, args):
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
        sf_vec_size = 16
        ufp8_type_weights = 1

        args.hidden_states_dequant = e2m1_and_ufp8sf_scale_to_float(
            args.hidden_states.cpu(),
            args.hidden_states_scale.cpu().view(torch.uint8).reshape(-1),
            (1 / args.hidden_states_scale_global).cpu(),
            sf_vec_size,
            ufp8_type_weights,
            True,  # is_sf_swizzled_layout
        ).cuda()

        args.gemm1_weights_dequant = e2m1_and_ufp8_scale_batches(
            args.gemm1_weights,
            args.gemm1_scales,
            1 / args.gemm1_scales_global,
            sf_vec_size,
            ufp8_type_weights,
        ).cuda()

        args.gemm2_weights_dequant = e2m1_and_ufp8_scale_batches(
            args.gemm2_weights,
            args.gemm2_scales,
            1 / args.gemm2_scales_global,
            sf_vec_size,
            ufp8_type_weights,
        ).cuda()

        return run_moe_dequant(args)

    def get_tolerances(self):
        """Get FP4-specific accuracy tolerances."""
        return {"atol": 0.1, "rtol": 0.85, "percent": 0.925}

class FP4MoeExecutor(FP4Moe):
    def prepare_static_weights_for_kernel(self, args):
        use_ue8m0 = self.is_mxfp4
        gemm1_scales_linear_fp4_bytes = args.gemm1_scales
        gemm2_scales_linear_fp4_bytes = args.gemm2_scales
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
        args.gemm1_weights_fp4_shuffled = gemm1_weights_fp4
        args.gemm1_scales_fp4_shuffled = gemm1_scales_linear_fp4
        args.gemm2_weights_fp4_shuffled = gemm2_weights_fp4
        args.gemm2_scales_fp4_shuffled = gemm2_scales_linear_fp4

    def call_moe(self, args):
        model_config = ModelConfig()
        model_config.expert_num = args.num_experts
        model_config.hidden_size = args.hidden_size
        model_config.moe_inter_size = args.intermediate_size
        model_config.moe_k = args.top_k
        parallelism_config = ParallelismConfig()
        parallelism_config.dp_size = 1
        parallelism_config.tp_size = 1
        parallelism_config.ep_size = 1
        config = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
        )
        payload = ExpertForwardPayload(
            expert_x=args.hidden_states_orig,
            expert_x_origin_dtype=torch.bfloat16,
            expert_topk_ids=args.topk_ids,
            expert_topk_weights=args.topk_weights,
        )
        weights = {
            W.moe_w1: args.gemm1_weights_fp4_shuffled,
            W.moe_w2: args.gemm2_weights_fp4_shuffled,
            W.moe_s1: args.gemm1_scales_fp4_shuffled,
            W.moe_s2: args.gemm2_scales_fp4_shuffled,
            "w13_input_scale": args.w13_input_scale,
            "w13_weight_scale_2": args.w13_weight_scale_2,
            "w2_input_scale": args.w2_input_scale,
            "w2_weight_scale_2": args.w2_weight_scale_2,
        }

        executor = TrtllmFp4Executor(config, weights, FusedMoEQuantConfig())
        output = executor.execute(payload, "silu", None, None, False, None)
        return output.to(torch.float)

# ====================================================================================
# Fp4Executor Implementation (using TrtllmFp4Executor as backend)
# ====================================================================================

@dataclass(frozen=False, slots=True)
class moe_args:
    num_tokens: int = None
    num_experts: int = None
    hidden_size: int = None
    intermediate_size: int = None
    top_k: int = None
    padding: int = None
    permute_info: torch.Tensor = None
    use_routing_scales_on_input: bool = None
    gated_act_type: GatedActType = None
    topk_ids: torch.Tensor = None
    topk_weights: torch.Tensor = None
    hidden_states: torch.Tensor = None
    hidden_states_scale: torch.Tensor = None
    hidden_states_scale_global: torch.Tensor = None
    hidden_states_dequant: torch.Tensor = None
    hidden_states_orig: torch.Tensor = None
    expert_logits: torch.Tensor = None
    gemm1_weights: torch.Tensor = None
    gemm1_scales: torch.Tensor = None
    gemm1_scales_global: torch.Tensor = None
    gemm1_scales_linear: torch.Tensor = None
    gemm1_weights_dequant: torch.Tensor = None
    gemm1_weights_orig: torch.Tensor = None
    gemm1_weights_fp4_shuffled: torch.Tensor = None
    gemm1_scales_fp4_shuffled: torch.Tensor = None
    gemm2_weights: torch.Tensor = None
    gemm2_scales: torch.Tensor = None
    gemm2_scales_global: torch.Tensor = None
    gemm2_scales_linear: torch.Tensor = None
    gemm2_weights_dequant: torch.Tensor = None
    gemm2_weights_orig: torch.Tensor = None
    gemm2_weights_fp4_shuffled: torch.Tensor = None
    gemm2_scales_fp4_shuffled: torch.Tensor = None
    c_global_sf: torch.Tensor = None
    scale_c_fc1: torch.Tensor = None
    scale_gate_fc1: torch.Tensor = None
    scale_c_fc2: torch.Tensor = None


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


def quant_nvfp4_batches(a, num_experts, is_sf_swizzled_layout=True):
    """FP4 batch quantization function with centralized global scale factor calculation."""
    quant_a = []
    sfs = []
    global_sfs = []
    for i in range(num_experts):
        a_global_sf = calculate_fp4_global_scale_factor(a[i], False)
        a_fp4, a_sf = fp4_quantize(a[i].cuda(), a_global_sf.cuda(), 16, False, is_sf_swizzled_layout)
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


def run_moe_dequant(args):
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

    activation_output, c_global_sf = quant_dequant_fp4(
        activation_output.to(torch.bfloat16), False, True
    )
    activation_output = activation_output.to(torch.float)
    args.c_global_sf = c_global_sf

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


def test_moe(
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

    # Create arguments for reference computation
    moe_info = {
        "num_tokens": num_tokens,
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "top_k": top_k,
        "padding": padding,
        "expert_logits": scores,
        "permute_info": permute_info,
        "use_routing_scales_on_input": routing_method_type == RoutingMethodType.Llama4,
        "gated_act_type": gated_act_type,
        "hidden_states_orig": hidden_states,
    }
    args = moe_args(**moe_info, **weights_data, **inputs_data)

    output_dequant_reference, args_dequant = moe_impl.compute_reference(args)

    assert output_dequant_reference is not None, "Reference computation failed to produce output"

    # Compute actual output
    enable_autotune = routing_config.get("enable_autotune", False)

    output_dequant_actual = moe_impl.compute_production(args)

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
    test_moe(
        num_tokens=3072,
        hidden_size=1024,
        intermediate_size=768,
        # moe_impl=FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4),
        moe_impl=FP4MoeExecutor(quant_mode=QuantMode.FP4_NVFP4_NVFP4),
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

