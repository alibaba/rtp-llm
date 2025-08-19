import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F

try:
    from rtp_llm.models_py.modules.fp8_kernel import sgl_per_token_group_quant_fp8
    FP8_AVAILABLE = True
    # FP8 quantization available
except ImportError as e:
    # FP8 quantization not available
    FP8_AVAILABLE = False

try:
    import deep_gemm
    from deep_gemm import fp8_gemm_nt
    DEEPGEMM_AVAILABLE = True
    
    # Setup CUTLASS include paths for JIT compilation
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Search for CUTLASS headers
    cutlass_paths = []
    parts = current_dir.split('/')
    for i in range(len(parts)):
        base_path = '/'.join(parts[:i+1])
        for subpath in ["deep_gemm/third-party/cutlass/include", 
                       "external/deep_gemm/third-party/cutlass/include"]:
            path = os.path.join(base_path, subpath)
            if os.path.exists(path):
                cutlass_paths.append(path)
    
    # Check runfiles directory
    if "runfiles" in current_dir:
        runfiles_root = current_dir.split("runfiles")[0] + "runfiles"
        for subdir in ["deep_gemm", "external/deep_gemm"]:
            path = os.path.join(runfiles_root, subdir, "third-party/cutlass/include")
            if os.path.exists(path):
                cutlass_paths.append(path)
    
    # Set environment variables if CUTLASS found
    if cutlass_paths:
        cutlass_path = cutlass_paths[0]
        for env_var in ["CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "CPATH"]:
            current_val = os.environ.get(env_var, "")
            os.environ[env_var] = f"{cutlass_path}:{current_val}" if current_val else cutlass_path
        
        nvcc_flags = os.environ.get("NVCC_PREPEND_FLAGS", "")
        os.environ["NVCC_PREPEND_FLAGS"] = f"-I{cutlass_path} {nvcc_flags}".strip()
        
except ImportError:
    DEEPGEMM_AVAILABLE = False


class Fp8Linear(nn.Module):
    """FP8 Linear layer with DeepGEMM quantized matrix multiplication."""
    
    # Debug options
    ENABLE_FP8_QUANTIZATION = True
    FORCE_USE_TORCH_MATMUL = False
    USE_MANUAL_QUANTIZATION = False
    
    def __init__(self, weight: torch.Tensor, weight_scales: torch.Tensor, 
                 bias: Optional[torch.Tensor] = None, config=None) -> None:
        super().__init__()
        self.hidden_size = weight.shape[0]  # k
        self.output_size = weight.shape[1]  # n
        self.weight = weight.reshape([weight.shape[1], weight.shape[0]])
        expected_weight_scales_shape = ((self.output_size + 127) // 128, (self.hidden_size + 127) // 128)
        self.weight_scales = weight_scales.reshape([weight_scales.shape[1], weight_scales.shape[0]])
        self.bias = bias
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        original_input = input.clone()
        
        # Get input dimensions
        input_m = input.shape[0]
        input_k = input.shape[1]
        output_n = self.output_size
        
        # Save original dtype for output conversion
        original_dtype = input.dtype
        
        # Convert to BF16 if needed
        if input.dtype != torch.bfloat16:
            input_bf16 = input.to(torch.bfloat16)
        else:
            input_bf16 = input
                        
        # Quantize input to FP8
        if FP8_AVAILABLE:
            
            # Calculate padding requirements
            alignment = self._get_padding_size(input_m)
            target_m = (input_m + alignment - 1) // alignment * alignment
            need_padding = target_m > input_m
                        
            # Prepare input for quantization
            if need_padding:
                input_for_quant = torch.zeros(target_m, input_k, dtype=torch.bfloat16, device=input.device)
                input_for_quant[:input_m, :] = input_bf16
            else:
                input_for_quant = input_bf16
            
            # Quantize using sgl_per_token_group_quant_fp8
            quantization_eps = 1e-4 
            use_column_major = need_padding
            
            input_fp8, input_scales = sgl_per_token_group_quant_fp8(
                input_for_quant, 
                group_size=128,
                eps=quantization_eps, 
                column_major_scales=use_column_major  
            )
            
            # Post-process scaling factors
            FP8_E4M3_MAX = 448.0
            min_scale_threshold = 1e-4 / FP8_E4M3_MAX  
            input_scales = torch.clamp(input_scales, min=min_scale_threshold)
            input_scales = input_scales.to(torch.float32)
            original_input_scales_shape = input_scales.shape
            
            # Create output tensor
            output_m = input_for_quant.shape[0]
            output = torch.zeros(output_m, output_n, dtype=torch.bfloat16, device=input.device)
            
            # Call DeepGEMM
            if DEEPGEMM_AVAILABLE:
                
                deepgemm_input_scales = input_scales
                input_fp8 = input_fp8.contiguous()
                deepgemm_input_scales = deepgemm_input_scales.contiguous()
                weight = self.weight.contiguous()
                weight_scales = self.weight_scales.contiguous()
                output = output.contiguous()
                try:
                    fp8_gemm_nt(
                        (input_fp8, deepgemm_input_scales),
                        (weight, weight_scales),
                        output, 
                        c=None,
                        disable_ue8m0_cast=True
                    )
                except Exception as e:
                    # DeepGEMM call failed, fallback to torch
                    print(f"Fp8Linear forward error type: {type(e)}")
                    import traceback
                    traceback.print_exc()
                    raise
                
            else:
                # DeepGEMM not available
                output = self._torch_fallback(input_fp8, input_scales)
                
        else:
            # FP8 not available
            output = self._fp16_fallback(input_for_quant)
             
        # Slice back to original size
        if need_padding:
            output = output[:input_m, :]
             
        # Handle bias and type conversion
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        
        # Convert back to original dtype
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
             
        final_output = output
        
        return final_output
        
    def _get_padding_size(self, m):
        """Calculate padding size based on DeepGEMM requirements."""
        if self._gemm_swap_ab_heuristic(m):
            if m < 16:
                return 16
            else:
                return 8
        else:
            return 64
            
    def _gemm_swap_ab_heuristic(self, m):
        # For DeepGemmType::Normal
        return m < 64
            
    def _torch_fallback(self, input_fp8, input_scales):
        """Fallback implementation using torch."""
        expanded_input_scales = self._expand_input_scales(input_scales, input_fp8.shape)
        input_fp32 = input_fp8.to(torch.float32) * expanded_input_scales
        
        # Dequantize weights
        weight_fp32 = self.weight.to(torch.float32) * self._expand_weight_scales()
        
        # Perform matrix multiplication: [m, k] @ [n, k].T = [m, n]
        output = torch.matmul(input_fp32, weight_fp32.T)
        return output.to(torch.bfloat16)
        
    def _fp16_fallback(self, input_tensor):
        """Pure FP16 fallback logic."""
        weight_fp32 = self.weight.to(torch.float32) * self._expand_weight_scales()
        weight_fp16 = weight_fp32.to(torch.float16)
        
        input_fp16 = input_tensor.to(torch.float16)
        output = torch.matmul(input_fp16, weight_fp16.T)
        return output.to(torch.bfloat16)
        
    def _expand_input_scales(self, input_scales, target_shape):
        """Expand input scales to target shape."""
        # input_scales: [m, k/128] - always row-major
        # target_shape: [m, k]
        m, k = target_shape
        
        # Validate scaling factor shape
        expected_scales_shape = (m, (k + 127) // 128)
        if input_scales.shape != expected_scales_shape:
            raise ValueError(f"Input scales shape mismatch! Expected {expected_scales_shape}, got {input_scales.shape}")
        
        # Expand scaling factors
        expanded = torch.zeros(target_shape, dtype=input_scales.dtype, device=input_scales.device)
        for i in range(input_scales.shape[0]):  # m tokens
            for j in range(input_scales.shape[1]):  # k/128 groups
                k_start = j * 128
                k_end = min((j + 1) * 128, k)
                expanded[i, k_start:k_end] = input_scales[i, j]
                
        return expanded        
        
    def _expand_weight_scales(self):
        """Expand weight scales to weight tensor shape."""
        expanded = torch.zeros_like(self.weight, dtype=torch.float32)
        for i in range(self.weight_scales.shape[0]):  # output_size blocks (60)
            for j in range(self.weight_scales.shape[1]):  # hidden_size blocks (20)
                h_start = i * 128  # output_size dimension
                h_end = min((i + 1) * 128, self.weight.shape[0])
                w_start = j * 128  # hidden_size dimension
                w_end = min((j + 1) * 128, self.weight.shape[1])
                
                expanded[h_start:h_end, w_start:w_end] = self.weight_scales[i, j]
                
        return expanded 
    
