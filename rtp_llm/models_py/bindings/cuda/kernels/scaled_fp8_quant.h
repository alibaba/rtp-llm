#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>
#include "rtp_llm/models_py/bindings/cuda/cuda_fp8_utils.h"

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

namespace rtp_llm {

#ifdef ENABLE_FP8

// Existing function
void per_tensor_quant_fp8(at::Tensor input, at::Tensor output_q, at::Tensor output_s, bool is_static);
void per_token_quant_fp8(at::Tensor input, at::Tensor output_q, at::Tensor output_s);

// SM90 QSA prefill path. Every call owns its scratch; frozen weights
// and independent column scales are packed by the model during preparation.
bool                              qsa_prefill_fp8_available();
std::pair<at::Tensor, at::Tensor> qsa_prefill_fp8_quant(const at::Tensor& input);
at::Tensor
qsa_prefill_fp8_project(const at::Tensor& input, const at::Tensor& packed_weight_t, const at::Tensor& column_scales);

// Capture tensor metadata once and validate it within the compute entry. The
// current Python tensor objects are passed again so TensorImpl swaps are visible.
class QsaFp8ProjectionGuard {
public:
    explicit QsaFp8ProjectionGuard(const std::array<at::Tensor, 4>& sources);
    std::optional<at::Tensor> project(const at::Tensor&                input,
                                      const at::Tensor&                packed_weight_t,
                                      const at::Tensor&                column_scales,
                                      const std::array<at::Tensor, 4>& current_sources) const;

private:
    struct SourceSnapshot {
        at::Tensor           tensor;
        const void*          data_ptr = nullptr;
        std::vector<int64_t> sizes;
        std::vector<int64_t> strides;
        int64_t              storage_offset = 0;
        at::ScalarType       dtype          = at::ScalarType::Undefined;
        c10::Device          device{c10::DeviceType::CPU};
        bool                 requires_grad = false;
        bool                 inference     = false;
        uint32_t             version       = 0;
    };
    std::array<SourceSnapshot, 4> sources_;
};

void invokeComputeFP8Quantize128(__nv_fp8_e4m3*       fp8_output,
                                 float*               quant_ptr,
                                 const __nv_bfloat16* weights,
                                 const int64_t        dim0,
                                 const int64_t        dim1,
                                 const int64_t        size,
                                 bool                 col_major_scale,
                                 cudaStream_t         stream);

#endif  // ENABLE_FP8
}  // namespace rtp_llm
