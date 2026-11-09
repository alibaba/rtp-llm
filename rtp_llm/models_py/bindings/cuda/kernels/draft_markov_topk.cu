#include "rtp_llm/models_py/bindings/cuda/kernels/deepselect_fp32.h"

#ifdef RTP_FLASHINFER_DRAFT_TOPK
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <flashinfer/fast_topk_clusters_exact.cuh>

namespace {
constexpr int kMaxDraftTopK = 4096;

template<typename Bias>
__global__ void correctedHalf(const float* logits, int64_t stride, const Bias* bias, __half* out, int pitch, int v) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (col < v) {
        float score = logits[row * stride + col] + static_cast<float>(bias[int64_t(row) * v + col]);
        // Keep masks at -inf, but never turn a finite logit into an infinity.
        if (isfinite(score))
            score = fminf(65504.f, fmaxf(-65504.f, score));
        out[int64_t(row) * pitch + col] = __float2half_rn(score);
    }
}

__global__ void scaleSelected(const __half* selected, const float* temperature, float* output, int k) {
    const int row = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < k) {
        const int i = row * k + col;
        output[i] = __half2float(selected[i]) / temperature[row];
    }
}
}  // namespace
#endif

namespace torch_ext {
std::tuple<torch::Tensor, torch::Tensor>
draft_markov_topk(const torch::Tensor& logits, const torch::Tensor& bias, const torch::Tensor& temperature, int64_t k) {
#ifdef RTP_FLASHINFER_DRAFT_TOPK
    if (k > 0 && k <= kMaxDraftTopK && logits.is_cuda() && logits.dim() == 2 && k < logits.size(1)) {
        const auto* p = at::cuda::getDeviceProperties(logits.get_device());
        if (p->major == 10 && (p->minor == 0 || p->minor == 3)) {
            TORCH_CHECK(logits.scalar_type() == at::kFloat && logits.stride(1) == 1
                            && logits.size(0) > 0 && logits.size(0) <= 65535
                            && logits.size(1) <= INT_MAX / 4 && bias.sizes() == logits.sizes() && bias.is_contiguous()
                            && bias.device() == logits.device() && temperature.device() == logits.device()
                            && temperature.scalar_type() == at::kFloat && temperature.dim() == 1
                            && temperature.size(0) == logits.size(0) && temperature.is_contiguous(),
                        "invalid DSpARK draft TopK inputs");
            const c10::cuda::CUDAGuard guard(logits.device());
            const int                  b = logits.size(0), v = logits.size(1), pitch = (v + 3) / 4 * 4;
            auto                       stream = at::cuda::getCurrentCUDAStream(logits.get_device());
            auto                       scores = torch::empty({b, pitch}, logits.options().dtype(at::kHalf));
            const dim3                 grid((v + 255) / 256, b);
            auto                       launch = [&](auto* input_bias) {
                correctedHalf<<<grid, 256, 0, stream>>>(logits.data_ptr<float>(),
                                                        logits.stride(0),
                                                        input_bias,
                                                        reinterpret_cast<__half*>(scores.data_ptr<at::Half>()),
                                                        pitch,
                                                        v);
            };
            if (bias.scalar_type() == at::kBFloat16)
                launch(bias.data_ptr<at::BFloat16>());
            else if (bias.scalar_type() == at::kHalf)
                launch(bias.data_ptr<at::Half>());
            else
                launch(bias.data_ptr<float>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            auto ids    = torch::empty({b, k}, logits.options().dtype(at::kLong));
            auto values = torch::empty({b, k}, scores.options());
            // FlashInfer topk.py dispatch policy. Budget for a complete 4096-element
            // vectorized CTA stripe per cluster, including the last partial round.
            const int clusters = v < 8192 ? 1 : b <= 32 ? 8 : b < 128 ? 4 : b < 256 ? 2 : 1;
            const int ovf      = ((v - 1) / (4096 * clusters) + 1) * 4096;
            const int cached   = (int(p->sharedMemPerBlockOptin) / 2 - (k + 5 + 3 * 256 + 8) * 4 - 1024) / 16;
            auto      work     = torch::empty({b, int64_t(4) * clusters * ovf}, logits.options().dtype(at::kInt));
            flashinfer::sampling::launch_fast_topk_clusters_exact<__half, int64_t>(
                reinterpret_cast<const __half*>(scores.data_ptr<at::Half>()),
                ids.data_ptr<int64_t>(),
                reinterpret_cast<__half*>(values.data_ptr<at::Half>()),
                v,
                nullptr,
                work.data_ptr<int>(),
                ovf,
                b,
                pitch,
                k,
                cached,
                clusters,
                false,
                k,
                stream);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            // Positive row temperatures preserve ordering. Scale after selection so
            // small temperatures cannot overflow FP16. All q and verification math stays FP32.
            auto output = torch::empty({b, k}, logits.options());
            scaleSelected<<<dim3((k + 255) / 256, b), 256, 0, stream>>>(
                reinterpret_cast<const __half*>(values.data_ptr<at::Half>()),
                temperature.data_ptr<float>(),
                output.data_ptr<float>(),
                k);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return {output, ids};
        }
    }
#endif
    return deepselect_fp32_topk(markov_scaled_logits(logits, bias, temperature), k);
}
}  // namespace torch_ext
