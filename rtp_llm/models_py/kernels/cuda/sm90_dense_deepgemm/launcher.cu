// SPDX-License-Identifier: MIT
// Host-only adaptation of SGLang DeepGEMM's SM90 1D2D launcher. Arithmetic is
// provided unchanged by vendor/sm90_fp8_gemm_1d2d.cuh; see PROVENANCE.md.
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "vendor/sm90_fp8_gemm_1d2d.cuh"

#ifndef RTP_DG_N
#error "RTP_DG_N must specialize the output columns"
#endif
#ifndef RTP_DG_K
#error "RTP_DG_K must specialize the reduction dimension"
#endif
#ifndef RTP_DG_SMS
#error "RTP_DG_SMS must match the guarded device"
#endif

namespace {

static_assert(RTP_DG_N > 0 && RTP_DG_N % 32 == 0);
static_assert(RTP_DG_K >= 128 && RTP_DG_K <= 16384 && RTP_DG_K % 128 == 0);

// Mirrors upstream make_tma_2d_desc: TMA swizzle constrains the inner box to
// one swizzle atom. A/B are K-major, output N-major and A scales M-major.
CUtensorMap tensor_map(const at::Tensor& tensor, CUtensorMapDataType dtype,
                      int inner, int outer, int box_inner, int box_outer,
                      int outer_stride, int swizzle) {
    TORCH_CHECK(tensor.data_ptr() != nullptr &&
                    reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
                "TMA requires a 16-byte aligned pointer");
    const int item_size = tensor.element_size();
    TORCH_CHECK((outer_stride * item_size) % 16 == 0,
                "TMA requires a 16-byte aligned outer stride");
    if (swizzle != 0) box_inner = swizzle / item_size;
    const cuuint64_t dims[] = {static_cast<cuuint64_t>(inner),
                              static_cast<cuuint64_t>(outer)};
    const cuuint64_t strides[] = {static_cast<cuuint64_t>(outer_stride * item_size)};
    const cuuint32_t box[] = {static_cast<cuuint32_t>(box_inner),
                             static_cast<cuuint32_t>(box_outer)};
    const cuuint32_t elem_strides[] = {1, 1};
    CUtensorMap result;
    auto status = cuTensorMapEncodeTiled(
        &result, dtype, 2, tensor.data_ptr(), dims, strides, box, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(status == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", int(status));
    return result;
}

template <int BlockM, int BlockN, int Stages>
void launch(const at::Tensor& a, const at::Tensor& a_scale,
            const at::Tensor& b, const at::Tensor& b_scale,
            const at::Tensor& out) {
    constexpr int k = RTP_DG_K, n = RTP_DG_N;
    constexpr int math_threads = BlockM <= 64 ? 128 : 256;
    // These quantities mirror the vendored kernel's shared-memory layout.
    constexpr int smem_d = ((BlockM * BlockN * 2 + 1023) / 1024) * 1024;
    constexpr int smem_ab_stage = (BlockM + BlockN) * 128;
    constexpr int smem_as_stage = ((BlockM * 4 * 4 + 127) / 128) * 128;
    constexpr int smem_bs = ((k / 32 * (BlockN / 32) * 4 + 7) / 8) * 8;
    constexpr int smem = smem_d + Stages * (smem_ab_stage + smem_as_stage) +
                         smem_bs + Stages * 8 * 2;
    static_assert(smem <= 232448, "SM90 opt-in shared-memory limit exceeded");
    static_assert(BlockN % 64 == 0, "Selected D layout uses 128-byte swizzling");
    auto kernel = &deep_gemm::sm90_fp8_gemm_1d2d_impl<
        cute::UMMA::Major::K, 0, n, k, 1,
        BlockM, BlockN, 128, 128, 128, 128, Stages,
        128, math_threads, 1, false, RTP_DG_SMS,
        deep_gemm::GemmType::Normal, cutlass::bfloat16_t,
        deep_gemm::epilogue::transform::EpilogueIdentity, 32>;
    // Each extension is keyed by physical device index and N/K, so this setup
    // never crosses devices and is completed by warmup before graph capture.
    static const bool configured = [&] {
        cudaStreamCaptureStatus capture_status;
        C10_CUDA_CHECK(cudaStreamIsCapturing(
            at::cuda::getCurrentCUDAStream(a.get_device()), &capture_status));
        TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
                    "Warm up each SM90 group32 tile config before CUDA graph capture");
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        return true;
    }();
    (void)configured;

    const int m = a.size(0), padded_m = ((m + 3) / 4) * 4;
    auto map_a = tensor_map(a, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                            k, m, 128, BlockM, k, 128);
    auto map_b = tensor_map(b, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                            k, n, 128, BlockN, k, 128);
    auto map_d = tensor_map(out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                            n, m, BlockN, BlockM, n, 128);
    auto map_as = tensor_map(a_scale, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                             padded_m, k / 32, BlockM, 4, padded_m, 0);
    kernel<<<RTP_DG_SMS, 128 + math_threads, smem,
             at::cuda::getCurrentCUDAStream(a.get_device())>>>(
        static_cast<float*>(b_scale.data_ptr()), nullptr,
        m, n, k, map_a, map_b, map_d, map_as);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void sm90_g32_gemm_cuda(const at::Tensor& a, const at::Tensor& a_scale,
                       const at::Tensor& b, const at::Tensor& b_scale,
                       const at::Tensor& out, int config) {
    TORCH_CHECK(a.is_cuda() && a.dim() == 2 && a.size(0) > 0 &&
                    a.size(1) == RTP_DG_K && a.is_contiguous() &&
                    a.scalar_type() == at::kFloat8_e4m3fn,
                "Invalid A for compiled SM90 group32 kernel");
    TORCH_CHECK(b.device() == a.device() && b.dim() == 2 &&
                    b.size(0) == RTP_DG_N && b.size(1) == RTP_DG_K &&
                    b.is_contiguous() && b.scalar_type() == at::kFloat8_e4m3fn,
                "Invalid B for compiled SM90 group32 kernel");
    const int64_t m = a.size(0), padded_m = ((m + 3) / 4) * 4;
    TORCH_CHECK(m * RTP_DG_N < INT32_MAX && m * RTP_DG_K < INT32_MAX &&
                    int64_t(RTP_DG_N) * RTP_DG_K < INT32_MAX,
                "Candidate requires 32-bit-safe tensor indexing");
    TORCH_CHECK(a_scale.device() == a.device() && a_scale.dim() == 2 &&
                    a_scale.scalar_type() == at::kFloat &&
                    a_scale.size(0) == m && a_scale.size(1) == RTP_DG_K / 32 &&
                    a_scale.stride(0) == 1 && a_scale.stride(1) == padded_m,
                "A scales must have exact TMA-aligned M-major group32 layout");
    TORCH_CHECK(a_scale.storage().nbytes() - a_scale.storage_offset() * 4 >=
                    padded_m * (RTP_DG_K / 32) * 4,
                "A scales must physically allocate the final TMA padding rows");
    TORCH_CHECK(b_scale.device() == a.device() && b_scale.dim() == 2 &&
                    b_scale.scalar_type() == at::kFloat &&
                    b_scale.size(0) == RTP_DG_N / 32 &&
                    b_scale.size(1) == RTP_DG_K / 32 && b_scale.is_contiguous(),
                "B scales must be contiguous 32x32 block scales");
    TORCH_CHECK(out.device() == a.device() && out.dim() == 2 &&
                    out.scalar_type() == at::kBFloat16 && out.size(0) == m &&
                    out.size(1) == RTP_DG_N && out.is_contiguous(), "Invalid output");
    at::assert_no_overlap(out, a);
    at::assert_no_overlap(out, a_scale);
    at::assert_no_overlap(out, b);
    at::assert_no_overlap(out, b_scale);
    c10::cuda::CUDAGuard guard(a.device());
    const auto* properties = at::cuda::getDeviceProperties(a.get_device());
    TORCH_CHECK(properties->major == 9 && properties->minor == 0 &&
                    properties->multiProcessorCount == RTP_DG_SMS,
                "Compiled sidecar requires its original SM90 device");
    switch (config) {
        case 0: launch<64, 64, 8>(a, a_scale, b, b_scale, out); break;
        case 1: launch<64, 128, 6>(a, a_scale, b, b_scale, out); break;
        case 2: launch<128, 128, 5>(a, a_scale, b, b_scale, out); break;
        default: TORCH_CHECK(false, "Unknown SM90 group32 tile config");
    }
}
