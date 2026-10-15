#include "rtp_llm/models_py/bindings/cuda/W4A16GemmOp.h"

#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <limits>

extern "C" int w4a16_gemm_splitk_sm120(const void*, const void*, const void*, void*, int, int, int, int, void*);
extern "C" int w4a16_weight_transform(const void*, int, int, void*, void*, void*, void*);
at::Tensor     fast_hadamard_transform(at::Tensor& input, float scale);

namespace rtp_llm {
namespace {
void checkTensor(const at::Tensor& tensor, at::ScalarType dtype, const at::Device& device, int alignment) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == device, "W4A16 tensors must share a CUDA device");
    TORCH_CHECK(tensor.scalar_type() == dtype && tensor.is_contiguous(), "Invalid W4A16 dtype or strides");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0, "Unaligned W4A16 tensor");
}

void checkDimensions(int64_t output_size, int64_t input_size) {
    TORCH_CHECK(output_size > 0 && input_size > 0 && output_size % 16 == 0 && input_size % 32 == 0,
                "W4A16 requires N % 16 == 0 and K % 32 == 0");
    TORCH_CHECK(output_size <= std::numeric_limits<int>::max() && input_size <= std::numeric_limits<int>::max()
                    && output_size <= std::numeric_limits<int>::max() / input_size,
                "W4A16 dimensions exceed kernel indexing limits");
}

void checkDevice(const at::Device& device) {
    TORCH_CHECK(device.is_cuda(), "W4A16 requires CUDA");
    cudaDeviceProp properties;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device.index()));
    TORCH_CHECK(properties.major == 12 && properties.minor == 0, "W4A16 requires SM120");
}

void checkStorage(const at::Tensor& packed,
                  const at::Tensor& scales,
                  int64_t           output_size,
                  int64_t           input_size,
                  const at::Device& device) {
    checkTensor(packed, at::kInt, device, 16);
    checkTensor(scales, at::kByte, device, 4);
    TORCH_CHECK(packed.numel() == output_size * input_size / 8 && scales.numel() == output_size * input_size / 8,
                "Invalid W4A16 packed weight or scale size");
}
}  // namespace

at::Tensor w4a16Sm120Hadamard(at::Tensor input, double scale) {
    checkDevice(input.device());
    TORCH_CHECK(input.dim() == 2 && input.size(1) == 128, "W4A16 Hadamard requires [rows,128]");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kFloat,
                "W4A16 Hadamard requires BF16 or FP32");
    if (input.size(0) == 0) {
        return at::empty_like(input);
    }
    checkTensor(input, input.scalar_type(), input.device(), 16);
    return ::fast_hadamard_transform(input, static_cast<float>(scale));
}

void w4a16Sm120Transform(const at::Tensor& weight, at::Tensor packed, at::Tensor scales, at::Tensor scratch) {
    checkDevice(weight.device());
    const c10::cuda::CUDAGuard guard(weight.device());
    TORCH_CHECK(weight.dim() == 2, "W4A16 weight must be [N,K]");
    const int64_t output_size = weight.size(0), input_size = weight.size(1);
    checkDimensions(output_size, input_size);
    checkTensor(weight, at::kBFloat16, weight.device(), 16);
    checkStorage(packed, scales, output_size, input_size, weight.device());
    checkTensor(scratch, at::kByte, weight.device(), 1);
    TORCH_CHECK(scratch.numel() >= weight.numel(), "W4A16 scratch is too small");
    for (const auto& destination : {packed, scales, scratch}) {
        at::assert_no_overlap(destination, weight);
    }
    at::assert_no_overlap(packed, scales);
    at::assert_no_overlap(packed, scratch);
    at::assert_no_overlap(scales, scratch);
    auto stream = c10::cuda::getCurrentCUDAStream(weight.get_device()).stream();
    TORCH_CHECK(w4a16_weight_transform(weight.data_ptr(),
                                       output_size,
                                       input_size,
                                       packed.data_ptr(),
                                       scales.data_ptr(),
                                       scratch.data_ptr(),
                                       stream)
                    == 0,
                "W4A16 weight transform failed");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void w4a16Sm120Gemm(const at::Tensor& input,
                    const at::Tensor& packed,
                    const at::Tensor& scales,
                    at::Tensor        output,
                    int64_t           output_size,
                    int64_t           input_size,
                    int64_t           split_k) {
    checkDevice(input.device());
    const c10::cuda::CUDAGuard guard(input.device());
    checkDimensions(output_size, input_size);
    TORCH_CHECK(input.dim() == 2 && input.size(1) == input_size, "W4A16 input must be [M,K]");
    const int64_t rows = input.size(0);
    TORCH_CHECK(rows < 64 && output.dim() == 2 && output.size(0) == rows && output.size(1) == output_size,
                "W4A16 output must be [M,N], with M < 64");
    TORCH_CHECK(split_k >= 0 && split_k <= 16, "W4A16 split_k must be between 0 and 16");
    const int tile_n = rows <= 8 ? 64 : (rows <= 16 ? 128 : 256);
    TORCH_CHECK(output_size % tile_n == 0 && input_size % (rows <= 8 ? 64 : 32) == 0,
                "Unmodified W4A16 kernel cannot safely load partial N/K tiles");
    checkTensor(input, at::kBFloat16, input.device(), 16);
    checkTensor(output, at::kBFloat16, input.device(), 4);
    checkStorage(packed, scales, output_size, input_size, input.device());
    for (const auto& source : {input, packed, scales}) {
        at::assert_no_overlap(output, source);
    }
    if (rows == 0) {
        return;
    }
    auto stream = c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
    TORCH_CHECK(w4a16_gemm_splitk_sm120(input.data_ptr(),
                                        packed.data_ptr(),
                                        scales.data_ptr(),
                                        output.data_ptr(),
                                        rows,
                                        output_size,
                                        input_size,
                                        split_k,
                                        stream)
                    == 0,
                "W4A16 GEMM failed");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace rtp_llm
