#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using DType = bf16_t;
constexpr int64_t kRtpRopeDim = 64;
constexpr uint32_t kRtpRopeBlockSize = 128;
constexpr uint32_t kRtpRopeWarps = kRtpRopeBlockSize / device::kWarpThreads;

struct RtpExactRopeParams {
  void* __restrict__ q;
  const float* __restrict__ freqs;
  int64_t q_stride_row;
  uint32_t rows;
  uint32_t freq_stride_n;
};

template <bool kUsePDL, bool kInverse>
__global__ __launch_bounds__(kRtpRopeBlockSize, 16) void rtp_exact_rope_kernel(
    const __grid_constant__ RtpExactRopeParams params) {
  using namespace device;
  using DType2 = packed_t<DType>;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto row_id = blockIdx.x * kRtpRopeWarps + warp_id;
  if (row_id >= params.rows) return;

  const auto freq_id = row_id / params.freq_stride_n;
  auto* input = static_cast<DType2*>(
      pointer::offset(params.q, static_cast<int64_t>(row_id) * params.q_stride_row));
  const auto* freq = reinterpret_cast<const fp32x2_t*>(
      params.freqs + static_cast<int64_t>(freq_id) * kRtpRopeDim);

  const auto [f_real, f_imag] = freq[lane_id];
  PDLWaitPrimary<kUsePDL>();
  const auto [x_real, x_imag] = cast<fp32x2_t>(input[lane_id]);

  fp32x2_t output;
  if constexpr (kInverse) {
    output = {
        fmaf(x_real, f_real, x_imag * f_imag),
        fmaf(x_imag, f_real, -x_real * f_imag),
    };
  } else {
    // Preserve RTP Triton's final FMA association exactly. Reordering either
    // expression changes BF16 ties for cancellation-heavy inputs.
    output = {
        fmaf(x_real, f_real, -x_imag * f_imag),
        fmaf(x_imag, f_real, x_real * f_imag),
    };
  }
  input[lane_id] = cast<DType2>(output);
  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct RtpExactRopeKernel {
  static constexpr auto kernel_fwd = rtp_exact_rope_kernel<kUsePDL, false>;
  static constexpr auto kernel_inv = rtp_exact_rope_kernel<kUsePDL, true>;

  static void forward(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView freqs,
      bool inverse) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto freq_rows = SymbolicSize{"freq_rows"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({rows, kRtpRopeDim})
        .with_strides({-1, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(q);
    TensorMatcher({freq_rows, kRtpRopeDim})
        .with_strides({kRtpRopeDim, 1})
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs);

    const auto row_count = static_cast<uint32_t>(rows.unwrap());
    const auto freq_count = static_cast<uint32_t>(freq_rows.unwrap());
    RuntimeCheck(freq_count > 0 && row_count % freq_count == 0,
                 "RTP exact RoPE rows must be divisible by frequency rows");
    if (row_count == 0) return;

    const auto params = RtpExactRopeParams{
        .q = q.data_ptr(),
        .freqs = static_cast<const float*>(freqs.data_ptr()),
        .q_stride_row = q.stride(0) * static_cast<int64_t>(sizeof(DType)),
        .rows = row_count,
        .freq_stride_n = row_count / freq_count,
    };
    const auto blocks = div_ceil(row_count, kRtpRopeWarps);
    using KernelType = decltype(kernel_fwd);
    const KernelType kernel = inverse ? kernel_inv : kernel_fwd;
    LaunchKernel(blocks, kRtpRopeBlockSize, device_.unwrap())
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
