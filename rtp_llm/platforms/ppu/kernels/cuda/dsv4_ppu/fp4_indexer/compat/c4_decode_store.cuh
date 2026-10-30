// Decode-only C4 state update and FP4 cache epilogue. One warp owns a request.
#include "c4_decode_v2.cuh"
#include "fused_norm_rope_v2.cuh"

namespace {
struct C4DecodeStoreParams {
  Compress4DecodeParams compress;
  FusedNormRopeStoreParams store;
};

template <bool kUsePDL>
__global__ __launch_bounds__(128, 4) void c4_decode_store(const __grid_constant__ C4DecodeStoreParams params) {
  using Trait = C4Trait<128>;
  const uint32_t row = (blockIdx.x * blockDim.x + threadIdx.x) / device::kWarpThreads;
  if (row >= params.compress.batch_size) return;
  const auto plan = params.compress.plan_d[row];
  if (plan.write_loc < 0) return;
  auto* state = static_cast<float*>(params.compress.kv_buffer);
  const auto* input = static_cast<const float*>(params.compress.kv_input) + row * Trait::kElementSize;
  device::PDLWaitPrimary<kUsePDL>();
  c4_write_decode<Trait>(state + plan.write_loc * Trait::kElementSize, input);
  if (plan.seq_len % 4 == 0) {
    // Compression's original JIT uses fast exp/div. Keep those intrinsics
    // explicit so the norm/rotary epilogue retains its original math flags.
    const auto compressed = c4_reduce<Trait, float, float, true>(
        state + plan.read_page_0 * Trait::kPageElementSize,
        state + plan.read_page_1 * Trait::kPageElementSize, input,
        static_cast<const float*>(params.compress.score_bias), plan.seq_len > 4, 8);
    norm_rope_store_indexer_fp4<float, ForwardMode::CompressDecode, 6, kUsePDL, true>(
        params.store, row, compressed);
  }
}

template <bool kUsePDL>
struct FusedC4DecodeStore {
  static void forward(
      tvm::ffi::TensorView state, tvm::ffi::TensorView input, tvm::ffi::TensorView ape,
      tvm::ffi::TensorView plan, tvm::ffi::TensorView weight, float eps,
      tvm::ffi::TensorView freqs, tvm::ffi::TensorView slots, tvm::ffi::TensorView cache) {
    using namespace host;
    auto rows = SymbolicSize{"tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({-1, 4, 512}).with_dtype<float>().with_device(device).verify(state);
    TensorMatcher({rows, 512}).with_dtype<float>().with_device(device).verify(input);
    TensorMatcher({8, 128}).with_dtype<float>().with_device(device).verify(ape);
    const auto plan_ptr = compress::verify_plan_d(plan, rows, device);
    TensorMatcher({128}).with_dtype<float>().with_device(device).verify(weight);
    TensorMatcher({-1, 64}).with_dtype<float>().with_device(device).verify(freqs);
    TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(slots);
    TensorMatcher({-1, 4352}).with_dtype<uint8_t>().with_device(device).verify(cache);
    const auto n = static_cast<uint32_t>(rows.unwrap());
    if (!n) return;
    const C4DecodeStoreParams params{
        .compress = {.kv_buffer = state.data_ptr(), .kv_input = input.data_ptr(), .kv_output = nullptr,
                     .score_bias = ape.data_ptr(), .plan_d = plan_ptr, .batch_size = n},
        .store = {.input = nullptr, .handle = plan_ptr, .weight = weight.data_ptr(),
                  .freqs_cis = static_cast<const float*>(freqs.data_ptr()),
                  .out_loc = static_cast<const int32_t*>(slots.data_ptr()),
                  .kvcache = static_cast<uint8_t*>(cache.data_ptr()), .eps = eps, .compress_ratio = 4, .num_tokens = n},
    };
    LaunchKernel(div_ceil(n, 4u), 128, device.unwrap()).enable_pdl(kUsePDL)(c4_decode_store<kUsePDL>, params);
  }
};
}  // namespace
