// Keep each request's speculative C4 transitions ordered inside one warp.
#include "c4_decode_store.cuh"

namespace {
struct C4DecodeQueriesParams {
  C4DecodeStoreParams base;
  const void* positions;
  const void* requests;
  const void* state_slots;
  const void* kv_slots;
  const void* block_table;
  int64_t block_stride;
  int32_t max_blocks;
  int32_t state_entries;
  int32_t state_tokens;
};

template <bool kInt64>
SGL_DEVICE int32_t c4_metadata(const void* ptr, int64_t index) {
  if constexpr (kInt64) return static_cast<int32_t>(static_cast<const int64_t*>(ptr)[index]);
  return static_cast<const int32_t*>(ptr)[index];
}

template <bool kUsePDL, int kQueries, int kInt64Mask>
__global__ __launch_bounds__(128, 4) void c4_decode_queries_store(
    const __grid_constant__ C4DecodeQueriesParams params) {
  using Trait = C4Trait<128>;
  const auto& base = params.base;
  const uint32_t row = (blockIdx.x * blockDim.x + threadIdx.x) / device::kWarpThreads;
  if (row >= base.compress.batch_size) return;
  auto* state = static_cast<float*>(base.compress.kv_buffer);
  device::PDLWaitPrimary<kUsePDL>();
  // Do not unroll the compressor/epilogue into multiple copies of this kernel.
#pragma unroll 1
  for (int query = 0; query < kQueries; ++query) {
    const auto token = row * kQueries + query;
    const auto pos = c4_metadata<(kInt64Mask & 1) != 0>(params.positions, token);
    const auto write = c4_metadata<(kInt64Mask & 4) != 0>(params.state_slots, token);
    if (pos < 0 || write < 0) continue;
    const auto seq = pos + 1;
    const auto* input = static_cast<const float*>(base.compress.kv_input) +
                        (query * base.compress.batch_size + row) * Trait::kElementSize;
    c4_write_decode<Trait>(state + int64_t{write} * Trait::kElementSize, input);
    // The old path has a kernel boundary between queries. Retain visibility
    // for the next query and the cross-lane epilogue within this owning warp.
    __syncwarp();
    if (seq % 4 == 0) {
      const auto req = c4_metadata<(kInt64Mask & 2) != 0>(params.requests, token);
      int32_t pages[2] = {0, 0};
#pragma unroll
      for (int part = 0; part < 2; ++part) {
        const auto window = seq - 8 + part * 4;
        if (window >= 0) {
          const auto block = (window / params.state_tokens) % params.max_blocks;
          const auto physical = c4_metadata<(kInt64Mask & 16) != 0>(
              params.block_table, int64_t{req} * params.block_stride + block);
          pages[part] = physical * (params.state_entries / 4) + (window % params.state_entries) / 4;
        }
      }
      const auto compressed = c4_reduce<Trait, float, float, true>(
          state + int64_t{pages[0]} * Trait::kPageElementSize,
          state + int64_t{pages[1]} * Trait::kPageElementSize, input,
          static_cast<const float*>(base.compress.score_bias), seq > 4, 8);
      // Reuse the unchanged epilogue with a register-resident one-row plan.
      const PlanD plan{static_cast<uint32_t>(seq), write, pages[0], pages[1]};
      const auto slot = c4_metadata<(kInt64Mask & 8) != 0>(params.kv_slots, token);
      auto store = base.store;
      store.handle = &plan;
      store.out_loc = &slot;
      store.num_tokens = 1;
      norm_rope_store_indexer_fp4<float, ForwardMode::CompressDecode, 6, kUsePDL, true>(
          store, 0, compressed);
    }
    __syncwarp();
  }
}

template <bool kInt64>
void c4_verify_metadata(tvm::ffi::TensorView tensor, host::SymbolicSize& rows, host::SymbolicDevice& device) {
  if constexpr (kInt64) {
    host::TensorMatcher({rows}).with_dtype<int64_t>().with_device(device).verify(tensor);
  } else {
    host::TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(tensor);
  }
}

template <bool kUsePDL, int kQueries, int kInt64Mask>
struct FusedC4DecodeQueriesStore {
  static void forward(
      tvm::ffi::TensorView state, tvm::ffi::TensorView input, tvm::ffi::TensorView ape,
      tvm::ffi::TensorView weight, float eps, tvm::ffi::TensorView freqs,
      tvm::ffi::TensorView positions, tvm::ffi::TensorView requests,
      tvm::ffi::TensorView state_slots, tvm::ffi::TensorView kv_slots,
      tvm::ffi::TensorView block_table, int32_t state_entries, int32_t state_tokens,
      tvm::ffi::TensorView cache) {
    using namespace host;
    static_assert(kQueries >= 2 && kQueries <= 4);
    auto batch = SymbolicSize{"batch"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({-1, 4, 512}).with_dtype<float>().with_device(device).verify(state);
    TensorMatcher({kQueries, batch, 512}).with_dtype<float>().with_device(device).verify(input);
    TensorMatcher({8, 128}).with_dtype<float>().with_device(device).verify(ape);
    TensorMatcher({128}).with_dtype<float>().with_device(device).verify(weight);
    TensorMatcher({-1, 64}).with_dtype<float>().with_device(device).verify(freqs);
    TensorMatcher({-1, 4352}).with_dtype<uint8_t>().with_device(device).verify(cache);
    auto rows = SymbolicSize{"tokens"};
    rows.set_value(batch.unwrap() * kQueries);
    c4_verify_metadata<(kInt64Mask & 1) != 0>(positions, rows, device);
    c4_verify_metadata<(kInt64Mask & 2) != 0>(requests, rows, device);
    c4_verify_metadata<(kInt64Mask & 4) != 0>(state_slots, rows, device);
    c4_verify_metadata<(kInt64Mask & 8) != 0>(kv_slots, rows, device);
    auto stride = SymbolicSize{"bt_stride"};
    if constexpr (kInt64Mask & 16) {
      TensorMatcher({-1, -1}).with_strides({stride, 1}).with_device(device).with_dtype<int64_t>().verify(block_table);
    } else {
      TensorMatcher({-1, -1}).with_strides({stride, 1}).with_device(device).with_dtype<int32_t>().verify(block_table);
    }
    const auto n = static_cast<uint32_t>(batch.unwrap());
    if (!n) return;
    const C4DecodeQueriesParams params{
        .base = {.compress = {.kv_buffer = state.data_ptr(), .kv_input = input.data_ptr(), .kv_output = nullptr,
                              .score_bias = ape.data_ptr(), .plan_d = nullptr, .batch_size = n},
                 .store = {.input = nullptr, .handle = nullptr, .weight = weight.data_ptr(),
                           .freqs_cis = static_cast<const float*>(freqs.data_ptr()), .out_loc = nullptr,
                           .kvcache = static_cast<uint8_t*>(cache.data_ptr()), .eps = eps,
                           .compress_ratio = 4, .num_tokens = n}},
        .positions = positions.data_ptr(), .requests = requests.data_ptr(),
        .state_slots = state_slots.data_ptr(), .kv_slots = kv_slots.data_ptr(),
        .block_table = block_table.data_ptr(), .block_stride = block_table.stride(0),
        .max_blocks = static_cast<int32_t>(block_table.size(1)),
        .state_entries = state_entries, .state_tokens = state_tokens,
    };
    LaunchKernel(div_ceil(n, 4u), 128, device.unwrap()).enable_pdl(kUsePDL)(
        c4_decode_queries_store<kUsePDL, kQueries, kInt64Mask>, params);
  }
};
}  // namespace
