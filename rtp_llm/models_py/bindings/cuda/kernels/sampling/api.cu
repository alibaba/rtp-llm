#include "sampling.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <tuple>

#include "flashinfer/air_top_p.cuh"
#include "flashinfer/sampling.cuh"

namespace flashinfer {
namespace sampling {

// ---------------------------------------------------------------------------
// DSpARK fused draft sample: softmax + top-p (nucleus) + CDF sampling in one
// kernel, replacing softmax_warp_forward + TopPRenormProb + SamplingFromProb for
// the compact draft path (rows = batch, width = candidate width = 1024).
//
// Semantics are pinned to the kernels it replaces:
//   * softmax:        max-subtracted exp, normalised by the row sum (exact
//                     division, like ATen); only the summation order differs.
//   * top-p:          line-by-line port of flashinfer::sampling::TopPRenormProbKernel
//                     (the K != 4096 path, which is what K=1024 uses): the same
//                     ternary search on the threshold, the same `ieee_*` arithmetic,
//                     the same stopping condition `min_gt_low != max_le_high`, and
//                     the same support rule `prob > low`. The support set is
//                     therefore identical, not just close.
//   * sampling:       Philox state from `curand_init(seed, row, offset)` -- exactly
//                     the mapping of SamplingFromProbDeviceScalarSeedKernel -- and
//                     the first candidate whose inclusive CDF exceeds `u`, in
//                     candidate order, with the same "last positive index" fallback.
// ---------------------------------------------------------------------------
template<uint32_t BLOCK_THREADS, uint32_t WIDTH>
__global__ void DSparkFusedDraftSampleKernel(const float* __restrict__ scaled_logits,
                                             float* __restrict__ q_out,
                                             int32_t* __restrict__ token_out,
                                             const float* __restrict__ top_p_arr,
                                             float top_p_val,
                                             const uint64_t* __restrict__ seed_scalar,
                                             const uint64_t* __restrict__ offset_scalar) {
    constexpr uint32_t ITEMS = WIDTH / BLOCK_THREADS;
    static_assert(WIDTH % BLOCK_THREADS == 0, "width must be a multiple of the block size");
    using FloatReduce = cub::BlockReduce<float, BLOCK_THREADS>;

    constexpr uint32_t VEC_SIZE = 16 / sizeof(float);   // 4: whole row in one helper call
    static_assert(WIDTH == BLOCK_THREADS * VEC_SIZE, "the row must fit one sampling helper call");
    using SampleStorage = flashinfer::sampling::SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>;

    __shared__ typename FloatReduce::TempStorage reduce_storage;
    __shared__ float                             smem_broadcast;
    __shared__ __align__(16) float               smem_probs[WIDTH];
    __shared__ SampleStorage                     sample_storage;

    const uint32_t row     = blockIdx.x, lane = threadIdx.x;
    const float    top_p   = top_p_arr == nullptr ? top_p_val : top_p_arr[row];
    const float*   logits  = scaled_logits + static_cast<size_t>(row) * WIDTH;
    float*         q_row   = q_out + static_cast<size_t>(row) * WIDTH;

    // ---- softmax -----------------------------------------------------------
    float values[ITEMS];
    float local_max = -cuda::std::numeric_limits<float>::infinity();
#pragma unroll
    for (uint32_t i = 0; i < ITEMS; ++i) {
        values[i] = logits[i * BLOCK_THREADS + lane];
        local_max = fmaxf(local_max, values[i]);
    }
    float row_max = FloatReduce(reduce_storage).Reduce(local_max, MaxReduceOp{});
    if (lane == 0) {
        smem_broadcast = row_max;
    }
    __syncthreads();
    row_max = smem_broadcast;

    float exp_values[ITEMS];
    float local_sum = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < ITEMS; ++i) {
        exp_values[i] = __expf(values[i] - row_max);
        local_sum += exp_values[i];
    }
    float row_sum = FloatReduce(reduce_storage).Sum(local_sum);
    if (lane == 0) {
        smem_broadcast = row_sum;
    }
    __syncthreads();
    row_sum = smem_broadcast;
    if (!(row_sum > 0.0f) || !isfinite(row_sum)) {
        // Degenerate row: emit zeros and token 0 rather than NaN.
        if (lane == 0) {
            token_out[row] = 0;
        }
        for (uint32_t i = 0; i < ITEMS; ++i) {
            q_row[i * BLOCK_THREADS + lane] = 0.0f;
        }
        return;
    }
#pragma unroll
    for (uint32_t i = 0; i < ITEMS; ++i) {
        smem_probs[i * BLOCK_THREADS + lane] = exp_values[i] / row_sum;
    }
    __syncthreads();

    // ---- top-p nucleus (port of TopPRenormProbKernel) ----------------------
    // `low` ends up as the largest threshold whose surviving mass still reaches
    // top_p, and the support is exactly {prob > low}.
    float normalizer = 1.0f;
    if (top_p >= 1.0f) {
        // Reference fast path: renormalise the row by its own sum.
        float thread_sum = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < ITEMS; ++i) {
            thread_sum += smem_probs[i * BLOCK_THREADS + lane];
        }
        float renorm_sum = FloatReduce(reduce_storage).Sum(thread_sum);
        if (lane == 0) {
            smem_broadcast = renorm_sum;
        }
        __syncthreads();
        renorm_sum     = smem_broadcast;
        const float denom = (renorm_sum <= 1e-8f) ? 1.0f : renorm_sum;
        normalizer        = math::ptx_rcp(denom);
#pragma unroll
        for (uint32_t i = 0; i < ITEMS; ++i) {
            const uint32_t idx = i * BLOCK_THREADS + lane;
            smem_probs[idx]    = smem_probs[idx] * normalizer;
        }
        __syncthreads();
    } else {
        float thread_max_prob = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < ITEMS; ++i) {
            thread_max_prob = fmaxf(thread_max_prob, smem_probs[i * BLOCK_THREADS + lane]);
        }
        float max_prob = FloatReduce(reduce_storage).Reduce(thread_max_prob, MaxReduceOp{});
        if (lane == 0) {
            smem_broadcast = max_prob;
        }
        __syncthreads();
        max_prob = smem_broadcast;

        float low = 0.0f, high = max_prob;
        float min_gt_low = 0.0f, max_le_high = 0.0f;
        float sum_low = 1.0f;
        do {
            // Same pivots and the same IEEE-754 helpers as the reference so the
            // search visits identical thresholds (fast-math cannot flush them).
            const float pivot_0 = ieee_div(ieee_add(high, ieee_mul(2.f, low)), 3.f);
            const float pivot_1 = ieee_div(ieee_add(ieee_mul(2.f, high), low), 3.f);
            float       thread_gt_pivot_0 = 0.0f, thread_gt_pivot_1 = 0.0f;
            float       thread_min_gt_low = high, thread_max_le_high = low;
#pragma unroll
            for (uint32_t i = 0; i < ITEMS; ++i) {
                const float    v = smem_probs[i * BLOCK_THREADS + lane];
                const uint32_t idx = i * BLOCK_THREADS + lane;
                if (v > pivot_0) {
                    thread_gt_pivot_0 += v;
                }
                if (v > pivot_1) {
                    thread_gt_pivot_1 += v;
                }
                if (v > low && idx < WIDTH) {
                    thread_min_gt_low = fminf(thread_min_gt_low, v);
                }
                if (v <= high && idx < WIDTH) {
                    thread_max_le_high = fmaxf(thread_max_le_high, v);
                }
            }
            float aggregate_gt_pivot_0 = FloatReduce(reduce_storage).Sum(thread_gt_pivot_0);
            __syncthreads();
            float aggregate_gt_pivot_1 = FloatReduce(reduce_storage).Sum(thread_gt_pivot_1);
            __syncthreads();
            float next_min_gt_low = FloatReduce(reduce_storage).Reduce(thread_min_gt_low, MinReduceOp{});
            __syncthreads();
            float next_max_le_high = FloatReduce(reduce_storage).Reduce(thread_max_le_high, MaxReduceOp{});
            if (lane == 0) {
                smem_broadcast = next_max_le_high;
            }
            __syncthreads();
            next_max_le_high = smem_broadcast;
            // The reference stores both reductions in shared memory and reloads them
            // for all threads; reproduce that broadcast so every thread compares the
            // same values.
            if (lane == 0) {
                smem_broadcast = aggregate_gt_pivot_0;
            }
            __syncthreads();
            aggregate_gt_pivot_0 = smem_broadcast;
            if (lane == 0) {
                smem_broadcast = aggregate_gt_pivot_1;
            }
            __syncthreads();
            aggregate_gt_pivot_1 = smem_broadcast;
            if (lane == 0) {
                smem_broadcast = next_min_gt_low;
            }
            __syncthreads();
            next_min_gt_low = smem_broadcast;
            min_gt_low      = next_min_gt_low;
            max_le_high     = next_max_le_high;

            if (aggregate_gt_pivot_1 >= top_p) {
                low     = pivot_1;
                sum_low = aggregate_gt_pivot_1;
            } else if (aggregate_gt_pivot_0 >= top_p) {
                low     = pivot_0;
                high    = fminf(pivot_1, max_le_high);
                sum_low = aggregate_gt_pivot_0;
            } else {
                high = fminf(pivot_0, max_le_high);
            }
            __syncthreads();
        } while (min_gt_low != max_le_high);

        normalizer = math::ptx_rcp(fmaxf(sum_low, 1e-8f));
#pragma unroll
        for (uint32_t i = 0; i < ITEMS; ++i) {
            const uint32_t idx = i * BLOCK_THREADS + lane;
            const float    v   = smem_probs[idx];
            smem_probs[idx]    = (v > low) ? v * normalizer : 0.0f;
        }
        __syncthreads();
    }

    // ---- CDF sampling: the reference helper, not a re-implementation --------
    // Same Philox state as SamplingFromProbDeviceScalarSeedKernel
    // (curand_init(seed, row, offset)), same `DeviceSamplingFromProb` reduction and
    // scan, and the whole row fits in one call (WIDTH == BLOCK_THREADS * VEC_SIZE),
    // exactly like the production sampler's single iteration.
    curandStatePhilox4_32_10_t state;
    curand_init(*seed_scalar, static_cast<uint64_t>(row), *offset_scalar, &state);
    sample_storage.sampled_id    = WIDTH;
    sample_storage.last_valid_id = -1;
    __syncthreads();
    const float                         u = curand_uniform(&state);
    vec_t<float, VEC_SIZE> probs_vec;
    probs_vec.fill(0.0f);
    probs_vec.cast_load(smem_probs + lane * VEC_SIZE);
    float aggregate = 0.0f;
    flashinfer::sampling::DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, /*DETERMINISTIC=*/true>(
        0, WIDTH, [](float x) { return x > 0; }, u, probs_vec, aggregate, &sample_storage);
    __syncthreads();
    if (lane == 0) {
        int sampled_id = sample_storage.sampled_id;
        if (sampled_id == static_cast<int>(WIDTH)) {
            // u landed past the row total through rounding: fall back to the last
            // positive candidate, exactly like the reference kernel.
            sampled_id = sample_storage.last_valid_id == -1 ? 0 : sample_storage.last_valid_id;
        }
        token_out[row] = sampled_id;
    }
    __syncthreads();

    // ---- write the top-p renormalised distribution -------------------------
#pragma unroll
    for (uint32_t i = 0; i < ITEMS; ++i) {
        q_row[i * BLOCK_THREADS + lane] = smem_probs[i * BLOCK_THREADS + lane];
    }
}


template<uint32_t             BLOCK_THREADS,
         BlockScanAlgorithm   SCAN_ALGORITHM,
         BlockReduceAlgorithm REDUCE_ALGORITHM,
         uint32_t             VEC_SIZE,
         bool                 DETERMINISTIC,
         typename DType,
         typename IdType>
__global__ void SamplingFromProbDeviceScalarSeedKernel(DType*    probs,
                                                       IdType*   output,
                                                       bool*     valid,
                                                       IdType*   indices,
                                                       uint32_t  d,
                                                       uint32_t  row_stride,
                                                       uint64_t* seed_scalar,
                                                       uint64_t* offset_scalar) {
    curandStatePhilox4_32_10_t state;
    const uint32_t             bx = blockIdx.x, tx = threadIdx.x;

    // Scalar-seed semantics: the batch shares one (seed, offset); the Philox
    // subsequence is the row index (SamplingPhiloxSubsequence without a seed array).
    curand_init(*seed_scalar, static_cast<uint64_t>(bx), *offset_scalar, &state);
    const uint32_t row_idx = indices == nullptr ? bx : indices[bx];

    extern __shared__ __align__(alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
        uint8_t       smem_sampling[];
    auto&             temp_storage =
        reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(smem_sampling);
    temp_storage.sampled_id    = d;
    temp_storage.last_valid_id = -1;
    __syncthreads();

    vec_t<float, VEC_SIZE> probs_vec;
    float                  aggregate(0);
    float                  u = curand_uniform(&state);

#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
        probs_vec.fill(0);
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
            probs_vec.cast_load(probs + static_cast<size_t>(row_idx) * row_stride + i * BLOCK_THREADS * VEC_SIZE
                                + tx * VEC_SIZE);
        }

        DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, DETERMINISTIC>(
            i, d, [](float x) { return x > 0; }, u, probs_vec, aggregate, &temp_storage);
        if (float(aggregate) > u) {
            break;
        }
    }
    int sampled_id = temp_storage.sampled_id;
    if (sampled_id == d) {
        // u is very close to 1 and the probability sum is smaller than u.
        if (temp_storage.last_valid_id == -1) {
            if (tx == 0) {
                output[bx] = 0;
                valid[bx]  = false;
            }
            return;
        }
        sampled_id = temp_storage.last_valid_id;
    }
    if (tx == 0) {
        output[bx] = sampled_id;
        valid[bx]  = true;
    }
}

}  // namespace sampling
}  // namespace flashinfer

namespace rtp_llm {
namespace {

void check_renorm_inputs(const torch::Tensor& probs, const torch::Tensor& renorm_probs) {
    TORCH_CHECK(probs.is_cuda(), "probs must be a CUDA tensor");
    TORCH_CHECK(renorm_probs.is_cuda(), "renorm_probs must be a CUDA tensor");
    TORCH_CHECK(probs.dim() == 2, "probs must be a 2D tensor");
    TORCH_CHECK(renorm_probs.dim() == 2, "renorm_probs must be a 2D tensor");
    TORCH_CHECK(probs.sizes() == renorm_probs.sizes(), "renorm_probs shape must match probs shape");
    TORCH_CHECK(probs.scalar_type() == renorm_probs.scalar_type(), "renorm_probs dtype must match probs dtype");
    TORCH_CHECK(probs.device() == renorm_probs.device(), "renorm_probs device must match probs device");
    TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
    TORCH_CHECK(renorm_probs.is_contiguous(), "renorm_probs must be contiguous");
}

void check_sampling_inputs(const torch::Tensor& probs,
                           const torch::Tensor& output,
                           const torch::Tensor& valid,
                           bool                 allow_row_stride = false) {
    TORCH_CHECK(probs.is_cuda(), "probs must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(valid.is_cuda(), "valid must be a CUDA tensor");
    TORCH_CHECK(probs.dim() == 2, "probs must be a 2D tensor");
    TORCH_CHECK(output.dim() == 1, "output must be a 1D tensor");
    TORCH_CHECK(valid.dim() == 1, "valid must be a 1D tensor");
    TORCH_CHECK(output.size(0) == valid.size(0), "output and valid must have the same batch size");
    TORCH_CHECK(output.size(0) == probs.size(0), "output batch size must match probs batch size");
    TORCH_CHECK(probs.scalar_type() == at::kFloat, "sampling from probs currently expects float32 probs");
    TORCH_CHECK(output.scalar_type() == at::kInt, "sampling output currently expects int32 dtype");
    TORCH_CHECK(valid.scalar_type() == at::kBool, "valid must be bool dtype");
    TORCH_CHECK(probs.device() == output.device(), "output device must match probs device");
    TORCH_CHECK(probs.device() == valid.device(), "valid device must match probs device");
    if (allow_row_stride) {
        TORCH_CHECK(probs.stride(1) == 1 && probs.stride(0) >= probs.size(1), "probs must be row-major");
        TORCH_CHECK(probs.stride(0) <= std::numeric_limits<uint32_t>::max(), "probs row stride exceeds uint32 range");
    } else {
        TORCH_CHECK(probs.is_contiguous(), "probs must be contiguous");
    }
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(valid.is_contiguous(), "valid must be contiguous");
}

torch::Tensor prepare_optional_param(const std::optional<torch::Tensor>& maybe_param,
                                     int64_t                             batch_size,
                                     c10::Device                         device,
                                     c10::ScalarType                     dtype,
                                     const char*                         name) {
    if (!maybe_param.has_value()) {
        return torch::Tensor();
    }

    const auto& param = maybe_param.value();
    TORCH_CHECK(param.dim() == 1, name, " must be a 1D tensor with shape [batch_size]");
    TORCH_CHECK(
        param.numel() == batch_size, name, " length must match batch_size, got ", param.numel(), " vs ", batch_size);

    // Return the input tensor directly when it already matches; callers must clone before any in-place mutation.
    if (param.device() == device && param.scalar_type() == dtype && param.is_contiguous()) {
        return param;
    }
    return param.to(torch::TensorOptions().device(device).dtype(dtype), /*non_blocking=*/true).contiguous();
}

torch::Tensor
prepare_optional_index(const std::optional<torch::Tensor>& maybe_indices, int64_t batch_size, c10::Device device) {
    if (!maybe_indices.has_value()) {
        return torch::Tensor();
    }

    const auto& indices = maybe_indices.value();
    TORCH_CHECK(indices.dim() == 1, "indices must be a 1D tensor with shape [batch_size]");
    TORCH_CHECK(indices.numel() == batch_size,
                "indices length must match batch_size, got ",
                indices.numel(),
                " vs ",
                batch_size);
    TORCH_CHECK(indices.scalar_type() == at::kInt, "indices currently expects int32 dtype");
    if (indices.device() == device && indices.is_contiguous()) {
        return indices;
    }
    return indices.to(torch::TensorOptions().device(device).dtype(at::kInt), /*non_blocking=*/true).contiguous();
}

cudaStream_t resolve_stream(int64_t cuda_stream) {
    if (cuda_stream == 0) {
        return at::cuda::getCurrentCUDAStream().stream();
    }
    return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
}

template<bool IsDeterministic, typename DType>
size_t air_top_p_workspace_size(uint32_t batch_size, uint32_t vocab_size) {
    namespace air = flashinfer::sampling::air_top_p;
    auto align256 = [](size_t x) { return ((x + 255) / 256) * 256; };

    using HistT        = air::HisT<IsDeterministic, DType>;
    const auto buf_len = static_cast<size_t>(air::calcBufLen<DType>(static_cast<air::IdxT>(vocab_size)));

    const size_t counters_size = align256(sizeof(air::Counter<DType>) * batch_size);
    const size_t hist_size     = align256(sizeof(HistT) * air::NUM_BUCKETS * batch_size);
    const size_t count_size    = align256(sizeof(air::IdxT) * air::NUM_BUCKETS * batch_size);
    const size_t buf_size      = align256(sizeof(DType) * buf_len * batch_size);
    return counters_size + hist_size + count_size + 2 * buf_size;
}

void check_cuda_status(cudaError_t status, const char* kernel_name) {
    TORCH_CHECK(status == cudaSuccess, kernel_name, " failed with error code ", cudaGetErrorString(status));
    status = cudaGetLastError();
    TORCH_CHECK(status == cudaSuccess, kernel_name, " launch failed with error code ", cudaGetErrorString(status));
}

}  // namespace

std::tuple<uint64_t, uint64_t> get_seed_and_offset(int increment_size, std::optional<at::Generator> generator) {
    auto gen =
        at::get_generator_or_default<at::CUDAGeneratorImpl>(generator, at::cuda::detail::getDefaultCUDAGenerator());
    std::lock_guard<std::mutex> lock(gen->mutex_);
    at::PhiloxCudaState         rng_engine_inputs = gen->philox_cuda_state(increment_size);
    return {rng_engine_inputs.seed_.val, rng_engine_inputs.offset_.val};
}

void get_sampling_seed_offsets(const std::vector<at::Generator>& generators,
                               int64_t                           batch_size,
                               int                               increment_size,
                               int64_t*                          seeds,
                               int64_t*                          offsets) {
    TORCH_CHECK(batch_size >= 0 && static_cast<uint64_t>(batch_size) <= generators.size(),
                "sampling generators must contain every batch row");
    TORCH_CHECK(increment_size >= 0, "sampling RNG increment must be nonnegative");
    if (batch_size == 0) {
        return;
    }
    TORCH_CHECK(seeds != nullptr && offsets != nullptr, "sampling seed/offset outputs must be nonnull");
    // A captured PhiloxCudaState contains device pointers rather than scalar
    // seed/offset values. These host arrays cannot represent that graph state.
    at::cuda::assertNotCapturing("Cannot materialize sampling seed/offset host arrays during CUDA graph capture");
    const auto& default_generator = at::cuda::detail::getDefaultCUDAGenerator();
    const auto  row_increment     = ((static_cast<uint64_t>(increment_size) + 3) / 4) * 4;
    auto        generator_for_row = [&](int64_t row) {
        return at::check_generator<at::CUDAGeneratorImpl>(generators[row].defined() ? generators[row] :
                                                                                      default_generator);
    };
    for (int64_t begin = 0; begin < batch_size;) {
        auto*   generator = generator_for_row(begin);
        int64_t end       = begin + 1;
        while (end < batch_size && generator_for_row(end) == generator) {
            ++end;
        }
        const auto count = static_cast<uint64_t>(end - begin);
        TORCH_CHECK(row_increment == 0 || count <= std::numeric_limits<uint64_t>::max() / row_increment,
                    "sampling RNG increment overflow");
        std::lock_guard<std::mutex> lock(generator->mutex_);
        const auto                  state = generator->philox_cuda_state(row_increment * count);
        for (int64_t row = begin; row < end; ++row) {
            seeds[row]   = static_cast<int64_t>(state.seed_.val);
            offsets[row] = static_cast<int64_t>(state.offset_.val + (row - begin) * row_increment);
        }
        begin = end;
    }
}




void dspark_fused_draft_sample(torch::Tensor                scaled_logits,
                               torch::Tensor                q_out,
                               torch::Tensor                token_out,
                               std::optional<torch::Tensor> maybe_top_p_arr,
                               double                       top_p_val,
                               const torch::Tensor&         seed_scalar,
                               const torch::Tensor&         offset_scalar,
                               int64_t                      cuda_stream) {
    constexpr int      width   = 1024;
    constexpr uint32_t threads = 256;
    TORCH_CHECK(scaled_logits.is_cuda() && scaled_logits.scalar_type() == torch::kFloat32
                    && scaled_logits.dim() == 2 && scaled_logits.size(1) == width
                    && scaled_logits.is_contiguous(),
                "dspark fused draft sample expects contiguous CUDA FP32 [rows, 1024] scaled logits");
    const auto rows = scaled_logits.size(0);
    TORCH_CHECK(q_out.is_cuda() && q_out.scalar_type() == torch::kFloat32 && q_out.sizes() == scaled_logits.sizes()
                    && q_out.is_contiguous(),
                "q_out must be a contiguous FP32 tensor matching the scaled logits shape");
    TORCH_CHECK(token_out.is_cuda() && token_out.scalar_type() == torch::kInt32 && token_out.dim() == 1
                    && token_out.numel() == rows && token_out.is_contiguous(),
                "token_out must be a contiguous CUDA int32 tensor with one entry per row");
    const auto scalar_ok = [](const torch::Tensor& tensor, const char* name) {
        TORCH_CHECK(tensor.defined() && tensor.is_cuda() && tensor.scalar_type() == at::kLong && tensor.numel() == 1
                        && tensor.is_contiguous(),
                    name,
                    " must be a one-element contiguous CUDA int64 tensor");
    };
    scalar_ok(seed_scalar, "seed_scalar");
    scalar_ok(offset_scalar, "offset_scalar");
    TORCH_CHECK(seed_scalar.device() == scaled_logits.device() && offset_scalar.device() == scaled_logits.device(),
                "seed_scalar/offset_scalar must live on the logits device");

    at::cuda::CUDAGuard device_guard(scaled_logits.device());
    auto               stream = resolve_stream(cuda_stream);
    const auto         top_p  = prepare_optional_param(maybe_top_p_arr, rows, scaled_logits.device(), at::kFloat, "top_p");

    const float*   logits_ptr = scaled_logits.data_ptr<float>();
    float*         q_ptr      = q_out.data_ptr<float>();
    int32_t*       token_ptr  = token_out.data_ptr<int32_t>();
    const float*   top_p_ptr  = top_p.defined() ? top_p.data_ptr<float>() : nullptr;
    float          top_p_v    = static_cast<float>(top_p_val);
    const uint64_t* seed_ptr  = reinterpret_cast<const uint64_t*>(seed_scalar.data_ptr<int64_t>());
    const uint64_t* offset_ptr = reinterpret_cast<const uint64_t*>(offset_scalar.data_ptr<int64_t>());

    auto kernel = flashinfer::sampling::DSparkFusedDraftSampleKernel<threads, width>;
    void* args[] = {&logits_ptr, &q_ptr, &token_ptr, &top_p_ptr, &top_p_v, &seed_ptr, &offset_ptr};
    const cudaError_t status =
        cudaLaunchKernel((void*)kernel, dim3(static_cast<uint32_t>(rows)), dim3(threads), args, 0, stream);
    TORCH_CHECK(status == cudaSuccess, "dspark fused draft sample launch failed: ", cudaGetErrorString(status));
}

void sampling_from_probs_device_scalar_seed(torch::Tensor                probs,
                                            torch::Tensor                output,
                                            torch::Tensor                valid,
                                            std::optional<torch::Tensor> maybe_indices,
                                            bool                         deterministic,
                                            const torch::Tensor&         seed_scalar,
                                            const torch::Tensor&         offset_scalar,
                                            int64_t                      cuda_stream) {
    check_sampling_inputs(probs, output, valid, /*allow_row_stride=*/true);
    const auto scalar_ok = [](const torch::Tensor& tensor, const char* name) {
        TORCH_CHECK(tensor.defined() && tensor.is_cuda() && tensor.scalar_type() == at::kLong && tensor.numel() == 1
                        && tensor.is_contiguous(),
                    name,
                    " must be a one-element contiguous CUDA int64 tensor");
    };
    scalar_ok(seed_scalar, "seed_scalar");
    scalar_ok(offset_scalar, "offset_scalar");
    TORCH_CHECK(seed_scalar.device() == probs.device() && offset_scalar.device() == probs.device(),
                "seed_scalar/offset_scalar must live on the probs device");
    // RTP always samples with the deterministic scan; keeping the check here means
    // only one template instantiation is emitted.
    TORCH_CHECK(deterministic, "device-scalar-seed sampling requires the deterministic scan");

    using namespace flashinfer::sampling;
    at::cuda::CUDAGuard device_guard(probs.device());
    const auto          batch_size = static_cast<uint32_t>(output.size(0));
    // Non-const so their addresses can go into the kernel argument array.
    uint32_t            vocab_size = static_cast<uint32_t>(probs.size(1));
    uint32_t            row_stride = static_cast<uint32_t>(probs.stride(0));
    auto                indices    = prepare_optional_index(maybe_indices, batch_size, probs.device());
    auto                stream     = resolve_stream(cuda_stream);

    float*   probs_ptr   = static_cast<float*>(probs.data_ptr());
    int32_t* output_ptr  = static_cast<int32_t*>(output.data_ptr());
    bool*    valid_ptr   = static_cast<bool*>(valid.data_ptr());
    int32_t* indices_ptr = indices.defined() ? static_cast<int32_t*>(indices.data_ptr()) : nullptr;
    uint64_t* seed_ptr   = reinterpret_cast<uint64_t*>(const_cast<int64_t*>(seed_scalar.data_ptr<int64_t>()));
    uint64_t* offset_ptr = reinterpret_cast<uint64_t*>(const_cast<int64_t*>(offset_scalar.data_ptr<int64_t>()));

    // Mirror flashinfer::sampling::SamplingFromProb's launch configuration exactly
    // (same vec_size derivation, same block-size dispatch) so the two kernels issue
    // identical work per row.
    const uint32_t vec_size =
        std::gcd(std::gcd(16 / sizeof(float), vocab_size),
                 std::gcd(row_stride, static_cast<uint32_t>(reinterpret_cast<uintptr_t>(probs_ptr) / sizeof(float))));

    auto compute_capacity = flashinfer::GetCudaComputeCapability();
    DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
        dim3  nblks(batch_size);
        dim3  nthrs(BLOCK_THREADS);
        void* args[] = {
            &probs_ptr, &output_ptr, &valid_ptr, &indices_ptr, &vocab_size, &row_stride, &seed_ptr, &offset_ptr};
        const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
        DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
            auto kernel = SamplingFromProbDeviceScalarSeedKernel<BLOCK_THREADS,
                                                                 SCAN_ALGO,
                                                                 REDUCE_ALGO,
                                                                 VEC_SIZE,
                                                                 /*DETERMINISTIC=*/true,
                                                                 float,
                                                                 int32_t>;
            const cudaError_t status = cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream);
            TORCH_CHECK(status == cudaSuccess,
                        "device-scalar-seed sampling launch failed: ",
                        cudaGetErrorString(status));
        });
    });
}

void sampling_from_probs(torch::Tensor                probs,
                         torch::Tensor                output,
                         torch::Tensor                valid,
                         std::optional<torch::Tensor> maybe_indices,
                         bool                         deterministic,
                         std::optional<torch::Tensor> maybe_seed_arr,
                         uint64_t                     seed_val,
                         std::optional<torch::Tensor> maybe_offset_arr,
                         uint64_t                     offset_val,
                         int64_t                      cuda_stream) {
    check_sampling_inputs(probs, output, valid, /*allow_row_stride=*/true);

    at::cuda::CUDAGuard device_guard(probs.device());
    const auto          batch_size = static_cast<uint32_t>(output.size(0));
    const auto          vocab_size = static_cast<uint32_t>(probs.size(1));
    auto                indices    = prepare_optional_index(maybe_indices, batch_size, probs.device());
    auto                seed = prepare_optional_param(maybe_seed_arr, batch_size, probs.device(), at::kLong, "seed");
    auto offset = prepare_optional_param(maybe_offset_arr, batch_size, probs.device(), at::kLong, "offset");
    auto stream = resolve_stream(cuda_stream);

    cudaError_t status = flashinfer::sampling::SamplingFromProb<float, int32_t>(
        static_cast<float*>(probs.data_ptr()),
        static_cast<int32_t*>(output.data_ptr()),
        static_cast<bool*>(valid.data_ptr()),
        indices.defined() ? static_cast<int32_t*>(indices.data_ptr()) : nullptr,
        batch_size,
        vocab_size,
        static_cast<uint32_t>(probs.stride(0)),
        deterministic,
        seed.defined() ? reinterpret_cast<uint64_t*>(seed.data_ptr<int64_t>()) : nullptr,
        seed_val,
        offset.defined() ? reinterpret_cast<uint64_t*>(offset.data_ptr<int64_t>()) : nullptr,
        offset_val,
        stream);
    check_cuda_status(status, "SamplingFromProb");
}

void top_p_sampling_from_probs(torch::Tensor                probs,
                               torch::Tensor                output,
                               torch::Tensor                valid,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_p_arr,
                               double                       top_p_val,
                               bool                         deterministic,
                               std::optional<torch::Tensor> maybe_seed_arr,
                               uint64_t                     seed_val,
                               std::optional<torch::Tensor> maybe_offset_arr,
                               uint64_t                     offset_val,
                               int64_t                      cuda_stream) {
    check_sampling_inputs(probs, output, valid);

    at::cuda::CUDAGuard device_guard(probs.device());
    const auto          batch_size = static_cast<uint32_t>(output.size(0));
    const auto          vocab_size = static_cast<uint32_t>(probs.size(1));
    auto                indices    = prepare_optional_index(maybe_indices, batch_size, probs.device());
    auto top_p  = prepare_optional_param(maybe_top_p_arr, batch_size, probs.device(), at::kFloat, "top_p");
    auto seed   = prepare_optional_param(maybe_seed_arr, batch_size, probs.device(), at::kLong, "seed");
    auto offset = prepare_optional_param(maybe_offset_arr, batch_size, probs.device(), at::kLong, "offset");
    auto stream = resolve_stream(cuda_stream);

    cudaError_t status = flashinfer::sampling::TopPSamplingFromProb<float, int32_t>(
        static_cast<float*>(probs.data_ptr()),
        static_cast<int32_t*>(output.data_ptr()),
        static_cast<bool*>(valid.data_ptr()),
        indices.defined() ? static_cast<int32_t*>(indices.data_ptr()) : nullptr,
        top_p.defined() ? static_cast<float*>(top_p.data_ptr()) : nullptr,
        batch_size,
        static_cast<float>(top_p_val),
        vocab_size,
        deterministic,
        seed.defined() ? reinterpret_cast<uint64_t*>(seed.data_ptr<int64_t>()) : nullptr,
        seed_val,
        offset.defined() ? reinterpret_cast<uint64_t*>(offset.data_ptr<int64_t>()) : nullptr,
        offset_val,
        stream);
    check_cuda_status(status, "TopPSamplingFromProb");
}

void top_k_sampling_from_probs(torch::Tensor                probs,
                               torch::Tensor                output,
                               torch::Tensor                valid,
                               std::optional<torch::Tensor> maybe_indices,
                               std::optional<torch::Tensor> maybe_top_k_arr,
                               int64_t                      top_k_val,
                               bool                         deterministic,
                               std::optional<torch::Tensor> maybe_seed_arr,
                               uint64_t                     seed_val,
                               std::optional<torch::Tensor> maybe_offset_arr,
                               uint64_t                     offset_val,
                               int64_t                      cuda_stream) {
    check_sampling_inputs(probs, output, valid);

    at::cuda::CUDAGuard device_guard(probs.device());
    const auto          batch_size = static_cast<uint32_t>(output.size(0));
    const auto          vocab_size = static_cast<uint32_t>(probs.size(1));
    auto                indices    = prepare_optional_index(maybe_indices, batch_size, probs.device());
    auto                top_k = prepare_optional_param(maybe_top_k_arr, batch_size, probs.device(), at::kInt, "top_k");
    auto                seed  = prepare_optional_param(maybe_seed_arr, batch_size, probs.device(), at::kLong, "seed");
    auto       offset = prepare_optional_param(maybe_offset_arr, batch_size, probs.device(), at::kLong, "offset");
    auto       stream = resolve_stream(cuda_stream);
    const auto top_k_limit =
        static_cast<uint32_t>(top_k_val <= 0 ? vocab_size : std::min<int64_t>(top_k_val, vocab_size));

    cudaError_t status = flashinfer::sampling::TopKSamplingFromProb<float, int32_t>(
        static_cast<float*>(probs.data_ptr()),
        static_cast<int32_t*>(output.data_ptr()),
        static_cast<bool*>(valid.data_ptr()),
        indices.defined() ? static_cast<int32_t*>(indices.data_ptr()) : nullptr,
        top_k.defined() ? static_cast<int32_t*>(top_k.data_ptr()) : nullptr,
        batch_size,
        top_k_limit,
        vocab_size,
        deterministic,
        seed.defined() ? reinterpret_cast<uint64_t*>(seed.data_ptr<int64_t>()) : nullptr,
        seed_val,
        offset.defined() ? reinterpret_cast<uint64_t*>(offset.data_ptr<int64_t>()) : nullptr,
        offset_val,
        stream);
    check_cuda_status(status, "TopKSamplingFromProb");
}

void top_k_top_p_sampling_from_probs(torch::Tensor                probs,
                                     torch::Tensor                output,
                                     torch::Tensor                valid,
                                     std::optional<torch::Tensor> maybe_indices,
                                     std::optional<torch::Tensor> maybe_top_k_arr,
                                     int64_t                      top_k_val,
                                     std::optional<torch::Tensor> maybe_top_p_arr,
                                     double                       top_p_val,
                                     bool                         deterministic,
                                     std::optional<torch::Tensor> maybe_seed_arr,
                                     uint64_t                     seed_val,
                                     std::optional<torch::Tensor> maybe_offset_arr,
                                     uint64_t                     offset_val,
                                     int64_t                      cuda_stream) {
    check_sampling_inputs(probs, output, valid);

    at::cuda::CUDAGuard device_guard(probs.device());
    const auto          batch_size = static_cast<uint32_t>(output.size(0));
    const auto          vocab_size = static_cast<uint32_t>(probs.size(1));
    auto                indices    = prepare_optional_index(maybe_indices, batch_size, probs.device());
    auto                top_k = prepare_optional_param(maybe_top_k_arr, batch_size, probs.device(), at::kInt, "top_k");
    auto       top_p  = prepare_optional_param(maybe_top_p_arr, batch_size, probs.device(), at::kFloat, "top_p");
    auto       seed   = prepare_optional_param(maybe_seed_arr, batch_size, probs.device(), at::kLong, "seed");
    auto       offset = prepare_optional_param(maybe_offset_arr, batch_size, probs.device(), at::kLong, "offset");
    auto       stream = resolve_stream(cuda_stream);
    const auto top_k_limit =
        static_cast<int32_t>(top_k_val <= 0 ? vocab_size : std::min<int64_t>(top_k_val, vocab_size));

    cudaError_t status = flashinfer::sampling::TopKTopPSamplingFromProb<float, int32_t>(
        static_cast<float*>(probs.data_ptr()),
        top_k.defined() ? static_cast<int32_t*>(top_k.data_ptr()) : nullptr,
        top_p.defined() ? static_cast<float*>(top_p.data_ptr()) : nullptr,
        static_cast<int32_t*>(output.data_ptr()),
        static_cast<bool*>(valid.data_ptr()),
        indices.defined() ? static_cast<int32_t*>(indices.data_ptr()) : nullptr,
        batch_size,
        top_k_limit,
        static_cast<float>(top_p_val),
        vocab_size,
        deterministic,
        seed.defined() ? reinterpret_cast<uint64_t*>(seed.data_ptr<int64_t>()) : nullptr,
        seed_val,
        offset.defined() ? reinterpret_cast<uint64_t*>(offset.data_ptr<int64_t>()) : nullptr,
        offset_val,
        stream);
    check_cuda_status(status, "TopKTopPSamplingFromProb");
}

void top_p_renorm_probs(torch::Tensor                probs,
                        torch::Tensor                renorm_probs,
                        std::optional<torch::Tensor> maybe_top_p_arr,
                        double                       top_p_val,
                        int64_t                      cuda_stream) {
    check_renorm_inputs(probs, renorm_probs);
    TORCH_CHECK(probs.scalar_type() == at::kFloat, "top_p_renorm_probs currently expects float32 probs");

    at::cuda::CUDAGuard device_guard(probs.device());
    const auto          batch_size = static_cast<uint32_t>(probs.size(0));
    const auto          vocab_size = static_cast<uint32_t>(probs.size(1));
    auto   top_p     = prepare_optional_param(maybe_top_p_arr, batch_size, probs.device(), at::kFloat, "top_p");
    auto   stream    = resolve_stream(cuda_stream);
    float* top_p_ptr = top_p.defined() ? static_cast<float*>(top_p.data_ptr()) : nullptr;

    cudaError_t status = cudaSuccess;
    if (vocab_size < flashinfer::sampling::air_top_p::NUM_BUCKETS || vocab_size == 4096) {
        status = flashinfer::sampling::TopPRenormProb<float>(static_cast<float*>(probs.data_ptr()),
                                                             static_cast<float*>(renorm_probs.data_ptr()),
                                                             top_p_ptr,
                                                             batch_size,
                                                             static_cast<float>(top_p_val),
                                                             vocab_size,
                                                             stream);
        check_cuda_status(status, "TopPRenormProb");
        return;
    }

    constexpr bool deterministic  = false;
    const auto     workspace_size = air_top_p_workspace_size<deterministic, float>(batch_size, vocab_size);
    auto           workspace = torch::empty({static_cast<int64_t>(workspace_size)}, probs.options().dtype(at::kByte));

    status = flashinfer::sampling::air_top_p::AirTopPRenormProb<deterministic, float>(
        static_cast<float*>(probs.data_ptr()),
        static_cast<float*>(renorm_probs.data_ptr()),
        top_p_ptr,
        batch_size,
        static_cast<float>(top_p_val),
        vocab_size,
        workspace.data_ptr(),
        stream);
    check_cuda_status(status, "AirTopPRenormProb");
}

void top_k_renorm_probs(torch::Tensor                probs,
                        torch::Tensor                renorm_probs,
                        std::optional<torch::Tensor> maybe_top_k_arr,
                        int64_t                      top_k_val,
                        int64_t                      cuda_stream) {
    check_renorm_inputs(probs, renorm_probs);
    TORCH_CHECK(probs.scalar_type() == at::kFloat || probs.scalar_type() == at::kHalf
                    || probs.scalar_type() == at::kBFloat16,
                "top_k_renorm_probs expects float32, float16, or bfloat16 probs");

    at::cuda::CUDAGuard device_guard(probs.device());
    const auto          batch_size = static_cast<uint32_t>(probs.size(0));
    const auto          vocab_size = static_cast<uint32_t>(probs.size(1));
    auto                top_k  = prepare_optional_param(maybe_top_k_arr, batch_size, probs.device(), at::kInt, "top_k");
    auto                stream = resolve_stream(cuda_stream);
    int*                top_k_ptr   = top_k.defined() ? static_cast<int*>(top_k.data_ptr()) : nullptr;
    const auto          top_k_limit = static_cast<uint32_t>(
        top_k_val <= 0 ? vocab_size : std::min<int64_t>(top_k_val, std::numeric_limits<uint32_t>::max()));

    auto  row_states     = torch::zeros({1024 * 1024}, probs.options().dtype(at::kByte));
    auto* row_states_ptr = reinterpret_cast<flashinfer::sampling::RadixRowState*>(row_states.data_ptr());

    cudaError_t status = cudaSuccess;
    switch (probs.scalar_type()) {
        case at::kFloat:
            status = flashinfer::sampling::RadixTopKRenormProbMultiCTA<float, int>(
                static_cast<float*>(probs.data_ptr()),
                static_cast<float*>(renorm_probs.data_ptr()),
                top_k_ptr,
                batch_size,
                top_k_limit,
                vocab_size,
                row_states_ptr,
                stream);
            break;
        case at::kHalf:
            status = flashinfer::sampling::RadixTopKRenormProbMultiCTA<half, int>(
                reinterpret_cast<half*>(probs.data_ptr()),
                reinterpret_cast<half*>(renorm_probs.data_ptr()),
                top_k_ptr,
                batch_size,
                top_k_limit,
                vocab_size,
                row_states_ptr,
                stream);
            break;
        case at::kBFloat16:
            status = flashinfer::sampling::RadixTopKRenormProbMultiCTA<nv_bfloat16, int>(
                reinterpret_cast<nv_bfloat16*>(probs.data_ptr()),
                reinterpret_cast<nv_bfloat16*>(renorm_probs.data_ptr()),
                top_k_ptr,
                batch_size,
                top_k_limit,
                vocab_size,
                row_states_ptr,
                stream);
            break;
        default:
            TORCH_CHECK(false, "unsupported dtype for top_k_renorm_probs");
    }
    check_cuda_status(status, "RadixTopKRenormProbMultiCTA");
}

}  // namespace rtp_llm
