// GLM5 ordinary BF16-normalized input and ordinary FP32 router
// logits -> GroupTopK-compatible routing plus group32 MegaMoE input.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace {
namespace cg          = cooperative_groups;
constexpr int kHidden = 6144, kExperts = 256, kTopK = 8;

__global__ void router_pack_kernel(const __nv_bfloat16* hidden,
                                   const float*         logits,
                                   const float*         bias,
                                   unsigned char*       activation,
                                   int*                 scales,
                                   int64_t*             ids,
                                   float*               weights) {
    const int        row = blockIdx.x, thread = threadIdx.x, lane = thread & 31;
    __shared__ float probability[kExperts];
    __shared__ float adjusted[kExperts];
    __shared__ int   selected[kTopK];
    // Keep ordinary CUDA sigmoid and its separate FP32 bias addition.
    const float probability_value = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-logits[row * kExperts + thread])));
    probability[thread]           = probability_value;
    adjusted[thread]              = __fadd_rn(probability_value, bias[thread]);
    __syncthreads();

    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<32>(block);
    if (thread < 32) {
        // Same lane ownership/comparisons as no_aux_tc_kernels.cu::topk_with_k2.
        float largest = -INFINITY, second = -INFINITY;
        float candidates[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float value = adjusted[lane + i * 32];
            if (value > largest) {
                second  = largest;
                largest = value;
            } else if (value > second) {
                second = value;
            }
            // Ordinary stable WarpSelect replaces nonfinite candidates with
            // numeric_limits<float>::lowest(), not with negative infinity.
            candidates[i] = isfinite(value) ? value : -FLT_MAX;
        }
        const float max1 = cg::reduce(tile, largest, cg::greater<float>());
        float       max2 = max1;
        if (__popc(__ballot_sync(0xffffffffu, largest == max1)) == 1) {
            largest = largest == max1 ? second : largest;
            max2    = cg::reduce(tile, largest, cg::greater<float>());
        }
        const float group_score = __fadd_rn(max1, max2);
        const bool  valid_group = isfinite(group_score) && group_score != -FLT_MAX;
        if (valid_group) {
            // Repeated exact max/lowest-id selection has the same total order
            // as ordinary greater=true,is_stable=true WarpSelect. No sums or
            // rounded arithmetic are used for the ranking itself.
#pragma unroll
            for (int rank = 0; rank < kTopK; ++rank) {
                float local_best = -INFINITY;
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    local_best = fmaxf(local_best, candidates[i]);
                const float best     = cg::reduce(tile, local_best, cg::greater<float>());
                int         local_id = 0x7fffffff;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (candidates[i] == best)
                        local_id = min(local_id, lane + i * 32);
                }
                const int best_id = cg::reduce(tile, local_id, cg::less<int>());
                if (lane == 0)
                    selected[rank] = best_id;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (lane + i * 32 == best_id)
                        candidates[i] = -INFINITY;
                }
            }
            __syncwarp();
            const float value = lane < kTopK ? probability[selected[lane]] : 0.0f;
            // Preserve GroupTopK's 32-lane reduction followed by +1e-20.
            const float sum = __fadd_rn(1.0e-20f, cg::reduce(tile, value, cg::plus<float>()));
            if (lane < kTopK) {
                ids[row * kTopK + lane]     = selected[lane];
                weights[row * kTopK + lane] = __fmul_rn(__fdiv_rn(value, sum), 2.5f);
            }
        } else if (lane < kTopK) {
            // Ordinary invalid-group fallback does NOT apply the 2.5 factor.
            ids[row * kTopK + lane]     = lane;
            weights[row * kTopK + lane] = 0.125f;
        }
    }

    // Independent work: other warps may quantize while warp0 selects experts.
    // No changed RMSNorm or BF16 rounding boundary is introduced here.
#pragma unroll
    for (int round = 0; round < 3; ++round) {
        const int column = round * 2048 + thread * 8;
        float     values[8], maximum = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            values[i] = __bfloat162float(hidden[row * kHidden + column + i]);
            // Triton tl.max/tl.maximum default to PropagateNan.NONE.
            maximum = fmaxf(maximum, fabsf(values[i]));
        }
        for (int offset = 2; offset; offset >>= 1) {
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset, 4));
        }
        const float    initial  = __fdiv_rn(fmaxf(maximum, 1.0e-4f), 448.0f);
        const unsigned bits     = __float_as_uint(initial);
        unsigned       exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0);
        exponent                = min(254u, max(1u, exponent));
        const float scale       = __uint_as_float(exponent << 23);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float value = __fdiv_rn(values[i], scale);
            // Match the existing packer's SM100 PTX, including NaN and -0:
            // tl.clamp lowers to min.xorsign.abs with PropagateNan.NONE.
            asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(value) : "f"(value), "f"(448.0f));
            activation[row * kHidden + column + i] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
        }
        const int      half_warp = lane & 16;
        const unsigned e0        = __shfl_sync(0xffffffffu, exponent, half_warp);
        const unsigned e1        = __shfl_sync(0xffffffffu, exponent, half_warp + 4);
        const unsigned e2        = __shfl_sync(0xffffffffu, exponent, half_warp + 8);
        const unsigned e3        = __shfl_sync(0xffffffffu, exponent, half_warp + 12);
        if ((lane & 15) == 0) {
            scales[row * 48 + round * 16 + thread / 16] = e0 | e1 << 8 | e2 << 16 | e3 << 24;
        }
    }
}
}  // namespace

void initialize_router_pack() {
    cudaFuncAttributes attributes{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, router_pack_kernel));
}

void launch_router_pack(const at::Tensor& hidden,
                        const at::Tensor& logits,
                        const at::Tensor& bias,
                        const at::Tensor& activation,
                        const at::Tensor& scales,
                        const at::Tensor& ids,
                        const at::Tensor& weights) {
    const c10::cuda::CUDAGuard guard(hidden.device());
    router_pack_kernel<<<hidden.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr()),
        logits.data_ptr<float>(),
        bias.data_ptr<float>(),
        reinterpret_cast<unsigned char*>(activation.data_ptr()),
        scales.data_ptr<int>(),
        ids.data_ptr<int64_t>(),
        weights.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
