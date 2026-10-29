/*
 * Copyright (c) 2024 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Lamport transport follows FlashInfer; norm preserves its native FP32 residual sum.
#include <flashinfer/comm/trtllm_allreduce_fusion.cuh>
#include <flashinfer/math.cuh>
#include "nv_internal/tensorrt_llm/kernels/quantization_utils.cuh"
#include "tvm_ffi_utils.h"

namespace rtp_llm {
using namespace flashinfer;
namespace ar = flashinfer::trtllm_allreduce_fusion;
using Params = ar::AllReduceFusionParams<nv_bfloat16>;

__global__ __launch_bounds__(640) void sm120_ar_norm_fp4(Params params) {
    constexpr int width = 5120;
    cudaGridDependencySynchronize();
    ar::IndexHelper<nv_bfloat16> index(params);
    ar::LamportComm<2>           comm(params.workspace, params.rank);
    vec_t<nv_bfloat16, 8>        clear;
    clear.fill(ar::neg_zero_v<nv_bfloat16>);
    const float      global_scale = params.scale_factor[0];
    __shared__ float sums[20];

    for (int idx = index.access_id; idx < index.tot_access; idx += index.access_stride) {
        vec_t<nv_bfloat16, 8> val;
        val.load(static_cast<nv_bfloat16*>(params.allreduce_in) + idx * 8);
        ar::remove_neg_zero<nv_bfloat16, 8>(val);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            val.store(reinterpret_cast<nv_bfloat16*>(comm.data_bufs[r]) + (params.rank * index.tot_access + idx) * 8);
        }
    }
    for (int idx = index.access_id; idx < comm.clear_size / 8; idx += index.access_stride) {
        clear.store(reinterpret_cast<nv_bfloat16*>(comm.clear_buf) + idx * 8);
    }
    for (int idx = index.access_id, token = index.token_id; idx < index.tot_access;
         idx += index.access_stride, token += index.token_stride) {
        vec_t<nv_bfloat16, 8> vals[2];
        bool                  done = false;
        while (!done) {
            done = true;
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                vals[r].load_global_volatile(reinterpret_cast<nv_bfloat16*>(comm.data_bufs[params.rank])
                                             + (r * index.tot_access + idx) * 8);
                done &= !ar::has_neg_zero<nv_bfloat16, 8>(vals[r]);
            }
        }
        auto                  reduced = ar::allreduce_sum<nv_bfloat16, 8, 2, true>(vals);
        vec_t<nv_bfloat16, 8> residual, rounded, gamma;
        vec_t<float, 8>       added;
        residual.load(static_cast<nv_bfloat16*>(params.residual_in) + idx * 8);
        gamma.load(static_cast<nv_bfloat16*>(params.rms_gamma) + index.access_id_in_token * 8);
        float sum_sq = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float value = float(reduced[j]);
            value += float(residual[j]);
            sum_sq += value * value;
            residual[j] = nv_bfloat16(value);
            added[j]    = value;
        }
        residual.store(static_cast<nv_bfloat16*>(params.residual_out) + idx * 8);
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_sq += math::shfl_xor_sync(sum_sq, offset);
        }
        const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
        sums[warp] = sum_sq;
        __syncthreads();
        if (warp == 0) {
            sum_sq = lane < 20 ? sums[lane] : 0.f;
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum_sq += math::shfl_xor_sync(sum_sq, offset);
            }
            sums[0] = sum_sq;
        }
        __syncthreads();
        const float rms = math::rsqrt(sums[0] / float(width) + params.rms_eps);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            rounded[j] = nv_bfloat16(added[j] * rms * (0.f + float(gamma[j])));
        }
        rounded.store(static_cast<nv_bfloat16*>(params.norm_out) + idx * 8);
        tensorrt_llm::kernels::PackedVec<nv_bfloat16, 8> packed;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            packed.elts[j] = __halves2bfloat162(rounded[j * 2], rounded[j * 2 + 1]);
        }
        auto sf = tensorrt_llm::kernels::cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 16, 2>(
            token, index.access_id_in_token, width, static_cast<uint32_t*>(params.scale_out));
        static_cast<uint32_t*>(params.quant_out)[idx] =
            tensorrt_llm::kernels::cvt_warp_fp16_to_fp4<nv_bfloat16, 16, 8, false>(packed, global_scale, sf);
        __syncthreads();
    }
    comm.update(params.size * 2);
    cudaTriggerProgrammaticLaunchCompletion();
}

void ar_norm_fp4(TensorView x,
                 TensorView r,
                 TensorView gamma,
                 TensorView scale,
                 TensorView q,
                 TensorView sf,
                 TensorView norm,
                 TensorView workspace,
                 int64_t    rank,
                 double     eps) {
    ffi::CUDADeviceGuard guard(x.device().device_id);
    TVM_FFI_ICHECK_EQ(x.ndim(), 2);
    TVM_FFI_ICHECK_EQ(x.size(1), 5120);
    TVM_FFI_ICHECK_GT(x.size(0), 0);
    TVM_FFI_ICHECK_LE(x.size(0), 512);
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(x.dtype()), bfloat16_code);
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(r.dtype()), bfloat16_code);
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(gamma.dtype()), bfloat16_code);
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(norm.dtype()), bfloat16_code);
    TVM_FFI_ICHECK(x.IsContiguous() && r.IsContiguous() && gamma.IsContiguous() && norm.IsContiguous());
    TVM_FFI_ICHECK(scale.IsContiguous() && q.IsContiguous() && sf.IsContiguous() && workspace.IsContiguous());
    TVM_FFI_ICHECK_EQ(r.numel(), x.numel());
    TVM_FFI_ICHECK_EQ(norm.numel(), x.numel());
    TVM_FFI_ICHECK_EQ(gamma.numel(), 5120);
    TVM_FFI_ICHECK_EQ(scale.numel(), 1);
    TVM_FFI_ICHECK_EQ(scale.dtype().code, kDLFloat);
    TVM_FFI_ICHECK_EQ(scale.dtype().bits, 32);
    TVM_FFI_ICHECK_EQ(q.dtype().code, kDLUInt);
    TVM_FFI_ICHECK_EQ(q.dtype().bits, 8);
    TVM_FFI_ICHECK_EQ(sf.dtype().code, kDLUInt);
    TVM_FFI_ICHECK_EQ(sf.dtype().bits, 8);
    TVM_FFI_ICHECK_EQ(q.numel(), x.size(0) * 2560);
    TVM_FFI_ICHECK_GE(sf.numel(), (x.size(0) + 127) / 128 * 128 * 320);
    TVM_FFI_ICHECK_EQ(workspace.numel(), 7);
    TVM_FFI_ICHECK_EQ(workspace.dtype().code, kDLInt);
    TVM_FFI_ICHECK_EQ(workspace.dtype().bits, 64);
    for (const auto* view : {&r, &gamma, &scale, &q, &sf, &norm, &workspace}) {
        TVM_FFI_ICHECK_EQ(view->device().device_id, x.device().device_id);
    }
    TVM_FFI_ICHECK(rank == 0 || rank == 1);
    Params params{};
    params.nranks       = 2;
    params.rank         = rank;
    params.size         = x.numel();
    params.hidden_dim   = 5120;
    params.workspace    = reinterpret_cast<void**>(workspace.data_ptr());
    params.allreduce_in = x.data_ptr();
    params.residual_in  = r.data_ptr();
    params.residual_out = r.data_ptr();
    params.rms_gamma    = gamma.data_ptr();
    params.rms_eps      = float(eps);
    params.norm_out     = norm.data_ptr();
    params.quant_out    = q.data_ptr();
    params.scale_out    = sf.data_ptr();
    params.scale_factor = static_cast<float*>(scale.data_ptr());
    params.use_oneshot  = true;
    static int sm_count = 0;
    if (sm_count == 0) {
        TVM_FFI_ICHECK_EQ(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, x.device().device_id),
                          cudaSuccess);
    }
    cudaLaunchConfig_t config{};
    config.gridDim  = dim3(std::min<int>(x.size(0), sm_count));
    config.blockDim = dim3(640);
    config.stream   = get_stream(x.device());
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    attrs[1].id                                         = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim                             = {1, 1, 1};
    config.attrs                                        = attrs;
    config.numAttrs                                     = 2;
    TVM_FFI_ICHECK_EQ(cudaLaunchKernelEx(&config, sm120_ar_norm_fp4, params), cudaSuccess);
    TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}
}  // namespace rtp_llm
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, rtp_llm::ar_norm_fp4);
