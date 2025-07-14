#include "rtp_llm/cpp/devices/arm_impl/ArmDevice.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/core/cpu_allocator.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include <cstring>
#include "autil/StringUtil.h"
#include "gemm_opt/ArmGemmKernel.h"
#include <cfloat>
#include "rtp_llm/cpp/devices/utils/Timer.h"

#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f16.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qsi4c32p/kai_matmul_clamp_f16_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qsi4c32p/kai_matmul_clamp_f16_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qsi4c32p/kai_matmul_clamp_f16_qsi8d32p_qsi4c32p_interface.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p_bf16p_interface.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_bf16p_bf16p/kai_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"
#include "kai/ukernels/matmul/pack/kai_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p12x4biasf16_f16_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p12x4biasf32_f16_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_bf16p8x4_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_bf16p8x4_f16_neon.h"

namespace rtp_llm {

static const float HALF_FLT_MAX = 65504.F;

struct kai_matmul_ukernel_f32_qa8d32p_qs4c32p {
    kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_ukernel ukernel;
    std::string                                    name = {};
};

struct kai_matmul_ukernel_f16_qa8d32p_qs4c32p {
    kai_matmul_clamp_f16_qsi8d32p_qsi4c32p_ukernel ukernel;
    std::string                                    name = {};
};

kai_matmul_ukernel_f32_qa8d32p_qs4c32p fp32_ukernel_variants[] = {
    {kai_get_m_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_n_step_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_dst_size_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     "matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod"},
    {kai_get_m_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_dst_size_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     "matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm"},
};

kai_matmul_ukernel_f16_qa8d32p_qs4c32p fp16_ukernel_variants[] = {
    {kai_get_m_step_matmul_clamp_f16_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_n_step_matmul_clamp_f16_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_mr_matmul_clamp_f16_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_nr_matmul_clamp_f16_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_kr_matmul_clamp_f16_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_sr_matmul_clamp_f16_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_lhs_packed_offset_matmul_clamp_f16_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_rhs_packed_offset_matmul_clamp_f16_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_dst_offset_matmul_clamp_f16_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_get_dst_size_matmul_clamp_f16_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     kai_run_matmul_clamp_f16_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
     "matmul_clamp_f16_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod"},
    {kai_get_m_step_matmul_clamp_f16_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_n_step_matmul_clamp_f16_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_mr_matmul_clamp_f16_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_nr_matmul_clamp_f16_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_kr_matmul_clamp_f16_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_sr_matmul_clamp_f16_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_lhs_packed_offset_matmul_clamp_f16_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_rhs_packed_offset_matmul_clamp_f16_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_dst_offset_matmul_clamp_f16_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_get_dst_size_matmul_clamp_f16_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     kai_run_matmul_clamp_f16_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
     "matmul_clamp_f16_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm"},
};

/// @brief   basic gemm ops
/// @details D = alpha * op(A) * op(B) + beta * C
///          A [b, ..., m, k]
///          B [b, ..., k, n]
///          C [b, ..., m, n]
BufferPtr ArmCpuDevice::gemm_kai_bf16(const GemmParams& params) {
#ifdef GEMM_DEBUG
    auto start = std::chrono::high_resolution_clock::now();
#endif
    params.check();

    std::vector<size_t> Ashape;
    std::vector<size_t> Bshape;
    std::vector<size_t> Dshape;

    size_t dim;
    size_t m;
    size_t k;
    size_t n;

    Ashape = params.A.shape();
    Bshape = params.B.shape();

    dim = params.A.dim();

    if (params.transA == TransposeOperation::TRANSPOSE) {
        std::iter_swap(Ashape.end() - 1, Ashape.end() - 2);
    }

    if (params.transB == TransposeOperation::TRANSPOSE) {
        std::iter_swap(Bshape.end() - 1, Bshape.end() - 2);
    }

    m = Ashape[dim - 2];
    k = Ashape[dim - 1];
    n = Bshape[dim - 1];

    auto data_type = params.compute_type == DataType::TYPE_INVALID ? params.A.type() : params.compute_type;

    Dshape = std::vector<size_t>(Ashape.begin(), Ashape.end() - 2);
    Dshape.insert(Dshape.end(), {m, n});

    BufferPtr output;
    if (params.D) {
        output = params.D;
        RUNTIME_ASSERT_OP_ARG((data_type == params.D->type()) && (Dshape == params.D->shape()),
                              "Gemm output D shape and dtype mismatch: expected [%d][%s] but got [%s]",
                              data_type,
                              autil::StringUtil::toString(Dshape).c_str(),
                              params.D->debugString().c_str());
    } else {
        output = allocateBuffer({data_type, Dshape, AllocationType::DEVICE}, {"gemm_output"});
    }

    const size_t mr = kai_get_mr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla();
    const size_t nr = kai_get_nr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla();
    const size_t kr = kai_get_kr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla();
    const size_t sr = kai_get_sr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla();

    uint8_t* rhs_packed;
    uint8_t* lhs_packed;

    float* lhs = (float*)params.A.data();

    // Packing only needs to be performed once if the contents of the bias and RHS matrices are expected to be constant.
    int n_step = nr;
    rhs_packed = (uint8_t*)params.B.data();
    float* dst = (float*)output->data();

    int m_step = mr;

    if (params.A.type() == DataType::TYPE_FP32) {
        // lhs in fp32
        const size_t lhs_stride      = k * sizeof(float);
        const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_bf16p8x4_f32_neon(m, k, mr, kr, sr);
        lhs_packed                   = new uint8_t[lhs_packed_size];

#pragma omp parallel for if (m > 1)
        for (int m_start = 0; m_start < m; m_start += m_step) {
            const size_t lhs_offset = kai_get_lhs_offset_lhs_quant_pack_bf16p8x4_f32_neon(m_start, lhs_stride);
            const size_t lhs_packed_offset =
                kai_get_lhs_packed_offset_lhs_quant_pack_bf16p8x4_f32_neon(m_start, k, mr, kr, sr);
            int tile_m = (m_start + m_step <= m) ? m_step : m - m_start;

            kai_run_lhs_quant_pack_bf16p8x4_f32_neon(tile_m,
                                                     k,
                                                     mr,
                                                     kr,
                                                     sr,
                                                     0 /* m_idx_start; should stay as 0 */,
                                                     ((uint8_t*)lhs + lhs_offset),  // adjust Lhs start position
                                                     lhs_stride,
                                                     (lhs_packed + lhs_packed_offset));
        }
    } else if (params.A.type() == DataType::TYPE_FP16) {
        // lhs in fp16
        const size_t lhs_stride      = k * sizeof(float16_t);
        const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_bf16p8x4_f16_neon(m, k, mr, kr, sr);
        lhs_packed                   = new uint8_t[lhs_packed_size];

#pragma omp parallel for if (m > 1)
        for (int m_start = 0; m_start < m; m_start += m_step) {
            const size_t lhs_offset = kai_get_lhs_offset_lhs_pack_bf16p8x4_f16_neon(m_start, lhs_stride);
            const size_t lhs_packed_offset =
                kai_get_lhs_packed_offset_lhs_pack_bf16p8x4_f16_neon(m_start, k, mr, kr, sr);
            int tile_m = (m_start + m_step <= m) ? m_step : m - m_start;

            kai_run_lhs_pack_bf16p8x4_f16_neon(tile_m,
                                               k,
                                               mr,
                                               kr,
                                               sr,
                                               0 /* m_idx_start; should stay as 0 */,
                                               ((uint8_t*)lhs + lhs_offset),  // adjust Lhs start position
                                               lhs_stride,
                                               (lhs_packed + lhs_packed_offset));
        }
    } else {
        RTP_LLM_LOG_WARNING("Not supported GEMM input type %d", params.A.type());
    }

    if (data_type == DataType::TYPE_FP32) {
        // matmul out fp32
        const size_t dst_stride_row = n * sizeof(float);
        const size_t dst_stride_col = sizeof(float);

#pragma omp parallel for
        for (int n_start = 0; n_start < n; n_start += n_step) {
            size_t lhs_offset;
            size_t rhs_offset;
            size_t dst_offset =
                kai_get_dst_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(0, n_start, n * sizeof(float));
            if (params.A.type() == DataType::TYPE_FP32) {
                lhs_offset = kai_get_lhs_packed_offset_lhs_quant_pack_bf16p8x4_f32_neon(0, k, mr, kr, sr);
                rhs_offset = kai_get_rhs_packed_offset_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(n_start, k, nr, kr);
            } else {  // For input type FP16 and compute type FP32.
                lhs_offset = kai_get_lhs_packed_offset_lhs_pack_bf16p8x4_f16_neon(0, k, mr, kr, sr);
                rhs_offset = kai_get_rhs_packed_offset_rhs_pack_kxn_bf16p12x4biasf32_f16_neon(n_start, k);
            }

            const void* lhs_ptr = (const void*)((const uint8_t*)lhs_packed + lhs_offset);
            const void* rhs_ptr = (const void*)((const uint8_t*)rhs_packed + rhs_offset);
            void*       dst_ptr = (void*)((uint8_t*)dst + dst_offset);

            assert(n % n_step == 0);
            assert(n_step % n_step == 0);

            int tile_n = (n_start + n_step <= n) ? n_step : n - n_start;
            kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(m,
                                                                        tile_n,
                                                                        k,               // Dimensions
                                                                        lhs_ptr,         // LHS
                                                                        rhs_ptr,         // RHS packed
                                                                        dst_ptr,         // DST
                                                                        dst_stride_row,  // DST stride (row)
                                                                        dst_stride_col,  // DST stride (col)
                                                                        -FLT_MAX,
                                                                        FLT_MAX  // Min and max for the clamp operation
            );
        }
    } else if (data_type == DataType::TYPE_FP16) {
        // matmul out fp16

        const size_t dst_stride_row = n * sizeof(float16_t);
        const size_t dst_stride_col = sizeof(float16_t);

#pragma omp parallel for
        for (int n_start = 0; n_start < n; n_start += n_step) {
            size_t lhs_offset = kai_get_lhs_packed_offset_lhs_pack_bf16p8x4_f16_neon(0, k, mr, kr, sr);
            size_t rhs_offset = kai_get_rhs_packed_offset_rhs_pack_kxn_bf16p12x4biasf16_f16_neon(n_start, k);
            size_t dst_offset = kai_get_dst_offset_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla(
                0, n_start, n * sizeof(bfloat16_t));

            const void* lhs_ptr = (const void*)((const uint8_t*)lhs_packed + lhs_offset);
            const void* rhs_ptr = (const void*)((const uint8_t*)rhs_packed + rhs_offset);
            void*       dst_ptr = (void*)((uint8_t*)dst + dst_offset);

            assert(n % n_step == 0);
            assert(n_step % n_step == 0);

            int tile_n = (n_start + n_step <= n) ? n_step : n - n_start;
            kai_run_matmul_clamp_f16_bf16p8x4_bf16p12x4b_8x12_neon_mmla(m,
                                                                        tile_n,
                                                                        k,               // Dimensions
                                                                        lhs_ptr,         // LHS
                                                                        rhs_ptr,         // RHS packed
                                                                        dst_ptr,         // DST
                                                                        dst_stride_row,  // DST stride (row)
                                                                        dst_stride_col,  // DST stride (col)
                                                                        -FLT_MAX,
                                                                        FLT_MAX  // Min and max for the clamp operation
            );
        }
    } else {
        RTP_LLM_LOG_WARNING("Not supported GEMM output type %d", data_type);
    }

    delete[] lhs_packed;

    /* TODO
    if (m == 1) {
        // gemv
    } else {
        // gemm
    }
    */

#ifdef GEMM_DEBUG
    auto  end         = std::chrono::high_resolution_clock::now();
    float during_time = std::chrono::duration<float>(end - start).count();
    printf("gemm_kai_bf16 m,n,k %ld %ld %ld %.3f\n", m, n, k, during_time * 1000);
#endif
    return output;
}

BufferPtr ArmCpuDevice::gemm_kai_a8w4(const GemmParams& params) {
#ifdef GEMM_DEBUG
    auto start = std::chrono::high_resolution_clock::now();
#endif
    params.check();

    std::vector<size_t> Ashape;
    std::vector<size_t> Bshape;
    std::vector<size_t> Dshape;

    size_t dim;
    size_t m;
    size_t k;
    size_t n;

    Ashape = params.A.shape();
    Bshape = params.B.shape();

    dim = params.A.dim();

    if (params.transA == TransposeOperation::TRANSPOSE) {
        std::iter_swap(Ashape.end() - 1, Ashape.end() - 2);
    }

    if (params.transB == TransposeOperation::TRANSPOSE) {
        std::iter_swap(Bshape.end() - 1, Bshape.end() - 2);
    }

    m = Ashape[dim - 2];
    k = Ashape[dim - 1];
    n = Bshape[dim - 1];

    auto data_type = params.compute_type == DataType::TYPE_INVALID ? params.A.type() : params.compute_type;

    Dshape = std::vector<size_t>(Ashape.begin(), Ashape.end() - 2);
    Dshape.insert(Dshape.end(), {m, n});

    BufferPtr output;
    if (params.D) {
        output = params.D;
        RUNTIME_ASSERT_OP_ARG((data_type == params.D->type()) && (Dshape == params.D->shape()),
                              "Gemm output D shape and dtype mismatch: expected [%d][%s] but got [%s]",
                              data_type,
                              autil::StringUtil::toString(Dshape).c_str(),
                              params.D->debugString().c_str());
    } else {
        output = allocateBuffer({data_type, Dshape, AllocationType::DEVICE}, {"gemm_output"});
    }

    size_t idx_variant = 0;
    // input FP16 or output FP16 case, currently support gemv only
    if (m == 1) {
        idx_variant = 0;
    } else {
        idx_variant = 1;
    }

    // Get the packing parameters
    size_t mr;
    size_t kr;
    size_t sr;
    if (data_type == DataType::TYPE_FP32) {
        mr = fp32_ukernel_variants[idx_variant].ukernel.get_mr();
        kr = fp32_ukernel_variants[idx_variant].ukernel.get_kr();
        sr = fp32_ukernel_variants[idx_variant].ukernel.get_sr();
    } else if (data_type == DataType::TYPE_FP16) {
        mr = fp16_ukernel_variants[idx_variant].ukernel.get_mr();
        kr = fp16_ukernel_variants[idx_variant].ukernel.get_kr();
        sr = fp16_ukernel_variants[idx_variant].ukernel.get_sr();
    } else {
        RTP_LLM_LOG_WARNING("Not supported GEMM output type %d", data_type);
    }

    const size_t lhs_stride     = k * sizeof(float);
    const size_t dst_stride_col = sizeof(float);

    const size_t bl = 32;

    const size_t lhs_packed_size       = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(m, k, bl, mr, kr, sr);
    uint8_t*     lhs_packed_mtx_qs8d32 = new uint8_t[lhs_packed_size];

    uint8_t* rhs_packed_mtx_qs4c32 = (uint8_t*)params.B.data();
    float*   lhs                   = (float*)params.A.data();

    int n_step = 32;  // 32 is the best for performance
    int m_step = mr;
    // LHS packing
    if (params.A.type() == DataType::TYPE_FP32) {
#pragma omp parallel for if (m > 1)
        for (int m_start = 0; m_start < m; m_start += m_step) {
            const size_t lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32(m_start, lhs_stride);
            const size_t lhs_packed_offset =
                kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32(m_start, k, bl, mr, kr, sr);
            int tile_m = (m_start + m_step <= m) ? m_step : m - m_start;

            kai_run_lhs_quant_pack_qsi8d32p_f32(tile_m,
                                                k,
                                                bl,
                                                mr,
                                                kr,
                                                sr,
                                                0,
                                                (const float*)((uint8_t*)lhs + lhs_offset),
                                                lhs_stride,
                                                ((uint8_t*)lhs_packed_mtx_qs8d32 + lhs_packed_offset));
        }
    } else if (params.A.type() == DataType::TYPE_FP16) {
#pragma omp parallel for if (m > 1)
        for (int m_start = 0; m_start < m; m_start += m_step) {
            const size_t lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f16(m_start, k * sizeof(float16_t));
            const size_t lhs_packed_offset =
                kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f16(m_start, k, bl, mr, kr, sr);
            int tile_m = (m_start + m_step <= m) ? m_step : m - m_start;

            kai_run_lhs_quant_pack_qsi8d32p_f16(tile_m,
                                                k,
                                                bl,
                                                mr,
                                                kr,
                                                sr,
                                                0,
                                                (const float16_t*)((uint8_t*)lhs + lhs_offset),
                                                k * sizeof(float16_t),
                                                ((uint8_t*)lhs_packed_mtx_qs8d32 + lhs_packed_offset));
        }
    } else {
        RTP_LLM_LOG_WARNING("Not supported GEMM A type %d", params.A.type());
    }

    // Matmul
    if (data_type == DataType::TYPE_FP32) {
#pragma omp parallel for
        for (int n_start = 0; n_start < n; n_start += n_step) {
            const size_t dst_stride = n * sizeof(float);
            const size_t lhs_offset = fp32_ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(0, k, bl);
            const size_t rhs_offset = fp32_ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(n_start, k, bl);
            const size_t dst_offset = fp32_ukernel_variants[idx_variant].ukernel.get_dst_offset(0, n_start, dst_stride);

            const void* lhs_ptr = (const void*)((const char*)lhs_packed_mtx_qs8d32 + lhs_offset);
            const void* rhs_ptr = (const void*)((const char*)rhs_packed_mtx_qs4c32 + rhs_offset);
            float*      dst_ptr = (float*)((uint8_t*)output->data() + dst_offset);

            int tile_n = (n_start + n_step <= n) ? n_step : n - n_start;

            fp32_ukernel_variants[idx_variant].ukernel.run_matmul(m,
                                                                  tile_n,
                                                                  k,
                                                                  bl,              // Dimensions
                                                                  lhs_ptr,         // LHS packed
                                                                  rhs_ptr,         // RHS packed
                                                                  dst_ptr,         // DST
                                                                  dst_stride,      // DST stride (row)
                                                                  dst_stride_col,  // DST stride (col)
                                                                  -FLT_MAX,
                                                                  FLT_MAX  // Min and max for the clamp operation
            );
        }
    } else if (data_type == DataType::TYPE_FP16) {
#pragma omp parallel for
        for (int n_start = 0; n_start < n; n_start += n_step) {
            const size_t dst_stride = n * sizeof(float16_t);
            const size_t lhs_offset = fp16_ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(0, k, bl);
            const size_t rhs_offset = fp16_ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(n_start, k, bl);
            const size_t dst_offset = fp16_ukernel_variants[idx_variant].ukernel.get_dst_offset(0, n_start, dst_stride);

            const void* lhs_ptr = (const void*)((const char*)lhs_packed_mtx_qs8d32 + lhs_offset);
            const void* rhs_ptr = (const void*)((const char*)rhs_packed_mtx_qs4c32 + rhs_offset);
            float16_t*  dst_ptr = (float16_t*)((uint8_t*)output->data() + dst_offset);

            int tile_n = (n_start + n_step <= n) ? n_step : n - n_start;

            fp16_ukernel_variants[idx_variant].ukernel.run_matmul(m,
                                                                  tile_n,
                                                                  k,
                                                                  bl,                 // Dimensions
                                                                  lhs_ptr,            // LHS packed
                                                                  rhs_ptr,            // RHS packed
                                                                  dst_ptr,            // DST
                                                                  dst_stride,         // DST stride (row)
                                                                  sizeof(float16_t),  // DST stride (col)
                                                                  -HALF_FLT_MAX,
                                                                  HALF_FLT_MAX  // Min and max for the clamp operation
            );
        }
    }

    delete[] lhs_packed_mtx_qs8d32;

#ifdef GEMM_DEBUG
    auto  end         = std::chrono::high_resolution_clock::now();
    float during_time = std::chrono::duration<float>(end - start).count();
    printf("gemm_kai_a8w4 m,n,k %ld %ld %ld %.3f\n", m, n, k, during_time * 1000);
#endif
    return output;
}

}  // namespace rtp_llm
