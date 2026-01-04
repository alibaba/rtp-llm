// Define commonly used types.
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/reference_tensor_operation/gpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_b_scale.hpp"

#include "../rocmCKGemmWrapper.h"

namespace rtp_llm {

// Define commonly used types.
template<ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row         = ck::tensor_layout::gemm::RowMajor;
using Col         = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
// using GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

using ADataType        = ck::half_t;
using BDataType        = ck::pk_i4_t;
using BScaleDataType   = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using CDataType        = ck::half_t;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

// using LOOP_SCHED = ck::BlockGemmPipelineScheduler::Intrawave;
using ComputeType = ck::half_t;

static constexpr bool PermuteA = false;
static constexpr bool PermuteB = false;

template<int BLOCK_SIZE,
         int MBLOCK,
         int NBLOCK,
         int KBLOCK,
         int BK1,
         int WAVE_TILE_M,
         int WAVE_TILE_N,
         int WAVE_MAP_M,
         int WAVE_MAP_N,
         typename ABLOCK_TRANSFER,
         typename BBLOCK_TRANSFER,
         int BBlockSPV,
         typename CBLOCK_TRANSFER,
         int                            CBLOCK_SPV,
         ck::BlockGemmPipelineScheduler LOOP_SCHED,
         ck::BlockGemmPipelineVersion   PIPELINE_VERSION>
using DeviceInt4GemmHelper =
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<ALayout,
                                                            BLayout,
                                                            CLayout,
                                                            ADataType,
                                                            BDataType,
                                                            BScaleDataType,
                                                            CDataType,
                                                            AccDataType,
                                                            CShuffleDataType,
                                                            AElementOp,
                                                            BElementOp,
                                                            CElementOp,
                                                            GemmDefault,
                                                            BLOCK_SIZE,       // Block Size
                                                            1,                // Scale block N
                                                            128,              // Scale black K = Group size
                                                            MBLOCK,           // M per Block
                                                            NBLOCK,           // N per Block
                                                            KBLOCK,           // K per Block
                                                            8,                // AK1
                                                            BK1,              // BK1
                                                            WAVE_TILE_M,      // M per Xdl
                                                            WAVE_TILE_N,      // N per Xdl
                                                            WAVE_MAP_M,       // Mxdl per Wave
                                                            WAVE_MAP_N,       // Nxdl per Wave
                                                            ABLOCK_TRANSFER,  // A Block Start
                                                            S<1, 0, 2>,
                                                            S<1, 0, 2>,
                                                            2,
                                                            8,
                                                            8,
                                                            0,                // A Block Lds ExtraMem
                                                            BBLOCK_TRANSFER,  // B block Start
                                                            S<1, 0, 2>,
                                                            S<1, 0, 2>,
                                                            2,
                                                            BBlockSPV,  // BBlockTransferSrcScalarPerVector
                                                            BBlockSPV,  // BBlockTransferDstScalarPerVector_BK1
                                                            0,          // B Block Lds ExtraMem
                                                            1,          // CShuffleMXdlPerWavePerShuffle
                                                            1,          // CShuffleNXdlPerWavePerShuffle
                                                            CBLOCK_TRANSFER,
                                                            CBLOCK_SPV,
                                                            LOOP_SCHED,
                                                            PIPELINE_VERSION,
                                                            ComputeType,
                                                            ComputeType,
                                                            PermuteA,
                                                            PermuteB>;

template<typename DeviceInt4GemmInstance>
void int4Gemm_impl(const ckGemmParam& params) {
    // Get input information.
    auto M            = params.M;
    auto N            = params.N;
    auto K            = params.K;
    auto GroupSize    = params.Group_size;
    auto StrideA      = params.StrideA;
    auto StrideB      = params.StrideB;
    auto StrideC      = params.StrideC;
    auto StrideScaleB = (K + params.Group_size - 1) / params.Group_size;
    auto KBatch       = 1;

    // for KBatch tuning
    if (N == 29696 && K == 8192) {
        if (M == 1 || M == 16 || M == 48 || M == 64) {
            KBatch = 2;
        } else if (M == 32) {
            KBatch = 4;
        }
    } else if (N == 8192 && K == 29696) {
        if (M == 64) {
            KBatch = 2;
        } else if (M == 16 || M == 48) {
            KBatch = 4;
        } else if (M == 1 || M == 32 || M == 80 || M == 96 || M == 112 || M == 128) {
            KBatch = 8;
        }
    } else if (N == 10240 && K == 8192) {
        if (M == 16 || M == 80 || M == 112 || M == 128) {
            KBatch = 2;
        } else if (M == 1 || M == 32) {
            KBatch = 4;
        }
    } else if (N == 8192 && K == 8192) {
        if (M == 112 || M == 128) {
            KBatch = 2;
        } else if (M == 1 || M == 16) {
            KBatch = 4;
        }
    }

    // Create gemm launcher and arguments.
    auto gemm    = DeviceInt4GemmInstance{};
    auto invoker = gemm.MakeInvoker();

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    auto argument = gemm.MakeArgument(static_cast<ADataType*>(params.A_input),
                                      static_cast<BDataType*>(params.B_input),
                                      static_cast<CDataType*>(params.C_input),
                                      M,
                                      N,
                                      K,
                                      StrideA,
                                      StrideB,
                                      StrideC,
                                      StrideScaleB,
                                      static_cast<BScaleDataType*>(params.B_scales_input),
                                      KBatch,
                                      a_element_op,
                                      b_element_op,
                                      c_element_op);

    if (!gemm.IsSupportedArgument(argument)) {
        std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return;
    }

    std::size_t workspace_size = gemm.GetWorkSpaceSize(&argument);
    if (workspace_size != 0) {
        ck::DeviceMem gemm_desc_workspace(workspace_size);
        gemm.SetWorkSpacePointer(&argument, gemm_desc_workspace.GetDeviceBuffer());
    }

    invoker.Run(argument, StreamConfig{params.stream, false});
}

}  // namespace rtp_llm