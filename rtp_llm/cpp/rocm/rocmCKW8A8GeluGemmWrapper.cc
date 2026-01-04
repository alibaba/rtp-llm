#include "rocmCKW8A8GeluGemmWrapper.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"


namespace rtp_llm {

template<ck::index_t... Is>
using S = ck::Sequence<Is...>;

using I8  = int8_t;
using I32 = int;
using F16 = ck::half_t;
using FP8 = ck::f8_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using FastGelu    = ck::tensor_operation::element_wise::FastGelu;

// w8a8 gelu gemm D0 D1

struct MultiplyMultiplyFastGelu
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, float, float, float>(
        ck::half_t& e, const float& c, const float& d0, const float& d1) const
    {
        const float x0_f = c * d0 * d1;
        ck::half_t gelu_x = 0;
        FastGelu{}.template operator()<ck::half_t, float>(gelu_x, x0_f);
        e = gelu_x;
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, int, float, float>(
        ck::half_t& e, const int& c, const float& d0, const float& d1) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * ck::type_convert<float>(d0) * ck::type_convert<float>(d1);
        ck::half_t gelu_x = 0;
        FastGelu{}.template operator()<ck::half_t, float>(gelu_x, x0_f);
        e = gelu_x;
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, int, ck::half_t, ck::half_t>(
        ck::half_t& e, const int& c, const ck::half_t& d0, const ck::half_t& d1) const
    {
        const ck::half_t x0_f = ck::type_convert<ck::half_t>(c) * d0 * d1;
        ck::half_t gelu_x = 0;
        FastGelu{}.template operator()<ck::half_t, ck::half_t>(gelu_x, x0_f);
        e = gelu_x;
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, ck::half_t, ck::half_t, ck::half_t>(
        ck::half_t& e, const ck::half_t& c, const ck::half_t& d0, const ck::half_t& d1) const
    {
        const ck::half_t x0_f = c * d0 * d1;
        ck::half_t gelu_x = 0;
        FastGelu{}.template operator()<ck::half_t, ck::half_t>(gelu_x, x0_f);
        e = gelu_x;
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::bhalf_t, int, float, float>(
        ck::bhalf_t& e, const int& c, const float& d0, const float& d1) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * ck::type_convert<float>(d0) * ck::type_convert<float>(d1);
        ck::bhalf_t gelu_x = 0;
        FastGelu{}.template operator()<ck::bhalf_t, float>(gelu_x, x0_f);
        e = gelu_x;
    }
};

void CKGemmW8A8GeluImpl(const ckW8A8GemmParam& params)
{
    using A0DataType       = I8;
    using B0DataType       = I8;
    using AccDataType      = I32;
    using CShuffleDataType = I32;
    using D0DataType       = F32;
    using D1DataType       = F32;
    using DsDataType       = ck::Tuple<D0DataType, D1DataType>;
    using EDataType        = F16;

    using A0Layout = Row;
    using B0Layout = Col;
    using D0Layout = Row;
    using D1Layout = Col;
    using DsLayout = ck::Tuple<D0Layout, D1Layout>;
    using ELayout  = Row;

    using AElementOp   = PassThrough;
    using BElementOp   = PassThrough;
    using CDEElementOp = MultiplyMultiplyFastGelu;

    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNPadding;

    using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3
    // clang-format off
///######|  ALayout|  BLayout| DsLayout| ELayout|      AData|      BData|     DsData|     EData|     AccData|         CShuffle|           A|           B|          CDE|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
///######|         |         |         |        |       Type|       Type|       Type|      Type|        Type|         DataType| Elementwise| Elementwise|  Elementwise| Spacialization|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
///######|         |         |         |        |           |           |           |          |            |                 |   Operation|   Operation|    Operation|               |      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
///######|         |         |         |        |           |           |           |          |            |                 |            |            |             |               |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |    S<C, D0, D1>|
///###### RRR
      ///<      Row,      Row, DsLayout, ELayout, A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   256,   256,   128,    64,  16,   4,  32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<16, 16, 1>,    S<0, 2, 1>,     S<0, 2, 1>,             1,               8,              4,          0,          1,           1,               S<1, 32, 1, 8>,      S<8, 8, 1>,  ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1, I8>;
///###### RCR
         <      Row,      Col, DsLayout, ELayout, A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   256,   256,   128,    64,  16,  16,  32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,     S<4, 64, 1>,    S<1, 0, 2>,     S<1, 0, 2>,             2,              16,             16,          0,          1,           1,               S<1, 32, 1, 8>,      S<8, 8, 1>,  ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1, I8>;

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DsDataType::Size();

    constexpr auto I0 = ck::Number<0>{};

    auto gemm = DeviceOpInstance{};
    auto invoker = gemm.MakeInvoker();
    auto argument =
        gemm.MakeArgument(static_cast<A0DataType*>(params.A_input),
                               static_cast<B0DataType*>(params.B_input),
                               std::array<const void*, NumDTensor>{params.D0_input,
                                                                   params.D1_input},
                               static_cast<EDataType*>(params.E_input),
                               params.M,
                               params.N,
                               params.K,
                               params.StrideA,
                               params.StrideB,
                               std::array<ck::index_t, NumDTensor>{I0, I0},
                               params.StrideE,
                               1 , // KBatch
                               a_element_op,
                               b_element_op,
                               cde_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;
        return;
    }
    std::size_t workspace_size = gemm.GetWorkSpaceSize(&argument);
    if(workspace_size!=0)
    {
        ck::DeviceMem gemm_desc_workspace(workspace_size);
        gemm.SetWorkSpacePointer(&argument, gemm_desc_workspace.GetDeviceBuffer());
    }
    invoker.Run(argument, StreamConfig{params.stream, false});
}

void rocmCKW8A8GeluGemmWrapper::runCKW8A8GeluGemm(const ckW8A8GemmParam& ckParams, DataType ADtype, DataType BDtype)
{
    if(ADtype==DataType::TYPE_QINT8 && BDtype==DataType::TYPE_QINT8)
    {
        CKGemmW8A8GeluImpl(ckParams);
    } else {
        RTP_LLM_LOG_ERROR("input A type: %d and B type: %d are not supported W8A8 GEMM by CK lib", ADtype, BDtype);
    }
};
}  // namespace rtp_llm