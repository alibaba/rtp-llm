#include "rocmMoeWrapper.h"
#include "maga_transformer/cpp/utils/StringUtil.h"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multi_abd_xdl_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_multi_abd.hpp"
// #include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/combined_element_wise_operation.hpp"

#include "ck/utility/data_type.hpp"
#include "ck/library/utility/device_memory.hpp"

namespace fastertransformer {

static void throwCKError(const char* const file, int const line, std::string const& info = "") {
    auto error_msg = std::string("[CK][ERROR] ") + info + " Assertion fail: " + file + ":" + std::to_string(line) + " \n";
    std::printf("%s", error_msg.c_str());
    fflush(stdout);
    fflush(stderr);
    abort();
    throw std::exception();
}
#define CK_FAIL(info, ...) throwCKError(__FILE__, __LINE__, rtp_llm::fmtstr(info, ##__VA_ARGS__))

void* add_offset(void* ptr, std::size_t offset, std::size_t element_size) {
    char* char_ptr = static_cast<char*>(ptr);
    char_ptr += offset * element_size;
    return static_cast<void*>(char_ptr);
}

template<ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;
using I8   = int8_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Multiply    = ck::tensor_operation::element_wise::Multiply;

// lip: CK not support bf16 act yet...
template<typename ActT>
struct F32ActImpl {
    ActT activationOp_;
    F32ActImpl(ActT activationOp): activationOp_(activationOp) {};

    template<typename E>
    __host__ __device__ constexpr void operator()(E& e, const F32& c) const {
        using ck::type_convert;
        F32 tmp = 0;

        activationOp_(tmp, c);

        e = type_convert<E>(tmp);
    };
};
using Silu    = ck::tensor_operation::element_wise::Silu;
using Gelu    = ck::tensor_operation::element_wise::Gelu;
using Relu    = ck::tensor_operation::element_wise::Relu;
using Sigmoid = ck::tensor_operation::element_wise::Sigmoid;

template<typename InputT, typename WeightT>
uint32_t MoeRunnerImpl(const rocmMoeParams& params,
                       void *               gemm_desc_workspace,
                       void *               gemm_kernel_args_dev) {
    auto totlaM    = params.total_rows_before_expert_host[params.num_experts - 1];
    auto dtypeSize = sizeof(InputT);

    uint32_t rslt = 0;
    // Lip: designed 2 stages,
    if (isGatedActivation(params.activation_type)) {
        // start stage1. do "input GEMM (gate_W, UP_W)", then "activation"
        auto gemmParamsGate = rocmGroupGEMMParams({params.input,
                                                   params.gate_weight,
                                                   params.gate_scales,
                                                   params.gate_zeros,
                                                   params.gate_bias,
                                                   params.output_gate,
                                                   params.num_experts,
                                                   params.total_rows_before_expert_host,
                                                   params.N,
                                                   params.K,
                                                   params.stream});

        if (params.isGate_RowMajor) {
            rslt = MoeRunnerImpl_groupGEMM_caller<InputT, WeightT, true>(gemmParamsGate, params.activation_type,gemm_desc_workspace,gemm_kernel_args_dev);
        } else {
            rslt = MoeRunnerImpl_groupGEMM_caller<InputT, WeightT, false>(gemmParamsGate, params.activation_type,gemm_desc_workspace,gemm_kernel_args_dev);
        }

        auto gemmParamsUp = rocmGroupGEMMParams({params.input,
                                                 params.up_weight,
                                                 params.up_scales,
                                                 params.up_zeros,
                                                 params.up_bias,
                                                 add_offset(params.output_gate, totlaM * params.N, dtypeSize),
                                                 params.num_experts,
                                                 params.total_rows_before_expert_host,
                                                 params.N,
                                                 params.K,
                                                 params.stream});

        if (params.isUp_RowMajor) {
            rslt = MoeRunnerImpl_groupGEMM_caller<InputT, WeightT, true>(gemmParamsUp, ActivationType::Identity,gemm_desc_workspace,gemm_kernel_args_dev);
        } else {
            rslt = MoeRunnerImpl_groupGEMM_caller<InputT, WeightT, false>(gemmParamsUp, ActivationType::Identity,gemm_desc_workspace,gemm_kernel_args_dev);
        }
        hipStreamSynchronize(params.stream);

        // start stage2. do "gate_O*UP_O, then GEMM with down_W"
        if (params.isDown_RowMajor) {
            rslt += MoeRunnerImpl_stage2<InputT, WeightT, true>(params,gemm_desc_workspace,gemm_kernel_args_dev);
        } else {
            rslt += MoeRunnerImpl_stage2<InputT, WeightT, false>(params,gemm_desc_workspace,gemm_kernel_args_dev);
        }
    } else {
        // start stage1. do "input GEMM with UP_W", then "activation"
        auto gemmParamsUp = rocmGroupGEMMParams({params.input,
                                                 params.up_weight,
                                                 params.up_scales,
                                                 params.up_zeros,
                                                 params.up_bias,
                                                 params.output_gate,
                                                 params.num_experts,
                                                 params.total_rows_before_expert_host,
                                                 params.N,
                                                 params.K,
                                                 params.stream});

        if (params.isUp_RowMajor) {
            rslt = MoeRunnerImpl_groupGEMM_caller<InputT, WeightT, true>(gemmParamsUp, params.activation_type,gemm_desc_workspace,gemm_kernel_args_dev);
        } else {
            rslt = MoeRunnerImpl_groupGEMM_caller<InputT, WeightT, false>(gemmParamsUp, params.activation_type,gemm_desc_workspace,gemm_kernel_args_dev);
        }

        // start stage2. do "GEMM with down_W"
        auto gemmParamsDown = rocmGroupGEMMParams({params.output_gate,
                                                   params.down_weight,
                                                   params.down_scales,
                                                   params.down_zeros,
                                                   params.down_bias,
                                                   params.input,
                                                   params.num_experts,
                                                   params.total_rows_before_expert_host,
                                                   params.K,
                                                   params.N,
                                                   params.stream});
        if (params.isDown_RowMajor) {
            rslt = MoeRunnerImpl_groupGEMM_caller<InputT, WeightT, true>(gemmParamsDown, ActivationType::Identity,gemm_desc_workspace,gemm_kernel_args_dev);
        } else {
            rslt = MoeRunnerImpl_groupGEMM_caller<InputT, WeightT, false>(gemmParamsDown, ActivationType::Identity,gemm_desc_workspace,gemm_kernel_args_dev);
        }
    }
    
    return rslt;
}

template<typename InputT, typename WeightT, bool isWeightRowMajor>
uint32_t MoeRunnerImpl_groupGEMM_caller(const rocmGroupGEMMParams& params, 
                                          ActivationType       activation_type,
                                          void *               gemm_desc_workspace,
                                          void *               gemm_kernel_args_dev) {
    uint32_t rslt = 0;                                            
    switch (activation_type) {
        case ActivationType::Gelu:
        case ActivationType::Geglu:
            rslt = MoeRunnerImpl_groupGEMM<InputT, WeightT, Gelu, isWeightRowMajor>(params,gemm_desc_workspace,gemm_kernel_args_dev);
            break;
        case ActivationType::Relu:
            rslt = MoeRunnerImpl_groupGEMM<InputT, WeightT, Relu, isWeightRowMajor>(params,gemm_desc_workspace,gemm_kernel_args_dev);
            break;
        case ActivationType::Silu:
        case ActivationType::Swiglu:
            rslt = MoeRunnerImpl_groupGEMM<InputT, WeightT, Silu, isWeightRowMajor>(params,gemm_desc_workspace,gemm_kernel_args_dev);
            break;
        case ActivationType::Sigmoid:
            rslt = MoeRunnerImpl_groupGEMM<InputT, WeightT, Sigmoid, isWeightRowMajor>(params,gemm_desc_workspace,gemm_kernel_args_dev);
            break;
        case ActivationType::Identity:
            rslt = MoeRunnerImpl_groupGEMM<InputT, WeightT, PassThrough, isWeightRowMajor>(params,gemm_desc_workspace,gemm_kernel_args_dev);
            break;
        case ActivationType::GeluNoneApproximate:
        case ActivationType::GeGluNoneApproximate:
            rslt = MoeRunnerImpl_groupGEMM<InputT, WeightT, Gelu, isWeightRowMajor>(params,gemm_desc_workspace,gemm_kernel_args_dev);
            break;
        default:
            CK_FAIL("not support activation type");
    }
    return rslt;
}

template<typename InputT, typename WeightT, typename ActiveT, bool isWeightRowMajor_>
uint32_t MoeRunnerImpl_groupGEMM(const rocmGroupGEMMParams& params,
    void * gemm_desc_workspace,
    void * gemm_kernel_args_dev) {
    int K = params.K;
    int N = params.N;

    using ADataType        = InputT;
    using BDataType        = WeightT;
    using AccDataType      = F32;
    using CShuffleDataType = F32;
    using DsDataType       = ck::Tuple<>;
    using EDataType        = InputT;

    using ALayout                    = Row;
    static constexpr bool isRowMajor = isWeightRowMajor_;
    using BLayout                    = std::conditional_t<isRowMajor, Row, Col>;
    using BAccessOrder               = std::conditional_t<isRowMajor, S<0, 2, 1>, S<1, 0, 2>>;
    const auto BSrcVectorDim         = isRowMajor ? 1 : 2;
    const auto BSrcScalar            = isRowMajor ? 4 : 8;
    auto       B_Srtide              = isRowMajor ? N : K;
    using DsLayout                   = ck::Tuple<>;
    using ELayout                    = Row;

    using AElementOp   = PassThrough;
    using BElementOp   = PassThrough;
    using CDEElementOp = F32ActImpl<ActiveT>;
    // using CDEElementOp = ActiveT;

    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

    auto totlaM = params.total_rows_before_expert_host[params.num_experts - 1];

    using DeviceGemmInstance = ck::tensor_operation::device::DeviceGroupedGemm_Xdl
        // clang-format off
//######| ALayout| BLayout| DsLayout| ELayout|     AData|     BData|     AccData|         CShuffle|     DsData|     EData|           A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
//######|        |        |         |        |      Type|      Type|        Type|         DataType|       Type|      Type| Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
//######|        |        |         |        |          |          |            |                 |           |          |   Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
//######|        |        |         |        |          |          |            |                 |           |          |            |            |             |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        < ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   128,    64,   128,    32,   8,   8,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,   BAccessOrder,   BAccessOrder, BSrcVectorDim,     BSrcScalar,              8,         1,           1,           1,               S<1, 16, 1, 8>,              8>;

    auto gemmRunner=DeviceGemmInstance{};

    // GEMM shape
    std::vector<ck::tensor_operation::device::GemmDesc> gemm_descs;

    std::vector<const void*> p_a, p_b;
    std::vector<void*>       p_c;

    gemm_descs.reserve(params.num_experts);

    auto dtypeSize  = sizeof(InputT);
    auto wtypeSize  = sizeof(WeightT);
    int  startRowID = 0;
    int  rowID      = 0;

    for (int i = 0; i < params.num_experts; i++) {
        rowID = params.total_rows_before_expert_host[i];

        gemm_descs.push_back({rowID - startRowID,  // M
                              N,           // N
                              K,           // K
                              (int)K,      // stride A
                              (int)B_Srtide,      // stride B
                              (int)N,      // stride C
                              {}});
        p_a.push_back(add_offset(params.input, startRowID * K, dtypeSize));
        p_b.push_back(add_offset(params.B_weight, i * K * N, wtypeSize));
        p_c.push_back(add_offset(params.output, startRowID * N, dtypeSize));

        startRowID = rowID;
    }

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CDEElementOp{ActiveT{}};
    // auto c_element_op = CDEElementOp{};

    auto invoker = gemmRunner.MakeInvoker();

    std::vector<std::array<const void*, 0>> p_Ds = {};

    // do GEMM
    auto argument = gemmRunner.MakeArgument(p_a, p_b, p_Ds, p_c, gemm_descs, a_element_op, b_element_op, c_element_op);

    if (!gemmRunner.IsSupportedArgument(argument)) {
        CK_FAIL("device_gemm with the specified compilation parameters does not support this GEMM problem");
    }

    if(gemm_desc_workspace == nullptr)
    {
        return gemmRunner.GetWorkSpaceSize(&argument);
    }

    //DeviceMem gemm_desc_workspace(gemmRunner.GetWorkSpaceSize(&argument));
    gemmRunner.SetWorkSpacePointer(&argument, gemm_desc_workspace);

    invoker.Run(argument, StreamConfig{params.stream, false});

    return 0;
}

template<typename InputT, typename WeightT, bool isWeightRowMajor_>
uint32_t MoeRunnerImpl_stage2(const rocmMoeParams& params,
    void * gemm_desc_workspace,
    void * gemm_kernel_args_dev){
    int N = params.K;
    int K = params.N;
    // stage 2, (Gate_O*UP_O), gemm down_W
    using AsDataType       = ck::Tuple<InputT, InputT>;
    using BsDataType       = ck::Tuple<WeightT>;
    using AccDataType      = F32;
    using CShuffleDataType = InputT;
    using DsDataType       = ck::Tuple<>;
    using EDataType        = InputT;

    using AsLayout = ck::Tuple<Row, Row>;
    static constexpr bool isRowMajor = isWeightRowMajor_;
    using BLayout                    = std::conditional_t<isRowMajor, Row, Col>;
    using BAccessOrder               = std::conditional_t<isRowMajor, S<0, 2, 1>, S<1, 0, 2>>;
    const auto BSrcVectorDim         = isRowMajor ? 1 : 2;
    const auto BSrcScalar            = isRowMajor ? 2 : 8;
    auto       B_Srtide              = isRowMajor ? N : K;
    using BsLayout = ck::Tuple<BLayout>;
    using DsLayout = ck::Tuple<>;
    using ELayout  = Row;

    using AsElementOp = ck::tensor_operation::element_wise::BinaryWithUnaryCombinedOp<Multiply, PassThrough, PassThrough>;
    using BElementOp   = PassThrough;
    using CDEElementOp = PassThrough;

    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;
    auto totlaM = params.total_rows_before_expert_host[params.num_experts - 1];

    using DeviceGemmInstance = ck::tensor_operation::device::DeviceGroupedGemm_Xdl_Multi_ABD_Fixed_NK
        // clang-format off
//######|  ALayout|  BLayout| DsLayout| ELayout|      AData|      BData|     AccData|         CShuffle|     DsData|     EData|            A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
//######|         |         |         |        |       Type|       Type|        Type|         DataType|       Type|      Type|  Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
//######|         |         |         |        |           |           |            |                 |           |          |    Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
//######|         |         |         |        |           |           |            |                 |           |          |             |            |             |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        < AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AsElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,    64,   128,    32,   8,   8,   32,   32,    1,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,   BAccessOrder,   BAccessOrder, BSrcVectorDim,     BSrcScalar,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8, InputT>;

    auto gemmRunner=DeviceGemmInstance{};

    // GEMM shape
    std::vector<ck::tensor_operation::device::GemmMultiABDDesc> gemm_descs;

    gemm_descs.reserve(params.num_experts);

    auto dtypeSize  = sizeof(InputT);
    auto wtypeSize  = sizeof(WeightT);
    int  startRowID = 0;
    int  rowID      = 0;

    constexpr ck::index_t NumATensor = 2;
    constexpr ck::index_t NumBTensor = 1;
    constexpr ck::index_t NumDTensor = 0;
    using GroupedGemmKernelArgument = ck::tensor_operation::device::GroupedGemmMultiABDKernelArgument<NumATensor, NumBTensor, NumDTensor>;
    std::vector<GroupedGemmKernelArgument> grouped_gemm_kernel_args_;
    grouped_gemm_kernel_args_.reserve(params.num_experts);
    int  M = 0;

    for (int i = 0; i < params.num_experts; i++) {
        rowID = params.total_rows_before_expert_host[i];
        M = rowID - startRowID;
        if (M==0)
        {
            continue;
        }

        gemm_descs.push_back({totlaM, N, K, {1,1}, {1}, {}, 1});

        grouped_gemm_kernel_args_.push_back(
            {std::array<const void*, NumATensor>{add_offset(params.output_gate, startRowID*K, dtypeSize),
                                                 add_offset(params.output_gate, totlaM*K+startRowID*K, dtypeSize)},
             std::array<const void*, NumBTensor>{add_offset(params.down_weight, i * K*N, wtypeSize)},
             std::array<const void*, NumDTensor>{},
             add_offset(params.input, startRowID*N, dtypeSize),//reuse input as output
             M, //M
             N, //N
             K, //K
             std::array<ck::index_t, NumATensor>{(int)K, (int)K},
             std::array<ck::index_t, NumBTensor>{(int)B_Srtide},
             std::array<ck::index_t, NumDTensor>{},
             (int)N});

        startRowID = rowID;
    }

    auto a_element_op = AsElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CDEElementOp{};

    auto invoker = gemmRunner.MakeInvoker();

    std::vector<std::array<const void*, NumATensor>> p_As = {};
    std::vector<std::array<const void*, NumBTensor>> p_Bs = {};
    std::vector<std::array<const void*, NumDTensor>> p_Ds = {};
    std::vector<void*> p_Cs                               = {};
    auto argument = gemmRunner.MakeArgument(p_As, p_Bs, p_Ds, p_Cs, gemm_descs);

    if (!gemmRunner.IsSupportedArgument(argument)) {
        CK_FAIL("device_gemm with the specified compilation parameters does not support this GEMM problem");
    }

    if(gemm_desc_workspace == nullptr || gemm_kernel_args_dev == nullptr)
    {
        uint32_t malloc_size = gemmRunner.GetWorkSpaceSize(&argument);
        malloc_size += gemmRunner.GetDeviceKernelArgSize(&argument);
        return malloc_size;
    }

    //DeviceMem gemm_desc_workspace(gemmRunner.GetWorkSpaceSize(&argument));
    gemmRunner.SetWorkSpacePointer(&argument, gemm_desc_workspace);

    //DeviceMem gemm_kernel_args_dev(gemmRunner.GetDeviceKernelArgSize(&argument));
    hip_check_error(hipMemcpy(gemm_kernel_args_dev,
                            grouped_gemm_kernel_args_.data(),
                            gemmRunner.GetDeviceKernelArgSize(&argument),
                            hipMemcpyHostToDevice));

    gemmRunner.SetDeviceKernelArgs(argument, gemm_kernel_args_dev);
    gemmRunner.SetKBatch(argument, 1);
    gemmRunner.SetElementwiseOps(argument, a_element_op, b_element_op, c_element_op);

    invoker.Run(argument, StreamConfig{params.stream, false});
    return 0;
}


uint32_t rocmMoeWrapper::runCKMoe(const rocmMoeParams& params,
                              DataType           dtype,
                              DataType           wtype,
                              void *             gemm_desc_workspace,
                              void *             gemm_kernel_args_dev) {
    uint32_t rslt = 0;
    if (dtype == DataType::TYPE_FP16 && wtype == DataType::TYPE_FP16) {
        using InputT = F16;
        using WeightT= F16;
        rslt = MoeRunnerImpl<InputT, WeightT>(params, gemm_desc_workspace, gemm_kernel_args_dev);
    } else if (dtype == DataType::TYPE_BF16 && wtype == DataType::TYPE_BF16) {
        using InputT = BF16;
        using WeightT  = BF16;
        rslt = MoeRunnerImpl<InputT, WeightT>(params, gemm_desc_workspace, gemm_kernel_args_dev);
    } else if (dtype == DataType::TYPE_FP32 && wtype == DataType::TYPE_FP32) {
        using InputT = F32;
        using WeightT = F32;
        rslt = MoeRunnerImpl<InputT, WeightT>(params, gemm_desc_workspace, gemm_kernel_args_dev);
    // TODO: int8/int4 dequantization
    // } else if (dtype == DataType::TYPE_FP16 && wtype == DataType::TYPE_QINT8) {
    //     using InputT = F16;
    //     using WeightT = I8;
    //     MoeRunnerImpl<InputT, WeightT>(params);
    // } else if (dtype == DataType::TYPE_BF16 && wtype == DataType::TYPE_QINT8) {
    //     using InputT = BF16;
    //     using WeightT = I8;
    //     MoeRunnerImpl<InputT, WeightT>(params);
    } else {
        CK_FAIL("input type %d and weights type %d not supported by CK", dtype, wtype);
    }
    return rslt;
}

}  // namespace fastertransformer