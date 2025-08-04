#include "rtp_llm/models_py/bindings/rocm/Gemm.h"
#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/rocm/hip_utils.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmAllocator.h"
#include "rtp_llm/cpp/rocm/hipblasMMWrapper.h"
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"

// void gemm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, int64_t hip_stream){
//     CHECK_INPUT(input);
//     CHECK_INPUT(weight);
//     auto device = input.device();
//     CHECK_EQ(weight.device(), device);
//     CHECK_DIM(2, input);
//     CHECK_DIM(2, weight);
//     CHECK_EQ(input.size(1), weight.size(0));
//     CHECK_EQ(input.size(0), output.size(0));
//     CHECK_EQ(weight.size(1), output.size(1));
//     int m = input.size(0);
//     int n = weight.size(1);
//     int k = input.size(1);
//     float alpha = 1.0f;
//     float beta  = 0.0f;

//     hipblasHandle_t hipblas_handle;
//     hipblasCreate(&hipblas_handle);
//     //hipStream_t stream = reinterpret_cast<hipStream_t>(hip_stream);

//     DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
//         hipblasDatatype_t dtype = (std::is_same<c_type, __half>::value) ? HIPBLAS_R_16F : HIPBLAS_R_16B;

//         hipblasGemmEx(hipblas_handle,
//                       HIPBLAS_OP_N,                             // weight矩阵不转置
//                       HIPBLAS_OP_N,                             // input矩阵不转置
//                       n,                                        // 矩阵维度
//                       m,                                        // 矩阵维度
//                       k,                                        // 矩阵维度
//                       &alpha,                                   // alpha值
//                       static_cast<c_type*>(weight.data_ptr()),  // weight矩阵
//                       dtype,                                    // weight的数据类型
//                       n,                                        // weight的leading dimension
//                       static_cast<c_type*>(input.data_ptr()),   // input矩阵
//                       dtype,                                    // input的数据类型
//                       k,                                        // input的leading dimension
//                       &beta,                                    // beta值
//                       static_cast<c_type*>(output.data_ptr()),  // output矩阵
//                       dtype,                                    // output的数据类型
//                       n,                                        // output的leading dimension
//                       HIPBLAS_R_32F,                            // 计算类型
//                       HIPBLAS_GEMM_DEFAULT);                    // 算法选择

//         hipblasDestroy(hipblas_handle);
//         return true;
//     });
// }

// void gemm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, int64_t hip_stream){
//     CHECK_INPUT(input);
//     CHECK_INPUT(weight);
//     auto device = input.device();
//     CHECK_EQ(weight.device(), device);
//     CHECK_DIM(2, input);
//     CHECK_DIM(2, weight);
//     CHECK_EQ(input.size(1), weight.size(0));
//     CHECK_EQ(input.size(0), output.size(0));
//     CHECK_EQ(weight.size(1), output.size(1));
//     int m = input.size(0);
//     int n = weight.size(1);
//     int k = input.size(1);
//     float alpha = 1.0f;
//     float beta  = 0.0f;
//     hipblasOperation_t transa = HIPBLAS_OP_N;
//     hipblasOperation_t transb = HIPBLAS_OP_N;
//     auto allocator_ptr = new rtp_llm::Allocator<rtp_llm::AllocatorType::ROCM>();
//     void* workSpace = allocator_ptr->malloc(HIPBLAS_WORKSPACE_SIZE);
//     size_t workspaceSize = HIPBLAS_WORKSPACE_SIZE;
//     hipblasHandle_t hipblas_handle;
//     hipblasCreate(&hipblas_handle);
//     hipStream_t stream = reinterpret_cast<hipStream_t>(hip_stream);

//     DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
//         hipDataType hdtype = (std::is_same<c_type, __half>::value) ? HIP_R_16F : HIP_R_16BF;

//         // Step 1: 创建矩阵描述符
//         hipblasLtMatrixLayout_t ADesc, BDesc, CDesc;
//         hipblasLtMatrixLayoutCreate(&ADesc, hdtype, k, m, k);
//         hipblasLtMatrixLayoutCreate(&BDesc, hdtype, n, k, n);
//         hipblasLtMatrixLayoutCreate(&CDesc, hdtype, n, m, n);

//         // Step 2: 创建矩阵乘法描述符
//         hipblasLtMatmulDesc_t matmul;
//         hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F);
//         hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(int32_t));
//         hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(int32_t));

//         // Step 3: 查询最优算法
//         hipblasLtMatmulHeuristicResult_t heuristicResult;
//         hipblasLtMatmulPreference_t blasLtPrefer;
//         hipblasLtMatmulPreferenceCreate(&blasLtPrefer);
//         hipblasLtMatmulPreferenceSetAttribute(blasLtPrefer,
//                                               HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
//                                               &workspaceSize,
//                                               sizeof(workspaceSize));
//         int returnedAlgoCount = 0;
//         hipblasLtMatmulAlgoGetHeuristic(
//             hipblas_handle, matmul, ADesc, BDesc, CDesc, CDesc,
//             blasLtPrefer, 1, &heuristicResult, &returnedAlgoCount);

//         //Step 4: 执行矩阵乘法
//         if (returnedAlgoCount > 0) {
//             hipblasLtMatmul(hipblas_handle,
//                             matmul,
//                             &alpha,
//                             static_cast<c_type*>(weight.data_ptr()),
//                             BDesc,
//                             static_cast<c_type*>(input.data_ptr()),
//                             ADesc,
//                             &beta,
//                             static_cast<c_type*>(output.data_ptr()),
//                             CDesc,
//                             static_cast<c_type*>(output.data_ptr()),
//                             CDesc,
//                             &heuristicResult.algo,
//                             workSpace,
//                             workspaceSize,
//                             stream);
//          }else{
//             hipblasDatatype_t dtype = (std::is_same<c_type, __half>::value) ? HIPBLAS_R_16F : HIPBLAS_R_16B;
//             hipblasGemmEx(hipblas_handle,
//                 HIPBLAS_OP_N,                             // weight矩阵不转置
//                 HIPBLAS_OP_N,                             // input矩阵不转置
//                 n,                                        // 矩阵维度
//                 m,                                        // 矩阵维度
//                 k,                                        // 矩阵维度
//                 &alpha,                                   // alpha值
//                 static_cast<c_type*>(weight.data_ptr()),  // weight矩阵
//                 dtype,                                    // weight的数据类型
//                 n,                                        // weight的leading dimension
//                 static_cast<c_type*>(input.data_ptr()),   // input矩阵
//                 dtype,                                    // input的数据类型
//                 k,                                        // input的leading dimension
//                 &beta,                                    // beta值
//                 static_cast<c_type*>(output.data_ptr()),  // output矩阵
//                 dtype,                                    // output的数据类型
//                 n,                                        // output的leading dimension
//                 HIPBLAS_R_32F,                            // 计算类型
//                 HIPBLAS_GEMM_DEFAULT);                    // 算法选择
//         }

//         // Step 5: 释放资源
//         hipblasLtMatrixLayoutDestroy(ADesc);
//         hipblasLtMatrixLayoutDestroy(BDesc);
//         hipblasLtMatrixLayoutDestroy(CDesc);
//         hipblasLtMatmulDescDestroy(matmul);
//         hipblasLtMatmulPreferenceDestroy(blasLtPrefer);
//         allocator_ptr->free((void**)(&workSpace));
//         hipblasDestroy(hipblas_handle);
//         return true;
//     });
// }

hipblasHandle_t   g_hipblas_handle   = nullptr;
hipblasLtHandle_t g_hipblaslt_handle = nullptr;
// hipStream_t       g_stream           = nullptr;

// 全局分配器和包装器
rtp_llm::Allocator<rtp_llm::AllocatorType::ROCM>* g_allocator = nullptr;
std::unique_ptr<rtp_llm::rocm::hipblasMMWrapper>  g_hipblas_mm_wrapper;

// 初始化标志
bool g_initialized = false;

// 初始化函数
void initialize_global_resources() {
    if (g_initialized)
        return;

    // 创建 HIPBLAS 句柄
    hipblasCreate(&g_hipblas_handle);
    hipblasLtCreate(&g_hipblaslt_handle);
    hipStream_t stream = at::hip::getCurrentHIPStream().stream();

    // 初始化分配器
    g_allocator = new rtp_llm::Allocator<rtp_llm::AllocatorType::ROCM>();

    // 初始化 hipblasMMWrapper
    rtp_llm::HWKernelConfig hw_kernel_config;
    g_hipblas_mm_wrapper = std::make_unique<rtp_llm::rocm::hipblasMMWrapper>(
        g_hipblas_handle, g_hipblaslt_handle, stream, g_allocator, hw_kernel_config);

    g_initialized = true;
}

void gemm(at::Tensor& output, at::Tensor& input, at::Tensor& weight) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    auto device = input.device();
    CHECK_EQ(weight.device(), device);
    CHECK_DIM(2, input);
    CHECK_DIM(2, weight);
    CHECK_EQ(input.size(1), weight.size(0));
    CHECK_EQ(input.size(0), output.size(0));
    CHECK_EQ(weight.size(1), output.size(1));
    const int   m     = input.size(0);
    const int   n     = weight.size(1);
    const int   k     = input.size(1);
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    initialize_global_resources();  // 仅首次调用时初始化

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
        hipDataType dtype = (std::is_same<c_type, __half>::value) ? HIP_R_16F : HIP_R_16BF;

        g_hipblas_mm_wrapper->setGemmConfig(dtype, dtype, dtype, HIP_R_32F);
        hipStream_t stream = at::hip::getCurrentHIPStream().stream();
        g_hipblas_mm_wrapper->setStream(stream);
        g_hipblas_mm_wrapper->Gemm(HIPBLAS_OP_N,
                                   HIPBLAS_OP_N,
                                   n,
                                   m,
                                   k,
                                   static_cast<c_type*>(weight.data_ptr()),
                                   n,
                                   static_cast<c_type*>(input.data_ptr()),
                                   k,
                                   static_cast<c_type*>(output.data_ptr()),
                                   n,
                                   alpha,
                                   beta);
        return true;
    });
}