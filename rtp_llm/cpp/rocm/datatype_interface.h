#pragma once
#include <hipblaslt/hipblaslt.h>

#include <map>

union computeTypeInterface
{
    float         f32;
    double        f64;
    hipblasLtHalf f16;
    int32_t       i32;
};

template <typename T>
constexpr auto hipblaslt_type2datatype()
{
    if(std::is_same<T, hipblasLtHalf>{})
        return HIP_R_16F;
    if(std::is_same<T, hip_bfloat16>{})
        return HIP_R_16BF;
    if(std::is_same<T, float>{})
        return HIP_R_32F;
    if(std::is_same<T, double>{})
        return HIP_R_64F;
    if(std::is_same<T, hipblaslt_f8_fnuz>{})
        return HIP_R_8F_E4M3_FNUZ;
    if(std::is_same<T, hipblaslt_bf8_fnuz>{})
        return HIP_R_8F_E5M2_FNUZ;
#ifdef ROCM_USE_FLOAT8
    if(std::is_same<T, hipblaslt_f8>{})
        return HIP_R_8F_E4M3;
    if(std::is_same<T, hipblaslt_bf8>{})
        return HIP_R_8F_E5M2;
#endif
    if(std::is_same<T, int32_t>{})
        return HIP_R_32I;
    if(std::is_same<T, hipblasLtInt8>{})
        return HIP_R_8I;

    return HIP_R_16F; // testing purposes we default to f32 ex
}

inline hipDataType computeTypeToRealDataType(hipblasComputeType_t ctype)
{
    static const std::map<hipblasComputeType_t, hipDataType> ctypeMap{
        {HIPBLAS_COMPUTE_16F, HIP_R_16F},
        {HIPBLAS_COMPUTE_16F_PEDANTIC, HIP_R_16F},
        {HIPBLAS_COMPUTE_32F, HIP_R_32F},
        {HIPBLAS_COMPUTE_32F_PEDANTIC, HIP_R_32F},
        {HIPBLAS_COMPUTE_32F_FAST_16F, HIP_R_32F},
        {HIPBLAS_COMPUTE_32F_FAST_16BF, HIP_R_32F},
        {HIPBLAS_COMPUTE_32F_FAST_TF32, HIP_R_32F},
        {HIPBLAS_COMPUTE_64F, HIP_R_64F},
        {HIPBLAS_COMPUTE_64F_PEDANTIC, HIP_R_64F},
        {HIPBLAS_COMPUTE_32I, HIP_R_32I},
        {HIPBLAS_COMPUTE_32I_PEDANTIC, HIP_R_32I}};

    return ctypeMap.at(ctype);
}

inline std::size_t realDataTypeSize(hipDataType dtype)
{
    // These types were not defined in older versions of ROCm, so need to be handled specially here.
    auto const dtype_int = static_cast<int>(dtype);
    if(dtype_int == HIP_R_4F_E2M1_EXT || dtype_int == HIP_R_6F_E2M3_EXT
       || dtype_int == HIP_R_6F_E3M2_EXT)
    {
        return 1;
    }

    static const std::map<hipDataType, std::size_t> dtypeMap{
        {HIP_R_32F, 4},
        {HIP_R_64F, 8},
        {HIP_R_16F, 2},
        {HIP_R_8I, 1},
        {HIP_R_8U, 1},
        {HIP_R_32I, 4},
        {HIP_R_32U, 4},
        {HIP_R_16BF, 2},
        {HIP_R_4I, 1},
        {HIP_R_4U, 1},
        {HIP_R_16I, 2},
        {HIP_R_16U, 2},
        {HIP_R_64I, 8},
        {HIP_R_64U, 8},
        {HIP_R_8F_E4M3_FNUZ, 1},
        {HIP_R_8F_E5M2_FNUZ, 1},
#ifdef ROCM_USE_FLOAT8
        {HIP_R_8F_E4M3, 1},
        {HIP_R_8F_E5M2, 1},
#endif
    };

    return dtypeMap.at(dtype);
}
