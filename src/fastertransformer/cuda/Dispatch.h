
#pragma once

#include "src/fastertransformer/utils/assert_utils.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/core/Types.h"

#if USING_CUDA
#include <cuda_runtime.h>
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif
#endif
#if USING_ROCM
#include "src/fastertransformer/rocm/cuda_shims.h"
#endif

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <functional>

namespace fastertransformer {

template<typename T>
struct FunctionTraits;

template<typename R, typename ...Args>
struct FunctionTraits<std::function<R(Args...)>> {
public:
    static const size_t nargs = sizeof...(Args);
    typedef std::tuple<Args...> args;
};

template<typename SrcT, typename DstT, typename WorkT>
constexpr bool IsCastingVoidPtrToWorkTPtr =
    std::is_pointer_v<SrcT> &&
    std::is_void_v<std::remove_pointer_t<SrcT>> &&
    std::is_pointer_v<DstT> &&
    std::is_same_v<std::remove_cv_t<std::remove_pointer_t<DstT>>, WorkT>;

template<typename SrcT, typename DstT, typename WorkT>
constexpr bool IsCastingFloatToWorkT =
    std::is_floating_point_v<SrcT> &&
    std::is_same_v<DstT, WorkT> &&
    std::is_convertible_v<SrcT, DstT>;

template<typename SrcT, typename DstT, std::enable_if_t<std::is_same<SrcT, DstT>::value, bool> = 0>
inline DstT simpleCast(SrcT src) {
    return src;
}

template<typename SrcT, typename DstT,
         std::enable_if_t<(!std::is_same_v<SrcT, DstT>) && std::is_convertible_v<SrcT, DstT>, bool> = 0>
inline DstT simpleCast(SrcT src) {
    return (DstT)src;
}

template<typename SrcT, typename DstT, typename WorkT,
         std::enable_if_t<IsCastingVoidPtrToWorkTPtr<SrcT, DstT, WorkT>, bool> = 0>
inline DstT cast(SrcT src) {
    return static_cast<DstT>(src);
}

template<typename SrcT, typename DstT, typename WorkT,
         std::enable_if_t<IsCastingFloatToWorkT<SrcT, DstT, WorkT>, bool> = 0>
inline DstT cast(SrcT src) {
    return (DstT)src;
}

template<typename SrcT, typename DstT, typename WorkT,
         std::enable_if_t<(!IsCastingVoidPtrToWorkTPtr<SrcT, DstT, WorkT>) &&
                          (!IsCastingFloatToWorkT<SrcT, DstT, WorkT>), bool> = 0>
inline DstT cast(SrcT src) {
    return simpleCast<SrcT, DstT>(src);
}

template<typename WorkT, typename ...DstTs, typename ...SrcTs, std::size_t ...Idx>
void castTuple(std::tuple<DstTs...> &dst, const std::tuple<SrcTs...> &src, std::index_sequence<Idx...>) {
    int unused_expander[] = { 0,
    ((void)[&] {
        using SrcT = std::tuple_element_t<Idx, std::tuple<SrcTs...>>;
        using DstT = std::tuple_element_t<Idx, std::tuple<DstTs...>>;
        std::get<Idx>(dst) = cast<SrcT, DstT, WorkT>(std::get<Idx>(src));
    }(), 0) ... };
    (void)unused_expander;
}

template<typename CastedTuple, typename WorkT, typename ...Args,
         std::enable_if_t<std::is_constructible_v<CastedTuple>, bool> = 0>
CastedTuple castArgs(const std::tuple<Args...>& args) {
    auto ret = CastedTuple();
    castTuple<WorkT>(ret, args, std::make_index_sequence<std::tuple_size_v<CastedTuple>>());
    return ret;
}

template<typename CastedTuple, typename WorkT, typename ...Args,
         std::enable_if_t<!std::is_constructible_v<CastedTuple>, bool> = 0>
CastedTuple castArgs(const std::tuple<Args...>& args) {
    return move(args);
}

#define ARG_CASTED_FUNC_CALL(T, func_name, ...) {                                           \
    using target_args_type = FunctionTraits<std::function<decltype(func_name<T>)>>::args;   \
    auto typed_args = castArgs<target_args_type, T>(std::make_tuple(__VA_ARGS__));          \
    std::apply(func_name<T>, typed_args);                                                   \
}

#define DISPATCH_FOR_EACH_COMPUTE_TYPE(MACRO, ...)         \
    MACRO(DataType::TYPE_FP32, float, __VA_ARGS__)         \
    MACRO(DataType::TYPE_FP16, half, __VA_ARGS__)          \
    MACRO(DataType::TYPE_BF16, __nv_bfloat16, __VA_ARGS__) \
    default: \
        FT_CHECK(false);

#define DISPATCH_FOR_EACH_NUMERIC_TYPE(MACRO, ...)         \
    MACRO(DataType::TYPE_INT8, int8_t, __VA_ARGS__)        \
    MACRO(DataType::TYPE_INT32, int32_t, __VA_ARGS__)      \
    MACRO(DataType::TYPE_INT64, int64_t, __VA_ARGS__)      \
    MACRO(DataType::TYPE_UINT8, uint8_t, __VA_ARGS__)      \
    MACRO(DataType::TYPE_UINT32, uint32_t, __VA_ARGS__)    \
    MACRO(DataType::TYPE_UINT64, uint64_t, __VA_ARGS__)    \
    DISPATCH_FOR_EACH_COMPUTE_TYPE(MACRO, __VA_ARGS__)     \

#define DP_FUNCTION_CALL_CASE(data_type, T, ...) \
    case data_type: {                            \
        ARG_CASTED_FUNC_CALL(T, __VA_ARGS__);    \
        break;                                   \
    }

#define DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, function, ...)                              \
    do {                                                                                        \
        switch (data_type) {                                                                    \
            DISPATCH_FOR_EACH_COMPUTE_TYPE(DP_FUNCTION_CALL_CASE, function, __VA_ARGS__)        \
        }                                                                                       \
    } while (0)

#define DISPATCH_CUDA_FUNCTION_GENERAL_TYPE(data_type, function, ...)                           \
    do {                                                                                        \
        switch (data_type) {                                                                    \
            DISPATCH_FOR_EACH_NUMERIC_TYPE(DP_FUNCTION_CALL_CASE, function, __VA_ARGS__)        \
        }                                                                                       \
    } while (0)

#define INNER_TYPE_CASE(dtype2, T2, function, T1, ...) \
    case dtype2: {                                     \
        function<T1, T2>(__VA_ARGS__);                 \
        break;                                         \
    }

#define GENERAL_OUTER_TYPE_CASE(dtype1, T1, dtype2, function, ...)                                   \
    case dtype1: {                                                                           \
        switch (dtype2) {                                                                    \
            DISPATCH_FOR_EACH_NUMERIC_TYPE(INNER_TYPE_CASE, function, T1, __VA_ARGS__);      \
        }                                                                                    \
        break;                                                                               \
    }

#define DISPATCH_CUDA_FUNCTION_TWO_TYPES(dtype1, dtype2, function, ...)                      \
    switch (dtype1) {                                                                        \
        GENERAL_OUTER_TYPE_CASE(DataType::TYPE_FP16, half, dtype2, function, __VA_ARGS__)            \
        GENERAL_OUTER_TYPE_CASE(DataType::TYPE_BF16, __nv_bfloat16, dtype2, function, __VA_ARGS__)   \
        GENERAL_OUTER_TYPE_CASE(DataType::TYPE_FP32, float, dtype2, function, __VA_ARGS__)           \
        GENERAL_OUTER_TYPE_CASE(DataType::TYPE_INT32, int32_t, dtype2, function, __VA_ARGS__)        \
        GENERAL_OUTER_TYPE_CASE(DataType::TYPE_UINT32, uint32_t, dtype2, function, __VA_ARGS__)      \
        GENERAL_OUTER_TYPE_CASE(DataType::TYPE_INT64, int64_t, dtype2, function, __VA_ARGS__)        \
        GENERAL_OUTER_TYPE_CASE(DataType::TYPE_UINT64, uint64_t, dtype2, function, __VA_ARGS__)      \
    }

#define COMPUTE_OUTER_TYPE_CASE(dtype1, T1, dtype2, function, ...)                           \
    case dtype1: {                                                                           \
        switch (dtype2) {                                                                    \
            DISPATCH_FOR_EACH_COMPUTE_TYPE(INNER_TYPE_CASE, function, T1, __VA_ARGS__);      \
        }                                                                                    \
        break;                                                                               \
    }

#define DISPATCH_CUDA_FUNCTION_TWO_COMPUTE_TYPES(dtype1, dtype2, function, ...)                      \
    switch (dtype1) {                                                                                \
        COMPUTE_OUTER_TYPE_CASE(DataType::TYPE_FP16, half, dtype2, function, __VA_ARGS__)            \
        COMPUTE_OUTER_TYPE_CASE(DataType::TYPE_BF16, __nv_bfloat16, dtype2, function, __VA_ARGS__)   \
        COMPUTE_OUTER_TYPE_CASE(DataType::TYPE_FP32, float, dtype2, function, __VA_ARGS__)           \
    }

}  // namespace fastertransformer
