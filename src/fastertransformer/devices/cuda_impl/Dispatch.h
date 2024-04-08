
#pragma once

#include "src/fastertransformer/utils/assert_utils.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/core/Types.h"

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <functional>

#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

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
}

template<typename CastedTuple, typename WorkT, typename ...Args>
CastedTuple castArgs(const std::tuple<Args...>& args) {
    auto ret = CastedTuple();
    castTuple<WorkT>(ret, args, std::make_index_sequence<std::tuple_size_v<CastedTuple>>());
    return ret;
}

#define ARG_CASTED_FUNC_CALL(T, func_name, ...) {                                           \
    using target_args_type = FunctionTraits<std::function<decltype(func_name<T>)>>::args;   \
    auto typed_args = castArgs<target_args_type, T>(std::make_tuple(__VA_ARGS__));          \
    std::apply(func_name<T>, typed_args);                                                   \
}

#define DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, function, ...)                              \
    do {                                                                                        \
        switch (data_type) {                                                                    \
            case DataType::TYPE_FP32:                                                           \
                ARG_CASTED_FUNC_CALL(float, function, __VA_ARGS__);                             \
                break;                                                                          \
            case DataType::TYPE_FP16:                                                           \
                ARG_CASTED_FUNC_CALL(half, function, __VA_ARGS__);                              \
                break;                                                                          \
            case DataType::TYPE_BF16:                                                           \
                ARG_CASTED_FUNC_CALL(__nv_bfloat16, function, __VA_ARGS__);                     \
                break;                                                                          \
            default:                                                                            \
                FT_CHECK(false);                                                                \
        }                                                                                       \
    } while (0)

}  // namespace fastertransformer
