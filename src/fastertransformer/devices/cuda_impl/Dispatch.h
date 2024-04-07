
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
struct function_traits;

template<typename R, typename ...Args>
struct function_traits<std::function<R(Args...)>> {
public:
    static const size_t nargs = sizeof...(Args);
    typedef std::tuple<Args...> args;
};

template<typename ...DstTs, typename ...SrcTs, std::size_t...Is>
void cast_tuple(std::tuple<DstTs...> &dst, const std::tuple<SrcTs...> &src, std::index_sequence<Is...>) {
    int unused_expander[] = { 0,
    ((void)[&] {
        using SrcT = std::tuple_element_t<Is, std::tuple<SrcTs...>>;
        using DstT = std::tuple_element_t<Is, std::tuple<DstTs...>>;
        if constexpr (std::is_same_v<SrcT, DstT>) {
            std::get<Is>(dst) = std::get<Is>(src);
        } else if constexpr (std::is_pointer_v<SrcT> && std::is_void_v<std::remove_pointer_t<SrcT>>) {
            std::get<Is>(dst) = static_cast<DstT>(std::get<Is>(src));
        } else {
            std::get<Is>(dst) = (DstT)std::get<Is>(src);
        }
    }(), 0) ... };
}

template<typename ...DstTs, typename ...SrcTs>
void cast_tuple(std::tuple<DstTs...> &dst, const std::tuple<SrcTs...> &src) {
    cast_tuple(dst, src, std::make_index_sequence<sizeof...(DstTs)>());
}

template<typename CastedTuple, typename CastT, typename ...Args>
CastedTuple cast_args(const std::tuple<Args...>& args) {
    auto ret = CastedTuple();
    cast_tuple(ret, args);
    return ret;
}

#define ARG_CASTED_FUNC_CALL(T, func_name, ...) {                                           \
    using target_args_type = function_traits<std::function<decltype(func_name<T>)>>::args;  \
    auto typed_args = cast_args<target_args_type, T>(std::make_tuple(__VA_ARGS__));         \
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
