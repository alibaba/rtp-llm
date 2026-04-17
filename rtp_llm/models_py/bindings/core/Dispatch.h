
#pragma once

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/Types.h"

#include <functional>

namespace rtp_llm {

template<typename T>
struct FunctionTraits;

template<typename R, typename... Args>
struct FunctionTraits<std::function<R(Args...)>> {
public:
    static const size_t         nargs = sizeof...(Args);
    typedef std::tuple<Args...> args;
};

template<typename SrcT, typename DstT, typename WorkT>
constexpr bool IsCastingVoidPtrToWorkTPtr =
    std::is_pointer_v<SrcT> && std::is_void_v<std::remove_pointer_t<SrcT>> && std::is_pointer_v<DstT>
    && std::is_same_v<std::remove_cv_t<std::remove_pointer_t<DstT>>, WorkT>;

template<typename SrcT, typename DstT, typename WorkT>
constexpr bool IsCastingFloatToWorkT =
    std::is_floating_point_v<SrcT> && std::is_same_v<DstT, WorkT> && std::is_convertible_v<SrcT, DstT>;

template<typename SrcT, typename DstT, std::enable_if_t<std::is_same<SrcT, DstT>::value, bool> = 0>
inline DstT simpleCast(SrcT src) {
    return src;
}

template<typename SrcT,
         typename DstT,
         std::enable_if_t<(!std::is_same_v<SrcT, DstT>) && std::is_convertible_v<SrcT, DstT>, bool> = 0>
inline DstT simpleCast(SrcT src) {
    return (DstT)src;
}

template<typename SrcT,
         typename DstT,
         std::enable_if_t<(!std::is_same_v<SrcT, DstT>) && (!std::is_convertible_v<SrcT, DstT>)
                              && (std::is_pointer_v<SrcT> && std::is_pointer_v<DstT>),
                          bool> = 0>
inline DstT simpleCast(SrcT src) {
    return reinterpret_cast<DstT>(src);
}

template<typename SrcT,
         typename DstT,
         typename WorkT,
         std::enable_if_t<IsCastingVoidPtrToWorkTPtr<SrcT, DstT, WorkT>, bool> = 0>
inline DstT cast(SrcT src) {
    return static_cast<DstT>(src);
}

template<typename SrcT,
         typename DstT,
         typename WorkT,
         std::enable_if_t<IsCastingFloatToWorkT<SrcT, DstT, WorkT>, bool> = 0>
inline DstT cast(SrcT src) {
    return (DstT)src;
}

template<
    typename SrcT,
    typename DstT,
    typename WorkT,
    std::enable_if_t<(!IsCastingVoidPtrToWorkTPtr<SrcT, DstT, WorkT>) && (!IsCastingFloatToWorkT<SrcT, DstT, WorkT>),
                     bool> = 0>
inline DstT cast(SrcT src) {
    return simpleCast<SrcT, DstT>(src);
}

template<typename WorkT, typename... DstTs, typename... SrcTs, std::size_t... Idx>
void castTuple(std::tuple<DstTs...>& dst, const std::tuple<SrcTs...>& src, std::index_sequence<Idx...>) {
    int unused_expander[] = {0,
                             (
                                 (void)[&] {
                                     using SrcT         = std::tuple_element_t<Idx, std::tuple<SrcTs...>>;
                                     using DstT         = std::tuple_element_t<Idx, std::tuple<DstTs...>>;
                                     std::get<Idx>(dst) = cast<SrcT, DstT, WorkT>(std::get<Idx>(src));
                                 }(),
                                 0)...};
    (void)unused_expander;
}

template<typename CastedTuple,
         typename WorkT,
         typename... Args,
         std::enable_if_t<std::is_constructible_v<CastedTuple>, bool> = 0>
CastedTuple castArgs(const std::tuple<Args...>& args) {
    auto ret = CastedTuple();
    castTuple<WorkT>(ret, args, std::make_index_sequence<std::tuple_size_v<CastedTuple>>());
    return ret;
}

template<typename CastedTuple,
         typename WorkT,
         typename... Args,
         std::enable_if_t<!std::is_constructible_v<CastedTuple>, bool> = 0>
CastedTuple castArgs(const std::tuple<Args...>& args) {
    return args;
}

#define ARG_CASTED_FUNC_CALL(T, func_name, ...)                                                                        \
    {                                                                                                                  \
        using target_args_type = FunctionTraits<std::function<decltype(func_name<T>)>>::args;                          \
        auto typed_args        = castArgs<target_args_type, T>(std::make_tuple(__VA_ARGS__));                          \
        std::apply(func_name<T>, typed_args);                                                                          \
    }

#define DISPATCH_FOR_EACH_COMPUTE_TYPE(MACRO, ...)                                                                     \
    MACRO(DataType::TYPE_FP32, float, __VA_ARGS__)                                                                     \
    MACRO(DataType::TYPE_FP16, half, __VA_ARGS__)                                                                      \
    MACRO(DataType::TYPE_BF16, __nv_bfloat16, __VA_ARGS__)                                                             \
    default:                                                                                                           \
        RTP_LLM_CHECK(false);

#define DP_FUNCTION_CALL_CASE(data_type, T, ...)                                                                       \
    case data_type: {                                                                                                  \
        ARG_CASTED_FUNC_CALL(T, __VA_ARGS__);                                                                          \
        break;                                                                                                         \
    }

#define DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type, function, ...)                                                     \
    do {                                                                                                               \
        switch (data_type) { DISPATCH_FOR_EACH_COMPUTE_TYPE(DP_FUNCTION_CALL_CASE, function, __VA_ARGS__) }            \
    } while (0)

}  // namespace rtp_llm
