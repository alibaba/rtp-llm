#pragma once

#include "hip/hip_runtime.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include "hip_utils.hpp"

namespace HipKernels {

template<int size, bool ascending, typename T, typename idxT>
struct BitonicMerge {
    // input should be a bitonic sequence, and sort it to be a monotonic sequence
    __device__ static void merge(T* __restrict__ val_arr, idxT* __restrict__ idx_arr) {
        static_assert(HipKernels::Utils::is_power_of_2(size));
        static_assert(size >= 2 * HipKernels::Utils::WARP_SIZE);
        constexpr int arr_len = size / HipKernels::Utils::WARP_SIZE;

        constexpr int stride = arr_len / 2;
        for (int i = 0; i < stride; ++i) {
            const int other_i = i + stride;
            T& val = val_arr[i];
            T& other_val = val_arr[other_i];
            if ((val > other_val && ascending) || (val < other_val && !ascending)) {
                T tmp = val;
                val = other_val;
                other_val = tmp;

                idxT tmp2 = idx_arr[i];
                idx_arr[i] = idx_arr[other_i];
                idx_arr[other_i] = tmp2;
            }
        }

        BitonicMerge<size / 2, ascending, T, idxT>::merge(val_arr, idx_arr);
        BitonicMerge<size / 2, ascending, T, idxT>::merge(val_arr + arr_len / 2,
                                                          idx_arr + arr_len / 2);
    }
};


template<int size, bool ascending, typename T, typename idxT>
struct BitonicSort {
    __device__ static void sort(T* __restrict__ val_arr, idxT* __restrict__ idx_arr) {
        static_assert(HipKernels::Utils::is_power_of_2(size));
        static_assert(size >= 2 * HipKernels::Utils::WARP_SIZE);
        constexpr int arr_len = size / HipKernels::Utils::WARP_SIZE;

        BitonicSort<size / 2, true, T, idxT>::sort(val_arr, idx_arr);
        BitonicSort<size / 2, false, T, idxT>::sort(val_arr + arr_len / 2,
                                                    idx_arr + arr_len / 2);
        BitonicMerge<size, ascending, T, idxT>::merge(val_arr, idx_arr);
    }
};

template<typename T>
__device__ __forceinline__ T dev_max(const T& a, const T& b) {
    return a > b ? a : b;
}

template<>
__device__ __forceinline__ float dev_max<float>(const float& a, const float& b) {
    return __builtin_fmaxf(a, b);
}

template<typename T>
__device__ __forceinline__ T dev_min(const T& a, const T& b) {
    return a > b ? b : a;
}

template<>
__device__ __forceinline__ float dev_min<float>(const float& a, const float& b) {
    return __builtin_fminf(a, b);
}

template<typename T>
__device__ __forceinline__ T dev_med3(const T& a, const T& b, const T& c) {
    if constexpr(std::is_same_v<T, float>) {
        return __builtin_amdgcn_fmed3f(a, b, c);
    } else if constexpr (std::is_same_v<T, __half>) {
        __fp16 a_fp16 = *reinterpret_cast<const __fp16*>(&a);
        __fp16 b_fp16 = *reinterpret_cast<const __fp16*>(&b);
        __fp16 c_fp16 = *reinterpret_cast<const __fp16*>(&c);
        __fp16 result = __builtin_amdgcn_fmed3h(a_fp16, b_fp16, c_fp16);
        return *reinterpret_cast<const __half*>(&result);
    } else {
        auto max_0 = dev_max(a, b);
        auto min_0 = dev_min(a, b);
        return dev_min(max_0, dev_max(min_0, c));
    }
}

template<typename idxT, typename T>
__device__ __forceinline__ idxT select_idx(const idxT& idx_a, const idxT& idx_b,
                                           const T& val_a, const T& val_b, const T& selected_val) {
    return (selected_val == val_a) ? idx_a : idx_b;
}

template<int stride>
struct StrideToDPP {
    static_assert(stride == 1 || stride == 2 || stride == 4 || stride == 8, "DPP only supports stride 1 ,2, 4, 8");
};

template<> struct StrideToDPP<1> {
    static constexpr int dpp_i = 0xb1;  // quad_perm: [1,0,3,2]
};
template<> struct StrideToDPP<2> {
    static constexpr int dpp_i = 0x4e;  // quad_perm: [2,3,0,1]
};

template<> struct StrideToDPP<4> {
    static constexpr int dpp_i_shl = 260;
    static constexpr int bank_mask_shl = 0b0101;
    static constexpr int dpp_i_shr = 276;
    static constexpr int bank_mask_shr = 0b1010;
};
template<> struct StrideToDPP<8> {
    static constexpr int dpp_i_shl = 264;
    static constexpr int bank_mask_shl = 0b0011;
    static constexpr int dpp_i_shr = 280;
    static constexpr int bank_mask_shr = 0b1100;
};

template<typename T, int stride>
__forceinline__ __device__ T mov_dpp(T x) {
    constexpr int dpp_i      = StrideToDPP<stride>::dpp_i;
    constexpr int row_mask   = 0xf;
    constexpr int bank_mask  = 0xf;
    constexpr bool bound_ctrl = true;

    if constexpr (sizeof(T) == 4) {
        return __builtin_bit_cast(T,
            __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x),
                                     dpp_i,
                                     row_mask,
                                     bank_mask,
                                     bound_ctrl));
    }
    else if constexpr (sizeof(T) == 2) {
        unsigned short x_u16 = __builtin_bit_cast(unsigned short, x);
        unsigned int x_u32 = x_u16;
        unsigned int result_u32 = __builtin_amdgcn_mov_dpp(x_u32,
                                                           dpp_i,
                                                           row_mask,
                                                           bank_mask,
                                                           bound_ctrl);
        unsigned short result_u16 = static_cast<unsigned short>(result_u32);
        return __builtin_bit_cast(T, result_u16);
    }
    else {
        static_assert(sizeof(T) == 4 || sizeof(T) == 2,
                      "mov_dpp only supports 32-bit and 16-bit types.");
        return x;
    }
}

template<typename T, int stride, bool shl>
__forceinline__ __device__ T upd_dpp(const T& old, T x) {
    constexpr int dpp_i       = shl? StrideToDPP<stride>::dpp_i_shl : StrideToDPP<stride>::dpp_i_shr;
    constexpr int row_mask    = 0xf;
    constexpr int bank_mask   = shl ? StrideToDPP<stride>::bank_mask_shl : StrideToDPP<stride>::bank_mask_shr;
    constexpr bool bound_ctrl = true;

    if constexpr (sizeof(T) == 4) {
        return __builtin_bit_cast(T,
            __builtin_amdgcn_update_dpp(__builtin_bit_cast(int, old),
                                        __builtin_bit_cast(int, x),
                                        dpp_i,
                                        row_mask,
                                        bank_mask,
                                        bound_ctrl));
    }
    else if constexpr (sizeof(T) == 2) {
        unsigned int old_u32 = __builtin_bit_cast(unsigned short, old);
        unsigned int x_u32   = __builtin_bit_cast(unsigned short, x);

        unsigned int result_u32 = __builtin_amdgcn_update_dpp(old_u32,
                                                              x_u32,
                                                              dpp_i,
                                                              row_mask,
                                                              bank_mask,
                                                              bound_ctrl);
        unsigned short result_u16 = static_cast<unsigned short>(result_u32);
        return __builtin_bit_cast(T, result_u16);
    }
    else {
        static_assert(sizeof(T) == 4 || sizeof(T) == 2,
                      "upd_dpp only supports 32-bit and 16-bit types.");
        return old;
    }
}

template<typename T>
__forceinline__ __device__ constexpr T get_guard(const bool x){
    if constexpr (std::is_same_v<T, __half>){
        auto inf = __half(0x7C00);
        return x ? -inf : inf;
    } else if constexpr (std::is_same_v<T, __hip_bfloat16>) {
        auto inf = __hip_bfloat16(0x7F80);
        return x ? -inf : inf;
    } else if constexpr (!std::is_floating_point_v<T>) {
        return x ? std::numeric_limits<T>::lowest() : std::numeric_limits<T>::max();
    } else {
        return x ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();
    }
}

// Optimized sort step using DPP for small strides
template<typename T, typename idxT, int stage, int stride>
__forceinline__ __device__ typename std::enable_if<(stride <= 2), void>::type
sort_step(T* __restrict__ val_arr, idxT* __restrict__ idx_arr) {
    const int lane = threadIdx.x & (HipKernels::Utils::WARP_SIZE - 1);
    bool reverse = (lane >> stage) & 2;
    bool is_second = lane & stride;

    const auto val = *val_arr;
    const auto idx = *idx_arr;
    T other = mov_dpp<T, stride>(val);
    idxT other_idx = mov_dpp<idxT, stride>(idx);

    T selected_val = dev_med3(val, other, get_guard<T>(reverse != is_second));
    idxT selected_idx = select_idx(idx, other_idx, val, other, selected_val);

    *val_arr = selected_val;
    *idx_arr = selected_idx;
}

// Optimized sort step using DPP for small strides
template<typename T, typename idxT, int stage, int stride>
__forceinline__ __device__ typename std::enable_if<(stride > 2 && stride <= 8), void>::type
sort_step(T* __restrict__ val_arr, idxT* __restrict__ idx_arr) {
    const int lane = threadIdx.x & (HipKernels::Utils::WARP_SIZE - 1);
    bool reverse = (lane >> stage) & 2;
    bool is_second = lane & stride;

    const auto val = *val_arr;
    const auto idx = *idx_arr;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
    T other;
    other = upd_dpp<T, stride, true>(other, val);
    other = upd_dpp<T, stride, false>(other, val);
    idxT other_idx;
    other_idx = upd_dpp<idxT, stride, true>(other_idx, idx);
    other_idx = upd_dpp<idxT, stride, false>(other_idx, idx);
#pragma clang diagnostic pop

    T selected_val = dev_med3(val, other, get_guard<T>(reverse != is_second));
    idxT selected_idx = select_idx(idx, other_idx, val, other, selected_val);

    *val_arr = selected_val;
    *idx_arr = selected_idx;
}

// Fallback to shuffle for larger strides
template<typename T, typename idxT, int stage, int stride>
__forceinline__ __device__ typename std::enable_if<(stride > 8), void>::type
sort_step(T* __restrict__ val_arr, idxT* __restrict__ idx_arr) {
    const int lane = threadIdx.x & (HipKernels::Utils::WARP_SIZE - 1);
    bool reverse = (lane >> stage) & 2;
    bool is_second = lane & stride;

    const auto val = *val_arr;
    const auto idx = *idx_arr;
    T other = __shfl_xor(val, stride);
    idxT other_idx = __shfl_xor(idx, stride);

    T selected_val = dev_med3(val, other, get_guard<T>(reverse != is_second));
    idxT selected_idx = select_idx(idx, other_idx, val, other, selected_val);

    *val_arr = selected_val;
    *idx_arr = selected_idx;
}

template<bool ascending, typename T, typename idxT>
struct BitonicSort<64, ascending, T, idxT> {
    __device__ static void sort(T* __restrict__ val_arr, idxT* __restrict__ idx_arr) {
        sort_step<T, idxT, 0, 1>(val_arr, idx_arr);

        sort_step<T, idxT, 1, 2>(val_arr, idx_arr);
        sort_step<T, idxT, 1, 1>(val_arr, idx_arr);

        sort_step<T, idxT, 2, 4>(val_arr, idx_arr);
        sort_step<T, idxT, 2, 2>(val_arr, idx_arr);
        sort_step<T, idxT, 2, 1>(val_arr, idx_arr);

        sort_step<T, idxT, 3, 8>(val_arr, idx_arr);
        sort_step<T, idxT, 3, 4>(val_arr, idx_arr);
        sort_step<T, idxT, 3, 2>(val_arr, idx_arr);
        sort_step<T, idxT, 3, 1>(val_arr, idx_arr);

        sort_step<T, idxT, 4, 16>(val_arr, idx_arr);
        sort_step<T, idxT, 4, 8>(val_arr, idx_arr);
        sort_step<T, idxT, 4, 4>(val_arr, idx_arr);
        sort_step<T, idxT, 4, 2>(val_arr, idx_arr);
        sort_step<T, idxT, 4, 1>(val_arr, idx_arr);

        BitonicMerge<64, ascending, T, idxT>::merge(val_arr, idx_arr);
    }
};

// Optimized merge using DPP for small strides
template<bool ascending, typename T, typename idxT, int stride>
__forceinline__ __device__ typename std::enable_if<(stride <= 2), void>::type
merge_step(T* __restrict__ val_arr, idxT* __restrict__ idx_arr) {
    const int lane = threadIdx.x & (HipKernels::Utils::WARP_SIZE - 1);
    bool is_second = lane & stride;
    T& val = *val_arr;
    idxT& idx = *idx_arr;

    T other = mov_dpp<T, stride>(val);
    idxT other_idx = mov_dpp<idxT, stride>(idx);

    T selected_val = dev_med3(val, other, get_guard<T>(ascending != is_second));
    idxT selected_idx = select_idx(idx, other_idx, val, other, selected_val);

    val = selected_val;
    idx = selected_idx;
}

template<bool ascending, typename T, typename idxT, int stride>
__forceinline__ __device__ typename std::enable_if<(stride > 2 && stride <= 8), void>::type
merge_step(T* __restrict__ val_arr, idxT* __restrict__ idx_arr) {
    const int lane = threadIdx.x & (HipKernels::Utils::WARP_SIZE - 1);
    bool is_second = lane & stride;
    T& val = *val_arr;
    idxT& idx = *idx_arr;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
    T other;
    other = upd_dpp<T, stride, true>(other, val);
    other = upd_dpp<T, stride, false>(other, val);
    idxT other_idx;
    other_idx = upd_dpp<idxT, stride, true>(other_idx, idx);
    other_idx = upd_dpp<idxT, stride, false>(other_idx, idx);
#pragma clang diagnostic pop

    T selected_val = dev_med3(val, other, get_guard<T>(ascending != is_second));
    idxT selected_idx = select_idx(idx, other_idx, val, other, selected_val);

    val = selected_val;
    idx = selected_idx;
}

// Fallback to shuffle for larger strides
template<bool ascending, typename T, typename idxT, int stride>
__forceinline__ __device__  typename std::enable_if<(stride > 8), void>::type
merge_step(T* __restrict__ val_arr, idxT* __restrict__ idx_arr) {
    const int lane = threadIdx.x & (HipKernels::Utils::WARP_SIZE - 1);
    bool is_second = lane & stride;
    T& val = *val_arr;
    idxT& idx = *idx_arr;

    T other = __shfl_xor(val, stride);
    idxT other_idx = __shfl_xor(idx, stride);

    T selected_val = dev_med3(val, other, get_guard<T>(ascending != is_second));
    idxT selected_idx = select_idx(idx, other_idx, val, other, selected_val);

    val = selected_val;
    idx = selected_idx;
}

template<bool ascending, typename T, typename idxT>
struct BitonicMerge<64, ascending, T, idxT> {
    __device__ static void merge(T* __restrict__ val_arr, idxT* __restrict__ idx_arr) {
        merge_step<ascending, T, idxT, 32>(val_arr, idx_arr);
        merge_step<ascending, T, idxT, 16>(val_arr, idx_arr);
        merge_step<ascending, T, idxT, 8>(val_arr, idx_arr);
        merge_step<ascending, T, idxT, 4>(val_arr, idx_arr);
        merge_step<ascending, T, idxT, 2>(val_arr, idx_arr);
        merge_step<ascending, T, idxT, 1>(val_arr, idx_arr);
    }
};

}  // namespace HipKernels
