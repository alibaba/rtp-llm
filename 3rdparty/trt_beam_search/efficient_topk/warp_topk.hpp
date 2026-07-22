#pragma once

#include "hip/hip_runtime.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#include "bitonic_sort.hpp"
#include "hip_utils.hpp"
#include "half_support.hpp"
#ifdef ENABLE_BF16
#include "bfloat16_support.hpp"
#endif // ENABLE_BF16

namespace HipKernels {
namespace buffer_load_helpers {

constexpr int MAX_CAPACITY = 1024;

enum struct AmdBufferCoherence
{
    coherence_default = 0,
    glc               = 1,
    slc               = 2,
    glc_slc           = 3,
    WAVE_NT0          = 0,
    WAVE_NT1          = 2,
    GROUP_NT0         = 1,
    GROUP_NT1         = 3,
    DEVICE_NT0        = 8,
    DEVICE_NT1        = 10,
    SYSTEM_NT0        = 9,
    SYSTEM_NT1        = 11,
};

using int32x4_t = int __attribute__((ext_vector_type(4)));
using floatx4_t = float __attribute__((ext_vector_type(4)));
using halfx8_t = _Float16 __attribute__((ext_vector_type(8)));
using uint32_t = unsigned int;
using index_t = uint32_t;

struct __attribute__((packed)) BufferResource
{
    const void* ptr;
    uint32_t range;
    uint32_t config;
};

#ifndef CK_TILE_BUFFER_RESOURCE_3RD_DWORD
#define CK_TILE_BUFFER_RESOURCE_3RD_DWORD 0x00020000
#endif

__device__ __forceinline__ int32x4_t make_wave_buffer_resource(const void* ptr, uint32_t size_in_bytes)
{
    BufferResource res{ptr, size_in_bytes, CK_TILE_BUFFER_RESOURCE_3RD_DWORD};
    return __builtin_bit_cast(int32x4_t, res);
}

extern "C" __device__ int32x4_t
llvm_amdgcn_raw_buffer_load_i32x4(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

template <typename T, AmdBufferCoherence coherence>
__device__ __forceinline__ T
buffer_load_dwordx4(const int32x4_t& src_wave_buffer_resource,
                    index_t src_thread_addr_offset_bytes,
                    index_t src_wave_addr_offset)
{
    static_assert(sizeof(T) == 16, "T must be 128 bits (4 dwords)");
    int32x4_t tmp = llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                      src_thread_addr_offset_bytes,
                                                      src_wave_addr_offset,
                                                      static_cast<index_t>(coherence));
    return __builtin_bit_cast(T, tmp);
}

template <typename ReturnT, AmdBufferCoherence coherence, typename SrcT, typename IdxT>
__device__ __forceinline__ ReturnT
buffer_load(const SrcT* p_src_wave,
            IdxT base_idx0,
            IdxT element_space_size)
{
    static_assert(sizeof(ReturnT) == 16, "ReturnT must be 128 bits (4 dwords)");
    const int32x4_t srsrc =
        make_wave_buffer_resource(p_src_wave, static_cast<uint32_t>(element_space_size * sizeof(SrcT)));
    const index_t voffset_bytes = static_cast<index_t>(base_idx0 * sizeof(SrcT));
    ReturnT packed = buffer_load_dwordx4<ReturnT, coherence>(srsrc, voffset_bytes, 0);
    return packed;
}

} // namespace buffer_load_helpers


// --- Warp-Level Sorting Primitives ---

template<int capacity, bool greater, typename T, typename IdxT>
class WarpSort {
public:
    __device__ WarpSort(IdxT k, T dummy)
        : lane_(threadIdx.x % Utils::WARP_SIZE), k_(k), dummy_(dummy) {
        static_assert(capacity >= Utils::WARP_SIZE && Utils::is_power_of_2(capacity));
        #pragma unroll
        for (int i = 0; i < max_elements_per_thread_; ++i) {
            values_[i] = dummy_;
        }
    }

    __device__ void load_sorted(const T* __restrict__ in,
                                 const IdxT* __restrict__ in_idx,
                                 IdxT start) {
        IdxT idx = start + Utils::WARP_SIZE - 1 - lane_;
        #pragma unroll
        for (int i = max_elements_per_thread_ - 1; i >= 0; --i, idx += Utils::WARP_SIZE) {
            if (idx < start + k_) {
                T t = in[idx];
                if (numeric::is_preferred<greater>(t, values_[i])) {
                    values_[i] = t;
                    indices_[i] = in_idx[idx];
                }
            }
        }
        BitonicMerge<capacity, !greater, T, IdxT>::merge(values_, indices_);
    }

    __device__ void dump(T* __restrict__ out, IdxT* __restrict__ out_idx) const {
        #pragma unroll
        for (int i = 0; i < max_elements_per_thread_; ++i) {
            IdxT out_i = i * Utils::WARP_SIZE + lane_;
            if (out_i < k_) {
                out[out_i] = values_[i];
                if (out_idx) {
                    out_idx[out_i] = indices_[i];
                }
            }
        }
    }

protected:
    static constexpr int max_elements_per_thread_ = capacity / Utils::WARP_SIZE;
    T values_[max_elements_per_thread_];
    IdxT indices_[max_elements_per_thread_];
    const int lane_;
    const IdxT k_;
    const T dummy_;
};

template<int capacity, bool greater, typename T, typename IdxT>
class WarpSelect : public WarpSort<capacity, greater, T, IdxT> {
public:
    __device__ WarpSelect(IdxT k, T dummy)
        : WarpSort<capacity, greater, T, IdxT>(k, dummy),
          k_th_(dummy),
          k_th_lane_((k - 1) % Utils::WARP_SIZE) {
        extern __shared__ char smem_buf[];
        const int num_warps = blockDim.x / Utils::WARP_SIZE;
        const int warp_id = threadIdx.x / Utils::WARP_SIZE;
        values_smem_ = reinterpret_cast<T*>(smem_buf);
        values_smem_ += warp_id * Utils::WARP_SIZE;
        indices_smem_ = reinterpret_cast<IdxT*>(smem_buf
            + Utils::round_up_to_multiple_of<16>(num_warps * sizeof(T) * Utils::WARP_SIZE));
        indices_smem_ += warp_id * Utils::WARP_SIZE;
    }

    __device__ void add(const T* __restrict__ in, IdxT start, IdxT end) {
        const IdxT n = end - start;
        const IdxT whole = n & ~(static_cast<IdxT>(Utils::WARP_SIZE) - 1);
        const IdxT padded = (n == whole) ? n : (whole + Utils::WARP_SIZE);
        const IdxT end_aligned = start + whole;
        const IdxT end_for_fullwarp = start + padded;

        for (IdxT i = start + this->lane_; i < end_aligned; i += Utils::WARP_SIZE) {
            add(in[i], i);
        }

        for (IdxT i = end_aligned + this->lane_; i < end_for_fullwarp; i += Utils::WARP_SIZE) {
            T val = (i < end) ? in[i] : this->dummy_;
            add(val, i);
        }
    }

    __device__ void add(T val, IdxT idx) {
        const bool do_add = numeric::is_preferred<greater>(val, k_th_);
        const uint64_t mask = __ballot(do_add);

        if (mask == 0) { return; }

        const int prefix = __popcll(mask & ((1ull << this->lane_) - 1));
        const int base = smem_buf_len_;
        const int pos = base + prefix;
        const bool in_place = do_add && (pos < Utils::WARP_SIZE);

        if (in_place) {
            values_smem_[pos] = val;
            indices_smem_[pos] = idx;
        }

        const int total = __popcll(mask);
        smem_buf_len_ = base + total;

        if (smem_buf_len_ >= Utils::WARP_SIZE) {
            __builtin_amdgcn_wave_barrier();
            merge_buf_(values_smem_[this->lane_], indices_smem_[this->lane_]);
            smem_buf_len_ -= Utils::WARP_SIZE;
        }

        const bool overflow = do_add && !in_place;
        if (overflow) {
            const int new_pos = pos - Utils::WARP_SIZE;
            values_smem_[new_pos] = val;
            indices_smem_[new_pos] = idx;
        }
        __builtin_amdgcn_wave_barrier();
    }

    __device__ void done() {
        if (smem_buf_len_) {
            T val = (this->lane_ < smem_buf_len_) ? values_smem_[this->lane_] : this->dummy_;
            IdxT idx = (this->lane_ < smem_buf_len_) ? indices_smem_[this->lane_] : 0;
            merge_buf_(val, idx);
        }
        __syncthreads();
    }

private:
    __device__ void set_k_th_() {
        k_th_ = __shfl(this->values_[this->max_elements_per_thread_ - 1], k_th_lane_);
    }

    __device__ void merge_buf_(T val, IdxT idx) {
        BitonicSort<Utils::WARP_SIZE, greater, T, IdxT>::sort(&val, &idx);
        T& old_val = this->values_[this->max_elements_per_thread_ - 1];
        if (numeric::is_preferred<greater>(val, old_val)) {
            old_val = val;
            this->indices_[this->max_elements_per_thread_ - 1] = idx;
        }
        BitonicMerge<capacity, !greater, T, IdxT>::merge(this->values_, this->indices_);
        set_k_th_();
    }

    using WarpSort<capacity, greater, T, IdxT>::max_elements_per_thread_;
    using WarpSort<capacity, greater, T, IdxT>::values_;
    using WarpSort<capacity, greater, T, IdxT>::indices_;
    using WarpSort<capacity, greater, T, IdxT>::lane_;
    using WarpSort<capacity, greater, T, IdxT>::k_;
    using WarpSort<capacity, greater, T, IdxT>::dummy_;

    T* values_smem_;
    IdxT* indices_smem_;
    int smem_buf_len_ = 0;
    T k_th_;
    const int k_th_lane_;
};


template<int capacity, bool greater, typename T, typename IdxT>
class WarpBitonic : public WarpSort<capacity, greater, T, IdxT> {
public:
    __device__ WarpBitonic(IdxT k, T dummy)
        : WarpSort<capacity, greater, T, IdxT>(k, dummy), buf_len_(0) {}

    __device__ void add(const T* __restrict__ in, IdxT start, IdxT end) {
        add_first_(in, start, end);
        start += capacity;
        while (start < end) {
            add_extra_(in, start, end);
            merge_();
            start += capacity;
        }
    }

    __device__ void add(T val, IdxT idx) {
        #pragma unroll
        for (int i = 0; i < this->max_elements_per_thread_; ++i) {
            if (i == buf_len_) {
                val_buf_[i] = val;
                idx_buf_[i] = idx;
            }
        }
        ++buf_len_;
        if (buf_len_ == this->max_elements_per_thread_) {
            BitonicSort<capacity, greater, T, IdxT>::sort(val_buf_, idx_buf_);
            merge_();
            buf_len_ = 0;
        }
    }

    __device__ void done() {
        if (buf_len_ != 0) {
            #pragma unroll
            for (int i = 0; i < this->max_elements_per_thread_; ++i) {
                if (i >= buf_len_) {
                    val_buf_[i] = this->dummy_;
                }
            }
            BitonicSort<capacity, greater, T, IdxT>::sort(val_buf_, idx_buf_);
            merge_();
        }
    }

private:
    __device__ void add_first_(const T* __restrict__ in, IdxT start, IdxT end) {
        IdxT idx = start + this->lane_;
        #pragma unroll
        for (int i = 0; i < this->max_elements_per_thread_; ++i, idx += Utils::WARP_SIZE) {
            if (idx < end) {
                this->values_[i] = in[idx];
                this->indices_[i] = idx;
            }
        }
        BitonicSort<capacity, !greater, T, IdxT>::sort(this->values_, this->indices_);
    }

    __device__ void add_extra_(const T* __restrict__ in, IdxT start, IdxT end) {
        IdxT idx = start + this->lane_;
        #pragma unroll
        for (int i = 0; i < this->max_elements_per_thread_; ++i, idx += Utils::WARP_SIZE) {
            val_buf_[i] = (idx < end) ? in[idx] : this->dummy_;
            idx_buf_[i] = idx;
        }
        BitonicSort<capacity, greater, T, IdxT>::sort(val_buf_, idx_buf_);
    }

    __device__ void merge_() {
        #pragma unroll
        for (int i = 0; i < this->max_elements_per_thread_; ++i) {
            if (numeric::is_preferred<greater>(val_buf_[i], this->values_[i])) {
                this->values_[i] = val_buf_[i];
                this->indices_[i] = idx_buf_[i];
            }
        }
        BitonicMerge<capacity, !greater, T, IdxT>::merge(this->values_, this->indices_);
    }

    using WarpSort<capacity, greater, T, IdxT>::max_elements_per_thread_;
    using WarpSort<capacity, greater, T, IdxT>::values_;
    using WarpSort<capacity, greater, T, IdxT>::indices_;
    using WarpSort<capacity, greater, T, IdxT>::lane_;
    using WarpSort<capacity, greater, T, IdxT>::k_;
    using WarpSort<capacity, greater, T, IdxT>::dummy_;

    T val_buf_[max_elements_per_thread_];
    IdxT idx_buf_[max_elements_per_thread_];
    int buf_len_;
};


template<int capacity, bool greater, typename T, typename IdxT>
class WarpMerge : public WarpSort<capacity, greater, T, IdxT> {
public:
    __device__ WarpMerge(IdxT k, T dummy)
        : WarpSort<capacity, greater, T, IdxT>(k, dummy) {}

    __device__ void add(const T* __restrict__ in, const IdxT* __restrict__ in_idx, IdxT start, IdxT end) {
        IdxT idx = start + this->lane_;
        IdxT first_end = (start + this->k_ < end) ? (start + this->k_) : end;
        #pragma unroll
        for (int i = 0; i < this->max_elements_per_thread_; ++i, idx += Utils::WARP_SIZE) {
            if (idx < first_end) {
                this->values_[i] = in[idx];
                this->indices_[i] = in_idx[idx];
            }
        }
        for (start += this->k_; start < end; start += this->k_) {
            this->load_sorted(in, in_idx, start);
        }
    }

    __device__ void done() {}

private:
    using WarpSort<capacity, greater, T, IdxT>::max_elements_per_thread_;
    using WarpSort<capacity, greater, T, IdxT>::values_;
    using WarpSort<capacity, greater, T, IdxT>::indices_;
    using WarpSort<capacity, greater, T, IdxT>::lane_;
    using WarpSort<capacity, greater, T, IdxT>::k_;
    using WarpSort<capacity, greater, T, IdxT>::dummy_;
};


// --- Block-Level Logic ---

template<template<int, bool, typename, typename> class WarpSortImpl, int capacity, bool greater, typename T, typename IdxT>
class WarpSortBlockWide {
public:
    __device__ WarpSortBlockWide(IdxT k, T dummy, void* smem_buf)
        : queue_(k, dummy), k_(k), dummy_(dummy) {
        val_smem_ = static_cast<T*>(smem_buf);
        const int num_warps = blockDim.x / Utils::WARP_SIZE;
        idx_smem_ = reinterpret_cast<IdxT*>(
            reinterpret_cast<char*>(smem_buf)
            + Utils::round_up_to_multiple_of<16>(num_warps / 2 * sizeof(T) * k_));
    }

    __device__ void add(const T* __restrict__ in, const IdxT* __restrict__ in_idx, IdxT start, IdxT end) {
        static_assert(std::is_same_v<WarpSortImpl<capacity, greater, T, IdxT>, WarpMerge<capacity, greater, T, IdxT>>);
        int num_warps = blockDim.x / Utils::WARP_SIZE;
        const int warp_id = threadIdx.x / Utils::WARP_SIZE;
        IdxT len_per_warp = (end - start - 1) / num_warps + 1;
        len_per_warp = ((len_per_warp - 1) / k_ + 1) * k_;
        IdxT warp_start = start + warp_id * len_per_warp;
        IdxT warp_end = std::min(warp_start + len_per_warp, end);
        queue_.add(in, in_idx, warp_start, warp_end);
    }

    __device__ void add(const T* __restrict__ in, IdxT start, IdxT end) {
        if constexpr (std::is_same_v<WarpSortImpl<capacity, greater, T, IdxT>, WarpSelect<capacity, greater, T, IdxT>>) {
            const IdxT n = end - start;
            const IdxT tid = threadIdx.x;
            const IdxT stride = blockDim.x;
            constexpr IdxT elements = 16 / sizeof(T);

            constexpr auto cache_policy = buffer_load_helpers::AmdBufferCoherence::slc;

#ifdef ENABLE_BF16
            constexpr bool use_half_vec_load = std::is_same_v<T, __half> || std::is_same_v<T, amd_bfloat16>;
#else
            constexpr bool use_half_vec_load = std::is_same_v<T, __half>;
#endif

            if constexpr (use_half_vec_load) {
                constexpr IdxT repetition = 2;
                constexpr IdxT tile = elements * repetition;
                const IdxT block_tile = blockDim.x * tile;
                const IdxT end_aligned = start + Utils::round_up_to_multiple_of(
                                                     n, block_tile);
                const IdxT tail = end_aligned - block_tile;

                using VecType = buffer_load_helpers::halfx8_t;

                VecType arr[repetition];
                for (IdxT i = start + tid * tile; i < tail;
                     i += stride * tile) {
#pragma unroll
                    for (IdxT idx = 0; idx < repetition; ++idx) {
                      arr[idx] = buffer_load_helpers::buffer_load<VecType,
                                                                  cache_policy>(
                          in, i + idx * elements, n);
                    }
#pragma unroll
                    for (IdxT idx = 0; idx < tile; ++idx) {
                      queue_.add(arr[idx / elements][idx % elements], i + idx);
                    }
                }

                for (IdxT i = tail + tid; i < end_aligned; i += stride) {
                    const auto val = (i < end) ? in[i] : dummy_;
                    queue_.add(val, i);
                }
            } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int>) {
                constexpr IdxT repetition = 2;
                constexpr IdxT tile = elements * repetition;
                const IdxT block_tile = blockDim.x * tile;
                const IdxT end_aligned  = start + Utils::round_up_to_multiple_of(n, block_tile);

                using VecType = std::conditional_t<std::is_same_v<T, float>,
                                                   buffer_load_helpers::floatx4_t,
                                                   buffer_load_helpers::int32x4_t>;
                VecType arr[repetition];
                for(IdxT i = start + tid * tile; i < end_aligned; i += stride * tile)
                {
#pragma unroll
                    for(IdxT idx = 0; idx < repetition; ++idx)
                    {
                        arr[idx] = buffer_load_helpers::buffer_load<VecType, cache_policy>(
                            in, i + idx * elements, n);
                    }
#pragma unroll
                    for(IdxT idx = 0; idx < tile; ++idx)
                    {
                        const auto val = (i + idx < end) ? arr[idx / elements][idx % elements] : dummy_;
                        queue_.add(val, i + idx);
                    }
                }
            } else {
                static_assert(std::is_same_v<T, __half> ||
#ifdef ENABLE_BF16
                              std::is_same_v<T, amd_bfloat16> ||
#endif
                              std::is_same_v<T, float> ||
                              std::is_same_v<T, int>,
                              "Unsupported type T: only "
                              "__half, "
#ifdef ENABLE_BF16
                              "amd_bfloat16, "
#endif
                              "float, and int are implemented");
            }
        } else if constexpr (std::is_same_v<WarpSortImpl<capacity, greater, T, IdxT>, WarpBitonic<capacity, greater, T, IdxT>>) {
            int num_warps = blockDim.x / Utils::WARP_SIZE;
            const int warp_id = threadIdx.x / Utils::WARP_SIZE;
            IdxT len_per_warp = (end - start - 1) / num_warps + 1;
            len_per_warp = Utils::round_up_to_multiple_of<Utils::WARP_SIZE>(len_per_warp);
            IdxT warp_start = start + warp_id * len_per_warp;
            IdxT warp_end = std::min(warp_start + len_per_warp, end);
            this->queue_.add(in, warp_start, warp_end);
        }
    }

    __device__ void add(T val, IdxT idx) { queue_.add(val, idx); }

    __device__ void done() {
        queue_.done();
        int num_warps = blockDim.x / Utils::WARP_SIZE;
        const int warp_id = threadIdx.x / Utils::WARP_SIZE;
        while (num_warps > 1) {
            int half_num_warps = (num_warps + 1) / 2;
            if (warp_id < num_warps && warp_id >= half_num_warps) {
                int dst_warp_id = warp_id - half_num_warps;
                queue_.dump(val_smem_ + dst_warp_id * k_, idx_smem_ + dst_warp_id * k_);
            }
            __syncthreads();
            if (warp_id < num_warps / 2) {
                queue_.load_sorted(val_smem_, idx_smem_, warp_id * k_);
            }
            __syncthreads();
            num_warps = half_num_warps;
        }
    }

    __device__ void dump(T* __restrict__ out, IdxT* __restrict__ out_idx) const {
        if (threadIdx.x < Utils::WARP_SIZE) {
            queue_.dump(out, out_idx);
        }
    }

private:
    WarpSortImpl<capacity, greater, T, IdxT> queue_;
    int k_;
    T dummy_;
    T* val_smem_;
    IdxT* idx_smem_;
};

// --- Kernel and Launch Logic ---

template<template<int, bool, typename, typename> class WarpSortClass, int capacity, bool greater, typename T, typename IdxT>
__global__ void __launch_bounds__(512, 2)
block_kernel(const T* __restrict__ in,
               const IdxT* __restrict__ in_idx,
               int batch_size,
               IdxT len,
               IdxT k,
               T* __restrict__ out,
               IdxT* __restrict__ out_idx,
               T dummy) {
    extern __shared__ char smem_buf[];
    const int num_of_block = gridDim.x / batch_size;
    const IdxT len_per_block = std::is_same_v<WarpSortClass<capacity, greater, T, IdxT>,
                                              WarpSelect<capacity, greater, T, IdxT>>
                                ? len
                                : (len - 1) / num_of_block + 1;
    const int batch_id = blockIdx.x / num_of_block;
    const int block_id_in_a_batch = blockIdx.x % num_of_block;
    IdxT start = block_id_in_a_batch * len_per_block;
    IdxT end = std::min(start + len_per_block, len);

    WarpSortBlockWide<WarpSortClass, capacity, greater, T, IdxT> queue(k, dummy, smem_buf);
    if constexpr (std::is_same_v<WarpSortClass<capacity, greater, T, IdxT>, WarpMerge<capacity, greater, T, IdxT>>) {
        queue.add(in + static_cast<size_t>(batch_id) * len, in_idx + static_cast<size_t>(batch_id) * len, start, end);
    } else {
        queue.add(in + static_cast<size_t>(batch_id) * len, start, end);
    }
    queue.done();
    queue.dump(out + static_cast<size_t>(blockIdx.x) * k, out_idx + static_cast<size_t>(blockIdx.x) * k);
}

template<bool greater, int Capacity, template<int, bool, typename, typename> class WarpSortClass, typename T, typename IdxT>
auto find_block_kernel_helper(int capacity) {
    if constexpr (Capacity == Utils::WARP_SIZE) {
        return greater ? block_kernel<WarpSortClass, Utils::WARP_SIZE, true, T, IdxT>
                       : block_kernel<WarpSortClass, Utils::WARP_SIZE, false, T, IdxT>;
    } else {
        if (capacity == Capacity) {
            return greater ? block_kernel<WarpSortClass, Capacity, true, T, IdxT>
                           : block_kernel<WarpSortClass, Capacity, false, T, IdxT>;
        }
        return find_block_kernel_helper<greater, Capacity / 2, WarpSortClass, T, IdxT>(capacity);
    }
}

template<bool greater, template<int, bool, typename, typename> class WarpSortClass, typename T, typename IdxT>
auto find_block_kernel(int k) {
    const int capacity = Utils::calc_capacity(k);
    assert(capacity <= buffer_load_helpers::MAX_CAPACITY);
    return find_block_kernel_helper<greater, buffer_load_helpers::MAX_CAPACITY, WarpSortClass, T, IdxT>(capacity);
}

template<typename T, typename IdxT>
int calc_smem_size_for_block_wide(int num_of_warp, IdxT k) {
    int n = std::max<int>(num_of_warp / 2 * k, num_of_warp * Utils::WARP_SIZE);
    return Utils::round_up_to_multiple_of<16>(n * sizeof(T)) + n * sizeof(IdxT);
}

template<template<int, bool, typename, typename> class WarpSortClass, typename T, typename IdxT>
void calc_launch_parameter_by_occupancy(IdxT k, int* block_size, int* min_grid_size) {
    // Cache occupancy-derived launch parameters per k. The occupancy API is expensive
    // and the result only depends on k for a given template instantiation.
    // Protected by a mutex because this function can be called concurrently from
    // multiple host threads. Uses std::mutex (not std::shared_mutex): the ROCm
    // clang bundled with ROCm 7.2 cannot parse gcc-toolset-12's <shared_mutex>
    // (unqualified `chrono` in try_lock_until) when this host-side header is
    // pulled into a HIP translation unit. The occupancy cache is looked up rarely
    // so an exclusive lock on the read path is negligible.
    static std::mutex cache_mutex;
    static std::unordered_map<IdxT, std::pair<int, int>> occupancy_cache;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = occupancy_cache.find(k);
        if (it != occupancy_cache.end()) {
            *block_size    = it->second.first;
            *min_grid_size = it->second.second;
            return;
        }
    }

    auto func = find_block_kernel<true, WarpSortClass, T, IdxT>(k);
    auto calc_smem = [k](int bs) {
        return calc_smem_size_for_block_wide<T, IdxT>(bs / Utils::WARP_SIZE, k);
    };
    HIP_CHECK(hipOccupancyMaxPotentialBlockSizeVariableSMem(min_grid_size, block_size, func, calc_smem));
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        occupancy_cache.emplace(k, std::make_pair(*block_size, *min_grid_size));
    }
}

template<template<int, bool, typename, typename> class WarpSortClass>
struct LaunchThreshold {};

template<> struct LaunchThreshold<WarpSelect> {
    static constexpr int len_factor_for_multi_block = 2;
    static constexpr int len_factor_for_single_block = 256;
};

template<> struct LaunchThreshold<WarpBitonic> {
    static constexpr int len_factor_for_choosing = 4;
    static constexpr int len_factor_for_multi_block = 2;
    static constexpr int len_factor_for_single_block = 4;
};

template<template<int, bool, typename, typename> class WarpSortClass, typename T, typename IdxT>
void calc_launch_parameter(int batch_size, IdxT len, IdxT k, int* p_num_of_block, int* p_num_of_warp) {
    const int capacity = Utils::calc_capacity(k);
    int block_size = 0;
    int min_grid_size = 0;
    calc_launch_parameter_by_occupancy<WarpSortClass, T, IdxT>(k, &block_size, &min_grid_size);

    int num_of_warp;
    int num_of_block;
    if (batch_size < min_grid_size) {
        num_of_warp = block_size / Utils::WARP_SIZE;
        num_of_block = min_grid_size / batch_size;
        IdxT len_per_block = (len - 1) / num_of_block + 1;
        IdxT len_per_warp = (len_per_block - 1) / num_of_warp + 1;
        len_per_warp = Utils::round_up_to_multiple_of<Utils::WARP_SIZE>(len_per_warp);
        len_per_block = len_per_warp * num_of_warp;
        num_of_block = (len - 1) / len_per_block + 1;
        constexpr int len_factor = LaunchThreshold<WarpSortClass>::len_factor_for_multi_block;
        if (len_per_warp < static_cast<IdxT>(capacity * len_factor)) {
            len_per_warp = capacity * len_factor;
            len_per_block = num_of_warp * len_per_warp;
            if (len_per_block > len) { len_per_block = len; }
            num_of_block = (len - 1) / len_per_block + 1;
            num_of_warp = (len_per_block - 1) / len_per_warp + 1;
        }
    } else {
        num_of_block = 1;
        float scale = static_cast<float>(batch_size) / min_grid_size;
        if (scale > 1) {
            if (0.8 * scale > 1) { scale = 0.8 * scale; }
            block_size /= scale;
            if (block_size < 1) { block_size = 1; }
            block_size = Utils::round_up_to_multiple_of<Utils::WARP_SIZE>(block_size);
        }
        num_of_warp = block_size / Utils::WARP_SIZE;
        IdxT len_per_warp = (len - 1) / num_of_warp + 1;
        len_per_warp = Utils::round_up_to_multiple_of<Utils::WARP_SIZE>(len_per_warp);
        num_of_warp = (len - 1) / len_per_warp + 1;
        constexpr int len_factor = LaunchThreshold<WarpSortClass>::len_factor_for_single_block;
        if (len_per_warp < static_cast<IdxT>(capacity * len_factor)) {
            len_per_warp = capacity * len_factor;
            num_of_warp = (len - 1) / len_per_warp + 1;
        }
    }
    *p_num_of_block = num_of_block;
    *p_num_of_warp = Utils::round_up_to_multiple_of<4>(num_of_warp);
}

template<typename T, typename IdxT>
void calc_launch_parameter_for_merge(IdxT len, IdxT k, int* num_of_block, int* num_of_warp) {
    *num_of_block = 1;
    int block_size = 0;
    int min_grid_size = 0;
    calc_launch_parameter_by_occupancy<WarpMerge, T, IdxT>(k, &block_size, &min_grid_size);
    *num_of_warp = block_size / Utils::WARP_SIZE;
    IdxT len_per_warp = (len - 1) / (*num_of_warp) + 1;
    len_per_warp = ((len_per_warp - 1) / k + 1) * k;
    *num_of_warp = (len - 1) / len_per_warp + 1;
}

template<bool greater, template<int, bool, typename, typename> class WarpSortClass, typename T, typename IdxT>
void warp_sort_topk_impl(int num_of_block,
                         int num_of_warp,
                         void* buf,
                         size_t& buf_size,
                         const T* __restrict__ in,
                         int batch_size,
                         IdxT len,
                         IdxT k,
                         T* __restrict__ out,
                         IdxT* __restrict__ out_idx,
                         hipStream_t stream) {
    T* tmp_val = nullptr;
    IdxT* tmp_idx = nullptr;

    if (num_of_block > 1) {
        using Aligner256 = Utils::MemoryAligner<256>;
        size_t sizes[2]   = {sizeof(T) * num_of_block * k * batch_size,
                             sizeof(IdxT) * num_of_block * k * batch_size};
        size_t total_size = Aligner256::calculate_required_size(sizes);
        if (!buf) {
            buf_size = total_size;
            return;
        }
        auto aligned_pointers = Aligner256::partition_buffer(buf, sizes);
        tmp_val = static_cast<T*>(aligned_pointers[0]);
        tmp_idx = static_cast<IdxT*>(aligned_pointers[1]);
    } else if (!buf) {
        buf_size = 1;
        return;
    }

    T dummy = numeric::get_sentinel_value<greater, T>();
    T* result_val = (num_of_block == 1) ? out : tmp_val;
    IdxT* result_idx = (num_of_block == 1) ? out_idx : tmp_idx;
    int block_dim = num_of_warp * Utils::WARP_SIZE;
    int smem_size = calc_smem_size_for_block_wide<T, IdxT>(num_of_warp, k);
    auto block_kernel_func = find_block_kernel<greater, WarpSortClass, T, IdxT>(k);

    block_kernel_func<<<batch_size * num_of_block, block_dim, smem_size, stream>>>(
        in, static_cast<IdxT*>(nullptr), batch_size, len, k, result_val, result_idx, dummy);

    if (num_of_block > 1) {
        len = k * num_of_block;
        calc_launch_parameter_for_merge<T, IdxT>(len, k, &num_of_block, &num_of_warp);
        block_dim = num_of_warp * Utils::WARP_SIZE;
        smem_size = calc_smem_size_for_block_wide<T, IdxT>(num_of_warp, k);
        auto merge_kernel_func = find_block_kernel<greater, WarpMerge, T, IdxT>(k);

        merge_kernel_func<<<batch_size* num_of_block, block_dim, smem_size, stream>>>(
            tmp_val, tmp_idx, batch_size, len, k, out, out_idx, dummy);
    }
}


// NaN contract: the bitonic sort comparators (dev_med3, select_idx) use IEEE
// equality, so NaN values break sort stability and may corrupt index tracking.
// Callers must ensure `in` is NaN-free.  In beam search this holds because
// logits pass through log_softmax before entering topk.
template<bool greater, typename T, typename IdxT>
void WarpSortTopk(void* buf,
                      size_t& buf_size,
                      const T* __restrict__ in,
                      int batch_size,
                      IdxT len,
                      IdxT k,
                      T* __restrict__ out,
                      IdxT* __restrict__ out_idx = nullptr,
                      hipStream_t stream = 0) {
    assert(k <= buffer_load_helpers::MAX_CAPACITY);
    const int capacity = Utils::calc_capacity(k);
    int num_of_block = 0;
    int num_of_warp = 0;

    calc_launch_parameter<WarpBitonic, T, IdxT>(batch_size, len, k, &num_of_block, &num_of_warp);
    int len_per_warp = (num_of_block * num_of_warp == 0) ? len : len / (num_of_block * num_of_warp);

    if (len_per_warp <= static_cast<IdxT>(capacity) * LaunchThreshold<WarpBitonic>::len_factor_for_choosing) {
        warp_sort_topk_impl<greater, WarpBitonic, T, IdxT>(
            num_of_block, num_of_warp, buf, buf_size, in, batch_size, len, k, out, out_idx, stream);
    } else {
        num_of_block = 1;
        num_of_warp = 2;
        warp_sort_topk_impl<greater, WarpSelect, T, IdxT>(
            num_of_block, num_of_warp, buf, buf_size, in, batch_size, len, k, out, out_idx, stream);
    }
}

} // namespace HipKernels
