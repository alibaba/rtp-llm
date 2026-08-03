// Minimal compatibility layer for SGLang's DeepSeek-v4 TopK v2 device code.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

using fp16_t = __half;

namespace device {

#define SGL_DEVICE __forceinline__ __device__

inline constexpr uint32_t kWarpThreads = 32;
inline constexpr std::size_t kMaxVecBytes = 16;

template <typename To, typename From>
SGL_DEVICE To cast(const From& value) {
    if constexpr (std::is_same_v<To, fp16_t> &&
                  std::is_same_v<From, float>) {
        return __float2half_rn(value);
    } else if constexpr (std::is_same_v<To, float> &&
                         std::is_same_v<From, fp16_t>) {
        return __half2float(value);
    } else {
        return static_cast<To>(value);
    }
}

template <typename T, std::size_t N>
struct alignas(sizeof(T) * N) AlignedStorage {
    T data[N];
};

template <typename T, std::size_t N>
struct AlignedVector {
  private:
    static_assert(N > 0 && (N & (N - 1)) == 0);
    static_assert(sizeof(T) * N <= kMaxVecBytes);
    using storage_t = AlignedStorage<T, N>;

  public:
    SGL_DEVICE void load(const void* ptr, int64_t offset = 0) {
        storage_ = reinterpret_cast<const storage_t*>(ptr)[offset];
    }
    SGL_DEVICE void load_unaligned(const void* ptr, int64_t offset = 0) {
        const auto* values = reinterpret_cast<const T*>(
            static_cast<const char*>(ptr) +
            offset * static_cast<int64_t>(sizeof(storage_t)));
#pragma unroll
        for (std::size_t i = 0; i < N; ++i) {
            storage_.data[i] = values[i];
        }
    }
    SGL_DEVICE void store(void* ptr, int64_t offset = 0) const {
        reinterpret_cast<storage_t*>(ptr)[offset] = storage_;
    }
    SGL_DEVICE void fill(T value) {
#pragma unroll
        for (std::size_t i = 0; i < N; ++i) storage_.data[i] = value;
    }
    SGL_DEVICE T& operator[](std::size_t i) { return storage_.data[i]; }
    SGL_DEVICE T operator[](std::size_t i) const {
        return storage_.data[i];
    }

  private:
    storage_t storage_;
};

namespace warp {

template <uint32_t kNumThreads = kWarpThreads, bool kInner = true, typename T>
SGL_DEVICE T reduce_sum(
    T value, uint32_t active_mask = 0xffffffffu) {
    static_assert(kNumThreads >= 1 && kNumThreads <= kWarpThreads);
    static_assert((kNumThreads & (kNumThreads - 1)) == 0);
    if constexpr (kInner) {
#pragma unroll
        for (uint32_t mask = kNumThreads / 2; mask >= 1; mask >>= 1) {
            value += __shfl_xor_sync(active_mask, value, mask, 32);
        }
    } else {
#pragma unroll
        for (uint32_t mask = kNumThreads;
             mask <= kWarpThreads / 2;
             mask <<= 1) {
            value += __shfl_xor_sync(active_mask, value, mask, 32);
        }
    }
    return value;
}

}  // namespace warp

template <typename T, typename U>
SGL_DEVICE constexpr auto div_ceil(T a, U b) {
    return (a + b - 1) / b;
}

SGL_DEVICE void enable_smem_spilling() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm(".pragma \"enable_smem_spilling\";");
#endif
}

}  // namespace device
