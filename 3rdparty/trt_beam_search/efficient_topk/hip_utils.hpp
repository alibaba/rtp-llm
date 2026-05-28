#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

//==================================================================================================
// Error Handling
//==================================================================================================

#define HIP_CHECK(val) \
  { HipKernels::Utils::hip_check_((val), __FILE__, __LINE__); }

namespace HipKernels {
namespace Utils {

constexpr int WARP_SIZE = 64;
constexpr unsigned long long FULL_WARP_MASK = 0xffffffffffffffffULL;

class HipException : public std::runtime_error {
 public:
  explicit HipException(const std::string& what) : runtime_error(what) {}
};

inline void hip_check_(hipError_t val, const char* file, int line) {
  if (val != hipSuccess) {
    throw HipException(std::string(file) + ":" + std::to_string(line) +
                       ": HIP error " + std::to_string(val) + ": " +
                       hipGetErrorString(val));
  }
}

/**
 * @class MemoryAligner
 * @brief A utility to manage memory alignment for sub-allocations within a
 * single buffer.
 */
template <size_t Alignment = 256>
class MemoryAligner {
 public:
  static_assert((Alignment > 0) && ((Alignment & (Alignment - 1)) == 0),
                "Alignment must be a non-zero power of two.");

  static inline size_t calculate_required_size(
      const std::vector<size_t>& sizes) {
    size_t total_aligned_size = 0;
    for (const auto size : sizes) {
      total_aligned_size += align_up(size);
    }
    return total_aligned_size + (Alignment - 1);
  }

  static inline std::vector<void*> partition_buffer(
      const void* buffer, const std::vector<size_t>& sizes) {
    auto* aligned_buffer_start =
        reinterpret_cast<char*>(align_up(reinterpret_cast<uintptr_t>(buffer)));

    std::vector<void*> pointers;
    pointers.reserve(sizes.size());

    char* current_ptr = aligned_buffer_start;
    for (const auto size : sizes) {
      pointers.push_back(current_ptr);
      current_ptr += align_up(size);
    }

    return pointers;
  }

 private:
  static constexpr size_t align_up(size_t value) {
    return (value + Alignment - 1) & ~(Alignment - 1);
  }
};

template <size_t Multiple, typename T>
__inline__ __host__ __device__ constexpr T round_up_to_multiple_of(T value) {
  if (value == 0) {
    return 0;
  }
  static_assert(Multiple > 0, "Multiple must be positive.");
  return ((value - 1) / Multiple + 1) * Multiple;
}

template <typename T>
__inline__ __host__ __device__ constexpr T round_up_to_multiple_of(
    T value, size_t multiple) {
  return value > 0 ? ((value - 1) / multiple + 1) * multiple : 0;
}

template <typename T>
__inline__ __host__ __device__ constexpr bool is_power_of_2(T value) {
  return (value && !(value & (value - 1)));
}

template <typename T>
__inline__ __host__ __device__ constexpr T ceil_to_power_of_2(T value) {
  if (value <= 1) {
    return 1;
  }
  T v = value - 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  if constexpr (sizeof(T) >= 2) v |= v >> 8;
  if constexpr (sizeof(T) >= 4) v |= v >> 16;
  if constexpr (sizeof(T) >= 8) v |= v >> 32;
  return v + 1;
}

template <typename T>
__inline__ __host__ __device__ constexpr int integer_log2(T n, int p = 0) {
  return (n <= 1) ? p : integer_log2(n / 2, p + 1);
}

__inline__ __host__ __device__ constexpr int calc_capacity(int k) {
  int capacity = HipKernels::Utils::ceil_to_power_of_2(k);
  return (capacity < WARP_SIZE) ? WARP_SIZE : capacity;
}

}  // namespace Utils

namespace numeric {

template <typename T>
__inline__ constexpr T get_lower_bound() {
  static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic.");
  if constexpr (std::is_floating_point_v<T> && std::is_signed_v<T>) {
    return -std::numeric_limits<T>::infinity();
  } else {
    return std::numeric_limits<T>::lowest();
  }
}

template <typename T>
__inline__ constexpr T get_upper_bound() {
  static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic.");
  if constexpr (std::is_floating_point_v<T>) {
    return std::numeric_limits<T>::infinity();
  } else {
    return std::numeric_limits<T>::max();
  }
}

template <bool FindLargest, typename T>
__inline__ constexpr T get_sentinel_value() {
  if constexpr (FindLargest) {
    static_assert(!std::is_unsigned_v<T>,
                  "Cannot determine a meaningful lower bound for finding the "
                  "'largest' unsigned value.");
    return get_lower_bound<T>();
  } else {
    return get_upper_bound<T>();
  }
}

template <bool FindLargest, typename T>
__device__ __host__ constexpr bool is_preferred(T val, T baseline) {
  if constexpr (FindLargest) {
    return val > baseline;
  } else {
    return val < baseline;
  }
}

}  // namespace numeric
}  // namespace HipKernels
