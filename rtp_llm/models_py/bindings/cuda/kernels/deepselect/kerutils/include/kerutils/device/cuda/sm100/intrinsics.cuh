#pragma once

// MIT DeepSelect d96d33afe1fa: retain only the two SM100 operations used
// by the BF16 selector; unrelated TMEM/MMA/CLC helpers are excluded.

// LDG.256 or LDG.256 with non-coherent cache
// (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ld) We use macro
// instead of function here, since we need a multi-level recursive dispatch based on template parameters if using
// function NC_STR should be either "" or ".nc" L1_CACHE_HINT_STR should be either "evict_first", "evict_normal",
// "evict_last", "evict_unchanged", or "no_allocate" L2_CACHE_HINT_STR should be either "evict_first", "evict_normal",
// or "evict_last" L2_PREFETCH_SIZE_STR should be either "64B", "128B", or "256B"
#define KU_LDG_256(global_addr, result, NC_STR, L1_CACHE_HINT_STR, L2_CACHE_HINT_STR, L2_PREFETCH_SIZE_STR)            \
    {                                                                                                                  \
        static_assert(std::is_pointer_v<decltype(global_addr)> || std::is_array_v<decltype(global_addr)>,              \
                      "`global_addr` must be a pointer");                                                              \
        static_assert(std::is_pointer_v<decltype(result)> || std::is_array_v<decltype(result)>,                        \
                      "`result` must be a pointer");                                                                   \
        uint64_t* result_as_uint64_ptr = (uint64_t*)(result);                                                          \
        asm volatile("ld.global" NC_STR ".L1::" L1_CACHE_HINT_STR ".L2::" L2_CACHE_HINT_STR                            \
                     ".L2::" L2_PREFETCH_SIZE_STR ".v4.u64 {%0, %1, %2, %3}, [%4];\n"                                  \
                     : "=l"(result_as_uint64_ptr[0]),                                                                  \
                       "=l"(result_as_uint64_ptr[1]),                                                                  \
                       "=l"(result_as_uint64_ptr[2]),                                                                  \
                       "=l"(result_as_uint64_ptr[3])                                                                   \
                     : "l"(global_addr));                                                                              \
    }

// STG.256 (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st)
// L1_CACHE_HINT_STR should be either "evict_first", "evict_normal", "evict_last", "evict_unchanged", or "no_allocate"
// L2_CACHE_HINT_STR should be either "evict_first", "evict_normal", or "evict_last"
#define KU_STG_256(global_addr, src, L1_CACHE_HINT_STR, L2_CACHE_HINT_STR)                                             \
    {                                                                                                                  \
        static_assert(std::is_pointer_v<decltype(global_addr)> || std::is_array_v<decltype(global_addr)>,              \
                      "`global_addr` must be a pointer");                                                              \
        static_assert(std::is_pointer_v<decltype(src)> || std::is_array_v<decltype(src)>, "`src` must be a pointer");  \
        uint64_t const* src_as_uint64_ptr = (uint64_t const*)(src);                                                    \
        asm volatile("st.global.L1::" L1_CACHE_HINT_STR ".L2::" L2_CACHE_HINT_STR ".v4.u64 [%0], {%1, %2, %3, %4};\n"  \
                     :                                                                                                 \
                     : "l"(global_addr),                                                                               \
                       "l"(src_as_uint64_ptr[0]),                                                                      \
                       "l"(src_as_uint64_ptr[1]),                                                                      \
                       "l"(src_as_uint64_ptr[2]),                                                                      \
                       "l"(src_as_uint64_ptr[3]));                                                                     \
    }
