#pragma once

#define FT_SWITCH(COND, CONST_NAME, ...)                                                                               \
    [&] {                                                                                                              \
        if (COND) {                                                                                                    \
            constexpr static bool CONST_NAME = true;                                                                   \
            return __VA_ARGS__();                                                                                      \
        } else {                                                                                                       \
            constexpr static bool CONST_NAME = false;                                                                  \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
    }()

#define FT_SWITCH_V(COND, V, A, B, ...)                                                                                \
    [&] {                                                                                                              \
        if (COND) {                                                                                                    \
            constexpr auto V = A;                                                                                      \
            return __VA_ARGS__();                                                                                      \
        } else {                                                                                                       \
            constexpr auto V = B;                                                                                      \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
    }()

#define FT_SWITCH_T(COND, NAME, T_TYPE, F_TYPE, ...)                                                                   \
    [&] {                                                                                                              \
        if (COND) {                                                                                                    \
            typedef T_TYPE NAME;                                                                                       \
            return __VA_ARGS__();                                                                                      \
        } else {                                                                                                       \
            typedef F_TYPE NAME;                                                                                       \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
    }()

#define FT_SWITCH_ONE_CASE(CONST_NAME, VALUE, ...)                                                                     \
    case VALUE: {                                                                                                      \
        constexpr static auto CONST_NAME = VALUE;                                                                      \
        return __VA_ARGS__();                                                                                          \
    }

#define FT_SWITCH_DEFAULT_CASE(CONST_NAME, VALUE, ...)                                                                 \
    default: {                                                                                                         \
        constexpr static auto CONST_NAME = VALUE;                                                                      \
        return __VA_ARGS__();                                                                                          \
    }

#define FT_SWITCH_ONE_CASE_T(NAME, VALUE, T_TYPE, ...)                                                                 \
    case VALUE: {                                                                                                      \
        typedef T_TYPE NAME;                                                                                           \
        return __VA_ARGS__();                                                                                          \
    }

#if defined(ENABLE_FP8) || defined(USING_ROCM)
#define ENABLE_FP8_CASE(NAME, TYPE, ...) FT_SWITCH_ONE_CASE_T(NAME, KvCacheDataType::FP8, TYPE, __VA_ARGS__)
#else
#define ENABLE_FP8_CASE(NAME, TYPE, ...)
#endif

#define FT_SWITCH_KV_CACHE_TYPE_CASE(COND, NAME, ...)                                                                  \
    [&] {                                                                                                              \
        switch (COND) {                                                                                                \
            FT_SWITCH_ONE_CASE_T(NAME, KvCacheDataType::BASE, T, __VA_ARGS__)                                          \
            FT_SWITCH_ONE_CASE_T(NAME, KvCacheDataType::INT8, int8_t, __VA_ARGS__)                                     \
            ENABLE_FP8_CASE(NAME, __nv_fp8_e4m3, __VA_ARGS__)                                                          \
        }                                                                                                              \
    }()

#define FT_SWITCH_KV_CACHE_TYPE_NON_INT8_CASE(COND, NAME, ...)                                                         \
    [&] {                                                                                                              \
        switch (COND) {                                                                                                \
            FT_SWITCH_ONE_CASE_T(NAME, KvCacheDataType::BASE, T, __VA_ARGS__)                                          \
            ENABLE_FP8_CASE(NAME, __nv_fp8_e4m3, __VA_ARGS__)                                                          \
        }                                                                                                              \
    }()
