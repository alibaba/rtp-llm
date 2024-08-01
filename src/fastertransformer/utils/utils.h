#pragma once

#define FT_SWITCH(COND, CONST_NAME, ...)                \
    [&] {                                               \
        if (COND) {                                     \
            constexpr static bool CONST_NAME = true;    \
            return __VA_ARGS__();                       \
        } else {                                        \
            constexpr static bool CONST_NAME = false;   \
            return __VA_ARGS__();                       \
        }                                               \
    }()

#define FT_SWITCH_V(COND, V, A, B, ...)         \
    [&] {                                       \
        if (COND) {                             \
            constexpr auto V = A;               \
            return __VA_ARGS__();               \
        } else {                                \
            constexpr auto V = B;               \
            return __VA_ARGS__();               \
        }                                       \
    }()

#define FT_SWITCH_T(COND, NAME, T_TYPE, F_TYPE, ...)    \
    [&] {                                               \
        if (COND) {                                     \
            typedef T_TYPE NAME;                        \
            return __VA_ARGS__();                       \
        } else {                                        \
            typedef F_TYPE NAME;                        \
            return __VA_ARGS__();                       \
        }                                               \
    }()

#define FT_SWITCH_ONE_CASE(CONST_NAME, VALUE, ...)      \
    case VALUE:                                         \
    {                                                   \
        constexpr static auto CONST_NAME = VALUE;       \
        return __VA_ARGS__();                           \
    }

#define FT_SWITCH_DEFAULT_CASE(CONST_NAME, VALUE, ...)  \
    default:                                            \
    {                                                   \
        constexpr static auto CONST_NAME = VALUE;       \
        return __VA_ARGS__();                           \
    }
