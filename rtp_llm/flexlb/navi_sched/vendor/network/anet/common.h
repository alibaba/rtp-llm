#pragma once

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

#define BEGIN_ANET_NS() namespace anet {
#define END_ANET_NS() }
#define USE_ANET_NS() using namespace anet;
#define DELETE_AND_SET_NULL_ANET(x)                                                                                    \
    if (x) {                                                                                                           \
        delete x;                                                                                                      \
        x = NULL;                                                                                                      \
    }

