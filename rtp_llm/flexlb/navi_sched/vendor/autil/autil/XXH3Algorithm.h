#pragma once

#include <stddef.h>
#include <stdint.h>

#include "autil/xxh3.h"

namespace autil {

class XXH3Algorithm {
public:
    static inline uint64_t hashString64(const char *str, size_t size) { return XXH3_64bits(str, size); }
};

} // namespace autil
