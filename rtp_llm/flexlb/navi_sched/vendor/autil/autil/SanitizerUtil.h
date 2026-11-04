#pragma once
#include "autil/CommonMacros.h"

#ifdef AUTIL_HAVE_ADDRESS_SANITIZER
#include <sanitizer/asan_interface.h>
#endif

#ifdef AUTILL_HAVE_MEMORY_SANITIZER
#include <sanitizer/msan_interface.h>
#endif

namespace autil::SanitizerUtil {
// Helper functions for asan and msan.
inline void PoisonMemoryRegion(const void *m, size_t s) {
#ifdef AUTIL_HAVE_ADDRESS_SANITIZER
    __asan_poison_memory_region(m, s);
#endif
#ifdef AUTIL_HAVE_MEMORY_SANITIZER
    __msan_poison(m, s);
#endif
    (void)m;
    (void)s;
}

inline void UnpoisonMemoryRegion(const void *m, size_t s) {
#ifdef AUTIL_HAVE_ADDRESS_SANITIZER
    __asan_unpoison_memory_region(m, s);
#endif
#ifdef AUTIL_HAVE_MEMORY_SANITIZER
    __msan_unpoison(m, s);
#endif
    (void)m;
    (void)s;
}

template <typename T>
inline void PoisonObject(const T *object) {
    PoisonMemoryRegion(object, sizeof(T));
}

template <typename T>
inline void UnpoisonObject(const T *object) {
    UnpoisonMemoryRegion(object, sizeof(T));
}

} // namespace autil::SanitizerUtil
