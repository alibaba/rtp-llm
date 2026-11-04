#pragma once

#include <algorithm>
#include <assert.h>
#include <iosfwd>
#include <stdint.h>
#include <string.h>
#include <string>

#include "autil/Span.h"

namespace autil {

namespace detail {

template <typename Allocator>
StringView MakeCStringImpl(const char *data, std::size_t size, Allocator *allocator) {
    // allocate one more byte '\0', but size not change
    // it makes StringView easier to debug
    // example: data = "abc", size = 3 => _data="abc\0",_size=3
    // then you print _data then show "abc" bur data (withou \0) not
    char *buf = (char *)allocator->allocate(size + 1);
    if (size > 0) {
        memcpy(buf, data, size);
    }
    buf[size] = '\0';
    return {buf, size};
}

} // namespace detail

template <typename Allocator>
StringView MakeCString(const char *data, Allocator *allocator) {
    return detail::MakeCStringImpl(data, ::strlen(data), allocator);
}

template <typename Allocator>
StringView MakeCString(const char *data, std::size_t size, Allocator *allocator) {
    return detail::MakeCStringImpl(data, size, allocator);
}

template <typename Allocator>
StringView MakeCString(const StringView &str, Allocator *allocator) {
    return detail::MakeCStringImpl(str.data(), str.size(), allocator);
}

} // namespace autil
