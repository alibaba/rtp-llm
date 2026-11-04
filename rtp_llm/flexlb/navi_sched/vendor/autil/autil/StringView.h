#pragma once

#include <functional>
#include <string_view>

#include "autil/Span.h" // IWYU pragma: export

namespace std {

template <>
struct hash<autil::StringView> {
    std::size_t operator()(const autil::StringView &key) const {
        return std::hash<std::string_view>()(std::string_view(key.data(), key.size()));
    }
};

} // namespace std
