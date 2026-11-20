#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace deep_jit::str {

inline std::string join(const std::vector<std::string>& values, const std::string_view separator = " ") {
    if (values.empty())
        return {};

    std::size_t size = separator.size() * (values.size() - 1);
    for (const auto& value : values)
        size += value.size();

    std::string result;
    result.reserve(size);
    result += values.front();
    for (std::size_t i = 1; i < values.size(); ++i) {
        result.append(separator);
        result.append(values[i]);
    }
    return result;
}

}  // namespace deep_jit::str
