#pragma once

#include <map>
#include <set>
#include <vector>

#include "fmt/core.h"
#include "fmt/format.h"

namespace autil {

struct FormatterWithDefaultParse {
    template <typename ParseContext>
    constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }
};

} // namespace autil

template <typename T>
struct fmt::formatter<std::vector<T>> : autil::FormatterWithDefaultParse {
    template <typename FormatContext>
    auto format(const std::vector<T> &values, FormatContext &ctx) {
        auto &&out = ctx.out();
        fmt::formatter<char>().format('[', ctx);
        for (const auto &value : values) {
            fmt::formatter<T>().format(value, ctx);
            fmt::formatter<char>().format(',', ctx);
        }
        fmt::formatter<char>().format(']', ctx);
        return out;
    }
};

template <typename T>
struct fmt::formatter<std::set<T>> : autil::FormatterWithDefaultParse {
    template <typename FormatContext>
    auto format(const std::set<T> &values, FormatContext &ctx) {
        auto &&out = ctx.out();
        fmt::formatter<char>().format('{', ctx);
        for (const auto &value : values) {
            fmt::formatter<T>().format(value, ctx);
            fmt::formatter<char>().format(',', ctx);
        }
        fmt::formatter<char>().format('}', ctx);
        return out;
    }
};

template <typename K, typename V>
struct fmt::formatter<std::pair<K, V>> : autil::FormatterWithDefaultParse {
    template <typename FormatContext>
    auto format(const std::pair<K, V> &value, FormatContext &ctx) {
        auto &&out = ctx.out();
        fmt::formatter<char>().format('<', ctx);
        fmt::formatter<K>().format(value.first, ctx);
        fmt::formatter<char>().format(',', ctx);
        fmt::formatter<V>().format(value.second, ctx);
        fmt::formatter<char>().format('>', ctx);
        return out;
    }
};

template <typename K, typename V>
struct fmt::formatter<std::map<K, V>> : autil::FormatterWithDefaultParse {
    template <typename FormatContext>
    auto format(const std::map<K, V> &value, FormatContext &ctx) {
        auto &&out = ctx.out();
        fmt::formatter<char>().format('{', ctx);
        for (const auto &pair : value) {
            fmt::formatter<K>().format(pair.first, ctx);
            fmt::formatter<char>().format(':', ctx);
            fmt::formatter<V>().format(pair.second, ctx);
            fmt::formatter<char>().format(',', ctx);
        }
        fmt::formatter<char>().format('}', ctx);
        return out;
    }
};
