#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace rtp_llm {

// Internal URI metadata contract shared with the origin-aware Manager. All
// offsets are exclusive computed-KV ends, never sampled-token ownership.
struct DecodeCacheOrigin {
    std::string session;
    int64_t     handoff_end{0}, begin{0}, end{0}, computed_end{0}, sampled_end{0};

    static std::optional<DecodeCacheOrigin> parse(std::string_view value) {
        if (value.size() > 140 || value.substr(0, 3) != "v1.")
            return std::nullopt;
        std::array<std::string_view, 6> fields;
        size_t                          offset = 3;
        for (size_t i = 0; i < fields.size(); ++i) {
            const auto next = value.find('.', offset);
            if ((i + 1 == fields.size()) != (next == std::string_view::npos))
                return std::nullopt;
            const auto end = next == std::string_view::npos ? value.size() : next;
            fields[i]      = value.substr(offset, end - offset);
            offset         = end + 1;
        }
        if (fields[0].size() != 32)
            return std::nullopt;
        for (const char c : fields[0])
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')))
                return std::nullopt;
        DecodeCacheOrigin       result;
        std::array<int64_t*, 5> output{
            &result.handoff_end, &result.begin, &result.end, &result.computed_end, &result.sampled_end};
        for (size_t i = 0; i < output.size(); ++i) {
            const auto field = fields[i + 1];
            if (field.empty() || field.front() < '0' || field.front() > '9'
                || (field.size() > 1 && field.front() == '0'))
                return std::nullopt;
            const auto [last, error] = std::from_chars(field.data(), field.data() + field.size(), *output[i]);
            if (error != std::errc{} || last != field.data() + field.size())
                return std::nullopt;
        }
        if (result.begin >= result.end || result.end > result.computed_end || result.handoff_end > result.computed_end
            || result.computed_end == std::numeric_limits<int64_t>::max()
            || result.sampled_end != result.computed_end + 1)
            return std::nullopt;
        result.session = std::string(fields[0]);
        return result;
    }

    std::optional<std::string> encode() const {
        auto value = "v1." + session + "." + std::to_string(handoff_end) + "." + std::to_string(begin) + "."
                     + std::to_string(end) + "." + std::to_string(computed_end) + "." + std::to_string(sampled_end);
        return parse(value) ? std::optional<std::string>(std::move(value)) : std::nullopt;
    }

    bool operator==(const DecodeCacheOrigin& other) const {
        return session == other.session && handoff_end == other.handoff_end && begin == other.begin && end == other.end
               && computed_end == other.computed_end && sampled_end == other.sampled_end;
    }

    int64_t logicalDecodeTokens(int64_t observed_end) const {
        const auto upper = std::min(end, observed_end);
        const auto lower = std::max(begin, handoff_end);
        return lower >= 0 && upper > lower ? upper - lower : 0;
    }
};

namespace decode_origin_detail {
inline std::optional<std::map<std::string, std::string>> queryFields(std::string_view uri) {
    if (uri.empty() || uri.size() > 16384 || uri.find('#') != std::string_view::npos)
        return std::nullopt;
    std::map<std::string, std::string> result;
    const auto                         question = uri.find('?');
    size_t                             begin    = question == std::string_view::npos ? uri.size() : question + 1;
    while (begin < uri.size()) {
        auto end = uri.find('&', begin);
        if (end == std::string_view::npos)
            end = uri.size();
        const auto equal = uri.find('=', begin);
        if (equal == std::string_view::npos || equal >= end || equal == begin
            || !result
                    .emplace(std::string(uri.substr(begin, equal - begin)),
                             std::string(uri.substr(equal + 1, end - equal - 1)))
                    .second)
            return std::nullopt;
        begin = end + 1;
    }
    return result;
}
}  // namespace decode_origin_detail

inline std::optional<DecodeCacheOrigin> decodeCacheOriginFromUri(std::string_view uri) {
    const auto fields = decode_origin_detail::queryFields(uri);
    if (!fields)
        return std::nullopt;
    const auto found = fields->find("dtop_origin");
    return found == fields->end() ? std::nullopt : DecodeCacheOrigin::parse(found->second);
}

// Preserve every original query value. The Manager independently verifies that
// this is its exact allocation, not a writer-selected address or changed core.
inline std::optional<std::string> attachDecodeCacheOrigin(std::string_view uri, const DecodeCacheOrigin& origin) {
    auto       fields = decode_origin_detail::queryFields(uri);
    const auto value  = origin.encode();
    if (!fields || !value || !fields->emplace("dtop_origin", *value).second)
        return std::nullopt;
    std::string result(uri.substr(0, uri.find('?')));
    char        separator = '?';
    for (const auto& [key, data] : *fields) {
        result += separator;
        result += key + "=" + data;
        separator = '&';
    }
    return result;
}

}  // namespace rtp_llm
