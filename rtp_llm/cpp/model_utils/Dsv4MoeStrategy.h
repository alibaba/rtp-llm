#pragma once

#include <string>
#include <string_view>

namespace rtp_llm {

// Mirrors the Python descriptor: an explicit strategy takes precedence over
// the legacy DSv4 environment alias.
inline std::string resolveDsv4MoeStrategy(std::string_view requested, const char* alias) {
    auto trim = [](std::string_view value) {
        const auto first = value.find_first_not_of(" \t\n\r\f\v");
        if (first == std::string_view::npos) {
            return std::string_view{};
        }
        return value.substr(first, value.find_last_not_of(" \t\n\r\f\v") - first + 1);
    };
    requested = trim(requested);
    if (requested.empty() || requested == "auto") {
        if (alias != nullptr) {
            const std::string_view alias_view = trim(alias);
            if (alias_view == "sm120_decode" || alias_view == "sm120_fused_moe") {
                return std::string(alias_view);
            }
        }
        return "auto";
    }
    return std::string(requested);
}

inline std::string
resolveModelMoeStrategy(std::string_view model_type, std::string_view requested, const char* dsv4_alias) {
    const bool uses_dsv4_descriptor =
        model_type == "deepseek_v4" || model_type == "deepseek_v4_mtp" || model_type == "deepseek_v4_dspark";
    return resolveDsv4MoeStrategy(requested, uses_dsv4_descriptor ? dsv4_alias : nullptr);
}

}  // namespace rtp_llm
