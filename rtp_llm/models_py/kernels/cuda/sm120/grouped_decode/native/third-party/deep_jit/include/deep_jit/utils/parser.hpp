#pragma once

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/hash.hpp>

namespace deep_jit {

class Parser {
public:
    const std::vector<std::filesystem::path> include_dirs;
    const std::vector<std::string> include_prefixes;
    std::unordered_map<std::string, std::string> cache;
    std::unordered_set<std::string> visiting;

    Parser(std::vector<std::filesystem::path> include_dirs, std::vector<std::string> include_prefixes)
        : include_dirs(std::move(include_dirs)), include_prefixes(std::move(include_prefixes)) {}

    static std::string_view trim_left(std::string_view value) {
        const auto begin = value.find_first_not_of(" \t\f\v");
        return begin == std::string_view::npos ? std::string_view{} : value.substr(begin);
    }

    static bool is_space(const char value) {
        return value == ' ' or value == '\t' or value == '\f' or value == '\v';
    }

    std::string_view parse_include(std::string_view line) const {
        line = trim_left(line);
        if (not line.starts_with('#'))
            return {};

        line = trim_left(line.substr(1));
        constexpr std::string_view directive = "include";
        if (not line.starts_with(directive))
            return {};

        line = trim_left(line.substr(directive.size()));
        if (line.empty() or (line.front() != '<' and line.front() != '"'))
            return {};

        const auto closing = line.find_first_of(">\"", 1);
        if (closing == std::string_view::npos)
            return {};

        const auto include = line.substr(0, closing + 1);
        const auto filename = line.substr(1, closing - 1);
        if (line.front() != '<' or line[closing] != '>' or filename.empty() or is_space(filename.front()) or is_space(filename.back())) {
            DJ_PANIC("non-standard include: {}", include);
        }

        for (const auto& prefix : include_prefixes) {
            if (filename.starts_with(prefix))
                return filename;
        }
        return {};
    }

    std::string parse_includes_into_hash(std::string_view filename) {
        // Check cache
        std::string key(filename);
        if (const auto iterator = cache.find(key); iterator != cache.end())
            return iterator->second;

        // No circular include
        if (visiting.contains(key))
            DJ_PANIC("circular include may occur: {}", key);
        visiting.insert(key);

        try {
            // Find path
            std::filesystem::path path;
            for (const auto& include_dir: include_dirs) {
                const auto try_path = include_dir / filename;
                if (std::filesystem::exists(try_path)) {
                    path = try_path;
                    break;
                }
            }

            // Read code and get hash
            std::ifstream input(path);
            DJ_HOST_ASSERT(input.is_open(), "failed to open: {}", path.string());
            const std::string code{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
            const auto hash = parse_into_hash(code);

            // Erase and return
            visiting.erase(key);
            return cache[key] = hash;
        } catch (...) {
            visiting.erase(key);
            throw;
        }
    }

    std::string parse_into_hash(const std::string& code) {
        // Parse code itself
        hash::FNV1a hash;
        hash.update(code);

        // Parse include files
        auto view = std::string_view(code);
        while (not view.empty()) {
            const auto newline = view.find('\n');
            const auto line = view.substr(0, newline);
            if (const auto filename = parse_include(line); not filename.empty())
                hash.update(parse_includes_into_hash(filename));
            view.remove_prefix(newline == std::string_view::npos ? view.size() : newline + 1);
        }
        return hash.get_hex_digest();
    }
};

}  // namespace deep_jit
