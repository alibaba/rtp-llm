#pragma once

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/json.hpp>

namespace deep_jit {

struct Config {
    std::filesystem::path python_library_root;
    std::string env_prefix;
    std::string extra_signature;
    std::vector<std::filesystem::path> include_dirs;
    std::vector<std::string> include_prefixes;

    explicit Config(std::filesystem::path python_library_root,
                    std::string env_prefix,
                    std::string extra_signature = {},
                    std::vector<std::filesystem::path> include_dirs = {},
                    std::vector<std::string> include_prefixes = {})
        : python_library_root(std::move(python_library_root)),
          env_prefix(std::move(env_prefix)),
          extra_signature(std::move(extra_signature)),
          include_dirs(std::move(include_dirs)),
          include_prefixes(std::move(include_prefixes)) {
        // Check library root
        DJ_HOST_ASSERT(not this->python_library_root.empty(), "Python library root must not be empty");
        DJ_HOST_ASSERT(this->python_library_root.is_absolute(),
                       "Python library root must be absolute: {}", this->python_library_root.string());
        this->python_library_root = this->python_library_root.lexically_normal();

        // Check environment prefix
        DJ_HOST_ASSERT(not this->env_prefix.empty(), "environment variable prefix must not be empty");
        DJ_HOST_ASSERT(this->env_prefix != "DJ", "DJ is reserved for global environment variables");

        // Check includes
        for (auto& include_dir: this->include_dirs) {
            DJ_HOST_ASSERT(not include_dir.empty(), "JIT include directory must not be empty");
            DJ_HOST_ASSERT(include_dir.is_absolute(), "JIT include directory must be absolute: {}", include_dir.string());
            include_dir = include_dir.lexically_normal();
        }
    }

    [[nodiscard]] std::filesystem::path get_python_path(const std::string& rel_path) const {
        return python_library_root / std::filesystem::path(rel_path).lexically_normal();
    }

    [[nodiscard]] json to_json() const {
        std::vector<std::string> include_dir_strings;
        include_dir_strings.reserve(include_dirs.size());
        for (const auto& include_dir : include_dirs)
            include_dir_strings.emplace_back(include_dir.string());
        return json::object_t {
            {"python_library_root", python_library_root.string()},
            {"extra_signature", extra_signature},
            {"include_dirs", include_dir_strings},
            {"include_prefixes", include_prefixes},
        };
    }
};

}  // namespace deep_jit
