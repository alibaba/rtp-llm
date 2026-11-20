#pragma once

#include <algorithm>
#include <filesystem>
#include <fmt/format.h>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/filesystem.hpp>
#include <deep_jit/utils/uuid.hpp>

namespace deep_jit {

inline constexpr std::string_view kCommitFileName = ".committed";

// One disk-cache slot. On a miss the entry points at a private temporary
// directory; commit() publishes it atomically (whole-directory rename) after
// fsync, so concurrent processes on (distributed) filesystems observe either
// nothing or a complete entry. Uncommitted temporaries are removed on
// destruction, including on exception paths.
class DiskCacheEntry {
public:
    bool hit = false, committed = false;

    // `path` is the temporary path while building, commit path after commit
    std::filesystem::path path;
    std::filesystem::path commit_path;

    DiskCacheEntry(const DiskCacheEntry&) = delete;
    DiskCacheEntry& operator=(const DiskCacheEntry&) = delete;
    DiskCacheEntry(DiskCacheEntry&&) = delete;
    DiskCacheEntry& operator=(DiskCacheEntry&&) = delete;

    DiskCacheEntry(const bool hit, std::filesystem::path path, std::filesystem::path commit_path = {})
        : hit(hit), path(std::move(path)), commit_path(std::move(commit_path)) {}

    std::filesystem::path commit() {
        if (hit or committed)
            return path;

        // Mark then fsync the whole tree before publishing
        write_file_sync(path / kCommitFileName, "");
        fsync_dir(path);

        // Atomically rename the temporary directory to the final cache path
        // NOTES: if another rank already created dir_path, rename will fail — that's fine
        make_dirs(commit_path.parent_path());
        std::error_code error_code;
        std::filesystem::rename(path, commit_path, error_code);
        if (error_code) {
            // Another rank beat us, then clean up our dir and use the existing one
            // NOTES: avoid `std::filesystem::remove_all` here — it can segfault on
            // distributed filesystems, when concurrent processes operate
            // on the same parent directory, causing stale directory entries
            safe_remove_all(path);
        }

        path = commit_path;
        hit = true;
        committed = true;
        return path;
    }

    ~DiskCacheEntry() {
        // Best-effort cleanup; abrupt process termination may bypass the destructor.
        if (not hit and not committed)
            safe_remove_all(path);
    }
};

// Directory-granularity cache on a possibly distributed filesystem.
// Entries live at `<root>/cache/<tag>.<digest>/` and are published with a
// ".committed" marker; builds go through `<root>/tmp/<uuid>`.
struct DiskCache {
    const std::vector<std::filesystem::path> paths = {};

    explicit DiskCache(std::vector<std::filesystem::path> paths)
        : paths(std::move(paths)) {
        DJ_HOST_ASSERT(not this->paths.empty());
    }

    static DiskCache from_env(const Env& env) {
        // Parse PATH_1:PATH_2:PATH_3 into a writable cache root followed by
        // zero or more read-only lookup roots.
        std::vector<std::filesystem::path> paths;
        if (const auto env_opt = env.get<std::string>("JIT_CACHE_DIR")) {
            size_t begin = 0;
            while (true) {
                const auto end = env_opt->find(':', begin);
                const auto item = env_opt->substr(begin, end - begin);
                DJ_HOST_ASSERT(not item.empty(), "disk cache path list contains an empty path: {}", env_opt.value());
                paths.emplace_back(item);
                if (end == std::string_view::npos)
                    break;
                begin = end + 1;
            }
        } else {
            const auto home = get_env<std::string>("HOME");
            DJ_HOST_ASSERT(not home.empty(), "HOME environment variable must not be empty");
            paths.emplace_back(std::filesystem::path(home) / ".dj");
        }

        return DiskCache(paths);
    }

    [[nodiscard]] DiskCacheEntry entry(const std::string& tag, const std::string& digest) const {
        const auto is_valid_char = [](const char& c) {
            return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or
                   (c >= '0' and c <= '9') or c == '_';
        };
        DJ_HOST_ASSERT(not tag.empty() and std::ranges::all_of(tag, is_valid_char),
                       "cache tag must contain only letters, digits, or underscores: {}", tag);

        // Check all cache dirs
        const auto entry_name = fmt::format("{}.{}", tag, digest);
        for (const auto& dir: paths) {
            const auto path = dir / "cache" / entry_name;
            if (std::filesystem::exists(path / kCommitFileName)) {
                try_update_mtime(path / kCommitFileName);
                return DiskCacheEntry{true, path};
            }
        }

        // Miss, create an empty temporary directory
        const auto temporary_path = paths[0] / "tmp" / get_uuid();
        std::filesystem::create_directories(temporary_path);
        return DiskCacheEntry{false, temporary_path, paths[0] / "cache" / entry_name};
    }
};

}  // namespace deep_jit
