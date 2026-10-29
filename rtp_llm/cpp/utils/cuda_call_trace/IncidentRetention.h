#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace rtp_llm::cuda_trace::detail {

struct ScopeEvidence {
    const char* name            = nullptr;
    uint64_t    parent_scope_id = 0;
    uint64_t    batch_size      = 0;
};

// A long CUDA call is a capture candidate, not proof of a hang. Requiring a
// request or a known non-empty engine batch excludes ordinary peer idle waits.
// The bound also prevents malformed/lost scope ancestry from looping forever.
template<typename Lookup>
bool longDemandCall(uint64_t begin_ns, uint64_t now_ns, int64_t request_id, uint64_t scope_id, Lookup lookup) {
    constexpr uint64_t threshold_ns = 5000000000ULL;
    if (now_ns < begin_ns || now_ns - begin_ns < threshold_ns) {
        return false;
    }
    if (request_id >= 0) {
        return true;
    }
    for (unsigned depth = 0; scope_id && depth != 64; ++depth) {
        ScopeEvidence scope{};
        if (!lookup(scope_id, scope)) {
            return false;
        }
        if (scope.name && std::strcmp(scope.name, "engine.execute") == 0 && scope.batch_size > 0) {
            return true;
        }
        scope_id = scope.parent_scope_id;
    }
    return false;
}

// Caller has flushed/closed its current segment. It must unlink a ring slot
// before reopening it, otherwise truncating that inode would destroy the pin.
// The single owned directory is replaced only after the previous trigger exits.
// Hard links keep ring + one retained incident bounded to at most two rings;
// there is intentionally no cross-filesystem copying fallback.
struct IncidentDirectory {
    bool     valid  = false;
    uint64_t device = 0;
    uint64_t inode  = 0;
};

inline IncidentDirectory directoryIdentity(const std::filesystem::path& path) {
    struct stat status{};
    if (::lstat(path.c_str(), &status) != 0 || !S_ISDIR(status.st_mode)) {
        throw std::runtime_error("incident directory identity is unavailable or is not a directory");
    }
    return {true, static_cast<uint64_t>(status.st_dev), static_cast<uint64_t>(status.st_ino)};
}

inline bool sameDirectory(const IncidentDirectory& left, const IncidentDirectory& right) {
    return left.valid && right.valid && left.device == right.device && left.inode == right.inode;
}

// Continuing to roll files after an unremovable temporary pin could exceed the
// fixed retention budget. Stop diagnostic writing on this failure, not CUDA.
struct RetentionCleanupError: std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline void removeOwnedDirectory(const std::filesystem::path& path, const IncidentDirectory& identity) {
    try {
        if (!sameDirectory(identity, directoryIdentity(path))) {
            throw RetentionCleanupError("refusing to remove a replaced incident directory");
        }
        std::filesystem::remove_all(path);
    } catch (...) {
        throw RetentionCleanupError("cannot safely remove an owned old/temporary incident directory");
    }
}

inline void pinClosedFiles(const std::filesystem::path&              directory,
                           const std::vector<std::filesystem::path>& files,
                           const std::string&                        manifest,
                           IncidentDirectory&                        owned_directory) {
    if (owned_directory.valid) {
        if (!sameDirectory(owned_directory, directoryIdentity(directory))) {
            throw std::runtime_error("refusing to replace an incident directory with a different device/inode");
        }
    } else if (std::filesystem::exists(directory)) {
        throw std::runtime_error("refusing to replace an unowned incident directory");
    }
    const std::filesystem::path pending  = directory.string() + ".pending";
    const std::filesystem::path previous = directory.string() + ".previous";
    if (std::filesystem::exists(pending) || std::filesystem::exists(previous)) {
        throw std::runtime_error("refusing to replace an existing incident staging directory");
    }
    if (!std::filesystem::create_directory(pending)) {
        throw std::runtime_error("cannot create incident staging directory");
    }
    const auto pending_identity = directoryIdentity(pending);
    try {
        for (const auto& source : files) {
            std::filesystem::create_hard_link(source, pending / source.filename());
        }
        std::ofstream out(pending / "manifest.json", std::ios::trunc);
        out << manifest << '\n';
        out.close();
        if (!out) {
            throw std::runtime_error("cannot write incident manifest; old complete pin is retained");
        }
        const auto old_identity = owned_directory;
        if (old_identity.valid) {
            if (!sameDirectory(old_identity, directoryIdentity(directory))) {
                throw std::runtime_error("incident directory changed before publication");
            }
            std::filesystem::rename(directory, previous);
        } else if (std::filesystem::exists(directory)) {
            throw std::runtime_error("incident destination appeared before publication");
        }
        try {
            std::filesystem::rename(pending, directory);
        } catch (...) {
            if (old_identity.valid && !std::filesystem::exists(directory)) {
                std::filesystem::rename(previous, directory);
            }
            throw;
        }
        owned_directory = pending_identity;
        if (old_identity.valid) {
            removeOwnedDirectory(previous, old_identity);
        }
    } catch (...) {
        if (std::filesystem::exists(pending)) {
            removeOwnedDirectory(pending, pending_identity);
        }
        throw;
    }
}

}  // namespace rtp_llm::cuda_trace::detail
