#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace rtp_llm {

struct KVSBuffer {
    uint64_t addr          = 0;
    size_t   size          = 0;
    size_t   object_offset = 0;
    bool     is_cuda       = false;
};

struct KVSObjectBuffer {
    std::string            object_key;
    std::vector<KVSBuffer> buffers;
    bool                   partial = false;

    size_t totalBytes() const {
        size_t total = 0;
        for (const auto& buffer : buffers) {
            total += buffer.size;
        }
        return total;
    }
};

struct KVSReadHandle {
    std::string                     handle_id;
    std::unordered_set<std::string> object_keys;

    bool valid() const {
        return !handle_id.empty();
    }

    bool contains(const std::string& object_key) const {
        return object_keys.count(object_key) != 0;
    }

    bool containsAll(const std::vector<std::string>& keys) const {
        for (const auto& key : keys) {
            if (!contains(key)) {
                return false;
            }
        }
        return true;
    }
};

struct KVSWriteHandle {
    std::string                     handle_id;
    std::unordered_set<std::string> object_keys;

    bool valid() const {
        return !handle_id.empty();
    }

    bool contains(const std::string& object_key) const {
        return object_keys.count(object_key) != 0;
    }

    bool containsAll(const std::vector<std::string>& keys) const {
        for (const auto& key : keys) {
            if (!contains(key)) {
                return false;
            }
        }
        return true;
    }
};

struct KVSObjectSpec {
    std::string object_key;
    size_t      size = 0;
};

}  // namespace rtp_llm
