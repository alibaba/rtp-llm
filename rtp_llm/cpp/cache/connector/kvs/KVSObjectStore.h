#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/connector/kvs/KVSObjectBackend.h"

namespace rtp_llm {

using KVSCacheKey = int64_t;

struct KVSObjectStoreConfig {
    std::string object_namespace  = "rtp_llm";
    std::string cache_key_version = "1";
};

struct KVSBlockIdentity {
    KVSCacheKey cache_key = 0;
    int         group_id  = 0;
};

class KVSObjectStore {
public:
    KVSObjectStore(KVSObjectStoreConfig config, std::shared_ptr<KVSObjectBackend> backend);

    std::string makeKey(const KVSBlockIdentity& identity) const;

    std::optional<KVSReadHandle> acquire(const std::vector<KVSObjectBuffer>& objects,
                                         const std::string&                  trace_id);
    bool fetch(const KVSReadHandle&                handle,
               const std::vector<KVSObjectBuffer>& objects,
               const std::string&                  trace_id);
    bool loadLocal(const std::vector<KVSObjectBuffer>& objects, const std::string& trace_id);
    std::optional<KVSWriteHandle> beginWrite(const std::vector<KVSObjectSpec>& objects,
                                             const std::string&                trace_id);
    bool writeLocal(const std::vector<KVSObjectBuffer>& objects, const std::string& trace_id);
    bool commitWrite(const KVSWriteHandle&             handle,
                     const std::vector<KVSObjectSpec>& objects,
                     const std::string&                trace_id);
    void abortWrite(const KVSWriteHandle& handle, const std::string& trace_id);
    bool write(const std::vector<KVSObjectBuffer>& objects, const std::string& trace_id);
    void release(const KVSReadHandle& handle, const std::string& trace_id);

private:
    std::vector<std::string> objectKeys(const std::vector<KVSObjectBuffer>& buffers) const;
    std::vector<std::string> objectKeys(const std::vector<KVSObjectSpec>& objects) const;
    std::vector<size_t>      objectSizes(const std::vector<KVSObjectBuffer>& buffers) const;
    std::vector<size_t>      objectSizes(const std::vector<KVSObjectSpec>& objects) const;

private:
    KVSObjectStoreConfig              config_;
    std::shared_ptr<KVSObjectBackend> backend_;
};

}  // namespace rtp_llm
