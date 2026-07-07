#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/connector/kvs/KVSObjectTypes.h"

namespace rtp_llm {

class KVSObjectBackend {
public:
    virtual ~KVSObjectBackend() = default;

    virtual std::optional<KVSReadHandle> get(const std::vector<std::string>& object_keys,
                                             const std::string&              trace_id) = 0;
    virtual std::optional<KVSReadHandle> getLocal(const std::vector<std::string>& object_keys,
                                                  const std::string&              trace_id) = 0;
    virtual std::optional<KVSWriteHandle> getMutableLocal(const std::vector<std::string>& object_keys,
                                                          const std::string&              trace_id) = 0;
    virtual std::optional<KVSWriteHandle> create(const std::vector<std::string>& object_keys,
                                                 const std::vector<size_t>&      object_sizes,
                                                 const std::string&              trace_id) = 0;
    virtual bool fetch(const KVSReadHandle&            handle,
                       const std::vector<std::string>& object_keys,
                       const std::string&              trace_id) = 0;
    virtual bool load(const KVSReadHandle& handle, const std::vector<KVSObjectBuffer>& dst_buffers) = 0;
    virtual bool store(const KVSWriteHandle& handle, const std::vector<KVSObjectBuffer>& src_buffers) = 0;
    virtual bool complete(const KVSReadHandle&            handle,
                          const std::vector<std::string>& object_keys,
                          const std::string&              trace_id) = 0;
    virtual bool complete(const KVSWriteHandle&           handle,
                          const std::vector<std::string>& object_keys,
                          const std::string&              trace_id) = 0;
    virtual void release(const KVSReadHandle& handle, const std::string& trace_id) = 0;
    virtual void release(const KVSWriteHandle& handle, const std::string& trace_id) = 0;
    virtual void discard(const KVSWriteHandle& handle, const std::string& trace_id) = 0;
};

}  // namespace rtp_llm
