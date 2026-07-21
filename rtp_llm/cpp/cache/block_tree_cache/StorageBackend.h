#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"

namespace rtp_llm {

// Pluggable remote storage interface (replaces old RemoteConnector).
// All remote I/O is asynchronous and batch-oriented.
// The tree orchestrates prefetch/backup scheduling but does not track
// remote state in GroupSlot — the backend manages its own key mapping.
class StorageBackend {
public:
    virtual ~StorageBackend() = default;

    // Asynchronous batch read: pull data from remote into host buffers.
    // results[i] corresponds to keys[i].
    virtual std::shared_ptr<AsyncContext> batchRead(const std::vector<std::string>& keys,
                                                    std::vector<std::vector<char>>& results) = 0;

    // Asynchronous batch write: push data to remote.
    virtual std::shared_ptr<AsyncContext>
    batchWrite(const std::vector<std::pair<std::string, std::vector<char>>>& items) = 0;

    // Batch existence check: query whether keys exist on remote.
    virtual std::shared_ptr<AsyncContext> batchExists(const std::vector<std::string>& keys,
                                                      std::vector<bool>&              results) = 0;

    // Batch delete: remove data from remote.
    virtual std::shared_ptr<AsyncContext> batchDelete(const std::vector<std::string>& keys) = 0;
};

}  // namespace rtp_llm
