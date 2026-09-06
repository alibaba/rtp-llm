#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/config/MMKvcmConfig.h"

namespace rtp_llm {

struct MMKvcmBuffer {
    std::string key;
    void*       data    = nullptr;
    uint64_t    nbytes  = 0;
    bool        is_cuda = false;
};

class MMKvcmClient {
public:
    virtual ~MMKvcmClient() = default;

    // Empty return value means success. Implementations must validate the
    // complete batch before issuing metadata or data-plane operations.
    virtual std::string save(const std::string& trace_id, const std::vector<MMKvcmBuffer>& objects) = 0;
    // timeout_ms is the remaining end-to-end request budget. Implementations
    // must stop admitting service-sized sub-batches after it expires.
    virtual std::string
    load(const std::string& trace_id, const std::vector<MMKvcmBuffer>& objects, int64_t timeout_ms) = 0;
    virtual std::string remove(const std::string& trace_id, const std::vector<std::string>& keys)   = 0;
};

// Returns an error string for invalid configuration, otherwise an empty string.
std::string validateMMKvcmConfig(const MMKvcmConfig& config);

// Validates a complete save/load request before an implementation performs
// metadata or data-plane I/O. Returns an empty string on success.
std::string
validateMMKvcmObjects(const std::vector<MMKvcmBuffer>& objects, uint64_t max_object_bytes, uint64_t max_receipt_bytes);

// Returns the exclusive end of the next service-admissible object batch.
// The caller must validate object sizes before invoking this helper.
std::size_t nextMMKvcmBatchEnd(const std::vector<MMKvcmBuffer>& objects, std::size_t begin);

// The default build supplies a fail-closed stub. A real implementation is
// selected only with --define=use_kvcm_emb_storage=true.
bool                          hasMMKvcmImplementation();
std::shared_ptr<MMKvcmClient> createMMKvcmClient(const MMKvcmConfig& config);

}  // namespace rtp_llm
