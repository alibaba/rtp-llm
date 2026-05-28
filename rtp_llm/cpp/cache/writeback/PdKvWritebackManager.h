#pragma once

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/writeback/PdKvWritebackManifest.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

enum class PdKvWritebackLaunchStatus {
    Started,
    Skipped,
    Failed,
};

struct PdKvWritebackCompatibility {
    int32_t              seq_size_per_block = 0;
    int32_t              layer_count        = 0;
    int32_t              group_count        = 0;
    int32_t              partition_count    = 1;
    std::vector<int32_t> layer_to_group_id;
    std::vector<int32_t> group_types;
};

struct PdKvWritebackLaunchRequest {
    PdKvWritebackManifest      manifest;
    PdKvWritebackCompatibility source;
    PdKvWritebackCompatibility destination;
    std::vector<std::string>   source_prefill_grpc_addrs;
};

struct PdKvWritebackLaunchResult {
    PdKvWritebackLaunchStatus status = PdKvWritebackLaunchStatus::Skipped;
    std::string               reason;
};

class PdKvWritebackCacheWriter {
public:
    virtual ~PdKvWritebackCacheWriter()                            = default;
    virtual absl::Status mallocWritebackBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                               size_t                         block_count) = 0;
    virtual void         commitWritebackBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                               const CacheKeysType&           cache_keys,
                                               bool                           is_resident)   = 0;
};

absl::Status validatePdKvWritebackCompatibility(const PdKvWritebackCompatibility& source,
                                                const PdKvWritebackCompatibility& destination);

class PdKvWritebackManager {
public:
    PdKvWritebackManager(const PDSepConfig& pd_config, PdKvWritebackCacheWriter* cache_writer);

    PdKvWritebackLaunchResult launchFromDecode(const PdKvWritebackLaunchRequest& request) const;
    absl::Status              receiveOnPrefill(const PdKvWritebackLaunchRequest& request,
                                               const BatchKVCacheResourcePtr&    destination_resource);

private:
    PDSepConfig               pd_config_;
    PdKvWritebackCacheWriter* cache_writer_ = nullptr;
};

using PdKvWritebackManagerPtr = std::shared_ptr<PdKvWritebackManager>;

}  // namespace rtp_llm
