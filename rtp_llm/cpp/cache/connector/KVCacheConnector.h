#pragma once

#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorLayerContext.h"

namespace rtp_llm {

class Meta;
class KVCacheResource;

class KVCacheConnector {
public:
    KVCacheConnector()          = default;
    virtual ~KVCacheConnector() = default;

public:
    virtual std::shared_ptr<AsyncMatchContext> asyncMatch(const std::shared_ptr<KVCacheResource>& resource,
                                                          const std::shared_ptr<Meta>&            meta) = 0;
    // Nonnegative `start_read_block_index` and `read_block_num` are global
    // cache-key blocks. A CP-canonical connector converts them to internal
    // entry ordinals. Connector-specific negative sentinels (for example the
    // P2P scheduler's count=-1 "remaining entries") must be handled before
    // conversion and remain sentinels, never unsigned block counts.
    virtual std::shared_ptr<AsyncContext> asyncRead(const std::shared_ptr<KVCacheResource>&   resource,
                                                    const std::shared_ptr<Meta>&              meta,
                                                    const std::shared_ptr<AsyncMatchContext>& match_context,
                                                    int                                       start_read_block_index,
                                                    int                                       read_block_num)                 = 0;
    virtual std::shared_ptr<AsyncContext> asyncWrite(const std::shared_ptr<KVCacheResource>& resource,
                                                     const std::shared_ptr<Meta>&            meta) = 0;
    virtual std::shared_ptr<AsyncContext>
    asyncWriteByLayer(int layer_id, const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context) = 0;
};

}  // namespace rtp_llm
