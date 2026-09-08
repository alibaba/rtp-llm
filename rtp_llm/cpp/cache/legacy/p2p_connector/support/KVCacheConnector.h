#pragma once

#include <memory>

#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/legacy/p2p_connector/support/AsyncContext.h"
#include "rtp_llm/cpp/cache/legacy/p2p_connector/support/KVCacheConnectorLayerContext.h"

namespace rtp_llm::legacy::p2p {

class Meta;

class KVCacheConnector {
public:
    virtual ~KVCacheConnector() = default;

    virtual std::shared_ptr<AsyncMatchContext> asyncMatch(const ::rtp_llm::KVCacheResourcePtr& resource,
                                                          const std::shared_ptr<Meta>&         meta) = 0;
    virtual std::shared_ptr<AsyncContext>      asyncRead(const ::rtp_llm::KVCacheResourcePtr&      resource,
                                                         const std::shared_ptr<Meta>&              meta,
                                                         const std::shared_ptr<AsyncMatchContext>& match_context,
                                                         int                                       start_read_block_index,
                                                         int                                       read_block_num)                 = 0;
    virtual std::shared_ptr<AsyncContext>      asyncWrite(const ::rtp_llm::KVCacheResourcePtr& resource,
                                                          const std::shared_ptr<Meta>&         meta) = 0;
    virtual std::shared_ptr<AsyncContext>
    asyncWriteByLayer(int layer_id, const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context) = 0;
};

}  // namespace rtp_llm::legacy::p2p
