#pragma once

#include "rtp_llm/cpp/cache_new/AsyncContext.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"

namespace rtp_llm {

class KVCacheConnector {
public:
    KVCacheConnector()          = default;
    virtual ~KVCacheConnector() = default;

public:
    enum class ConnectorType {
        Memory = 0,
        Remote = 1,
        P2P    = 2
    };

    class Meta {
    public:
        virtual ~Meta()                                = default;
        virtual std::pair<int, int> blockRange() const = 0;  // <start_block_index, size>
    };

    class AsyncMatchContext: public AsyncContext {
    public:
        ~AsyncMatchContext() override                   = default;
        virtual size_t        matchedBlockCount() const = 0;
        virtual ConnectorType connectorType() const     = 0;
    };

public:
    virtual std::shared_ptr<AsyncMatchContext> asyncMatch(const std::shared_ptr<KVCacheResourceV1>& resource,
                                                          const std::shared_ptr<Meta>&              meta)                      = 0;
    virtual std::shared_ptr<AsyncContext>      asyncRead(const std::shared_ptr<KVCacheResourceV1>& resource,
                                                         const std::shared_ptr<Meta>&              meta,
                                                         const std::shared_ptr<AsyncMatchContext>& match_context) = 0;
    virtual std::shared_ptr<AsyncContext>      asyncWrite(const std::shared_ptr<KVCacheResourceV1>& resource,
                                                          const std::shared_ptr<Meta>&              meta)                      = 0;
    virtual std::shared_ptr<AsyncContext>      asyncWriteByLayer(int                                       layer_id,
                                                                 const std::shared_ptr<KVCacheResourceV1>& resource,
                                                                 const std::shared_ptr<Meta>&              meta)               = 0;
};

}  // namespace rtp_llm