#pragma once

#include "rtp_llm/cpp/cache/connector/AsyncContext.h"

namespace rtp_llm {

class Meta;
class KVCacheResource;

class KVCacheConnector {
public:
    KVCacheConnector()          = default;
    virtual ~KVCacheConnector() = default;

public:
    class AsyncMatchContext: public AsyncContext {
    public:
        ~AsyncMatchContext() override            = default;
        virtual size_t matchedBlockCount() const = 0;
    };

public:
    virtual std::shared_ptr<AsyncMatchContext> asyncMatch(const std::shared_ptr<KVCacheResource>& resource,
                                                          const std::shared_ptr<Meta>&            meta)        = 0;
    virtual std::shared_ptr<AsyncContext>      asyncRead(const std::shared_ptr<KVCacheResource>&   resource,
                                                         const std::shared_ptr<Meta>&              meta,
                                                         const std::shared_ptr<AsyncMatchContext>& match_context,
                                                         int                                       start_read_block_index,
                                                         int                                       read_block_num)                        = 0;
    virtual std::shared_ptr<AsyncContext>      asyncWrite(const std::shared_ptr<KVCacheResource>& resource,
                                                          const std::shared_ptr<Meta>&            meta)        = 0;
    virtual std::shared_ptr<AsyncContext>      asyncWriteByLayer(int                                     layer_id,
                                                                 const std::shared_ptr<KVCacheResource>& resource,
                                                                 const std::shared_ptr<Meta>&            meta) = 0;
};

}  // namespace rtp_llm