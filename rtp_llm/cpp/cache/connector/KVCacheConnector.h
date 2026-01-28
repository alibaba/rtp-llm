#pragma once

#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/IGenerateStream.h"
#include "rtp_llm/cpp/core/Event.h"

namespace rtp_llm {

class KVCacheResource;

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
        virtual ~Meta()                      = default;
        int                start_block_index = 0;
        int                block_size        = 0;
        int64_t            request_id;
        std::string        unique_key;
        int64_t            deadline_ms;
        IGenerateStreamPtr generate_stream;
        DeviceEventPtr     attention_event;

    public:
        Meta()                  = default;
        Meta(const Meta& other) = default;
    };

    class AsyncMatchContext: public AsyncContext {
    public:
        ~AsyncMatchContext() override                   = default;
        virtual size_t        matchedBlockCount() const = 0;
        virtual ConnectorType connectorType() const     = 0;
    };

public:
    virtual std::shared_ptr<AsyncMatchContext> asyncMatch(const std::shared_ptr<KVCacheResource>& resource,
                                                          const std::shared_ptr<Meta>&            meta)                      = 0;
    virtual std::shared_ptr<AsyncContext>      asyncRead(const std::shared_ptr<KVCacheResource>&   resource,
                                                         const std::shared_ptr<Meta>&              meta,
                                                         const std::shared_ptr<AsyncMatchContext>& match_context) = 0;
    virtual std::shared_ptr<AsyncContext>      asyncWrite(const std::shared_ptr<KVCacheResource>& resource,
                                                          const std::shared_ptr<Meta>&            meta)                      = 0;
    virtual std::shared_ptr<AsyncContext>      asyncWriteByLayer(int                                     layer_id,
                                                                 const std::shared_ptr<KVCacheResource>& resource,
                                                                 const std::shared_ptr<Meta>&            meta)               = 0;
};

}  // namespace rtp_llm