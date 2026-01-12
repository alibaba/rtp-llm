#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/core/Event.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include <torch/torch.h>

namespace rtp_llm {

class ICompleteTokenIds {
public:
    virtual ~ICompleteTokenIds() = default;

public:
    virtual void             appendTokenId(int batch_id, int token_id) = 0;
    virtual std::vector<int> currentExecuteTokens(int batch_id)        = 0;
};

typedef std::shared_ptr<ICompleteTokenIds> ICompleteTokenIdsPtr;

class IGenerateStream {
public:
    virtual ~IGenerateStream() = default;

public:
    virtual void             appendTokenId(int batch_id, int token_id) = 0;
    virtual std::vector<int> currentExecuteTokens(int batch_id)        = 0;

    virtual void appendSPInfo(const std::vector<int>& propose_tokens,
                              const TensorPB&         propose_probs,
                              const TensorPB&         propose_hidden)                             = 0;
    virtual std::optional<std::tuple<std::vector<int>, TensorPB, TensorPB>> getSPInfoPB() = 0;

    virtual int                       reuseBlockNum()  = 0;
    virtual std::tuple<int, int, int> getReuseLength() = 0;  // reuse_length, local_reuse_length, remote_reuse_length
    virtual void setPrefillReuseLength(int reuse_length, int local_reuse_length, int remote_reuse_length) = 0;

    virtual std::pair<std::string, uint32_t> getPrefillAddr() = 0;  // prefill_ip, prefill_port

    virtual std::vector<int32_t> getContextPositionIdsPB()                              = 0;
    virtual void                 setContextPositionIds(const std::vector<int32_t>& ids) = 0;
};
using IGenerateStreamPtr = std::shared_ptr<IGenerateStream>;

struct KVCacheConnectorMeta {
    int64_t            request_id;
    std::string        unique_key;
    int64_t            deadline_ms;
    IGenerateStreamPtr generate_stream;
    DeviceEventPtr     attention_event;
};

class KVCacheConnector {
public:
    KVCacheConnector()          = default;
    virtual ~KVCacheConnector() = default;

public:
    class Meta {
    public:
        virtual ~Meta() = default;
    };

    virtual bool init() = 0;

public:
    virtual std::shared_ptr<AsyncMatchContext> asyncMatch(const std::shared_ptr<KVCacheResource>&      resource,
                                                          const std::shared_ptr<KVCacheConnectorMeta>& meta)        = 0;
    virtual std::shared_ptr<AsyncContext>      asyncRead(const std::shared_ptr<KVCacheResource>&      resource,
                                                         const std::shared_ptr<KVCacheConnectorMeta>& meta,
                                                         const std::shared_ptr<AsyncMatchContext>&    match_context,
                                                         const std::pair<int, int>&                   block_range)                    = 0;
    virtual std::shared_ptr<AsyncContext>      asyncWrite(const std::shared_ptr<KVCacheResource>&      resource,
                                                          const std::shared_ptr<KVCacheConnectorMeta>& meta)        = 0;
    virtual std::shared_ptr<AsyncContext>      asyncWriteByLayer(int                                          layer_id,
                                                                 const std::shared_ptr<KVCacheResource>&      resource,
                                                                 const std::shared_ptr<KVCacheConnectorMeta>& meta) = 0;
};

}  // namespace rtp_llm