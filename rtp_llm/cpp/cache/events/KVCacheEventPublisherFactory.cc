#include "rtp_llm/cpp/cache/events/KVCacheEventPublisherFactory.h"

#include <utility>

#include "rtp_llm/cpp/cache/events/KVCMPublisher.h"
#include "rtp_llm/cpp/cache/events/LogPublisher.h"
#include "rtp_llm/cpp/cache/events/NullPublisher.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

KVCacheEventPublisherPtr createNullKVCacheEventPublisher() {
    return std::make_shared<NullPublisher>();
}

KVCacheEventPublisherPtr createKVCacheEventPublisher(const KVCacheEventPublisherConfig&    config,
                                                     const KVCacheEventPublisherContext&   context,
                                                     KVCacheSnapshotProvider               snapshot_provider,
                                                     std::shared_ptr<KVCacheEventReporter> reporter) {
    if (config.type == "log") {
        return std::make_shared<LogPublisher>(config, context);
    }
    if (config.type == "kvcm") {
        return std::make_shared<KVCMPublisher>(config, context, std::move(snapshot_provider), std::move(reporter));
    }
    if (config.type != "none" && !config.type.empty()) {
        RTP_LLM_LOG_WARNING("unknown KV cache event publisher type=%s, using NullPublisher", config.type.c_str());
    }
    return createNullKVCacheEventPublisher();
}

}  // namespace rtp_llm
