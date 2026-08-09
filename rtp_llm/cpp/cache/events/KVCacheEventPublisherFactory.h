#pragma once

#include <memory>

#include "rtp_llm/cpp/cache/events/KVCacheEventPublisher.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventPublisherConfig.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventReporter.h"

namespace rtp_llm {

KVCacheEventPublisherPtr createNullKVCacheEventPublisher();

KVCacheEventPublisherPtr createKVCacheEventPublisher(const KVCacheEventPublisherConfig&    config,
                                                     const KVCacheEventPublisherContext&   context,
                                                     KVCacheSnapshotProvider               snapshot_provider = {},
                                                     std::shared_ptr<KVCacheEventReporter> reporter          = nullptr);

}  // namespace rtp_llm
