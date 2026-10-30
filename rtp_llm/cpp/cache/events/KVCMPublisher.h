#pragma once

#include <memory>
#include <functional>

#include "rtp_llm/cpp/cache/events/KVCacheEventPublisher.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventPublisherConfig.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventReporter.h"

namespace rtp_llm {

class KVCMPublisher final: public KVCacheEventPublisher {
public:
    // Runs under the registration lock; must be nonblocking and must not
    // reenter the publisher. An absent/invalid value never invokes it.
    using RemoteCacheControl = std::function<void(bool)>;
    KVCMPublisher(KVCacheEventPublisherConfig           config,
                  KVCacheEventPublisherContext          context,
                  KVCacheSnapshotProvider               snapshot_provider,
                  std::shared_ptr<KVCacheEventReporter> reporter = nullptr,
                  RemoteCacheControl                   remote_cache_control = {});
    ~KVCMPublisher() override;

    bool            start() noexcept override;
    PublishResult   tryPublish(KVCacheEvent event) noexcept override;
    void            stop() noexcept override;
    PublisherStatus status() const noexcept override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtp_llm
