#pragma once

#include <memory>

#include "rtp_llm/cpp/cache/events/KVCacheEventPublisher.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventPublisherConfig.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventReporter.h"

namespace rtp_llm {

class KVCMPublisher final: public KVCacheEventPublisher {
public:
    KVCMPublisher(KVCacheEventPublisherConfig           config,
                  KVCacheEventPublisherContext          context,
                  KVCacheSnapshotProvider               snapshot_provider,
                  std::shared_ptr<KVCacheEventReporter> reporter = nullptr);
    ~KVCMPublisher() override;

    bool            start() noexcept override;
    PublishResult   tryPublish(KVCacheEvent event) noexcept override;
    void            stop() noexcept override;
    PublisherStatus status() const noexcept override;
    bool            enabled() const noexcept override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtp_llm
