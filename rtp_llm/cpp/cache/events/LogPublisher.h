#pragma once

#include <memory>

#include "rtp_llm/cpp/cache/events/KVCacheEventPublisher.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventPublisherConfig.h"

namespace rtp_llm {

class LogPublisher final: public KVCacheEventPublisher {
public:
    LogPublisher(KVCacheEventPublisherConfig config, KVCacheEventPublisherContext context);
    ~LogPublisher() override;

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
