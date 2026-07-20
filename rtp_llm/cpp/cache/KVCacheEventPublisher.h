#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

enum class KVCacheEventType {
    BLOCK_ADD,
    BLOCK_DELETE,
};

struct KVCacheEvent {
    KVCacheEventType type      = KVCacheEventType::BLOCK_ADD;
    int64_t          block_key = 0;
    uint64_t         sequence  = 0;
};

struct KVCacheSnapshot {
    int64_t              version = -1;
    std::vector<int64_t> block_keys;
};

using KVCacheSnapshotProvider = std::function<KVCacheSnapshot()>;

enum class PublishResult {
    ACCEPTED,
    DISABLED,
    NOT_RUNNING,
    QUEUE_BUSY,
    QUEUE_FULL,
};

enum class PublisherState {
    DISABLED,
    STARTING,
    LOGGING,
    REGISTERING,
    RESYNCING,
    READY,
    DEGRADED,
    STOPPED,
};

struct PublisherStatus {
    PublisherState state          = PublisherState::DISABLED;
    size_t         queue_size     = 0;
    uint64_t       accepted_count = 0;
    uint64_t       dropped_count  = 0;
};

struct KVCacheEventPublisherConfig {
    std::string type = "none";

    std::string manager_endpoint;

    size_t queue_capacity         = 100000;
    size_t report_batch_size      = 1000;
    int    flush_interval_ms      = 20;
    int    heartbeat_interval_ms  = 1000;
    int    request_timeout_ms     = 1500;
    int    retry_interval_ms      = 500;
    int    snapshot_interval_ms   = 300000;
    size_t log_max_keys_per_batch = 8;
};

struct KVCacheEventPublisherContext {
    std::string instance_group;
    std::string instance_id;
    std::string host_ip_port;
    std::string model_name;
    std::string dtype;
    std::string spec_name;
    std::string location_uri;

    int32_t block_size_tokens = 0;
    int64_t spec_size_bytes   = 0;
    int32_t tp_size           = 1;
    int32_t dp_size           = 1;
    int32_t pp_size           = 1;
    int32_t dp_rank           = 0;
    bool    use_mla           = false;
};

class KVCacheEventReporter {
public:
    virtual ~KVCacheEventReporter() = default;

    virtual bool post(const std::string& route, const std::string& request, std::string& response) noexcept = 0;
};

class KVCacheEventPublisher {
public:
    virtual ~KVCacheEventPublisher() = default;

    virtual bool            start() noexcept                        = 0;
    virtual PublishResult   tryPublish(KVCacheEvent event) noexcept = 0;
    virtual void            stop() noexcept                         = 0;
    virtual PublisherStatus status() const noexcept                 = 0;
    virtual bool            enabled() const noexcept                = 0;
};

using KVCacheEventPublisherPtr = std::shared_ptr<KVCacheEventPublisher>;

class NullPublisher final: public KVCacheEventPublisher {
public:
    bool            start() noexcept override;
    PublishResult   tryPublish(KVCacheEvent event) noexcept override;
    void            stop() noexcept override;
    PublisherStatus status() const noexcept override;
    bool            enabled() const noexcept override;
};

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

KVCacheEventPublisherPtr createKVCacheEventPublisher(const KVCacheEventPublisherConfig&    config,
                                                     const KVCacheEventPublisherContext&   context,
                                                     KVCacheSnapshotProvider               snapshot_provider = {},
                                                     std::shared_ptr<KVCacheEventReporter> reporter          = nullptr);

}  // namespace rtp_llm
