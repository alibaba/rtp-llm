#include "rtp_llm/cpp/cache/block_tree_cache/CacheStageMetrics.h"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string_view>
#include <thread>
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
namespace rtp_llm {
struct CacheStageMetrics::Impl {
    static constexpr size_t kCapacity = 512, kLabelSize = 32, kMaxProbes = 32;
    struct Slot {
        std::atomic<uint8_t>                        state{0};  // empty / initializing / published
        std::atomic_flag                            busy = ATOMIC_FLAG_INIT;
        std::array<std::array<char, kLabelSize>, 5> labels{};
        int64_t                                     rank{-1};
        bool                                        success{false};
        uint64_t                                    count{0}, sum_us{0};
        std::unique_ptr<kmonitor::MetricsTags>      tags;  // consumer-only
    };
    explicit Impl(std::shared_ptr<kmonitor::MetricsReporter> reporter, std::chrono::milliseconds interval):
        reporter(std::move(reporter)) {
        worker = std::thread([this, interval] {
            std::unique_lock<std::mutex> lock(wake_mutex);
            while (!stopped) {
                wake.wait_for(lock, interval, [this] { return stopped; });
                lock.unlock();
                flush();
                lock.lock();
            }
        });
    }
    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(wake_mutex);
            stopped = true;
        }
        wake.notify_one();
        worker.join();
    }
    void record(const std::array<std::string_view, 5>& labels, int64_t latency, int64_t rank, bool success) noexcept {
        uint64_t hash = 1469598103934665603ULL;
        for (auto label : labels) {
            if (label.size() >= kLabelSize) {
                dropped.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            for (auto c : label)
                hash = (hash ^ static_cast<unsigned char>(c)) * 1099511628211ULL;
            hash = (hash ^ 255) * 1099511628211ULL;
        }
        hash ^= static_cast<uint64_t>(rank) * 1099511628211ULL;
        hash ^= success;
        for (size_t probe = 0; probe < kMaxProbes; ++probe) {
            auto&      slot  = slots[(hash + probe) % kCapacity];
            const auto state = slot.state.load(std::memory_order_acquire);
            if (state == 0) {
                uint8_t expected = 0;
                if (!slot.state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
                    dropped.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                for (size_t i = 0; i < labels.size(); ++i)
                    std::memcpy(slot.labels[i].data(), labels[i].data(), labels[i].size());
                slot.rank    = rank;
                slot.success = success;
                slot.state.store(2, std::memory_order_release);
            } else if (state == 1) {
                dropped.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            bool match = slot.rank == rank && slot.success == success;
            for (size_t i = 0; match && i < labels.size(); ++i)
                match = labels[i] == slot.labels[i].data();
            if (!match)
                continue;
            if (slot.busy.test_and_set(std::memory_order_acquire)) {
                dropped.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            ++slot.count;
            slot.sum_us += static_cast<uint64_t>(latency);
            slot.busy.clear(std::memory_order_release);
            return;
        }
        dropped.fetch_add(1, std::memory_order_relaxed);
    }
    void flush() noexcept {
        std::lock_guard<std::mutex> lock(flush_mutex);
        for (auto& slot : slots) {
            if (slot.state.load(std::memory_order_acquire) != 2 || slot.busy.test_and_set(std::memory_order_acquire))
                continue;
            const auto count = slot.count, sum = slot.sum_us;
            slot.count = slot.sum_us = 0;
            slot.busy.clear(std::memory_order_release);
            if (count == 0)
                continue;
            try {
                if (!slot.tags) {
                    auto        tags    = std::make_unique<kmonitor::MetricsTags>();
                    const char* names[] = {"stage", "scope", "operation", "source_tier", "target_tier"};
                    for (size_t i = 0; i < 5; ++i)
                        tags->AddTag(names[i], slot.labels[i].data());
                    tags->AddTag("success", slot.success ? "true" : "false");
                    if (slot.rank >= 0)
                        tags->AddTag("transfer_rank", std::to_string(slot.rank));
                    slot.tags = std::move(tags);
                }
                RtpLLMCacheTransferStageMetricsCollector c;
                c.tags_prepared  = true;
                c.latency_us     = static_cast<double>(sum) / count;
                c.latency_sum_us = sum;
                c.sample_count   = count;
                reporter->report<RtpLLMCacheTransferMetrics, RtpLLMCacheTransferStageMetricsCollector>(slot.tags.get(),
                                                                                                       &c);
            } catch (...) {
                dropped.fetch_add(count, std::memory_order_relaxed);
            }
        }
        const auto loss = dropped.exchange(0, std::memory_order_relaxed);
        if (loss) {
            try {
                RtpLLMCacheTransferStageMetricsCollector c;
                c.sample_count  = 0;
                c.dropped_count = loss;
                reporter->report<RtpLLMCacheTransferMetrics, RtpLLMCacheTransferStageMetricsCollector>(nullptr, &c);
            } catch (...) {
                dropped.fetch_add(loss, std::memory_order_relaxed);
            }
        }
    }
    std::shared_ptr<kmonitor::MetricsReporter> reporter;
    std::array<Slot, kCapacity>                slots;
    std::atomic<uint64_t>                      dropped{0};
    std::mutex                                 flush_mutex, wake_mutex;
    std::condition_variable                    wake;
    bool                                       stopped{false};
    std::thread                                worker;
};
CacheStageMetrics::CacheStageMetrics(std::shared_ptr<kmonitor::MetricsReporter> reporter,
                                     std::chrono::milliseconds                  interval):
    impl_(std::make_unique<Impl>(std::move(reporter), interval)) {}
CacheStageMetrics::~CacheStageMetrics() = default;
void CacheStageMetrics::record(const char* stage,
                               int64_t     latency,
                               bool        success,
                               int64_t     rank,
                               const char* scope,
                               const char* operation,
                               const char* source,
                               const char* target) noexcept {
    impl_->record({stage, scope, operation, source, target}, latency, rank, success);
}
void CacheStageMetrics::flush() {
    impl_->flush();
}
}  // namespace rtp_llm
