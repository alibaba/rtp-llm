#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaReader.h"

namespace rtp_llm {
namespace {

using Read                      = rdma_transport::RdmaRead;
using ReadResult                = rdma_transport::RdmaReadResult;
using Descriptor                = rdma_transport::RdmaDescriptor;
using Clock                     = std::chrono::steady_clock;
constexpr int64_t kReadWindowMs = 5000;

MultimodalOutputPB receipt(const std::vector<std::string>& leases) {
    MultimodalOutputPB result;
    for (const auto& lease : leases) {
        auto* slot       = result.add_output_rdma_slots();
        auto* descriptor = slot->mutable_rdma_descriptor();
        descriptor->set_host("127.0.0.1");
        descriptor->set_port(1);
        descriptor->set_remote_addr(0x1000);
        descriptor->set_lease_id(lease);
        descriptor->set_payload_bytes(16);
        auto* key = descriptor->add_nic_keys();
        key->set_nic_id(0);
        key->set_rkey(1);
        auto* tensor = descriptor->add_tensors();
        tensor->add_shape(1);
        tensor->add_shape(4);
        tensor->set_data_type(RDMA_TENSOR_FLOAT32);
        tensor->set_nbytes(16);
        slot->add_roles(MMRdmaSlotPB::EMBEDDING);
    }
    result.add_split_size(static_cast<int64_t>(leases.size()));
    return result;
}

class RecordingControl final: public MMControlClient {
public:
    ErrorResult<MultimodalOutputPB> request(const std::string&, MultimodalInputsPB&, DeadlineBudget&) override {
        return ErrorInfo(ErrorCode::UNKNOWN_ERROR, "consume tests do not issue control requests");
    }

    void release(const std::string&, const std::vector<std::string>& leases, DeadlineBudget&) override {
        record(leases, false);
    }
    void releaseAsync(std::string, std::vector<std::string> leases) override {
        record(leases, true);
    }
    int releases(const std::string& lease) {
        std::lock_guard<std::mutex> lock(mutex_);
        return sync_[lease] + async_[lease];
    }
    int asyncReleases(const std::string& lease) {
        std::lock_guard<std::mutex> lock(mutex_);
        return async_[lease];
    }

private:
    void record(const std::vector<std::string>& leases, bool async) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& lease : leases) {
            ++(async ? async_ : sync_)[lease];
        }
    }
    std::mutex                 mutex_;
    std::map<std::string, int> sync_, async_;
};

// The provider blocks until the test opens the gate or its actual supplied read
// budget expires. Every wait is bounded; no sleep estimates stand in for overlap.
class ReadGate {
public:
    bool enterAndWait(Clock::time_point deadline) {
        std::unique_lock<std::mutex> lock(mutex_);
        ++entered_;
        cv_.notify_all();
        return Clock::now() < deadline && cv_.wait_until(lock, deadline, [&] { return open_; });
    }
    bool waitForEntries(size_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(2), [&] { return entered_ >= count; });
    }
    size_t entries() {
        std::lock_guard<std::mutex> lock(mutex_);
        return entered_;
    }
    void open() {
        std::lock_guard<std::mutex> lock(mutex_);
        open_ = true;
        cv_.notify_all();
    }
    void openAt(Clock::time_point when) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_until(lock, when, [&] { return open_; });
        open_ = true;
        cv_.notify_all();
    }

private:
    std::mutex              mutex_;
    std::condition_variable cv_;
    size_t                  entered_ = 0;
    bool                    open_    = false;
};

class GatedRead final: public Read {
public:
    struct Invocation {
        Clock::time_point started;
        Clock::time_point deadline;
    };

    enum class Outcome {
        SUCCESS,
        ERROR,
        EXCEPTION,
        UNKNOWN_EXCEPTION,
        DEFERRED_ERROR
    };

    explicit GatedRead(std::shared_ptr<ReadGate> gate): gate_(std::move(gate)) {}

    ReadResult read(const std::vector<Descriptor>& descriptors, int64_t timeout_ms) override {
        bool deferred = false;
        return readWithCleanup(descriptors, timeout_ms, {}, &deferred);
    }

    ReadResult readWithCleanup(const std::vector<Descriptor>& descriptors,
                               int64_t                        timeout_ms,
                               std::function<void()>          release,
                               bool*                          release_deferred) override {
        *release_deferred   = false;
        const auto started  = Clock::now();
        const auto deadline = started + std::chrono::milliseconds(timeout_ms);
        const int  active   = active_.fetch_add(1) + 1;
        struct Exit {
            std::atomic<int>& active;
            ~Exit() {
                --active;
            }
        } exit{active_};
        int previous = max_active_.load();
        while (active > previous && !max_active_.compare_exchange_weak(previous, active)) {}
        {
            std::lock_guard<std::mutex> lock(mutex_);
            timeouts_.push_back(timeout_ms);
            for (const auto& descriptor : descriptors) {
                invocations_[descriptor.lease_id] = {started, deadline};
            }
        }
        if (timeout_ms <= 0 || !gate_->enterAndWait(deadline)) {
            return {ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "fake provider read deadline expired"), {}};
        }
        switch (next_.exchange(Outcome::SUCCESS)) {
            case Outcome::EXCEPTION:
                throw std::runtime_error("injected read exception");
            case Outcome::UNKNOWN_EXCEPTION:
                throw 17;
            case Outcome::ERROR:
                return {ErrorInfo(ErrorCode::REMOTE_ALLOCATE_RESOURCE_READ_FAILED, "injected read failure"), {}};
            case Outcome::DEFERRED_ERROR: {
                std::lock_guard<std::mutex> lock(mutex_);
                pending_storage_  = std::make_shared<int>(17);
                pending_release_  = std::move(release);
                *release_deferred = true;
                return {ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "one read chunk has not completed"), {}};
            }
            case Outcome::SUCCESS:
                break;
        }
        ReadResult result{ErrorInfo::OkStatus(), {}};
        for (const auto& descriptor : descriptors) {
            for (const auto& meta : descriptor.tensors) {
                result.tensors.push_back(torch::full(meta.shape, 17.0f, torch::kFloat32));
            }
        }
        return result;
    }

    void failNext(Outcome outcome) {
        next_.store(outcome);
    }
    int maxActive() const {
        return max_active_.load();
    }
    std::vector<int64_t> timeouts() {
        std::lock_guard<std::mutex> lock(mutex_);
        return timeouts_;
    }
    std::map<std::string, Invocation> invocations() {
        std::lock_guard<std::mutex> lock(mutex_);
        return invocations_;
    }
    std::weak_ptr<int> pendingStorage() {
        std::lock_guard<std::mutex> lock(mutex_);
        return pending_storage_;
    }
    void completeDeferred() {
        std::function<void()> release;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_storage_.reset();
            release = std::move(pending_release_);
        }
        if (release) {
            release();
        }
    }

private:
    std::shared_ptr<ReadGate>         gate_;
    std::atomic<int>                  active_{0}, max_active_{0};
    std::atomic<Outcome>              next_{Outcome::SUCCESS};
    std::mutex                        mutex_;
    std::vector<int64_t>              timeouts_;
    std::map<std::string, Invocation> invocations_;
    std::shared_ptr<int>              pending_storage_;
    std::function<void()>             pending_release_;
};

ConsumeResult consume(MMRdmaReader&                            reader,
                      const std::shared_ptr<RecordingControl>& control,
                      const MultimodalOutputPB&                output,
                      int64_t                                  budget_ms  = 10000,
                      std::function<void(Clock::time_point)>   on_started = {}) {
    const std::string endpoint = "provider-pool-test";
    DeadlineBudget    budget(budget_ms);
    DeliveryContext   context{endpoint, budget, *control, control};
    if (on_started) {
        on_started(Clock::now());
    }
    return reader.consume(output, context);
}

// Opens all fake reads before joining, including when an ASSERT returns early.
// Provider deadlines bound join even if the implementation under test is broken.
class Consumers {
public:
    Consumers(MMRdmaReader&                     reader,
              std::shared_ptr<RecordingControl> control,
              std::shared_ptr<ReadGate>         gate,
              size_t                            count):
        results(count),
        reader_(reader),
        control_(std::move(control)),
        gate_(std::move(gate)),
        done_(count, false),
        started_at_(count) {}
    ~Consumers() {
        gate_->open();
        join();
    }
    void start(size_t index, std::string lease, int64_t budget_ms = 10000) {
        threads_.emplace_back([this, index, lease = std::move(lease), budget_ms] {
            try {
                results[index] = consume(reader_, control_, receipt({lease}), budget_ms, [&](Clock::time_point start) {
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        started_at_[index] = start;
                    }
                    cv_.notify_all();
                });
            } catch (...) {
                results[index] = ConsumeResult::failure(ErrorInfo(ErrorCode::UNKNOWN_ERROR, "unexpected throw"));
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                done_[index] = true;
                ++finished_;
            }
            cv_.notify_all();
        });
    }
    bool waitForFinished(size_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(2), [&] { return finished_ >= count; });
    }
    bool waitForCompletion(size_t index) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(2), [&] { return done_[index]; });
    }
    bool waitForStart(size_t index) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(2), [&] { return started_at_[index] != Clock::time_point{}; });
    }
    Clock::time_point startedAt(size_t index) {
        std::lock_guard<std::mutex> lock(mutex_);
        return started_at_[index];
    }
    void join() {
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    std::vector<ConsumeResult> results;

private:
    MMRdmaReader&                     reader_;
    std::shared_ptr<RecordingControl> control_;
    std::shared_ptr<ReadGate>         gate_;
    std::mutex                        mutex_;
    std::condition_variable           cv_;
    size_t                            finished_ = 0;
    std::vector<bool>                 done_;
    std::vector<Clock::time_point>    started_at_;
    std::vector<std::thread>          threads_;
};

TEST(MMRdmaProviderPoolTest, FourDistinctProvidersOverlapAndFifthDeadlineDoesNotReenterOne) {
    auto                                    gate    = std::make_shared<ReadGate>();
    auto                                    control = std::make_shared<RecordingControl>();
    std::vector<std::shared_ptr<GatedRead>> providers;
    std::vector<std::shared_ptr<Read>>      pool;
    for (size_t i = 0; i < 4; ++i) {
        providers.push_back(std::make_shared<GatedRead>(gate));
        pool.push_back(providers.back());
    }
    RdmaConfig config;
    config.read_timeout_ms = kReadWindowMs;
    MMRdmaReader reader(std::move(pool), config);
    Consumers    consumers(reader, control, gate, 5);
    for (size_t i = 0; i < 4; ++i) {
        consumers.start(i, "parallel-" + std::to_string(i));
    }
    ASSERT_TRUE(gate->waitForEntries(4));
    consumers.start(4, "waiter", 40);
    ASSERT_TRUE(consumers.waitForCompletion(4));
    EXPECT_FALSE(consumers.results[4].succeeded());
    EXPECT_EQ(consumers.results[4].error().code(), ErrorCode::MM_PROCESS_ERROR);
    EXPECT_EQ(gate->entries(), 4u);
    EXPECT_EQ(control->releases("waiter"), 1);
    EXPECT_EQ(control->asyncReleases("waiter"), 1);
    gate->open();
    consumers.join();
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_TRUE(consumers.results[i].succeeded());
        ASSERT_EQ(consumers.results[i].output().mm_features.size(), 1u);
        EXPECT_TRUE(consumers.results[i].output().mm_features[0].eq(17.0f).all().item<bool>());
        EXPECT_EQ(control->releases("parallel-" + std::to_string(i)), 1);
        EXPECT_EQ(providers[i]->maxActive(), 1);
        ASSERT_EQ(providers[i]->timeouts().size(), 1u);
        EXPECT_GT(providers[i]->timeouts()[0], 0);
        EXPECT_LE(providers[i]->timeouts()[0], kReadWindowMs);
    }
}

TEST(MMRdmaProviderPoolTest, SingleProviderAndLegacyConfiguredConstructorKeepCallSerialization) {
    for (const bool legacy_constructor : {false, true}) {
        SCOPED_TRACE(legacy_constructor);
        auto       gate     = std::make_shared<ReadGate>();
        auto       control  = std::make_shared<RecordingControl>();
        auto       provider = std::make_shared<GatedRead>(gate);
        RdmaConfig config;
        config.read_provider_count = 4;
        config.read_timeout_ms     = kReadWindowMs;
        std::unique_ptr<MMRdmaReader> reader;
        if (legacy_constructor) {
            reader = std::make_unique<MMRdmaReader>(provider, config);
        } else {
            reader = std::make_unique<MMRdmaReader>(std::vector<std::shared_ptr<Read>>{provider}, config);
        }
        Consumers consumers(*reader, control, gate, 4);
        consumers.start(0, "holder");
        ASSERT_TRUE(gate->waitForEntries(1));
        for (size_t i = 1; i < 4; ++i) {
            consumers.start(i, "queued-" + std::to_string(i), 40);
        }
        ASSERT_TRUE(consumers.waitForFinished(3));
        EXPECT_EQ(gate->entries(), 1u);
        for (size_t i = 1; i < 4; ++i) {
            EXPECT_FALSE(consumers.results[i].succeeded());
            EXPECT_EQ(consumers.results[i].error().code(), ErrorCode::MM_PROCESS_ERROR);
            EXPECT_EQ(control->releases("queued-" + std::to_string(i)), 1);
        }
        gate->open();
        consumers.join();
        EXPECT_TRUE(consumers.results[0].succeeded());
        EXPECT_EQ(provider->maxActive(), 1);
    }
}

TEST(MMRdmaProviderPoolTest, ErrorsAndBothExceptionKindsReturnProviderSlot) {
    for (const auto failure :
         {GatedRead::Outcome::ERROR, GatedRead::Outcome::EXCEPTION, GatedRead::Outcome::UNKNOWN_EXCEPTION}) {
        SCOPED_TRACE(static_cast<int>(failure));
        auto gate = std::make_shared<ReadGate>();
        gate->open();
        auto         control  = std::make_shared<RecordingControl>();
        auto         provider = std::make_shared<GatedRead>(gate);
        RdmaConfig   config;
        MMRdmaReader reader(std::vector<std::shared_ptr<Read>>{provider}, config);
        provider->failNext(failure);
        EXPECT_FALSE(consume(reader, control, receipt({"failure"})).succeeded());
        EXPECT_EQ(control->releases("failure"), 1);
        EXPECT_TRUE(consume(reader, control, receipt({"next"})).succeeded());
        EXPECT_EQ(control->releases("next"), 1);
        EXPECT_EQ(provider->timeouts().size(), 2u);
        EXPECT_EQ(provider->maxActive(), 1);
    }
}

TEST(MMRdmaProviderPoolTest, ConfiguredReadWindowExpiresBeforeLongDeliveryBudget) {
    auto       gate     = std::make_shared<ReadGate>();
    auto       control  = std::make_shared<RecordingControl>();
    auto       provider = std::make_shared<GatedRead>(gate);
    RdmaConfig config;
    config.read_timeout_ms = 60;
    MMRdmaReader reader(std::vector<std::shared_ptr<Read>>{provider}, config);
    Consumers    consumers(reader, control, gate, 2);
    consumers.start(0, "window-holder");
    ASSERT_TRUE(gate->waitForEntries(1));
    consumers.start(1, "window-waiter");
    ASSERT_TRUE(consumers.waitForFinished(2));
    consumers.join();
    EXPECT_FALSE(consumers.results[0].succeeded());
    EXPECT_FALSE(consumers.results[1].succeeded());
    EXPECT_EQ(control->releases("window-holder"), 1);
    EXPECT_EQ(control->releases("window-waiter"), 1);
    EXPECT_EQ(provider->maxActive(), 1);
    // Even with a longer delivery budget, provider calls retain the configured cap.
    for (const auto timeout : provider->timeouts()) {
        EXPECT_GT(timeout, 0);
        EXPECT_LE(timeout, config.read_timeout_ms);
    }
}

TEST(MMRdmaProviderPoolTest, QueuedReadKeepsDeadlineFromBeforeProviderAcquisition) {
    auto       gate     = std::make_shared<ReadGate>();
    auto       control  = std::make_shared<RecordingControl>();
    auto       provider = std::make_shared<GatedRead>(gate);
    RdmaConfig config;
    config.read_timeout_ms = 500;
    MMRdmaReader reader(std::vector<std::shared_ptr<Read>>{provider}, config);
    Consumers    consumers(reader, control, gate, 2);
    consumers.start(0, "deadline-holder");
    ASSERT_TRUE(gate->waitForEntries(1));
    consumers.start(1, "deadline-waiter");
    ASSERT_TRUE(consumers.waitForStart(1));
    const auto     waiter_started = consumers.startedAt(1);
    constexpr auto queue_time     = std::chrono::milliseconds(100);
    // A bounded gate holds the slot while the waiter queues. This is a deadline
    // regression check, not a simulated production throughput measurement.
    gate->openAt(waiter_started + queue_time);
    consumers.join();
    ASSERT_TRUE(consumers.results[0].succeeded());
    ASSERT_TRUE(consumers.results[1].succeeded());
    const auto invocations = provider->invocations();
    ASSERT_EQ(invocations.count("deadline-holder"), 1u);
    ASSERT_EQ(invocations.count("deadline-waiter"), 1u);
    const auto& waiter = invocations.at("deadline-waiter");
    EXPECT_GE(waiter.started, waiter_started + queue_time);
    // Allow 25ms for receipt parsing, scheduling and millisecond rounding. Granting a
    // fresh 500ms after the controlled 100ms queue wait must fail this check.
    EXPECT_LE(waiter.deadline,
              waiter_started + std::chrono::milliseconds(config.read_timeout_ms) + std::chrono::milliseconds(25));
    EXPECT_EQ(provider->maxActive(), 1);
    EXPECT_EQ(control->releases("deadline-holder"), 1);
    EXPECT_EQ(control->releases("deadline-waiter"), 1);
}

TEST(MMRdmaProviderPoolTest, DeferredPartialReadRetainsLeaseWhileProviderSlotCanBeReused) {
    auto gate = std::make_shared<ReadGate>();
    gate->open();
    auto         control  = std::make_shared<RecordingControl>();
    auto         provider = std::make_shared<GatedRead>(gate);
    RdmaConfig   config;
    MMRdmaReader reader(std::vector<std::shared_ptr<Read>>{provider}, config);
    provider->failNext(GatedRead::Outcome::DEFERRED_ERROR);
    EXPECT_FALSE(consume(reader, control, receipt({"complete-chunk", "pending-chunk"})).succeeded());
    auto retained = provider->pendingStorage();
    EXPECT_FALSE(retained.expired());
    EXPECT_EQ(control->releases("complete-chunk"), 0);
    EXPECT_EQ(control->releases("pending-chunk"), 0);

    EXPECT_TRUE(consume(reader, control, receipt({"later-request"})).succeeded());
    EXPECT_EQ(control->releases("later-request"), 1);
    EXPECT_FALSE(retained.expired());
    provider->completeDeferred();
    provider->completeDeferred();
    EXPECT_TRUE(retained.expired());
    EXPECT_EQ(control->releases("complete-chunk"), 1);
    EXPECT_EQ(control->releases("pending-chunk"), 1);
}

TEST(MMRdmaProviderPoolTest, ExpiredBudgetDoesNotBorrowProviderAndReleasesLeaseOnce) {
    auto         gate     = std::make_shared<ReadGate>();
    auto         control  = std::make_shared<RecordingControl>();
    auto         provider = std::make_shared<GatedRead>(gate);
    RdmaConfig   config;
    MMRdmaReader reader(std::vector<std::shared_ptr<Read>>{provider}, config);
    EXPECT_FALSE(consume(reader, control, receipt({"already-expired"}), -1).succeeded());
    EXPECT_EQ(gate->entries(), 0u);
    EXPECT_EQ(control->releases("already-expired"), 1);
    EXPECT_EQ(control->asyncReleases("already-expired"), 1);
}

TEST(MMRdmaProviderPoolTest, InvalidProviderVectorsFailBeforeAdvertisingRdma) {
    auto       gate     = std::make_shared<ReadGate>();
    auto       provider = std::make_shared<GatedRead>(gate);
    RdmaConfig config;
    auto       construct = [&](std::vector<std::shared_ptr<Read>> providers) {
        return std::make_unique<MMRdmaReader>(std::move(providers), config);
    };
    EXPECT_THROW(construct({}), std::invalid_argument);
    EXPECT_THROW(construct({nullptr}), std::invalid_argument);
    EXPECT_THROW(construct({provider, nullptr}), std::invalid_argument);
    EXPECT_THROW(construct({provider, provider}), std::invalid_argument);
    EXPECT_EQ(gate->entries(), 0u);
}

}  // namespace
}  // namespace rtp_llm
