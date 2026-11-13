#include "kmonitor/client/core/MetricsSystem.h"
#include "kmonitor/client/core/MetricsSource.h"
#include "kmonitor/client/core/MetricsInfo.h"
#include "kmonitor/client/sink/Sink.h"
#include <gtest/gtest.h>
#include <chrono>
#include <future>
#include <thread>

namespace {
class Source: public kmonitor::MetricsSource {
public:
    const std::string& Name() override {
        return name;
    }
    void
    GetMetrics(kmonitor::MetricsCollector* collector, const std::set<kmonitor::MetricLevel>&, int64_t now) override {
        if (pending != 0) {
            auto* record = collector->AddRecord(name, std::make_shared<kmonitor::MetricsTags>(), now);
            record->AddValue(std::make_shared<kmonitor::MetricsInfo>(name, ""), pending);
        }
        pending = 0;
    }
    std::string name    = "load";
    double      pending = 0;
};

class Sink: public kmonitor::Sink {
public:
    Sink(): kmonitor::Sink("test") {}
    bool Init() override {
        return true;
    }
    void PutMetrics(kmonitor::MetricsRecord* record) override {
        for (const auto* value : record->Values()) {
            values.push_back(value->Value());
        }
    }
    void Flush() override {
        ++flushes;
    }
    std::vector<std::string> values;
    int                      flushes = 0;
};

class BlockingSink: public Sink {
public:
    explicit BlockingSink(std::shared_future<void> release): release_(std::move(release)) {}
    void Flush() override {
        if (flushes == 0) {
            entered.set_value();
            release_.wait();
        }
        Sink::Flush();
    }
    std::promise<void> entered;

private:
    std::shared_future<void> release_;
};

TEST(MetricsReportingTest, NormalSamplingDoesNotWaitForSlowTransport) {
    class SamplingSource: public Source {
    public:
        void GetMetrics(kmonitor::MetricsCollector*            collector,
                        const std::set<kmonitor::MetricLevel>& levels,
                        int64_t                                now) override {
            Source::GetMetrics(collector, levels, now);
            if (++samples == 2) {
                sampled_twice.set_value();
            }
        }
        int                samples = 0;
        std::promise<void> sampled_twice;
    } source;
    kmonitor::MetricsSystem system;
    system.AddSource(&source);
    std::promise<void> release;
    auto               sink = std::make_shared<BlockingSink>(release.get_future().share());
    system.AddSink(sink);
    source.pending = 11;
    auto sender    = std::async(std::launch::async, [&] { system.ManuallySnapshot(); });
    if (sink->entered.get_future().wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        release.set_value();
        sender.get();
        FAIL() << "sender did not enter the transport";
    }
    // ManuallySnapshot aligns its clock to milliseconds.
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    auto       sampler = std::async(std::launch::async, [&] { system.ManuallySnapshot(); });
    const auto sampled = source.sampled_twice.get_future().wait_for(std::chrono::seconds(1));
    release.set_value();
    sender.get();
    sampler.get();
    EXPECT_EQ(sampled, std::future_status::ready);
}

TEST(MetricsReportingTest, SleepStillWaitsForInflightTransport) {
    Source                  source;
    kmonitor::MetricsSystem system;
    system.AddSource(&source);
    std::promise<void> release;
    auto               sink = std::make_shared<BlockingSink>(release.get_future().share());
    system.AddSink(sink);
    source.pending = 11;
    auto sender    = std::async(std::launch::async, [&] { system.ManuallySnapshot(); });
    if (sink->entered.get_future().wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        release.set_value();
        sender.get();
        FAIL() << "sender did not enter the transport";
    }
    auto paused = std::async(std::launch::async, [&] { return system.SetReportingEnabled(false); });
    EXPECT_EQ(paused.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    release.set_value();
    sender.get();
    EXPECT_TRUE(paused.get());
    source.pending = 99;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    system.ManuallySnapshot();
    EXPECT_EQ(sink->flushes, 1);
}

TEST(MetricsReportingTest, SleepDisconnectsAndWakeDoesNotReplayBufferedSamples) {
    Source                  source;
    kmonitor::MetricsSystem system;
    system.AddSource(&source);
    auto sink = std::make_shared<Sink>();
    system.AddSink(sink);
    source.pending = 11;
    system.ManuallySnapshot();
    ASSERT_EQ(sink->values, std::vector<std::string>({"11"}));
    ASSERT_TRUE(system.SetReportingEnabled(false));
    EXPECT_EQ(system.GetSink("test"), nullptr);
    const int flushes = sink->flushes;
    source.pending    = 99;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    system.ManuallySnapshot();
    EXPECT_EQ(source.pending, 0);
    EXPECT_EQ(sink->flushes, flushes);
    source.pending = 88;  // unsampled at the moment wake begins
    ASSERT_TRUE(system.SetReportingEnabled(true));
    EXPECT_EQ(source.pending, 0);
    system.AddSink(sink);
    source.pending = 22;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    system.ManuallySnapshot();
    EXPECT_EQ(sink->values, std::vector<std::string>({"11", "22"}));
}
}  // namespace
