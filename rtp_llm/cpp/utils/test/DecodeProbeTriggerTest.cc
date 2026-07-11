#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "rtp_llm/cpp/utils/DecodeProbeTrigger.h"

namespace rtp_llm {
namespace {

std::string normalizedName(const std::string& name) {
    return name.front() == '/' ? name : "/" + name;
}

std::string uniqueShmName() {
    static std::atomic<uint64_t> sequence{0};
    return "rtpllm_decode_probe_trigger_" + std::to_string(getpid()) + "_" + std::to_string(sequence.fetch_add(1));
}

uint64_t nowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

class SharedRecordMapping {
public:
    explicit SharedRecordMapping(const std::string& name): name_(normalizedName(name)) {
        fd_ = shm_open(name_.c_str(), O_RDWR, 0);
        EXPECT_NE(fd_, -1);
        if (fd_ == -1) {
            return;
        }
        void* address = mmap(nullptr,
                             sizeof(detail::DecodeProbeTriggerSharedRecord),
                             PROT_READ | PROT_WRITE,
                             MAP_SHARED,
                             fd_,
                             0);
        EXPECT_NE(address, MAP_FAILED);
        if (address != MAP_FAILED) {
            record_ = static_cast<detail::DecodeProbeTriggerSharedRecord*>(address);
        }
    }

    ~SharedRecordMapping() {
        if (record_ != nullptr) {
            munmap(record_, sizeof(detail::DecodeProbeTriggerSharedRecord));
        }
        if (fd_ != -1) {
            close(fd_);
        }
    }

    detail::DecodeProbeTriggerSharedRecord* get() const {
        return record_;
    }

private:
    std::string name_;
    int         fd_{-1};
    detail::DecodeProbeTriggerSharedRecord* record_{nullptr};
};

class DecodeProbeTriggerTest: public ::testing::Test {
protected:
    void SetUp() override {
        name_ = uniqueShmName();
        shm_unlink(normalizedName(name_).c_str());
    }

    void TearDown() override {
        shm_unlink(normalizedName(name_).c_str());
    }

    std::string name_;
};

TEST_F(DecodeProbeTriggerTest, PublishPeekAndAcknowledgeAcrossMappings) {
    DecodeProbeTriggerRegistry writer(name_.c_str(), true);
    DecodeProbeTriggerRegistry reader(name_.c_str(), true);
    ASSERT_TRUE(writer.enabled());
    ASSERT_TRUE(reader.enabled());

    DecodeProbeTriggerEvent published;
    published.generation               = 1;
    published.timestamp_us             = nowUs();
    published.observed_sequence_length = 42;
    published.trace_id                 = "trace-abc";
    published.reason                   = "checksum-mismatch";
    published.required_rank_mask       = 0b11;
    ASSERT_TRUE(writer.publish(published));

    DecodeProbeTriggerEvent observed;
    ASSERT_TRUE(reader.peek(observed));
    EXPECT_EQ(observed.generation, 1);
    EXPECT_EQ(observed.timestamp_us, published.timestamp_us);
    EXPECT_EQ(observed.observed_sequence_length, 42);
    EXPECT_EQ(observed.trace_id, "trace-abc");
    EXPECT_EQ(observed.reason, "checksum-mismatch");
    EXPECT_EQ(observed.required_rank_mask, 0b11);
    EXPECT_EQ(observed.ack_rank_mask, 0);

    EXPECT_TRUE(reader.acknowledge(1, 0));
    EXPECT_TRUE(reader.acknowledge(1, 1));
    ASSERT_TRUE(writer.peek(observed));
    EXPECT_EQ(observed.ack_rank_mask, 0b11);
}

TEST_F(DecodeProbeTriggerTest, ReaderRejectsTornOrWrongVersionRecord) {
    DecodeProbeTriggerRegistry writer(name_.c_str(), true);
    DecodeProbeTriggerRegistry reader(name_.c_str(), true);
    ASSERT_TRUE(writer.enabled());

    DecodeProbeTriggerEvent published;
    published.generation         = 1;
    published.timestamp_us       = nowUs();
    published.trace_id           = "trace";
    published.reason             = "reason";
    published.required_rank_mask = 1;
    ASSERT_TRUE(writer.publish(published));

    SharedRecordMapping mapping(name_);
    mapping.get()->version = detail::kDecodeProbeTriggerRecordVersion + 1;
    DecodeProbeTriggerEvent observed;
    EXPECT_FALSE(reader.peek(observed));

    mapping.get()->version = detail::kDecodeProbeTriggerRecordVersion;
    mapping.get()->generation.store(0, std::memory_order_release);
    DecodeProbeTriggerRegistry torn_reader(name_.c_str(), true);
    EXPECT_FALSE(torn_reader.peek(observed));
}

TEST_F(DecodeProbeTriggerTest, UnackedGenerationIsNotOverwrittenBeforeExpiry) {
    constexpr uint64_t kExpiryUs = 1000;
    DecodeProbeTriggerRegistry registry(name_.c_str(), true, kExpiryUs);
    ASSERT_TRUE(registry.enabled());

    DecodeProbeTriggerEvent first;
    first.generation         = 1;
    first.timestamp_us       = nowUs();
    first.trace_id           = "first";
    first.reason             = "first-reason";
    first.required_rank_mask = 0b11;
    ASSERT_TRUE(registry.publish(first));

    DecodeProbeTriggerEvent second = first;
    second.generation              = 2;
    second.trace_id                = "second";
    EXPECT_FALSE(registry.publish(second));

    std::this_thread::sleep_for(std::chrono::microseconds(kExpiryUs + 1000));
    EXPECT_TRUE(registry.publish(second));

    DecodeProbeTriggerEvent observed;
    ASSERT_TRUE(registry.peek(observed));
    EXPECT_EQ(observed.generation, 2);
    EXPECT_EQ(observed.trace_id, "second");
}

TEST_F(DecodeProbeTriggerTest, DisabledModeDoesNotCreateSharedMemory) {
    unsetenv("RTPLLM_RETROSPECTIVE_PROBE_DEBUG");
    setenv("RTPLLM_RETROSPECTIVE_PROBE_SHM_NAME", name_.c_str(), 1);
    EXPECT_FALSE(DecodeProbeTrigger::enabled());

    DecodeProbeTriggerEvent event;
    EXPECT_FALSE(DecodeProbeTrigger::publish(event));

    const auto normalized_name = normalizedName(name_);
    const int  fd              = shm_open(normalized_name.c_str(), O_RDONLY, 0);
    EXPECT_EQ(fd, -1);
    if (fd != -1) {
        close(fd);
    }
}

}  // namespace
}  // namespace rtp_llm
