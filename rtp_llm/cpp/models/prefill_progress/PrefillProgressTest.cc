#include "rtp_llm/cpp/models/prefill_progress/PrefillProgress.h"

#include <cstdlib>
#include <endian.h>
#include <gtest/gtest.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

namespace rtp_llm {
namespace {

struct TestSlot {
    ~TestSlot() {
        if (fd >= 0) {
            close(fd);
            unlink(path);
        }
    }

    uint64_t read() const {
        uint64_t word = 0;
        EXPECT_EQ(pread(fd, &word, sizeof(word), 0), static_cast<ssize_t>(sizeof(word)));
        return le64toh(word);
    }

    char path[sizeof("/tmp/rtp-prefill-progress-XXXXXX")] = "/tmp/rtp-prefill-progress-XXXXXX";
    int  fd                                               = mkstemp(path);
};

TEST(PrefillProgressTest, PublishesStagesAndClearsOnExit) {
    TestSlot slot;
    ASSERT_GE(slot.fd, 0);
    ASSERT_EQ(ftruncate(slot.fd, sizeof(uint64_t)), 0);
    ASSERT_EQ(setenv("RTP_LLM_PREFILL_PROGRESS_SLOT", slot.path, 1), 0);
    ASSERT_TRUE(PrefillProgressScope::enabled());

    {
        PrefillProgressScope ineligible(false);
        PrefillProgressScope::mark(PrefillPhase::GPU_WAIT);
        EXPECT_FALSE(PrefillProgressScope::active());
        EXPECT_EQ(slot.read(), uint64_t{0});
    }

    uint64_t last_sequence = 0;
    auto     check         = [&](PrefillPhase phase, uint64_t layer = 0) {
        const uint64_t word = slot.read();
        EXPECT_EQ(word & 15, static_cast<uint64_t>(phase));
        EXPECT_EQ((word >> 4) & 1023, layer);
        EXPECT_EQ((word >> 14) & ((uint64_t{1} << 22) - 1), static_cast<uint64_t>(syscall(SYS_gettid)));
        EXPECT_EQ(word >> 36, ++last_sequence);
    };
    auto run = [&]() {
        PrefillProgressScope progress(true);
        ASSERT_TRUE(PrefillProgressScope::active());
        check(PrefillPhase::PREPARE);
        std::thread other([] {
            PrefillProgressScope contender(true);
            EXPECT_FALSE(PrefillProgressScope::active());
            PrefillProgressScope::mark(PrefillPhase::TAIL);
        });
        other.join();
        PrefillProgressScope::markLayer(-1);
        check(PrefillPhase::LAYER);
        PrefillProgressScope::markLayer(2);
        check(PrefillPhase::LAYER, 3);
        PrefillProgressScope::mark(PrefillPhase::GRAPH);
        check(PrefillPhase::GRAPH);
        PrefillProgressScope::mark(PrefillPhase::TAIL);
        check(PrefillPhase::TAIL);
        PrefillProgressScope::mark(PrefillPhase::GPU_WAIT);
        check(PrefillPhase::GPU_WAIT);
        PrefillProgressScope::mark(PrefillPhase::GPU_DONE);
        check(PrefillPhase::GPU_DONE);
        throw 123;
    };
    EXPECT_THROW(run(), int);
    EXPECT_FALSE(PrefillProgressScope::active());
    EXPECT_EQ(slot.read(), uint64_t{0});

    {
        PrefillProgressScope next(true);
        check(PrefillPhase::PREPARE);
    }
    EXPECT_EQ(slot.read(), uint64_t{0});
    EXPECT_EQ(unsetenv("RTP_LLM_PREFILL_PROGRESS_SLOT"), 0);
}

}  // namespace
}  // namespace rtp_llm
