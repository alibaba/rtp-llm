#include "rtp_llm/cpp/models/prefill_progress/PrefillProgress.h"

#include <cstdlib>
#include <endian.h>
#include <gtest/gtest.h>
#include <unistd.h>

namespace rtp_llm {

TEST(PrefillProgressTest, PublishesPrepareAndClearsOnExit) {
    char      path[] = "/tmp/rtp-prefill-progress-XXXXXX";
    const int fd     = mkstemp(path);
    ASSERT_GE(fd, 0);
    uint64_t word = 0;
    EXPECT_EQ(ftruncate(fd, sizeof(word)), 0);
    PrefillProgressScope::initialize(path);
    unlink(path);

    {
        PrefillProgressScope progress(true);
        EXPECT_EQ(pread(fd, &word, sizeof(word), 0), static_cast<ssize_t>(sizeof(word)));
        EXPECT_EQ(le64toh(word) & 15, static_cast<uint64_t>(PrefillPhase::PREPARE));
    }
    EXPECT_EQ(pread(fd, &word, sizeof(word), 0), static_cast<ssize_t>(sizeof(word)));
    EXPECT_EQ(word, uint64_t{0});
    close(fd);
}

}  // namespace rtp_llm
