#include "rtp_llm/cpp/cache/connector/memory/MemoryCopyOperationId.h"

#include <set>
#include <string>

#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(MemoryCopyOperationIdTest, OneInstanceNeverRepeatsAnId) {
    MemoryCopyOperationIdGenerator generator;
    std::set<std::string>           ids;
    for (int i = 0; i < 1000; ++i) {
        EXPECT_TRUE(ids.emplace(generator.next()).second);
    }
}

TEST(MemoryCopyOperationIdTest, ConnectorInstancesHaveDifferentEpochs) {
    MemoryCopyOperationIdGenerator first;
    MemoryCopyOperationIdGenerator second;

    EXPECT_NE(first.next(), second.next());
}

TEST(MemoryCopyOperationIdTest, GeneratorCreatedBeforeForkDoesNotRepeatIds) {
    MemoryCopyOperationIdGenerator generator;
    int                            pipe_fds[2];
    ASSERT_EQ(pipe(pipe_fds), 0);

    const pid_t child = fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        close(pipe_fds[0]);
        const std::string child_id = generator.next();
        const auto        written  = write(pipe_fds[1], child_id.data(), child_id.size());
        close(pipe_fds[1]);
        _exit(written == static_cast<ssize_t>(child_id.size()) ? 0 : 1);
    }

    close(pipe_fds[1]);
    const std::string parent_id = generator.next();
    char              child_id_buffer[512];
    const auto        child_id_size = read(pipe_fds[0], child_id_buffer, sizeof(child_id_buffer));
    close(pipe_fds[0]);
    int child_status = 0;
    ASSERT_EQ(waitpid(child, &child_status, 0), child);
    ASSERT_TRUE(WIFEXITED(child_status));
    ASSERT_EQ(WEXITSTATUS(child_status), 0);
    ASSERT_GT(child_id_size, 0);

    const std::string child_id(child_id_buffer, static_cast<size_t>(child_id_size));
    EXPECT_NE(parent_id, child_id);
}

}  // namespace
}  // namespace rtp_llm
