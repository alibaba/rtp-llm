#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockIO.h"

#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "gtest/gtest.h"

namespace rtp_llm::block_tree_cache {
namespace {

std::string testFilePath(const std::string& name) {
    return ::testing::TempDir() + "/disk_block_io_test_" + name;
}

}  // namespace

TEST(DiskBlockIOTest, OpenPreallocateWriteReadAndClose) {
    const std::string path = testFilePath("basic.bin");
    ::remove(path.c_str());

    PosixDiskBlockIO io;
    ASSERT_EQ(io.openAndPreallocate(path, 4096, /*buffered_io=*/true), DiskBlockIOStatus::OK);

    // The file must be preallocated to at least the requested size.
    struct stat st {};
    ASSERT_EQ(::stat(path.c_str(), &st), 0);
    EXPECT_GE(static_cast<size_t>(st.st_size), 4096u);

    std::vector<char> write_buf(4096);
    for (size_t i = 0; i < write_buf.size(); ++i) {
        write_buf[i] = static_cast<char>(i % 251);
    }
    ASSERT_EQ(io.write(0, write_buf.data(), write_buf.size()), DiskBlockIOStatus::OK);

    std::vector<char> read_buf(4096, 0);
    ASSERT_EQ(io.read(0, read_buf.data(), read_buf.size()), DiskBlockIOStatus::OK);
    EXPECT_EQ(read_buf, write_buf);

    EXPECT_FALSE(io.debugString().empty());

    io.close();
    // close() must be idempotent.
    io.close();

    ::remove(path.c_str());
}

TEST(DiskBlockIOTest, DirectIOAlignmentError) {
    const std::string path = testFilePath("direct_align.bin");
    ::remove(path.c_str());

    PosixDiskBlockIO io;
    // Best-effort open: some sandbox filesystems (tmpfs/overlayfs) reject O_DIRECT
    // with EINVAL, so openAndPreallocate may legitimately fail here. Alignment
    // checking in read()/write() is driven purely by the requested I/O mode and the
    // call parameters, not by whether the underlying open() call actually succeeded
    // with O_DIRECT, so the assertions below must hold regardless of filesystem
    // support.
    io.openAndPreallocate(path, 8192, /*buffered_io=*/false);

    alignas(4096) char buffer[8192];

    // Misaligned offset.
    EXPECT_EQ(io.write(1, buffer, 4096), DiskBlockIOStatus::ALIGNMENT_ERROR);
    EXPECT_EQ(io.read(1, buffer, 4096), DiskBlockIOStatus::ALIGNMENT_ERROR);

    // Misaligned size.
    EXPECT_EQ(io.write(0, buffer, 100), DiskBlockIOStatus::ALIGNMENT_ERROR);
    EXPECT_EQ(io.read(0, buffer, 100), DiskBlockIOStatus::ALIGNMENT_ERROR);

    // Misaligned buffer address.
    char* misaligned_buffer = buffer + 1;
    EXPECT_EQ(io.write(0, misaligned_buffer, 4096), DiskBlockIOStatus::ALIGNMENT_ERROR);
    EXPECT_EQ(io.read(0, misaligned_buffer, 4096), DiskBlockIOStatus::ALIGNMENT_ERROR);

    io.close();
    ::remove(path.c_str());
}

TEST(DiskBlockIOTest, BatchReadWriteLoopsAllRequests) {
    const std::string path = testFilePath("batch.bin");
    ::remove(path.c_str());

    PosixDiskBlockIO io;
    ASSERT_EQ(io.openAndPreallocate(path, 4096 * 4, /*buffered_io=*/true), DiskBlockIOStatus::OK);

    std::vector<char> buf_a(1024, 'A');
    std::vector<char> buf_b(1024, 'B');
    std::vector<char> buf_c(1024, 'C');

    std::vector<DiskWrite> writes = {
        {0, buf_a.data(), buf_a.size()},
        {1024, buf_b.data(), buf_b.size()},
        {2048, buf_c.data(), buf_c.size()},
    };
    ASSERT_EQ(io.write(writes), DiskBlockIOStatus::OK);

    std::vector<char> read_a(1024, 0);
    std::vector<char> read_b(1024, 0);
    std::vector<char> read_c(1024, 0);
    std::vector<DiskRead> reads = {
        {0, read_a.data(), read_a.size()},
        {1024, read_b.data(), read_b.size()},
        {2048, read_c.data(), read_c.size()},
    };
    ASSERT_EQ(io.read(reads), DiskBlockIOStatus::OK);
    EXPECT_EQ(read_a, buf_a);
    EXPECT_EQ(read_b, buf_b);
    EXPECT_EQ(read_c, buf_c);

    // A batch with a failing request (zero-length, invalid) in the middle must stop
    // at the first failure and must not execute the request that follows it.
    std::vector<char> buf_d(1024, 'D');
    std::vector<DiskWrite> writes_with_failure = {
        {3072, buf_a.data(), buf_a.size()},
        {0, buf_b.data(), 0},  // INVALID_SIZE, should short-circuit the batch
        {2048, buf_d.data(), buf_d.size()},
    };
    EXPECT_EQ(io.write(writes_with_failure), DiskBlockIOStatus::INVALID_SIZE);

    // Confirm the request after the failing one was never issued: offset 2048 must
    // still hold buf_c's pattern, not buf_d's.
    std::vector<char> check(1024, 0);
    ASSERT_EQ(io.read(2048, check.data(), check.size()), DiskBlockIOStatus::OK);
    EXPECT_EQ(check, buf_c);

    io.close();
    ::remove(path.c_str());
}

}  // namespace rtp_llm::block_tree_cache
