#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockIO.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"

#include <cstdio>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "gtest/gtest.h"

namespace rtp_llm {
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

namespace {

// physical_block_count = disk_size_bytes / stride_bytes = 4, so totalBlocksNum() (which
// excludes reserved block 0) is 3.
std::shared_ptr<DiskBlockPoolConfig> makeDiskConfig(const std::string& work_dir,
                                                     size_t             disk_size_bytes = 4 * 4096,
                                                     size_t             stride_bytes    = 4096,
                                                     size_t             payload_bytes   = 1024,
                                                     bool               buffered_io     = true) {
    auto config                     = std::make_shared<DiskBlockPoolConfig>();
    config->pool_type               = BlockPoolType::DISK;
    config->pool_name                = "disk";
    config->free_block_order_policy = FreeBlockOrderPolicy::ASCENDING_ORDER;
    config->work_dir                = work_dir;
    config->local_rank              = 0;
    config->world_rank              = 0;
    config->disk_size_bytes         = disk_size_bytes;
    config->payload_bytes           = payload_bytes;
    config->stride_bytes            = stride_bytes;
    config->buffered_io             = buffered_io;
    return config;
}

// A DiskBlockIO stub whose read/write always report PARTIAL_FAILURE, used to prove
// that DiskBlockPool surfaces the underlying DiskBlockIOStatus -> BlockIOStatus
// mapping faithfully instead of only ever returning OK/error-from-real-IO.
class FailingDiskBlockIO: public DiskBlockIO {
public:
    DiskBlockIOStatus openAndPreallocate(const std::string&, size_t, bool) override {
        return DiskBlockIOStatus::OK;
    }
    DiskBlockIOStatus read(uint64_t, void*, size_t) override {
        return DiskBlockIOStatus::PARTIAL_FAILURE;
    }
    DiskBlockIOStatus write(uint64_t, const void*, size_t) override {
        return DiskBlockIOStatus::PARTIAL_FAILURE;
    }
    DiskBlockIOStatus read(const std::vector<DiskRead>&) override {
        return DiskBlockIOStatus::PARTIAL_FAILURE;
    }
    DiskBlockIOStatus write(const std::vector<DiskWrite>&) override {
        return DiskBlockIOStatus::PARTIAL_FAILURE;
    }
    void close() override {}
    std::string debugString() const override {
        return "FailingDiskBlockIO";
    }
};

}  // namespace

TEST(DiskBlockPoolTest, InitPreallocatesFileAndSkipsOffsetZero) {
    auto          config = makeDiskConfig(::testing::TempDir());
    DiskBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    struct stat st {};
    ASSERT_EQ(::stat(pool.filePath().c_str(), &st), 0);
    EXPECT_GE(static_cast<size_t>(st.st_size), config->disk_size_bytes);

    // physical_block_count(4) - reserved block 0 = 3 usable blocks.
    EXPECT_EQ(pool.totalBlocksNum(), 3u);
    EXPECT_EQ(pool.blockOffset(0), 0u);

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_EQ(*block, 1);
    EXPECT_EQ(pool.blockOffset(*block), pool.strideBytes());

    ::remove(pool.filePath().c_str());
}

TEST(DiskBlockPoolTest, MallocReturnsAscendingBlocks) {
    auto          config = makeDiskConfig(::testing::TempDir());
    DiskBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto b1 = pool.malloc();
    auto b2 = pool.malloc();
    auto b3 = pool.malloc();
    ASSERT_TRUE(b1.has_value());
    ASSERT_TRUE(b2.has_value());
    ASSERT_TRUE(b3.has_value());
    EXPECT_EQ(*b1, 1);
    EXPECT_EQ(*b2, 2);
    EXPECT_EQ(*b3, 3);

    ::remove(pool.filePath().c_str());
}

TEST(DiskBlockPoolTest, ReadWriteRequireAllocatedBlock) {
    auto          config = makeDiskConfig(::testing::TempDir());
    DiskBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());

    std::vector<unsigned char> data(pool.strideBytes(), 0x5a);
    EXPECT_EQ(pool.write(*block, data.data(), data.size()), BlockIOStatus::OK);

    std::vector<unsigned char> read_buf(pool.strideBytes(), 0);
    EXPECT_EQ(pool.read(*block, read_buf.data(), read_buf.size()), BlockIOStatus::OK);
    EXPECT_EQ(read_buf, data);

    // bytes > stride_bytes -> INVALID_SIZE.
    std::vector<unsigned char> too_big(pool.strideBytes() + 1, 0);
    EXPECT_EQ(pool.write(*block, too_big.data(), too_big.size()), BlockIOStatus::INVALID_SIZE);

    // A valid, in-range block index that was never malloc'd -> INVALID_BLOCK.
    const BlockIdxType unallocated = *block + 1;
    EXPECT_EQ(pool.write(unallocated, data.data(), data.size()), BlockIOStatus::INVALID_BLOCK);
    EXPECT_EQ(pool.read(unallocated, read_buf.data(), read_buf.size()), BlockIOStatus::INVALID_BLOCK);

    // Once freed, the same block index becomes unallocated again.
    pool.free(*block);
    EXPECT_EQ(pool.write(*block, data.data(), data.size()), BlockIOStatus::INVALID_BLOCK);

    ::remove(pool.filePath().c_str());
}

TEST(DiskBlockPoolTest, BatchReadWriteUsesBlockOrder) {
    auto          config = makeDiskConfig(::testing::TempDir());
    DiskBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    auto blocks_opt = pool.malloc(3);
    ASSERT_TRUE(blocks_opt.has_value());
    const BlockIdList blocks = *blocks_opt;
    ASSERT_EQ(blocks.size(), 3u);

    std::vector<unsigned char> buf_a(pool.strideBytes(), 'A');
    std::vector<unsigned char> buf_b(pool.strideBytes(), 'B');
    std::vector<unsigned char> buf_c(pool.strideBytes(), 'C');
    std::vector<const void*>   srcs = {buf_a.data(), buf_b.data(), buf_c.data()};
    ASSERT_EQ(pool.write(blocks, srcs, pool.strideBytes()), BlockIOStatus::OK);

    std::vector<unsigned char> read_a(pool.strideBytes(), 0);
    std::vector<unsigned char> read_b(pool.strideBytes(), 0);
    std::vector<unsigned char> read_c(pool.strideBytes(), 0);
    std::vector<void*>         dsts = {read_a.data(), read_b.data(), read_c.data()};
    ASSERT_EQ(pool.read(blocks, dsts, pool.strideBytes()), BlockIOStatus::OK);
    EXPECT_EQ(read_a, buf_a);
    EXPECT_EQ(read_b, buf_b);
    EXPECT_EQ(read_c, buf_c);

    // Directly reading each block by index proves the batch call mapped
    // (block, buffer) pairs to (offset, buffer) in blocks[] order, not some other
    // order.
    const std::vector<unsigned char>* expected[] = {&buf_a, &buf_b, &buf_c};
    for (size_t i = 0; i < blocks.size(); ++i) {
        std::vector<unsigned char> check(pool.strideBytes(), 0);
        ASSERT_EQ(pool.read(blocks[i], check.data(), check.size()), BlockIOStatus::OK);
        EXPECT_EQ(check, *expected[i]);
    }

    ::remove(pool.filePath().c_str());
}

TEST(DiskBlockPoolTest, BlockZeroReadWriteReturnsInvalidBlock) {
    auto          config = makeDiskConfig(::testing::TempDir());
    DiskBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    std::vector<unsigned char> data(pool.strideBytes(), 0x11);
    EXPECT_EQ(pool.write(0, data.data(), data.size()), BlockIOStatus::INVALID_BLOCK);
    EXPECT_EQ(pool.read(0, data.data(), data.size()), BlockIOStatus::INVALID_BLOCK);

    ::remove(pool.filePath().c_str());
}

TEST(DiskBlockPoolTest, IoErrorStatusIsMapped) {
    auto          config = makeDiskConfig(::testing::TempDir());
    DiskBlockPool pool(config, std::make_unique<FailingDiskBlockIO>());
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());

    std::vector<unsigned char> data(pool.strideBytes(), 0);
    EXPECT_EQ(pool.write(*block, data.data(), data.size()), BlockIOStatus::PARTIAL_FAILURE);
    EXPECT_EQ(pool.read(*block, data.data(), data.size()), BlockIOStatus::PARTIAL_FAILURE);

    std::vector<const void*> srcs = {data.data()};
    EXPECT_EQ(pool.write(BlockIdList{*block}, srcs, pool.strideBytes()), BlockIOStatus::PARTIAL_FAILURE);
    std::vector<void*> dsts = {data.data()};
    EXPECT_EQ(pool.read(BlockIdList{*block}, dsts, pool.strideBytes()), BlockIOStatus::PARTIAL_FAILURE);
}

}  // namespace rtp_llm
