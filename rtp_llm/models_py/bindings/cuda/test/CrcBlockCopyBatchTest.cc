#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#ifdef RTP_LLM_CRC_INTERNAL_TESTS
#include "rtp_llm/models_py/bindings/cuda/CrcBlockCopyInternal.cuh"
#endif

namespace rtp_llm {
namespace {
void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
uint32_t crc32c(const uint8_t* data, size_t bytes) {
    uint32_t crc = ~uint32_t(0);
    for (size_t i = 0; i < bytes; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ ((crc & 1) ? 0x82f63b78U : 0);
    }
    return ~crc;
}
struct Block {
    uint8_t*             host{nullptr};
    uint8_t*             device{nullptr};
    size_t               bytes;
    size_t               encoded;
    std::vector<uint8_t> expected;
    explicit Block(size_t size, int seed = 0):
        bytes(size), encoded(CrcBlockCopyBatch::encodedBytes(size)), expected(size) {
        cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&host), encoded, cudaHostAllocDefault));
        auto error = cudaMalloc(reinterpret_cast<void**>(&device), bytes);
        if (error != cudaSuccess) {
            cudaFreeHost(host);
            host = nullptr;
            cudaCheck(error);
        }
        for (size_t i = 0; i < bytes; ++i)
            expected[i] = uint8_t(i * 13 + seed * 37);
        cudaCheck(cudaMemcpy(device, expected.data(), bytes, cudaMemcpyHostToDevice));
        std::memset(host, 0x6b, encoded);
    }
    ~Block() {
        if (device)
            cudaFree(device);
        if (host)
            cudaFreeHost(host);
    }
    CrcCopyItem item(bool ragged = false) {
        CrcCopyItem result{host, bytes, encoded, {}};
        if (ragged && bytes > 20)
            result.tiles = {{device, 0, 3}, {device + 3, 3, 17}, {device + 20, 20, bytes - 20}};
        else
            result.tiles = {{device, 0, bytes}};
        return result;
    }
    std::vector<uint8_t> readDevice() {
        std::vector<uint8_t> data(bytes);
        cudaCheck(cudaMemcpy(data.data(), device, bytes, cudaMemcpyDeviceToHost));
        return data;
    }
    uint32_t footer() const {
        uint32_t value;
        std::memcpy(&value, host + encoded - 4, 4);
        return value;
    }
};

struct TestAllocation {
    void*  pointer{nullptr};
    size_t bytes;
    bool   pinned;
    explicit TestAllocation(size_t size, bool host = false): bytes(size), pinned(host) {
        cudaCheck(host ? cudaHostAlloc(&pointer, bytes, cudaHostAllocDefault) : cudaMalloc(&pointer, bytes));
    }
    ~TestAllocation() {
        if (pointer) {
            if (pinned)
                cudaFreeHost(pointer);
            else
                cudaFree(pointer);
        }
    }
    TestAllocation(const TestAllocation&)            = delete;
    TestAllocation& operator=(const TestAllocation&) = delete;
    uint8_t*        data() const {
        return static_cast<uint8_t*>(pointer);
    }
};

// Setups explicitly complete before a backend's nonblocking stream can run.
void upload(void* target, const void* source, size_t bytes) {
    cudaCheck(cudaMemcpyAsync(target, source, bytes, cudaMemcpyHostToDevice, nullptr));
    cudaCheck(cudaStreamSynchronize(nullptr));
}

struct GuardedBlock {
    static constexpr uint8_t kHostGuard    = 0xa5;
    static constexpr uint8_t kDeviceGuard  = 0xd3;
    static constexpr size_t  kHostPrefix   = 3;
    static constexpr size_t  kDevicePrefix = 17;
    size_t                   bytes, encoded;
    TestAllocation           host_allocation, device_allocation;
    uint8_t*                 host;
    uint8_t*                 device;
    std::vector<uint8_t>     expected;
    uint32_t                 expected_crc;

    explicit GuardedBlock(size_t payload, unsigned seed):
        bytes(payload),
        encoded(CrcBlockCopyBatch::encodedBytes(payload)),
        host_allocation(encoded + kHostPrefix + 64, true),
        device_allocation(bytes + kDevicePrefix + 64),
        host(host_allocation.data() + kHostPrefix),
        device(device_allocation.data() + kDevicePrefix),
        expected(bytes) {
        uint32_t state = seed + 1;
        for (auto& value : expected) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            value = uint8_t(state);
        }
        expected_crc = crc32c(expected.data(), expected.size());
        cudaPointerAttributes attributes{};
        cudaCheck(cudaPointerGetAttributes(&attributes, host));
        if (attributes.type != cudaMemoryTypeHost)
            throw std::runtime_error("guarded CRC fixture must use pinned host storage");
        resetHost();
        resetDevice(true);
    }

    CrcCopyItem item() const {
        CrcCopyItem value{host, bytes, encoded, {}};
        if (bytes > 20)
            value.tiles = {{device, 0, 3}, {device + 3, 3, 17}, {device + 20, 20, bytes - 20}};
        else
            value.tiles = {{device, 0, bytes}};
        return value;
    }
    void resetHost() {
        std::memset(host_allocation.pointer, kHostGuard, host_allocation.bytes);
    }
    void writeCpuRecord() {
        resetHost();
        std::memcpy(host, expected.data(), bytes);
        std::memset(host + bytes, 0x7b, encoded - 4 - bytes);
        std::memcpy(host + encoded - 4, &expected_crc, sizeof(expected_crc));
    }
    void resetDevice(bool valid) {
        std::vector<uint8_t> initial(device_allocation.bytes, kDeviceGuard);
        if (valid)
            std::memcpy(initial.data() + kDevicePrefix, expected.data(), bytes);
        upload(device_allocation.pointer, initial.data(), initial.size());
    }
    void expectDevice(bool valid) const {
        std::vector<uint8_t> actual(device_allocation.bytes);
        cudaCheck(cudaMemcpy(actual.data(), device_allocation.pointer, actual.size(), cudaMemcpyDeviceToHost));
        EXPECT_TRUE(std::all_of(
            actual.begin(), actual.begin() + kDevicePrefix, [](uint8_t value) { return value == kDeviceGuard; }));
        if (valid)
            EXPECT_EQ(std::memcmp(actual.data() + kDevicePrefix, expected.data(), bytes), 0);
        else
            EXPECT_TRUE(std::all_of(actual.begin() + kDevicePrefix,
                                    actual.begin() + kDevicePrefix + bytes,
                                    [](uint8_t value) { return value == kDeviceGuard; }));
        EXPECT_TRUE(std::all_of(
            actual.begin() + kDevicePrefix + bytes, actual.end(), [](uint8_t value) { return value == kDeviceGuard; }));
    }
    void expectRecord(bool stored) const {
        EXPECT_EQ(std::memcmp(host, expected.data(), bytes), 0);
        uint32_t actual_crc = 0;
        std::memcpy(&actual_crc, host + encoded - 4, sizeof(actual_crc));
        EXPECT_EQ(actual_crc, expected_crc);
        EXPECT_TRUE(std::all_of(
            host + bytes, host + encoded - 4, [stored](uint8_t value) { return value == (stored ? 0 : 0x7b); }));
        EXPECT_TRUE(std::all_of(host_allocation.data(), host, [](uint8_t value) { return value == kHostGuard; }));
        EXPECT_TRUE(std::all_of(host + encoded, host_allocation.data() + host_allocation.bytes, [](uint8_t value) {
            return value == kHostGuard;
        }));
    }
};

using GuardedBlocks = std::vector<std::unique_ptr<GuardedBlock>>;
GuardedBlocks makeGuardedBlocks(size_t count, const std::vector<size_t>& sizes) {
    GuardedBlocks blocks;
    for (size_t i = 0; i < count; ++i)
        blocks.emplace_back(std::make_unique<GuardedBlock>(sizes[i % sizes.size()], unsigned(i + 193)));
    return blocks;
}
std::vector<CrcCopyItem> itemsFor(const GuardedBlocks& blocks, size_t count, size_t rotation = 0) {
    std::vector<CrcCopyItem> items;
    for (size_t i = 0; i < count; ++i)
        items.push_back(blocks[(i + rotation) % blocks.size()]->item());
    return items;
}
std::vector<CrcCopyItem> withoutTiles(std::vector<CrcCopyItem> items) {
    for (auto& item : items)
        item.tiles.clear();
    return items;
}

class CrcBatchTest: public ::testing::Test {
    void SetUp() override {
        if (!CrcBlockCopyBatch::available())
            GTEST_SKIP() << "CUDA13 CRC backend unavailable";
        int        count  = 0;
        const auto result = cudaGetDeviceCount(&count);
        if (result != cudaSuccess || !count)
            GTEST_SKIP() << "CUDA device unavailable";
        cudaCheck(cudaSetDevice(0));
    }
};

TEST(CrcBlockCopyLayoutTest, EncodedSizeIsCheckedAndAligned) {
    EXPECT_EQ(CrcBlockCopyBatch::encodedBytes(4080), 4096u);
    EXPECT_EQ(CrcBlockCopyBatch::encodedBytes(4096), 4112u);
    EXPECT_THROW(CrcBlockCopyBatch::encodedBytes(std::numeric_limits<size_t>::max()), std::overflow_error);
}

TEST_F(CrcBatchTest, KnownCastagnoliVector) {
    Block       block(9);
    const char* input = "123456789";
    cudaCheck(cudaMemcpy(block.device, input, 9, cudaMemcpyHostToDevice));
    CrcBlockCopyBatch copy(0, 1, 9, 1);
    ASSERT_EQ(copy.store({block.item()}), CrcCopyStatus::OK);
    EXPECT_EQ(std::memcmp(block.host, input, 9), 0);
    EXPECT_EQ(block.footer(), 0xe3069283U);
    EXPECT_EQ(block.footer(), crc32c(block.host, 9));
}

TEST_F(CrcBatchTest, MixedRaggedBatchRoundTripAndReuse) {
    Block                    a(35, 1), b(4096, 2), c(96, 3);
    CrcBlockCopyBatch        copy(0, 3, 4096, 9);
    std::vector<CrcCopyItem> items{a.item(true), b.item(true), c.item(true)};
    for (int round = 0; round < 3; ++round) {
        for (auto* block : {&a, &b, &c}) {
            for (auto& value : block->expected)
                value ^= uint8_t(17 + round);
            cudaCheck(cudaMemcpy(block->device, block->expected.data(), block->bytes, cudaMemcpyHostToDevice));
            std::memset(block->host, 0x6b, block->encoded);
            for (size_t i = 0; i < block->bytes; ++i)
                block->host[i] = block->expected[i] ^ 0xff;
            const uint32_t wrong_crc = crc32c(block->expected.data(), block->bytes) ^ 0xffffffffU;
            std::memcpy(block->host + block->encoded - 4, &wrong_crc, sizeof(wrong_crc));
        }
        ASSERT_EQ(copy.store(items), CrcCopyStatus::OK);
        for (auto* block : {&a, &b, &c}) {
            EXPECT_EQ(std::memcmp(block->host, block->expected.data(), block->bytes), 0);
            EXPECT_EQ(block->footer(), crc32c(block->host, block->bytes));
            EXPECT_TRUE(std::all_of(block->host + block->bytes, block->host + block->encoded - 4, [](uint8_t value) {
                return value == 0;
            }));
            auto poison = block->expected;
            for (auto& value : poison)
                value ^= 0xff;
            cudaCheck(cudaMemcpy(block->device, poison.data(), block->bytes, cudaMemcpyHostToDevice));
        }
        ASSERT_EQ(copy.load(items), CrcCopyStatus::OK);
        for (auto* block : {&a, &b, &c})
            EXPECT_EQ(block->readDevice(), block->expected);
        // Shrink the batch between full calls, exercising reused metadata slots.
        ASSERT_EQ(copy.store({a.item()}), CrcCopyStatus::OK);
    }
}

TEST_F(CrcBatchTest, LastItemCorruptionPreventsEveryScatter) {
    Block             a(35, 1), b(96, 2);
    CrcBlockCopyBatch copy(0, 2, 96, 6);
    auto              items = std::vector<CrcCopyItem>{a.item(true), b.item(true)};
    for (bool footer : {false, true}) {
        for (auto* block : {&a, &b})
            cudaCheck(cudaMemcpy(block->device, block->expected.data(), block->bytes, cudaMemcpyHostToDevice));
        ASSERT_EQ(copy.store(items), CrcCopyStatus::OK);
        b.host[footer ? b.encoded - 4 : b.bytes / 2] ^= 1;
        for (auto* block : {&a, &b})
            cudaCheck(cudaMemset(block->device, 0xd3, block->bytes));
        cudaCheck(cudaStreamSynchronize(nullptr));
        EXPECT_EQ(copy.load(items), CrcCopyStatus::CRC_MISMATCH);
        for (auto* block : {&a, &b})
            EXPECT_EQ(block->readDevice(), std::vector<uint8_t>(block->bytes, 0xd3));
    }
}

TEST_F(CrcBatchTest, ValidateNeedsNoDeviceTilesAndPaddingIsNotPayload) {
    Block             block(35, 7);
    CrcBlockCopyBatch copy(0, 1, 35, 3);
    ASSERT_EQ(copy.store({block.item(true)}), CrcCopyStatus::OK);
    auto item = block.item();
    item.tiles.clear();
    block.host[block.bytes] ^= 1;
    EXPECT_EQ(copy.validate({item}), CrcCopyStatus::OK);
    block.host[0] ^= 1;
    EXPECT_EQ(copy.validate({item}), CrcCopyStatus::CRC_MISMATCH);
    EXPECT_EQ(block.readDevice(), block.expected);
}

TEST_F(CrcBatchTest, SegmentedMixedRecordsInteroperateWithWholeRecordPath) {
    constexpr size_t  full = 732672, swa = 878160;
    auto              blocks = makeGuardedBlocks(32, {35, 16383, 16384, 16385, 32768, 32769, full, swa});
    CrcBlockCopyBatch segmented(0, 32, swa, 32 * 3);
    CrcBlockCopyBatch whole(0, 1, swa, 3);
    for (size_t count : {16u, 32u}) {
        SCOPED_TRACE(count);
        auto items = itemsFor(blocks, count);
        for (size_t i = 0; i < count; ++i) {
            blocks[i]->resetDevice(true);
            blocks[i]->resetHost();
        }
        ASSERT_EQ(segmented.store(items), CrcCopyStatus::OK);
        for (size_t i = 0; i < count; ++i) {
            SCOPED_TRACE(i);
            blocks[i]->expectRecord(true);  // Independent bitwise CPU CRC.
            blocks[i]->resetDevice(false);
            ASSERT_EQ(whole.load({items[i]}), CrcCopyStatus::OK);
            blocks[i]->expectDevice(true);
            ASSERT_EQ(whole.store({items[i]}), CrcCopyStatus::OK);
            blocks[i]->expectRecord(true);
            // Outside the CRC domain; neither load nor validate may require zero padding.
            std::memset(blocks[i]->host + blocks[i]->bytes, 0x7b, blocks[i]->encoded - 4 - blocks[i]->bytes);
            blocks[i]->resetDevice(false);
        }
        ASSERT_EQ(segmented.validate(withoutTiles(items)), CrcCopyStatus::OK);
        for (size_t i = 0; i < count; ++i) {
            blocks[i]->expectRecord(false);
            blocks[i]->expectDevice(false);
        }
        ASSERT_EQ(segmented.load(items), CrcCopyStatus::OK);
        for (size_t i = 0; i < count; ++i) {
            blocks[i]->expectDevice(true);
            blocks[i]->expectRecord(false);
        }
    }
}

TEST_F(CrcBatchTest, SegmentedCorruptionRejectsTheEntireBatchAndRecovers) {
    auto              blocks = makeGuardedBlocks(32, {49157, 65535, 65536, 65537, 32769});
    CrcBlockCopyBatch copy(0, 32, 65537, 32 * 3);
    const auto        items            = itemsFor(blocks, 32);
    const auto        validation_items = withoutTiles(items);
    for (auto& block : blocks)
        block->writeCpuRecord();
    ASSERT_EQ(copy.load(items), CrcCopyStatus::OK);
    for (size_t damaged_index : {0u, 1u, 16u, 31u}) {
        auto& damaged = *blocks[damaged_index];
        for (size_t offset : {size_t(0), size_t(16384 + 11), damaged.bytes - 1, damaged.encoded - 4}) {
            SCOPED_TRACE(::testing::Message() << "backing=" << damaged_index << " offset=" << offset);
            damaged.host[offset] ^= 1;
            const std::vector<uint8_t> before(damaged.host, damaged.host + damaged.encoded);
            for (auto& block : blocks)
                block->resetDevice(false);
            ASSERT_EQ(copy.validate(validation_items), CrcCopyStatus::CRC_MISMATCH);
            EXPECT_EQ(std::memcmp(damaged.host, before.data(), before.size()), 0);
            ASSERT_EQ(copy.load(items), CrcCopyStatus::CRC_MISMATCH);
            for (const auto& block : blocks)
                block->expectDevice(false);
            EXPECT_EQ(std::memcmp(damaged.host, before.data(), before.size()), 0);
            damaged.host[offset] ^= 1;
            ASSERT_EQ(copy.load(items), CrcCopyStatus::OK);
            for (const auto& block : blocks) {
                block->expectDevice(true);
                block->expectRecord(false);
            }
        }
    }
}

TEST_F(CrcBatchTest, WorkspaceReuseAcrossStrategiesAndPayloadShapes) {
    constexpr size_t  swa   = 878160;
    auto              large = makeGuardedBlocks(32, {35, 16383, 16384, 16385, 32768, 32769, 732672, swa});
    auto              tiny  = makeGuardedBlocks(32, {1, 9, 15, 16, 17, 16383, 16384});
    CrcBlockCopyBatch copy(0, 32, swa, 32 * 3);
    size_t            round = 0;
    for (size_t count : {1u, 2u, 4u, 8u, 16u, 32u, 32u, 16u, 1u, 32u}) {
        SCOPED_TRACE(::testing::Message() << "round=" << round << " count=" << count);
        auto&        blocks   = round == 6 ? tiny : large;
        const size_t rotation = (round * 5) % blocks.size();
        auto         items    = itemsFor(blocks, count, rotation);
        for (auto& block : blocks) {
            block->writeCpuRecord();
            block->resetDevice(false);
        }
        ASSERT_EQ(copy.validate(withoutTiles(items)), CrcCopyStatus::OK);
        ASSERT_EQ(copy.load(items), CrcCopyStatus::OK);
        for (size_t i = 0; i < blocks.size(); ++i) {
            const bool active = (i + blocks.size() - rotation) % blocks.size() < count;
            blocks[i]->expectDevice(active);
            blocks[i]->expectRecord(false);
        }
        for (size_t i = 0; i < count; ++i)
            blocks[(i + rotation) % blocks.size()]->resetHost();
        ASSERT_EQ(copy.store(items), CrcCopyStatus::OK);
        for (size_t i = 0; i < count; ++i)
            blocks[(i + rotation) % blocks.size()]->expectRecord(true);
        auto& last = *blocks[(count - 1 + rotation) % blocks.size()];
        last.host[last.encoded - 4] ^= 1;
        for (auto& block : blocks)
            block->resetDevice(false);
        ASSERT_EQ(copy.load(items), CrcCopyStatus::CRC_MISMATCH);
        for (const auto& block : blocks)
            block->expectDevice(false);
        last.host[last.encoded - 4] ^= 1;
        // The next call changes count/order/shape after a failed verdict.
        if (round == 9) {
            ASSERT_EQ(copy.load(items), CrcCopyStatus::OK);
            for (const auto& block : blocks)
                block->expectDevice(true);
        }
        ++round;
    }
}

TEST_F(CrcBatchTest, InvalidAndOverflowingCapacitiesAreRejectedBeforeAllocation) {
    const size_t maximum     = std::numeric_limits<size_t>::max();
    const size_t int_maximum = size_t(std::numeric_limits<int>::max());
    EXPECT_THROW(CrcBlockCopyBatch(-1, 1, 35, 1), std::invalid_argument);
    EXPECT_THROW(CrcBlockCopyBatch(0, 0, 35, 1), std::invalid_argument);
    EXPECT_THROW(CrcBlockCopyBatch(0, 1, 0, 1), std::invalid_argument);
    EXPECT_THROW(CrcBlockCopyBatch(0, 1, 35, 0), std::invalid_argument);
    EXPECT_THROW(CrcBlockCopyBatch(0, int_maximum + 1, 35, 1), std::invalid_argument);
    EXPECT_THROW(CrcBlockCopyBatch(0, 1, 35, int_maximum + 1), std::invalid_argument);
    EXPECT_THROW(CrcBlockCopyBatch(0, 1, maximum, 1), std::overflow_error);
    EXPECT_THROW(CrcBlockCopyBatch(0, 2, maximum - 19, 1), std::overflow_error);
    EXPECT_THROW(CrcBlockCopyBatch(0, int_maximum, 32769, 1), std::invalid_argument);
}

TEST_F(CrcBatchTest, MatrixCacheReplacementPreservesShapesUsedLaterInTheBatch) {
    std::vector<size_t> sizes;
    for (size_t i = 0; i < 49; ++i)
        sizes.push_back(32769 + i * 37);
    auto              blocks = makeGuardedBlocks(sizes.size(), sizes);
    CrcBlockCopyBatch copy(0, 16, sizes.back(), 16 * 3);
    auto              consecutive = [](size_t first) {
        std::vector<size_t> indices;
        for (size_t i = 0; i < 16; ++i)
            indices.push_back(first + i);
        return indices;
    };
    auto new_before_hits = [](size_t fresh, size_t first_hit) {
        std::vector<size_t> indices{fresh};
        for (size_t i = 0; i < 15; ++i)
            indices.push_back(first_hit + i);
        return indices;
    };
    // More historical shapes than cache capacity, followed by an old shape set.
    // A miss at item zero must not evict matrices needed by later cache hits.
    for (const auto& indices : {consecutive(0),
                                new_before_hits(16, 0),
                                consecutive(17),
                                new_before_hits(33, 17),
                                consecutive(0),
                                new_before_hits(48, 0)}) {
        SCOPED_TRACE(indices.front());
        std::vector<CrcCopyItem> items;
        for (size_t index : indices) {
            blocks[index]->writeCpuRecord();
            blocks[index]->resetDevice(false);
            items.push_back(blocks[index]->item());
        }
        ASSERT_EQ(copy.validate(withoutTiles(items)), CrcCopyStatus::OK);
        ASSERT_EQ(copy.load(items), CrcCopyStatus::OK);
        for (size_t index : indices) {
            blocks[index]->expectDevice(true);
            blocks[index]->expectRecord(false);
            blocks[index]->resetHost();
        }
        ASSERT_EQ(copy.store(items), CrcCopyStatus::OK);
        for (size_t index : indices)
            blocks[index]->expectRecord(true);
    }
}

TEST_F(CrcBatchTest, SharedWorkspaceSerializesConcurrentMixedShapeCalls) {
    std::vector<size_t> first_sizes, second_sizes;
    for (size_t i = 0; i < 16; ++i) {
        first_sizes.push_back(32769 + i * 19);
        second_sizes.push_back(49157 + i * 23);
    }
    std::array<GuardedBlocks, 2>      blocks{makeGuardedBlocks(16, first_sizes), makeGuardedBlocks(16, second_sizes)};
    CrcBlockCopyBatch                 copy(0, 16, second_sizes.back(), 16 * 3);
    std::array<std::exception_ptr, 2> failures{};
    std::mutex                        start_mutex;
    std::condition_variable           start_condition;
    size_t                            ready  = 0;
    bool                              start  = false;
    auto                              worker = [&](size_t index) {
        {
            std::unique_lock<std::mutex> lock(start_mutex);
            ++ready;
            start_condition.notify_all();
            start_condition.wait(lock, [&] { return start; });
        }
        try {
            cudaCheck(cudaSetDevice(0));
            const auto items            = itemsFor(blocks[index], 16);
            const auto validation_items = withoutTiles(items);
            for (int round = 0; round < 2; ++round) {
                if (copy.store(items) != CrcCopyStatus::OK)
                    throw std::runtime_error("concurrent store failed");
                if (copy.validate(validation_items) != CrcCopyStatus::OK)
                    throw std::runtime_error("concurrent validation failed");
                for (auto& block : blocks[index])
                    block->resetDevice(false);
                if (copy.load(items) != CrcCopyStatus::OK)
                    throw std::runtime_error("concurrent load failed");
            }
        } catch (...) {
            failures[index] = std::current_exception();
        }
    };
    std::thread first(worker, 0), second(worker, 1);
    {
        std::unique_lock<std::mutex> lock(start_mutex);
        start_condition.wait(lock, [&] { return ready == 2; });
        start = true;
    }
    start_condition.notify_all();
    first.join();
    second.join();
    for (size_t index = 0; index < blocks.size(); ++index) {
        SCOPED_TRACE(index);
        if (failures[index])
            std::rethrow_exception(failures[index]);
        for (const auto& block : blocks[index]) {
            block->expectRecord(true);
            block->expectDevice(true);
        }
    }
}

#ifdef RTP_LLM_CRC_INTERNAL_TESTS
TEST_F(CrcBatchTest, ProductionCombineChecksEverySegmentStatusWithoutResealingReads) {
    namespace internal     = crc_block_copy_internal;
    constexpr size_t chunk = 16384;
    // Exercise every one of the 45 FULL and 54 SWA segment statuses, including
    // active threads on both sides of the warp boundary and unequal tail sizes.
    const std::vector<size_t>          sizes{732672, 878160};
    const size_t                       stride = CrcBlockCopyBatch::encodedBytes(sizes.back()) + 32;
    std::vector<uint8_t>               record(sizes.size() * stride, 0x4e);
    std::vector<uint32_t>              whole_crc, segment_crc, matrices;
    std::vector<internal::BackingInfo> backings;
    for (size_t i = 0; i < sizes.size(); ++i) {
        auto* payload = record.data() + i * stride;
        for (size_t j = 0; j < sizes[i]; ++j)
            payload[j] = uint8_t((j * 73 + (j >> 7) * 31 + i * 97) ^ (j >> 3));
        const size_t segments = (sizes[i] + chunk - 1) / chunk;
        backings.push_back({sizes[i], segment_crc.size(), segments, matrices.size()});
        for (size_t offset = 0; offset < sizes[i]; offset += chunk)
            segment_crc.push_back(crc32c(payload + offset, std::min(chunk, sizes[i] - offset)));
        whole_crc.push_back(crc32c(payload, sizes[i]));
        const size_t encoded = CrcBlockCopyBatch::encodedBytes(sizes[i]);
        std::memset(payload + sizes[i], 0x7b, encoded - 4 - sizes[i]);
        std::memcpy(payload + encoded - 4, &whole_crc.back(), 4);
        matrices.resize(matrices.size() + 32 * segments);
        internal::makeSuffixMatrices(sizes[i], matrices.data() + backings.back().matrix_offset);
    }
    TestAllocation staging(record.size()), device_backings(backings.size() * sizeof(internal::BackingInfo)),
        device_matrices(matrices.size() * sizeof(uint32_t)), checksums(segment_crc.size() * sizeof(uint32_t)),
        statuses(segment_crc.size() * sizeof(nvcompStatus_t)), results((sizes.size() + 1) * sizeof(internal::Result));
    upload(device_backings.pointer, backings.data(), device_backings.bytes);
    upload(device_matrices.pointer, matrices.data(), device_matrices.bytes);
    upload(checksums.pointer, segment_crc.data(), checksums.bytes);

    auto run = [&](const std::vector<nvcompStatus_t>& injected, bool store, bool corrupt_footer = false) {
        SCOPED_TRACE(::testing::Message() << "store=" << store << " corrupt_footer=" << corrupt_footer);
        auto input = record;
        if (store || corrupt_footer) {
            // Reading must compare with this original footer, not overwrite it
            // with the combined checksum and accidentally compare it to itself.
            for (size_t i = 0; i < sizes.size(); ++i)
                input[i * stride + CrcBlockCopyBatch::encodedBytes(sizes[i]) - 4] ^= 0x80;
        }
        upload(staging.pointer, input.data(), input.size());
        upload(statuses.pointer, injected.data(), statuses.bytes);
        std::vector<uint8_t> result_bytes(results.bytes, 0xac);
        upload(results.pointer, result_bytes.data(), result_bytes.size());
        cudaCheck(internal::launchCombineAndFinish(staging.data(),
                                                   stride,
                                                   static_cast<const internal::BackingInfo*>(device_backings.pointer),
                                                   static_cast<const uint32_t*>(device_matrices.pointer),
                                                   static_cast<const uint32_t*>(checksums.pointer),
                                                   static_cast<const nvcompStatus_t*>(statuses.pointer),
                                                   static_cast<internal::Result*>(results.pointer),
                                                   sizes.size(),
                                                   store,
                                                   nullptr));
        cudaCheck(cudaStreamSynchronize(nullptr));
        std::vector<internal::Result> actual(sizes.size());
        cudaCheck(cudaMemcpy(
            actual.data(), results.pointer, actual.size() * sizeof(internal::Result), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(result_bytes.data(), results.pointer, result_bytes.size(), cudaMemcpyDeviceToHost));
        EXPECT_TRUE(std::all_of(result_bytes.begin() + actual.size() * sizeof(internal::Result),
                                result_bytes.end(),
                                [](uint8_t value) { return value == 0xac; }));
        std::vector<uint8_t> output(record.size());
        cudaCheck(cudaMemcpy(output.data(), staging.pointer, output.size(), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < sizes.size(); ++i) {
            SCOPED_TRACE(i);
            const auto&    backing     = backings[i];
            nvcompStatus_t first_error = nvcompSuccess;
            for (size_t segment = 0; segment < backing.segment_count; ++segment) {
                if (injected[backing.first_segment + segment] != nvcompSuccess) {
                    first_error = injected[backing.first_segment + segment];
                    break;
                }
            }
            EXPECT_EQ(actual[i].status, first_error);
            const size_t footer = i * stride + CrcBlockCopyBatch::encodedBytes(sizes[i]) - 4;
            if (!store) {
                uint32_t original_footer = 0;
                std::memcpy(&original_footer, input.data() + footer, 4);
                EXPECT_EQ(actual[i].expected, original_footer);
            }
            if (first_error == nvcompSuccess) {
                EXPECT_EQ(actual[i].actual, whole_crc[i]);
                if (store) {
                    uint32_t stored_footer = 0;
                    std::memcpy(&stored_footer, output.data() + footer, 4);
                    EXPECT_EQ(stored_footer, whole_crc[i]);
                    EXPECT_EQ(actual[i].expected, whole_crc[i]);
                } else if (corrupt_footer) {
                    EXPECT_NE(actual[i].expected, actual[i].actual);
                } else {
                    EXPECT_EQ(actual[i].expected, actual[i].actual);
                }
            }
            EXPECT_EQ(std::memcmp(output.data() + i * stride, input.data() + i * stride, sizes[i]), 0);
            if (store) {
                EXPECT_TRUE(std::all_of(output.begin() + i * stride + sizes[i],
                                        output.begin() + footer,
                                        [](uint8_t value) { return value == 0; }));
            }
            const size_t encoded_end = i * stride + CrcBlockCopyBatch::encodedBytes(sizes[i]);
            EXPECT_TRUE(std::all_of(output.begin() + encoded_end, output.begin() + (i + 1) * stride, [](uint8_t value) {
                return value == 0x4e;
            }));
        }
        if (!store) {
            EXPECT_EQ(output, input);  // Includes padding and even corrupt footers.
        }
    };

    for (bool store : {false, true}) {
        std::vector<nvcompStatus_t> injected(segment_crc.size(), nvcompSuccess);
        run(injected, store);
        for (size_t failed = 0; failed < injected.size(); ++failed) {
            SCOPED_TRACE(failed);
            injected[failed] = nvcompErrorInternal;
            run(injected, store);
            injected[failed] = nvcompSuccess;
        }
        injected[0]                             = nvcompErrorInvalidValue;
        injected[backings[0].segment_count - 1] = nvcompErrorInternal;
        injected.back()                         = nvcompErrorInternal;
        run(injected, store);  // Keep the first error independently for each backing.
        std::fill(injected.begin(), injected.end(), nvcompSuccess);
        run(injected, store);  // Good -> every bad position -> good recovery.
    }
    run(std::vector<nvcompStatus_t>(segment_crc.size(), nvcompSuccess), false, true);
}
#endif

TEST_F(CrcBatchTest, InvalidGeometryNeverWritesTargets) {
    Block                    block(35, 4);
    CrcBlockCopyBatch        copy(0, 1, 35, 3);
    const auto               valid = block.item(true);
    std::vector<CrcCopyItem> cases;
    auto                     bad = valid;
    bad.host                     = nullptr;
    cases.push_back(bad);
    bad = valid;
    --bad.capacity_bytes;
    cases.push_back(bad);
    bad               = valid;
    bad.payload_bytes = 36;
    cases.push_back(bad);
    bad = valid;
    bad.tiles.clear();
    cases.push_back(bad);
    bad                 = valid;
    bad.tiles[0].device = nullptr;
    cases.push_back(bad);
    bad                 = valid;
    bad.tiles[1].offset = 2;
    cases.push_back(bad);  // overlap
    bad                 = valid;
    bad.tiles[1].offset = 4;
    cases.push_back(bad);  // gap
    bad = valid;
    bad.tiles[2].bytes += 1;
    cases.push_back(bad);  // overrun
    bad = valid;
    bad.tiles[2].bytes -= 1;
    cases.push_back(bad);  // incomplete
    bad                = valid;
    bad.tiles[0].bytes = 0;
    cases.push_back(bad);
    for (const auto& item : cases) {
        std::memset(block.host, 0x6b, block.encoded);
        EXPECT_EQ(copy.store({item}), CrcCopyStatus::INVALID_ARGS);
        EXPECT_EQ(copy.load({item}), CrcCopyStatus::INVALID_ARGS);
        EXPECT_TRUE(std::all_of(block.host, block.host + block.encoded, [](uint8_t v) { return v == 0x6b; }));
        EXPECT_EQ(block.readDevice(), block.expected);
    }
    EXPECT_EQ(copy.store({}), CrcCopyStatus::INVALID_ARGS);
    EXPECT_EQ(copy.validate({}), CrcCopyStatus::INVALID_ARGS);
    EXPECT_EQ(copy.store({valid, valid}), CrcCopyStatus::INVALID_ARGS);
    CrcBlockCopyBatch fewer_tiles(0, 1, 35, 2);
    EXPECT_EQ(fewer_tiles.store({valid}), CrcCopyStatus::INVALID_ARGS);
    CrcBlockCopyBatch two(0, 2, 35, 6);
    EXPECT_EQ(two.store({valid, valid}), CrcCopyStatus::INVALID_ARGS);  // aliasing host backing
}
}  // namespace
}  // namespace rtp_llm
