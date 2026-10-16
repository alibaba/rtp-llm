#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcDumpWriter.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <iterator>
#include <unistd.h>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

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
    explicit TestAllocation(size_t size, bool host = false, unsigned flags = cudaHostAllocDefault):
        bytes(size), pinned(host) {
        cudaCheck(host ? cudaHostAlloc(&pointer, bytes, flags) : cudaMalloc(&pointer, bytes));
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
    // The ungated scatter lands the host payload even when the verdict fails;
    // guards stay intact outside the tile spans.
    void expectDeviceScattered() const {
        std::vector<uint8_t> actual(device_allocation.bytes);
        cudaCheck(cudaMemcpy(actual.data(), device_allocation.pointer, actual.size(), cudaMemcpyDeviceToHost));
        EXPECT_TRUE(std::all_of(
            actual.begin(), actual.begin() + kDevicePrefix, [](uint8_t value) { return value == kDeviceGuard; }));
        EXPECT_EQ(std::memcmp(actual.data() + kDevicePrefix, host, bytes), 0);
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
    void TearDown() override {
        if (!dump_root_.empty()) {
            std::error_code ignored;
            std::filesystem::remove_all(dump_root_, ignored);
        }
    }

protected:
    std::shared_ptr<CrcDumpWriter> makeDumpWriter(size_t backup_count = CrcDumpWriter::kBackupCount) {
        const char* temp    = std::getenv("TEST_TMPDIR");
        std::string pattern = std::string(temp ? temp : "/tmp") + "/crc_batch_dump_XXXXXX";
        if (!mkdtemp(pattern.data()))
            throw std::runtime_error("cannot create CRC batch test directory");
        dump_root_ = pattern;
        return std::make_shared<CrcDumpWriter>(0, pattern, backup_count);
    }
    std::vector<std::filesystem::path> dumps() const {
        std::vector<std::filesystem::path> result;
        const auto                         rank_dir = dump_root_ / "rank_0";
        if (std::filesystem::exists(rank_dir))
            for (const auto& entry : std::filesystem::directory_iterator(rank_dir))
                if (std::filesystem::exists(entry.path() / "manifest.json"))
                    result.push_back(entry.path());
        return result;
    }
    static std::vector<uint8_t> readDump(const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }
    std::filesystem::path dump_root_;
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

TEST_F(CrcBatchTest, LastItemCorruptionScattersAllItemsAndReports) {
    Block             a(35, 1), b(96, 2);
    CrcBlockCopyBatch copy(0, 2, 96, 6, makeDumpWriter(1));
    auto              items = std::vector<CrcCopyItem>{a.item(true), b.item(true)};
    for (bool footer : {false, true}) {
        for (auto* block : {&a, &b})
            cudaCheck(cudaMemcpy(block->device, block->expected.data(), block->bytes, cudaMemcpyHostToDevice));
        ASSERT_EQ(copy.store(items), CrcCopyStatus::OK);
        b.host[footer ? b.encoded - 4 : b.bytes / 2] ^= 1;
        for (auto* block : {&a, &b})
            cudaCheck(cudaMemset(block->device, 0xd3, block->bytes));
        cudaCheck(cudaStreamSynchronize(nullptr));
        std::vector<CrcCopyFailure> failures;
        EXPECT_EQ(copy.load(items, &failures), CrcCopyStatus::CRC_MISMATCH);
        ASSERT_EQ(failures.size(), 1u);
        EXPECT_EQ(failures.front().item_index, 1u);
        EXPECT_NE(failures.front().expected, failures.front().actual);
        EXPECT_EQ(failures.front().status, CrcCopyStatus::CRC_MISMATCH);
        EXPECT_TRUE(failures.front().cpu_snapshot.empty());
        EXPECT_TRUE(failures.front().staging_snapshot.empty());
        const auto paths = dumps();
        ASSERT_EQ(paths.size(), 1u);
        EXPECT_EQ(readDump(paths[0] / "cpu.bin"), std::vector<uint8_t>(b.host, b.host + b.encoded));
        EXPECT_EQ(readDump(paths[0] / "gpu_staging.bin"), std::vector<uint8_t>(b.host, b.host + b.bytes));
        EXPECT_TRUE(failures.front().checked_footer_available);
        // The ungated scatter lands every item, corrupted or not; the caller
        // discards failures through the reported status.
        for (auto* block : {&a, &b})
            EXPECT_EQ(block->readDevice(), std::vector<uint8_t>(block->host, block->host + block->bytes));
    }
}

TEST_F(CrcBatchTest, FailureSnapshotsKeepCheckedStagingEvenWhenHostChangesAfterVerdict) {
    Block             block(35, 7);
    CrcBlockCopyBatch copy(0, 1, 35, 3, makeDumpWriter());
    auto              item = block.item(true);
    ASSERT_EQ(copy.store({item}), CrcCopyStatus::OK);
    block.host[1] ^= 1;
    const std::vector<uint8_t> checked(block.host, block.host + block.encoded);
    item.dump_metadata = [&] {
        // Metadata is evaluated after the verdict, before snapshot capture.
        block.host[2] ^= 1;
        block.host[block.encoded - 4] ^= 1;
        return CrcCopyMetadata{};
    };
    std::vector<CrcCopyFailure> failures;
    ASSERT_EQ(copy.load({item}, &failures), CrcCopyStatus::CRC_MISMATCH);
    ASSERT_EQ(failures.size(), 1u);
    const auto paths = dumps();
    ASSERT_EQ(paths.size(), 1u);
    const std::vector<uint8_t> checked_payload(checked.begin(), checked.begin() + block.bytes);
    EXPECT_EQ(readDump(paths[0] / "gpu_staging.bin"), checked_payload);
    EXPECT_EQ(readDump(paths[0] / "cpu.bin"), std::vector<uint8_t>(block.host, block.host + block.encoded));
    EXPECT_NE(std::vector<uint8_t>(block.host, block.host + block.bytes), checked_payload);
    uint32_t checked_footer;
    std::memcpy(&checked_footer, checked.data() + block.encoded - 4, 4);
    EXPECT_EQ(failures[0].expected, checked_footer);
    EXPECT_EQ(failures[0].actual, crc32c(checked.data(), block.bytes));
    ASSERT_EQ(copy.store({item}), CrcCopyStatus::OK);
    EXPECT_EQ(readDump(paths[0] / "gpu_staging.bin"), checked_payload);
}

TEST_F(CrcBatchTest, DumpFailureDoesNotChangeVerdictAndSuccessNeverBuildsMetadata) {
    Block             block(35, 7);
    CrcBlockCopyBatch copy(0, 1, 35, 3, makeDumpWriter());
    unsigned          metadata_calls = 0;
    bool              fail_metadata  = false;
    auto              item           = block.item();
    item.dump_metadata               = [&] {
        ++metadata_calls;
        if (fail_metadata)
            throw std::runtime_error("diagnostic unavailable");
        return CrcCopyMetadata{};
    };
    std::vector<CrcCopyFailure> failures;
    ASSERT_EQ(copy.store({item}, &failures), CrcCopyStatus::OK);
    ASSERT_EQ(copy.load({item}, &failures), CrcCopyStatus::OK);
    EXPECT_EQ(metadata_calls, 0u);
    EXPECT_TRUE(dumps().empty());
    block.host[1] ^= 1;
    for (bool throws : {false, true}) {
        fail_metadata = throws;
        EXPECT_EQ(copy.load({item}, &failures), CrcCopyStatus::CRC_MISMATCH);
        ASSERT_EQ(failures.size(), 1u);
        EXPECT_EQ(failures[0].status, CrcCopyStatus::CRC_MISMATCH);
        EXPECT_TRUE(failures[0].cpu_snapshot.empty());
        EXPECT_TRUE(failures[0].staging_snapshot.empty());
        EXPECT_NE(failures[0].expected, failures[0].actual);
    }
    EXPECT_EQ(metadata_calls, 2u);
    EXPECT_EQ(dumps().size(), 1u);
}

TEST_F(CrcBatchTest, SlotWriterDumpsWithoutFailureOutput) {
    Block             block(35, 7);
    CrcBlockCopyBatch copy(0, 1, 35, 3, makeDumpWriter());
    ASSERT_EQ(copy.store({block.item()}), CrcCopyStatus::OK);
    block.host[1] ^= 1;
    EXPECT_EQ(copy.load({block.item()}), CrcCopyStatus::CRC_MISMATCH);
    const auto paths = dumps();
    ASSERT_EQ(paths.size(), 1u);
    EXPECT_EQ(readDump(paths[0] / "gpu_staging.bin"), std::vector<uint8_t>(block.host, block.host + block.bytes));
}

TEST_F(CrcBatchTest, DumpFailureDoesNotSkipLaterFailures) {
    Block             a(35, 1), b(96, 2);
    CrcBlockCopyBatch copy(0, 2, 96, 6, makeDumpWriter());
    auto              items = std::vector<CrcCopyItem>{a.item(true), b.item(true)};
    ASSERT_EQ(copy.store(items), CrcCopyStatus::OK);
    items[0].dump_metadata = []() -> CrcCopyMetadata { throw std::runtime_error("dump unavailable"); };
    for (auto* block : {&a, &b}) {
        block->host[1] ^= 1;
        cudaCheck(cudaMemset(block->device, 0xd3, block->bytes));
    }
    cudaCheck(cudaStreamSynchronize(nullptr));
    std::vector<CrcCopyFailure> failures;
    EXPECT_EQ(copy.load(items, &failures), CrcCopyStatus::CRC_MISMATCH);
    ASSERT_EQ(failures.size(), 2u);
    for (const auto& failure : failures)
        EXPECT_EQ(failure.status, CrcCopyStatus::CRC_MISMATCH);
    const auto paths = dumps();
    ASSERT_EQ(paths.size(), 1u);
    EXPECT_EQ(readDump(paths[0] / "cpu.bin"), std::vector<uint8_t>(b.host, b.host + b.encoded));
    // The ungated scatter lands the (corrupted) payload; the reported status
    // tells the caller to discard it.
    for (auto* block : {&a, &b})
        EXPECT_EQ(block->readDevice(), std::vector<uint8_t>(block->host, block->host + block->bytes));
}

TEST_F(CrcBatchTest, LoadIgnoresPaddingAndReportsCorruption) {
    Block             block(35, 7);
    CrcBlockCopyBatch copy(0, 1, 35, 3);
    ASSERT_EQ(copy.store({block.item(true)}), CrcCopyStatus::OK);
    const auto                 item = block.item(true);
    const std::vector<uint8_t> untouched(block.bytes, 0xd3);
    upload(block.device, untouched.data(), untouched.size());
    block.host[block.bytes] ^= 1;
    ASSERT_EQ(copy.load({item}), CrcCopyStatus::OK);
    EXPECT_EQ(block.readDevice(), block.expected);
    upload(block.device, untouched.data(), untouched.size());
    block.host[0] ^= 1;
    std::vector<CrcCopyFailure> failures;
    ASSERT_EQ(copy.load({item}, &failures), CrcCopyStatus::CRC_MISMATCH);
    ASSERT_EQ(failures.size(), 1u);
    EXPECT_TRUE(failures.front().checked_footer_available);
    // The scatter is not gated on the verdict: the corrupted bytes land in the
    // tiles and the caller discards the item through the reported status.
    EXPECT_EQ(block.readDevice(), std::vector<uint8_t>(block.host, block.host + block.bytes));
    block.host[0] ^= 1;
    ASSERT_EQ(copy.load({item}, &failures), CrcCopyStatus::OK);
    EXPECT_TRUE(failures.empty());
    EXPECT_EQ(block.readDevice(), block.expected);
}

TEST_F(CrcBatchTest, PlainHostRecordsRoundTripAndReportCorruption) {
    Block             block(35, 7);
    CrcBlockCopyBatch copy(0, 1, 35, 3);
    // Pageable std::vector storage exercises the same load path as pinned
    // storage; the observable contract stays the same.
    std::vector<uint8_t> record(block.encoded, 0x6b);
    auto                 item = block.item(true);
    item.host                 = record.data();
    ASSERT_EQ(copy.store({item}), CrcCopyStatus::OK);
    EXPECT_EQ(std::vector<uint8_t>(record.begin(), record.begin() + block.bytes), block.expected);
    uint32_t footer = 0;
    std::memcpy(&footer, record.data() + block.encoded - 4, sizeof(footer));
    EXPECT_EQ(footer, crc32c(record.data(), block.bytes));
    record[block.bytes / 2] ^= 1;
    cudaCheck(cudaMemset(block.device, 0xd3, block.bytes));
    cudaCheck(cudaStreamSynchronize(nullptr));
    std::vector<CrcCopyFailure> failures;
    EXPECT_EQ(copy.load({item}, &failures), CrcCopyStatus::CRC_MISMATCH);
    ASSERT_EQ(failures.size(), 1u);
    EXPECT_EQ(failures.front().item_index, 0u);
    EXPECT_EQ(failures.front().status, CrcCopyStatus::CRC_MISMATCH);
    EXPECT_NE(failures.front().expected, failures.front().actual);
    // The ungated scatter lands the corrupted payload despite the verdict.
    EXPECT_EQ(block.readDevice(), std::vector<uint8_t>(record.data(), record.data() + block.bytes));
    record[block.bytes / 2] ^= 1;
    cudaCheck(cudaMemset(block.device, 0xd3, block.bytes));
    ASSERT_EQ(copy.load({item}, &failures), CrcCopyStatus::OK);
    EXPECT_TRUE(failures.empty());
    EXPECT_EQ(block.readDevice(), block.expected);
    record[block.encoded - 4] ^= 1;
    cudaCheck(cudaMemset(block.device, 0xd3, block.bytes));
    EXPECT_EQ(copy.load({item}, &failures), CrcCopyStatus::CRC_MISMATCH);
    ASSERT_EQ(failures.size(), 1u);
    EXPECT_NE(failures.front().expected, failures.front().actual);
    EXPECT_EQ(block.readDevice(), block.expected);
}

TEST_F(CrcBatchTest, MisalignedPinnedHostRecordVerifies) {
    Block             block(35, 7);
    CrcBlockCopyBatch copy(0, 1, 35, 3);
    // Pinned storage whose record starts one byte past the allocation base;
    // the declared capacity still covers the record.
    TestAllocation backing(block.encoded + 16, true);
    uint8_t*       host = backing.data() + 1;
    auto           item = block.item(true);
    item.host           = host;
    item.capacity_bytes = block.encoded + 15;
    ASSERT_EQ(copy.store({item}), CrcCopyStatus::OK);
    EXPECT_EQ(std::vector<uint8_t>(host, host + block.bytes), block.expected);
    cudaCheck(cudaMemset(block.device, 0xd3, block.bytes));
    ASSERT_EQ(copy.load({item}), CrcCopyStatus::OK);
    EXPECT_EQ(block.readDevice(), block.expected);
    host[block.bytes / 2] ^= 1;
    cudaCheck(cudaMemset(block.device, 0xd3, block.bytes));
    cudaCheck(cudaStreamSynchronize(nullptr));
    std::vector<CrcCopyFailure> failures;
    EXPECT_EQ(copy.load({item}, &failures), CrcCopyStatus::CRC_MISMATCH);
    ASSERT_EQ(failures.size(), 1u);
    EXPECT_NE(failures.front().expected, failures.front().actual);
    // The ungated scatter lands the corrupted payload despite the verdict.
    EXPECT_EQ(block.readDevice(), std::vector<uint8_t>(host, host + block.bytes));
}

TEST_F(CrcBatchTest, MixedPinnedAndPlainHostRecordsVerifyTogether) {
    Block             pinned(35, 1), plain(96, 2);
    CrcBlockCopyBatch copy(0, 2, 96, 6);
    // Mixed host allocation kinds in one batch.
    std::vector<uint8_t> plain_record(plain.encoded, 0x6b);
    auto                 items = std::vector<CrcCopyItem>{pinned.item(true), plain.item(true)};
    items[1].host              = plain_record.data();
    ASSERT_EQ(copy.store(items), CrcCopyStatus::OK);
    EXPECT_EQ(std::vector<uint8_t>(plain_record.begin(), plain_record.begin() + plain.bytes), plain.expected);
    for (auto* block : {&pinned, &plain})
        cudaCheck(cudaMemset(block->device, 0xd3, block->bytes));
    ASSERT_EQ(copy.load(items), CrcCopyStatus::OK);
    EXPECT_EQ(pinned.readDevice(), pinned.expected);
    EXPECT_EQ(plain.readDevice(), plain.expected);
    plain_record[plain.bytes / 2] ^= 1;
    for (auto* block : {&pinned, &plain})
        cudaCheck(cudaMemset(block->device, 0xd3, block->bytes));
    cudaCheck(cudaStreamSynchronize(nullptr));
    std::vector<CrcCopyFailure> failures;
    EXPECT_EQ(copy.load(items, &failures), CrcCopyStatus::CRC_MISMATCH);
    ASSERT_EQ(failures.size(), 1u);
    EXPECT_EQ(failures.front().item_index, 1u);
    EXPECT_EQ(failures.front().status, CrcCopyStatus::CRC_MISMATCH);
    EXPECT_NE(failures.front().expected, failures.front().actual);
    // The pinned item still lands its payload; the corrupted plain record is
    // scattered and discarded through the reported status.
    EXPECT_EQ(pinned.readDevice(), pinned.expected);
    EXPECT_EQ(plain.readDevice(), std::vector<uint8_t>(plain_record.data(), plain_record.data() + plain.bytes));
}

TEST_F(CrcBatchTest, LargeBatchLoadPipelinesWavesAndVerifies) {
    // At least kH2dPipelineMinBytes with multiple records: load() splits the
    // batch into record waves (6 records -> 2/2/1/1) and streams wave w+1 on
    // the copy stream while the SMs CRC wave w. 6 * ~0.8MB crosses the 4MiB
    // threshold.
    auto              blocks = makeGuardedBlocks(6, {732672, 878160});
    CrcBlockCopyBatch copy(0, 6, 878160, 6 * 3);
    auto              items = itemsFor(blocks, 6);
    for (auto& block : blocks)
        block->resetDevice(true);
    ASSERT_EQ(copy.store(items), CrcCopyStatus::OK);
    for (auto& block : blocks)
        block->expectRecord(true);
    for (auto& block : blocks)
        block->resetDevice(false);
    ASSERT_EQ(copy.load(items), CrcCopyStatus::OK);
    for (auto& block : blocks)
        block->expectDevice(true);
    // A corrupted payload in a late wave is still caught after its DMA.
    blocks[4]->host[blocks[4]->bytes / 2] ^= 1;
    for (auto& block : blocks)
        block->resetDevice(false);
    std::vector<CrcCopyFailure> failures;
    EXPECT_EQ(copy.load(items, &failures), CrcCopyStatus::CRC_MISMATCH);
    ASSERT_EQ(failures.size(), 1u);
    EXPECT_EQ(failures.front().item_index, 4u);
    // Payload corruption: the stored footer is intact, the recomputed CRC is not.
    EXPECT_EQ(failures.front().expected, blocks[4]->expected_crc);
    EXPECT_NE(failures.front().actual, blocks[4]->expected_crc);
    blocks[4]->expectDeviceScattered();
    for (size_t i : {0u, 1u, 2u, 3u, 5u})
        blocks[i]->expectDevice(true);
    blocks[4]->host[blocks[4]->bytes / 2] ^= 1;
    // A corrupted footer in the first wave fails only that item.
    blocks[0]->host[blocks[0]->encoded - 4] ^= 1;
    for (auto& block : blocks)
        block->resetDevice(false);
    EXPECT_EQ(copy.load(items, &failures), CrcCopyStatus::CRC_MISMATCH);
    ASSERT_EQ(failures.size(), 1u);
    EXPECT_EQ(failures.front().item_index, 0u);
    // Footer corruption: the corrupted footer is reported as expected, the
    // recomputed CRC of the intact payload as actual.
    EXPECT_EQ(failures.front().expected, blocks[0]->expected_crc ^ 1u);
    EXPECT_EQ(failures.front().actual, blocks[0]->expected_crc);
    for (size_t i = 0; i < 6; ++i)
        blocks[i]->expectDevice(true);
    blocks[0]->host[blocks[0]->encoded - 4] ^= 1;
    ASSERT_EQ(copy.load(items, &failures), CrcCopyStatus::OK);
    EXPECT_TRUE(failures.empty());
    for (auto& block : blocks)
        block->expectDevice(true);
}

TEST_F(CrcBatchTest, H2dWaveThresholdAndCountBoundaries) {
    // Two records just past the 4 MiB threshold split into two single-record
    // waves (waves = min(4, count)); a lone record past the threshold stays on
    // the serial path because a single wave buys no overlap.
    {
        auto              blocks = makeGuardedBlocks(2, {2195400});
        CrcBlockCopyBatch copy(0, 2, 2195400, 2 * 3);
        auto              items = itemsFor(blocks, 2);
        for (auto& block : blocks)
            block->resetDevice(true);
        ASSERT_EQ(copy.store(items), CrcCopyStatus::OK);
        for (auto& block : blocks)
            block->expectRecord(true);
        for (auto& block : blocks)
            block->resetDevice(false);
        ASSERT_EQ(copy.load(items), CrcCopyStatus::OK);
        for (auto& block : blocks)
            block->expectDevice(true);
        blocks[1]->host[blocks[1]->bytes / 2] ^= 1;
        for (auto& block : blocks)
            block->resetDevice(false);
        std::vector<CrcCopyFailure> failures;
        EXPECT_EQ(copy.load(items, &failures), CrcCopyStatus::CRC_MISMATCH);
        ASSERT_EQ(failures.size(), 1u);
        EXPECT_EQ(failures.front().item_index, 1u);
        blocks[1]->expectDeviceScattered();
        blocks[0]->expectDevice(true);
    }
    {
        auto              blocks = makeGuardedBlocks(1, {4400000});
        CrcBlockCopyBatch copy(0, 1, 4400000, 3);
        auto              items = itemsFor(blocks, 1);
        blocks[0]->resetDevice(true);
        ASSERT_EQ(copy.store(items), CrcCopyStatus::OK);
        blocks[0]->expectRecord(true);
        blocks[0]->resetDevice(false);
        ASSERT_EQ(copy.load(items), CrcCopyStatus::OK);
        blocks[0]->expectDevice(true);
    }
}

TEST_F(CrcBatchTest, ExactH2dThresholdAndThreeWaveBatchesReuseWorkspace) {
    constexpr size_t threshold = size_t{4} << 20;
    struct Case {
        std::vector<size_t> payloads;
        size_t              encoded_bytes;
    };
    // The threshold counts encoded records, including each footer and padding.
    // Reuse the workspace across wave -> serial -> wave -> three-wave -> serial.
    const std::vector<Case> cases = {
        {{2097148, 2097164}, threshold + 16},
        {{2097148, 2097132}, threshold - 16},
        {{2097148, 2097148}, threshold},
        {{1398108, 1398108, 1398108}, threshold + 32},
        {{2097148, 2097132}, threshold - 16},
    };
    CrcBlockCopyBatch copy(0, 3, 2097164, 9);
    for (const auto& test : cases) {
        SCOPED_TRACE(::testing::Message() << "encoded=" << test.encoded_bytes << " count=" << test.payloads.size());
        auto   blocks = makeGuardedBlocks(test.payloads.size(), test.payloads);
        auto   items  = itemsFor(blocks, blocks.size());
        size_t total  = 0;
        for (const auto& block : blocks)
            total += block->encoded;
        ASSERT_EQ(total, test.encoded_bytes);
        ASSERT_EQ(copy.store(items), CrcCopyStatus::OK);
        for (auto& block : blocks) {
            block->expectRecord(true);
            block->resetDevice(false);
        }
        ASSERT_EQ(copy.load(items), CrcCopyStatus::OK);
        for (const auto& block : blocks)
            block->expectDevice(true);
    }
}

TEST_F(CrcBatchTest, FailuresAcrossH2dWavesKeepIndicesAndRecover) {
    auto              blocks = makeGuardedBlocks(6, {732672, 878160});
    CrcBlockCopyBatch copy(0, 6, 878160, 18);
    const auto        items = itemsFor(blocks, blocks.size());
    ASSERT_EQ(copy.store(items), CrcCopyStatus::OK);
    // Waves contain 2/2/1/1 records. Damage payloads and a footer in three
    // different waves, including the final one, to check global result indices.
    blocks[0]->host[0] ^= 1;
    blocks[2]->host[blocks[2]->encoded - 4] ^= 1;
    blocks[5]->host[blocks[5]->bytes - 1] ^= 1;
    for (auto& block : blocks)
        block->resetDevice(false);
    std::vector<CrcCopyFailure> failures;
    ASSERT_EQ(copy.load(items, &failures), CrcCopyStatus::CRC_MISMATCH);
    ASSERT_EQ(failures.size(), 3u);
    const std::array<size_t, 3> failed_indices{0, 2, 5};
    for (size_t i = 0; i < failures.size(); ++i) {
        const auto& failure = failures[i];
        const auto& block   = *blocks[failed_indices[i]];
        uint32_t    footer  = 0;
        std::memcpy(&footer, block.host + block.encoded - 4, sizeof(footer));
        EXPECT_EQ(failure.item_index, failed_indices[i]);
        EXPECT_EQ(failure.status, CrcCopyStatus::CRC_MISMATCH);
        EXPECT_TRUE(failure.checked_footer_available);
        EXPECT_EQ(failure.expected, footer);
        EXPECT_EQ(failure.actual, crc32c(block.host, block.bytes));
        EXPECT_NE(failure.expected, failure.actual);
    }
    for (const auto& block : blocks)
        block->expectDeviceScattered();
    blocks[0]->host[0] ^= 1;
    blocks[2]->host[blocks[2]->encoded - 4] ^= 1;
    blocks[5]->host[blocks[5]->bytes - 1] ^= 1;
    // Shrink after a failed verdict, then restore the full batch. Every reused
    // flag and arrival counter must be overwritten before collecting results.
    ASSERT_EQ(copy.load({items.back()}, &failures), CrcCopyStatus::OK);
    EXPECT_TRUE(failures.empty());
    ASSERT_EQ(copy.load(items, &failures), CrcCopyStatus::OK);
    EXPECT_TRUE(failures.empty());
    for (const auto& block : blocks)
        block->expectDevice(true);
}

TEST_F(CrcBatchTest, BatchedMixedRecordsInteroperateWithSingleRecordCopies) {
    constexpr size_t  full = 732672, swa = 878160;
    auto              blocks = makeGuardedBlocks(32, {35, 16383, 16384, 16385, 32768, 32769, full, swa});
    CrcBlockCopyBatch batched(0, 32, swa, 32 * 3);
    CrcBlockCopyBatch single(0, 1, swa, 3);
    for (size_t count : {16u, 32u}) {
        SCOPED_TRACE(count);
        auto items = itemsFor(blocks, count);
        for (size_t i = 0; i < count; ++i) {
            blocks[i]->resetDevice(true);
            blocks[i]->resetHost();
        }
        ASSERT_EQ(batched.store(items), CrcCopyStatus::OK);
        for (size_t i = 0; i < count; ++i) {
            SCOPED_TRACE(i);
            blocks[i]->expectRecord(true);  // Independent bitwise CPU CRC.
            blocks[i]->resetDevice(false);
            ASSERT_EQ(single.load({items[i]}), CrcCopyStatus::OK);
            blocks[i]->expectDevice(true);
            ASSERT_EQ(single.store({items[i]}), CrcCopyStatus::OK);
            blocks[i]->expectRecord(true);
            // Padding is outside the CRC domain; load must accept nonzero padding.
            std::memset(blocks[i]->host + blocks[i]->bytes, 0x7b, blocks[i]->encoded - 4 - blocks[i]->bytes);
            blocks[i]->resetDevice(false);
        }
        ASSERT_EQ(batched.load(items), CrcCopyStatus::OK);
        for (size_t i = 0; i < count; ++i) {
            blocks[i]->expectDevice(true);
            blocks[i]->expectRecord(false);
        }
    }
}

TEST_F(CrcBatchTest, LargeBatchCorruptionStillScattersAndRecovers) {
    auto              blocks = makeGuardedBlocks(32, {49157, 65535, 65536, 65537, 32769});
    CrcBlockCopyBatch copy(0, 32, 65537, 32 * 3);
    const auto        items = itemsFor(blocks, 32);
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
            ASSERT_EQ(copy.load(items), CrcCopyStatus::CRC_MISMATCH);
            for (const auto& block : blocks)
                block->expectDeviceScattered();
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

TEST_F(CrcBatchTest, WorkspaceReuseAcrossBatchSizesAndPayloadShapes) {
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
        for (size_t i = 0; i < blocks.size(); ++i) {
            const bool active = (i + blocks.size() - rotation) % blocks.size() < count;
            if (active)
                blocks[i]->expectDeviceScattered();
            else
                blocks[i]->expectDevice(false);
        }
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
}

TEST_F(CrcBatchTest, SharedWorkspaceSerializesConcurrentMixedShapeCalls) {
    // Each 16-item batch exceeds the 4 MiB wave threshold, so the workers
    // contend on the wave-pipelined load path, not only the serial one.
    std::vector<size_t> first_sizes, second_sizes;
    for (size_t i = 0; i < 16; ++i) {
        first_sizes.push_back(300000 + i * 19);
        second_sizes.push_back(350000 + i * 23);
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
            const auto items = itemsFor(blocks[index], 16);
            for (int round = 0; round < 2; ++round) {
                if (copy.store(items) != CrcCopyStatus::OK)
                    throw std::runtime_error("concurrent store failed");
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
    EXPECT_EQ(copy.load({}), CrcCopyStatus::INVALID_ARGS);
    EXPECT_EQ(copy.store({valid, valid}), CrcCopyStatus::INVALID_ARGS);
    CrcBlockCopyBatch fewer_tiles(0, 1, 35, 2);
    EXPECT_EQ(fewer_tiles.store({valid}), CrcCopyStatus::INVALID_ARGS);
    CrcBlockCopyBatch two(0, 2, 35, 6);
    EXPECT_EQ(two.store({valid, valid}), CrcCopyStatus::INVALID_ARGS);  // aliasing host backing
}
}  // namespace
}  // namespace rtp_llm
