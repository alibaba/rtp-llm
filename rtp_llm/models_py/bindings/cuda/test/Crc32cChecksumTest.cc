// Kernel-level correctness for crc32cBatchedKernel, exercised through the
// production store() path (tables + kernel + fold + seal). Every checksum is
// checked three ways: the sealed footer, an independent byte-wise CRC32C
// reference, and nvCOMP's own batched CRC32C on the identical device bytes.
#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

#include <cuda_runtime.h>
#include <nvcomp/crc32.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <vector>

namespace rtp_llm {
namespace {

void cudaCheck(cudaError_t error) {
    ASSERT_EQ(error, cudaSuccess) << cudaGetErrorString(error);
}

void checkNvcomp(nvcompStatus_t error) {
    ASSERT_EQ(error, nvcompSuccess);
}

// Independent reflected CRC-32C (Castagnoli): poly 0x1EDC6F41, init/xorout
// 0xFFFFFFFF, ref_in/ref_out true. One byte per step: correct by construction.
uint32_t crc32cReference(const uint8_t* data, size_t bytes) {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> built{};
        for (uint32_t index = 0; index < 256; ++index) {
            uint32_t state = index;
            for (int bit = 0; bit < 8; ++bit)
                state = (state & 1u) ? (0x82F63B78u ^ (state >> 1)) : (state >> 1);
            built[index] = state;
        }
        return built;
    }();
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t index = 0; index < bytes; ++index)
        crc = table[(crc ^ data[index]) & 0xFFu] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

struct DeviceBuffer {
    uint8_t* pointer{nullptr};

    explicit DeviceBuffer(size_t bytes) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&pointer), bytes));
    }
    ~DeviceBuffer() {
        if (pointer)
            cudaFree(pointer);
    }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct PinnedBuffer {
    uint8_t* pointer{nullptr};

    explicit PinnedBuffer(size_t bytes) {
        cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&pointer), bytes, cudaHostAllocDefault));
    }
    ~PinnedBuffer() {
        if (pointer)
            cudaFreeHost(pointer);
    }
    PinnedBuffer(const PinnedBuffer&)            = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

// Byte patterns that stress different data-loop paths (all-zero tables hits,
// table-index variety, unaligned uint4 lanes, pseudo-random mixing).
enum class Pattern {
    Zeros = 0,
    AllOnes,
    Counter,
    Xorshift,
    AscendingMod251,
};

void fillPattern(uint8_t* data, size_t bytes, Pattern pattern, unsigned seed) {
    switch (pattern) {
        case Pattern::Zeros:
            std::memset(data, 0, bytes);
            break;
        case Pattern::AllOnes:
            std::memset(data, 0xFF, bytes);
            break;
        case Pattern::Counter:
            for (size_t index = 0; index < bytes; ++index)
                data[index] = static_cast<uint8_t>(index + seed);
            break;
        case Pattern::Xorshift: {
            uint32_t state = seed * 2654435761u + 12345u;
            for (size_t index = 0; index < bytes; ++index) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                data[index] = static_cast<uint8_t>(state >> 24);
            }
            break;
        }
        case Pattern::AscendingMod251:
            for (size_t index = 0; index < bytes; ++index)
                data[index] = static_cast<uint8_t>((index * (seed + 1)) % 251);
            break;
    }
}

const char* patternName(Pattern pattern) {
    switch (pattern) {
        case Pattern::Zeros:
            return "zeros";
        case Pattern::AllOnes:
            return "all_ones";
        case Pattern::Counter:
            return "counter";
        case Pattern::Xorshift:
            return "xorshift";
        case Pattern::AscendingMod251:
            return "ascending_mod_251";
    }
    return "?";
}

// One pinned host record plus one device tile per message; the record is
// encoded-capacity and the footer is read back after store().
struct MessageFixture {
    struct Message {
        PinnedBuffer host;
        DeviceBuffer device;
        const size_t length;
        const size_t encoded;

        explicit Message(size_t payload, Pattern pattern, unsigned seed):
            host(CrcBlockCopyBatch::encodedBytes(payload)),
            device(payload),
            length(payload),
            encoded(CrcBlockCopyBatch::encodedBytes(payload)) {
            fillPattern(host.pointer, length, pattern, seed);
            cudaCheck(cudaMemcpy(device.pointer, host.pointer, length, cudaMemcpyHostToDevice));
        }

        CrcCopyItem item() const {
            CrcCopyItem value{host.pointer, length, encoded, {}};
            value.tiles.push_back({device.pointer, 0, length});
            return value;
        }

        uint32_t sealedFooter() const {
            uint32_t value;
            std::memcpy(&value, host.pointer + encoded - 4, sizeof(value));
            return value;
        }
    };

    std::vector<std::unique_ptr<Message>> messages;
    std::vector<CrcCopyItem>              items;

    MessageFixture(const std::vector<size_t>& lengths, Pattern pattern, unsigned seed) {
        messages.reserve(lengths.size());
        for (size_t length : lengths)
            messages.push_back(std::make_unique<Message>(length, pattern, seed++));
        items.reserve(messages.size());
        for (const auto& message : messages)
            items.push_back(message->item());
    }

    // nvCOMP's checksums for the identical device bytes.
    std::vector<uint32_t> nvcompChecksums() const {
        const size_t        count = messages.size();
        std::vector<void*>  hosts(count);
        std::vector<size_t> lengths(count);
        size_t              largest = 0;
        for (size_t index = 0; index < count; ++index) {
            hosts[index]   = messages[index]->device.pointer;
            lengths[index] = messages[index]->length;
            largest        = std::max(largest, lengths[index]);
        }
        DeviceBuffer inputs(count * sizeof(void*));
        DeviceBuffer device_lengths(count * sizeof(size_t));
        DeviceBuffer checksums(count * sizeof(uint32_t));
        DeviceBuffer statuses(count * sizeof(nvcompStatus_t));
        cudaCheck(cudaMemcpy(inputs.pointer, hosts.data(), count * sizeof(void*), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(device_lengths.pointer, lengths.data(), count * sizeof(size_t), cudaMemcpyHostToDevice));

        nvcompBatchedCRC32Opts_t opts{};
        opts.spec = nvcompCRC32_C;
        checkNvcomp(nvcompBatchedCRC32GetHeuristicConf(nullptr, count, &opts.kernel_conf, largest, nullptr));
        checkNvcomp(nvcompBatchedCRC32Async(reinterpret_cast<const void* const*>(inputs.pointer),
                                            reinterpret_cast<const size_t*>(device_lengths.pointer),
                                            count,
                                            reinterpret_cast<uint32_t*>(checksums.pointer),
                                            opts,
                                            nvcompCRC32OnlySegment,
                                            reinterpret_cast<nvcompStatus_t*>(statuses.pointer),
                                            nullptr));
        std::vector<uint32_t>       result(count);
        std::vector<nvcompStatus_t> returned(count);
        cudaCheck(cudaMemcpy(result.data(), checksums.pointer, count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        cudaCheck(
            cudaMemcpy(returned.data(), statuses.pointer, count * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost));
        for (nvcompStatus_t status : returned)
            EXPECT_EQ(status, nvcompSuccess);
        return result;
    }
};

class Crc32cChecksumTest: public ::testing::Test {
protected:
    void SetUp() override {
        cudaCheck(cudaSetDevice(0));
        cudaCheck(cudaFree(nullptr));
    }

    // Runs one (lengths, pattern) batch through the shared workspace and checks
    // the three-way agreement. Reusing one workspace across batches also
    // exercises fold-table rebuilds and arrived resets between launches.
    void checkBatch(const std::vector<size_t>& lengths, Pattern pattern, unsigned seed) {
        SCOPED_TRACE(::testing::Message() << "pattern=" << patternName(pattern) << " seed=" << seed);
        MessageFixture fixture(lengths, pattern, seed);
        const auto     nvcomp = fixture.nvcompChecksums();
        ASSERT_EQ(copy_.store(fixture.items), CrcCopyStatus::OK);
        for (size_t index = 0; index < lengths.size(); ++index) {
            const auto& message = *fixture.messages[index];
            SCOPED_TRACE(::testing::Message() << "length=" << message.length << " index=" << index);
            const uint32_t reference = crc32cReference(message.host.pointer, message.length);
            EXPECT_EQ(message.sealedFooter(), reference);
            EXPECT_EQ(nvcomp[index], reference);
        }
    }

    // Sized once for the largest matrix below: 64 messages, SWA payload, one
    // tile per message.
    CrcBlockCopyBatch copy_{0, 64, 1235040, 64};
};

// Length matrix. One batch shares crc_blocks = min(32, ceil(largest/16KiB)),
// so each group fixes a different block geometry while lengths sweep the
// per-message slice/ragged structure:
//   single_block: largest 16384 -> 1 block
//   two_blocks:   largest 32768 -> 2 blocks, ragged 1/2, empty trailing blocks
//                 for tiny lengths
//   four_blocks:  largest 65536 -> 4 blocks with empty trailing blocks
//   max_blocks:   largest 1235040 -> 32 blocks with per-message pow2 slice
//                 folds, ragged remainders, and the production FULL/SWA shapes
TEST_F(Crc32cChecksumTest, ChecksumsMatchReferenceAndNvcomp) {
    const std::vector<size_t> single_block = {1,    2,    3,    7,    8,    9,    15,   16,   17,   31,    32,   33,
                                              63,   64,   65,   127,  128,  129,  255,  256,  257,  511,   512,  513,
                                              1023, 1024, 1025, 4095, 4096, 4097, 8191, 8192, 8193, 16383, 16384};
    const std::vector<size_t> two_blocks   = {1, 2, 3, 16, 17, 16384, 16385, 16386, 24576, 24577, 32767, 32768};
    const std::vector<size_t> four_blocks  = {35, 49152, 49153, 65535, 65536};
    const std::vector<size_t> max_blocks   = {49152,
                                              49153,
                                              65536,
                                              131072,
                                              262144,
                                              262145,
                                              262144 + 16383,
                                              262144 + 16384,
                                              524287,
                                              524288,
                                              524289,
                                              510912,
                                              511000,
                                              732672,
                                              878160,
                                              1235040};

    const std::vector<Pattern> patterns = {
        Pattern::Zeros, Pattern::AllOnes, Pattern::Counter, Pattern::Xorshift, Pattern::AscendingMod251};

    unsigned seed = 1;
    for (Pattern pattern : patterns) {
        checkBatch(single_block, pattern, seed++);
        checkBatch(two_blocks, pattern, seed++);
        checkBatch(four_blocks, pattern, seed++);
        checkBatch(max_blocks, pattern, seed++);
    }
}

// Standard CRC-32C vectors anchor the whole chain to the specification.
TEST_F(Crc32cChecksumTest, KnownVectors) {
    struct Vector {
        const char* bytes;
        size_t      length;
        uint32_t    expected;
    };
    const std::vector<Vector> vectors = {
        {"123456789", 9, 0xE3069283u},
        {"The quick brown fox jumps over the lazy dog", 43, 0x22620404u},
    };
    for (const auto& vector : vectors) {
        SCOPED_TRACE(vector.bytes);
        MessageFixture fixture({vector.length}, Pattern::Xorshift, 7);
        std::memcpy(fixture.messages[0]->host.pointer, vector.bytes, vector.length);
        cudaCheck(cudaMemcpy(fixture.messages[0]->device.pointer, vector.bytes, vector.length, cudaMemcpyHostToDevice));
        ASSERT_EQ(copy_.store(fixture.items), CrcCopyStatus::OK);
        EXPECT_EQ(fixture.messages[0]->sealedFooter(), vector.expected);
        const auto nvcomp = fixture.nvcompChecksums();
        EXPECT_EQ(nvcomp[0], vector.expected);
    }
}

TEST_F(Crc32cChecksumTest, Rfc3720ThirtyTwoByteVectors) {
    struct Vector {
        std::vector<uint8_t> bytes;
        uint32_t             expected;
    };
    std::vector<Vector> vectors = {
        {std::vector<uint8_t>(32, 0x00), 0x8A9136AAu},
        {std::vector<uint8_t>(32, 0xFF), 0x62A8AB43u},
    };
    {
        std::vector<uint8_t> ascending(32);
        std::iota(ascending.begin(), ascending.end(), 0);
        vectors.push_back({std::move(ascending), 0x46DD794Eu});
    }
    {
        std::vector<uint8_t> descending(32);
        std::iota(descending.begin(), descending.end(), 0);
        std::reverse(descending.begin(), descending.end());
        vectors.push_back({std::move(descending), 0x113FDB5Cu});
    }
    {
        std::vector<uint8_t> spaces(32, 0x20);
        vectors.push_back({std::move(spaces), 0x6DFC1D19u});
    }

    std::vector<size_t> lengths;
    for (const auto& vector : vectors)
        lengths.push_back(vector.bytes.size());
    MessageFixture fixture(lengths, Pattern::Zeros, 3);
    for (size_t index = 0; index < vectors.size(); ++index) {
        std::memcpy(fixture.messages[index]->host.pointer, vectors[index].bytes.data(), lengths[index]);
        cudaCheck(cudaMemcpy(fixture.messages[index]->device.pointer,
                             vectors[index].bytes.data(),
                             lengths[index],
                             cudaMemcpyHostToDevice));
    }
    ASSERT_EQ(copy_.store(fixture.items), CrcCopyStatus::OK);
    const auto nvcomp = fixture.nvcompChecksums();
    for (size_t index = 0; index < vectors.size(); ++index) {
        SCOPED_TRACE(index);
        EXPECT_EQ(fixture.messages[index]->sealedFooter(), vectors[index].expected);
        EXPECT_EQ(nvcomp[index], vectors[index].expected);
        EXPECT_EQ(crc32cReference(vectors[index].bytes.data(), lengths[index]), vectors[index].expected);
    }
}

// The seal must depend on every payload byte: for the 97-byte message every
// single bit flip must fail the verdict; 16395 crosses the 16 KiB block
// boundary, so every byte position (alternating bit masks) is checked there.
TEST_F(Crc32cChecksumTest, SingleBitAndByteFlipsAreDetectedOnLoad) {
    {
        MessageFixture fixture({97}, Pattern::Xorshift, 11);
        auto&          message = *fixture.messages[0];
        ASSERT_EQ(copy_.store(fixture.items), CrcCopyStatus::OK);
        for (size_t byte = 0; byte < message.length; ++byte) {
            for (int bit = 0; bit < 8; ++bit) {
                const uint8_t mask = uint8_t(1u << bit);
                message.host.pointer[byte] ^= mask;
                EXPECT_EQ(copy_.load(fixture.items), CrcCopyStatus::CRC_MISMATCH)
                    << "bit flip at byte " << byte << " bit " << bit << " was not detected";
                message.host.pointer[byte] ^= mask;
            }
        }
        EXPECT_EQ(copy_.load(fixture.items), CrcCopyStatus::OK);
    }
    {
        MessageFixture fixture({16395}, Pattern::Xorshift, 12);
        auto&          message = *fixture.messages[0];
        ASSERT_EQ(copy_.store(fixture.items), CrcCopyStatus::OK);
        for (size_t byte = 0; byte < message.length; ++byte) {
            const uint8_t mask = uint8_t(1u << (byte & 7u));
            message.host.pointer[byte] ^= mask;
            EXPECT_EQ(copy_.load(fixture.items), CrcCopyStatus::CRC_MISMATCH)
                << "flip at byte " << byte << " was not detected";
            message.host.pointer[byte] ^= mask;
        }
        EXPECT_EQ(copy_.load(fixture.items), CrcCopyStatus::OK);
    }
}

// Byte positions around the 32-block message's slice boundaries (slice = 16384
// after the pow2 fold, full = 31 blocks, ragged start 507904) and the data
// loop's alignment peel points: every flip must fail.
TEST_F(Crc32cChecksumTest, BoundaryByteFlipsAreDetected) {
    const size_t   length = 510912;
    MessageFixture fixture({length}, Pattern::Xorshift, 13);
    auto&          message = *fixture.messages[0];
    ASSERT_EQ(copy_.store(fixture.items), CrcCopyStatus::OK);
    for (size_t byte : {
             size_t(0),      size_t(1),      size_t(7),      size_t(8),      size_t(15),     size_t(16),
             size_t(17),     size_t(31),     size_t(4095),   size_t(4096),   size_t(4097),   size_t(16383),
             size_t(16384),  size_t(16385),  size_t(32767),  size_t(32768),  size_t(507903), size_t(507904),
             size_t(507905), size_t(510895), size_t(510896), size_t(510908), size_t(510911),
         }) {
        SCOPED_TRACE(::testing::Message() << "byte=" << byte);
        message.host.pointer[byte] ^= 0x5A;
        EXPECT_EQ(copy_.load(fixture.items), CrcCopyStatus::CRC_MISMATCH);
        message.host.pointer[byte] ^= 0x5A;
    }
    EXPECT_EQ(copy_.load(fixture.items), CrcCopyStatus::OK);
}

}  // namespace
}  // namespace rtp_llm
