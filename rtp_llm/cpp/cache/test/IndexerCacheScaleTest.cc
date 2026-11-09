#include "rtp_llm/cpp/cache/IndexerCacheScale.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace rtp_llm {
namespace {
void roundTrip(c10::Device device) {
    // Include every finite UE8M0 exponent, the FP32 subnormal boundary,
    // multiple kernel pages, payload sentinels and non-payload stride padding.
    constexpr int tokens = 255, stride = tokens * 132 + 12;
    auto          original = at::full({2, stride}, 0x5a, at::TensorOptions().dtype(at::kByte));
    auto*         data     = original.data_ptr<uint8_t>();
    for (int page = 0; page < 2; ++page) {
        for (int token = 0; token < tokens; ++token) {
            const float scale = std::ldexp(1.0f, token - 127);
            std::memcpy(data + page * stride + tokens * 128 + token * 4, &scale, 4);
        }
    }
    auto      buffer = original.to(device).clone();
    BlockInfo destination;
    destination.addr           = buffer.data_ptr();
    destination.size_bytes     = buffer.nbytes();
    destination.is_cuda        = device.is_cuda();
    destination.indexer_layout = {IndexerScaleFormat::UE8M0, tokens, stride};
    convertIndexerCacheScales({{destination, {IndexerScaleFormat::FP32, tokens, stride}}});
    auto        encoded = buffer.cpu();
    const auto* bytes   = encoded.data_ptr<uint8_t>();
    for (int page = 0; page < 2; ++page) {
        for (int i = 0; i < tokens * 128; ++i)
            EXPECT_EQ(bytes[page * stride + i], 0x5a);
        for (int token = 0; token < tokens; ++token) {
            for (int group = 0; group < 4; ++group)
                EXPECT_EQ(bytes[page * stride + tokens * 128 + token * 4 + group], token);
        }
        for (int i = tokens * 132; i < stride; ++i)
            EXPECT_EQ(bytes[page * stride + i], 0x5a);
    }
    // Identical encoding must leave the entire block byte-identical.
    convertIndexerCacheScales({{destination, destination.indexer_layout}});
    EXPECT_TRUE(buffer.cpu().equal(encoded));
    destination.indexer_layout.format = IndexerScaleFormat::FP32;
    convertIndexerCacheScales({{destination, {IndexerScaleFormat::UE8M0, tokens, stride}}});
    EXPECT_TRUE(buffer.cpu().equal(original));
}
// Independent E4M3 decoder and nearest-even oracle, without tensor dtype casts.
double decodeFp8(uint8_t code) {
    const int exp = (code >> 3) & 15, fraction = code & 7;
    double value = exp == 0 ? std::ldexp(static_cast<double>(fraction), -9) : std::ldexp(1.0 + fraction / 8.0, exp - 7);
    return (code & 128) ? -value : value;
}
uint8_t roundFp8(float value) {
    const double magnitude     = std::abs(static_cast<double>(value));
    double       best_distance = magnitude;
    uint8_t      best          = 0;
    for (uint8_t code = 1; code < 127; ++code) {
        const double distance = std::abs(decodeFp8(code) - magnitude);
        if (distance < best_distance || (distance == best_distance && (code & 1) == 0)) {
            best          = code;
            best_distance = distance;
        }
    }
    return best | (std::signbit(value) ? 128 : 0);
}
void arbitraryScales(c10::Device device) {
    constexpr int tokens = 128, stride = tokens * 132 + 16, pages = 33;
    const float   scales[] = {3.0f / 448, 0.1f / 448, 1.001f / 448, 57.0f / 448, 0.03125f, 1e-4f / 448, 65504.0f / 448};
    uint8_t       expected[7][256] = {};
    float         target_scales[7];
    uint32_t      expected_scale_bits[7];
    for (int s = 0; s < 7; ++s) {
        int          exponent;
        const double mantissa  = std::frexp(static_cast<double>(scales[s]), &exponent);
        const int    ceiling   = mantissa == 0.5 ? exponent - 1 : exponent;
        target_scales[s]       = std::ldexp(1.0f, ceiling);
        expected_scale_bits[s] = static_cast<uint32_t>(ceiling + 127) * 0x01010101u;
        for (int sign : {0, 128})
            for (int code = 0; code < 127; ++code) {
                const float value        = static_cast<float>(decodeFp8(sign | code)) * (scales[s] / target_scales[s]);
                expected[s][sign | code] = roundFp8(value);
            }
    }
    auto  original = at::full({pages, stride}, 0x5a, at::TensorOptions().dtype(at::kByte));
    auto* bytes    = original.data_ptr<uint8_t>();
    for (int page = 0; page < pages; ++page)
        for (int row = 0; row < tokens; ++row) {
            const int global_row = page * tokens + row;
            for (int column = 0; column < 128; ++column) {
                const int n                               = global_row + column;
                bytes[page * stride + row * 128 + column] = (n % 127) | ((n & 1) ? 128 : 0);
            }
            std::memcpy(bytes + page * stride + tokens * 128 + row * 4, &scales[global_row % 7], 4);
        }
    auto      cache = original.to(device).clone();
    BlockInfo target;
    target.addr           = cache.data_ptr();
    target.size_bytes     = cache.nbytes();
    target.is_cuda        = device.is_cuda();
    target.indexer_layout = {IndexerScaleFormat::FP32, tokens, stride};
    convertIndexerCacheScales({{target, target.indexer_layout}});
    EXPECT_TRUE(cache.cpu().equal(original));
    target.indexer_layout.format = IndexerScaleFormat::UE8M0;
    convertIndexerCacheScales({{target, {IndexerScaleFormat::FP32, tokens, stride}}});
    auto        converted = cache.cpu();
    const auto* result    = converted.data_ptr<uint8_t>();
    for (int page = 0; page < pages; ++page) {
        for (int row = 0; row < tokens; ++row) {
            const int scale_index = (page * tokens + row) % 7;
            for (int col = 0; col < 128; ++col) {
                const int offset = page * stride + row * 128 + col;
                ASSERT_EQ(result[offset], expected[scale_index][bytes[offset]])
                    << "row=" << page * tokens + row << " col=" << col;
            }
            uint32_t packed;
            std::memcpy(&packed, result + page * stride + tokens * 128 + row * 4, 4);
            ASSERT_EQ(packed, expected_scale_bits[scale_index]);
        }
        for (int tail = tokens * 132; tail < stride; ++tail)
            EXPECT_EQ(result[page * stride + tail], 0x5a);
    }
}
}  // namespace

TEST(IndexerCacheScaleTest, CPURoundTripAllExponents) {
    roundTrip(c10::Device(c10::kCPU));
}
TEST(IndexerCacheScaleTest, GPURoundTripAllExponents) {
    if (at::getNumGPUs() == 0)
        GTEST_SKIP() << "requires one CUDA/ROCm device";
    roundTrip(c10::Device(c10::kCUDA, 0));
}
TEST(IndexerCacheScaleTest, CPUArbitraryScalesRequantizeWithBoundedBatches) {
    arbitraryScales(c10::Device(c10::kCPU));
}
TEST(IndexerCacheScaleTest, GPUArbitraryScalesRequantizeWithBoundedBatches) {
    if (at::getNumGPUs() == 0)
        GTEST_SKIP() << "requires one CUDA/ROCm device";
    arbitraryScales(c10::Device(c10::kCUDA, 0));
}
TEST(IndexerCacheScaleTest, RejectMissingUnknownAndMisalignedMetadata) {
    alignas(4) uint8_t data[528] = {};
    BlockInfo          target;
    target.addr           = data;
    target.size_bytes     = sizeof(data);
    target.indexer_layout = {IndexerScaleFormat::FP32, 4, 528};
    EXPECT_NO_THROW(convertIndexerCacheScales({{target, {}}}));
    EXPECT_THROW(convertIndexerCacheScales({{target, {IndexerScaleFormat::NONE, 4, 528}}}), std::runtime_error);
    EXPECT_THROW(convertIndexerCacheScales({{target, {static_cast<IndexerScaleFormat>(99), 4, 528}}}),
                 std::runtime_error);
    EXPECT_THROW(convertIndexerCacheScales({{target, {IndexerScaleFormat::UE8M0, 2, 528}}}), std::runtime_error);
    target.size_bytes -= 1;
    EXPECT_THROW(convertIndexerCacheScales({{target, {IndexerScaleFormat::UE8M0, 4, 528}}}), std::runtime_error);
    target.size_bytes += 1;
    target.addr = data + 1;
    EXPECT_THROW(convertIndexerCacheScales({{target, {IndexerScaleFormat::UE8M0, 4, 528}}}), std::runtime_error);
}
TEST(IndexerCacheScaleTest, LegacyWireDefaultsToFp32AndTransferCopyPreservesSource) {
    auto  tensor = at::full({528}, 0x5a, at::TensorOptions().dtype(at::kByte));
    auto* scales = reinterpret_cast<uint32_t*>(tensor.data_ptr<uint8_t>() + 512);
    for (int i = 0; i < 4; ++i)
        scales[i] = 0x7a7a7a7a;
    BlockInfo source;
    source.addr           = tensor.data_ptr();
    source.size_bytes     = tensor.nbytes();
    source.indexer_layout = {IndexerScaleFormat::UE8M0, 4, 528};
    const auto copy       = legacyIndexerTransfer(source, 1024);
    EXPECT_NE(copy.block.addr, source.addr);
    EXPECT_EQ(copy.allocation_bytes, 1024);
    EXPECT_EQ(copy.block.indexer_layout.format, IndexerScaleFormat::FP32);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(scales[i], 0x7a7a7a7a);
        EXPECT_EQ(reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(copy.block.addr) + 512)[i], 0x3d000000);
    }
    EXPECT_EQ(std::memcmp(copy.block.addr, source.addr, 512), 0);
    auto receiver = source;
    receiver.addr = copy.block.addr;
    convertIndexerCacheScales({{receiver, {}}});
    EXPECT_EQ(std::memcmp(copy.block.addr, source.addr, 528), 0);
    const auto same = legacyIndexerTransfer(copy.block);
    EXPECT_EQ(same.block.addr, copy.block.addr);
    EXPECT_FALSE(same.owner);
}

TEST(IndexerCacheScaleTest, GPULegacyTransferCopyPreservesNativeCache) {
    if (at::getNumGPUs() == 0)
        GTEST_SKIP() << "requires one CUDA/ROCm device";
    auto  host   = at::full({528}, 0x5a, at::TensorOptions().dtype(at::kByte));
    auto* scales = reinterpret_cast<uint32_t*>(host.data_ptr<uint8_t>() + 512);
    for (int i = 0; i < 4; ++i)
        scales[i] = 0x7a7a7a7a;
    auto      tensor = host.to(c10::Device(c10::kCUDA, 0));
    BlockInfo source;
    source.is_cuda        = true;
    source.addr           = tensor.data_ptr();
    source.size_bytes     = 528;
    source.indexer_layout = {IndexerScaleFormat::UE8M0, 4, 528};
    const auto copy       = legacyIndexerTransfer(source, 1024);
    auto       result     = at::from_blob(copy.block.addr, {528}, tensor.options()).cpu();
    EXPECT_TRUE(tensor.cpu().equal(host));
    EXPECT_EQ(std::memcmp(result.data_ptr(), host.data_ptr(), 512), 0);
    const auto* output_scales = reinterpret_cast<const uint32_t*>(result.data_ptr<uint8_t>() + 512);
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(output_scales[i], 0x3d000000);
}

TEST(IndexerCacheScaleTest, OrdinaryCacheDoesNotInspectMemoryOrDevice) {
    BlockInfo ordinary;
    ordinary.is_cuda = true;
    ordinary.addr    = nullptr;
    EXPECT_NO_THROW(convertIndexerCacheScales({{ordinary, {}}}));
    ordinary.addr           = reinterpret_cast<void*>(4);  // Deliberately unreadable, aligned address.
    ordinary.size_bytes     = 528;
    ordinary.indexer_layout = {IndexerScaleFormat::UE8M0, 4, 528};
    EXPECT_NO_THROW(convertIndexerCacheScales({{ordinary, ordinary.indexer_layout}}));
}
TEST(IndexerCacheScaleTest, DescriptorPreservesCompressedKernelPageGeometry) {
    KVCacheSpecDesc desc;
    desc.tag                         = "indexer_kv";
    desc.cache_type                  = KVCacheSpecType::OpaqueKV;
    desc.entry_elems                 = 132;
    desc.entry_dtype                 = DataType::TYPE_UINT8;
    desc.entry_count_mode            = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
    desc.compression_ratio           = 4;
    desc.block_stride_bytes_override = 4352;
    desc.indexer_scale_format        = IndexerScaleFormat::FP32;
    SpecBuildContext context;
    context.seq_size_per_block      = 256;
    context.kernel_tokens_per_block = 128;
    auto       spec                 = SpecBuilder::build(desc, context);
    const auto layout               = spec->indexerCacheLayout();
    EXPECT_EQ(layout.page_tokens, 32);
    EXPECT_EQ(layout.page_stride, 4352);
    EXPECT_EQ(spec->k_block_size_bytes(), 8704);
    auto copy = spec->clone();
    EXPECT_EQ(copy->indexerCacheLayout().format, IndexerScaleFormat::FP32);
    copy->indexer_scale_format = IndexerScaleFormat::UE8M0;
    EXPECT_NE(copy->fingerprint(), spec->fingerprint());
    auto ordinary                   = spec->clone();
    ordinary->indexer_scale_format  = IndexerScaleFormat::NONE;
    const auto ordinary_fingerprint = ordinary->fingerprint();
    EXPECT_EQ(ordinary_fingerprint.find("indexer_scale_format"), std::string::npos);
    EXPECT_EQ(ordinary_fingerprint, spec->fingerprint().substr(0, spec->fingerprint().find(";indexer_scale_format=")));
    desc.entry_count_mode     = OpaqueBlockEntryCountMode::EXPLICIT;
    desc.explicit_entry_count = 32;
    EXPECT_ANY_THROW(SpecBuilder::build(desc, context));
    desc.entry_count_mode = OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED;
    desc.entry_elems      = 264;
    EXPECT_ANY_THROW(SpecBuilder::build(desc, context));
    desc.entry_elems          = 68;
    desc.indexer_scale_format = IndexerScaleFormat::NONE;
    EXPECT_NO_THROW(SpecBuilder::build(desc, context));
}
}  // namespace rtp_llm
