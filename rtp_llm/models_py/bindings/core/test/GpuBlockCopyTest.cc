#include "rtp_llm/models_py/bindings/core/GpuBlockCopy.h"

#if USING_ROCM
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPException.h>
#define GPU_CHECK C10_HIP_CHECK
#define GPU_RUNTIME(name) hip##name
#else
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#define GPU_CHECK C10_CUDA_CHECK
#define GPU_RUNTIME(name) cuda##name
#endif
#include <gtest/gtest.h>
#include <cstring>

namespace rtp_llm {
namespace {

#if USING_ROCM
using StreamGuard = c10::hip::HIPStreamGuardMasqueradingAsCUDA;
auto testStream() {
    return c10::hip::getStreamFromPoolMasqueradingAsCUDA();
}
#else
using StreamGuard = c10::cuda::CUDAStreamGuard;
auto testStream() {
    return at::cuda::getStreamFromPool();
}
#endif

torch::Tensor mapping(const std::vector<int32_t>& ids) {
    return torch::tensor(ids, torch::kInt32).reshape({-1, 3});
}

torch::Tensor pattern(int rows, int stride, int seed = 0) {
    return (torch::arange(rows * stride, torch::kInt64).reshape({rows, stride}) * 13
            + torch::arange(rows, torch::kInt64).reshape({rows, 1}) * 17 + seed)
        .to(torch::kUInt8);
}

void applyReference(torch::Tensor& expected, const torch::Tensor& ids, size_t copy_bytes) {
    auto        original = expected.clone();
    const auto* entries  = ids.data_ptr<int32_t>();
    for (int64_t i = 0; i < ids.size(0); ++i) {
        std::memcpy(expected.data_ptr<uint8_t>() + entries[3 * i + 2] * expected.stride(0),
                    original.data_ptr<uint8_t>() + entries[3 * i + 1] * original.stride(0),
                    copy_bytes);
    }
}

TEST(GpuBlockCopyTest, BeamFanoutAllLayersAndScale) {
    c10::InferenceMode             inference;
    std::vector<GpuBlockCopyPlane> planes;
    std::vector<torch::Tensor>     expected;
    std::vector<int32_t>           ids;
    for (int beam = 0; beam < 1024; ++beam) {
        ids.insert(ids.end(), {0, beam % 160, 160 + beam});
    }
    auto mappings = mapping(ids);
    for (int layer = 0; layer < 24; ++layer) {
        for (int bytes : {8192, 32}) {
            auto host = pattern(1186, bytes, layer);
            planes.push_back({host.cuda(), static_cast<size_t>(bytes)});
            applyReference(host, mappings, bytes);
            expected.push_back(host);
        }
    }
    auto op = GpuBlockCopy::create(planes);
    op->copy(mappings);
    for (size_t i = 0; i < planes.size(); ++i) {
        EXPECT_TRUE(torch::equal(planes[i].blocks.cpu(), expected[i])) << "plane " << i;
    }
}

TEST(GpuBlockCopyTest, PaddingUnalignedPointersAndByteTails) {
    for (auto sizes : {std::pair<int, int>{64, 37}, {47, 35}, {16, 3}}) {
        auto host   = pattern(9, sizes.first);
        auto device = host.cuda();
        auto op     = GpuBlockCopy::create({{device, static_cast<size_t>(sizes.second)}});
        auto ids    = mapping({0, 0, 5, 0, 1, 6, 0, 1, 7, 0, 8, 8});
        applyReference(host, ids, sizes.second);
        op->copy(ids);
        EXPECT_TRUE(torch::equal(device.cpu(), host));
    }
}

TEST(GpuBlockCopyTest, VariableCountAndImmediateHostMutation) {
    auto host   = pattern(1200, 64);
    auto device = host.cuda();
    auto op     = GpuBlockCopy::create({{device, 64}});
    for (int count : {1, 160, 1024, 0, 12, 1024, 1}) {
        std::vector<int32_t> entries;
        for (int i = 0; i < count; ++i) {
            entries.insert(entries.end(), {0, (i + count) % 160, i + 160});
        }
        auto ids = mapping(entries);
        applyReference(host, ids, 64);
        op->copy(ids);
        // The op must own H2D staging instead of borrowing the caller's buffer.
        ids.fill_(-1);
    }
    EXPECT_TRUE(torch::equal(device.cpu(), host));
}

TEST(GpuBlockCopyTest, SlotReuseAndSwitchingGpuStreams) {
    auto       host   = pattern(8, 8192);
    auto       device = host.cuda();
    auto       op     = GpuBlockCopy::create({{device, 8192}});
    const auto first  = testStream();
    const auto second = testStream();
    for (int i = 0; i < 100; ++i) {
        StreamGuard guard(i % 2 ? first : second);
        const int   src = i % 8, dst = (i + 1) % 8;
        auto        ids = mapping({0, src, dst});
        applyReference(host, ids, 8192);
        op->copy(ids);
    }
    // Destructor also has to drain non-default-stream copies.
    op.reset();
    EXPECT_TRUE(torch::equal(device.cpu(), host));
}

TEST(GpuBlockCopyTest, CopyPrecedesGpuGraphReplay) {
    auto        host     = pattern(6, 256);
    auto        device   = host.cuda();
    auto        observed = torch::empty({256}, device.options());
    auto        op       = GpuBlockCopy::create({{device, 256}});
    const auto  stream   = testStream();
    StreamGuard guard(stream);
    GPU_RUNTIME(Graph_t) graph;
    GPU_RUNTIME(GraphExec_t) exec;
    GPU_CHECK(GPU_RUNTIME(StreamBeginCapture)(stream.stream(), GPU_RUNTIME(StreamCaptureModeThreadLocal)));
    GPU_CHECK(GPU_RUNTIME(MemcpyAsync)(observed.data_ptr(),
                                       device.data_ptr<uint8_t>() + 5 * 256,
                                       256,
                                       GPU_RUNTIME(MemcpyDeviceToDevice),
                                       stream.stream()));
    GPU_CHECK(GPU_RUNTIME(StreamEndCapture)(stream.stream(), &graph));
    GPU_CHECK(GPU_RUNTIME(GraphInstantiate)(&exec, graph, nullptr, nullptr, 0));
    for (int source : {0, 3, 1, 4, 2}) {
        op->copy(mapping({0, source, 5}));
        GPU_CHECK(GPU_RUNTIME(GraphLaunch)(exec, stream.stream()));
        EXPECT_TRUE(torch::equal(observed.cpu(), host[source]));
    }
    GPU_CHECK(GPU_RUNTIME(GraphExecDestroy)(exec));
    GPU_CHECK(GPU_RUNTIME(GraphDestroy)(graph));
}

TEST(GpuBlockCopyTest, RejectInvalidMappingsBeforeEnqueue) {
    auto host   = pattern(8, 16);
    auto device = host.cuda();
    auto op     = GpuBlockCopy::create({{device, 16}});
    EXPECT_THROW(op->copy(mapping({1, 0, 1})), c10::Error);
    EXPECT_THROW(op->copy(mapping({0, -1, 1})), c10::Error);
    EXPECT_THROW(op->copy(mapping({0, 0, 8})), c10::Error);
    EXPECT_THROW(op->copy(torch::zeros({2, 3}, torch::kInt64)), c10::Error);
    EXPECT_THROW(op->copy(torch::zeros({2, 2}, torch::kInt32)), c10::Error);
    EXPECT_THROW(op->copy(torch::zeros({3, 2}, torch::kInt32).t()), c10::Error);
    EXPECT_THROW(op->copy(mapping({0, 0, 1}).cuda()), c10::Error);
    EXPECT_TRUE(torch::equal(device.cpu(), host));
}

TEST(GpuBlockCopyTest, OpaquePagesBeyondFourGiB) {
    // Real MI308 KV pools exceed 4 GiB per layer. Only touch the selected pages;
    // address arithmetic must retain the high bits for both source and target.
    constexpr int64_t page_bytes = 65536;
    constexpr int64_t high_page  = (int64_t{1} << 32) / page_bytes + 1;
    auto              device =
        torch::empty({high_page + 3, page_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    auto source = pattern(1, page_bytes, 29).squeeze(0);
    device[high_page].copy_(source);
    device[1].fill_(0);
    device[high_page + 1].fill_(0);
    auto op = GpuBlockCopy::create({{device, page_bytes}});
    // Opaque (including ROCm shuffled) layouts ignore live-token hints and
    // preserve every byte of the page, including the currently inactive tail.
    op->copy(mapping({0, high_page, 1, 0, high_page, high_page + 1}), torch::tensor({1, 7}, torch::kInt32));
    EXPECT_TRUE(torch::equal(device[1].cpu(), source));
    EXPECT_TRUE(torch::equal(device[high_page + 1].cpu(), source));
}

TEST(GpuBlockCopyTest, HndLivePrefixesAcrossLayersAndMixedParents) {
    c10::InferenceMode inference;
    constexpr int      tokens = 16, token_bytes = 256, kv_heads = 16;
    constexpr int      bytes = kv_heads * tokens * token_bytes;
    // Distinct per-mapping lengths must survive a mixed-request batch. Zero is
    // the legacy full-copy sentinel; valid==16 is also a full page.
    auto                           ids     = mapping({0, 1, 3, 0, 0, 4, 0, 2, 5, 0, 1, 6, 0, 0, 7, 0, 2, 8});
    auto                           lengths = torch::tensor({1, 4, 5, 15, 16, 0}, torch::kInt32);
    std::vector<GpuBlockCopyPlane> planes;
    std::vector<torch::Tensor>     expected;
    for (int layer = 0; layer < 3; ++layer) {
        // Include inter-page padding. Both the unused tokens and padding must
        // keep their original sentinel bytes, rather than being zeroed/copied.
        auto host      = pattern(10, bytes + 32, layer * 13);
        auto reference = host.clone();
        for (int i = 0; i < ids.size(0); ++i) {
            const int src = ids[i][1].item<int>(), dst = ids[i][2].item<int>();
            const int valid = lengths[i].item<int>() ? lengths[i].item<int>() : tokens;
            for (int head = 0; head < kv_heads; ++head) {
                const int offset = head * tokens * token_bytes;
                std::memcpy(reference.data_ptr<uint8_t>() + dst * reference.stride(0) + offset,
                            host.data_ptr<uint8_t>() + src * host.stride(0) + offset,
                            valid * token_bytes);
            }
        }
        planes.push_back({host.cuda(), bytes, tokens, token_bytes});
        expected.push_back(reference);
    }
    auto op = GpuBlockCopy::create(planes);
    op->copy(ids, lengths);
    ids.fill_(-1);
    lengths.fill_(-1);  // asynchronous transfer must not borrow either input
    for (size_t layer = 0; layer < planes.size(); ++layer) {
        EXPECT_TRUE(torch::equal(planes[layer].blocks.cpu(), expected[layer])) << "layer=" << layer;
    }
}

TEST(GpuBlockCopyTest, OpaqueLayoutIgnoresLiveLengthAndOmittedLengthsCopyAll) {
    auto host   = pattern(5, 1024);
    auto device = host.cuda();
    auto ids    = mapping({0, 0, 3, 0, 1, 4});
    auto opaque = GpuBlockCopy::create({{device, 1024}});
    opaque->copy(ids, torch::tensor({4, 5}, torch::kInt32));
    applyReference(host, ids, 1024);
    EXPECT_TRUE(torch::equal(device.cpu(), host));

    auto hnd  = GpuBlockCopy::create({{device, 1024, 16, 16}});
    auto next = mapping({0, 2, 3});
    hnd->copy(next);
    applyReference(host, next, 1024);
    EXPECT_TRUE(torch::equal(device.cpu(), host));
}

TEST(GpuBlockCopyTest, HndUnalignedHeadsAndByteTails) {
    auto host      = pattern(5, 99);
    auto device    = host.cuda();
    auto op        = GpuBlockCopy::create({{device, 96, 16, 3}});
    auto reference = host.clone();
    for (int head = 0; head < 2; ++head) {
        std::memcpy(reference.data_ptr<uint8_t>() + 4 * 99 + head * 48, host.data_ptr<uint8_t>() + head * 48, 5 * 3);
    }
    op->copy(mapping({0, 0, 4}), torch::tensor({5}, torch::kInt32));
    EXPECT_TRUE(torch::equal(device.cpu(), reference));
}

TEST(GpuBlockCopyTest, HndPrefixRefreshBeforeGraphAndNextTokenWrite) {
    c10::InferenceMode inference;
    constexpr int      heads = 2, tokens = 16, dim = 8;
    constexpr int      bytes = 2 * heads * tokens * dim * 2;
    auto               host  = (torch::arange(6 * 2 * heads * tokens * dim, torch::kFloat32).remainder(29))
                    .reshape({6, 2, heads, tokens, dim})
                    .to(torch::kBFloat16);
    auto        device      = host.cuda();
    auto        op          = GpuBlockCopy::create({{device, bytes, tokens, dim * 2}});
    auto        destination = device[5];
    const auto  stream      = testStream();
    StreamGuard guard(stream);
    auto        positions = torch::arange(tokens, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA))
                         .reshape({1, 1, tokens, 1});
    auto live_length = torch::zeros({1}, positions.options());
    auto zeros       = torch::zeros_like(destination);
    auto observed    = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    // Warm allocator/operators before capture; the captured consumer only reads
    // logically live KV. Dirty bytes past that bound may remain from a prior owner.
    observed.copy_(torch::where(positions < live_length, destination, zeros).to(torch::kFloat32).sum());
    GPU_CHECK(GPU_RUNTIME(StreamSynchronize)(stream.stream()));
    GPU_RUNTIME(Graph_t) graph;
    GPU_RUNTIME(GraphExec_t) exec;
    GPU_CHECK(GPU_RUNTIME(StreamBeginCapture)(stream.stream(), GPU_RUNTIME(StreamCaptureModeThreadLocal)));
    observed.copy_(torch::where(positions < live_length, destination, zeros).to(torch::kFloat32).sum());
    GPU_CHECK(GPU_RUNTIME(StreamEndCapture)(stream.stream(), &graph));
    GPU_CHECK(GPU_RUNTIME(GraphInstantiate)(&exec, graph, nullptr, nullptr, 0));
    int iteration = 0;
    for (int valid : {1, 4, 5, 15, 4, 1, 5}) {
        const int source = iteration++ % 3;
        auto      counts = torch::tensor({valid}, torch::kInt32);
        op->copy(mapping({0, source, 5}), counts);
        counts.fill_(-1);
        // The next forward writes the selected new token after COW, then reads
        // prefix+new token. Include K and V and every head in the comparison.
        destination.select(2, valid).fill_(37);
        live_length.fill_(valid + 1);
        GPU_CHECK(GPU_RUNTIME(GraphLaunch)(exec, stream.stream()));
        auto reference = host[source].clone();
        reference.select(2, valid).fill_(37);
        EXPECT_EQ(observed.cpu().item<float>(),
                  reference.slice(2, 0, valid + 1).to(torch::kFloat32).sum().item<float>());
    }
    GPU_CHECK(GPU_RUNTIME(GraphExecDestroy)(exec));
    GPU_CHECK(GPU_RUNTIME(GraphDestroy)(graph));
}

TEST(GpuBlockCopyTest, RejectInvalidLiveLengthsBeforeEnqueue) {
    auto host   = pattern(6, 1024);
    auto device = host.cuda();
    auto op     = GpuBlockCopy::create({{device, 1024, 16, 16}});
    auto ids    = mapping({0, 0, 5});
    for (int value : {-1, 17}) {
        EXPECT_THROW(op->copy(ids, torch::tensor({value}, torch::kInt32)), c10::Error);
    }
    EXPECT_THROW(op->copy(ids, torch::zeros({2}, torch::kInt32)), c10::Error);
    EXPECT_THROW(op->copy(ids, torch::zeros({1, 1}, torch::kInt32)), c10::Error);
    EXPECT_THROW(op->copy(ids, torch::zeros({1}, torch::kInt64)), c10::Error);
    EXPECT_THROW(op->copy(ids, torch::zeros({1}, torch::kInt32).cuda()), c10::Error);
    EXPECT_THROW(GpuBlockCopy::create({{device, 1024, 16, 0}}), c10::Error);
    EXPECT_THROW(GpuBlockCopy::create({{device, 1024, 16, 17}}), c10::Error);
    EXPECT_TRUE(torch::equal(device.cpu(), host));
}

}  // namespace
}  // namespace rtp_llm
