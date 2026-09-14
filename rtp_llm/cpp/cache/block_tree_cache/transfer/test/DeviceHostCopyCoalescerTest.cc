#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyCoalescer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace rtp_llm {
namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<typename Exception, typename Function>
void expectThrows(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    } catch (...) {
        throw std::runtime_error(message + ": wrong exception type");
    }
    throw std::runtime_error(message + ": no exception");
}

CopyCoalescingTile tile(void* src, void* dst, size_t bytes, size_t layer) {
    return {src, dst, bytes, 0, 0, 0, 0, layer, 0};
}

void expectSingleton(const CopyCoalescingRegion& region, void* src, void* dst, size_t width) {
    expect(region.src == src, "singleton: wrong source");
    expect(region.dst == dst, "singleton: wrong destination");
    expect(region.width == width, "singleton: wrong width");
    expect(region.height == 1, "singleton: wrong height");
    expect(region.src_pitch == width, "singleton: wrong source pitch");
    expect(region.dst_pitch == width, "singleton: wrong destination pitch");
}

void emulateCopies(const std::vector<CopyCoalescingRegion>& regions) {
    for (const auto& region : regions) {
        const auto src = reinterpret_cast<uintptr_t>(region.src);
        const auto dst = reinterpret_cast<uintptr_t>(region.dst);
        for (size_t row = 0; row < region.height; ++row) {
            std::memcpy(reinterpret_cast<void*>(dst + row * region.dst_pitch),
                        reinterpret_cast<const void*>(src + row * region.src_pitch),
                        region.width);
        }
    }
}

void testSingletonRegion() {
    std::vector<uint8_t> src(64);
    std::vector<uint8_t> dst(64);

    const auto regions = coalesceDeviceHostTiles({tile(src.data(), dst.data(), 64, 0)});

    expect(regions.size() == 1, "singleton: expected one region");
    expectSingleton(regions[0], src.data(), dst.data(), 64);
}

void testThreeRowsCoalesce() {
    std::vector<uint8_t> src(192);
    std::vector<uint8_t> dst(320);
    const std::vector<CopyCoalescingTile> tiles = {tile(src.data(), dst.data(), 64, 0),
                                                   tile(src.data() + 64, dst.data() + 128, 64, 1),
                                                   tile(src.data() + 128, dst.data() + 256, 64, 2)};

    const auto regions = coalesceDeviceHostTiles(tiles);

    expect(regions.size() == 1, "three rows: expected one coalesced region");
    expect(regions[0].src == src.data(), "three rows: wrong source");
    expect(regions[0].dst == dst.data(), "three rows: wrong destination");
    expect(regions[0].width == 64, "three rows: wrong width");
    expect(regions[0].height == 3, "three rows: wrong height");
    expect(regions[0].src_pitch == 64, "three rows: wrong source pitch");
    expect(regions[0].dst_pitch == 128, "three rows: wrong destination pitch");
}

void testIdentityBoundariesDoNotMerge() {
    std::vector<uint8_t> src(128);
    std::vector<uint8_t> dst(128);
    const auto            first = tile(src.data(), dst.data(), 64, 0);

    auto expectBoundary = [&](CopyCoalescingTile second, const std::string& name) {
        const auto regions = coalesceDeviceHostTiles({first, second});
        expect(regions.size() == 2, name + ": crossed identity boundary");
    };

    auto second             = tile(src.data() + 64, dst.data() + 64, 64, 1);
    second.descriptor_index = 1;
    expectBoundary(second, "descriptor");

    second              = tile(src.data() + 64, dst.data() + 64, 64, 1);
    second.layout_index = 1;
    expectBoundary(second, "layout");

    second                 = tile(src.data() + 64, dst.data() + 64, 64, 1);
    second.member_group_id = 1;
    expectBoundary(second, "member group");

    second                 = tile(src.data() + 64, dst.data() + 64, 64, 1);
    second.component_index = 1;
    expectBoundary(second, "component");

    second              = tile(src.data() + 64, dst.data() + 64, 64, 1);
    second.device_index = 1;
    expectBoundary(second, "device");
}

void testInterleavedComponentsCoalesceIndependently() {
    std::vector<uint8_t> src(768);
    std::vector<uint8_t> dst(768);

    auto value0            = tile(src.data(), dst.data(), 64, 0);
    value0.component_index = 0;
    auto scale0            = tile(src.data() + 384, dst.data() + 384, 64, 0);
    scale0.component_index = 1;
    auto value1            = tile(src.data() + 64, dst.data() + 128, 64, 1);
    value1.component_index = 0;
    auto scale1            = tile(src.data() + 448, dst.data() + 512, 64, 1);
    scale1.component_index = 1;
    auto value2            = tile(src.data() + 128, dst.data() + 256, 64, 2);
    value2.component_index = 0;
    auto scale2            = tile(src.data() + 512, dst.data() + 640, 64, 2);
    scale2.component_index = 1;

    const auto regions = coalesceDeviceHostTiles({value0, scale0, value1, scale1, value2, scale2});

    expect(regions.size() == 2, "interleaved components: expected two regions");
    expect(regions[0].src == value0.src && regions[0].height == 3, "interleaved value: wrong region");
    expect(regions[0].src_pitch == 64 && regions[0].dst_pitch == 128, "interleaved value: wrong pitches");
    expect(regions[1].src == scale0.src && regions[1].height == 3, "interleaved scale: wrong region");
    expect(regions[1].src_pitch == 64 && regions[1].dst_pitch == 128, "interleaved scale: wrong pitches");
    expect(regions[0].height + regions[1].height == 6, "interleaved components: a tile was lost or duplicated");
}

void testPitchChangeAndWidthChangeSplitRuns() {
    std::vector<uint8_t> src(320);
    std::vector<uint8_t> dst(320);

    const auto pitch_regions = coalesceDeviceHostTiles({tile(src.data(), dst.data(), 64, 0),
                                                         tile(src.data() + 64, dst.data() + 64, 64, 1),
                                                         tile(src.data() + 144, dst.data() + 128, 64, 2)});
    expect(pitch_regions.size() == 2, "pitch change: expected two regions");
    expect(pitch_regions[0].height == 2, "pitch change: expected first two rows together");
    expectSingleton(pitch_regions[1], src.data() + 144, dst.data() + 128, 64);

    const auto width_regions = coalesceDeviceHostTiles(
        {tile(src.data(), dst.data(), 64, 0), tile(src.data() + 64, dst.data() + 64, 32, 1)});
    expect(width_regions.size() == 2, "width change: expected two regions");
}

void testLayerDiscontinuityAndNonpositivePitchSplitRuns() {
    std::vector<uint8_t> src(256);
    std::vector<uint8_t> dst(256);

    const auto layer_regions = coalesceDeviceHostTiles(
        {tile(src.data(), dst.data(), 64, 0), tile(src.data() + 64, dst.data() + 64, 64, 2)});
    expect(layer_regions.size() == 2, "layer discontinuity: expected two regions");

    const auto same_address_regions =
        coalesceDeviceHostTiles({tile(src.data(), dst.data(), 64, 0), tile(src.data(), dst.data() + 64, 64, 1)});
    expect(same_address_regions.size() == 2, "zero source pitch: expected two regions");

    const auto reverse_regions = coalesceDeviceHostTiles(
        {tile(src.data() + 64, dst.data(), 64, 0), tile(src.data(), dst.data() + 64, 64, 1)});
    expect(reverse_regions.size() == 2, "negative source pitch: expected two regions");

    const auto subwidth_regions = coalesceDeviceHostTiles(
        {tile(src.data(), dst.data(), 64, 0), tile(src.data() + 32, dst.data() + 64, 64, 1)});
    expect(subwidth_regions.size() == 2, "sub-width source pitch: expected two regions");

    const auto duplicate_layer_regions = coalesceDeviceHostTiles(
        {tile(src.data(), dst.data(), 64, 0), tile(src.data() + 64, dst.data() + 64, 64, 0)});
    expect(duplicate_layer_regions.size() == 2, "duplicate layer: expected two regions");

    const auto out_of_order_regions = coalesceDeviceHostTiles({tile(src.data(), dst.data(), 64, 0),
                                                                tile(src.data() + 128, dst.data() + 128, 64, 2),
                                                                tile(src.data() + 64, dst.data() + 64, 64, 1)});
    expect(out_of_order_regions.size() == 3, "out-of-order layers: expected every transition to split");
}

void testMissingIdentityPreservesSingletons() {
    std::vector<uint8_t> src(128);
    std::vector<uint8_t> dst(128);
    CopyCoalescingTile   first;
    first.src   = src.data();
    first.dst   = dst.data();
    first.bytes = 64;
    CopyCoalescingTile second = first;
    second.src                = src.data() + 64;
    second.dst                = dst.data() + 64;

    const auto regions = coalesceDeviceHostTiles({first, second});

    expect(regions.size() == 2, "missing identity: expected singleton regions");
    expectSingleton(regions[0], first.src, first.dst, first.bytes);
    expectSingleton(regions[1], second.src, second.dst, second.bytes);
}

void testZeroTilesAreOmittedAndInvalidRangesThrow() {
    const CopyCoalescingTile zero;
    expect(coalesceDeviceHostTiles({zero}).empty(), "zero tile: expected omission");

    auto       invalid_src  = tile(nullptr, reinterpret_cast<void*>(uintptr_t{1}), 1, 0);
    auto       invalid_dst  = tile(reinterpret_cast<void*>(uintptr_t{1}), nullptr, 1, 0);
    const auto maximum      = std::numeric_limits<uintptr_t>::max();
    auto       overflow_src = tile(reinterpret_cast<void*>(maximum - 31), reinterpret_cast<void*>(uintptr_t{1}), 64, 0);
    auto       overflow_dst = tile(reinterpret_cast<void*>(uintptr_t{1}), reinterpret_cast<void*>(maximum - 31), 64, 0);

    expectThrows<std::invalid_argument>([&] { coalesceDeviceHostTiles({invalid_src}); }, "null source");
    expectThrows<std::invalid_argument>([&] { coalesceDeviceHostTiles({invalid_dst}); }, "null destination");
    expectThrows<std::overflow_error>([&] { coalesceDeviceHostTiles({overflow_src}); }, "source overflow");
    expectThrows<std::overflow_error>([&] { coalesceDeviceHostTiles({overflow_dst}); }, "destination overflow");
}

void testSentinelGapsAndDirectionReversal() {
    constexpr size_t width     = 64;
    constexpr size_t src_pitch = 96;
    constexpr size_t dst_pitch = 128;
    std::vector<uint8_t> src(src_pitch * 3, 0xEE);
    std::vector<uint8_t> dst(dst_pitch * 3, 0xA5);
    for (size_t row = 0; row < 3; ++row) {
        std::fill_n(src.data() + row * src_pitch, width, static_cast<uint8_t>(0x10 + row));
    }

    std::vector<CopyCoalescingTile> forward;
    for (size_t row = 0; row < 3; ++row) {
        forward.push_back(tile(src.data() + row * src_pitch, dst.data() + row * dst_pitch, width, row));
    }
    const auto forward_regions = coalesceDeviceHostTiles(forward);
    expect(forward_regions.size() == 1, "sentinel copy: expected one region");
    emulateCopies(forward_regions);
    for (size_t row = 0; row < 3; ++row) {
        expect(std::all_of(dst.begin() + row * dst_pitch,
                           dst.begin() + row * dst_pitch + width,
                           [row](uint8_t value) { return value == static_cast<uint8_t>(0x10 + row); }),
               "sentinel copy: row contents differ");
        expect(std::all_of(dst.begin() + row * dst_pitch + width,
                           dst.begin() + (row + 1) * dst_pitch,
                           [](uint8_t value) { return value == 0xA5; }),
               "sentinel copy: destination gap was overwritten");
    }

    std::fill(src.begin(), src.end(), 0x3C);
    std::vector<CopyCoalescingTile> reverse;
    for (size_t row = 0; row < 3; ++row) {
        reverse.push_back(tile(dst.data() + row * dst_pitch, src.data() + row * src_pitch, width, row));
    }
    const auto reverse_regions = coalesceDeviceHostTiles(reverse);
    expect(reverse_regions.size() == 1, "reversed copy: expected one region");
    expect(reverse_regions[0].src_pitch == dst_pitch, "reversed copy: wrong source pitch");
    expect(reverse_regions[0].dst_pitch == src_pitch, "reversed copy: wrong destination pitch");
    emulateCopies(reverse_regions);
    for (size_t row = 0; row < 3; ++row) {
        expect(std::all_of(src.begin() + row * src_pitch,
                           src.begin() + row * src_pitch + width,
                           [row](uint8_t value) { return value == static_cast<uint8_t>(0x10 + row); }),
               "reversed copy: row contents differ");
        expect(std::all_of(src.begin() + row * src_pitch + width,
                           src.begin() + (row + 1) * src_pitch,
                           [](uint8_t value) { return value == 0x3C; }),
               "reversed copy: destination gap was overwritten");
    }
}

void testOutputOrderingIsDeterministic() {
    std::vector<uint8_t> src(512);
    std::vector<uint8_t> dst(512);
    auto                 component1_layer0 = tile(src.data() + 256, dst.data() + 256, 64, 0);
    component1_layer0.component_index      = 1;
    auto component0_layer0                 = tile(src.data(), dst.data(), 64, 0);
    auto component1_layer1                 = tile(src.data() + 320, dst.data() + 320, 64, 1);
    component1_layer1.component_index      = 1;
    auto component0_layer1                 = tile(src.data() + 64, dst.data() + 64, 64, 1);
    const std::vector<CopyCoalescingTile> tiles =
        {component1_layer0, component0_layer0, component1_layer1, component0_layer1};

    const auto first  = coalesceDeviceHostTiles(tiles);
    const auto second = coalesceDeviceHostTiles(tiles);

    expect(first.size() == 2 && second.size() == first.size(), "deterministic ordering: wrong region count");
    for (size_t index = 0; index < first.size(); ++index) {
        expect(first[index].src == second[index].src && first[index].dst == second[index].dst
                   && first[index].width == second[index].width && first[index].height == second[index].height
                   && first[index].src_pitch == second[index].src_pitch
                   && first[index].dst_pitch == second[index].dst_pitch,
               "deterministic ordering: results differ");
    }
    expect(first[0].src == component1_layer0.src, "deterministic ordering: first identity moved");
    expect(first[1].src == component0_layer0.src, "deterministic ordering: second identity moved");
}

}  // namespace
}  // namespace rtp_llm

int main() {
    try {
        rtp_llm::testSingletonRegion();
        rtp_llm::testThreeRowsCoalesce();
        rtp_llm::testIdentityBoundariesDoNotMerge();
        rtp_llm::testInterleavedComponentsCoalesceIndependently();
        rtp_llm::testPitchChangeAndWidthChangeSplitRuns();
        rtp_llm::testLayerDiscontinuityAndNonpositivePitchSplitRuns();
        rtp_llm::testMissingIdentityPreservesSingletons();
        rtp_llm::testZeroTilesAreOmittedAndInvalidRangesThrow();
        rtp_llm::testSentinelGapsAndDirectionReversal();
        rtp_llm::testOutputOrderingIsDeterministic();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "PASS: 10 tests\n";
    return 0;
}
