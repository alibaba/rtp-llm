#include <iostream>
#include <filesystem>
#include <unistd.h>
#include <type_traits>
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
static_assert(std::is_constructible_v<rtp_llm::BlockTransferDispatcher, std::nullptr_t>);
#include <stdexcept>
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/offline_replay/MetadataReplay.h"
using namespace rtp_llm::benchmark;
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
MetadataRequest request(int id, std::vector<int64_t> keys) {
    return {id, static_cast<uint64_t>(id), static_cast<int>(keys.size() * 256 + 128), std::move(keys)};
}
int main() {
    MetadataReplayConfig c;
    c.device_blocks = 32;
    c.swa_blocks    = 32;
    MetadataReplay replay(c);
    auto           a = replay.run(request(1, {1, 2, 3, 4}));
    auto           b = replay.run(request(2, {1, 2, 3, 4}));
    require(a.success && a.reuse_tokens == 0, "first miss");
    require(b.success && b.reuse_tokens == 1024, "repeat hit");
    MetadataReplay pinned(c);
    require(pinned.checkPinnedForceDrop(), "pinned force_drop must fail without destroying refs");
    MetadataReplayConfig f;
    f.swa           = false;
    f.device_blocks = 8;
    f.host_blocks   = 0;
    f.disk_blocks   = 0;
    MetadataReplay force(f);
    require(force.run(request(1, {1, 2, 3, 4})).success, "first allocation");
    require(force.run(request(2, {11, 12, 13, 14})).success, "force_drop allocation");
    require(force.stats().force_blocks > 0, "native force_drop must reclaim");
    MetadataReplayConfig tiered;
    tiered.swa           = false;
    tiered.device_blocks = 16;
    tiered.host_blocks   = 16;
    tiered.disk_blocks   = 128;
    tiered.device_low = tiered.host_low = 0.5;
    tiered.device_high = tiered.host_high = 0.75;
    MetadataReplay pressure(tiered);
    for (int j = 0; j < 12; ++j) {
        int x = j * 100;
        require(pressure.run(request(j + 1, {x + 1, x + 2, x + 3, x + 4})).success, "pressure fill");
    }
    require(pressure.stats().watermark_blocks > 0 && pressure.stats().host_disk > 0, "watermark cascade");
    // Rebuild the deterministic fixture for each candidate.
    std::vector<int64_t> cold_keys;
    for (int j = 0; j < 12 && cold_keys.empty(); ++j) {
        MetadataReplay probe(tiered);
        for (int seed = 0; seed < 12; ++seed) {
            const int x = seed * 100;
            require(probe.run(request(seed + 1, {x + 1, x + 2, x + 3, x + 4})).success, "probe fill");
        }
        const int x = j * 100;
        std::vector<int64_t> keys{x + 1, x + 2, x + 3, x + 4};
        const auto before = probe.stats();
        const auto hit = probe.run(request(13, keys));
        const auto moves = probe.stats();
        if (hit.success && hit.reuse_tokens == 1024 && (moves.host_device > before.host_device || moves.disk_device > before.disk_device))
            cold_keys = keys;
    }
    require(!cold_keys.empty(), "fixture requires a verified lower-tier hit");
    pressure.failNextTransfer();
    auto broken = pressure.run(request(13, cold_keys));
    require(!broken.success && broken.status == "load_failed" && broken.reuse_tokens == 0, "injected load failure");
    auto recovered = pressure.run(request(14, cold_keys));
    require(recovered.success && recovered.reuse_tokens == 1024 && pressure.stats().failed_transfers == 1,
            "successful retry after rollback");
    std::cout << "PASS watermark cascade, injected load failure, release and recovery\n";
    std::cout << "PASS basic metadata lifecycle\n";
}
