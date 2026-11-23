#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <unistd.h>
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/offline_replay/MetadataReplay.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/offline_replay/EvaluationStats.h"
using namespace rtp_llm::benchmark;
namespace fs = std::filesystem;
MetadataRequest request(int64_t id, std::vector<int64_t> keys) {
    MetadataRequest r;
    r.id           = id;
    r.timestamp_ns = id;
    r.input_tokens = keys.size() * 256 + 128;
    r.prefix_keys  = std::move(keys);
    return r;
}
std::string contents(const fs::path& p) {
    std::ifstream f(p);
    return {std::istreambuf_iterator<char>(f), {}};
}
void checkMetricAggregation() {
    EvaluationStats original;
    original.add(1, 1024, 512, 256, {});
    original.add(2, 1024, 0, 0, {});
    original.add(300000000001ULL, 1024, 1024, 1024, {});
    stateRequire(original.total.full.percentile(50) == 512 && original.total.gap.sum == 256
                     && original.total.zero_full == 1 && original.timeline.size() == 2,
                 "metric aggregation differs from known values");
}
int main() {
    try {
        checkMetricAggregation();
        const auto dir = fs::temp_directory_path() / ("btc-v2-contract-" + std::to_string(getpid()));
        stateRequire(fs::create_directory(dir), "temporary directory exists");
        MetadataReplayConfig c;
        c.device_blocks     = 16;
        c.swa_blocks        = 16;
        c.host_blocks       = 16;
        c.disk_blocks       = 32;
        c.last_n            = 2;
        c.min_replay_tokens = 128;
        c.device_low        = .5;
        c.device_high       = .75;
        c.host_low          = .5;
        c.host_high         = .75;
        c.disk_low          = .5;
        c.disk_high         = .75;
        MetadataReplay sparse(c);
        stateRequire(sparse.run(request(1, {1, 2, 3, 4, 5, 6, 7, 8})).success, "sparse seed failed");
        const auto gap = sparse.run(request(2, {1, 2, 3, 4, 5, 6}));
        stateRequire(gap.success && gap.full_match_tokens == 1536 && gap.joint_match_tokens == 0,
                     "SWA hole was not reflected in FULL/joint gap");
        // A continued chain is not a structural fork, and a new root child is not a fork.
        MetadataReplayConfig structure_config;
        structure_config.swa        = false;
        structure_config.device_low = structure_config.host_low = structure_config.disk_low = 0.8;
        structure_config.device_high = structure_config.host_high = structure_config.disk_high = 0.9;
        std::cout << "CHECK static tree" << std::endl;
        MetadataReplay structure(structure_config);
        stateRequire(structure.run(request(1, {1, 2})).success, "structure seed");
        stateRequire(structure.run(request(2, {1, 2, 3})).success, "structure extension");
        stateRequire(structure.run(request(3, {10, 11})).success, "second root context");
        structure.writeTreeStructure(dir.string());
        stateRequire(contents(dir / "tree-structure.json").find("\"branch_nodes\":0") != std::string::npos,
                     "chain extension or root was counted as structural fork");
        stateRequire(structure.run(request(4, {1, 2, 4})).success, "structure fork");
        structure.writeTreeStructure(dir.string());
        const auto snapshot = contents(dir / "tree-structure.csv");
        stateRequire(snapshot.find("2,1,2,2,1") != std::string::npos, "fork at key 2 missing");
        stateRequire(contents(dir / "tree-structure.json").find("\"branch_nodes\":1") != std::string::npos,
                     "expected exactly one structural fork");
        const auto before = structure.stats();
        structure.writeTreeStructure(dir.string());
        stateRequire(contents(dir / "tree-structure.csv") == snapshot
                         && structure.stats().tree_nodes == before.tree_nodes
                         && structure.stats().watermark_blocks == before.watermark_blocks,
                     "export mutated cache");
        // A prefix match can survive an allocation failure; no insertion occurred.
        std::vector<int64_t> oversized{1, 2, 3, 4, 5, 6};
        for (int64_t i = 0; i < 128; ++i)
            oversized.push_back(9000 + i);
        const auto failed = sparse.run(request(3, oversized));
        stateRequire(!failed.success && failed.full_match_tokens > 0, "expected FULL before allocation failure");
        MetadataReplay pinned(c);
        stateRequire(pinned.checkPinnedForceDrop(), "pinned allocation did not trigger native force_drop");
        std::cout << "CHECK pressure replay" << std::endl;
        MetadataReplay continuous(c);
        auto           a = continuous.run(request(1, {1, 2, 3, 4, 5, 6, 7, 8}));
        auto           b = continuous.run(request(2, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
        stateRequire(a.success && b.success && a.joint_match_tokens == 0 && b.full_match_tokens == 2048
                         && b.joint_match_tokens == 2048,
                     "native F/J matching failed");
        for (int id = 3; id < 35; ++id) {
            std::vector<int64_t> keys;
            for (int j = 1; j <= 8; ++j)
                keys.push_back(id * 100 + j);
            stateRequire(continuous.run(request(id, keys)).success, "pressure request failed");
        }
        std::cout << "PASS native F/J, read-only structural analysis and pressure replay\n";
        ::unlink((dir / "tree-structure.csv").c_str());
        ::unlink((dir / "tree-structure.json").c_str());
        ::rmdir(dir.c_str());
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
