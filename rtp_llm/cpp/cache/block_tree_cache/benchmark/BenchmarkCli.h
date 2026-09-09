#pragma once

#include <functional>
#include <string>

#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/CommonBenchmarkOptions.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/ModelProfile.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeBenchmarkOptions.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TransferBenchmarkOptions.h"

namespace rtp_llm::benchmark {

struct BenchmarkCommand {
    std::string      subcommand;
    BenchmarkOptions common;
    TreeOptions      tree;
    TransferOptions  transfer;
    ModelProfile     profile;
};

// Help, parsing, validation and profile loading complete before invoking the GPU runner.
int runBenchmarkCli(int argc, char** argv, const std::function<bool(const BenchmarkCommand&)>& run);

}  // namespace rtp_llm::benchmark
