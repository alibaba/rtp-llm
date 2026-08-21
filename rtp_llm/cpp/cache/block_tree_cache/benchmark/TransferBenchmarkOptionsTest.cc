#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TransferBenchmarkOptions.h"

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace rtp_llm::benchmark {
namespace {

TransferOptions parseOptions(std::initializer_list<const char*> arguments) {
    std::vector<std::string> storage{"transfer_benchmark"};
    for (const char* argument : arguments) {
        storage.emplace_back(argument);
    }

    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (auto& argument : storage) {
        argv.push_back(argument.data());
    }

    int    argc     = static_cast<int>(argv.size());
    char** argv_ptr = argv.data();
    return TransferOptions::parse(argc, argv_ptr);
}

TEST(TransferBenchmarkOptionsTest, CudaBatchSerializationDefaultsToEnabled) {
    const auto options = parseOptions({});
    EXPECT_TRUE(options.cuda_batch_serialize);
}

TEST(TransferBenchmarkOptionsTest, ParsesEnabledCudaBatchSerializationSwitch) {
    const auto options = parseOptions({"--cuda-batch-serialize=1"});
    EXPECT_TRUE(options.cuda_batch_serialize);
}

TEST(TransferBenchmarkOptionsTest, ParsesDisabledCudaBatchSerializationSwitch) {
    const auto options = parseOptions({"--cuda-batch-serialize=0"});
    EXPECT_FALSE(options.cuda_batch_serialize);
}

TEST(TransferBenchmarkOptionsTest, RejectsNonBooleanCudaBatchSerializationValue) {
    EXPECT_THROW(parseOptions({"--cuda-batch-serialize=2"}), std::runtime_error);
}

}  // namespace
}  // namespace rtp_llm::benchmark
