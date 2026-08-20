#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/CommonBenchmarkOptions.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkArgumentParser.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace rtp_llm::benchmark {

BenchmarkOptions BenchmarkOptions::parse(int& argc, char**& argv) {
    BenchmarkOptions opts;
    consumeOptions(argc, argv, [&](const std::string& key, const NextArgumentValue& next) {
        if (key == "model-profile")
            opts.model_profile_path = next();
        else if (key == "cuda-device")
            opts.cuda_device = parseInteger(key, next);
        else if (key == "seed")
            opts.seed = parseUnsigned(key, next);
        else if (key == "repetition-id")
            opts.repetition_id = parseUnsigned(key, next);
        else if (key == "output-json")
            opts.output_json_path = next();
        else if (key == "max-device-memory-fraction")
            opts.max_device_memory_fraction = parseDouble(key, next);
        else if (key == "help") {
            printHelp();
            std::exit(0);
        } else
            return false;
        return true;
    });
    if (opts.cuda_device < 0) {
        throw std::runtime_error("--cuda-device must be non-negative");
    }
    if (!std::isfinite(opts.max_device_memory_fraction) || opts.max_device_memory_fraction <= 0.0
        || opts.max_device_memory_fraction > 1.0) {
        throw std::runtime_error("--max-device-memory-fraction must be in (0, 1]");
    }
    return opts;
}

void BenchmarkOptions::printHelp() {
    std::cout << "Common options:\n"
              << "  --model-profile=PATH          Tree model profile or transfer descriptor-size profile\n"
              << "  --cuda-device=N               CUDA device ordinal (default: 0)\n"
              << "  --seed=N                      Random seed (default: 42)\n"
              << "  --repetition-id=N             Stable repetition identity (default: 0)\n"
              << "  --output-json=PATH            Output JSON path\n"
              << "  --max-device-memory-fraction=R  Max device memory fraction (default: 0.8)\n"
              << "  --help                        Show this help\n";
}

}  // namespace rtp_llm::benchmark
