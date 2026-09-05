#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rtp_llm::benchmark {

struct BenchmarkOptions {
    std::string model_profile_path;
    int         cuda_device{0};
    uint64_t    seed{42};
    uint64_t    repetition_id{0};
    std::string output_json_path;
    double      max_device_memory_fraction{0.8};

    static BenchmarkOptions parse(int& argc, char**& argv);
    static void             printHelp();
};

}  // namespace rtp_llm::benchmark
