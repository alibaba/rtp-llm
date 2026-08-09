#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/CommonBenchmarkOptions.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace rtp_llm::benchmark {

namespace {

std::pair<std::string, std::string> parseArg(const char* arg) {
    if (arg[0] != '-' || arg[1] != '-')
        return {};
    std::string full = arg + 2;
    auto        eq   = full.find('=');
    if (eq == std::string::npos)
        return {full, ""};
    return {full.substr(0, eq), full.substr(eq + 1)};
}

}  // anonymous namespace

BenchmarkOptions BenchmarkOptions::parse(int& argc, char**& argv) {
    BenchmarkOptions opts;
    int              write_idx = 1;  // keep argv[0] (program name)

    for (int i = 1; i < argc; ++i) {
        auto [key, value] = parseArg(argv[i]);
        if (key.empty()) {
            argv[write_idx++] = argv[i];
            continue;
        }

        auto next = [&]() -> std::string {
            if (!value.empty())
                return value;
            if (i + 1 < argc) {
                ++i;
                return argv[i];
            }
            throw std::runtime_error("Missing value for --" + key);
        };

        auto parseInteger = [&]() -> int {
            const auto text   = next();
            size_t     parsed = 0;
            const auto result = std::stoi(text, &parsed);
            if (parsed != text.size())
                throw std::runtime_error("Invalid integer for --" + key + ": " + text);
            return result;
        };
        auto parseUnsigned = [&]() -> uint64_t {
            const auto text   = next();
            size_t     parsed = 0;
            const auto result = std::stoull(text, &parsed);
            if (text.empty() || text.front() == '-' || parsed != text.size())
                throw std::runtime_error("Invalid integer for --" + key + ": " + text);
            return result;
        };
        auto parseDouble = [&]() -> double {
            const auto text   = next();
            size_t     parsed = 0;
            const auto result = std::stod(text, &parsed);
            if (parsed != text.size())
                throw std::runtime_error("Invalid number for --" + key + ": " + text);
            return result;
        };

        if (key == "model-profile")
            opts.model_profile_path = next();
        else if (key == "cuda-device")
            opts.cuda_device = parseInteger();
        else if (key == "seed")
            opts.seed = parseUnsigned();
        else if (key == "output-json")
            opts.output_json_path = next();
        else if (key == "max-device-memory-fraction")
            opts.max_device_memory_fraction = parseDouble();
        else if (key == "log-level")
            opts.log_level = next();
        else if (key == "help") {
            printHelp();
            std::exit(0);
        } else {
            // Unknown option, pass through for subcommand parsing
            argv[write_idx++] = argv[i];
        }
    }

    argc = write_idx;
    return opts;
}

void BenchmarkOptions::printHelp() {
    std::cout << "Common options:\n"
              << "  --model-profile=PATH          Model profile JSON path\n"
              << "  --cuda-device=N               CUDA device ordinal (default: 0)\n"
              << "  --seed=N                      Random seed (default: 42)\n"
              << "  --output-json=PATH            Output JSON path\n"
              << "  --max-device-memory-fraction=R  Max device memory fraction (default: 0.8)\n"
              << "  --log-level=LEVEL             Log level: ERROR|WARN|INFO (default: INFO)\n"
              << "  --help                        Show this help\n";
}

}  // namespace rtp_llm::benchmark
