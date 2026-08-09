#include <cmath>
#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>

#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/ModelProfile.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/CommonBenchmarkOptions.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeBenchmarkOptions.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TransferBenchmarkOptions.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TreeBenchmarkRunner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TransferBenchmarkRunner.h"
namespace rtp_llm::benchmark {

namespace {

void printUsage(const char* program) {
    std::cerr << "Usage: " << program << " <tree|transfer> [options...]\n\n";
    BenchmarkOptions::printHelp();
    std::cerr << "\nFor tree subcommand help:\n";
    std::cerr << "  " << program << " tree --help\n\n";
    std::cerr << "For transfer subcommand help:\n";
    std::cerr << "  " << program << " transfer --help\n";
}

void rejectUnknownArguments(int argc, char** argv) {
    if (argc > 2) {
        throw std::runtime_error("Unknown argument: " + std::string(argv[2]));
    }
}

void validateCommonOptions(const BenchmarkOptions& options) {
    if (options.cuda_device < 0) {
        throw std::runtime_error("--cuda-device must be non-negative");
    }
    if (!std::isfinite(options.max_device_memory_fraction) || options.max_device_memory_fraction <= 0.0
        || options.max_device_memory_fraction > 1.0) {
        throw std::runtime_error("--max-device-memory-fraction must be in (0, 1]");
    }
}

void validateTreeOptions(const TreeOptions& options) {
    if (options.payload_mode != "model_sized" && options.payload_mode != "scaled")
        throw std::runtime_error("--payload-mode must be model_sized or scaled");
    if (options.tree_node_count == 0 || options.max_path_length == 0 || options.tree_branching_factor == 0
        || options.initial_min_path_length == 0 || options.initial_max_path_length == 0 || options.append_length == 0
        || options.inserts_per_match == 0 || options.active_path_limit == 0 || options.operation_trace_count == 0
        || options.steady_threads == 0 || options.warmup_seconds == 0 || options.min_measured_seconds == 0) {
        throw std::runtime_error("tree count and duration options must be positive");
    }
    if (options.initial_min_path_length > options.initial_max_path_length
        || options.initial_max_path_length > options.max_path_length) {
        throw std::runtime_error("initial path lengths must satisfy 0 < min <= max <= --max-path-length");
    }
    if (options.inserts_per_match > options.max_path_length / options.append_length
        || options.initial_max_path_length
               > options.max_path_length - options.inserts_per_match * options.append_length) {
        throw std::runtime_error("initial path plus all incremental inserts must fit --max-path-length");
    }
    if (options.operation_trace_count < options.steady_threads) {
        throw std::runtime_error("--operation-trace-count must be at least --steady-threads");
    }
    if (!std::isfinite(options.continuation_ratio) || !std::isfinite(options.fork_ratio)
        || !std::isfinite(options.fork_reuse_min_ratio) || !std::isfinite(options.fork_reuse_max_ratio)
        || !std::isfinite(options.hot_path_ratio) || options.continuation_ratio < 0.0 || options.fork_ratio < 0.0
        || options.continuation_ratio + options.fork_ratio > 1.0 || options.fork_reuse_min_ratio < 0.0
        || options.fork_reuse_min_ratio > options.fork_reuse_max_ratio || options.fork_reuse_max_ratio > 1.0
        || options.hot_path_ratio < 0.0 || options.hot_path_ratio > 1.0) {
        throw std::runtime_error("tree workload ratios are outside their valid ranges");
    }
}

void validateTransferOptions(const TransferOptions& options) {
    if (options.group_set.empty() || options.transfer_directions.empty())
        throw std::runtime_error("transfer group set and directions must not be empty");
    if (options.transfer_operation_count == 0 || options.transfer_concurrency == 0 || options.min_measured_seconds == 0
        || options.device_disk_staging_block_count == 0) {
        throw std::runtime_error("transfer count, concurrency and duration options must be positive");
    }
    if (options.host_memory != "pinned" && options.host_memory != "pageable")
        throw std::runtime_error("--host-memory must be pinned or pageable");
    if (options.disk_io_mode != "direct" && options.disk_io_mode != "buffered")
        throw std::runtime_error("--disk-io-mode must be direct or buffered");
    if (options.disk_access_pattern != "sequential" && options.disk_access_pattern != "random")
        throw std::runtime_error("--disk-access-pattern must be sequential or random");
    if (options.copy_strategy != "auto" && options.copy_strategy != "batch" && options.copy_strategy != "staged-sm") {
        throw std::runtime_error("--copy-strategy must be auto, batch or staged-sm");
    }
}

int runTree(int argc, char** argv) {
    // Parse common options
    auto common = BenchmarkOptions::parse(argc, argv);

    // Parse tree options
    auto tree_opts = TreeOptions::parse(argc, argv);
    rejectUnknownArguments(argc, argv);
    validateCommonOptions(common);
    validateTreeOptions(tree_opts);

    // Load profile
    if (common.model_profile_path.empty()) {
        std::cerr << "Error: --model-profile is required" << std::endl;
        return 1;
    }

    ModelProfile profile;
    try {
        profile = ModelProfile::load(common.model_profile_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading profile: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Loaded profile: " << profile.profile_id << " (SHA256: " << profile.sha256_hex.substr(0, 16) << "...)"
              << std::endl;

    // Set CUDA device
    cudaSetDevice(common.cuda_device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, common.cuda_device);
    std::cout << "Using GPU: " << prop.name << " (device " << common.cuda_device << ")" << std::endl;

    // Run tree benchmark
    try {
        TreeBenchmarkRunner runner(profile, tree_opts, common.seed, common.output_json_path);
        bool                ok = runner.run();

        if (!common.output_json_path.empty()) {
            std::cout << "Results written to: " << common.output_json_path << std::endl;
        }

        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Tree benchmark failed: " << e.what() << std::endl;
        return 1;
    }
}

int runTransfer(int argc, char** argv) {
    // Parse common options
    auto common = BenchmarkOptions::parse(argc, argv);

    // Parse transfer options
    auto transfer_opts = TransferOptions::parse(argc, argv);
    rejectUnknownArguments(argc, argv);
    validateCommonOptions(common);
    validateTransferOptions(transfer_opts);

    // Load profile
    if (common.model_profile_path.empty()) {
        std::cerr << "Error: --model-profile is required" << std::endl;
        return 1;
    }

    ModelProfile profile;
    try {
        profile = ModelProfile::load(common.model_profile_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading profile: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Loaded profile: " << profile.profile_id << " (SHA256: " << profile.sha256_hex.substr(0, 16) << "...)"
              << std::endl;

    // Set CUDA device
    cudaSetDevice(common.cuda_device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, common.cuda_device);
    std::cout << "Using GPU: " << prop.name << " (device " << common.cuda_device << ")" << std::endl;

    // Run transfer benchmark
    try {
        TransferBenchmarkRunner runner(profile, transfer_opts, common.seed, common.output_json_path);
        bool                    ok = runner.run();

        if (!common.output_json_path.empty()) {
            std::cout << "Results written to: " << common.output_json_path << std::endl;
        }

        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Transfer benchmark failed: " << e.what() << std::endl;
        return 1;
    }
}

}  // anonymous namespace

}  // namespace rtp_llm::benchmark

int main(int argc, char* argv[]) {
    if (argc < 2) {
        rtp_llm::benchmark::printUsage(argv[0]);
        return 1;
    }

    // Subcommand name stays at argv[1]; the parse functions scan the full
    // argv array and treat non-option entries (like the subcommand name) as
    // passthrough, so no shifting is needed.
    std::string subcommand = argv[1];

    if (subcommand == "tree") {
        return rtp_llm::benchmark::runTree(argc, argv);
    } else if (subcommand == "transfer") {
        return rtp_llm::benchmark::runTransfer(argc, argv);
    } else if (subcommand == "--help" || subcommand == "-h") {
        rtp_llm::benchmark::printUsage(argv[0]);
        return 0;
    } else {
        std::cerr << "Unknown subcommand: " << subcommand << std::endl;
        std::cerr << "Use '" << argv[0] << " --help' for usage" << std::endl;
        return 1;
    }
}
