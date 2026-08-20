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

cudaDeviceProp selectCudaDevice(int cuda_device) {
    const cudaError_t set_status = cudaSetDevice(cuda_device);
    if (set_status != cudaSuccess) {
        throw std::runtime_error("cudaSetDevice(" + std::to_string(cuda_device)
                                 + ") failed: " + cudaGetErrorString(set_status));
    }
    cudaDeviceProp    prop{};
    const cudaError_t prop_status = cudaGetDeviceProperties(&prop, cuda_device);
    if (prop_status != cudaSuccess) {
        throw std::runtime_error("cudaGetDeviceProperties(" + std::to_string(cuda_device)
                                 + ") failed: " + cudaGetErrorString(prop_status));
    }
    return prop;
}

template<typename Options, typename ParseOptions, typename ValidateOptions, typename LoadProfile, typename Run>
int runBenchmark(const char*       label,
                 int               argc,
                 char**            argv,
                 ParseOptions&&    parse_options,
                 ValidateOptions&& validate_options,
                 LoadProfile&&     load_profile,
                 Run&&             run) {
    const BenchmarkOptions common  = BenchmarkOptions::parse(argc, argv);
    const Options          options = parse_options(argc, argv);
    rejectUnknownArguments(argc, argv);
    validate_options(options);
    if (common.model_profile_path.empty()) {
        std::cerr << "Error: --model-profile is required" << std::endl;
        return 1;
    }

    try {
        const ModelProfile profile = load_profile(common.model_profile_path);
        std::cout << "Loaded profile: " << profile.profile_id << " (SHA256: " << profile.sha256_hex.substr(0, 16)
                  << "...)" << std::endl;
        const cudaDeviceProp prop = selectCudaDevice(common.cuda_device);
        std::cout << "Using GPU: " << prop.name << " (device " << common.cuda_device << ")" << std::endl;
        const bool ok = run(profile, options, common);
        if (!common.output_json_path.empty()) {
            std::cout << "Results written to: " << common.output_json_path << std::endl;
        }
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << label << " benchmark failed: " << e.what() << std::endl;
        return 1;
    }
}

int runTree(int argc, char** argv) {
    return runBenchmark<TreeOptions>(
        "Tree",
        argc,
        argv,
        TreeOptions::parse,
        [](const TreeOptions&) {},
        ModelProfile::load,
        [](const ModelProfile& profile, const TreeOptions& options, const BenchmarkOptions& common) {
            return TreeBenchmarkRunner(profile,
                                       options,
                                       common.seed,
                                       common.repetition_id,
                                       common.cuda_device,
                                       common.max_device_memory_fraction,
                                       common.output_json_path)
                .run();
        });
}

int runTransfer(int argc, char** argv) {
    return runBenchmark<TransferOptions>(
        "Transfer",
        argc,
        argv,
        TransferOptions::parse,
        validateTransferOptions,
        ModelProfile::load,
        [](const ModelProfile& profile, const TransferOptions& options, const BenchmarkOptions& common) {
            return TransferBenchmarkRunner(profile, options, common.seed, common.output_json_path).run();
        });
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
