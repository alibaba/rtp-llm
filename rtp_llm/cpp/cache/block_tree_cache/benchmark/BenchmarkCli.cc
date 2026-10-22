#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkCli.h"

#include <iostream>
#include <stdexcept>

namespace rtp_llm::benchmark {
namespace {
void printUsage(const char* program) {
    std::cout << "Usage: " << program << " <tree|transfer> [options...]\n\n";
    BenchmarkOptions::printHelp();
    std::cout << "\nSubcommand help: " << program << " tree --help | transfer --help\n";
}

void validateTransferOptions(const TransferOptions& options) {
    if (options.group_set.empty() || options.transfer_directions.empty())
        throw std::runtime_error("transfer group set and directions must not be empty");
    if (options.transfer_operation_count == 0 || options.transfer_concurrency == 0 || options.min_measured_seconds == 0
        || options.device_disk_staging_block_count == 0) {
        throw std::runtime_error("transfer count, concurrency and duration options must be positive");
    }
    if (options.disk_io_mode != "direct" && options.disk_io_mode != "buffered")
        throw std::runtime_error("--disk-io-mode must be direct or buffered");
    if (options.disk_access_pattern != "sequential" && options.disk_access_pattern != "random")
        throw std::runtime_error("--disk-access-pattern must be sequential or random");
    if (options.copy_strategy != "auto" && options.copy_strategy != "batch" && options.copy_strategy != "staged-sm") {
        throw std::runtime_error("--copy-strategy must be auto, batch or staged-sm");
    }
}

}  // namespace

int runBenchmarkCli(int argc, char** argv, const std::function<bool(const BenchmarkCommand&)>& run) {
    try {
        if (argc < 2) {
            printUsage(argv[0]);
            return 1;
        }
        BenchmarkCommand command;
        command.subcommand = argv[1];
        if (command.subcommand == "--help" || command.subcommand == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        if (command.subcommand != "tree" && command.subcommand != "transfer") {
            throw std::runtime_error("Unknown subcommand: " + command.subcommand);
        }
        for (int index = 2; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--help" || argument == "-h") {
                BenchmarkOptions::printHelp();
                if (command.subcommand == "tree") {
                    TreeOptions::printHelp();
                } else {
                    TransferOptions::printHelp();
                }
                return 0;
            }
        }
        command.common = BenchmarkOptions::parse(argc, argv);
        if (command.subcommand == "tree") {
            command.tree = TreeOptions::parse(argc, argv);
        } else {
            command.transfer = TransferOptions::parse(argc, argv);
            validateTransferOptions(command.transfer);
        }
        if (argc > 2) {
            throw std::runtime_error("Unknown argument: " + std::string(argv[2]));
        }
        if (command.common.model_profile_path.empty()) {
            throw std::runtime_error("--model-profile is required");
        }
        command.profile = ModelProfile::load(command.common.model_profile_path);
        std::cout << "Loaded profile: " << command.profile.profile_id
                  << " (SHA256: " << command.profile.sha256_hex.substr(0, 16) << "...)\n";
        const bool ok = run(command);
        if (!command.common.output_json_path.empty()) {
            std::cout << "Results written to: " << command.common.output_json_path << '\n';
        }
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace rtp_llm::benchmark
