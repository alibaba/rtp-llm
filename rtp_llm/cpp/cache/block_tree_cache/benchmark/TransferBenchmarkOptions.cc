#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TransferBenchmarkOptions.h"

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

TransferOptions TransferOptions::parse(int& argc, char**& argv) {
    TransferOptions opts;
    int             write_idx = 1;

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

        auto parseInt = [&]() -> size_t {
            const auto text   = next();
            size_t     parsed = 0;
            const auto result = std::stoull(text, &parsed);
            if (text.empty() || text.front() == '-' || parsed != text.size())
                throw std::runtime_error("Invalid integer for --" + key + ": " + text);
            return result;
        };

        if (key == "group-set")
            opts.group_set = next();
        else if (key == "transfer-direction" || key == "transfer-directions") {
            opts.transfer_directions.clear();
            std::string dirs = next();
            size_t      pos  = 0;
            while (pos <= dirs.size()) {
                auto comma = dirs.find(',', pos);
                if (comma == std::string::npos)
                    comma = dirs.size();
                std::string dir = dirs.substr(pos, comma - pos);
                if (!dir.empty())
                    opts.transfer_directions.push_back(dir);
                pos = comma + 1;
            }
        } else if (key == "transfer-operation-count")
            opts.transfer_operation_count = parseInt();
        else if (key == "transfer-concurrency")
            opts.transfer_concurrency = parseInt();
        else if (key == "copy-strategy")
            opts.copy_strategy = next();
        else if (key == "min-measured-seconds")
            opts.min_measured_seconds = parseInt();
        else if (key == "host-memory")
            opts.host_memory = next();
        else if (key == "disk-path")
            opts.disk_path = next();
        else if (key == "disk-io-mode")
            opts.disk_io_mode = next();
        else if (key == "disk-access-pattern")
            opts.disk_access_pattern = next();
        else if (key == "working-set-blocks")
            opts.working_set_blocks = parseInt();
        else if (key == "device-disk-staging-block-count")
            opts.device_disk_staging_block_count = parseInt();
        else if (key == "help") {
            printHelp();
            std::exit(0);
        } else {
            argv[write_idx++] = argv[i];
        }
    }

    argc = write_idx;
    return opts;
}

void TransferOptions::printHelp() {
    std::cout
        << "Transfer benchmark options:\n"
        << "  --group-set=NAME             full_context | swa (default: full_context)\n"
        << "  --transfer-directions=DIRS   comma list, e.g. d2h,h2d mixes both in one case\n"
        << "                               (d2h|h2d|d2disk|disk2d|h2disk|disk2h, default: d2h)\n"
        << "  --transfer-operation-count=N Number of transfer operations (default: 1024)\n"
        << "  --transfer-concurrency=N     Concurrent workers (default: 1)\n"
        << "  --copy-strategy=STRATEGY     auto | batch | staged-sm (default: auto)\n"
        << "  --min-measured-seconds=N     Measured phase duration floor; pilot run scales op count (default: 30)\n"
        << "  --host-memory=TYPE           pinned | pageable (default: pinned)\n"
        << "  --disk-path=PATH             Disk directory for disk transfers\n"
        << "  --disk-io-mode=MODE          direct | buffered (default: direct)\n"
        << "  --disk-access-pattern=PAT    sequential | random (default: sequential)\n"
        << "  --working-set-blocks=N       Transfer pool size (0 = auto: concurrency*4, workers rotate)\n"
        << "  --device-disk-staging-block-count=N  Device<->Disk staging buffers, caps in-flight ops (default: 4)\n"
        << "  --help                       Show this help\n";
}

}  // namespace rtp_llm::benchmark
