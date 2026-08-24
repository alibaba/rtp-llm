#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TransferBenchmarkOptions.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkArgumentParser.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace rtp_llm::benchmark {

TransferOptions TransferOptions::parse(int& argc, char**& argv) {
    TransferOptions opts;
    consumeOptions(argc, argv, [&](const std::string& key, const NextArgumentValue& next) {
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
            opts.transfer_operation_count = parseUnsigned(key, next);
        else if (key == "transfer-concurrency")
            opts.transfer_concurrency = parseUnsigned(key, next);
        else if (key == "business-concurrency")
            opts.business_concurrency = parseUnsigned(key, next);
        else if (key == "descriptors-per-business")
            opts.descriptors_per_business = parseUnsigned(key, next);
        else if (key == "transfer-worker-count")
            opts.transfer_worker_count = parseUnsigned(key, next);
        else if (key == "transfer-descriptor-batch-size")
            opts.transfer_descriptor_batch_size = parseUnsigned(key, next);
        else if (key == "copy-strategy")
            opts.copy_strategy = next();
        else if (key == "min-measured-seconds")
            opts.min_measured_seconds = parseUnsigned(key, next);
        else if (key == "host-memory")
            opts.host_memory = next();
        else if (key == "disk-path")
            opts.disk_path = next();
        else if (key == "disk-io-mode")
            opts.disk_io_mode = next();
        else if (key == "disk-access-pattern")
            opts.disk_access_pattern = next();
        else if (key == "working-set-blocks")
            opts.working_set_blocks = parseUnsigned(key, next);
        else if (key == "device-disk-staging-block-count")
            opts.device_disk_staging_block_count = parseUnsigned(key, next);
        else if (key == "help") {
            printHelp();
            std::exit(0);
        } else
            return false;
        return true;
    });
    return opts;
}

void TransferOptions::printHelp() {
    std::cout
        << "Transfer benchmark options:\n"
        << "  --group-set=NAME             full_context | swa (default: full_context)\n"
        << "  --transfer-directions=DIRS   comma list, e.g. d2h,h2d mixes both in one case\n"
        << "                               (d2h|h2d|d2disk|disk2d|h2disk|disk2h, default: d2h)\n"
        << "  --transfer-operation-count=N Number of transfer operations (default: 1024)\n"
        << "  --transfer-concurrency=N     Concurrent endpoint lanes per wave (default: 1)\n"
        << "  --business-concurrency=N     Concurrent independent business requests (0 = legacy wave mode)\n"
        << "  --descriptors-per-business=N Same-direction descriptors owned by each business request\n"
        << "  --transfer-worker-count=N    Lower transfer workers (default: 1)\n"
        << "  --transfer-descriptor-batch-size=N  Descriptors per engine submit (0 = concurrency)\n"
        << "  --copy-strategy=STRATEGY     auto | batch | staged-sm (default: auto)\n"
        << "  --min-measured-seconds=N     Measured phase duration floor; pilot run scales op count (default: 30)\n"
        << "  --host-memory=TYPE           pinned | pageable (default: pinned)\n"
        << "  --disk-path=PATH             Disk directory for disk transfers\n"
        << "  --disk-io-mode=MODE          direct | buffered (default: direct)\n"
        << "  --disk-access-pattern=PAT    sequential | random (default: sequential)\n"
        << "  --working-set-blocks=N       Transfer pool size (0 = auto: concurrency*4, waves rotate)\n"
        << "  --device-disk-staging-block-count=N  Device<->Disk staging buffers, passed through unchanged "
           "(default: 4)\n"
        << "  --help                       Show this help\n";
}

}  // namespace rtp_llm::benchmark
