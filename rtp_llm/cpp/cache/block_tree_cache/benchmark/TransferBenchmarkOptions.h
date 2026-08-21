#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace rtp_llm::benchmark {

struct TransferOptions {
    // Group set selection
    std::string group_set{"full_context"};  // "full_context" or "swa"

    // Transfer directions (comma separated, e.g. "d2h,h2d" mixes both in one case)
    std::vector<std::string> transfer_directions{"d2h"};  // d2h, h2d, d2disk, disk2d, h2disk, disk2h

    // Operation count and concurrency
    size_t transfer_operation_count{1024};
    size_t transfer_concurrency{1};

    // Number of same-direction descriptors submitted in one transfer-engine
    // API call. 0 follows transfer_concurrency. Device->Disk is the only
    // direction whose engine contract still requires singleton submissions.
    size_t transfer_descriptor_batch_size{0};

    // Device<->Host copy strategy. Explicit strategies disable the other
    // optimized path; the runner observes the actual strategy and rejects
    // fallback or mixed execution.
    std::string copy_strategy{"auto"};  // "auto", "batch" or "staged-sm"

    // Measured-phase duration floor (seconds). A pilot run scales the
    // operation count so the measured phase lasts at least this long.
    size_t min_measured_seconds{30};

    // Host memory type
    std::string host_memory{"pinned"};  // "pinned" or "pageable"

    // Disk configuration
    std::string disk_path;
    std::string disk_io_mode{"direct"};             // "direct" or "buffered"
    std::string disk_access_pattern{"sequential"};  // "sequential" or "random"

    // Pool sizing for pure-path transfer. 0 = auto (concurrency * 4), so the
    // working set exceeds concurrency and buffered IO does not fully fit in
    // page cache; workers rotate blocks round-robin over the pool.
    size_t working_set_blocks{0};

    // Device<->Disk staging buffer count. This pool caps in-flight
    // device-disk ops (throughput ~= pool size / per-op hold time), so it
    // should be >= transfer_concurrency for path-capability measurements.
    // Default 4 matches the production BlockTreeCacheConfig default.
    size_t device_disk_staging_block_count{4};

    static TransferOptions parse(int& argc, char**& argv);
    static void            printHelp();
};

}  // namespace rtp_llm::benchmark
