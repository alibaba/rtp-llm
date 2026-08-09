#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rtp_llm::benchmark {

struct ScheduledTransferOperation {
    size_t operation_index{0};
    size_t direction_index{0};
    size_t logical_coordinate{0};
    size_t working_set_index{0};
};

size_t transferWorkingSetIndex(size_t logical_coordinate, size_t working_set_blocks, bool random_io, uint64_t seed);

std::vector<ScheduledTransferOperation> scheduleTransferWorker(size_t   operation_count,
                                                               size_t   direction_count,
                                                               size_t   start_coordinate,
                                                               size_t   working_set_blocks,
                                                               bool     random_io,
                                                               uint64_t seed,
                                                               size_t   worker_count,
                                                               size_t   worker_id);

}  // namespace rtp_llm::benchmark
