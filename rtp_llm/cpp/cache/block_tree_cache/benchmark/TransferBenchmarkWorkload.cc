#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/TransferBenchmarkWorkload.h"

#include <algorithm>
#include <numeric>

namespace rtp_llm::benchmark {

size_t transferWorkingSetIndex(size_t logical_coordinate, size_t working_set_blocks, bool random_io, uint64_t seed) {
    if (!random_io) {
        return logical_coordinate % working_set_blocks;
    }
    size_t step = static_cast<size_t>((seed * 2 + 1) % working_set_blocks);
    if (step == 0)
        step = 1;
    while (std::gcd(step, working_set_blocks) != 1)
        ++step;
    return (static_cast<size_t>(seed % working_set_blocks) + logical_coordinate * step) % working_set_blocks;
}

std::vector<ScheduledTransferOperation> scheduleTransferWave(size_t   operation_count,
                                                             size_t   direction_count,
                                                             size_t   start_coordinate,
                                                             size_t   working_set_blocks,
                                                             bool     random_io,
                                                             uint64_t seed,
                                                             size_t   coordinate_offset_begin,
                                                             size_t   wave_width) {
    std::vector<ScheduledTransferOperation> operations;
    const size_t coordinate_count = (operation_count + direction_count - 1) / direction_count;
    const size_t coordinate_end   = std::min(coordinate_offset_begin + wave_width, coordinate_count);
    operations.reserve((coordinate_end - coordinate_offset_begin) * direction_count);
    for (size_t coordinate_offset = coordinate_offset_begin; coordinate_offset < coordinate_end; ++coordinate_offset) {
        const size_t logical_coordinate = start_coordinate + coordinate_offset;
        const size_t working_set_index =
            transferWorkingSetIndex(logical_coordinate, working_set_blocks, random_io, seed);
        for (size_t direction_index = 0; direction_index < direction_count; ++direction_index) {
            const size_t operation_index = coordinate_offset * direction_count + direction_index;
            if (operation_index >= operation_count) {
                break;
            }
            operations.push_back({operation_index,
                                  direction_index,
                                  logical_coordinate,
                                  working_set_index,
                                  coordinate_offset - coordinate_offset_begin});
        }
    }
    return operations;
}

}  // namespace rtp_llm::benchmark
