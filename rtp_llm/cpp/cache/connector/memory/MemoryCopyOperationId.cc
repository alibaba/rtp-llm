#include "rtp_llm/cpp/cache/connector/memory/MemoryCopyOperationId.h"

#include <chrono>
#include <iomanip>
#include <random>
#include <sstream>

#include <unistd.h>

namespace rtp_llm {
namespace {

std::atomic<uint64_t> process_epoch_counter{0};

std::string makeEpoch() {
    const auto wall_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count();
    const auto monotonic_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
    std::random_device entropy;
    const uint64_t     random_bits = (static_cast<uint64_t>(entropy()) << 32U) ^ entropy();
    const uint64_t     process_sequence = process_epoch_counter.fetch_add(1, std::memory_order_relaxed);

    std::ostringstream stream;
    stream << std::hex << static_cast<uint64_t>(getpid()) << '-' << static_cast<uint64_t>(wall_time) << '-'
           << static_cast<uint64_t>(monotonic_time) << '-' << random_bits << '-' << process_sequence;
    return stream.str();
}

}  // namespace

MemoryCopyOperationIdGenerator::MemoryCopyOperationIdGenerator(): epoch_(makeEpoch()) {}

std::string MemoryCopyOperationIdGenerator::next() {
    return epoch_ + ':' + std::to_string(getpid()) + ':'
           + std::to_string(counter_.fetch_add(1, std::memory_order_relaxed));
}

}  // namespace rtp_llm
