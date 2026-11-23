#include <stdexcept>
#include <atomic>
#include <cstdint>
namespace rtp_llm {
namespace {
std::atomic<uint64_t> attempts{0};
}
namespace benchmark {
uint64_t physicalCopyAttempts() {
    return attempts.load();
}
}  // namespace benchmark
struct BatchCopyParams;
// This executable must never fall through to the physical forward-copy operation.
void execBatchCopy(const BatchCopyParams&) {
    ++attempts;
    throw std::logic_error("physical execBatchCopy is forbidden in metadata replay");
}
}  // namespace rtp_llm
