#pragma once
#include <cstddef>
namespace rtp_llm::block_transfer_engine_test {
enum class RemoteAllocationSite {
    NONE,
    INTENT,
    QUEUE,
    BATCH,
    SLICE
};
void failNextRemoteAllocationOfSize(size_t bytes);
void armRemoteAllocationFailure(RemoteAllocationSite site);
bool remoteAllocationFailed();
void clearRemoteAllocationFailure();
}  // namespace rtp_llm::block_transfer_engine_test
