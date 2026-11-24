#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/RemoteAllocationFailureTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/remote_mock/RemoteMockAdapter.h"
#include <atomic>
#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>
#include <new>
namespace rtp_llm::block_transfer_engine_test {
namespace {
std::atomic<RemoteAllocationSite> site{RemoteAllocationSite::NONE};
std::atomic<bool>                 failed{false};
thread_local size_t               selected_size = 0;
thread_local bool                 inspecting    = false;
}  // namespace
void failNextRemoteAllocationOfSize(size_t bytes) {
    selected_size = bytes;
    failed        = false;
}
void armRemoteAllocationFailure(RemoteAllocationSite value) {
    failed = false;
    site   = value;
}
bool remoteAllocationFailed() {
    return failed;
}
void clearRemoteAllocationFailure() {
    site = RemoteAllocationSite::NONE;
}
void             remoteCopyRegistrationAllocation() __attribute__((weak));
bool             remoteCopyRegistrationArmed() __attribute__((weak));
extern "C" void* __real__Znwm(size_t);
extern "C" void* __wrap__Znwm(size_t bytes) {
    if (selected_size && selected_size == bytes) {
        selected_size = 0;
        failed        = true;
        throw std::bad_alloc();
    }
    const auto selected = site.load();
    if (!inspecting
        && (selected != RemoteAllocationSite::NONE || (remoteCopyRegistrationArmed && remoteCopyRegistrationArmed()))) {
        inspecting = true;
        void*      frames[16];
        const auto count = backtrace(frames, 16);
        bool       match = false;
        for (int i = 1; i < count && !match; ++i) {
            Dl_info info{};
            if (!dladdr(frames[i], &info) || !info.dli_sname)
                continue;
            const char* name = info.dli_sname;
            if (remoteCopyRegistrationAllocation && std::strstr(name, "TransferBatchAsyncContext6onDone")) {
                inspecting = false;
                remoteCopyRegistrationAllocation();
                inspecting = true;
            }
            if (selected == RemoteAllocationSite::INTENT)
                match = std::strstr(name, "BlockTreeCache21completeRemoteRequest") != nullptr;
            if (selected == RemoteAllocationSite::QUEUE)
                match = bytes == 512 && i <= 2 && std::strstr(name, "BlockTreeCache21completeRemoteRequest");
            if (selected == RemoteAllocationSite::BATCH)
                match = bytes == sizeof(TransferDescriptor) && std::strstr(name, "RemoteCompletionState8runBatch");
            if (selected == RemoteAllocationSite::SLICE)
                match = std::strstr(name, "RemoteMockWriteback4Impl4send") != nullptr;
        }
        inspecting = false;
        if (match) {
            site   = RemoteAllocationSite::NONE;
            failed = true;
            throw std::bad_alloc();
        }
    }
    return __real__Znwm(bytes);
}
}  // namespace rtp_llm::block_transfer_engine_test
