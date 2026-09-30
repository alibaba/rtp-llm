#pragma once

#include <condition_variable>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

namespace rtp_llm {

// A bounded set of preallocated workspaces shared by all local transfer directions.
// Each synchronous operation owns its slot until CUDA completion, including errors.
class CrcTransferService {
public:
    CrcTransferService(const std::vector<GroupSetPtr>& groups, size_t max_batch, size_t workers);
    TransferStatus copy(const std::vector<HostBufferView>&     hosts,
                        const std::vector<TransferDescriptor>& descriptors,
                        const std::vector<const GroupSet*>&    groups);
    TransferStatus validate(const std::vector<HostBufferView>& hosts, const std::vector<const GroupSet*>& groups);

private:
    struct Slot {
        std::unique_ptr<CrcBlockCopyBatch> copy;
        bool                               busy{false};
    };
    enum class Operation {
        STORE,
        LOAD,
        VALIDATE
    };
    TransferStatus        execute(int device, const std::vector<CrcCopyItem>& items, Operation operation);
    static int            deviceFor(const GroupSet& group);
    static TransferStatus statusFor(CrcCopyStatus status);

    std::map<int, std::vector<Slot>> slots_;
    std::mutex                       mutex_;
    std::condition_variable          cv_;
    std::atomic<unsigned>            logged_operations_{0};
};

}  // namespace rtp_llm
