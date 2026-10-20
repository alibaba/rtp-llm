#pragma once

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcDumpWriter.h"
#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

namespace rtp_llm {

// A bounded set of preallocated workspaces shared by all local transfer directions.
// Each synchronous operation owns its slot until CUDA completion, including errors.
class CrcTransferService {
public:
    CrcTransferService(const std::vector<GroupSetPtr>& groups,
                       size_t                          max_batch,
                       size_t                          workers,
                       int64_t                         world_rank  = 0,
                       std::shared_ptr<CrcDumpWriter>  dump_writer = nullptr);
    TransferResult copy(const std::vector<HostBufferView>&     hosts,
                        const std::vector<TransferDescriptor>& descriptors,
                        const std::vector<const GroupSet*>&    groups);

private:
    struct Slot {
        std::unique_ptr<CrcBlockCopyBatch> copy;
        bool                               busy{false};
    };
    enum class Operation {
        STORE,
        LOAD
    };
    TransferResult        execute(int                                    device,
                                  const std::vector<CrcCopyItem>&        items,
                                  Operation                              operation,
                                  const std::vector<TransferDescriptor>& descriptors);
    static int            deviceFor(const GroupSet& group);
    static TransferStatus statusFor(CrcCopyStatus status);

    std::map<int, std::vector<Slot>> slots_;
    std::mutex                       mutex_;
    std::condition_variable          cv_;
};

}  // namespace rtp_llm
