#pragma once

#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcDumpWriter.h"
#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

namespace rtp_llm {

// One direction's CRC backend; no business callbacks run here. Async calls,
// including poll(), stopAdmission() and drain(), are serialized by the WorkLoop
// owner. Lifecycle callers may access it only after joining that owner. Legacy
// workers share a separate instance per direction and synchronize slot use.
// drain() proves physical completion but leaves stage callbacks for poll().
class CrcTransferService {
public:
    using DoneCallback = std::function<void(TransferResult)>;

    // Slot count is per device in this direction's service.
    // TODO(legacy removal): remove the async parameter. Legacy passes false and
    // shares slots_per_device among the workers of one direction.
    CrcTransferService(const std::vector<GroupSetPtr>& groups,
                       size_t                          max_batch,
                       size_t                          slots_per_device = 8,
                       int64_t                         world_rank       = 0,
                       std::shared_ptr<CrcDumpWriter>  dump_writer      = nullptr,
                       bool                            async            = true);
    ~CrcTransferService();

    // TODO(legacy removal): remove copy() and its synchronous implementation.
    // Internal callers supply resolved descriptors and configured GroupSets,
    // with one direction/device per batch, as arranged by the dispatcher.
    TransferResult copy(const std::vector<HostBufferView>&     hosts,
                        const std::vector<TransferDescriptor>& descriptors,
                        const std::vector<const GroupSet*>&    groups);

    // false = Busy, without submission or callback. true transfers responsibility
    // for exactly one result: immediate errors call done in this stack, accepted
    // work calls it from poll(). A preparation exception precedes CUDA
    // submission and has no callback. Callbacks must not allocate, throw or reenter
    // this service. They only fill the owner's already allocated stage record.
    // The caller keeps the storage behind hosts and business block references
    // valid through completion, including diagnostics/recovery.
    bool trySubmit(const std::vector<HostBufferView>&     hosts,
                   const std::vector<TransferDescriptor>& descriptors,
                   const GroupSetPtr&                     group,
                   DoneCallback                           done);
    // Scans every device. Safe slots retire before their callbacks run.
    // true means undelivered work remains. Async slot access needs no mutex.
    // Result-mapping exceptions propagate from poll() and drain().
    bool poll();
    void stopAdmission();
    void drain();

private:
    struct Operation {
        std::optional<TransferResult>   result;
        GroupSetPtr                     group;
        std::vector<TransferDescriptor> descriptors;
        DoneCallback                    done;
    };
    struct Slot {
        std::unique_ptr<CrcBlockCopyBatch> copy;
        int                                device{0};
        // TODO(legacy removal): remove updating; only legacy workers use it.
        bool                       updating{false};
        std::unique_ptr<Operation> operation;
    };
    void initialize(const std::vector<GroupSetPtr>& groups,
                    size_t                          max_batch,
                    size_t                          slots_per_device,
                    int64_t                         world_rank,
                    std::shared_ptr<CrcDumpWriter>  dump_writer);
    // Construct metadata from the caller's resolved descriptor and GroupSet owner.
    static CrcCopyItem
    makeItem(const HostBufferView& host, const TransferDescriptor& descriptor, const GroupSetPtr& group);
    // TODO(legacy removal): remove this synchronous slot execution helper.
    TransferResult        execute(int                                    device,
                                  const std::vector<CrcCopyItem>&        items,
                                  CrcCopyDirection                       direction,
                                  const std::vector<TransferDescriptor>& descriptors);
    static TransferResult mapResult(CrcCopyState                           status,
                                    CrcCopyDirection                       direction,
                                    const std::vector<TransferDescriptor>& descriptors,
                                    const std::vector<CrcCopyFailure>&     failures);
    void                  advance(Slot& slot, bool deliver);
    static int            deviceFor(const GroupSet& group);
    static TransferStatus statusFor(CrcCopyState status);

    std::map<int, std::vector<Slot>>       slots_;
    std::map<const GroupSet*, GroupSetPtr> groups_;
    // TODO(legacy removal): remove mutex_/cv_; only legacy workers use them.
    std::mutex              mutex_;
    std::condition_variable cv_;
    // TODO(legacy removal): remove async_ and its mode checks; keep accepting_.
    bool async_{false}, accepting_{true};
};

}  // namespace rtp_llm
