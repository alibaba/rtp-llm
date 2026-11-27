#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcDumpWriter.h"
#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

namespace rtp_llm {

// One direction's CRC backend, serialized by the WorkLoop owner. Lifecycle
// callers may access it after joining the owner. stopAndWaitForIo() proves
// physical completion while leaving stage callbacks for poll().
class CrcCopyExecutor {
public:
    using DoneCallback = std::function<void(TransferResult)>;

    // Optional borrowed output: one zero-initialized record per batch, valid
    // through done. Only raw steady-clock boundaries; profiling stays upstream.
    using Timing = TransferCrcTiming;

    // Groups have validated physical copy geometry. Slot count is per device.
    CrcCopyExecutor(const std::vector<GroupSetPtr>& groups,
                    CrcCopyDirection                direction,
                    size_t                          max_batch,
                    size_t                          slots_per_device = 8,
                    int64_t                         world_rank       = 0,
                    std::shared_ptr<CrcDumpWriter>  dump_writer      = nullptr);
    ~CrcCopyExecutor();

    // false = Busy, without submission or callback; the caller retries later.
    // true = handled: this service has taken responsibility for exactly one result,
    // not necessarily a successful CUDA submission. Immediate errors call done
    // in this stack; accepted work calls it from poll(). A preparation exception precedes CUDA
    // submission and has no callback. Callbacks must not allocate, throw or reenter
    // this service. They only fill the owner's already allocated stage record.
    // The caller keeps descriptor storage, host buffers and business block
    // references valid through done, including deferred diagnostics.
    // WorkLoop supplies a validated, homogeneous CRC batch and a nonempty callback.
    bool trySubmit(const std::vector<HostBufferView>& hosts,
                   DescriptorView                     descriptors,
                   const GroupSetPtr&                 group,
                   DoneCallback                       done,
                   Timing*                            timing = nullptr);
    // Scans every device. Safe slots retire before their callbacks run.
    // true means undelivered work remains. Async slot access needs no mutex.
    // Result-mapping exceptions propagate from poll() and stopAndWaitForIo().
    bool poll();
    size_t availableSlots(int device) const;
    void stopAdmission();
    void stopAndWaitForIo();

private:
    struct Operation {
        explicit Operation(DescriptorView descriptors): descriptors(descriptors) {}
        std::optional<TransferResult>   result;
        GroupSetPtr                     group;
        DescriptorView                  descriptors;
        DoneCallback                    done;
        Timing*                         timing{nullptr};
    };
    struct Slot {
        std::unique_ptr<CrcCopyWorkspace> copy;
        int                               device{0};
        std::optional<Operation> operation;
    };
    // Construct metadata from the caller's resolved descriptor and GroupSet owner.
    static CrcCopyItem
    makeItem(const HostBufferView& host, const TransferDescriptor& descriptor, const GroupSetPtr& group);
    static TransferResult mapResult(CrcCopyState                       state,
                                    CrcCopyDirection                   direction,
                                    DescriptorView                     descriptors,
                                    const std::vector<CrcCopyFailure>& failures);
    void                  advance(Slot& slot, bool deliver);
    static int            deviceFor(const GroupSet& group);
    static TransferStatus statusFor(CrcCopyState state);

    const CrcCopyDirection                 direction_;
    std::map<int, std::vector<Slot>>       slots_;
    std::map<const GroupSet*, GroupSetPtr> groups_;
    bool                                   accepting_{true};
};

}  // namespace rtp_llm
