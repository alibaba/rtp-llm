#include "aios/navi_sched/data/MasterPendingCacheSyncToAgentData.h"
#include "aios/navi_sched/resource/AgentReportR.h"
#include "navi/engine/Kernel.h"

namespace navi_sched {

class AgentPendingCacheSyncKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    KERNEL_DEPEND_DECLARE();

private:
    KERNEL_DEPEND_ON(AgentReportR, _agentReportR);
};

void AgentPendingCacheSyncKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.agent_pending_cache_sync_k")
        .input("request", MASTER_PENDING_CACHE_SYNC_TO_AGENT_TYPE_ID);
}

navi::ErrorCode AgentPendingCacheSyncKernel::compute(
    navi::KernelComputeContext &ctx)
{
    navi::DataPtr data;
    bool eof = false;
    if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
        NAVI_KERNEL_LOG(ERROR, "read Agent pending cache sync input failed");
        return navi::EC_ABORT;
    }
    auto request =
        std::dynamic_pointer_cast<MasterPendingCacheSyncToAgentData>(data);
    if (request) {
        const auto reported =
            _agentReportR->reportPendingCacheSnapshot(request->collectorId);
        NAVI_KERNEL_LOG(
            WARN,
            "pending cache full snapshot fallback handled [%d], collectorId [%s], "
            "reason [%s], fullSnapshot [%d], snapshotReadyBefore [%d], "
            "masterRevisionBefore [%lu], expectedRevision [%lu], "
            "rejectedRevision [%lu], masterChecksumBefore [%lu], "
            "candidateChecksum [%lu], receivedChecksum [%lu], "
            "candidateStateComputed [%d], masterBatchCountBefore [%ld], "
            "candidateBatchCount [%ld], receivedBatchCount [%ld], "
            "masterBlockCountBefore [%ld], candidateBlockCount [%ld], "
            "receivedBlockCount [%ld], masterTailBlockCountBefore [%ld], "
            "candidateTailBlockCount [%ld], receivedTailBlockCount [%ld]",
            static_cast<int>(reported), request->collectorId.c_str(),
            request->reason.c_str(), static_cast<int>(request->fullSnapshot),
            static_cast<int>(request->snapshotReadyBefore),
            request->masterRevisionBefore, request->expectedRevision,
            request->rejectedRevision, request->masterChecksumBefore,
            request->candidateChecksum, request->receivedChecksum,
            static_cast<int>(request->candidateStateComputed),
            request->masterBatchCountBefore, request->candidateBatchCount,
            request->receivedBatchCount, request->masterBlockCountBefore,
            request->candidateBlockCount, request->receivedBlockCount,
            request->masterTailBlockCountBefore,
            request->candidateTailBlockCount,
            request->receivedTailBlockCount);
        if (!reported) {
            NAVI_KERNEL_LOG(ERROR,
                            "pending cache full snapshot fallback failed, "
                            "collectorId [%s], reason [%s], rejectedRevision [%lu]",
                            request->collectorId.c_str(), request->reason.c_str(),
                            request->rejectedRevision);
        }
    }
    return navi::EC_NONE;
}

REGISTER_KERNEL(AgentPendingCacheSyncKernel);

}
