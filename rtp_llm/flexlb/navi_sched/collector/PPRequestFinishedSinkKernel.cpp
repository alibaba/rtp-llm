#include "aios/navi_sched/collector/PPCacheShardReportR.h"
#include "navi/engine/Kernel.h"

namespace navi_sched {

class PPRequestFinishedSinkKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override
    {
        builder.name("navi_sched.pp_request_finished_sink_k")
            .input("update", PP_CACHE_SHARD_UPDATE_TYPE_ID)
            .output("finished", PP_CACHE_SHARD_UPDATE_TYPE_ID);
    }
    bool config(navi::KernelConfigContext &ctx) override
    {
        NAVI_JSONIZE(ctx, "pp_group", _group);
        NAVI_JSONIZE(ctx, "shard_id", _shardId);
        NAVI_JSONIZE(ctx, "shard_count", _shardCount);
        NAVI_JSONIZE(ctx, "collector_id", _collectorId);
        NAVI_JSONIZE(ctx, "stream_generation", _generation);
        return !_group.empty() && !_collectorId.empty() && _shardCount > 1 &&
               _shardId < _shardCount && _generation > 0;
    }
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override
    {
        navi::DataPtr data;
        bool eof = false;
        if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof))
            return navi::EC_NONE;
        if (data) {
            auto update = std::dynamic_pointer_cast<PPCacheShardUpdateData>(data);
            if (!update || update->update.shardId != _shardId ||
                update->update.streamGeneration != _generation ||
                !_report->publishRequestFinished(_group, _shardCount, _collectorId,
                                                 *update))
                return navi::EC_ABORT;
        }
        if (eof)
            return ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, true)
                       ? navi::EC_NONE
                       : navi::EC_ABORT;
        return navi::EC_NONE;
    }

private:
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON(PPCacheShardReportR, _report);
    std::string _group;
    std::string _collectorId;
    uint32_t _shardId = 0;
    uint32_t _shardCount = 0;
    uint64_t _generation = 0;
};

REGISTER_KERNEL(PPRequestFinishedSinkKernel);

}
