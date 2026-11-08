#include "aios/navi_sched/collector/PPCacheMergeR.h"
#include "aios/navi_sched/collector/PPCacheShardUpdateData.h"
#include "navi/engine/Kernel.h"

namespace navi_sched {

class PPCacheMergeKernel : public navi::Kernel {
public:
    ~PPCacheMergeKernel() override
    {
        if (_connected) _merge->disconnect(_shardId, _generation);
    }
    void def(navi::KernelDefBuilder &builder) const override
    {
        builder.name("navi_sched.pp_cache_merge_k")
            .input("update", PP_CACHE_SHARD_UPDATE_TYPE_ID)
            .output("finished", PP_CACHE_SHARD_UPDATE_TYPE_ID);
    }
    bool config(navi::KernelConfigContext &ctx) override
    {
        NAVI_JSONIZE(ctx, "pp_group", _group);
        NAVI_JSONIZE(ctx, "shard_id", _shardId);
        NAVI_JSONIZE(ctx, "shard_count", _shardCount);
        NAVI_JSONIZE(ctx, "stream_generation", _generation);
        return !_group.empty() && _shardCount > 0 && _shardId < _shardCount &&
               _generation > 0;
    }
    navi::ErrorCode init(navi::KernelInitContext &) override
    {
        _connected = _merge->connect(_group, _shardId, _shardCount, _generation).accepted;
        return _connected ? navi::EC_NONE : navi::EC_INIT_SUB_GRAPH;
    }
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override
    {
        navi::DataPtr data;
        bool eof = false;
        if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof))
            return navi::EC_NONE;
        if (data) {
            const auto update = std::dynamic_pointer_cast<PPCacheShardUpdateData>(data);
            if (!update || !update->valid() || update->update.shardId != _shardId ||
                update->update.streamGeneration != _generation)
                return navi::EC_ABORT;
            const auto changes = _merge->apply(update->update);
            if (!changes.accepted && changes.needsFullSnapshot) return navi::EC_ABORT;
        }
        if (eof) {
            _merge->disconnect(_shardId, _generation);
            _connected = false;
            return ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, true)
                       ? navi::EC_NONE
                       : navi::EC_ABORT;
        }
        return navi::EC_NONE;
    }

private:
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON(PPCacheMergeR, _merge);
    std::string _group;
    uint32_t _shardId = 0;
    uint32_t _shardCount = 0;
    uint64_t _generation = 0;
    bool _connected = false;
};

REGISTER_KERNEL(PPCacheMergeKernel);

}
