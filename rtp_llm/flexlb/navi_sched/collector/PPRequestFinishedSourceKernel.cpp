#include "aios/navi_sched/collector/PPRequestFinishedR.h"
#include "navi/engine/Kernel.h"

namespace navi_sched {

class PPRequestFinishedSourceKernel : public navi::Kernel {
public:
    ~PPRequestFinishedSourceKernel() override
    {
        if (_pipe) _finished->unsubscribe(_shardId, _generation, _pipe);
    }
    void def(navi::KernelDefBuilder &builder) const override
    {
        builder.name("navi_sched.pp_request_finished_source_k")
            .output("update", PP_CACHE_SHARD_UPDATE_TYPE_ID);
    }
    bool config(navi::KernelConfigContext &ctx) override
    {
        NAVI_JSONIZE(ctx, "pp_group", _group);
        NAVI_JSONIZE(ctx, "shard_id", _shardId);
        NAVI_JSONIZE(ctx, "shard_count", _shardCount);
        NAVI_JSONIZE(ctx, "stream_generation", _generation);
        return !_group.empty() && _shardCount > 1 && _shardId < _shardCount &&
               _generation > 0;
    }
    navi::ErrorCode init(navi::KernelInitContext &ctx) override
    {
        _pipe = ctx.createAsyncPipe();
        return _pipe && _finished->subscribe(_group, _shardId, _shardCount, _generation,
                                             _pipe)
                   ? navi::EC_NONE
                   : navi::EC_INIT_SUB_GRAPH;
    }
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override
    {
        navi::DataPtr data;
        bool eof = false;
        if (!_pipe->getData(data, eof)) return navi::EC_NONE;
        // Both directions share one PP connection. Ending only this output
        // would leave the peer's input waiting and retain the Collector graph.
        if (eof) return navi::EC_ABORT;
        auto update = std::dynamic_pointer_cast<PPCacheShardUpdateData>(data);
        if (data && (!update || !update->valid())) return navi::EC_ABORT;
        return ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), update, eof)
                   ? navi::EC_NONE
                   : navi::EC_ABORT;
    }

private:
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON(PPRequestFinishedR, _finished);
    navi::AsyncPipePtr _pipe;
    std::string _group;
    uint32_t _shardId = 0;
    uint32_t _shardCount = 0;
    uint64_t _generation = 0;
};

REGISTER_KERNEL(PPRequestFinishedSourceKernel);

}
