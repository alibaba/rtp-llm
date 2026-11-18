#include "aios/navi_sched/data/V6dCacheTouchData.h"
#include "aios/navi_sched/resource/AgentRequestR.h"
#include "navi/engine/Kernel.h"

namespace navi_sched {

class CacheTouchSourceKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override
    {
        builder.name("navi_sched.client_cache_touch_source_k")
            .output("touch", V6D_CACHE_TOUCH_TYPE_ID);
    }
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override
    {
        if (!_request) {
            ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, true);
            return navi::EC_NONE;
        }
        auto request = std::make_shared<V6dCacheTouchData>();
        request->version = _request->version;
        request->requestId = _request->requestId;
        request->blockHashes = _request->blockHashes;
        ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), request, true);
        return navi::EC_NONE;
    }
private:
    KERNEL_DEPEND_DECLARE();
    KERNEL_NAMED_DATA(V6dCacheTouchData, _request, V6D_CACHE_TOUCH_DATA_NAME);
};

class AgentCacheTouchKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override
    {
        builder.name("navi_sched.agent_cache_touch_k")
            .input("request", V6D_CACHE_TOUCH_TYPE_ID);
    }
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override
    {
        navi::DataPtr data;
        bool eof = false;
        if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
            return navi::EC_NONE;
        }
        const auto request = std::dynamic_pointer_cast<V6dCacheTouchData>(data);
        if (request && request->version == NAVI_SCHED_DATA_VERSION && _requests) {
            _requests->touchV6dCache(request->requestId, request->blockHashes);
        }
        return navi::EC_NONE;
    }
private:
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON_FALSE(AgentRequestR, _requests);
};

REGISTER_KERNEL(CacheTouchSourceKernel);
REGISTER_KERNEL(AgentCacheTouchKernel);

}
