#include "navi/engine/NaviKernel.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "navi/engine/NaviThreadPool.h"

namespace navi {

NaviKernel::NaviKernel() {
}

NaviKernel::~NaviKernel() {
}

void NaviKernel::def(KernelDefBuilder &builder) const {
    builder.name(NAVI_KERNEL_NAME)
        .output(NAVI_KERNEL_OUTPUT_PORT, "");
}

bool NaviKernel::config(KernelConfigContext &ctx) {
    return true;
}

ErrorCode NaviKernel::init(KernelInitContext &ctx) {
    return _naviR->Init();
}

ErrorCode NaviKernel::compute(KernelComputeContext &ctx) {
    auto currentThreadPool = current_thread_pool;
    current_thread_pool = nullptr;
    auto threadPoolSignalCacheCount = thread_pool_signal_cache_count;
    thread_pool_signal_cache_count = INVALID_SIGNAL_CACHE_COUNT;
    auto inlineItemQueue = inline_item_queue;
    auto tailItemQueue = tail_item_queue;
    inline_item_queue = nullptr;
    tail_item_queue = nullptr;
    auto ec = _naviR->update();
    if (EC_NONE == ec) {
        if (!ctx.setOutput(0, nullptr, true)) {
            inline_item_queue = inlineItemQueue;
            tail_item_queue = tailItemQueue;
            thread_pool_signal_cache_count = threadPoolSignalCacheCount;
            current_thread_pool = currentThreadPool;
            return EC_ABORT;
        }
    }
    inline_item_queue = inlineItemQueue;
    tail_item_queue = tailItemQueue;
    thread_pool_signal_cache_count = threadPoolSignalCacheCount;
    current_thread_pool = currentThreadPool;
    return ec;
}

REGISTER_BUILDIN_KERNEL(NaviKernel);

}
