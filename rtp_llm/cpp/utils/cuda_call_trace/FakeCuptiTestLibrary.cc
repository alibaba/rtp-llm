// CPU-only contract test double. This file is linked into a test DSO only and
// never into the inference library. No CUDA context/device work is performed.
#include <cupti.h>

#include <atomic>
#include <chrono>
#include <thread>

namespace {
CUpti_CallbackFunc    callback = nullptr;
void*                 userdata = nullptr;
std::atomic<bool>     flush_entered{false}, release_flush{false};
std::atomic<unsigned> version_calls{0};
std::atomic<bool>     fail_required{false};
thread_local uint64_t correlation_data = 0;
}  // namespace

extern "C" CUptiResult CUPTIAPI cuptiGetVersion(uint32_t* version) {
    ++version_calls;
    *version = CUPTI_API_VERSION;
    return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiGetTimestamp(uint64_t* timestamp) {
    *timestamp =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiSubscribe(CUpti_SubscriberHandle* subscriber,
                                               CUpti_CallbackFunc      function,
                                               void*                   data) {
    callback    = function;
    userdata    = data;
    *subscriber = reinterpret_cast<CUpti_SubscriberHandle>(uintptr_t(1));
    return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiUnsubscribe(CUpti_SubscriberHandle) {
    callback = nullptr;
    return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiEnableDomain(uint32_t, CUpti_SubscriberHandle, CUpti_CallbackDomain) {
    return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiEnableCallback(uint32_t,
                                                    CUpti_SubscriberHandle,
                                                    CUpti_CallbackDomain domain,
                                                    CUpti_CallbackId     id) {
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API && id == CUPTI_DRIVER_TRACE_CBID_cuGraphExecGetFlags) {
        return CUPTI_ERROR_INVALID_PARAMETER;  // Named does not imply subscribable; this getter must be excluded.
    }
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API && id == CUPTI_DRIVER_TRACE_CBID_cuGraphCreate) {
        return CUPTI_ERROR_INVALID_PARAMETER;  // Test-only stable ID names an optional newer API below.
    }
    if (fail_required.load() && domain == CUPTI_CB_DOMAIN_DRIVER_API
        && id == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2) {
        return CUPTI_ERROR_INVALID_PARAMETER;
    }
    return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiGetCallbackName(CUpti_CallbackDomain domain,
                                                     CUpti_CallbackId     id,
                                                     const char**         name) {
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
#define DRIVER(NAME)                                                                                                   \
    case CUPTI_DRIVER_TRACE_CBID_##NAME:                                                                               \
        *name = #NAME;                                                                                                 \
        return CUPTI_SUCCESS
        switch (id) {
            DRIVER(cuMemcpyHtoDAsync_v2);
            DRIVER(cuMemcpyDtoHAsync_v2);
            DRIVER(cuLaunchKernel);
            DRIVER(cuEventRecord);
            DRIVER(cuStreamWaitEvent);
            DRIVER(cuStreamSynchronize);
            DRIVER(cuMemAlloc_v2);
            DRIVER(cuCtxSynchronize);
            DRIVER(cuGraphExecGetFlags);
            case CUPTI_DRIVER_TRACE_CBID_cuGraphCreate:
                *name = "cuGraphCreateWithArguments";
                return CUPTI_SUCCESS;
        }
#undef DRIVER
    } else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
#define RUNTIME(NAME)                                                                                                  \
    case CUPTI_RUNTIME_TRACE_CBID_##NAME:                                                                              \
        *name = #NAME;                                                                                                 \
        return CUPTI_SUCCESS
        switch (id) {
            RUNTIME(cudaMemcpy_v3020);
            RUNTIME(cudaMemcpyAsync_v3020);
            RUNTIME(cudaLaunchKernel_v7000);
            RUNTIME(cudaEventRecord_v3020);
            RUNTIME(cudaStreamWaitEvent_v3020);
            RUNTIME(cudaStreamSynchronize_v3020);
            RUNTIME(cudaMalloc_v3020);
            RUNTIME(cudaDeviceSynchronize_v3020);
        }
#undef RUNTIME
    }
    return CUPTI_ERROR_INVALID_PARAMETER;
}
extern "C" CUptiResult CUPTIAPI cuptiActivityRegisterCallbacks(CUpti_BuffersCallbackRequestFunc,
                                                               CUpti_BuffersCallbackCompleteFunc) {
    return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiActivityEnable(CUpti_ActivityKind) {
    return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiActivityEnableAndDump(CUpti_ActivityKind) {
    return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiActivityDisable(CUpti_ActivityKind) {
    return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiActivityGetNextRecord(uint8_t*, size_t, CUpti_Activity**) {
    return CUPTI_ERROR_MAX_LIMIT_REACHED;
}
extern "C" CUptiResult CUPTIAPI cuptiActivityGetNumDroppedRecords(CUcontext, uint32_t, size_t* dropped) {
    *dropped = 0;
    return CUPTI_SUCCESS;
}
extern "C" CUptiResult CUPTIAPI cuptiActivityFlushAll(uint32_t) {
    flush_entered.store(true, std::memory_order_release);
    while (!release_flush.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return CUPTI_SUCCESS;
}

extern "C" unsigned fakeVersionCalls() {
    return version_calls.load();
}
extern "C" void fakeFailRequiredCallback() {
    fail_required.store(true);
}
extern "C" bool fakeFlushEntered() {
    return flush_entered.load(std::memory_order_acquire);
}
extern "C" void fakeReleaseFlush() {
    release_flush.store(true, std::memory_order_release);
}
extern "C" void fakeApi(bool exiting) {
    cuMemcpyDtoHAsync_v2_params params{};
    params.srcDevice          = 0x100000;
    params.dstHost            = reinterpret_cast<void*>(uintptr_t(0x200000));
    params.ByteCount          = 4759552;
    params.hStream            = reinterpret_cast<CUstream>(uintptr_t(0x300000));
    CUresult           result = CUDA_SUCCESS;
    CUpti_CallbackData data{};
    data.callbackSite        = exiting ? CUPTI_API_EXIT : CUPTI_API_ENTER;
    data.functionName        = "cuMemcpyDtoHAsync_v2";
    data.functionParams      = &params;
    data.functionReturnValue = exiting ? &result : nullptr;
    data.context             = reinterpret_cast<CUcontext>(uintptr_t(0x400000));
    data.contextUid          = 42;
    data.correlationId       = 900;
    data.correlationData     = &correlation_data;
    callback(userdata, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2, &data);
}

extern "C" void fakeFlood(unsigned count, bool queries) {
    for (unsigned i = 0; i != count; ++i) {
        uint64_t           storage = 0;
        CUresult           result  = CUDA_SUCCESS;
        CUpti_CallbackData data{};
        data.functionName        = queries ? (i % 2 ? "cuCtxGetDevice_v2" : "cuGraphExecGetId") : "cuLaunchKernel";
        data.functionReturnValue = &result;
        data.correlationId       = i + 1000;
        data.correlationData     = &storage;
        data.callbackSite        = CUPTI_API_ENTER;
        callback(userdata, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, &data);
        data.callbackSite = CUPTI_API_EXIT;
        callback(userdata, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, &data);
    }
}

extern "C" void fakeUntrackedApi() {
    CUpti_CallbackData data{};
    data.functionName  = "cuCtxSynchronize";
    data.correlationId = 800;
    data.callbackSite  = CUPTI_API_ENTER;
    callback(userdata, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize, &data);
    data.callbackSite = CUPTI_API_EXIT;
    callback(userdata, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize, &data);
}
