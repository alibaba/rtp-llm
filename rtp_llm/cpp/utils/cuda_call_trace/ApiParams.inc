// Included inside the CUDA-only implementation. All pointees copied here are
// CUDA API argument/output storage valid during this callback. Never retain
// functionParams, invoke CUDA, or dereference tensor/device memory here.
bool apiName(const char* actual, const char* base) noexcept {
    const size_t n = std::strlen(base);
    return std::strncmp(actual, base, n) == 0
           && (actual[n] == '\0' || std::strcmp(actual + n, "_ptsz") == 0 || std::strcmp(actual + n, "_ptds") == 0);
}

bool startsWith(const char* name, const char* prefix) noexcept {
    return std::strncmp(name, prefix, std::strlen(prefix)) == 0;
}

bool isLaunchApi(const char* name) noexcept {
    return name && !startsWith(name, "cuLaunchHostFunc") && !startsWith(name, "cudaLaunchHostFunc")
           && (startsWith(name, "cuLaunch") || startsWith(name, "cudaLaunch") || startsWith(name, "cuGraphLaunch")
               || startsWith(name, "cudaGraphLaunch"));
}

// Applied both when enabling individual CUPTI callback IDs and before building
// a callback Record. Deliberately exclude polling/getter APIs; their NOT_READY
// results are not CUDA failures and previously drowned out dependency records.
// CUPTI callback names may contain ABI suffixes, so selection uses an explicit
// operation-prefix allowlist while parameter decoding still uses exact names.
bool trackedApi(const char* name) noexcept {
    if (!name) {
        return false;
    }
    // Some operation families include introspection CBIDs (e.g.
    // cuGraphExecGetId); GetCallbackName can name them even when CUPTI cannot
    // subscribe to them. They are outside the dependency/mutation contract.
    if (std::strstr(name, "Get") || std::strstr(name, "Query")) {
        return false;
    }
    constexpr const char* prefixes[] = {"cuMemcpy",
                                        "cudaMemcpy",
                                        "cuMemset",
                                        "cudaMemset",
                                        "cuMemAlloc",
                                        "cuMemFree",
                                        "cuMemHostAlloc",
                                        "cuMemHostRegister",
                                        "cuMemHostUnregister",
                                        "cudaMalloc",
                                        "cudaFree",
                                        "cudaHostAlloc",
                                        "cudaHostRegister",
                                        "cudaHostUnregister",
                                        "cuMemCreate",
                                        "cuMemRelease",
                                        "cuMemMap",
                                        "cuMemUnmap",
                                        "cuMemAddressReserve",
                                        "cuMemAddressFree",
                                        "cuStreamCreate",
                                        "cuStreamDestroy",
                                        "cuStreamSynchronize",
                                        "cuStreamWaitEvent",
                                        "cuStreamAddCallback",
                                        "cuStreamBeginCapture",
                                        "cuStreamEndCapture",
                                        "cuStreamUpdateCaptureDependencies",
                                        "cuStreamWaitValue",
                                        "cuStreamWriteValue",
                                        "cuStreamBatchMemOp",
                                        "cuLaunchHostFunc",
                                        "cudaStreamCreate",
                                        "cudaStreamDestroy",
                                        "cudaStreamSynchronize",
                                        "cudaStreamWaitEvent",
                                        "cudaStreamAddCallback",
                                        "cudaStreamBeginCapture",
                                        "cudaStreamEndCapture",
                                        "cudaStreamUpdateCaptureDependencies",
                                        "cudaStreamWaitValue",
                                        "cudaStreamWriteValue",
                                        "cudaStreamBatchMemOp",
                                        "cudaLaunchHostFunc",
                                        "cuWaitExternalSemaphores",
                                        "cuSignalExternalSemaphores",
                                        "cudaWaitExternalSemaphores",
                                        "cudaSignalExternalSemaphores",
                                        "cuEventCreate",
                                        "cuEventDestroy",
                                        "cuEventRecord",
                                        "cuEventSynchronize",
                                        "cudaEventCreate",
                                        "cudaEventDestroy",
                                        "cudaEventRecord",
                                        "cudaEventSynchronize",
                                        "cuCtxSynchronize",
                                        "cudaDeviceSynchronize",
                                        "cuCtxSetCurrent",
                                        "cuCtxPushCurrent",
                                        "cuCtxPopCurrent",
                                        "cuCtxCreate",
                                        "cuCtxDestroy",
                                        "cudaSetDevice",
                                        "cudaDeviceReset",
                                        "cuGraphCreate",
                                        "cuGraphDestroy",
                                        "cuGraphInstantiate",
                                        "cuGraphUpload",
                                        "cuGraphExec",
                                        "cuGraphAdd",
                                        "cuGraphRemove",
                                        "cudaGraphCreate",
                                        "cudaGraphDestroy",
                                        "cudaGraphInstantiate",
                                        "cudaGraphUpload",
                                        "cudaGraphExec",
                                        "cudaGraphAdd",
                                        "cudaGraphRemove"};
    if (isLaunchApi(name)) {
        return true;
    }
    for (const char* prefix : prefixes) {
        if (startsWith(name, prefix)) {
            return true;
        }
    }
    return false;
}

// CUPTI 13.2 names these newer driver APIs even on drivers where subscribing
// returns INVALID_PARAMETER. Only these observed optional additions may be
// skipped, and every skip is retained as a coverage gap. All other failures,
// including the established copy/event/synchronization/launch APIs, fail closed.
bool optionalNewCallback(const char* name) noexcept {
    constexpr const char* names[] = {"cuStreamBeginCaptureToCig",
                                     "cuStreamEndCaptureToCig",
                                     "cuGraphCreateWithArguments",
                                     "cuGraphLaunchWithArguments",
                                     "cuLaunchHostFunc_v2",
                                     "cuMemcpyWithAttributesAsync",
                                     "cuMemcpy3DWithAttributesAsync"};
    for (const auto* candidate : names) {
        if (apiName(name, candidate))
            return true;
    }
    return false;
}

bool requiredCallback(CUpti_CallbackDomain domain, uint32_t id) noexcept {
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        const uint32_t required[] = {CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2,
                                     CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2,
                                     CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
                                     CUPTI_DRIVER_TRACE_CBID_cuEventRecord,
                                     CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent,
                                     CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize,
                                     CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2,
                                     CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize};
        for (auto required_id : required)
            if (id == required_id)
                return true;
    } else {
        const uint32_t required[] = {CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020,
                                     CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020,
                                     CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,
                                     CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_v3020,
                                     CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_v3020,
                                     CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020,
                                     CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020,
                                     CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020};
        for (auto required_id : required)
            if (id == required_id)
                return true;
    }
    return false;
}

template<typename T>
uint64_t address(T* pointer) noexcept {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pointer));
}

void decodeParams(Record& r, const CUpti_CallbackData& cb, bool output) noexcept {
    const char* name = cb.functionName;
    const void* raw  = cb.functionParams;
    if (name == nullptr) {
        return;
    }
    if (apiName(name, "cudaDeviceSynchronize") || apiName(name, "cuCtxSynchronize")) {
        r.params_known = true;
        return;
    }
    if (raw == nullptr) {
        return;
    }
    const bool success = output && r.result == 0;
#define PARAM(API, TYPE, BODY)                                                                                         \
    if (apiName(name, API)) {                                                                                          \
        TYPE p{};                                                                                                      \
        std::memcpy(&p, raw, sizeof(p));                                                                               \
        r.params_known = true;                                                                                         \
        BODY;                                                                                                          \
        return;                                                                                                        \
    }
#define HANDLE_OUTPUT(P, FIELD)                                                                                        \
    if (success && (P) != nullptr) {                                                                                   \
        r.FIELD = address(*(P));                                                                                       \
    }

    if (r.domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        PARAM("cuMemcpyHtoDAsync_v2", cuMemcpyHtoDAsync_v2_params, r.dst = p.dstDevice; r.src = address(p.srcHost);
              r.bytes  = p.ByteCount;
              r.stream = address(p.hStream))
        PARAM("cuMemcpyDtoHAsync_v2", cuMemcpyDtoHAsync_v2_params, r.dst = address(p.dstHost); r.src = p.srcDevice;
              r.bytes  = p.ByteCount;
              r.stream = address(p.hStream))
        PARAM("cuMemcpyDtoDAsync_v2", cuMemcpyDtoDAsync_v2_params, r.dst = p.dstDevice; r.src = p.srcDevice;
              r.bytes  = p.ByteCount;
              r.stream = address(p.hStream))
        PARAM("cuMemcpyHtoD_v2", cuMemcpyHtoD_v2_params, r.dst = p.dstDevice; r.src = address(p.srcHost);
              r.bytes = p.ByteCount)
        PARAM("cuMemcpyDtoH_v2", cuMemcpyDtoH_v2_params, r.dst = address(p.dstHost); r.src = p.srcDevice;
              r.bytes = p.ByteCount)
        PARAM("cuMemcpyDtoD_v2", cuMemcpyDtoD_v2_params, r.dst = p.dstDevice; r.src = p.srcDevice;
              r.bytes = p.ByteCount)
        PARAM("cuMemcpyAsync", cuMemcpyAsync_params, r.dst = p.dst; r.src = p.src; r.bytes = p.ByteCount;
              r.stream = address(p.hStream))
        PARAM("cuMemcpy", cuMemcpy_params, r.dst = p.dst; r.src = p.src; r.bytes = p.ByteCount)
        PARAM(
            "cuMemcpy2DAsync_v2", cuMemcpy2DAsync_v2_params, r.stream = address(p.hStream); if (p.pCopy) {
                r.src = p.pCopy->srcMemoryType == CU_MEMORYTYPE_HOST ? address(p.pCopy->srcHost) : p.pCopy->srcDevice;
                r.dst = p.pCopy->dstMemoryType == CU_MEMORYTYPE_HOST ? address(p.pCopy->dstHost) : p.pCopy->dstDevice;
                r.value[0] = p.pCopy->WidthInBytes;
                r.value[1] = p.pCopy->Height;
                r.value[2] = p.pCopy->srcPitch;
                r.value[3] = p.pCopy->dstPitch;
            })
        PARAM("cuStreamCreate", cuStreamCreate_params, r.value[0] = p.Flags; HANDLE_OUTPUT(p.phStream, stream))
        PARAM("cuStreamCreateWithPriority", cuStreamCreateWithPriority_params, r.value[0] = p.flags;
              r.value[1] = static_cast<uint64_t>(static_cast<int64_t>(p.priority));
              HANDLE_OUTPUT(p.phStream, stream))
        PARAM("cuStreamDestroy_v2", cuStreamDestroy_v2_params, r.stream = address(p.hStream))
        PARAM("cuStreamSynchronize", cuStreamSynchronize_params, r.stream = address(p.hStream))
        PARAM("cuStreamQuery", cuStreamQuery_params, r.stream = address(p.hStream))
        PARAM("cuStreamWaitEvent", cuStreamWaitEvent_params, r.stream = address(p.hStream); r.event = address(p.hEvent);
              r.value[0] = p.Flags)
        PARAM("cuEventCreate", cuEventCreate_params, r.value[0] = p.Flags; HANDLE_OUTPUT(p.phEvent, event))
        PARAM("cuEventDestroy_v2", cuEventDestroy_v2_params, r.event = address(p.hEvent))
        PARAM("cuEventRecord", cuEventRecord_params, r.event = address(p.hEvent); r.stream = address(p.hStream))
        PARAM("cuEventRecordWithFlags", cuEventRecordWithFlags_params, r.event = address(p.hEvent);
              r.stream   = address(p.hStream);
              r.value[0] = p.flags)
        PARAM("cuEventSynchronize", cuEventSynchronize_params, r.event = address(p.hEvent))
        PARAM("cuEventQuery", cuEventQuery_params, r.event = address(p.hEvent))
        PARAM("cuLaunchKernel", cuLaunchKernel_params, r.src = address(p.f); r.stream = address(p.hStream);
              r.bytes    = p.sharedMemBytes;
              r.value[0] = p.gridDimX;
              r.value[1] = p.gridDimY;
              r.value[2] = p.gridDimZ;
              r.value[3] = p.blockDimX;
              r.value[4] = p.blockDimY;
              r.value[5] = p.blockDimZ)
        PARAM(
            "cuLaunchKernelEx", cuLaunchKernelEx_params, r.src = address(p.f); if (p.config) {
                r.stream   = address(p.config->hStream);
                r.bytes    = p.config->sharedMemBytes;
                r.value[0] = p.config->gridDimX;
                r.value[1] = p.config->gridDimY;
                r.value[2] = p.config->gridDimZ;
                r.value[3] = p.config->blockDimX;
                r.value[4] = p.config->blockDimY;
                r.value[5] = p.config->blockDimZ;
                r.value[6] = p.config->numAttrs;
            })
        PARAM("cuLaunchCooperativeKernel", cuLaunchCooperativeKernel_params, r.src = address(p.f);
              r.stream   = address(p.hStream);
              r.bytes    = p.sharedMemBytes;
              r.value[0] = p.gridDimX;
              r.value[1] = p.gridDimY;
              r.value[2] = p.gridDimZ;
              r.value[3] = p.blockDimX;
              r.value[4] = p.blockDimY;
              r.value[5] = p.blockDimZ)
        PARAM("cuGraphLaunch", cuGraphLaunch_params, r.src = address(p.hGraph); r.stream = address(p.hStream))
        PARAM("cuLaunchHostFunc", cuLaunchHostFunc_params, r.stream = address(p.hStream);
              r.src = reinterpret_cast<uintptr_t>(p.fn);
              r.dst = address(p.userData))
        PARAM("cuStreamAddCallback", cuStreamAddCallback_params, r.stream = address(p.hStream);
              r.src      = reinterpret_cast<uintptr_t>(p.callback);
              r.dst      = address(p.userData);
              r.value[0] = p.flags)
        PARAM("cuMemAlloc_v2", cuMemAlloc_v2_params, r.bytes = p.bytesize; if (success && p.dptr) { r.dst = *p.dptr; })
        PARAM(
            "cuMemAllocAsync", cuMemAllocAsync_params, r.bytes = p.bytesize; r.stream = address(p.hStream);
            if (success && p.dptr) { r.dst = *p.dptr; })
        PARAM("cuMemFree_v2", cuMemFree_v2_params, r.src = p.dptr)
        PARAM("cuMemFreeAsync", cuMemFreeAsync_params, r.src = p.dptr; r.stream = address(p.hStream))
        PARAM("cuMemHostAlloc", cuMemHostAlloc_params, r.bytes = p.bytesize; r.value[0] = p.Flags;
              HANDLE_OUTPUT(p.pp, dst))
        PARAM("cuMemAllocHost_v2", cuMemAllocHost_v2_params, r.bytes = p.bytesize; HANDLE_OUTPUT(p.pp, dst))
        PARAM("cuMemFreeHost", cuMemFreeHost_params, r.src = address(p.p))
        PARAM("cuMemHostRegister_v2", cuMemHostRegister_v2_params, r.src = address(p.p); r.bytes = p.bytesize;
              r.value[0] = p.Flags)
        PARAM("cuMemHostUnregister", cuMemHostUnregister_params, r.src = address(p.p))
        PARAM("cuMemsetD8Async", cuMemsetD8Async_params, r.dst = p.dstDevice; r.bytes = p.N; r.value[0] = p.uc;
              r.stream = address(p.hStream))
        PARAM("cuCtxSetCurrent", cuCtxSetCurrent_params, r.value[0] = address(p.ctx))
        PARAM("cuCtxPushCurrent_v2", cuCtxPushCurrent_v2_params, r.value[0] = address(p.ctx))
    } else if (r.domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        PARAM("cudaMemcpyAsync", cudaMemcpyAsync_v3020_params, r.src = address(p.src); r.dst = address(p.dst);
              r.bytes    = p.count;
              r.stream   = address(p.stream);
              r.value[0] = p.kind)
        PARAM("cudaMemcpy", cudaMemcpy_v3020_params, r.src = address(p.src); r.dst = address(p.dst); r.bytes = p.count;
              r.value[0] = p.kind)
        PARAM("cudaMemsetAsync", cudaMemsetAsync_v3020_params, r.dst = address(p.devPtr); r.bytes = p.count;
              r.value[0] = p.value;
              r.stream   = address(p.stream))
        PARAM("cudaStreamCreate", cudaStreamCreate_v3020_params, HANDLE_OUTPUT(p.pStream, stream))
        PARAM("cudaStreamCreateWithFlags", cudaStreamCreateWithFlags_v5000_params, r.value[0] = p.flags;
              HANDLE_OUTPUT(p.pStream, stream))
        PARAM("cudaStreamCreateWithPriority", cudaStreamCreateWithPriority_v5050_params, r.value[0] = p.flags;
              r.value[1] = static_cast<uint64_t>(static_cast<int64_t>(p.priority));
              HANDLE_OUTPUT(p.pStream, stream))
        PARAM("cudaStreamDestroy", cudaStreamDestroy_v5050_params, r.stream = address(p.stream))
        PARAM("cudaStreamSynchronize", cudaStreamSynchronize_v3020_params, r.stream = address(p.stream))
        PARAM("cudaStreamQuery", cudaStreamQuery_v3020_params, r.stream = address(p.stream))
        PARAM("cudaStreamWaitEvent", cudaStreamWaitEvent_v3020_params, r.stream = address(p.stream);
              r.event    = address(p.event);
              r.value[0] = p.flags)
        PARAM("cudaEventCreate", cudaEventCreate_v3020_params, HANDLE_OUTPUT(p.event, event))
        PARAM("cudaEventCreateWithFlags", cudaEventCreateWithFlags_v3020_params, r.value[0] = p.flags;
              HANDLE_OUTPUT(p.event, event))
        PARAM("cudaEventDestroy", cudaEventDestroy_v3020_params, r.event = address(p.event))
        PARAM("cudaEventRecord", cudaEventRecord_v3020_params, r.event = address(p.event); r.stream = address(p.stream))
        PARAM("cudaEventRecordWithFlags", cudaEventRecordWithFlags_v11010_params, r.event = address(p.event);
              r.stream   = address(p.stream);
              r.value[0] = p.flags)
        PARAM("cudaEventSynchronize", cudaEventSynchronize_v3020_params, r.event = address(p.event))
        PARAM("cudaEventQuery", cudaEventQuery_v3020_params, r.event = address(p.event))
        PARAM("cudaLaunchKernel", cudaLaunchKernel_v7000_params, r.src = address(p.func); r.stream = address(p.stream);
              r.bytes    = p.sharedMem;
              r.value[0] = p.gridDim.x;
              r.value[1] = p.gridDim.y;
              r.value[2] = p.gridDim.z;
              r.value[3] = p.blockDim.x;
              r.value[4] = p.blockDim.y;
              r.value[5] = p.blockDim.z)
        PARAM(
            "cudaLaunchKernelExC", cudaLaunchKernelExC_v11060_params, r.src = address(p.func); if (p.config) {
                r.stream   = address(p.config->stream);
                r.bytes    = p.config->dynamicSmemBytes;
                r.value[0] = p.config->gridDim.x;
                r.value[1] = p.config->gridDim.y;
                r.value[2] = p.config->gridDim.z;
                r.value[3] = p.config->blockDim.x;
                r.value[4] = p.config->blockDim.y;
                r.value[5] = p.config->blockDim.z;
                r.value[6] = p.config->numAttrs;
            })
        PARAM("cudaGraphLaunch", cudaGraphLaunch_v10000_params, r.src = address(p.graphExec);
              r.stream = address(p.stream))
        PARAM("cudaLaunchHostFunc", cudaLaunchHostFunc_v10000_params, r.stream = address(p.stream);
              r.src = reinterpret_cast<uintptr_t>(p.fn);
              r.dst = address(p.userData))
        PARAM("cudaStreamAddCallback", cudaStreamAddCallback_v5000_params, r.stream = address(p.stream);
              r.src      = reinterpret_cast<uintptr_t>(p.callback);
              r.dst      = address(p.userData);
              r.value[0] = p.flags)
        PARAM("cudaMalloc", cudaMalloc_v3020_params, r.bytes = p.size; HANDLE_OUTPUT(p.devPtr, dst))
        PARAM("cudaMallocAsync", cudaMallocAsync_v11020_params, r.bytes = p.size; r.stream = address(p.hStream);
              HANDLE_OUTPUT(p.devPtr, dst))
        PARAM("cudaFree", cudaFree_v3020_params, r.src = address(p.devPtr))
        PARAM("cudaFreeAsync", cudaFreeAsync_v11020_params, r.src = address(p.devPtr); r.stream = address(p.hStream))
        PARAM("cudaMallocHost", cudaMallocHost_v3020_params, r.bytes = p.size; HANDLE_OUTPUT(p.ptr, dst))
        PARAM("cudaHostAlloc", cudaHostAlloc_v3020_params, r.bytes = p.size; r.value[0] = p.flags;
              HANDLE_OUTPUT(p.pHost, dst))
        PARAM("cudaFreeHost", cudaFreeHost_v3020_params, r.src = address(p.ptr))
        PARAM("cudaHostRegister", cudaHostRegister_v4000_params, r.src = address(p.ptr); r.bytes = p.size;
              r.value[0] = p.flags)
        PARAM("cudaHostUnregister", cudaHostUnregister_v4000_params, r.src = address(p.ptr))
        PARAM("cudaSetDevice", cudaSetDevice_v3020_params, r.value[0] = p.device)
    }
    // Selected APIs without a decoder retain their names/correlations, with
    // explicitly unknown parameters rather than fabricated zero arguments.
#undef HANDLE_OUTPUT
#undef PARAM
}
