#include <cupti.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <mutex>
#include <string>

#include <dlfcn.h>
#include <unistd.h>

namespace {

struct Allocation {
    uint64_t    bytes;
    const char* api;
    bool        registered;
};

struct ProbeState {
    std::mutex                      init_mutex;
    std::mutex                      ledger_mutex;
    std::map<uintptr_t, Allocation>  allocations;
    uint64_t                        active_bytes          = 0;
    uint64_t                        total_allocated_bytes = 0;
    uint64_t                        host_vmm_calls              = 0;
    uint64_t                        host_vmm_created_bytes      = 0;
    int                             last_host_vmm_location_type = 0;
    int                             last_host_vmm_location_id   = 0;
    std::atomic<uint64_t>           driver_calls{0};
    std::atomic<uint64_t>           unclassified_import_calls{0};
    std::atomic<uint64_t>           errors{0};
    std::atomic<uint32_t>           last_error_cbid{0};
    std::atomic<bool>               initialized{false};
    bool                            init_attempted = false;
    std::string                     library_path;
    void*                           library    = nullptr;
    CUpti_SubscriberHandle          subscriber = nullptr;
};

ProbeState& probeState() {
    // Keep the ledger and CUPTI library alive for callbacks during process teardown.
    static ProbeState* state = new ProbeState;
    return *state;
}

void recordError(ProbeState& state, CUpti_CallbackId cbid = 0) noexcept {
    if (cbid != 0) {
        state.last_error_cbid.store(cbid, std::memory_order_relaxed);
    }
    state.errors.fetch_add(1, std::memory_order_relaxed);
}

bool isHostMemoryCall(CUpti_CallbackId cbid) noexcept {
    switch (cbid) {
        case CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc:
        case CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost:
        case CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost_v2:
        case CUPTI_DRIVER_TRACE_CBID_cuMemHostRegister:
        case CUPTI_DRIVER_TRACE_CBID_cuMemHostRegister_v2:
        case CUPTI_DRIVER_TRACE_CBID_cuMemFreeHost:
        case CUPTI_DRIVER_TRACE_CBID_cuMemHostUnregister:
        case CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc_v2:
        case CUPTI_DRIVER_TRACE_CBID_cuMemCreate:
        case CUPTI_DRIVER_TRACE_CBID_cuMemPoolCreate:
        case CUPTI_DRIVER_TRACE_CBID_cuMemGetDefaultMemPool:
        case CUPTI_DRIVER_TRACE_CBID_cuMemGetMemPool:
        case CUPTI_DRIVER_TRACE_CBID_cuMemImportFromShareableHandle:
        case CUPTI_DRIVER_TRACE_CBID_cuMemPoolImportFromShareableHandle:
        case CUPTI_DRIVER_TRACE_CBID_cuMemPoolImportPointer:
            return true;
        default:
            return false;
    }
}

void checkHostLocation(ProbeState& state, const CUmemLocation* location, uint64_t created_bytes = 0) noexcept {
    if (location == nullptr) {
        recordError(state);
        return;
    }
    switch (location->type) {
        case CU_MEM_LOCATION_TYPE_HOST:
        case CU_MEM_LOCATION_TYPE_HOST_NUMA:
        case CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT:
            // Host KV backings must still be covered by the classic pinned ledger.
            ++state.host_vmm_calls;
            state.host_vmm_created_bytes += created_bytes;
            state.last_host_vmm_location_type = location->type;
            state.last_host_vmm_location_id = location->id;
            break;
        default:
            break;
    }
}

void addAllocation(ProbeState& state, void* ptr, uint64_t bytes, const char* api, bool registered) {
    if (ptr == nullptr || bytes == 0 || bytes > std::numeric_limits<uint64_t>::max() - state.active_bytes
        || bytes > std::numeric_limits<uint64_t>::max() - state.total_allocated_bytes) {
        recordError(state);
        return;
    }
    const auto result = state.allocations.emplace(reinterpret_cast<uintptr_t>(ptr), Allocation{bytes, api, registered});
    if (!result.second) {
        recordError(state);
        return;
    }
    state.active_bytes += bytes;
    state.total_allocated_bytes += bytes;
}

template<typename Params>
void addHostAllocation(ProbeState& state, const void* params, const char* api) {
    const auto& p = *static_cast<const Params*>(params);
    if (p.pp == nullptr) {
        recordError(state);
        return;
    }
    addAllocation(state, *p.pp, p.bytesize, api, false);
}

template<typename Params>
void addHostRegistration(ProbeState& state, const void* params, const char* api) {
    const auto& p = *static_cast<const Params*>(params);
    if ((p.Flags & CU_MEMHOSTREGISTER_IOMEMORY) != 0) {
        recordError(state);
        return;
    }
    addAllocation(state, p.p, p.bytesize, api, true);
}

void removeAllocation(ProbeState& state, void* ptr, bool registered) {
    const auto it = state.allocations.find(reinterpret_cast<uintptr_t>(ptr));
    if (it == state.allocations.end() || it->second.registered != registered) {
        recordError(state);
        return;
    }
    state.active_bytes -= it->second.bytes;
    state.allocations.erase(it);
}

void CUPTIAPI driverCallback(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                            const void* cbdata) noexcept {
    auto& state = *static_cast<ProbeState*>(userdata);
    try {
        if (domain != CUPTI_CB_DOMAIN_DRIVER_API) {
            return;
        }
        if (cbdata == nullptr) {
            recordError(state);
            return;
        }
        const auto& data = *static_cast<const CUpti_CallbackData*>(cbdata);
        if (data.callbackSite != CUPTI_API_EXIT) {
            return;
        }
        state.driver_calls.fetch_add(1, std::memory_order_relaxed);
        if (!isHostMemoryCall(cbid)) {
            return;
        }
        std::lock_guard<std::mutex> lock(state.ledger_mutex);
        if (data.functionReturnValue == nullptr) {
            recordError(state, cbid);
            return;
        }
        const auto result = *static_cast<const CUresult*>(data.functionReturnValue);
        if (result != CUDA_SUCCESS) {
            return;
        }
        if (data.functionParams == nullptr) {
            recordError(state, cbid);
            return;
        }
        const auto errors_before = state.errors.load(std::memory_order_relaxed);
        switch (cbid) {
            case CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc:
                addHostAllocation<cuMemHostAlloc_params>(state, data.functionParams, "cuMemHostAlloc");
                break;
            case CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost:
                addHostAllocation<cuMemAllocHost_params>(state, data.functionParams, "cuMemAllocHost");
                break;
            case CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost_v2:
                addHostAllocation<cuMemAllocHost_v2_params>(state, data.functionParams, "cuMemAllocHost_v2");
                break;
            case CUPTI_DRIVER_TRACE_CBID_cuMemHostRegister:
                addHostRegistration<cuMemHostRegister_params>(state, data.functionParams, "cuMemHostRegister");
                break;
            case CUPTI_DRIVER_TRACE_CBID_cuMemHostRegister_v2:
                addHostRegistration<cuMemHostRegister_v2_params>(state, data.functionParams, "cuMemHostRegister_v2");
                break;
            case CUPTI_DRIVER_TRACE_CBID_cuMemFreeHost:
                removeAllocation(state, static_cast<const cuMemFreeHost_params*>(data.functionParams)->p, false);
                break;
            case CUPTI_DRIVER_TRACE_CBID_cuMemHostUnregister:
                removeAllocation(state, static_cast<const cuMemHostUnregister_params*>(data.functionParams)->p, true);
                break;
            case CUPTI_DRIVER_TRACE_CBID_cuMemCreate: {
                const auto& params = *static_cast<const cuMemCreate_params*>(data.functionParams);
                checkHostLocation(state, params.prop == nullptr ? nullptr : &params.prop->location, params.size);
                break;
            }
            case CUPTI_DRIVER_TRACE_CBID_cuMemPoolCreate: {
                const auto* props = static_cast<const cuMemPoolCreate_params*>(data.functionParams)->poolProps;
                checkHostLocation(state, props == nullptr ? nullptr : &props->location);
                break;
            }
            case CUPTI_DRIVER_TRACE_CBID_cuMemGetDefaultMemPool:
                checkHostLocation(state, static_cast<const cuMemGetDefaultMemPool_params*>(data.functionParams)->location);
                break;
            case CUPTI_DRIVER_TRACE_CBID_cuMemGetMemPool:
                checkHostLocation(state, static_cast<const cuMemGetMemPool_params*>(data.functionParams)->location);
                break;
            case CUPTI_DRIVER_TRACE_CBID_cuMemImportFromShareableHandle:
            case CUPTI_DRIVER_TRACE_CBID_cuMemPoolImportFromShareableHandle:
            case CUPTI_DRIVER_TRACE_CBID_cuMemPoolImportPointer:
                // Host KV backings are checked separately against the classic host allocation ledger.
                state.unclassified_import_calls.fetch_add(1, std::memory_order_relaxed);
                break;
            default:
                // cuMemHostAlloc_v2 has a CBID but no matching metadata struct.
                recordError(state);
                break;
        }
        if (state.errors.load(std::memory_order_relaxed) != errors_before) {
            state.last_error_cbid.store(cbid, std::memory_order_relaxed);
        }
    } catch (...) {
        recordError(state, cbid);
    }
}

}

extern "C" __attribute__((visibility("default"))) int scrPinnedProbeInit(const char* library_path) noexcept {
    ProbeState* state = nullptr;
    try {
        state = &probeState();
        std::lock_guard<std::mutex> lock(state->init_mutex);
        if (state->init_attempted) {
            if (library_path != nullptr && state->initialized.load() && state->library_path == library_path
                && state->errors.load(std::memory_order_relaxed) == 0) {
                return 0;
            }
            recordError(*state);
            return 1;
        }
        state->init_attempted = true;
        if (library_path == nullptr || library_path[0] != '/') {
            recordError(*state);
            return 1;
        }
        state->library_path = library_path;
        state->library = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
        if (state->library == nullptr) {
            recordError(*state);
            return 1;
        }
        const auto subscribe = reinterpret_cast<decltype(&cuptiSubscribe)>(dlsym(state->library, "cuptiSubscribe"));
        const auto enable_domain =
            reinterpret_cast<decltype(&cuptiEnableDomain)>(dlsym(state->library, "cuptiEnableDomain"));
        const auto unsubscribe =
            reinterpret_cast<decltype(&cuptiUnsubscribe)>(dlsym(state->library, "cuptiUnsubscribe"));
        if (subscribe == nullptr || enable_domain == nullptr || unsubscribe == nullptr) {
            recordError(*state);
            return 1;
        }
        const CUptiResult subscribe_result = subscribe(&state->subscriber, driverCallback, state);
        if (subscribe_result != CUPTI_SUCCESS) {
            recordError(*state);
            return static_cast<int>(subscribe_result);
        }
        const CUptiResult enable_result = enable_domain(1, state->subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
        if (enable_result != CUPTI_SUCCESS) {
            recordError(*state);
            if (unsubscribe(state->subscriber) != CUPTI_SUCCESS) {
                recordError(*state);
            }
            return static_cast<int>(enable_result);
        }
        if (state->errors.load(std::memory_order_relaxed) != 0) {
            return 1;
        }
        state->initialized.store(true);
        return 0;
    } catch (...) {
        if (state != nullptr) {
            recordError(*state);
        }
        return 1;
    }
}

extern "C" __attribute__((visibility("default"))) const char* scrPinnedProbeSnapshot() noexcept {
    ProbeState* state = nullptr;
    try {
        thread_local std::string snapshot;
        state = &probeState();
        std::lock_guard<std::mutex> lock(state->ledger_mutex);
        std::string json = "{\"pid\":" + std::to_string(static_cast<long long>(getpid()));
        json += ",\"initialized\":";
        json += state->initialized.load() ? "true" : "false";
        json += ",\"active_bytes\":" + std::to_string(state->active_bytes);
        json += ",\"total_allocated_bytes\":" + std::to_string(state->total_allocated_bytes);
        json += ",\"driver_calls\":" + std::to_string(state->driver_calls.load(std::memory_order_relaxed));
        json += ",\"unclassified_import_calls\":"
                + std::to_string(state->unclassified_import_calls.load(std::memory_order_relaxed));
        json += ",\"host_vmm_calls\":" + std::to_string(state->host_vmm_calls);
        json += ",\"host_vmm_created_bytes\":" + std::to_string(state->host_vmm_created_bytes);
        json += ",\"last_host_vmm_location_type\":" + std::to_string(state->last_host_vmm_location_type);
        json += ",\"last_host_vmm_location_id\":" + std::to_string(state->last_host_vmm_location_id);
        json += ",\"errors\":" + std::to_string(state->errors.load(std::memory_order_relaxed));
        json += ",\"last_error_cbid\":" + std::to_string(state->last_error_cbid.load(std::memory_order_relaxed));
        json += ",\"allocations\":[";
        bool first = true;
        for (const auto& entry : state->allocations) {
            if (!first) {
                json += ',';
            }
            first = false;
            json += "{\"ptr\":" + std::to_string(entry.first);
            json += ",\"bytes\":" + std::to_string(entry.second.bytes);
            json += ",\"api\":\"";
            json += entry.second.api;
            json += "\"}";
        }
        json += "]}";
        snapshot.swap(json);
        return snapshot.c_str();
    } catch (...) {
        uint64_t errors = 1;
        uint64_t calls  = 0;
        if (state != nullptr) {
            recordError(*state);
            errors = state->errors.load(std::memory_order_relaxed);
            calls  = state->driver_calls.load(std::memory_order_relaxed);
        }
        thread_local char failure[256];
        std::snprintf(failure, sizeof(failure),
                      "{\"pid\":%lld,\"initialized\":false,\"active_bytes\":0,\"total_allocated_bytes\":0,"
                      "\"driver_calls\":%llu,\"errors\":%llu,\"allocations\":[]}",
                      static_cast<long long>(getpid()), static_cast<unsigned long long>(calls),
                      static_cast<unsigned long long>(errors));
        return failure;
    }
}
