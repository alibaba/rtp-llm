// Development slice based on RTP-LLM main 492ecf45; not a production storage API.
// Driver VMM owns storage. ATen receives non-owning uint8 views with lifetime leases.
#include <torch/extension.h>
#include <ATen/ops/from_blob.h>
#include <cuda.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace {

// Python launch policy keeps the original small defaults. This is only the
// low-level hard ceiling; it does not establish available NUMA/container memory.
constexpr uint64_t kMaxAllocation = uint64_t{128} << 30;
constexpr uint64_t kMaxLoadChunk = uint64_t{64} << 20;
constexpr size_t kMaxRank = 32;
static_assert(sizeof(size_t) >= sizeof(uint64_t), "EGM bridge requires 64-bit size_t");

struct Counters {
    std::atomic<int64_t> contexts{0}, owner_allocations{0}, imports{0};
    std::atomic<int64_t> reservations{0}, mappings{0}, view_storages{0};
    std::atomic<int64_t> cleanup_failures{0}, deferred_cleanups{0};
} counters;

std::string cuda_error(CUresult result, const char* stage) {
    const char *name = nullptr, *message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    return std::string(stage) + ": " + (name ? name : "CUDA_UNKNOWN") +
           " (code=" + std::to_string(int(result)) + "): " +
           (message ? message : "unknown driver error");
}

void check(CUresult result, const char* stage) {
    if (result != CUDA_SUCCESS) throw std::runtime_error(cuda_error(result, stage));
}

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

class CurrentContext {
public:
    explicit CurrentContext(CUcontext context) {
        check(cuCtxPushCurrent(context), "context_push");
        pushed_ = true;
    }
    ~CurrentContext() noexcept {
        if (!pushed_) return;
        CUcontext previous{};
        CUresult result = cuCtxPopCurrent(&previous);
        if (result != CUDA_SUCCESS) {
            ++counters.cleanup_failures;
            std::fprintf(stderr, "engram_native: context_pop during unwind failed, CUDA code=%d\n", int(result));
        }
    }
    void finish() {
        CUcontext previous{};
        pushed_ = false; // Never retry an uncertain pop and accidentally pop a caller's context.
        check(cuCtxPopCurrent(&previous), "context_pop");
    }
    CurrentContext(const CurrentContext&) = delete;
    CurrentContext& operator=(const CurrentContext&) = delete;
private:
    bool pushed_ = false;
};

void validate_fail_stage(const std::string& value, bool owner) {
    const std::vector<std::string> common = {"", "after_context", "after_reserve", "after_map", "after_access"};
    for (const auto& stage : common) if (stage == value) return;
    if (owner && value == "after_create") return;
    if (!owner && (value == "after_import" || value == "after_properties")) return;
    throw std::invalid_argument("unknown fail_stage for this role");
}

void inject(const std::string& selected, const char* point) {
    if (selected == point) throw std::runtime_error(std::string("TEST_FAULT_INJECTION: ") + point);
}

class Mapping;
struct ViewLease {
    explicit ViewLease(std::shared_ptr<Mapping> owner);
    ~ViewLease() noexcept;
    std::shared_ptr<Mapping> mapping;
};

class Mapping : public std::enable_shared_from_this<Mapping> {
public:
    static uint64_t host_numa_granularity(int numa, int device) {
        require(numa >= 0, "numa must be a local nonnegative HOST_NUMA ID");
        auto state = std::shared_ptr<Mapping>(new Mapping(true));
        state->init_context(device);
        CurrentContext current(state->context_);
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        prop.location.id = numa;
        size_t granularity = 0;
        check(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
              "query_host_numa_granularity");
        require(granularity > 0 && granularity <= kMaxAllocation, "unsupported CUDA allocation granularity");
        current.finish();
        state->close();
        return granularity;
    }

    static std::shared_ptr<Mapping> make_owner(uint64_t size, int numa, int device,
                                               const std::string& fail_stage) {
        require(size > 0 && size <= kMaxAllocation, "size must be in 1..128GiB for this development slice");
        require(numa >= 0, "numa must be a local nonnegative HOST_NUMA ID");
        validate_fail_stage(fail_stage, true);
        auto state = std::shared_ptr<Mapping>(new Mapping(true));
        state->logical_bytes_ = size;
        state->numa_ = numa;
        state->init_context(device);
        inject(fail_stage, "after_context");
        CurrentContext current(state->context_);
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        prop.location.id = numa;
        check(cuMemGetAllocationGranularity(&state->granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
              "owner_granularity");
        state->validate_granularity();
        const uint64_t padding = (state->granularity_ - size % state->granularity_) % state->granularity_;
        require(padding <= kMaxAllocation - size, "padded allocation exceeds development limit 128GiB");
        state->bytes_ = static_cast<size_t>(size + padding);
        check(cuMemCreate(&state->handle_, state->bytes_, &prop, 0), "owner_create");
        state->has_handle_ = true;
        ++counters.owner_allocations;
        inject(fail_stage, "after_create");
        state->reserve_and_map(fail_stage);
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        access.location.id = numa; // Access descriptor's HOST_NUMA ID is ignored by CUDA.
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        check(cuMemSetAccess(state->address_, state->bytes_, &access, 1), "owner_cpu_access");
        inject(fail_stage, "after_access");
        // Initialize padding and any not-yet-loaded logical bytes; never publish stale pages.
        std::memset(reinterpret_cast<void*>(state->address_), 0, state->bytes_);
        state->ready_ = true;
        current.finish();
        return state;
    }

    static std::shared_ptr<Mapping> make_consumer(const CUmemFabricHandle& fabric, uint64_t bytes,
                                                  uint64_t advertised_granularity, int device,
                                                  const std::string& fail_stage) {
        require(bytes > 0 && bytes <= kMaxAllocation, "allocation_bytes must be in 1..128GiB");
        require(advertised_granularity > 0 && advertised_granularity <= kMaxAllocation,
                "invalid advertised granularity");
        require(bytes % advertised_granularity == 0, "allocation_bytes must be aligned to granularity");
        validate_fail_stage(fail_stage, false);
        auto state = std::shared_ptr<Mapping>(new Mapping(false));
        state->logical_bytes_ = bytes; // Logical tensor schema is validated by the Python manifest layer.
        state->bytes_ = bytes;
        state->init_context(device);
        inject(fail_stage, "after_context");
        CurrentContext current(state->context_);
        CUmemFabricHandle copy = fabric;
        check(cuMemImportFromShareableHandle(&state->handle_, &copy, CU_MEM_HANDLE_TYPE_FABRIC), "consumer_import");
        state->has_handle_ = true;
        ++counters.imports;
        inject(fail_stage, "after_import");
        CUmemAllocationProp actual{};
        check(cuMemGetAllocationPropertiesFromHandle(&actual, state->handle_), "consumer_properties");
        require(actual.type == CU_MEM_ALLOCATION_TYPE_PINNED, "imported allocation must be PINNED");
        require(actual.location.type == CU_MEM_LOCATION_TYPE_HOST_NUMA, "imported allocation must be HOST_NUMA");
        state->numa_ = actual.location.id; // Owner's NUMA identity, not a remote allocation request.
        check(cuMemGetAllocationGranularity(&state->granularity_, &actual, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
              "consumer_runtime_granularity");
        state->validate_granularity();
        require(advertised_granularity == state->granularity_, "advertised granularity differs from CUDA runtime properties");
        require(bytes % state->granularity_ == 0, "allocation_bytes is not runtime-granularity aligned");
        inject(fail_stage, "after_properties");
        // cuMemMap additionally validates that the actual object covers this extent.
        state->reserve_and_map(fail_stage);
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
        check(cuMemSetAccess(state->address_, state->bytes_, &access, 1), "consumer_gpu_read_access");
        unsigned long long flags = 0;
        check(cuMemGetAccess(&flags, &access.location, state->address_), "consumer_confirm_access");
        require(flags == CU_MEM_ACCESS_FLAGS_PROT_READ, "CUDA did not confirm READ-only access");
        inject(fail_stage, "after_access");
        state->ready_ = true;
        current.finish();
        return state;
    }

    ~Mapping() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) return;
        try {
            // Constructor rollback never exposed a GPU pointer, so no drain is needed.
            // Once a view was exposed, the last lease must drain before freeing storage.
            close_locked(may_have_gpu_work_);
        } catch (const std::exception& error) {
            ++counters.deferred_cleanups;
            // Leave CUDA refs and VA intact when completion/release cannot be proven.
            // No global ownership registry: the standalone worker must exit on this fault.
            std::fprintf(stderr, "engram_native: CLEANUP_DEFERRED until process exit: %s\n", error.what());
        }
    }

    void load(const char* data, size_t length, uint64_t offset) {
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_ready();
        require(owner_, "load is owner-only");
        require(!sealed_, "owner is sealed after first export; load is forbidden");
        require(length <= kMaxLoadChunk, "one load chunk must be <=64MiB");
        require(offset <= logical_bytes_ && length <= logical_bytes_ - offset, "load exceeds logical allocation bounds");
        if (length) std::memcpy(reinterpret_cast<void*>(address_ + offset), data, length);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        loaded_bytes_ += length; // Observability only; overlapping loads are permitted before export.
    }

    CUmemFabricHandle export_handle() {
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_ready();
        require(owner_, "export is owner-only");
        CurrentContext current(context_);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        check(cuCtxSynchronize(), "owner_publish_drain");
        CUmemFabricHandle fabric{};
        check(cuMemExportToShareableHandle(&fabric, handle_, CU_MEM_HANDLE_TYPE_FABRIC, 0), "owner_export");
        current.finish();
        sealed_ = true;
        return fabric;
    }

    at::Tensor tensor_view(const std::vector<int64_t>& shape, uint64_t offset) {
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_ready();
        require(!owner_, "tensor_view is consumer-only");
        require(shape.size() <= kMaxRank, "tensor rank exceeds development limit 32");
        require(offset < bytes_, "view offset must lie inside the mapped allocation");
        uint64_t count = 1;
        for (int64_t dimension : shape) {
            require(dimension >= 0, "tensor dimensions must be nonnegative");
            if (dimension == 0) { count = 0; continue; }
            require(uint64_t(dimension) <= kMaxAllocation, "tensor dimension exceeds development limit");
            require(count <= (bytes_ - offset) / uint64_t(dimension), "tensor shape exceeds mapping bounds");
            count *= uint64_t(dimension);
        }
        require(count <= bytes_ - offset, "tensor view exceeds mapping bounds");
        CurrentContext current(context_);
        auto lease = std::make_shared<ViewLease>(shared_from_this());
        may_have_gpu_work_ = true;
        const auto device = c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device_));
        auto options = at::TensorOptions().device(device).dtype(at::kByte).requires_grad(false);
        // Explicit target_device is important for a host-backed EGM address. CUDA
        // access was established and checked above; do not infer its tensor device
        // from host-allocation pointer attributes. No allocator/copy is invoked.
        auto tensor = at::from_blob(reinterpret_cast<void*>(address_ + offset), shape,
            [lease = std::move(lease)](void*) mutable { lease.reset(); }, options, device);
        current.finish();
        return tensor;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) return;
        require(active_views_.load() == 0, "cannot close: live tensor storage/view leases remain");
        close_locked(true); // Only this process's selected primary context, not other processes.
    }

    py::dict info() const {
        std::lock_guard<std::mutex> lock(mutex_);
        py::dict result;
        result["role"] = owner_ ? "owner" : "consumer";
        result["state"] = closed_ ? "CLOSED" : (cleanup_failed_ ? "CLEANUP_FAILED" : (ready_ ? "READY" : "CONSTRUCTING"));
        result["logical_bytes"] = logical_bytes_;
        result["allocation_bytes"] = bytes_;
        result["granularity"] = granularity_;
        result["device"] = device_;
        result["numa"] = numa_;
        result["base_address"] = uint64_t(address_);
        result["sealed"] = sealed_;
        result["active_views"] = active_views_.load();
        result["loaded_bytes_total"] = loaded_bytes_;
        result["mapped"] = mapped_;
        result["reserved"] = reserved_;
        result["has_allocation_reference"] = has_handle_;
        result["context_retained"] = context_retained_;
        result["gpu_access"] = (!owner_ && mapped_) ? "READ" : "NONE";
        return result;
    }

    void acquire_view() noexcept { ++active_views_; ++counters.view_storages; }
    void release_view() noexcept { --active_views_; --counters.view_storages; }

private:
    explicit Mapping(bool owner): owner_(owner) {}

    void init_context(int ordinal) {
        require(ordinal >= 0 && ordinal <= std::numeric_limits<c10::DeviceIndex>::max(), "device ordinal is out of range");
        check(cuInit(0), "driver_init");
        check(cuDeviceGet(&cuda_device_, ordinal), "device_get");
        device_ = ordinal;
        check(cuDevicePrimaryCtxRetain(&context_, cuda_device_), "primary_context_retain");
        context_retained_ = true;
        ++counters.contexts;
    }

    void validate_granularity() const {
        require(granularity_ > 0 && granularity_ <= kMaxAllocation, "unsupported CUDA allocation granularity");
    }

    void reserve_and_map(const std::string& fail_stage) {
        check(cuMemAddressReserve(&address_, bytes_, 0, 0, 0), "address_reserve");
        reserved_ = true;
        ++counters.reservations;
        inject(fail_stage, "after_reserve");
        check(cuMemMap(address_, bytes_, 0, handle_, 0), "map");
        mapped_ = true;
        ++counters.mappings;
        inject(fail_stage, "after_map");
    }

    void ensure_ready() const {
        require(ready_ && !closed_ && !cleanup_failed_ && mapped_ && has_handle_, "mapping is not ready");
    }

    void close_locked(bool drain) {
        try {
            if (context_retained_ && (mapped_ || reserved_ || has_handle_)) {
                CurrentContext current(context_);
                if (drain) check(cuCtxSynchronize(), "close_context_drain");
                if (mapped_) {
                    check(cuMemUnmap(address_, bytes_), "unmap");
                    mapped_ = false;
                    --counters.mappings;
                }
                if (reserved_) {
                    check(cuMemAddressFree(address_, bytes_), "address_free");
                    reserved_ = false;
                    address_ = 0;
                    --counters.reservations;
                }
                if (has_handle_) {
                    check(cuMemRelease(handle_), "allocation_release");
                    has_handle_ = false;
                    if (owner_) --counters.owner_allocations; else --counters.imports;
                }
                current.finish();
            }
            if (context_retained_) {
                check(cuDevicePrimaryCtxRelease(cuda_device_), "primary_context_release");
                context_retained_ = false;
                --counters.contexts;
            }
            ready_ = false;
            closed_ = true;
        } catch (...) {
            cleanup_failed_ = true;
            ++counters.cleanup_failures;
            throw;
        }
    }

    mutable std::mutex mutex_;
    const bool owner_;
    bool ready_ = false, closed_ = false, sealed_ = false, cleanup_failed_ = false;
    bool context_retained_ = false, has_handle_ = false, reserved_ = false, mapped_ = false;
    bool may_have_gpu_work_ = false;
    int device_ = -1, numa_ = -1;
    CUdevice cuda_device_{};
    CUcontext context_{};
    CUmemGenericAllocationHandle handle_{};
    CUdeviceptr address_{};
    uint64_t logical_bytes_ = 0, loaded_bytes_ = 0;
    size_t bytes_ = 0, granularity_ = 0;
    std::atomic<int64_t> active_views_{0};
};

ViewLease::ViewLease(std::shared_ptr<Mapping> owner): mapping(std::move(owner)) { mapping->acquire_view(); }
ViewLease::~ViewLease() noexcept { mapping->release_view(); }

class Owner {
public:
    Owner(uint64_t size, int numa, int device, const std::string& fail_stage) {
        py::gil_scoped_release release;
        mapping_ = Mapping::make_owner(size, numa, device, fail_stage);
    }
    void load(const py::bytes& data, uint64_t offset) {
        char* address = nullptr; Py_ssize_t length = 0;
        if (PyBytes_AsStringAndSize(data.ptr(), &address, &length) != 0) throw py::error_already_set();
        py::gil_scoped_release release;
        mapping_->load(address, static_cast<size_t>(length), offset);
    }
    py::bytes export_handle() {
        CUmemFabricHandle fabric{};
        { py::gil_scoped_release release; fabric = mapping_->export_handle(); }
        return py::bytes(reinterpret_cast<const char*>(&fabric), sizeof(fabric));
    }
    void close() { mapping_->close(); }
    py::dict info() const { return mapping_->info(); }
private:
    std::shared_ptr<Mapping> mapping_;
};

class Consumer {
public:
    Consumer(const py::bytes& handle, uint64_t bytes, uint64_t granularity, int device,
             const std::string& fail_stage) {
        char* address = nullptr; Py_ssize_t length = 0;
        if (PyBytes_AsStringAndSize(handle.ptr(), &address, &length) != 0) throw py::error_already_set();
        require(length == sizeof(CUmemFabricHandle), "fabric handle has incorrect byte length");
        CUmemFabricHandle fabric{};
        std::memcpy(&fabric, address, sizeof(fabric));
        py::gil_scoped_release release;
        mapping_ = Mapping::make_consumer(fabric, bytes, granularity, device, fail_stage);
    }
    at::Tensor tensor_view(const std::vector<int64_t>& shape, uint64_t offset) { return mapping_->tensor_view(shape, offset); }
    void close() { mapping_->close(); }
    py::dict info() const { return mapping_->info(); }
private:
    std::shared_ptr<Mapping> mapping_;
};

py::dict resource_counters() {
    py::dict result;
    result["contexts"] = counters.contexts.load();
    result["owner_allocations"] = counters.owner_allocations.load();
    result["imports"] = counters.imports.load();
    result["reservations"] = counters.reservations.load();
    result["mappings"] = counters.mappings.load();
    result["view_storages"] = counters.view_storages.load();
    result["cleanup_failures"] = counters.cleanup_failures.load();
    result["deferred_cleanups"] = counters.deferred_cleanups.load();
    return result;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Development-only immutable EGM owner and read-only external CUDA uint8 tensor views.";
    module.attr("max_allocation_bytes") = kMaxAllocation;
    module.attr("max_load_chunk_bytes") = kMaxLoadChunk;
    module.attr("fabric_handle_bytes") = sizeof(CUmemFabricHandle);
    module.def("resource_counters", &resource_counters);
    module.def("host_numa_granularity", &Mapping::host_numa_granularity,
               py::arg("numa"), py::arg("device"), py::call_guard<py::gil_scoped_release>());
    py::class_<Owner>(module, "Owner")
        .def(py::init<uint64_t, int, int, const std::string&>(), py::arg("size"), py::arg("numa"),
             py::arg("device"), py::arg("fail_stage") = "")
        .def("load", &Owner::load, py::arg("data"), py::arg("offset") = 0)
        .def("export_handle", &Owner::export_handle)
        .def("info", &Owner::info)
        .def("close", &Owner::close, py::call_guard<py::gil_scoped_release>());
    py::class_<Consumer>(module, "Consumer")
        .def(py::init<const py::bytes&, uint64_t, uint64_t, int, const std::string&>(), py::arg("handle"),
             py::arg("allocation_bytes"), py::arg("granularity"), py::arg("device"), py::arg("fail_stage") = "")
        .def("tensor_view", &Consumer::tensor_view, py::arg("shape"), py::arg("offset") = 0)
        .def("info", &Consumer::info)
        .def("close", &Consumer::close, py::call_guard<py::gil_scoped_release>());
}
