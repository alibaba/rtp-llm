// CUDA Fabric allocations and virtual address mappings with lifetime-managed ATen views.
#include <torch/extension.h>
#include <ATen/ops/from_blob.h>
#include <cuda.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace {

// Hard limit per physical allocation. A mapping can span multiple allocations.
constexpr uint64_t kMaxAllocation = uint64_t{128} << 30;
constexpr uint64_t kMaxLoadChunk = uint64_t{64} << 20;
constexpr size_t kMaxRank = 32;
static_assert(sizeof(size_t) >= sizeof(uint64_t), "Fabric mappings require 64-bit size_t");

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

class CudaError : public std::runtime_error {
public:
    CudaError(CUresult result, const char* operation)
        : std::runtime_error(cuda_error(result, operation)), result(result), operation(operation) {}

    CUresult result;
    const char* operation;
    int64_t segment = -1;
};

void check(CUresult result, const char* stage) {
    if (result != CUDA_SUCCESS) throw CudaError(result, stage);
}

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

int visibleDeviceCount() {
    check(cuInit(0), "driver_init");
    int count = 0;
    check(cuDeviceGetCount(&count), "device_count");
    return count;
}

CUdevice getDevice(int ordinal) {
    CUdevice device;
    check(cuDeviceGet(&device, ordinal), "device_get");
    return device;
}

std::vector<int> deviceNumaNodes() {
    const int count = visibleDeviceCount();
    std::vector<int> nodes(count);
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        check(cuDeviceGetAttribute(&nodes[ordinal], CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, getDevice(ordinal)),
              "device_host_numa");
    }
    return nodes;
}

void checkFabricDevices(const std::vector<int>& ordinals) {
    const int count = visibleDeviceCount();
    for (const int ordinal : ordinals) {
        if (ordinal < 0 || ordinal >= count) {
            throw std::out_of_range("device ordinal is outside visible CUDA devices");
        }
        const CUdevice device = getDevice(ordinal);
        int fabric_supported = 0;
        check(cuDeviceGetAttribute(&fabric_supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device),
              "device_fabric_support");
        require(fabric_supported != 0, "device does not support CUDA Fabric handles");
    }
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
            std::fprintf(stderr, "fabric_memory: context_pop during unwind failed, CUDA code=%d\n", int(result));
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
    if (owner && (value == "after_create" || value == "after_zero")) return;
    if (!owner && (value == "after_import" || value == "after_properties")) return;
    throw std::invalid_argument("unknown fail_stage for this role");
}

void inject(const std::string& selected, const char* point) {
    if (selected == point) throw std::runtime_error(std::string("TEST_FAULT_INJECTION: ") + point);
}

CUmemAllocationHandleType handleType(const std::string& value, bool gpu) {
    if (value == "fabric") return CU_MEM_HANDLE_TYPE_FABRIC;
    require(value == "posix_fd" && gpu, "handle_type must be fabric, or posix_fd for GPU memory");
    return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
}

struct FabricSegment {
    CUmemFabricHandle fabric{};
    int fd = -1;
    size_t bytes;
    size_t granularity;
};

struct MappedSegment {
    size_t offset;
    size_t bytes;
    CUmemGenericAllocationHandle handle{};
    bool retained = false;
    bool mapped = false;
};

class Mapping;
struct ViewLease {
    explicit ViewLease(std::shared_ptr<Mapping> owner);
    ~ViewLease() noexcept;
    std::shared_ptr<Mapping> mapping;
};

class Mapping : public std::enable_shared_from_this<Mapping> {
public:
    static uint64_t allocation_granularity(int numa, int device, bool gpu = false,
                                           CUmemAllocationHandleType handle_type = CU_MEM_HANDLE_TYPE_FABRIC) {
        require(gpu || numa >= 0, "numa must be a local nonnegative HOST_NUMA ID");
        auto state = std::shared_ptr<Mapping>(new Mapping(true));
        state->init_context(device);
        CurrentContext current(state->context_);
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.requestedHandleTypes = handle_type;
        prop.location.type = gpu ? CU_MEM_LOCATION_TYPE_DEVICE : CU_MEM_LOCATION_TYPE_HOST_NUMA;
        prop.location.id = gpu ? state->cuda_device_ : numa;
        size_t granularity = 0;
        check(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
              "query_allocation_granularity");
        require(granularity > 0 && granularity <= kMaxAllocation, "unsupported CUDA allocation granularity");
        current.finish();
        state->close();
        return granularity;
    }

    static std::shared_ptr<Mapping> make_owner(uint64_t size, int numa, int device,
                                               const std::string& fail_stage, bool gpu = false,
                                               CUmemAllocationHandleType handle_type = CU_MEM_HANDLE_TYPE_FABRIC) {
        require(size > 0 && size <= kMaxAllocation, "size must be in 1..128GiB");
        require(gpu || numa >= 0, "numa must be a local nonnegative HOST_NUMA ID");
        validate_fail_stage(fail_stage, true);
        auto state = std::shared_ptr<Mapping>(new Mapping(true));
        state->logical_bytes_ = size;
        state->numa_ = gpu ? -1 : numa;
        state->gpu_memory_ = gpu;
        state->handle_type_ = handle_type;
        state->init_context(device);
        inject(fail_stage, "after_context");
        CurrentContext current(state->context_);
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.requestedHandleTypes = handle_type;
        prop.location.type = gpu ? CU_MEM_LOCATION_TYPE_DEVICE : CU_MEM_LOCATION_TYPE_HOST_NUMA;
        prop.location.id = gpu ? state->cuda_device_ : numa;
        check(cuMemGetAllocationGranularity(&state->granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
              "owner_granularity");
        state->validate_granularity();
        const uint64_t padding = (state->granularity_ - size % state->granularity_) % state->granularity_;
        require(padding <= kMaxAllocation - size, "padded allocation exceeds allocation limit 128GiB");
        state->bytes_ = static_cast<size_t>(size + padding);
        state->segments_.push_back({0, state->bytes_});
        auto& segment = state->segments_.front();
        check(cuMemCreate(&segment.handle, state->bytes_, &prop, 0), "allocation_create");
        segment.retained = true;
        ++counters.owner_allocations;
        inject(fail_stage, "after_create");
        state->reserve(0);
        inject(fail_stage, "after_reserve");
        state->map(segment);
        inject(fail_stage, "after_map");
        CUmemAccessDesc access{};
        access.location.type = gpu ? CU_MEM_LOCATION_TYPE_DEVICE : CU_MEM_LOCATION_TYPE_HOST_NUMA;
        access.location.id = gpu ? state->cuda_device_ : numa;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        check(cuMemSetAccess(state->address_, state->bytes_, &access, 1), gpu ? "owner_gpu_access" : "owner_cpu_access");
        inject(fail_stage, "after_access");
        // Initialize padding and any not-yet-loaded logical bytes; never publish stale pages.
        if (gpu) {
            // Device memset can outlive the host call, including its failure path.
            state->may_have_gpu_work_ = true;
            check(cuMemsetD8(state->address_, 0, state->bytes_), "owner_gpu_zero");
        } else std::memset(reinterpret_cast<void*>(state->address_), 0, state->bytes_);
        inject(fail_stage, "after_zero");
        state->ready_ = true;
        current.finish();
        return state;
    }

    static std::shared_ptr<Mapping> make_mapping(int device, bool gpu, CUmemAllocationHandleType handle_type) {
        validate_device(device);
        auto state = std::shared_ptr<Mapping>(new Mapping(false));
        state->device_ = device;
        state->gpu_memory_ = gpu;
        state->handle_type_ = handle_type;
        return state;
    }

    void attach(const std::vector<FabricSegment>& parts, const std::string& fail_stage,
                size_t fail_segment, bool fail_rollback) {
        std::lock_guard<std::mutex> lock(mutex_);
        require(!ready_ && !closed_ && !cleanup_failed_, "mapping is not new");
        require(!parts.empty(), "at least one Fabric segment is required");
        validate_fail_stage(fail_stage, false);
        require(fail_segment < parts.size(), "fault segment is out of range");

        // Validate the entire descriptor and preallocate bookkeeping before acquiring CUDA resources.
        std::vector<MappedSegment> segments;
        segments.reserve(parts.size());
        size_t span = 0, alignment = 0;
        for (const auto& part : parts) {
            require(part.bytes > 0 && part.bytes <= kMaxAllocation, "allocation_bytes must be in 1..128GiB");
            require(part.granularity > 0 && part.granularity <= kMaxAllocation &&
                    (part.granularity & (part.granularity - 1)) == 0, "invalid advertised granularity");
            require(part.bytes % part.granularity == 0, "allocation_bytes must be aligned to granularity");
            require(span % part.granularity == 0, "segment offset must be aligned to granularity");
            require(part.bytes <= size_t(std::numeric_limits<int64_t>::max()) - span, "mapping span exceeds tensor limits");
            segments.push_back({span, part.bytes});
            span += part.bytes;
            alignment = std::max(alignment, part.granularity);
        }
        segments_ = std::move(segments);
        logical_bytes_ = bytes_ = span;
        granularity_ = alignment;

        try {
            init_context(device_);
            inject(fail_stage, "after_context");
            CurrentContext current(context_);
            reserve(alignment);
            inject(fail_stage, "after_reserve");
            for (size_t i = 0; i < parts.size(); ++i) {
                try {
                    attach_segment(parts[i], segments_[i], i == fail_segment ? fail_stage : "");
                } catch (CudaError& error) {
                    error.segment = static_cast<int64_t>(i);
                    throw;
                }
            }
            current.finish();
            ready_ = true;
        } catch (...) {
            // The caller keeps this object if rollback fails. Preserve the attach error
            // and leave the unreleased ranges/refs recorded for a later close().
            try {
                close_locked(false, fail_rollback);
            } catch (...) {
                // info() exposes CLEANUP_FAILED and the remaining resource counts.
            }
            throw;
        }
    }

    ~Mapping() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) return;
        try {
            // Drain GPU initialization and any work submitted through exposed views.
            // Rollback before GPU work needs no synchronization.
            close_locked(may_have_gpu_work_);
        } catch (const std::exception& error) {
            ++counters.deferred_cleanups;
            // Leave CUDA refs and VA intact when completion/release cannot be proven.
            // No global ownership registry: the standalone worker must exit on this fault.
            std::fprintf(stderr, "fabric_memory: CLEANUP_DEFERRED until process exit: %s\n", error.what());
        }
    }

    void load(const char* data, size_t length, uint64_t offset) {
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_ready();
        require(owner_, "load is owner-only");
        require(!sealed_, "owner is sealed after first export; load is forbidden");
        require(length <= kMaxLoadChunk, "one load chunk must be <=64MiB");
        require(offset <= logical_bytes_ && length <= logical_bytes_ - offset, "load exceeds logical allocation bounds");
        if (length) {
            if (gpu_memory_) {
                CurrentContext current(context_);
                check(cuMemcpyHtoD(address_ + offset, data, length), "owner_gpu_load");
                current.finish();
            } else std::memcpy(reinterpret_cast<void*>(address_ + offset), data, length);
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        loaded_bytes_ += length; // Observability only; overlapping loads are permitted before export.
    }

    CUmemFabricHandle export_handle() {
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_ready();
        require(owner_, "export is owner-only");
        require(handle_type_ == CU_MEM_HANDLE_TYPE_FABRIC, "use export_fd for POSIX_FD memory");
        CurrentContext current(context_);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        check(cuCtxSynchronize(), "owner_publish_drain");
        CUmemFabricHandle fabric{};
        check(cuMemExportToShareableHandle(&fabric, segments_.front().handle, CU_MEM_HANDLE_TYPE_FABRIC, 0), "owner_export");
        current.finish();
        sealed_ = true;
        return fabric;
    }

    int export_fd() {
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_ready();
        require(owner_ && handle_type_ == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                "export_fd requires a GPU POSIX_FD allocation");
        CurrentContext current(context_);
        check(cuCtxSynchronize(), "owner_publish_drain");
        int fd = -1;
        check(cuMemExportToShareableHandle(&fd, segments_.front().handle, handle_type_, 0), "owner_export_fd");
        sealed_ = true;
        try {
            if (fcntl(fd, F_SETFD, FD_CLOEXEC) == -1) throw std::runtime_error("cannot set exported FD CLOEXEC");
            current.finish();
            return fd; // The caller owns the FD; CUDA allocation references are independent.
        } catch (...) {
            ::close(fd);
            throw;
        }
    }

    at::Tensor tensor_view(const std::vector<int64_t>& shape, uint64_t offset, at::ScalarType dtype) {
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_ready();
        require(!owner_, "tensor_view is consumer-only");
        require(shape.size() <= kMaxRank, "tensor rank exceeds limit 32");
        require(offset < bytes_, "view offset must lie inside the mapped allocation");
        switch (dtype) {
            case at::kByte:
            case at::kHalf:
            case at::kBFloat16:
            case at::kFloat:
            case at::kInt:
            case at::kLong:
                break;
            default:
                throw std::invalid_argument("unsupported Fabric view dtype");
        }
        const uint64_t item_size = c10::elementSize(dtype);
        require(offset % item_size == 0, "view byte offset must be aligned to dtype");
        uint64_t extent = 1;
        bool empty = false;
        for (int64_t dimension : shape) {
            require(dimension >= 0, "tensor dimensions must be nonnegative");
            require(uint64_t(dimension) <= bytes_ / item_size, "tensor dimension exceeds mapping bounds");
            // Even an empty tensor needs representable contiguous strides.
            const auto stride_dim = uint64_t(std::max<int64_t>(dimension, 1));
            require(extent <= uint64_t(std::numeric_limits<int64_t>::max()) / stride_dim,
                    "tensor shape exceeds contiguous stride limits");
            extent *= stride_dim;
            empty = empty || dimension == 0;
        }
        require(empty || extent <= (bytes_ - offset) / item_size, "tensor view exceeds mapping bounds");
        CurrentContext current(context_);
        auto lease = std::make_shared<ViewLease>(shared_from_this());
        may_have_gpu_work_ = true;
        const auto device = c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device_));
        auto options = at::TensorOptions().device(device).dtype(dtype).requires_grad(false);
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
        close_locked(owner_ || may_have_gpu_work_);
    }

    py::dict info() const {
        std::lock_guard<std::mutex> lock(mutex_);
        py::dict result;
        result["role"] = owner_ ? "allocation" : "mapping";
        result["state"] = closed_ ? "CLOSED" : (cleanup_failed_ ? "CLEANUP_FAILED" : (ready_ ? "READY" : "NEW"));
        result["logical_bytes"] = logical_bytes_;
        result["allocation_bytes"] = bytes_;
        result["granularity"] = granularity_;
        result["device"] = device_;
        result["numa"] = numa_;
        result["memory_kind"] = gpu_memory_ ? "gpu" : "host";
        result["handle_type"] = handle_type_ == CU_MEM_HANDLE_TYPE_FABRIC ? "FABRIC" : "POSIX_FD";
        result["base_address"] = uint64_t(address_);
        result["sealed"] = sealed_;
        result["active_views"] = active_views_.load();
        result["loaded_bytes_total"] = loaded_bytes_;
        size_t mapped_ranges = 0, references = 0;
        for (const auto& segment : segments_) {
            mapped_ranges += segment.mapped;
            references += segment.retained;
        }
        result["mapped"] = mapped_ranges != 0;
        result["mapped_ranges"] = mapped_ranges;
        result["imported_handles"] = owner_ ? 0 : references;
        result["reserved"] = reserved_;
        result["has_allocation_reference"] = references != 0;
        result["context_retained"] = context_retained_;
        result["gpu_access"] = ready_ && !cleanup_failed_ && mapped_ranges ? (owner_ ? (gpu_memory_ ? "READWRITE" : "NONE") : "READ") : "NONE";
        return result;
    }

    void acquire_view() noexcept { ++active_views_; ++counters.view_storages; }
    void release_view() noexcept { --active_views_; --counters.view_storages; }

private:
    explicit Mapping(bool owner): owner_(owner) {}

    static void validate_device(int ordinal) {
        require(ordinal >= 0 && ordinal <= std::numeric_limits<c10::DeviceIndex>::max(), "device ordinal is out of range");
    }

    void init_context(int ordinal) {
        validate_device(ordinal);
        check(cuInit(0), "driver_init");
        cuda_device_ = getDevice(ordinal);
        device_ = ordinal;
        check(cuDevicePrimaryCtxRetain(&context_, cuda_device_), "primary_context_retain");
        context_retained_ = true;
        ++counters.contexts;
    }

    void validate_granularity() const {
        require(granularity_ > 0 && granularity_ <= kMaxAllocation, "unsupported CUDA allocation granularity");
    }

    void reserve(size_t alignment) {
        check(cuMemAddressReserve(&address_, bytes_, alignment, 0, 0), "address_reserve");
        reserved_ = true;
        ++counters.reservations;
    }

    void map(MappedSegment& segment) {
        check(cuMemMap(address_ + segment.offset, segment.bytes, 0, segment.handle, 0), "map");
        segment.mapped = true;
        ++counters.mappings;
    }

    void attach_segment(const FabricSegment& part, MappedSegment& segment, const std::string& fail_stage) {
        CUmemFabricHandle fabric = part.fabric;
        void* handle = handle_type_ == CU_MEM_HANDLE_TYPE_FABRIC
                           ? static_cast<void*>(&fabric)
                           : reinterpret_cast<void*>(static_cast<intptr_t>(part.fd));
        check(cuMemImportFromShareableHandle(&segment.handle, handle, handle_type_), "import_handle");
        segment.retained = true;
        ++counters.imports;
        inject(fail_stage, "after_import");
        CUmemAllocationProp actual{};
        check(cuMemGetAllocationPropertiesFromHandle(&actual, segment.handle), "allocation_properties");
        require(actual.type == CU_MEM_ALLOCATION_TYPE_PINNED, "imported allocation must be PINNED");
        require(actual.location.type == (gpu_memory_ ? CU_MEM_LOCATION_TYPE_DEVICE : CU_MEM_LOCATION_TYPE_HOST_NUMA),
                "imported allocation memory kind mismatch");
        size_t granularity = 0;
        check(cuMemGetAllocationGranularity(&granularity, &actual, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
              "runtime_granularity");
        require(granularity == part.granularity, "advertised granularity differs from CUDA runtime properties");
        if (!gpu_memory_ && segments_.size() == 1) numa_ = actual.location.id;
        inject(fail_stage, "after_properties");
        map(segment);
        inject(fail_stage, "after_map");
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = cuda_device_;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
        check(cuMemSetAccess(address_ + segment.offset, segment.bytes, &access, 1), "set_access");
        unsigned long long flags = 0;
        check(cuMemGetAccess(&flags, &access.location, address_ + segment.offset), "confirm_access");
        require(flags == CU_MEM_ACCESS_FLAGS_PROT_READ, "CUDA did not confirm READ-only access");
        inject(fail_stage, "after_access");
    }

    void ensure_ready() const {
        require(ready_ && !closed_ && !cleanup_failed_, "mapping is not ready");
    }

    void close_locked(bool drain, bool fail_rollback = false) {
        try {
            if (context_retained_) {
                CurrentContext current(context_);
                if (drain) check(cuCtxSynchronize(), "close_context_drain");
                for (auto segment = segments_.rbegin(); segment != segments_.rend(); ++segment) {
                    if (segment->mapped) {
                        check(cuMemUnmap(address_ + segment->offset, segment->bytes), "unmap");
                        segment->mapped = false;
                        --counters.mappings;
                    }
                    if (segment->retained) {
                        if (fail_rollback) throw std::runtime_error("TEST_FAULT_INJECTION: allocation_release");
                        check(cuMemRelease(segment->handle), "allocation_release");
                        segment->retained = false;
                        if (owner_) --counters.owner_allocations; else --counters.imports;
                    }
                }
                if (reserved_) {
                    check(cuMemAddressFree(address_, bytes_), "address_free");
                    reserved_ = false;
                    address_ = 0;
                    --counters.reservations;
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
    bool context_retained_ = false, reserved_ = false;
    bool may_have_gpu_work_ = false;
    bool gpu_memory_ = false;
    CUmemAllocationHandleType handle_type_ = CU_MEM_HANDLE_TYPE_FABRIC;
    int device_ = -1, numa_ = -1;
    CUdevice cuda_device_{};
    CUcontext context_{};
    std::vector<MappedSegment> segments_;
    CUdeviceptr address_{};
    uint64_t logical_bytes_ = 0, loaded_bytes_ = 0;
    size_t bytes_ = 0, granularity_ = 0;
    std::atomic<int64_t> active_views_{0};
};

ViewLease::ViewLease(std::shared_ptr<Mapping> owner): mapping(std::move(owner)) { mapping->acquire_view(); }
ViewLease::~ViewLease() noexcept { mapping->release_view(); }

class FabricAllocation {
public:
    FabricAllocation(uint64_t size, int numa, int device, const std::string& fail_stage, const std::string& memory_kind,
                     const std::string& handle_type) {
        require(memory_kind == "host" || memory_kind == "gpu", "memory_kind must be host or gpu");
        const auto type = handleType(handle_type, memory_kind == "gpu");
        py::gil_scoped_release release;
        mapping_ = Mapping::make_owner(size, numa, device, fail_stage, memory_kind == "gpu", type);
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
    int export_fd() { return mapping_->export_fd(); }
    void close() { mapping_->close(); }
    py::dict info() const { return mapping_->info(); }
private:
    std::shared_ptr<Mapping> mapping_;
};

class FabricMapping {
public:
    FabricMapping(int device, const std::string& memory_kind, const std::string& handle_type) {
        require(memory_kind == "host" || memory_kind == "gpu", "memory_kind must be host or gpu");
        handle_type_ = handleType(handle_type, memory_kind == "gpu");
        mapping_ = Mapping::make_mapping(device, memory_kind == "gpu", handle_type_);
    }

    void attach(const py::sequence& segments, const std::string& fail_stage,
                size_t fail_segment, bool fail_rollback) {
        std::vector<FabricSegment> parts;
        parts.reserve(py::len(segments));
        for (const auto& item : segments) {
            require(py::isinstance<py::tuple>(item) && py::len(item) == 3,
                    "each segment must be a (handle, allocation_bytes, granularity) tuple");
            auto tuple = py::reinterpret_borrow<py::tuple>(item);
            require(py::isinstance<py::int_>(tuple[1]) && !py::isinstance<py::bool_>(tuple[1]) &&
                    py::isinstance<py::int_>(tuple[2]) && !py::isinstance<py::bool_>(tuple[2]),
                    "allocation_bytes and granularity must be integers");
            FabricSegment part{};
            if (handle_type_ == CU_MEM_HANDLE_TYPE_FABRIC) {
                require(py::isinstance<py::bytes>(tuple[0]), "Fabric handle must be bytes");
                char* address = nullptr;
                Py_ssize_t length = 0;
                if (PyBytes_AsStringAndSize(tuple[0].ptr(), &address, &length) != 0) throw py::error_already_set();
                require(length == sizeof(part.fabric), "fabric handle has incorrect byte length");
                std::memcpy(&part.fabric, address, sizeof(part.fabric));
            } else {
                require(py::isinstance<py::int_>(tuple[0]) && !py::isinstance<py::bool_>(tuple[0]),
                        "POSIX_FD handle must be a local integer FD");
                const auto fd = PyLong_AsLongLong(tuple[0].ptr());
                if (PyErr_Occurred()) throw py::error_already_set();
                require(fd >= 0 && fd <= std::numeric_limits<int>::max(), "POSIX_FD handle is out of range");
                part.fd = static_cast<int>(fd);
            }
            part.bytes = PyLong_AsUnsignedLongLong(tuple[1].ptr());
            if (PyErr_Occurred()) throw py::error_already_set();
            part.granularity = PyLong_AsUnsignedLongLong(tuple[2].ptr());
            if (PyErr_Occurred()) throw py::error_already_set();
            parts.push_back(part);
        }
        try {
            py::gil_scoped_release release;
            mapping_->attach(parts, fail_stage, fail_segment, fail_rollback);
        } catch (const CudaError& error) {
            const char* name = nullptr;
            cuGetErrorName(error.result, &name);
            auto value = py::reinterpret_borrow<py::object>(PyExc_RuntimeError)(error.what());
            value.attr("operation") = error.operation;
            value.attr("segment") = error.segment;
            value.attr("cuda_error") = int(error.result);
            value.attr("cuda_error_name") = name ? name : "CUDA_UNKNOWN";
            PyErr_SetObject(PyExc_RuntimeError, value.ptr());
            throw py::error_already_set();
        }
    }

    at::Tensor tensor_view(const std::vector<int64_t>& shape, uint64_t offset, at::ScalarType dtype) {
        return mapping_->tensor_view(shape, offset, dtype);
    }
    void close() { mapping_->close(); }
    py::dict info() const { return mapping_->info(); }

private:
    std::shared_ptr<Mapping> mapping_;
    CUmemAllocationHandleType handle_type_;
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
    module.doc() = "CUDA Fabric host/GPU storage and lifetime-managed ATen tensor views.";
    module.attr("max_allocation_bytes") = kMaxAllocation;
    module.attr("max_load_chunk_bytes") = kMaxLoadChunk;
    module.attr("fabric_handle_bytes") = sizeof(CUmemFabricHandle);
    module.def("device_numa_nodes", &deviceNumaNodes, py::call_guard<py::gil_scoped_release>(),
               "Return each visible CUDA device's host NUMA node in ordinal order.");
    module.def("check_fabric_devices", &checkFabricDevices, py::arg("devices"),
               py::call_guard<py::gil_scoped_release>(), "Check Fabric support on the requested CUDA ordinals.");
    module.def("device_granularity", [](int device, const std::string& handle_type) {
        return Mapping::allocation_granularity(-1, device, true, handleType(handle_type, true));
    }, py::arg("device") = 0, py::arg("handle_type") = "fabric", py::call_guard<py::gil_scoped_release>());
    module.def("resource_counters", &resource_counters);
    module.def("host_numa_granularity", [](int numa, int device) { return Mapping::allocation_granularity(numa, device); },
               py::arg("numa"), py::arg("device"), py::call_guard<py::gil_scoped_release>());
    py::class_<FabricAllocation>(module, "FabricAllocation", "One local Fabric-exportable allocation.")
        .def(py::init<uint64_t, int, int, const std::string&, const std::string&, const std::string&>(), py::arg("size"), py::arg("numa"),
             py::arg("device"), py::arg("fail_stage") = "", py::arg("memory_kind") = "host", py::arg("handle_type") = "fabric")
        .def("load", &FabricAllocation::load, py::arg("data"), py::arg("offset") = 0)
        .def("export_handle", &FabricAllocation::export_handle)
        .def("export_fd", &FabricAllocation::export_fd, py::call_guard<py::gil_scoped_release>(),
             "Export a new close-on-exec FD owned by the caller; transfer it only with SCM_RIGHTS.")
        .def("info", &FabricAllocation::info)
        .def("close", &FabricAllocation::close, py::call_guard<py::gil_scoped_release>());
    py::class_<FabricMapping>(module, "FabricMapping", "One read-only CUDA VA span backed by ordered Fabric segments.")
        .def(py::init<int, const std::string&, const std::string&>(), py::arg("device"),
             py::arg("memory_kind") = "host", py::arg("handle_type") = "fabric")
        .def("attach", &FabricMapping::attach, py::arg("segments"), py::kw_only(),
             py::arg("_fail_stage") = "", py::arg("_fail_segment") = 0, py::arg("_fail_rollback") = false,
             "Map ordered (handle, bytes, granularity) tuples. Fabric uses bytes; POSIX_FD borrows local integer FDs.")
        .def("tensor_view", &FabricMapping::tensor_view, py::arg("shape"), py::arg("offset") = 0,
             py::kw_only(), py::arg("dtype") = at::kByte,
             "Return a read-only, zero-copy CUDA tensor view at a byte offset. The view retains the mapping.")
        .def("info", &FabricMapping::info)
        .def("close", &FabricMapping::close, py::call_guard<py::gil_scoped_release>(),
             "Drain submitted work and release the mapping. Release all tensor aliases before closing.");
}
