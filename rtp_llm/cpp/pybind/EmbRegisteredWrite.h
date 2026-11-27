#pragma once

#if USING_CUDA
#include "rtp_llm/cpp/pybind/EmbNativeObjectApi.h"
#include "rtp_llm/cpp/pybind/EmbRegisteredArena.h"
#include "rtp_llm/cpp/pybind/EmbTensorCodec.h"
#include <c10/util/ScopeExit.h>
#include <chrono>

namespace rtp_llm {
using WriteClock = std::chrono::steady_clock;

enum class EmbWriteState {
    Success,
    Expired,
    LeaseUnavailable,
    ProducerFailure,
    CudaFailure,
    SdkFailure
};

struct EmbWriteResult {
    EmbWriteState state;
    int32_t       code            = 0;
    bool          inflight        = false;
    int64_t       readiness_ns    = 0;
    int64_t       acquire_ns      = 0;
    int64_t       pack_ns         = 0;
    int64_t       transfer_ns     = 0;
    int64_t       acquire_lock_ns = 0;
    int64_t       release_lock_ns = 0;
    int64_t       lease_ns        = 0;
};

inline EmbWriteResult writeEmbObject(EmbRegisteredArena&                       arena,
                                     const at::Tensor&                         root,
                                     int64_t                                   length,
                                     std::string                               prefix,
                                     int64_t                                   payload_offset,
                                     const std::vector<EmbTensorInput>&        tensors,
                                     const std::vector<EmbCudaSnapshotReady*>& ready,
                                     const KvcmObjectNativeApiV2&              api,
                                     const std::string&                        key,
                                     const std::string&                        trace,
                                     int64_t                                   budget_ns,
                                     bool                                      sample_locks,
                                     bool                                      wait_for_extent = false) {
    TORCH_CHECK(root.is_cuda() && root.is_contiguous() && root.scalar_type() == at::kByte && root.dim() == 1,
                "expected a registered CUDA byte arena");
    TORCH_CHECK(length > 0 && length <= root.numel() && budget_ns <= 600'000'000'000,
                "invalid write length or admission deadline");
    const auto started = WriteClock::now();
    if (budget_ns <= 0) {
        return {EmbWriteState::Expired};
    }
    const auto deadline = started + std::chrono::nanoseconds(budget_ns);
    try {
        for (auto* event : ready) {
            event->synchronize();
        }
    } catch (...) {
        return {EmbWriteState::ProducerFailure};
    }
    const auto lease_requested = WriteClock::now();
    const auto readiness_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(lease_requested - started).count();
    if (lease_requested >= deadline) {
        return {EmbWriteState::Expired, 0, false, readiness_ns};
    }

    // Python callers preserve their nonblocking lease attempt. The native
    // worker may wait up to the deadline, and arena stop wakes it on shutdown.
    const auto remaining = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - lease_requested).count();
    const auto [acquired, ticket, offset, extent, lock_ns] =
        arena.acquireForNativeWrite(length, wait_for_extent ? remaining : 0, sample_locks);
    if (acquired != EmbRegisteredArena::Result::Acquired) {
        return {EmbWriteState::LeaseUnavailable, 0, false, readiness_ns};
    }
    const auto held_since = WriteClock::now();
    bool       reusable   = true;
    auto       cleanup    = c10::make_scope_exit([&] { arena.releaseWrittenExtent(ticket, offset, reusable, false); });
    const auto acquire_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(held_since - lease_requested).count();
    auto       result = EmbWriteResult{EmbWriteState::CudaFailure, 0, true, readiness_ns, acquire_ns, 0, 0, lock_ns};
    if (held_since >= deadline) {
        result.state           = EmbWriteState::Expired;
        result.inflight        = false;
        result.release_lock_ns = arena.releaseWrittenExtent(ticket, offset, true, sample_locks);
        cleanup.release();
        return result;
    }

    std::optional<EmbTransportCudaStream> transport;
    try {
        const c10::cuda::CUDAGuard device(root.device());
        transport.emplace(root.get_device());
        const c10::cuda::CUDAStreamGuard stream(
            c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(transport->address()), root.get_device()));
        const auto output = root.narrow(0, offset, length);
        packEmbTensorSlices(output, std::move(prefix), payload_offset, tensors);
        waitEmbCudaStream(output);
        const auto pack_finished = WriteClock::now();
        result.pack_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(pack_finished - held_since).count();
        if (pack_finished >= deadline) {
            result.state    = EmbWriteState::Expired;
            result.inflight = false;
        } else {
            const KvcmObjectNativeSaveRequestV2 request{
                key.data(),
                key.size(),
                trace.data(),
                trace.size(),
                output.data_ptr(),
                static_cast<uint64_t>(length),
                1,
                0,
                0,
                0,
                nullptr,
                0,
            };
            const auto transfer = api.save(api.context, &request);
            result.transfer_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(WriteClock::now() - pack_finished).count();
            reusable        = !transfer.data_transfer_may_be_inflight;
            result.state    = transfer.code == 0 && reusable ? EmbWriteState::Success : EmbWriteState::SdkFailure;
            result.code     = transfer.code;
            result.inflight = !reusable;
        }
    } catch (...) {
        reusable = false;
        if (transport) {
            transport->quarantine();
        }
        arena.failDevice();
    }
    result.lease_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(WriteClock::now() - held_since).count();
    result.release_lock_ns = arena.releaseWrittenExtent(ticket, offset, reusable, sample_locks);
    cleanup.release();
    return result;
}

inline bool validEmbNativeWriteApi(const KvcmObjectNativeApiV2* api) noexcept {
    return api != nullptr && api->version == KVCM_OBJECT_NATIVE_API_VERSION_2 && api->struct_size >= sizeof(*api)
           && api->context != nullptr && api->save != nullptr;
}

inline void registerEmbRegisteredWrite(pybind11::module& m) {
    namespace py = pybind11;
    m.def("supports_registered_emb_write", [](py::handle capsule) {
        if (!PyCapsule_IsValid(capsule.ptr(), KVCM_OBJECT_NATIVE_API_CAPSULE_V2)) {
            return false;
        }
        return validEmbNativeWriteApi(static_cast<const KvcmObjectNativeApiV2*>(
            PyCapsule_GetPointer(capsule.ptr(), KVCM_OBJECT_NATIVE_API_CAPSULE_V2)));
    });
    py::enum_<EmbWriteState>(m, "EmbRegisteredWriteState")
        .value("Success", EmbWriteState::Success)
        .value("Expired", EmbWriteState::Expired)
        .value("LeaseUnavailable", EmbWriteState::LeaseUnavailable)
        .value("ProducerFailure", EmbWriteState::ProducerFailure)
        .value("CudaFailure", EmbWriteState::CudaFailure)
        .value("SdkFailure", EmbWriteState::SdkFailure);
    m.def("write_emb_registered_object",
          [](EmbRegisteredArena&                       arena,
             const at::Tensor&                         root,
             int64_t                                   length,
             std::string                               prefix,
             int64_t                                   payload_offset,
             const std::vector<EmbTensorInput>&        tensors,
             const std::vector<EmbCudaSnapshotReady*>& ready,
             py::capsule                               capsule,
             const std::string&                        key,
             const std::string&                        trace,
             int64_t                                   budget_ns,
             bool                                      sample_locks) {
              for (const auto* event : ready) {
                  if (!event) {
                      throw std::invalid_argument("null producer event");
                  }
              }
              const auto* api = static_cast<const KvcmObjectNativeApiV2*>(
                  PyCapsule_GetPointer(capsule.ptr(), KVCM_OBJECT_NATIVE_API_CAPSULE_V2));
              if (!api) {
                  throw py::error_already_set();
              }
              if (!validEmbNativeWriteApi(api)) {
                  throw std::invalid_argument("invalid exact-object native API");
              }
              EmbWriteResult result;
              {
                  py::gil_scoped_release release;
                  result = writeEmbObject(arena,
                                          root,
                                          length,
                                          std::move(prefix),
                                          payload_offset,
                                          tensors,
                                          ready,
                                          *api,
                                          key,
                                          trace,
                                          budget_ns,
                                          sample_locks);
              }
              return py::make_tuple(result.state,
                                    result.code,
                                    result.inflight,
                                    result.readiness_ns,
                                    result.acquire_ns,
                                    result.pack_ns,
                                    result.transfer_ns,
                                    result.acquire_lock_ns,
                                    result.release_lock_ns,
                                    result.lease_ns);
          });
}
}  // namespace rtp_llm
#endif
