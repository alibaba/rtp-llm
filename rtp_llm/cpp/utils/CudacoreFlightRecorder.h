#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace rtp_llm {

// A small host-only history of the CUDA submissions nearest a fatal error.
// It never calls CUDA, allocates, or writes a file on the normal copy path.
enum class CudacoreFlightKind {
    BatchSubmit,
    BatchSubmitResult,
    BatchCompletion,
};

struct CudacoreFlightEvent {
    CudacoreFlightKind kind{CudacoreFlightKind::BatchSubmit};
    uint64_t           related_sequence{0};
    int                device_index{-1};
    uintptr_t          stream{0};
    uint64_t           tile_count{0};
    uint64_t           total_bytes{0};
    uintptr_t          first_dst{0};
    uintptr_t          first_src{0};
    uint64_t           first_bytes{0};
    uintptr_t          last_dst{0};
    uintptr_t          last_src{0};
    uint64_t           last_bytes{0};
    int                cuda_error{0};
};

// Returns zero if disabled or contended. A missed event never blocks CUDA work.
// RTP_LLM_CUDACORE_FLIGHT_RECORDER=0 disables the recorder; this diagnostic
// image enables it by default. The ring contains at most 256 short records.
void        prepareCudacoreFlightRecorder() noexcept;
uint64_t    recordCudacoreFlightEvent(const CudacoreFlightEvent& event) noexcept;
void        freezeCudacoreFlightRecorder() noexcept;
std::string snapshotCudacoreFlightRecorderJson() noexcept;

namespace cudacore_test {
void resetFlightRecorder() noexcept;
}

}  // namespace rtp_llm
