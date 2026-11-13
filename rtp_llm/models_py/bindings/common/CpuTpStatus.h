#pragma once
#include <cstddef>
#include <cstdint>

// The caller owns an aligned, shared mapping and initializes it before either
// rank publishes. Negative reduce results poison the group; never fall back to
// another collective after publication. A normal unmap does not poison peers.
extern "C" size_t cpu_status_size();
extern "C" int    cpu_status_initialize(void* pointer, size_t bytes);
extern "C" int    cpu_status_break(void* pointer);
extern "C" int    cpu_status_reduce(void* pointer, int rank, uint64_t generation, int local_ok, int timeout_ms);
