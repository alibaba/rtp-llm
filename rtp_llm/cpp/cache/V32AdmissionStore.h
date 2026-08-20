#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace rtp_llm {

// Admission-time mirror store for the v32 decode KV offload (staging-ring
// admission). Owned by the engine: DecodeRpcServer::loadCache fills it while
// pulling the prefix through the staging ring; the python-side hook adopts the
// buffers (via the exported C symbol below) instead of mirroring the prefix
// itself during the first decode steps.
//
// Keyed by the request's block0 physical block id: that is the only stable
// identity the python hook can observe from the kernel block table.
class V32AdmissionStore {
public:
    struct Entry {
        void*   host_kv            = nullptr;  // pinned: layers * cap_tokens * kv_bytes_per_token
        void*   idxp_dev           = nullptr;  // device: layers * nb_cap * idx_block_bytes
        int64_t cap_tokens         = 0;
        int64_t nb_cap             = 0;
        int64_t kv_bytes_per_token = 0;
        int64_t idx_block_bytes    = 0;
        int64_t durable_tokens     = 0;  // contiguous-from-0 watermark
        int32_t layers             = 0;
        int32_t seq_size_per_block = 0;
        int32_t device_id          = -1;

        int64_t hostLayerStride() const {
            return cap_tokens * kv_bytes_per_token;
        }
        int64_t idxLayerStride() const {
            return nb_cap * idx_block_bytes;
        }
    };

    static V32AdmissionStore& instance();

    // Allocate (or re-allocate on key reuse) the mirror for one request.
    // Returns false on allocation failure; the caller must then fail admission.
    bool prepare(int64_t key,
                 int32_t layers,
                 int64_t cap_tokens,
                 int64_t nb_cap,
                 int64_t kv_bytes_per_token,
                 int64_t idx_block_bytes,
                 int32_t seq_size_per_block,
                 int32_t device_id);

    // Enqueue one ring-block drain on the internal stream (no sync):
    //   kv_src  (device) -> host_kv[layer]  at token position block_pos*spb
    //   idx_src (device) -> idxp_dev[layer] at block slot block_pos
    bool enqueueDrain(int64_t     key,
                      int32_t     layer,
                      const void* kv_src,
                      int64_t     kv_bytes,
                      const void* idx_src,
                      int64_t     idx_bytes,
                      int64_t     block_pos);

    // Block until all enqueued drains are durable.
    bool sync();

    // Advance the contiguous durable watermark (tokens from position 0).
    void setDurable(int64_t key, int64_t tokens);

    std::shared_ptr<Entry> find(int64_t key);

    void release(int64_t key);

private:
    V32AdmissionStore() = default;

    cudaStream_t stream();
    // Free graveyard entries older than the grace window. The python-side
    // fetch thread may still read the host mirror for a step or two after the
    // stream is released (zero-sync design reads last-step metadata), so the
    // actual cudaFreeHost/cudaFree is deferred.
    void reapLocked(int64_t now_us);

    std::mutex                                              mu_;
    std::unordered_map<int64_t, std::shared_ptr<Entry>>     map_;
    std::vector<std::pair<int64_t, std::shared_ptr<Entry>>> graveyard_;  // (release_time_us, entry)
    cudaStream_t                                            stream_ = nullptr;
};

}  // namespace rtp_llm

// C export for the python-side torch extension (v32_ctx.so) to adopt the
// admission mirror without a pybind dependency on the wheel. Looked up with
// dlopen(libth_transformer.so)+dlsym.
extern "C" {
// Returns 1 and fills the out params if an entry exists for (key, layer).
int rtp_v32_admission_lookup(int64_t  key,
                             int32_t  layer,
                             void**   host_kv,
                             int64_t* cap_tokens,
                             int64_t* kv_bytes_per_token,
                             void**   idxp_dev,
                             int64_t* nb_cap,
                             int64_t* idx_block_bytes,
                             int64_t* durable_tokens,
                             int32_t* device_id);
// Release the mirror for one request (idempotent). Worker ranks have no
// GenerateStream, so the python hook's purge drives this there.
void rtp_v32_admission_release(int64_t key);
}
