#include "rtp_llm/cpp/cache/V32AdmissionStore.h"

#include <chrono>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {
constexpr int64_t kReleaseGraceUs = 30LL * 1000 * 1000;

int64_t nowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void freeEntry(V32AdmissionStore::Entry& e) {
    if (e.host_kv != nullptr) {
        cudaFreeHost(e.host_kv);
        e.host_kv = nullptr;
    }
    if (e.idxp_dev != nullptr) {
        cudaFree(e.idxp_dev);
        e.idxp_dev = nullptr;
    }
}
}  // namespace

void V32AdmissionStore::reapLocked(int64_t now_us) {
    // caller holds mu_; free outside would be nicer but freeing pinned memory
    // under the lock is acceptable at this (per-request) frequency.
    size_t kept = 0;
    for (auto& item : graveyard_) {
        if (now_us - item.first >= kReleaseGraceUs) {
            freeEntry(*item.second);
        } else {
            graveyard_[kept++] = item;
        }
    }
    graveyard_.resize(kept);
}

V32AdmissionStore& V32AdmissionStore::instance() {
    static V32AdmissionStore store;
    return store;
}

cudaStream_t V32AdmissionStore::stream() {
    if (stream_ == nullptr) {
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    }
    return stream_;
}

bool V32AdmissionStore::prepare(int64_t key,
                                int32_t layers,
                                int64_t cap_tokens,
                                int64_t nb_cap,
                                int64_t kv_bytes_per_token,
                                int64_t idx_block_bytes,
                                int32_t seq_size_per_block,
                                int32_t device_id) {
    auto entry                = std::make_shared<Entry>();
    entry->cap_tokens         = cap_tokens;
    entry->nb_cap             = nb_cap;
    entry->kv_bytes_per_token = kv_bytes_per_token;
    entry->idx_block_bytes    = idx_block_bytes;
    entry->layers             = layers;
    entry->seq_size_per_block = seq_size_per_block;
    entry->device_id          = device_id;

    const size_t host_bytes = static_cast<size_t>(layers) * entry->hostLayerStride();
    const size_t dev_bytes  = static_cast<size_t>(layers) * entry->idxLayerStride();
    auto         rc         = cudaHostAlloc(&entry->host_kv, host_bytes, cudaHostAllocDefault);
    if (rc != cudaSuccess) {
        RTP_LLM_LOG_WARNING(
            "v32 admission: cudaHostAlloc %zu bytes failed: %s (key=%ld)", host_bytes, cudaGetErrorString(rc), key);
        return false;
    }
    rc = cudaMalloc(&entry->idxp_dev, dev_bytes);
    if (rc != cudaSuccess) {
        RTP_LLM_LOG_WARNING(
            "v32 admission: cudaMalloc %zu bytes failed: %s (key=%ld)", dev_bytes, cudaGetErrorString(rc), key);
        cudaFreeHost(entry->host_kv);
        entry->host_kv = nullptr;
        return false;
    }

    std::shared_ptr<Entry> old;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto                        it = map_.find(key);
        if (it != map_.end()) {
            old = it->second;  // block0 physical id recycled by a new request
        }
        map_[key]         = entry;
        const int64_t now = nowUs();
        if (old != nullptr) {
            graveyard_.emplace_back(now, old);
        }
        reapLocked(now);
    }
    if (old != nullptr) {
        RTP_LLM_LOG_INFO("v32 admission: key %ld recycled, stale mirror deferred to graveyard", key);
    }
    RTP_LLM_LOG_INFO("v32 admission: prepared mirror key=%ld layers=%d cap_tokens=%ld host=%.2fGB dev=%.2fGB",
                     key,
                     layers,
                     cap_tokens,
                     host_bytes / 1024.0 / 1024.0 / 1024.0,
                     dev_bytes / 1024.0 / 1024.0 / 1024.0);
    return true;
}

bool V32AdmissionStore::enqueueDrain(int64_t     key,
                                     int32_t     layer,
                                     const void* kv_src,
                                     int64_t     kv_bytes,
                                     const void* idx_src,
                                     int64_t     idx_bytes,
                                     int64_t     block_pos) {
    std::shared_ptr<Entry> entry;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto                        it = map_.find(key);
        if (it == map_.end()) {
            RTP_LLM_LOG_WARNING(
                "v32 admission: drain miss, no mirror entry for key=%ld (layer=%d pos=%ld)", key, layer, block_pos);
            return false;
        }
        entry = it->second;
    }
    if (layer < 0 || layer >= entry->layers || block_pos < 0 || block_pos >= entry->nb_cap
        || kv_bytes > entry->seq_size_per_block * entry->kv_bytes_per_token || idx_bytes > entry->idx_block_bytes) {
        RTP_LLM_LOG_WARNING("v32 admission: drain out of range key=%ld layer=%d pos=%ld kv=%ld idx=%ld",
                            key,
                            layer,
                            block_pos,
                            kv_bytes,
                            idx_bytes);
        return false;
    }
    auto  s        = stream();
    char* host_dst = static_cast<char*>(entry->host_kv) + static_cast<int64_t>(layer) * entry->hostLayerStride()
                     + block_pos * entry->seq_size_per_block * entry->kv_bytes_per_token;
    auto rc = cudaMemcpyAsync(host_dst, kv_src, kv_bytes, cudaMemcpyDeviceToHost, s);
    if (rc != cudaSuccess) {
        RTP_LLM_LOG_WARNING("v32 admission: drain kv memcpy failed key=%ld layer=%d pos=%ld: %s",
                            key,
                            layer,
                            block_pos,
                            cudaGetErrorString(rc));
        return false;
    }
    if (idx_src != nullptr && idx_bytes > 0) {
        char* dev_dst = static_cast<char*>(entry->idxp_dev) + static_cast<int64_t>(layer) * entry->idxLayerStride()
                        + block_pos * entry->idx_block_bytes;
        rc = cudaMemcpyAsync(dev_dst, idx_src, idx_bytes, cudaMemcpyDeviceToDevice, s);
        if (rc != cudaSuccess) {
            RTP_LLM_LOG_WARNING("v32 admission: drain idx memcpy failed key=%ld layer=%d pos=%ld: %s",
                                key,
                                layer,
                                block_pos,
                                cudaGetErrorString(rc));
            return false;
        }
    }
    return true;
}

bool V32AdmissionStore::sync() {
    if (stream_ == nullptr) {
        return true;
    }
    return cudaStreamSynchronize(stream_) == cudaSuccess;
}

void V32AdmissionStore::setDurable(int64_t key, int64_t tokens) {
    std::lock_guard<std::mutex> lk(mu_);
    auto                        it = map_.find(key);
    if (it != map_.end() && tokens > it->second->durable_tokens) {
        it->second->durable_tokens = tokens;
    }
}

std::shared_ptr<V32AdmissionStore::Entry> V32AdmissionStore::find(int64_t key) {
    std::lock_guard<std::mutex> lk(mu_);
    auto                        it = map_.find(key);
    return it == map_.end() ? nullptr : it->second;
}

void V32AdmissionStore::release(int64_t key) {
    std::lock_guard<std::mutex> lk(mu_);
    auto                        it  = map_.find(key);
    const int64_t               now = nowUs();
    if (it != map_.end()) {
        graveyard_.emplace_back(now, it->second);
        map_.erase(it);
        RTP_LLM_LOG_INFO("v32 admission: released mirror key=%ld (deferred %llds)",
                         key,
                         static_cast<long long>(kReleaseGraceUs / 1000000));
    }
    reapLocked(now);
}

}  // namespace rtp_llm

extern "C" int rtp_v32_admission_lookup(int64_t  key,
                                        int32_t  layer,
                                        void**   host_kv,
                                        int64_t* cap_tokens,
                                        int64_t* kv_bytes_per_token,
                                        void**   idxp_dev,
                                        int64_t* nb_cap,
                                        int64_t* idx_block_bytes,
                                        int64_t* durable_tokens,
                                        int32_t* device_id) {
    auto entry = rtp_llm::V32AdmissionStore::instance().find(key);
    if (entry == nullptr || layer < 0 || layer >= entry->layers) {
        return 0;
    }
    *host_kv            = static_cast<char*>(entry->host_kv) + static_cast<int64_t>(layer) * entry->hostLayerStride();
    *cap_tokens         = entry->cap_tokens;
    *kv_bytes_per_token = entry->kv_bytes_per_token;
    *idxp_dev           = static_cast<char*>(entry->idxp_dev) + static_cast<int64_t>(layer) * entry->idxLayerStride();
    *nb_cap             = entry->nb_cap;
    *idx_block_bytes    = entry->idx_block_bytes;
    *durable_tokens     = entry->durable_tokens;
    *device_id          = entry->device_id;
    return 1;
}

extern "C" void rtp_v32_admission_release(int64_t key) {
    rtp_llm::V32AdmissionStore::instance().release(key);
}
