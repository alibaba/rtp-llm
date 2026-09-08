#include "rtp_llm/cpp/cache/V32AdmissionStore.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {
constexpr int64_t kReleaseGraceUs = 30LL * 1000 * 1000;

size_t poolMaxBytes() {
    static const size_t value = []() {
        const char* env = std::getenv("RTP_KV_ADMISSION_HOST_POOL_GB");
        const auto  gb  = env ? std::max(atoll(env), 0LL) : 12LL;
        return static_cast<size_t>(gb) * 1024 * 1024 * 1024;
    }();
    return value;
}

int64_t nowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void freeEntry(V32AdmissionStore::Entry& e) {
    if (e.host_kv != nullptr) {
        cudaFreeHost(e.host_kv);
        e.host_kv = nullptr;
    }
}
}  // namespace

void V32AdmissionStore::reapLocked(int64_t now_us) {
    size_t kept = 0;
    for (auto& item : graveyard_) {
        auto& entry = item.second;
        if (now_us - item.first < kReleaseGraceUs) {
            graveyard_[kept++] = item;
            continue;
        }
        entry->durable_tokens = 0;
        entry->generation     = 0;
        entry->prepare_us     = 0;
        entry->pool_hit       = false;
        if (entry->host_kv != nullptr && pooled_bytes_ + entry->allocation_bytes <= poolMaxBytes()) {
            pooled_bytes_ += entry->allocation_bytes;
            free_pool_[entry->allocation_bytes].push_back(std::move(entry));
        } else {
            freeEntry(*entry);
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
                                int64_t kv_bytes_per_token,
                                int32_t seq_size_per_block,
                                int32_t device_id) {
    const int64_t          start_us   = nowUs();
    const size_t           host_bytes = static_cast<size_t>(layers) * cap_tokens * kv_bytes_per_token;
    std::shared_ptr<Entry> entry;
    {
        std::lock_guard<std::mutex> lk(mu_);
        reapLocked(start_us);
        auto pool_it = free_pool_.find(host_bytes);
        if (pool_it != free_pool_.end() && !pool_it->second.empty()) {
            entry = std::move(pool_it->second.back());
            pool_it->second.pop_back();
            pooled_bytes_ -= host_bytes;
            entry->pool_hit = true;
        }
    }
    if (!entry) {
        entry                   = std::make_shared<Entry>();
        entry->allocation_bytes = host_bytes;
        auto rc                 = cudaHostAlloc(&entry->host_kv, host_bytes, cudaHostAllocDefault);
        if (rc != cudaSuccess) {
            RTP_LLM_LOG_WARNING(
                "v32 admission: cudaHostAlloc %zu bytes failed: %s (key=%ld)", host_bytes, cudaGetErrorString(rc), key);
            return false;
        }
    }
    entry->cap_tokens         = cap_tokens;
    entry->kv_bytes_per_token = kv_bytes_per_token;
    entry->durable_tokens     = 0;
    entry->layers             = layers;
    entry->seq_size_per_block = seq_size_per_block;
    entry->device_id          = device_id;
    entry->prepare_us         = nowUs() - start_us;

    std::shared_ptr<Entry> old;
    size_t                 pooled_after = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto                        it = map_.find(key);
        if (it != map_.end()) {
            old = it->second;
        }
        entry->generation = ++next_generation_;
        map_[key]         = entry;
        const int64_t now = nowUs();
        if (old != nullptr) {
            graveyard_.emplace_back(now, old);
        }
        reapLocked(now);
        pooled_after = pooled_bytes_;
    }
    if (old != nullptr) {
        RTP_LLM_LOG_INFO("v32 admission: key %ld recycled, stale mirror deferred to graveyard", key);
    }
    RTP_LLM_LOG_INFO("v32 admission: prepared host mirror key=%ld generation=%ld layers=%d cap_tokens=%ld host=%.2fGB "
                     "pool_hit=%d prepare=%ldms pooled=%.2fGB",
                     key,
                     entry->generation,
                     layers,
                     cap_tokens,
                     host_bytes / 1024.0 / 1024.0 / 1024.0,
                     static_cast<int>(entry->pool_hit),
                     entry->prepare_us / 1000,
                     pooled_after / 1024.0 / 1024.0 / 1024.0);
    return true;
}

bool V32AdmissionStore::prepareAsync(int64_t key,
                                     int32_t layers,
                                     int64_t cap_tokens,
                                     int64_t kv_bytes_per_token,
                                     int32_t seq_size_per_block,
                                     int32_t device_id) {
    std::lock_guard<std::mutex> lk(mu_);
    if (map_.count(key) || pending_.count(key)) {
        return true;
    }
    pending_[key] =
        std::async(std::launch::async,
                   [this, key, layers, cap_tokens, kv_bytes_per_token, seq_size_per_block, device_id]() {
                       cudaSetDevice(device_id);
                       return prepare(key, layers, cap_tokens, kv_bytes_per_token, seq_size_per_block, device_id);
                   })
            .share();
    return true;
}

bool V32AdmissionStore::waitPrepared(int64_t key, int64_t* wait_us) {
    std::shared_future<bool> pending;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto                        it = pending_.find(key);
        if (it == pending_.end()) {
            return map_.count(key) != 0;
        }
        pending = it->second;
    }
    const int64_t start_us = nowUs();
    bool          ok       = false;
    try {
        ok = pending.get();
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("v32 admission: async prepare failed for key=%ld: %s", key, e.what());
    }
    if (wait_us != nullptr) {
        *wait_us = nowUs() - start_us;
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        pending_.erase(key);
    }
    return ok;
}

bool V32AdmissionStore::enqueueDrain(
    int64_t key, int32_t layer, const void* kv_src, int64_t kv_bytes, int64_t block_pos) {
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
    const int64_t max_blocks = entry->cap_tokens / entry->seq_size_per_block;
    if (layer < 0 || layer >= entry->layers || block_pos < 0 || block_pos >= max_blocks
        || kv_bytes > entry->seq_size_per_block * entry->kv_bytes_per_token) {
        RTP_LLM_LOG_WARNING(
            "v32 admission: drain out of range key=%ld layer=%d pos=%ld kv=%ld", key, layer, block_pos, kv_bytes);
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
    waitPrepared(key);
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

void V32AdmissionStore::releaseIfGeneration(int64_t key, int64_t generation) {
    std::lock_guard<std::mutex> lk(mu_);
    auto                        it  = map_.find(key);
    const int64_t               now = nowUs();
    if (it != map_.end() && it->second->generation == generation) {
        graveyard_.emplace_back(now, it->second);
        map_.erase(it);
        RTP_LLM_LOG_INFO("v32 admission: released mirror key=%ld generation=%ld (deferred %llds)",
                         key,
                         generation,
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
    // ABI-compatible fields: idxp remains retired; nb_cap now transports the
    // admission generation to the matching v32_ctx extension.
    *idxp_dev        = nullptr;
    *nb_cap          = entry->generation;
    *idx_block_bytes = 0;
    *durable_tokens  = entry->durable_tokens;
    *device_id       = entry->device_id;
    return 1;
}

extern "C" void rtp_v32_admission_release(int64_t key) {
    rtp_llm::V32AdmissionStore::instance().release(key);
}

extern "C" int64_t rtp_v32_admission_generation(int64_t key) {
    auto entry = rtp_llm::V32AdmissionStore::instance().find(key);
    return entry == nullptr ? -1 : entry->generation;
}

extern "C" void rtp_v32_admission_release_generation(int64_t key, int64_t generation) {
    rtp_llm::V32AdmissionStore::instance().releaseIfGeneration(key, generation);
}
