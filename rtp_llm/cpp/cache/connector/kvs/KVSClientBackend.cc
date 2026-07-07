#include "rtp_llm/cpp/cache/connector/kvs/KVSClientBackend.h"

#include <cstring>
#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"
#if USING_CUDA || USING_ROCM
#include <cuda_runtime.h>
#endif

namespace rtp_llm {
namespace {

class KVSNativeClientImpl: public KVSNativeClient {
public:
    vineyard::Status init(const v6d::kvs::KVSClientConfig& config) override {
        return client_.init(config);
    }

    std::optional<v6d::kvs::LeaseHandle> get(const std::vector<std::string>& object_keys,
                                             const std::string&              peer,
                                             bool                            unsafe,
                                             const std::string&              trace_id) override {
        return client_.get(object_keys, peer, unsafe, trace_id);
    }

    std::optional<v6d::kvs::LeaseHandle> create(const std::vector<std::string>& object_keys,
                                                const std::vector<size_t>&      object_sizes,
                                                const std::string&              trace_id) override {
        return client_.create(object_keys, object_sizes, trace_id);
    }

    vineyard::Status fetch(v6d::kvs::LeaseHandle&         lease_handle,
                           const std::vector<std::string>& object_keys,
                           const std::string&              trace_id) override {
        return client_.fetch(lease_handle, object_keys, trace_id);
    }

    vineyard::Status complete(v6d::kvs::LeaseHandle&         lease_handle,
                              const std::vector<std::string>& object_keys,
                              const std::string&              trace_id) override {
        return client_.complete(lease_handle, object_keys, trace_id);
    }

    vineyard::Status release(const std::string& lease_id, const std::string& trace_id) override {
        return client_.release(lease_id, trace_id);
    }

    vineyard::Status discard(const std::string& lease_id, const std::string& trace_id) override {
        return client_.discard(lease_id, trace_id);
    }

    std::optional<v6d::kvs::BufferView> localBuffer(const v6d::kvs::ObjectHandle& handle) const override {
        return client_.localBuffer(handle);
    }

private:
    v6d::kvs::KVSClient client_;
};

bool copyBytes(void* dst, const void* src, size_t bytes, bool dst_cuda, bool src_cuda) {
    if (bytes == 0) {
        return true;
    }
    if (dst == nullptr || src == nullptr) {
        return false;
    }
    if (!dst_cuda && !src_cuda) {
        std::memcpy(dst, src, bytes);
        return true;
    }
#if USING_CUDA || USING_ROCM
    cudaMemcpyKind kind = cudaMemcpyDefault;
    if (dst_cuda && !src_cuda) {
        kind = cudaMemcpyHostToDevice;
    } else if (!dst_cuda && src_cuda) {
        kind = cudaMemcpyDeviceToHost;
    } else {
        kind = cudaMemcpyDeviceToDevice;
    }
    const auto err = cudaMemcpy(dst, src, bytes, kind);
    if (err != cudaSuccess) {
        RTP_LLM_LOG_WARNING("KVSClientBackend cudaMemcpy failed: %s", cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    (void) dst_cuda;
    (void) src_cuda;
    return false;
#endif
}

bool synchronizeCudaSources(const std::vector<KVSObjectBuffer>& src_buffers) {
    bool has_cuda_source = false;
    for (const auto& object : src_buffers) {
        for (const auto& buffer : object.buffers) {
            has_cuda_source = has_cuda_source || (buffer.is_cuda && buffer.size > 0);
        }
    }
    if (!has_cuda_source) {
        return true;
    }
#if USING_CUDA || USING_ROCM
    const auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        RTP_LLM_LOG_WARNING("KVSClientBackend failed to synchronize CUDA source buffers: %s",
                            cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    return false;
#endif
}

v6d::kvs::KVSClientConfig toClientConfig(const KVSConnectorConfig& config) {
    v6d::kvs::KVSClientConfig client_config;
    client_config.endpoint_url   = config.endpoint_url;
    client_config.socket_path    = config.socket_path;
    client_config.read_peer      = config.read_peer;
    client_config.timeout_ms     = config.timeout_ms;
    client_config.lease_term_sec = config.lease_term_sec;
    return client_config;
}

}  // namespace

KVSClientBackend::KVSClientBackend(KVSConnectorConfig config):
    KVSClientBackend(std::move(config), std::make_unique<KVSNativeClientImpl>()) {}

KVSClientBackend::KVSClientBackend(KVSConnectorConfig config, std::unique_ptr<KVSNativeClient> client):
    config_(std::move(config)), client_(std::move(client)) {}

bool KVSClientBackend::init() {
    std::lock_guard<std::mutex> lock(init_mutex_);
    if (inited_) {
        return true;
    }
    auto status = client_->init(toClientConfig(config_));
    if (!status.ok()) {
        RTP_LLM_LOG_WARNING("KVSClientBackend init failed: %s", status.message().c_str());
        return false;
    }
    inited_ = true;
    return true;
}

bool KVSClientBackend::ensureInit() {
    if (inited_) {
        return true;
    }
    return init();
}

std::optional<KVSReadHandle>
KVSClientBackend::get(const std::vector<std::string>& object_keys, const std::string& trace_id) {
    return getWithPeer(object_keys, config_.read_peer, trace_id);
}

std::optional<KVSReadHandle>
KVSClientBackend::getLocal(const std::vector<std::string>& object_keys, const std::string& trace_id) {
    return getWithPeer(object_keys, "local", trace_id);
}

std::optional<KVSReadHandle> KVSClientBackend::getWithPeer(const std::vector<std::string>& object_keys,
                                                           const std::string&              peer,
                                                           const std::string&              trace_id) {
    if (!ensureInit()) {
        return std::nullopt;
    }
    auto lease_handle = client_->get(object_keys, peer, false, trace_id);
    if (!lease_handle.has_value()) {
        return std::nullopt;
    }
    auto handle = makeReadHandle(*lease_handle);
    if (!handle.has_value()) {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(lease_mutex_);
    lease_handles_[handle->handle_id] = std::move(*lease_handle);
    return handle;
}

std::optional<KVSWriteHandle>
KVSClientBackend::getMutableLocal(const std::vector<std::string>& object_keys, const std::string& trace_id) {
    if (!ensureInit()) {
        return std::nullopt;
    }
    auto lease_handle = client_->get(object_keys, "local", true, trace_id);
    if (!lease_handle.has_value()) {
        return std::nullopt;
    }
    auto handle = makeWriteHandle(*lease_handle);
    if (!handle.has_value()) {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(lease_mutex_);
    lease_handles_[handle->handle_id] = std::move(*lease_handle);
    return handle;
}

std::optional<KVSWriteHandle> KVSClientBackend::create(const std::vector<std::string>& object_keys,
                                                        const std::vector<size_t>&      object_sizes,
                                                        const std::string&              trace_id) {
    if (!ensureInit()) {
        return std::nullopt;
    }
    auto lease_handle = client_->create(object_keys, object_sizes, trace_id);
    if (!lease_handle.has_value()) {
        return std::nullopt;
    }
    if (lease_handle->lease_id.empty() && lease_handle->object_handles.empty()
        && lease_handle->scope == "CREATE") {
        return KVSWriteHandle{};
    }
    auto handle = makeWriteHandle(*lease_handle);
    if (!handle.has_value()) {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(lease_mutex_);
    lease_handles_[handle->handle_id] = std::move(*lease_handle);
    return handle;
}

bool KVSClientBackend::fetch(const KVSReadHandle&            handle,
                             const std::vector<std::string>& object_keys,
                             const std::string&              trace_id) {
    if (!ensureInit()) {
        return false;
    }
    v6d::kvs::LeaseHandle lease_handle;
    {
        std::lock_guard<std::mutex> lock(lease_mutex_);
        auto iter = lease_handles_.find(handle.handle_id);
        if (iter == lease_handles_.end()) {
            return false;
        }
        lease_handle = iter->second;
    }
    auto status = client_->fetch(lease_handle, object_keys, trace_id);
    if (!status.ok()) {
        RTP_LLM_LOG_WARNING("KVSClientBackend fetch failed, lease_id: %s, error: %s",
                            handle.handle_id.c_str(),
                            status.message().c_str());
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(lease_mutex_);
        lease_handles_[handle.handle_id] = std::move(lease_handle);
    }
    return true;
}

bool KVSClientBackend::load(const KVSReadHandle& handle, const std::vector<KVSObjectBuffer>& dst_buffers) {
    std::unordered_map<std::string, v6d::kvs::ObjectHandle> handles;
    {
        std::lock_guard<std::mutex> lock(lease_mutex_);
        auto iter = lease_handles_.find(handle.handle_id);
        if (iter == lease_handles_.end()) {
            return false;
        }
        handles = iter->second.object_handles;
    }
    for (const auto& dst_buffer : dst_buffers) {
        auto handle_iter = handles.find(dst_buffer.object_key);
        if (handle_iter == handles.end()) {
            RTP_LLM_LOG_WARNING("KVSClientBackend load missing handle, key: %s", dst_buffer.object_key.c_str());
            return false;
        }
        if (!copyFromLocalBuffer(handle_iter->second, dst_buffer)) {
            RTP_LLM_LOG_WARNING("KVSClientBackend load copy failed, key: %s", dst_buffer.object_key.c_str());
            return false;
        }
    }
    return true;
}

bool KVSClientBackend::store(const KVSWriteHandle& handle, const std::vector<KVSObjectBuffer>& src_buffers) {
    // The write RPC runs on a connector worker thread. Make the model stream's
    // KV writes visible before copying the rank-local HBM segments to vineyard.
    if (!synchronizeCudaSources(src_buffers)) {
        return false;
    }
    std::unordered_map<std::string, v6d::kvs::ObjectHandle> handles;
    {
        std::lock_guard<std::mutex> lock(lease_mutex_);
        auto iter = lease_handles_.find(handle.handle_id);
        if (iter == lease_handles_.end()) {
            return false;
        }
        handles = iter->second.object_handles;
    }
    for (const auto& src_buffer : src_buffers) {
        auto handle_iter = handles.find(src_buffer.object_key);
        if (handle_iter == handles.end() || !copyToLocalBuffer(handle_iter->second, src_buffer)) {
            return false;
        }
    }
    return true;
}

bool KVSClientBackend::complete(const KVSReadHandle&            handle,
                                const std::vector<std::string>& object_keys,
                                const std::string&              trace_id) {
    return completeHandle(handle.handle_id, object_keys, trace_id);
}

bool KVSClientBackend::complete(const KVSWriteHandle&           handle,
                                const std::vector<std::string>& object_keys,
                                const std::string&              trace_id) {
    return completeHandle(handle.handle_id, object_keys, trace_id);
}

bool KVSClientBackend::completeHandle(const std::string&              handle_id,
                                      const std::vector<std::string>& object_keys,
                                      const std::string&              trace_id) {
    v6d::kvs::LeaseHandle lease_handle;
    {
        std::lock_guard<std::mutex> lock(lease_mutex_);
        auto iter = lease_handles_.find(handle_id);
        if (iter == lease_handles_.end()) {
            return false;
        }
        lease_handle = iter->second;
    }
    auto status = client_->complete(lease_handle, object_keys, trace_id);
    if (!status.ok()) {
        RTP_LLM_LOG_WARNING("KVSClientBackend complete failed, lease_id: %s, error: %s",
                            handle_id.c_str(),
                            status.message().c_str());
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(lease_mutex_);
        lease_handles_[handle_id] = std::move(lease_handle);
    }
    return true;
}

void KVSClientBackend::release(const KVSReadHandle& handle, const std::string& trace_id) {
    releaseHandle(handle.handle_id, trace_id);
}

void KVSClientBackend::release(const KVSWriteHandle& handle, const std::string& trace_id) {
    releaseHandle(handle.handle_id, trace_id);
}

void KVSClientBackend::releaseHandle(const std::string& handle_id, const std::string& trace_id) {
    if (handle_id.empty()) {
        return;
    }
    if (ensureInit()) {
        auto status = client_->release(handle_id, trace_id);
        if (!status.ok()) {
            RTP_LLM_LOG_WARNING("KVSClientBackend release failed, lease_id: %s, error: %s",
                                handle_id.c_str(),
                                status.message().c_str());
        }
    }
    std::lock_guard<std::mutex> lock(lease_mutex_);
    lease_handles_.erase(handle_id);
}

void KVSClientBackend::discard(const KVSWriteHandle& handle, const std::string& trace_id) {
    if (!handle.valid()) {
        return;
    }
    if (ensureInit()) {
        auto status = client_->discard(handle.handle_id, trace_id);
        if (!status.ok()) {
            RTP_LLM_LOG_WARNING("KVSClientBackend discard failed, lease_id: %s, error: %s",
                                handle.handle_id.c_str(),
                                status.message().c_str());
        }
    }
    std::lock_guard<std::mutex> lock(lease_mutex_);
    lease_handles_.erase(handle.handle_id);
}

std::optional<KVSReadHandle> KVSClientBackend::makeReadHandle(const v6d::kvs::LeaseHandle& lease_handle) const {
    if (lease_handle.lease_id.empty()) {
        return std::nullopt;
    }
    KVSReadHandle handle;
    handle.handle_id = lease_handle.lease_id;
    for (const auto& [key, _] : lease_handle.object_handles) {
        handle.object_keys.insert(key);
    }
    return handle;
}

std::optional<KVSWriteHandle> KVSClientBackend::makeWriteHandle(const v6d::kvs::LeaseHandle& lease_handle) const {
    if (lease_handle.lease_id.empty()) {
        return std::nullopt;
    }
    KVSWriteHandle handle;
    handle.handle_id = lease_handle.lease_id;
    for (const auto& [key, _] : lease_handle.object_handles) {
        handle.object_keys.insert(key);
    }
    return handle;
}

bool KVSClientBackend::copyFromLocalBuffer(const v6d::kvs::ObjectHandle& handle,
                                           const KVSObjectBuffer&         dst_buffer) {
    auto local = client_->localBuffer(handle);
    if (!local.has_value()) {
        RTP_LLM_LOG_WARNING("KVSClientBackend localBuffer missing, key: %s, handle_bytes: %zu",
                            handle.object_key.c_str(),
                            handle.bytes);
        return false;
    }
    return copySegments(local->addr, local->size, dst_buffer, true);
}

bool KVSClientBackend::copyToLocalBuffer(const v6d::kvs::ObjectHandle& handle,
                                         const KVSObjectBuffer&         src_buffer) {
    auto local = client_->localBuffer(handle);
    if (!local.has_value()) {
        return false;
    }
    return copySegments(local->addr, local->size, src_buffer, false);
}

bool KVSClientBackend::copySegments(uint64_t local_addr,
                                    size_t                 local_size,
                                    const KVSObjectBuffer& buffers,
                                    bool                   local_to_rtp) {
    size_t copied = 0;
    for (const auto& buffer : buffers.buffers) {
        if (buffer.size == 0) {
            continue;
        }
        const size_t local_offset = buffers.partial ? buffer.object_offset : copied;
        if (buffer.addr == 0 || local_offset + buffer.size > local_size) {
            RTP_LLM_LOG_WARNING("KVSClientBackend copySegments invalid segment, offset: %zu, buffer_size: %zu, "
                                "local_size: %zu, addr: %lu",
                                local_offset,
                                buffer.size,
                                local_size,
                                buffer.addr);
            return false;
        }
        auto* local_ptr = reinterpret_cast<void*>(local_addr + local_offset);
        auto* rtp_ptr   = reinterpret_cast<void*>(buffer.addr);
        const bool ok   = local_to_rtp ? copyBytes(rtp_ptr, local_ptr, buffer.size, buffer.is_cuda, false) :
                                         copyBytes(local_ptr, rtp_ptr, buffer.size, false, buffer.is_cuda);
        if (!ok) {
            return false;
        }
        copied += buffer.size;
    }
    if (!buffers.partial && copied != local_size) {
        RTP_LLM_LOG_WARNING("KVSClientBackend copySegments size mismatch, copied: %zu, local_size: %zu",
                            copied,
                            local_size);
        return false;
    }
    return true;
}

}  // namespace rtp_llm
