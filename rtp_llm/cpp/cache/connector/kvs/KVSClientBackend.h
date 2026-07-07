#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/util/status.h"
#include "rtp_llm/cpp/cache/connector/kvs/KVSConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/kvs/KVSObjectBackend.h"
#include "v6d/kvs/kvs_client.h"

namespace rtp_llm {

class KVSNativeClient {
public:
    virtual ~KVSNativeClient() = default;

    virtual vineyard::Status init(const v6d::kvs::KVSClientConfig& config) = 0;
    virtual std::optional<v6d::kvs::LeaseHandle> get(const std::vector<std::string>& object_keys,
                                                     const std::string&              peer,
                                                     bool                            unsafe,
                                                     const std::string&              trace_id) = 0;
    virtual std::optional<v6d::kvs::LeaseHandle> create(const std::vector<std::string>& object_keys,
                                                        const std::vector<size_t>&      object_sizes,
                                                        const std::string&              trace_id) = 0;
    virtual vineyard::Status fetch(v6d::kvs::LeaseHandle&         lease_handle,
                                   const std::vector<std::string>& object_keys,
                                   const std::string&              trace_id) = 0;
    virtual vineyard::Status complete(v6d::kvs::LeaseHandle&         lease_handle,
                                      const std::vector<std::string>& object_keys,
                                      const std::string&              trace_id) = 0;
    virtual vineyard::Status release(const std::string& lease_id, const std::string& trace_id) = 0;
    virtual vineyard::Status discard(const std::string& lease_id, const std::string& trace_id) = 0;
    virtual std::optional<v6d::kvs::BufferView> localBuffer(const v6d::kvs::ObjectHandle& handle) const = 0;
};

class KVSClientBackend: public KVSObjectBackend {
public:
    explicit KVSClientBackend(KVSConnectorConfig config);
    KVSClientBackend(KVSConnectorConfig config, std::unique_ptr<KVSNativeClient> client);
    ~KVSClientBackend() override = default;

    bool init();

    std::optional<KVSReadHandle> get(const std::vector<std::string>& object_keys,
                                     const std::string&              trace_id) override;
    std::optional<KVSReadHandle> getLocal(const std::vector<std::string>& object_keys,
                                          const std::string&              trace_id) override;
    std::optional<KVSWriteHandle> getMutableLocal(const std::vector<std::string>& object_keys,
                                                  const std::string&              trace_id) override;
    std::optional<KVSWriteHandle> create(const std::vector<std::string>& object_keys,
                                         const std::vector<size_t>&      object_sizes,
                                         const std::string&              trace_id) override;
    bool fetch(const KVSReadHandle&            handle,
               const std::vector<std::string>& object_keys,
               const std::string&              trace_id) override;
    bool load(const KVSReadHandle& handle, const std::vector<KVSObjectBuffer>& dst_buffers) override;
    bool store(const KVSWriteHandle& handle, const std::vector<KVSObjectBuffer>& src_buffers) override;
    bool complete(const KVSReadHandle&            handle,
                  const std::vector<std::string>& object_keys,
                  const std::string&              trace_id) override;
    bool complete(const KVSWriteHandle&           handle,
                  const std::vector<std::string>& object_keys,
                  const std::string&              trace_id) override;
    void release(const KVSReadHandle& handle, const std::string& trace_id) override;
    void release(const KVSWriteHandle& handle, const std::string& trace_id) override;
    void discard(const KVSWriteHandle& handle, const std::string& trace_id) override;

private:
    bool ensureInit();
    std::optional<KVSReadHandle> getWithPeer(const std::vector<std::string>& object_keys,
                                             const std::string&              peer,
                                             const std::string&              trace_id);
    std::optional<KVSReadHandle> makeReadHandle(const v6d::kvs::LeaseHandle& lease_handle) const;
    std::optional<KVSWriteHandle> makeWriteHandle(const v6d::kvs::LeaseHandle& lease_handle) const;
    bool completeHandle(const std::string&              handle_id,
                        const std::vector<std::string>& object_keys,
                        const std::string&              trace_id);
    void releaseHandle(const std::string& handle_id, const std::string& trace_id);
    bool copyFromLocalBuffer(const v6d::kvs::ObjectHandle& handle, const KVSObjectBuffer& dst_buffer);
    bool copyToLocalBuffer(const v6d::kvs::ObjectHandle& handle, const KVSObjectBuffer& src_buffer);
    bool copySegments(uint64_t local_addr, size_t local_size, const KVSObjectBuffer& buffers, bool local_to_rtp);

private:
    KVSConnectorConfig               config_;
    std::unique_ptr<KVSNativeClient> client_;
    std::atomic<bool>                inited_{false};

    std::mutex                                             init_mutex_;
    std::mutex                                             lease_mutex_;
    std::unordered_map<std::string, v6d::kvs::LeaseHandle> lease_handles_;
};

}  // namespace rtp_llm
