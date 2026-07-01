#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rapidjson/document.h"
#include "rtp_llm/cpp/cache/BlockInfo.h"

namespace rtp_llm {
namespace kvs {

struct KVSClientConfig {
    KVSClientConfig() = default;
    KVSClientConfig(std::string url, std::string socket_path, int timeout_ms, int lease_term_sec):
        v6d_url(std::move(url)),
        v6d_socket_path(std::move(socket_path)),
        timeout_ms(timeout_ms),
        lease_term_sec(lease_term_sec) {}

    std::string v6d_url;
    std::string v6d_socket_path;
    int         timeout_ms     = 12000;
    int         lease_term_sec = 60;
};

struct KVSObjectHandle {
    std::string object_key;
    size_t      bytes       = 0;
    uint64_t    data_offset = 0;
};

struct KVSReadSession {
    std::string                                      lease_id;
    std::unordered_map<std::string, KVSObjectHandle> handles;
    std::unordered_set<std::string>                  fetched_keys;
};

struct KVSObjectBuffer {
    std::string            object_key;
    std::vector<BlockInfo> iovs;
};

class KVSClient {
public:
    KVSClient();
    virtual ~KVSClient();

    virtual bool init(const KVSClientConfig& config);

    virtual std::optional<KVSReadSession> acquireForRead(const std::vector<std::string>& object_keys,
                                                         const std::string&              trace_id);

    virtual bool fetch(KVSReadSession&                 session,
                       const std::vector<std::string>& object_keys,
                       const std::string&              trace_id);

    virtual bool load(const KVSReadSession& session, const std::vector<KVSObjectBuffer>& dst_buffers);

    virtual bool complete(KVSReadSession&                 session,
                          const std::vector<std::string>& object_keys,
                          const std::string&              trace_id);

    virtual bool store(const std::vector<KVSObjectBuffer>& src_buffers, const std::string& trace_id);

    virtual void release(const std::string& lease_id);
    virtual void discard(const std::string& lease_id);

private:
    struct HttpResponse {
        long        status_code = 0;
        std::string body;
    };

    std::optional<HttpResponse> httpGet(const std::string& endpoint);
    std::optional<HttpResponse> httpPost(const std::string& endpoint, const std::string& payload);
    bool                        initMmap();
    bool                        copyFromObject(const KVSObjectHandle& handle, const std::vector<BlockInfo>& dst_iovs);
    bool                        copyToObject(const KVSObjectHandle& handle, const std::vector<BlockInfo>& src_iovs);
    std::optional<KVSObjectHandle> parseObjectHandle(const rapidjson::Value& object_value) const;

private:
    KVSClientConfig config_;
    void*           mmap_base_ = nullptr;
    size_t          mmap_size_ = 0;
    int             mmap_fd_   = -1;
    int             socket_fd_ = -1;
    bool            inited_    = false;
};

}  // namespace kvs
}  // namespace rtp_llm
