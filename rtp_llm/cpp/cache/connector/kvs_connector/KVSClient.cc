#include "rtp_llm/cpp/cache/connector/kvs_connector/KVSClient.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_set>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "curl/curl.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "rtp_llm/cpp/utils/Logger.h"

#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#endif

namespace rtp_llm::kvs {
namespace {

constexpr uint64_t kMaxJsonMessageBytes = 16ULL * 1024ULL * 1024ULL;

void curlGlobalInitOnce() {
    static std::once_flag once;
    std::call_once(once, []() { curl_global_init(CURL_GLOBAL_ALL); });
}

size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* output = static_cast<std::string*>(userdata);
    output->append(ptr, size * nmemb);
    return size * nmemb;
}

std::string joinUrl(std::string base, const std::string& endpoint) {
    while (!base.empty() && base.back() == '/') {
        base.pop_back();
    }
    if (!endpoint.empty() && endpoint.front() == '/') {
        return base + endpoint;
    }
    return base + "/" + endpoint;
}

bool sendAll(int fd, const void* data, size_t bytes) {
    const char* ptr = static_cast<const char*>(data);
    while (bytes > 0) {
        ssize_t sent = ::send(fd, ptr, bytes, 0);
        if (sent <= 0) {
            return false;
        }
        ptr += sent;
        bytes -= static_cast<size_t>(sent);
    }
    return true;
}

bool recvAll(int fd, void* data, size_t bytes) {
    char* ptr = static_cast<char*>(data);
    while (bytes > 0) {
        ssize_t got = ::recv(fd, ptr, bytes, 0);
        if (got <= 0) {
            return false;
        }
        ptr += got;
        bytes -= static_cast<size_t>(got);
    }
    return true;
}

bool sendJsonMessage(int fd, const std::string& message) {
    uint64_t len = message.size();
    return sendAll(fd, &len, sizeof(len)) && sendAll(fd, message.data(), message.size());
}

std::optional<std::string> recvJsonMessage(int fd) {
    uint64_t len = 0;
    if (!recvAll(fd, &len, sizeof(len))) {
        return std::nullopt;
    }
    if (len > kMaxJsonMessageBytes) {
        RTP_LLM_LOG_WARNING("KVSClient recv message too large: %llu", static_cast<unsigned long long>(len));
        return std::nullopt;
    }
    std::string message;
    message.resize(static_cast<size_t>(len));
    if (message.empty()) {
        return message;
    }
    if (!recvAll(fd, &message[0], message.size())) {
        return std::nullopt;
    }
    return message;
}

size_t totalBytes(const std::vector<BlockInfo>& iovs) {
    size_t bytes = 0;
    for (const auto& iov : iovs) {
        if (iov.size_bytes > std::numeric_limits<size_t>::max() - bytes) {
            return std::numeric_limits<size_t>::max();
        }
        bytes += iov.size_bytes;
    }
    return bytes;
}

bool copyHostAndDevice(void* dst, const void* src, size_t bytes, bool dst_cuda, bool src_cuda) {
    if (bytes == 0) {
        return true;
    }
    if (!dst || !src) {
        return false;
    }
    if (!dst_cuda && !src_cuda) {
        std::memcpy(dst, src, bytes);
        return true;
    }
#if USING_CUDA || USING_ROCM
    cudaMemcpyKind kind = cudaMemcpyDefault;
    if (dst_cuda && src_cuda) {
        kind = cudaMemcpyDeviceToDevice;
    } else if (dst_cuda) {
        kind = cudaMemcpyHostToDevice;
    } else if (src_cuda) {
        kind = cudaMemcpyDeviceToHost;
    }
    auto err = cudaMemcpy(dst, src, bytes, kind);
    if (err != cudaSuccess) {
        RTP_LLM_LOG_WARNING("KVSClient cudaMemcpy failed: %s", cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    RTP_LLM_LOG_WARNING("KVSClient copy requires GPU runtime, dst_cuda[%d], src_cuda[%d]", dst_cuda, src_cuda);
    return false;
#endif
}

std::optional<uint64_t> getJsonUint64(const rapidjson::Value& value) {
    if (value.IsUint64()) {
        return value.GetUint64();
    }
    if (value.IsInt64() && value.GetInt64() >= 0) {
        return static_cast<uint64_t>(value.GetInt64());
    }
    if (value.IsUint()) {
        return value.GetUint();
    }
    if (value.IsInt() && value.GetInt() >= 0) {
        return static_cast<uint64_t>(value.GetInt());
    }
    return std::nullopt;
}

void writeStringArray(rapidjson::Writer<rapidjson::StringBuffer>& writer,
                      const char*                                 name,
                      const std::vector<std::string>&             values) {
    writer.Key(name);
    writer.StartArray();
    for (const auto& value : values) {
        writer.String(value.c_str());
    }
    writer.EndArray();
}

}  // namespace

KVSClient::KVSClient() = default;

KVSClient::~KVSClient() {
    if (mmap_base_ && mmap_size_ > 0) {
        ::munmap(mmap_base_, mmap_size_);
    }
    if (mmap_fd_ >= 0) {
        ::close(mmap_fd_);
    }
    if (socket_fd_ >= 0) {
        ::close(socket_fd_);
    }
}

bool KVSClient::init(const KVSClientConfig& config) {
    config_ = config;
    if (config_.v6d_url.empty()) {
        RTP_LLM_LOG_WARNING("KVSClient init failed, v6d url is empty");
        return false;
    }
    auto health = httpGet("health");
    if (!health || health->status_code != 200) {
        RTP_LLM_LOG_WARNING("KVSClient health check failed, url[%s], status[%ld]",
                            config_.v6d_url.c_str(),
                            health ? health->status_code : -1);
        return false;
    }
    if (!initMmap()) {
        return false;
    }
    inited_ = true;
    RTP_LLM_LOG_INFO("KVSClient initialized, v6d_url[%s], socket_path[%s], mmap_base[%p], mmap_size[%zu]",
                     config_.v6d_url.c_str(),
                     config_.v6d_socket_path.c_str(),
                     mmap_base_,
                     mmap_size_);
    return true;
}

std::optional<KVSClient::HttpResponse> KVSClient::httpGet(const std::string& endpoint) {
    curlGlobalInitOnce();
    CURL* curl = curl_easy_init();
    if (!curl) {
        return std::nullopt;
    }
    std::string response;
    std::string url = joinUrl(config_.v6d_url, endpoint);
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, config_.timeout_ms);
    CURLcode rc     = curl_easy_perform(curl);
    long     status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_easy_cleanup(curl);
    if (rc != CURLE_OK) {
        RTP_LLM_LOG_WARNING("KVSClient GET [%s] failed: %s", url.c_str(), curl_easy_strerror(rc));
        return std::nullopt;
    }
    return HttpResponse{status, std::move(response)};
}

std::optional<KVSClient::HttpResponse> KVSClient::httpPost(const std::string& endpoint, const std::string& payload) {
    curlGlobalInitOnce();
    CURL* curl = curl_easy_init();
    if (!curl) {
        return std::nullopt;
    }
    std::string        response;
    std::string        url     = joinUrl(config_.v6d_url, endpoint);
    struct curl_slist* headers = nullptr;
    headers                    = curl_slist_append(headers, "Content-Type: application/json");
    headers                    = curl_slist_append(headers, "Accept: application/json");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, payload.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, config_.timeout_ms);
    CURLcode rc     = curl_easy_perform(curl);
    long     status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    if (rc != CURLE_OK) {
        RTP_LLM_LOG_WARNING("KVSClient POST [%s] failed: %s", url.c_str(), curl_easy_strerror(rc));
        return std::nullopt;
    }
    return HttpResponse{status, std::move(response)};
}

bool KVSClient::initMmap() {
    if (config_.v6d_socket_path.empty()) {
        RTP_LLM_LOG_WARNING("KVSClient init mmap failed, v6d socket path is empty");
        return false;
    }
    socket_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (socket_fd_ < 0) {
        RTP_LLM_LOG_WARNING("KVSClient create unix socket failed");
        return false;
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", config_.v6d_socket_path.c_str());
    if (::connect(socket_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        RTP_LLM_LOG_WARNING("KVSClient connect vineyard socket [%s] failed", config_.v6d_socket_path.c_str());
        return false;
    }

    const std::string register_request =
        R"({"type":"register_request","version":"0.0.0","store_type":0,"session_id":0,"username":"","password":""})";
    if (!sendJsonMessage(socket_fd_, register_request)) {
        RTP_LLM_LOG_WARNING("KVSClient send register_request failed");
        return false;
    }
    auto register_reply = recvJsonMessage(socket_fd_);
    if (!register_reply || register_reply->find("register_reply") == std::string::npos) {
        RTP_LLM_LOG_WARNING("KVSClient bad register_reply [%s]", register_reply ? register_reply->c_str() : "");
        return false;
    }
    if (!sendJsonMessage(socket_fd_, R"({"type":"get_vineyard_mmap_fd_request"})")) {
        RTP_LLM_LOG_WARNING("KVSClient send get_vineyard_mmap_fd_request failed");
        return false;
    }
    auto fd_reply = recvJsonMessage(socket_fd_);
    if (!fd_reply) {
        RTP_LLM_LOG_WARNING("KVSClient receive get_vineyard_mmap_fd_reply failed");
        return false;
    }
    rapidjson::Document doc;
    doc.Parse(fd_reply->c_str());
    if (doc.HasParseError() || !doc.IsObject() || !doc.HasMember("size")) {
        RTP_LLM_LOG_WARNING("KVSClient invalid mmap reply [%s]", fd_reply->c_str());
        return false;
    }
    auto mmap_size = getJsonUint64(doc["size"]);
    if (!mmap_size) {
        RTP_LLM_LOG_WARNING("KVSClient invalid mmap size reply [%s]", fd_reply->c_str());
        return false;
    }
    mmap_size_ = static_cast<size_t>(*mmap_size);

    char   control_buffer[CMSG_SPACE(sizeof(int))] = {};
    char   data_buffer[1]                          = {};
    iovec  iov{data_buffer, sizeof(data_buffer)};
    msghdr msg{};
    msg.msg_iov        = &iov;
    msg.msg_iovlen     = 1;
    msg.msg_control    = control_buffer;
    msg.msg_controllen = sizeof(control_buffer);
    if (::recvmsg(socket_fd_, &msg, 0) < 0) {
        RTP_LLM_LOG_WARNING("KVSClient receive mmap fd failed");
        return false;
    }
    for (cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
            std::memcpy(&mmap_fd_, CMSG_DATA(cmsg), sizeof(int));
            break;
        }
    }
    if (mmap_fd_ < 0) {
        RTP_LLM_LOG_WARNING("KVSClient no mmap fd in SCM_RIGHTS");
        return false;
    }
    mmap_base_ = ::mmap(nullptr, mmap_size_, PROT_READ | PROT_WRITE, MAP_SHARED, mmap_fd_, 0);
    if (mmap_base_ == MAP_FAILED) {
        mmap_base_ = nullptr;
        RTP_LLM_LOG_WARNING("KVSClient mmap failed, size[%zu]", mmap_size_);
        return false;
    }
    return true;
}

std::optional<KVSObjectHandle> KVSClient::parseObjectHandle(const rapidjson::Value& object_value) const {
    if (!object_value.IsObject() || !object_value.HasMember("key") || !object_value["key"].IsString()
        || !object_value.HasMember("blobs") || !object_value["blobs"].IsArray() || object_value["blobs"].Empty()) {
        return std::nullopt;
    }
    const auto& blob = object_value["blobs"][0];
    if (!blob.IsObject() || !blob.HasMember("type") || !blob["type"].IsString()
        || std::string(blob["type"].GetString()) != "vineyard" || !blob.HasMember("handle")
        || !blob["handle"].IsObject()) {
        return std::nullopt;
    }
    const auto& handle = blob["handle"];
    if (!handle.HasMember("data_offset") || !handle.HasMember("data_size")) {
        return std::nullopt;
    }
    auto data_offset = getJsonUint64(handle["data_offset"]);
    auto data_size   = getJsonUint64(handle["data_size"]);
    if (!data_offset || !data_size) {
        return std::nullopt;
    }
    if (handle.HasMember("object_id")) {
        auto object_id = getJsonUint64(handle["object_id"]);
        if (!object_id || *object_id == std::numeric_limits<uint64_t>::max()) {
            return std::nullopt;
        }
    }
    KVSObjectHandle result;
    result.object_key  = object_value["key"].GetString();
    result.data_offset = *data_offset;
    result.bytes       = static_cast<size_t>(*data_size);
    if (object_value.HasMember("meta") && object_value["meta"].IsObject()) {
        const auto& meta = object_value["meta"];
        if (meta.HasMember("size")) {
            auto meta_size = getJsonUint64(meta["size"]);
            if (meta_size && *meta_size > 0) {
                result.bytes = static_cast<size_t>(*meta_size);
            }
        }
    }
    if (result.data_offset > mmap_size_ || result.bytes > mmap_size_ - result.data_offset) {
        RTP_LLM_LOG_WARNING("KVSClient object [%s] out of mmap range, offset[%llu], bytes[%zu], mmap_size[%zu]",
                            result.object_key.c_str(),
                            static_cast<unsigned long long>(result.data_offset),
                            result.bytes,
                            mmap_size_);
        return std::nullopt;
    }
    return result;
}

std::optional<KVSReadSession> KVSClient::acquireForRead(const std::vector<std::string>& object_keys,
                                                        const std::string&              trace_id) {
    if (!inited_ || object_keys.empty()) {
        return KVSReadSession{};
    }
    RTP_LLM_LOG_INFO(
        "KVSClient acquire READ start, trace_id[%s], object_count[%zu]", trace_id.c_str(), object_keys.size());
    rapidjson::StringBuffer                    buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.Key("scope");
    writer.String("read");
    writeStringArray(writer, "object_keys", object_keys);
    writer.Key("term");
    writer.Int(config_.lease_term_sec);
    writer.Key("peer");
    writer.String("local");
    writer.Key("wait_timeout");
    writer.Int(0);
    writer.Key("request_id");
    writer.String(trace_id.c_str());
    writer.Key("best_effort");
    writer.Bool(true);
    writer.EndObject();

    auto response = httpPost("acquire", buffer.GetString());
    if (!response || response->status_code != 200) {
        if (response) {
            RTP_LLM_LOG_WARNING(
                "KVSClient acquire READ failed, status[%ld], body[%s]", response->status_code, response->body.c_str());
        }
        return std::nullopt;
    }
    rapidjson::Document doc;
    doc.Parse(response->body.c_str());
    if (doc.HasParseError() || !doc.IsObject() || !doc.HasMember("lease") || !doc.HasMember("objects")) {
        RTP_LLM_LOG_WARNING("KVSClient parse acquire READ response failed, body[%s]", response->body.c_str());
        return std::nullopt;
    }
    KVSReadSession session;
    if (doc["lease"].IsObject() && doc["lease"].HasMember("lease_id") && doc["lease"]["lease_id"].IsString()) {
        session.lease_id = doc["lease"]["lease_id"].GetString();
    }
    const auto& objects = doc["objects"];
    if (!objects.IsArray()) {
        return std::nullopt;
    }
    for (auto it = objects.Begin(); it != objects.End(); ++it) {
        if (it->IsNull()) {
            continue;
        }
        auto handle = parseObjectHandle(*it);
        if (handle) {
            session.handles.emplace(handle->object_key, *handle);
        }
    }
    RTP_LLM_LOG_INFO("KVSClient acquire READ done, trace_id[%s], requested[%zu], handles[%zu], lease[%s]",
                     trace_id.c_str(),
                     object_keys.size(),
                     session.handles.size(),
                     session.lease_id.c_str());
    return session;
}

bool KVSClient::copyFromObject(const KVSObjectHandle& handle, const std::vector<BlockInfo>& dst_iovs) {
    const char* src    = static_cast<const char*>(mmap_base_) + handle.data_offset;
    size_t      offset = 0;
    for (const auto& dst : dst_iovs) {
        if (offset > handle.bytes || dst.size_bytes > handle.bytes - offset) {
            return false;
        }
        if (!copyHostAndDevice(dst.addr, src + offset, dst.size_bytes, dst.is_cuda, false)) {
            return false;
        }
        offset += dst.size_bytes;
    }
    return offset == handle.bytes;
}

bool KVSClient::copyToObject(const KVSObjectHandle& handle, const std::vector<BlockInfo>& src_iovs) {
    char*  dst    = static_cast<char*>(mmap_base_) + handle.data_offset;
    size_t offset = 0;
    for (const auto& src : src_iovs) {
        if (offset + src.size_bytes > handle.bytes) {
            return false;
        }
        if (!copyHostAndDevice(dst + offset, src.addr, src.size_bytes, false, src.is_cuda)) {
            return false;
        }
        offset += src.size_bytes;
    }
    return offset == handle.bytes;
}

bool KVSClient::load(const KVSReadSession& session, const std::vector<KVSObjectBuffer>& dst_buffers) {
    RTP_LLM_LOG_INFO(
        "KVSClient load start, object_count[%zu], lease[%s]", dst_buffers.size(), session.lease_id.c_str());
    for (const auto& dst : dst_buffers) {
        const auto handle_it = session.handles.find(dst.object_key);
        if (handle_it == session.handles.end()) {
            RTP_LLM_LOG_WARNING("KVSClient load miss handle, object_key[%s]", dst.object_key.c_str());
            return false;
        }
        const auto expected = totalBytes(dst.iovs);
        auto       handle   = handle_it->second;
        const auto server_bytes = handle.bytes;
        if (expected > server_bytes) {
            RTP_LLM_LOG_WARNING("KVSClient load object too small, object_key[%s], expected[%zu], server[%zu]",
                                dst.object_key.c_str(),
                                expected,
                                server_bytes);
            return false;
        }
        if (handle.data_offset > mmap_size_ || server_bytes > mmap_size_ - handle.data_offset) {
            RTP_LLM_LOG_WARNING("KVSClient load out of mmap range, object_key[%s], offset[%llu], server[%zu], mmap[%zu]",
                                dst.object_key.c_str(),
                                static_cast<unsigned long long>(handle.data_offset),
                                server_bytes,
                                mmap_size_);
            return false;
        }
        handle.bytes = expected;
        if (!copyFromObject(handle, dst.iovs)) {
            RTP_LLM_LOG_WARNING("KVSClient load copy failed, object_key[%s]", dst.object_key.c_str());
            return false;
        }
    }
    RTP_LLM_LOG_INFO("KVSClient load done, object_count[%zu]", dst_buffers.size());
    return true;
}

bool KVSClient::store(const std::vector<KVSObjectBuffer>& src_buffers, const std::string& trace_id) {
    if (!inited_ || src_buffers.empty()) {
        return true;
    }
    RTP_LLM_LOG_INFO("KVSClient store start, trace_id[%s], object_count[%zu]", trace_id.c_str(), src_buffers.size());
    rapidjson::StringBuffer                    buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.Key("scope");
    writer.String("create");
    writer.Key("object_keys");
    writer.StartArray();
    for (const auto& src : src_buffers) {
        writer.String(src.object_key.c_str());
    }
    writer.EndArray();
    writer.Key("object_metas");
    writer.StartArray();
    for (const auto& src : src_buffers) {
        writer.StartObject();
        writer.Key("size");
        writer.Uint64(totalBytes(src.iovs));
        writer.Key("layout");
        writer.String("rtp_llm_group_block_v1");
        writer.EndObject();
    }
    writer.EndArray();
    writer.Key("term");
    writer.Int(config_.lease_term_sec);
    writer.Key("request_id");
    writer.String(trace_id.c_str());
    // When ignore_existing is true, KVS may omit already sealed objects from the
    // create lease response. Only returned objects need data copy and seal.
    writer.Key("ignore_existing");
    writer.Bool(true);
    writer.EndObject();

    auto response = httpPost("acquire", buffer.GetString());
    if (!response || response->status_code != 200) {
        if (response) {
            RTP_LLM_LOG_WARNING("KVSClient acquire CREATE failed, status[%ld], body[%s]",
                                response->status_code,
                                response->body.c_str());
        }
        return false;
    }
    rapidjson::Document doc;
    doc.Parse(response->body.c_str());
    if (doc.HasParseError() || !doc.IsObject() || !doc.HasMember("lease") || !doc.HasMember("objects")) {
        RTP_LLM_LOG_WARNING("KVSClient parse acquire CREATE response failed, body[%s]", response->body.c_str());
        return false;
    }
    std::string lease_id;
    if (doc["lease"].IsObject() && doc["lease"].HasMember("lease_id") && doc["lease"]["lease_id"].IsString()) {
        lease_id = doc["lease"]["lease_id"].GetString();
    }
    if (lease_id.empty()) {
        RTP_LLM_LOG_INFO("KVSClient store no-op, trace_id[%s], empty lease", trace_id.c_str());
        return true;
    }
    std::unordered_map<std::string, const KVSObjectBuffer*> src_by_key;
    for (const auto& src : src_buffers) {
        src_by_key.emplace(src.object_key, &src);
    }
    bool        ok      = true;
    const auto& objects = doc["objects"];
    if (!objects.IsArray()) {
        RTP_LLM_LOG_WARNING("KVSClient acquire CREATE response objects is not array, body[%s]", response->body.c_str());
        discard(lease_id);
        return false;
    }
    std::unordered_set<std::string> copied_keys;
    for (auto it = objects.Begin(); it != objects.End(); ++it) {
        auto handle = parseObjectHandle(*it);
        if (!handle) {
            ok = false;
            break;
        }
        const auto src_it = src_by_key.find(handle->object_key);
        if (src_it == src_by_key.end()) {
            ok = false;
            break;
        }
        const auto expected = totalBytes(src_it->second->iovs);
        if (handle->data_offset > mmap_size_ || expected > mmap_size_ - handle->data_offset) {
            RTP_LLM_LOG_WARNING("KVSClient store out of mmap range, object_key[%s], offset[%llu], src[%zu], mmap[%zu]",
                                handle->object_key.c_str(),
                                static_cast<unsigned long long>(handle->data_offset),
                                expected,
                                mmap_size_);
            ok = false;
            break;
        }
        handle->bytes = expected;
        if (!copyToObject(*handle, src_it->second->iovs)) {
            ok = false;
            break;
        }
        copied_keys.insert(handle->object_key);
    }
    if (ok && copied_keys.empty() && !src_buffers.empty()) {
        RTP_LLM_LOG_INFO("KVSClient store no new objects to copy, trace_id[%s]", trace_id.c_str());
    }
    if (ok) {
        rapidjson::StringBuffer                    seal_buffer;
        rapidjson::Writer<rapidjson::StringBuffer> seal_writer(seal_buffer);
        seal_writer.StartObject();
        seal_writer.Key("lease_id");
        seal_writer.String(lease_id.c_str());
        seal_writer.Key("request_id");
        seal_writer.String(trace_id.c_str());
        seal_writer.EndObject();
        auto seal_response = httpPost("seal", seal_buffer.GetString());
        ok                 = seal_response && seal_response->status_code == 204;
    }
    if (ok) {
        release(lease_id);
    } else {
        discard(lease_id);
    }
    RTP_LLM_LOG_INFO(
        "KVSClient store done, trace_id[%s], object_count[%zu], ok[%d]", trace_id.c_str(), src_buffers.size(), ok);
    return ok;
}

void KVSClient::release(const std::string& lease_id) {
    if (lease_id.empty()) {
        return;
    }
    rapidjson::StringBuffer                    buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.Key("lease_id");
    writer.String(lease_id.c_str());
    writer.EndObject();
    auto response = httpPost("release", buffer.GetString());
    if (!response || response->status_code != 204) {
        RTP_LLM_LOG_WARNING("KVSClient release failed, lease_id[%s]", lease_id.c_str());
    }
}

void KVSClient::discard(const std::string& lease_id) {
    if (lease_id.empty()) {
        return;
    }
    rapidjson::StringBuffer                    buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    writer.StartObject();
    writer.Key("lease_id");
    writer.String(lease_id.c_str());
    writer.EndObject();
    auto response = httpPost("discard", buffer.GetString());
    if (!response || response->status_code != 204) {
        RTP_LLM_LOG_WARNING("KVSClient discard failed, lease_id[%s]", lease_id.c_str());
    }
}

}  // namespace rtp_llm::kvs
