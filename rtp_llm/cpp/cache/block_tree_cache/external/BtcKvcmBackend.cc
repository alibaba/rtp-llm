#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#define XXH_INLINE_ALL
#include <xxhash.h>

#include "3rdparty/remote_kv_cache_manager/include/TransferClientCompat.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/SdkBufferSafety.h"

namespace {
namespace sdk                        = kv_cache_manager;
namespace kvcm                       = rtp_llm::kvcm;
using Json                           = nlohmann::json;
using Clock                          = std::chrono::steady_clock;
using Ms                             = std::chrono::milliseconds;
constexpr size_t      kQueryBatch    = 256;
constexpr size_t      kTransferBatch = 16;
constexpr size_t      kMaxResponse   = 16 * 1024 * 1024;
constexpr size_t      kSampleStride  = 16 * 1024;
constexpr size_t      kSampleBytes   = 64;
constexpr const char* kTair          = "ST_TAIRMEMPOOL";

class BufferOwnershipUnknown final: public std::runtime_error {
    using std::runtime_error::runtime_error;
};

size_t receive(char* bytes, size_t size, size_t count, void* target) noexcept {
    auto& output = *static_cast<std::string*>(target);
    if (size == 0 || count > (kMaxResponse - output.size()) / size) {
        return 0;
    }
    try {
        output.append(bytes, size * count);
        return size * count;
    } catch (...) {
        return 0;
    }
}

void checkCurl(CURLcode code) {
    if (code != CURLE_OK) {
        throw std::runtime_error(std::string("KVCM HTTP request failed: ") + curl_easy_strerror(code));
    }
}

std::string traceId() {
    static const uint64_t process_id = [] {
        std::random_device random;
        return (static_cast<uint64_t>(random()) << 32) | random();
    }();
    static std::atomic<uint64_t> sequence{0};
    return "btc-kvcm-" + std::to_string(process_id) + "-" + std::to_string(sequence.fetch_add(1));
}

bool validDigests(const Json& digests, size_t rank_count) {
    if (!digests.is_object() || digests.size() != rank_count) {
        return false;
    }
    for (size_t rank = 0; rank < rank_count; ++rank) {
        const auto it = digests.find("ch3:" + std::to_string(rank));
        if (it == digests.end() || !it->is_string()) {
            return false;
        }
        const auto& digest = it->get_ref<const std::string&>();
        if (digest.size() != 16 || digest.find_first_not_of("0123456789abcdef") != std::string::npos) {
            return false;
        }
    }
    return true;
}

class Metadata {
public:
    explicit Metadata(Json config):
        url_(config.at("url").get<std::string>()),
        instance_(config.at("registration").at("instance_id").get<std::string>()),
        request_timeout_(config.at("request_timeout_ms").get<long>()),
        commit_timeout_(config.at("commit_timeout_ms").get<Ms::rep>()),
        write_timeout_(config.at("write_timeout_seconds").get<int64_t>()),
        replicas_(config.at("min_replica_count").get<int32_t>()),
        registration_(std::move(config.at("registration"))),
        options_(std::move(config.at("options"))),
        groups_(std::move(config.at("groups"))) {}

    Json connect() {
        if (storage_.empty()) {
            const auto reply    = rpc("registerInstance", registration_);
            const auto storage  = reply.at("storage_configs").get<std::string>();
            const auto backends = Json::parse(storage);
            if (!std::any_of(backends.begin(), backends.end(), [](const auto& backend) {
                    return backend.value("is_available", false);
                })) {
                throw std::runtime_error("KVCM did not return an available storage backend");
            }
            storage_ = storage;
        }
        return Json::array({storage_, options_});
    }

    Json proofs(const Json& items) {
        Json proofs = Json::array();
        if (items.empty()) {
            return proofs;
        }
        connect();
        for (size_t start = 0; start < items.size(); start += kQueryBatch) {
            const size_t end   = std::min(start + kQueryBatch, items.size());
            Json         keys  = Json::array();
            Json         names = Json::array();
            for (size_t index = start; index < end; ++index) {
                keys.push_back(items.at(index).at(1));
                names.push_back(items.at(index).at(2));
            }
            const auto reply =
                rpc("getCacheLocationsByBackend",
                    {{"query_type", "QT_BATCH_GET"},
                     {"block_keys", std::move(keys)},
                     {"block_mask", {{"bool_masks", {{"values", std::vector<bool>(end - start)}}}}},
                     {"location_spec_names", std::move(names)},
                     {"backend_selectors", {{{"backend_type", kTair}, {"strategy", "LSS_WEIGHTED_RANDOM"}}}},
                     {"include_checksums", true}});
            const auto& rows = reply.at("key_locations");
            if (!rows.is_array() || rows.size() != end - start) {
                throw std::runtime_error("KVCM response omitted location rows");
            }
            for (size_t index = start; index < end; ++index) {
                const auto& item           = items.at(index);
                const auto& name           = item.at(2).get_ref<const std::string&>();
                const auto& specs          = group(item.at(0).get<int64_t>()).at("specs");
                const auto  expected_ranks = specs.at(name).get<size_t>();
                const auto  proof          = findProof(rows.at(index - start), name, expected_ranks);
                if (!proof.is_null()) {
                    proofs.push_back(Json::array({index, proof.at(0), proof.at(1)}));
                }
            }
        }
        return proofs;
    }

    Json prepareWrite(int64_t group_id, int64_t key) {
        const auto& descriptor = group(group_id);
        const auto  found      = checksums(group_id, key);
        if (found.size() == descriptor.at("specs").size()) {
            return {{"proofs", found}};
        }
        const auto reply     = rpc("startWriteCache",
                                   {{"block_keys", {key}},
                                    {"location_spec_group_names", {descriptor.at("name")}},
                                    {"write_timeout_seconds", write_timeout_},
                                    {"min_replica_count", replicas_}});
        const auto session   = reply.value("write_session_id", std::string{});
        const auto locations = reply.value("locations", Json::array());
        Json       uris      = Json::object();
        if (locations.empty()) {
            if (!session.empty()) {
                finish(session, Json::object());
            }
            return {{"session", ""}, {"uris", uris}};
        }
        if (session.empty()) {
            throw std::runtime_error("KVCM returned a write location without a session");
        }
        if (locations.size() == 1 && locations.at(0).value("type", "") == std::string_view(kTair)) {
            const auto specs = locations.at(0).value("location_specs", Json::array());
            for (const auto& spec : specs) {
                const auto name = spec.value("name", std::string{});
                const auto uri  = spec.value("uri", std::string{});
                if (!uri.empty() && descriptor.at("specs").contains(name)) {
                    uris[name] = uri;
                }
            }
            if (!uris.empty() && uris.size() == specs.size()) {
                return {{"session", session}, {"uris", uris}};
            }
        }
        finish(session, Json::object());
        throw std::runtime_error("KVCM returned an invalid PACE write layout");
    }

    bool finishWrite(int64_t group_id, int64_t key, const Json& plan, const Json& written) {
        const auto& session  = plan.at("session").get_ref<const std::string&>();
        const auto& uris     = plan.at("uris");
        const auto& specs    = group(group_id).at("specs");
        bool        complete = written.is_object() && written.size() == uris.size();
        for (const auto& [name, uri] : uris.items()) {
            if (!written.contains(name) || !validDigests(written.at(name), specs.at(name).get<size_t>())) {
                complete = false;
                break;
            }
        }
        finish(session, complete ? written : Json::object());
        return complete && confirm(group_id, key, written);
    }

    bool confirm(int64_t group_id, int64_t key, const Json& expected) {
        const auto& specs    = group(group_id).at("specs");
        const auto  deadline = Clock::now() + commit_timeout_;
        for (;;) {
            const auto found = checksums(group_id, key);
            if (found.size() == specs.size()) {
                for (const auto& [name, digest] : expected.items()) {
                    if (!found.contains(name) || found.at(name) != digest) {
                        return false;
                    }
                }
                return true;
            }
            const auto now = Clock::now();
            if (now >= deadline) {
                return false;
            }
            std::this_thread::sleep_until(std::min(deadline, now + Ms(50)));
        }
    }

private:
    Json rpc(const std::string& method, Json fields) {
        static const CURLcode initialized = curl_global_init(CURL_GLOBAL_DEFAULT);
        checkCurl(initialized);
        std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>         request(curl_easy_init(), &curl_easy_cleanup);
        std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(
            curl_slist_append(nullptr, "Content-Type: application/json"), &curl_slist_free_all);
        if (!request || !headers) {
            throw std::runtime_error("unable to create KVCM HTTP client");
        }
        fields["instance_id"] = instance_;
        fields["trace_id"]    = traceId();
        const auto  body      = fields.dump();
        const auto  address   = url_ + "/api/" + method;
        std::string output;
        auto*       handle = request.get();
        checkCurl(curl_easy_setopt(handle, CURLOPT_URL, address.c_str()));
        checkCurl(curl_easy_setopt(handle, CURLOPT_NOPROXY, "*"));
        checkCurl(curl_easy_setopt(handle, CURLOPT_NOSIGNAL, 1L));
        checkCurl(curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, request_timeout_));
        checkCurl(curl_easy_setopt(handle, CURLOPT_HTTPHEADER, headers.get()));
        checkCurl(curl_easy_setopt(handle, CURLOPT_POSTFIELDS, body.c_str()));
        checkCurl(curl_easy_setopt(handle, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(body.size())));
        checkCurl(curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, receive));
        checkCurl(curl_easy_setopt(handle, CURLOPT_WRITEDATA, &output));
        checkCurl(curl_easy_perform(handle));
        long status = 0;
        checkCurl(curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &status));
        if (status / 100 != 2) {
            throw std::runtime_error("KVCM " + method + " HTTP status " + std::to_string(status));
        }
        auto       reply  = Json::parse(output);
        const auto header = reply.value("header", Json::object()).value("status", Json::object());
        if (header.value("code", "") != std::string_view("OK")) {
            throw std::runtime_error("KVCM " + method + " failed: " + header.dump());
        }
        return reply;
    }

    static Json findProof(const Json& row, const std::string& name, size_t rank_count) {
        for (const auto& location : row.value("locations", Json::array())) {
            if (location.value("type", "") != std::string_view(kTair)) {
                continue;
            }
            for (const auto& spec : location.value("location_specs", Json::array())) {
                const auto uri = spec.value("uri", std::string{});
                if (spec.value("name", "") != name || uri.empty()) {
                    continue;
                }
                const auto checksum = spec.find("checksum");
                if (checksum == spec.end() || !checksum->is_string()) {
                    continue;
                }
                const auto digest = Json::parse(checksum->get_ref<const std::string&>(),
                                                nullptr,
                                                /*allow_exceptions=*/false);
                if (validDigests(digest, rank_count)) {
                    return Json::array({uri, digest});
                }
            }
        }
        return nullptr;
    }

    const Json& group(int64_t id) const {
        return groups_.at(std::to_string(id));
    }

    Json checksums(int64_t group_id, int64_t key) {
        Json items = Json::array();
        for (const auto& [name, count] : group(group_id).at("specs").items()) {
            items.push_back(Json::array({group_id, key, name}));
        }
        Json result = Json::object();
        for (const auto& proof : proofs(items)) {
            result[items.at(proof.at(0).get<size_t>()).at(2).get_ref<const std::string&>()] = proof.at(2);
        }
        return result;
    }

    void finish(const std::string& session, const Json& digests) {
        Json fields = {{"write_session_id", session}, {"success_blocks", {{"offset", 0}}}};
        if (!digests.empty()) {
            fields["success_blocks"] = {{"bool_masks", {{"values", {true}}}}};
            Json batches             = Json::array();
            for (const auto& [name, value] : digests.items()) {
                batches.push_back({{"location_spec_name", name}, {"checksums", {value.dump()}}});
            }
            fields["checksum_batches"] = std::move(batches);
        }
        rpc("finishWriteCache", std::move(fields));
    }

    std::string url_;
    std::string instance_;
    long        request_timeout_;
    Ms          commit_timeout_;
    int64_t     write_timeout_;
    int32_t     replicas_;
    Json        registration_;
    Json        options_;
    Json        groups_;
    std::string storage_;
};

std::string digest(const uint8_t* data, size_t size) {
    XXH3_state_t state;
    XXH3_64bits_reset(&state);
    std::array<uint8_t, sizeof(uint64_t)> length{};
    for (size_t i = 0; i < length.size(); ++i) {
        length[i] = static_cast<uint64_t>(size) >> (8 * i);
    }
    XXH3_64bits_update(&state, length.data(), length.size());
    for (size_t offset = 0; offset < size;) {
        const size_t remaining = size - offset;
        XXH3_64bits_update(&state, data + offset, std::min(remaining, kSampleBytes));
        offset += std::min(remaining, kSampleStride);
    }
    uint64_t    bits = XXH3_64bits_digest(&state);
    std::string hex(16, '0');
    for (auto it = hex.rbegin(); it != hex.rend(); ++it) {
        *it = "0123456789abcdef"[bits & 15];
        bits >>= 4;
    }
    return hex;
}

class HostTransfer {
public:
    HostTransfer(void* base, size_t size, int fd): base_(static_cast<uint8_t*>(base)), size_(size), fd_(fd) {
        if (!base || !size || fd < 0) {
            throw std::invalid_argument("BTC requires a writable fd-backed Host mapping");
        }
    }

    void connect(const Json& registration) {
        if (client_) {
            return;
        }
        sdk::RegistSpan span{base_, size_};
        sdk::InitParams params;
        params.role_type               = sdk::RoleType::WORKER;
        params.regist_span             = &span;
        params.self_location_spec_name = registration.at("name").get<std::string>();
        params.storage_configs         = registration.at("storage").get<std::string>();
        const sdk::SharedMemoryRegistration memory{base_, size_, fd_};
        client_ = sdk::TransferClient::Create(registration.at("options").dump(), params, memory);
        if (!client_) {
            throw std::runtime_error("PACE TransferClient failed to register the BTC Host mmap");
        }
    }

    Json digests(size_t offset, const std::vector<size_t>& rank_sizes) const {
        Json result = Json::object();
        for (size_t rank = 0; rank < rank_sizes.size(); ++rank) {
            const size_t size                     = rank_sizes[rank];
            result["ch3:" + std::to_string(rank)] = digest(address(offset, size), size);
            offset += size;
        }
        return result;
    }

    bool read(const Json& entries) {
        ensureConnected();
        // Validate every destination before submitting the first SDK call.
        sdk::BlockBuffers buffers;
        sdk::UriStrVec    uris;
        buffers.reserve(entries.size());
        uris.reserve(entries.size());
        for (const auto& entry : entries) {
            const auto sizes  = entry.at(2).get<std::vector<size_t>>();
            const auto offset = entry.at(1).get<size_t>();
            if (!validDigests(entry.at(3), sizes.size())) {
                throw std::invalid_argument("BTC read proof does not match the registered Host layout");
            }
            uris.push_back(entry.at(0).get<std::string>());
            buffers.push_back(makeBuffer(offset, sizes));
        }
        for (size_t start = 0; start < entries.size(); start += kTransferBatch) {
            const size_t            end = std::min(start + kTransferBatch, entries.size());
            const sdk::BlockBuffers batch_buffers(buffers.begin() + start, buffers.begin() + end);
            const sdk::UriStrVec    batch_uris(uris.begin() + start, uris.begin() + end);
            if (!check([&] { return kvcm::loadSdkKvCaches(*client_, batch_uris, batch_buffers, 0, nullptr); })) {
                return false;
            }
            for (size_t index = start; index < end; ++index) {
                const auto& entry = entries.at(index);
                if (digests(entry.at(1).get<size_t>(), entry.at(2).get<std::vector<size_t>>()) != entry.at(3)) {
                    return false;
                }
            }
        }
        return true;
    }

    Json write(const std::string& uri, size_t offset, const std::vector<size_t>& rank_sizes) {
        ensureConnected();
        const sdk::BlockBuffers buffers{makeBuffer(offset, rank_sizes)};
        const auto              checksums = digests(offset, rank_sizes);
        const sdk::UriStrVec    uris{uri};
        const bool ok = check([&] { return kvcm::saveSdkKvCaches(*client_, uris, buffers, 0, nullptr).first; });
        return ok ? checksums : Json(nullptr);
    }

private:
    uint8_t* address(size_t offset, size_t count) const {
        if (!count || offset > size_ || count > size_ - offset) {
            throw std::invalid_argument("BTC PACE span lies outside the registered Host mmap");
        }
        return base_ + offset;
    }

    sdk::BlockBuffer makeBuffer(size_t offset, const std::vector<size_t>& rank_sizes) const {
        const size_t begin = offset;
        for (size_t size : rank_sizes) {
            address(offset, size);
            offset += size;
        }
        const size_t     bytes = offset - begin;
        sdk::BlockBuffer result;
        result.iovs.push_back({sdk::MemoryType::CPU, address(begin, bytes), bytes, false});
        return result;
    }

    void ensureConnected() const {
        if (!client_) {
            throw std::logic_error("BTC PACE transfer is not connected");
        }
    }

    template<typename Operation>
    static bool check(Operation&& perform) {
        sdk::ClientErrorCode code;
        try {
            code = perform();
        } catch (...) {
            throw BufferOwnershipUnknown("PACE threw while registered Host memory was in use");
        }
        if (code == sdk::ClientErrorCode::ER_OK) {
            return true;
        }
        if (!kvcm::sdkCodeProvesBufferSafety(code)) {
            throw BufferOwnershipUnknown("PACE returned status " + std::to_string(static_cast<int>(code))
                                         + " with unconfirmed Host buffer lifetime");
        }
        return false;
    }

    uint8_t*                             base_;
    size_t                               size_;
    int                                  fd_;
    std::unique_ptr<sdk::TransferClient> client_;
};

thread_local std::string last_error;

template<typename Result, typename Function>
Result ffi(Function&& function, Result failure, Result unknown) noexcept {
    try {
        last_error.clear();
        return function();
    } catch (const BufferOwnershipUnknown& error) {
        last_error = error.what();
        return unknown;
    } catch (const std::exception& error) {
        last_error = error.what();
    } catch (...) {
        last_error = "unknown native BTC KVCM failure";
    }
    return failure;
}

void reply(char** output, const Json& value) {
    const auto text  = value.dump();
    auto*      bytes = static_cast<char*>(std::malloc(text.size() + 1));
    if (!bytes) {
        throw std::bad_alloc();
    }
    std::memcpy(bytes, text.c_str(), text.size() + 1);
    *output = bytes;
}
}  // namespace

#define BTC_KVCM_API extern "C" __attribute__((visibility("default")))

BTC_KVCM_API const char* btc_kvcm_last_error() noexcept {
    return last_error.c_str();
}

BTC_KVCM_API void btc_kvcm_free(void* result) noexcept {
    std::free(result);
}

BTC_KVCM_API void* btc_kvcm_create_metadata(const char* config) noexcept {
    return ffi<void*>([&] { return new Metadata(Json::parse(config)); }, nullptr, nullptr);
}

BTC_KVCM_API void btc_kvcm_destroy_metadata(void* handle) noexcept {
    delete static_cast<Metadata*>(handle);
}

BTC_KVCM_API int btc_kvcm_metadata(void* handle, const char* input, char** output) noexcept {
    return ffi<int>(
        [&] {
            auto&      metadata  = *static_cast<Metadata*>(handle);
            const auto query     = Json::parse(input);
            const auto operation = query.at("op").get<std::string>();
            if (operation == "connect") {
                reply(output, metadata.connect());
            } else if (operation == "proofs") {
                reply(output, metadata.proofs(query.at("items")));
            } else if (operation == "prepare_write") {
                reply(output, metadata.prepareWrite(query.at("group").get<int64_t>(), query.at("key").get<int64_t>()));
            } else if (operation == "finish_write") {
                reply(output,
                      metadata.finishWrite(query.at("group").get<int64_t>(),
                                           query.at("key").get<int64_t>(),
                                           query.at("plan"),
                                           query.at("written")));
            } else if (operation == "confirm") {
                reply(output,
                      metadata.confirm(
                          query.at("group").get<int64_t>(), query.at("key").get<int64_t>(), query.at("expected")));
            } else {
                throw std::invalid_argument("unknown BTC KVCM metadata operation");
            }
            return 0;
        },
        -1,
        -2);
}

BTC_KVCM_API void* btc_kvcm_create_transfer(void* base, size_t bytes, int fd) noexcept {
    return ffi<void*>([&] { return new HostTransfer(base, bytes, fd); }, nullptr, nullptr);
}

BTC_KVCM_API void btc_kvcm_destroy_transfer(void* handle) noexcept {
    delete static_cast<HostTransfer*>(handle);
}

BTC_KVCM_API int btc_kvcm_transfer(void* handle, const char* input, char** output) noexcept {
    return ffi<int>(
        [&] {
            auto&      transfer  = *static_cast<HostTransfer*>(handle);
            const auto command   = Json::parse(input);
            const auto operation = command.at("op").get<std::string>();
            if (operation == "connect") {
                transfer.connect(command);
                reply(output, nullptr);
            } else if (operation == "read") {
                reply(output, transfer.read(command.at("entries")));
            } else if (operation == "digest") {
                reply(output,
                      transfer.digests(command.at("offset").get<size_t>(),
                                       command.at("rank_bytes").get<std::vector<size_t>>()));
            } else if (operation == "write") {
                reply(output,
                      transfer.write(command.at("uri").get<std::string>(),
                                     command.at("offset").get<size_t>(),
                                     command.at("rank_bytes").get<std::vector<size_t>>()));
            } else {
                throw std::invalid_argument("unknown BTC PACE transfer operation");
            }
            return 0;
        },
        -1,
        -2);
}
