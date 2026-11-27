#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
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

#include "ReplicaProof.h"
#include "SdkBufferSafety.h"
#include "SdkClient.h"
#include "SdkWriteSession.h"

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
using PayloadCallback                = int (*)(const char*, const char**);

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
        rpc_address_(config.value("rpc_address", std::string{})),
        instance_(config.at("registration").at("instance_id").get<std::string>()),
        request_timeout_(config.at("request_timeout_ms").get<long>()),
        commit_timeout_(config.at("commit_timeout_ms").get<Ms::rep>()),
        write_timeout_(config.at("write_timeout_seconds").get<int64_t>()),
        replicas_(config.at("min_replica_count").get<int32_t>()),
        model_deployment_(std::move(config.at("registration").at("model_deployment"))),
        options_(std::move(config.at("options"))),
        groups_(Json::object()) {
        registerLayout(config.at("groups"));
    }

    Json connect() {
        if (!client_) {
            auto sdk_options = options_;
            sdk_options["address"] = {rpcAddress()};
            sdk_options["default_query_type"] = static_cast<int>(sdk::QueryType::QT_PREFIX_MATCH_WITH_MAMBA);
            sdk_options["model_deployment"] = model_deployment_;
            sdk_options["location_spec_groups"] = sdk_spec_groups_;
            sdk_options["meta_channel_config"] = {{"retry_time", 1},
                                                   {"connection_timeout", request_timeout_},
                                                   {"call_timeout", request_timeout_}};
            sdk::InitParams params;
            params.role_type = sdk::RoleType::SCHEDULER;
            auto client = kvcm::registerSdkMetadata(sdk_options.dump(), params);
            if (!client) {
                throw std::runtime_error("KVCM SDK metadata registration failed");
            }
            const auto& storage = client->GetStorageConfig();
            const auto backends = Json::parse(storage);
            if (!std::any_of(backends.begin(), backends.end(), [](const auto& backend) {
                    return backend.value("is_available", false);
                })) {
                throw std::runtime_error("KVCM did not return an available storage backend");
            }
            storage_ = storage;
            client_ = std::move(client);
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

    Json describe(const Json& rows) {
        Json                items = Json::array();
        std::vector<size_t> owners;
        for (size_t row = 0; row < rows.size(); ++row) {
            const auto& identity = rows.at(row);
            for (const auto& [name, count] : group(identity.at(0).get<int64_t>()).at("specs").items()) {
                (void)count;
                items.push_back(Json::array({identity.at(0), identity.at(1), name}));
                owners.push_back(row);
            }
        }
        std::vector<Json> found(rows.size(), Json::object());
        for (const auto& proof : proofs(items)) {
            const auto item_index = proof.at(0).get<size_t>();
            const auto& name = items.at(item_index).at(2).get_ref<const std::string&>();
            found.at(owners.at(item_index))[name] = Json::array({proof.at(1), proof.at(2)});
        }
        Json result = Json::array();
        for (size_t row = 0; row < rows.size(); ++row) {
            const auto& identity = rows.at(row);
            const auto selected = completeGroup(group(identity.at(0).get<int64_t>()).at("specs"), found[row]);
            if (!selected.empty()) {
                result.push_back(Json::array({row, Json::array({identity.at(0), identity.at(2), selected}).dump()}));
            }
        }
        return result;
    }

    bool writeGroup(int64_t group_id, int64_t key, bool verify_existing, PayloadCallback payload) {
        if (!payload) {
            throw std::invalid_argument("KVCM requires a payload executor");
        }
        const auto& descriptor = group(group_id);
        const auto existing = checksums(group_id, key);
        if (existing.size() == descriptor.at("specs").size()) {
            return !verify_existing || existing == invokePayload(payload, {{"op", "digest"}});
        }
        auto [code, allocation] = kvcm::startSdkWrite(*client_, traceId(), {key}, {},
                                                       {descriptor.at("name").get<std::string>()},
                                                       write_timeout_, replicas_);
        checkSdk(code, "StartWrite");
        kvcm::SdkWriteSession session(std::move(allocation));
        using Proof = kvcm::WorkerPayloadProof::Result;
        const auto abort = [&](const std::string& id, bool) { finishWithoutPayload(id); };
        try {
            const auto keys = session.allocatedKeys(1);
            if (keys.empty()) {
                session.abort(Proof::BUFFER_SAFE_FAILURE, false, abort);
                return confirm(group_id, key,
                               verify_existing ? invokePayload(payload, {{"op", "digest"}}) : Json::object());
            }
            const auto& specs = descriptor.at("specs");
            const auto uris = writeUris(session.location().locations.front(), specs);
            const auto written = invokePayload(payload, {{"op", "write"}, {"uris", uris}});
            return session.finish(
                writeProof(uris, specs, written),
                [] {},
                [&](const std::string& id, bool complete) {
                    if (complete) {
                        finishWithChecksums(id, written);
                    } else {
                        finishWithoutPayload(id);
                    }
                },
                [&] { return confirm(group_id, key, written); });
        } catch (const BufferOwnershipUnknown&) {
            // The legacy Host executor keeps the session pending when it
            // cannot prove the worker released the registered mmap.
            session.abort(Proof::UNKNOWN, false, abort);
            throw;
        } catch (...) {
            try {
                session.abort(Proof::BUFFER_SAFE_FAILURE, false, abort);
            } catch (...) {
                // Keep the original error if the session cannot be closed.
            }
            throw;
        }
    }

private:
    void registerLayout(const Json& layout) {
        std::map<std::string, int64_t> sizes;
        std::map<std::string, std::vector<std::string>> spec_groups;
        for (const auto& spec_group : layout) {
            const auto id = spec_group.at("id").get<std::string>();
            const auto& specs = spec_group.at("specs");
            if (!specs.is_array() || specs.empty()) {
                throw std::invalid_argument("KVCM page group has no location specs: " + id);
            }
            const auto group_name = specs.size() > 1 ? "BTC_" + id : specs.at(0).at("name").get<std::string>();
            auto [group, new_group] = spec_groups.try_emplace(group_name);
            if (!new_group) {
                throw std::invalid_argument("duplicate KVCM location group: " + group_name);
            }
            auto counts = Json::object();
            for (const auto& spec : specs) {
                const auto name = spec.at("name").get<std::string>();
                const auto ranks = spec.at("ranks").get<size_t>();
                const auto size = spec.at("size").get<int64_t>();
                if (!ranks || !sizes.emplace(name, size).second) {
                    throw std::invalid_argument("invalid or duplicate KVCM location spec: " + name);
                }
                group->second.push_back(name);
                counts[name] = ranks;
            }
            groups_[id] = {{"name", group_name}, {"specs", std::move(counts)}};
        }
        if (sizes.empty()) {
            throw std::invalid_argument("KVCM needs at least one local page spec");
        }
        kvcm::validateReplicaRegistration(sizes, spec_groups);
        sdk_spec_groups_ = std::move(spec_groups);
        options_["location_spec_infos"] = sizes;
    }

    std::string rpcAddress() {
        if (!rpc_address_.empty()) {
            return rpc_address_;
        }
        const auto cluster = rpc("getClusterInfo", Json::object());
        const auto& leader = cluster.at("leader_endpoint");
        auto host = leader.at("host").get<std::string>();
        const auto port = leader.at("meta_rpc_port").get<int>();
        if (host.empty() || port <= 0 || port > 65535) {
            throw std::runtime_error("KVCM cluster discovery did not return a usable metadata RPC endpoint");
        }
        if (host.find(':') != std::string::npos) {
            host = "[" + host + "]";
        }
        return host + ":" + std::to_string(port);
    }

    static Json invokePayload(PayloadCallback payload, const Json& request) {
        const char* response = nullptr;
        if (payload(request.dump().c_str(), &response) != 0 || response == nullptr) {
            // The executor may still be using the lease. Leave it to the server.
            throw BufferOwnershipUnknown("KVCM payload executor did not return a completion receipt");
        }
        auto result = Json::parse(response, nullptr, /*allow_exceptions=*/false);
        if (result.is_discarded()) {
            throw BufferOwnershipUnknown("KVCM payload executor returned an invalid completion receipt");
        }
        return result;
    }

    static Json completeGroup(const Json& required_specs, const Json& found) {
        std::vector<kvcm::ReplicaMember> required;
        std::vector<kvcm::ReplicaLocation> locations;
        for (const auto& [name, count] : required_specs.items()) {
            (void)count;
            required.push_back({name, 0});
        }
        for (const auto& [name, proof] : found.items()) {
            locations.push_back({name, proof.at(0).get<std::string>(), {{name, 0}}});
        }
        Json result = Json::object();
        for (const auto* selected : kvcm::coverReplicaMembers(locations, required)) {
            result[selected->name] = found.at(selected->name);
        }
        return result;
    }

    static void checkSdk(sdk::ClientErrorCode code, const char* operation) {
        if (code != sdk::ClientErrorCode::ER_OK) {
            throw std::runtime_error(std::string("KVCM SDK ") + operation + " failed, code " + std::to_string(code));
        }
    }

    static Json writeUris(const sdk::Location& locations, const Json& specs) {
        Json uris = Json::object();
        for (const auto& location : locations) {
            if (location.uri.rfind("pace://", 0) != 0 || !specs.contains(location.spec_name)
                || uris.contains(location.spec_name)) {
                throw std::runtime_error("KVCM SDK returned an invalid PACE write spec");
            }
            uris[location.spec_name] = location.uri;
        }
        if (uris.empty()) {
            throw std::runtime_error("KVCM SDK returned an empty PACE write location");
        }
        return uris;
    }

    static kvcm::WorkerPayloadProof::Result writeProof(const Json& uris, const Json& specs, const Json& written) {
        kvcm::WorkerPayloadProof proof;
        const bool shape = written.is_object() && written.size() == uris.size();
        proof.observe(true, shape, !shape);
        for (const auto& [name, uri] : uris.items()) {
            (void)uri;
            const bool success = written.contains(name) && validDigests(written.at(name), specs.at(name).get<size_t>());
            proof.observe(true, success, !success);
        }
        return proof.result();
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
        const auto& specs = group(group_id).at("specs");
        for (const auto& [name, count] : specs.items()) {
            (void)count;
            items.push_back(Json::array({group_id, key, name}));
        }
        Json found = Json::object();
        for (const auto& proof : proofs(items)) {
            found[items.at(proof.at(0).get<size_t>()).at(2).get_ref<const std::string&>()] =
                Json::array({proof.at(1), proof.at(2)});
        }
        Json result = Json::object();
        const auto complete = completeGroup(specs, found);
        for (const auto& [name, proof] : complete.items()) {
            result[name] = proof.at(1);
        }
        return result;
    }

    void finishWithoutPayload(const std::string& session) {
        checkSdk(kvcm::finishSdkWrite(*client_, traceId(), session, sdk::BlockMaskOffset{0}, {}), "FinishWrite");
    }

    void finishWithChecksums(const std::string& session, const Json& digests) {
        // The deployed SDK has no checksum_batches argument.
        Json batches = Json::array();
        for (const auto& [name, value] : digests.items()) {
            batches.push_back({{"location_spec_name", name}, {"checksums", {value.dump()}}});
        }
        rpc("finishWriteCache",
            {{"write_session_id", session},
             {"success_blocks", {{"bool_masks", {{"values", {true}}}}}},
             {"checksum_batches", std::move(batches)}});
    }

    std::string url_;
    std::string rpc_address_;
    std::string instance_;
    long        request_timeout_;
    Ms          commit_timeout_;
    int64_t     write_timeout_;
    int32_t     replicas_;
    Json        model_deployment_;
    Json        sdk_spec_groups_;
    Json        options_;
    Json        groups_;
    std::string storage_;
    std::unique_ptr<sdk::MetaClient> client_;
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
        const sdk::ClientMemoryRegistrations memory{{base_, size_, fd_}, {}};
        client_ = kvcm::SdkTransferClient(
            kvcm::SdkTransferClient::registerClient(registration.at("options").dump(), params, &memory));
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
            if (!check([&] { return client_.load(batch_uris, batch_buffers); })) {
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
        const bool ok = check([&] { return client_.save(uris, buffers).first; });
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
    kvcm::SdkTransferClient client_;
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
            } else if (operation == "describe") {
                reply(output, metadata.describe(query.at("rows")));
            } else {
                throw std::invalid_argument("unknown BTC KVCM metadata operation");
            }
            return 0;
        },
        -1,
        -2);
}

BTC_KVCM_API int btc_kvcm_write_group(
    void* handle, int64_t group, int64_t key, int verify_existing, PayloadCallback payload) noexcept {
    return ffi<int>(
        [&] { return static_cast<Metadata*>(handle)->writeGroup(group, key, verify_existing != 0, payload) ? 1 : 0; },
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
