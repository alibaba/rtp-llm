#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include "kvcm_client/meta_client.h"
#include "kvcm_client/HostCacheStateCompat.h"
#include "kvcm_client/TransferClientCompat.h"
#include "kvcm_client/transfer_client.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/SdkBufferSafety.h"

#if defined(USING_CUDA13)
#include <cuda_runtime_api.h>
#endif

namespace {
using namespace kv_cache_manager;

void require(bool ok, const std::string& message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

void check(ClientErrorCode code, const std::string& operation) {
    require(code == ER_OK, operation + " failed: " + std::to_string(code));
}

constexpr size_t full_block_bytes = 262144;
constexpr size_t gpu_pool_bytes = 2 * 1024 * 1024;

#if defined(USING_CUDA13)
void checkCuda(cudaError_t code, const char* operation) {
    require(code == cudaSuccess, std::string(operation) + ": " + cudaGetErrorString(code));
}
#endif

class GpuPool {
public:
    explicit GpuPool(bool enabled) {
        if (!enabled) {
            return;
        }
#if defined(USING_CUDA13)
        void* buffer = nullptr;
        checkCuda(cudaMalloc(&buffer, gpu_pool_bytes), "cudaMalloc");
        base_.reset(static_cast<uint8_t*>(buffer));
#else
        throw std::runtime_error("GDR requires a CUDA 13 build");
#endif
    }

    std::unique_ptr<TransferClient> createClient(const std::string& config, const InitParams& params,
                                                 const GpuPool& extra_pool) const {
#if defined(USING_CUDA13)
        if (base_) {
            int device_id = 0;
            checkCuda(cudaGetDevice(&device_id), "cudaGetDevice");
            ClientMemoryRegistrations registrations;
            registrations.gpu.push_back({base_.get(), gpu_pool_bytes, device_id});
            if (extra_pool.base_) {
                registrations.gpu.push_back({extra_pool.base_.get(), gpu_pool_bytes, device_id});
            }
            return TransferClient::Create(config, params, registrations);
        }
#else
        (void)extra_pool;
#endif
        return TransferClient::Create(config, params);
    }

    uint8_t* base() const {
        return base_.get();
    }

private:
    struct Deleter {
        void operator()(uint8_t* buffer) const {
#if defined(USING_CUDA13)
            cudaFree(buffer);
#else
            (void)buffer;
#endif
        }
    };
    std::unique_ptr<uint8_t, Deleter> base_;
};

bool present(const Location& location) {
    return std::any_of(location.begin(), location.end(), [](const auto& spec) { return !spec.uri.empty(); });
}

size_t count(const Locations& locations) {
    return std::count_if(locations.begin(), locations.end(), present);
}

Locations waitForMatch(MetaClient& meta, const std::string& trace, const std::vector<int64_t>& keys,
                       const std::function<bool(const Locations&)>& expected) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    do {
        auto [code, locations] = meta.MatchLocation(trace, QueryType::QT_BATCH_GET, keys, {}, BlockMaskOffset{0}, 0, {});
        check(code, "Wait for asynchronous removal");
        if (expected(locations)) {
            return locations;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    } while (std::chrono::steady_clock::now() < deadline);
    throw std::runtime_error("Asynchronous RemoveCache did not become visible");
}

BlockBuffers buffers(std::vector<std::vector<uint8_t>>& data, uint8_t* gpu) {
    BlockBuffers result;
    size_t offset = 0;
    for (auto& block : data) {
        // Deliberately split across an odd boundary to catch ordering/truncation.
        const size_t split = gpu ? block.size() / 2 : block.size() / 3 + 1;
        const auto type = gpu ? MemoryType::GPU : MemoryType::CPU;
        auto* base = gpu ? gpu + offset : block.data();
        result.push_back({{{type, base, split}, {type, base + split, block.size() - split}}});
        offset += block.size();
    }
    return result;
}

template<typename Code>
bool expectedTimeout(Code code) {
    if (code == Code::ER_SDK_TIMEOUT) {
        return true;
    }
    if constexpr (rtp_llm::kvcm::detail::HasExplicitBufferSafeCodes<Code>::value) {
        return code == Code::ER_SDK_TIMEOUT_BUFFER_SAFE;
    }
    return false;
}

void exerciseFault(TransferClient& transfer, const std::string& uri, const BlockBuffers& block_buffers,
                   const std::string& scenario, const std::string& socket_path) {
    require(uri.rfind("pace://", 0) == 0, "Fault requires a PACE URI");
    require(rtp_llm::kvcm::detail::HasTransferDeadline<TransferClient>::value,
            "TCP faults require a KVCM SDK with per-call deadlines");
    struct Connection {
        const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        ~Connection() {
            if (fd >= 0) {
                close(fd);
            }
        }
    } connection;
    require(connection.fd >= 0, "Cannot create PACE fault socket");
    require(fcntl(connection.fd, F_SETFD, FD_CLOEXEC) == 0, "Cannot initialize PACE fault socket");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    require(socket_path.size() < sizeof(address.sun_path), "PACE fault socket path is too long");
    std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);
    const timeval timeout{25, 0};
    require(setsockopt(connection.fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0
                && setsockopt(connection.fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0,
            "Cannot set PACE fault socket timeouts");
    require(connect(connection.fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
            "Cannot connect to PACE fault control at " + socket_path);
    const auto exchange = [&](const std::string& request) {
        char reply = 0;
        require(send(connection.fd, request.data(), request.size(), MSG_NOSIGNAL)
                    == static_cast<ssize_t>(request.size())
                    && recv(connection.fd, &reply, 1, 0) == 1 && reply == '1',
                "PACE fault control failed for " + scenario + ". See faults.log");
    };

    const UriStrVec uris{uri};
    exchange(scenario + "\n");
    const auto deadline_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch() + std::chrono::seconds(1)).count();
    const auto code = scenario == "slow-write"
        ? rtp_llm::kvcm::saveSdkKvCaches(transfer, uris, block_buffers, deadline_ms, nullptr).first
        : rtp_llm::kvcm::loadSdkKvCaches(transfer, uris, block_buffers, deadline_ms, nullptr);
    exchange("\n");
    require(expectedTimeout(code), scenario + " returned unexpected SDK status " + std::to_string(code));
    std::cout << "PASS: " << scenario << " hit, SDK status=" << code << '\n';
}

void verifyPool(TransferClient& transfer, WriteLocation& write, const std::string& spec_name, size_t bytes,
                const GpuPool& pool, bool save = true) {
    UriStrVec uris;
    std::vector<LocationSpecUnit*> specs;
    for (auto& location : write.locations) {
        for (auto& spec : location) {
            if (spec.spec_name == spec_name && !spec.uri.empty()) {
                require(spec.uri.rfind("pace://", 0) == 0, "Expected a real PACE URI");
                specs.push_back(&spec);
                uris.push_back(spec.uri);
            }
        }
    }
    require(!uris.empty(), "No write locations for " + spec_name);
    std::vector<std::vector<uint8_t>> expected(uris.size(), std::vector<uint8_t>(bytes));
    for (size_t block = 0; block < expected.size(); ++block) {
        for (size_t byte = 0; byte < bytes; ++byte) {
            expected[block][byte] = static_cast<uint8_t>(byte * 31 + (byte >> 8) + block * 97);
        }
    }
    auto source = expected;
    auto actual = uris;
    auto* gpu = pool.base();
#if defined(USING_CUDA13)
    require(!gpu || uris.size() <= gpu_pool_bytes / bytes, "GPU smoke pool is too small");
#endif
    if (save) {
#if defined(USING_CUDA13)
        if (gpu) {
            for (size_t block = 0; block < source.size(); ++block) {
                checkCuda(cudaMemcpy(gpu + block * bytes, source[block].data(), bytes, cudaMemcpyHostToDevice),
                          "initialize GPU source");
            }
            checkCuda(cudaDeviceSynchronize(), "synchronize GPU source before PACE write");
        }
#endif
        auto saved = transfer.SaveKvCaches(uris, buffers(source, gpu));
        check(saved.first, "SaveKvCaches " + spec_name);
        actual = std::move(saved.second);
    }
    require(actual.size() == specs.size(), "Actual URI count changed");
    for (size_t i = 0; i < actual.size(); ++i) {
        require(actual[i].rfind("pace://", 0) == 0, "Saved URI is not PACE");
        specs[i]->uri = actual[i];
        std::fill(source[i].begin(), source[i].end(), 0);
    }
    std::vector<std::vector<uint8_t>> loaded(actual.size(), std::vector<uint8_t>(bytes, 0xA5));
#if defined(USING_CUDA13)
    if (gpu) {
        checkCuda(cudaMemset(gpu, 0xA5, actual.size() * bytes), "initialize GPU destination");
        checkCuda(cudaDeviceSynchronize(), "synchronize GPU destination before PACE read");
    }
#endif
    check(transfer.LoadKvCaches(actual, buffers(loaded, gpu)), "LoadKvCaches " + spec_name);
#if defined(USING_CUDA13)
    if (gpu) {
        for (size_t block = 0; block < loaded.size(); ++block) {
            checkCuda(cudaMemcpy(loaded[block].data(), gpu + block * bytes, bytes, cudaMemcpyDeviceToHost),
                      "verify GPU readback");
        }
    }
#endif
    require(loaded == expected, "Byte mismatch in " + spec_name);
    // URI/buffer count mismatches must be rejected before submitting I/O.
    require(transfer.LoadKvCaches(actual, {}) != ER_OK, "Invalid buffer count accepted");
}

}  // namespace

int main(int argc, char** argv) {
    using namespace kv_cache_manager;
    try {
        const std::string usage = "Expected config path, backend type, [--gdr|--aft-gdr] [--fault CASE SOCKET]";
        require(argc >= 3, usage);
        const std::string buffer_mode = argc > 3 ? argv[3] : "";
        const bool aft_gdr = buffer_mode == "--aft-gdr";
        const bool use_gpu = buffer_mode == "--gdr" || aft_gdr;
        const int fault_arg = use_gpu ? 4 : 3;
        require(argc == fault_arg || (argc == fault_arg + 3 && std::string(argv[fault_arg]) == "--fault"), usage);
        const std::string fault = argc > fault_arg ? argv[fault_arg + 1] : "";
        const std::string fault_socket = argc > fault_arg ? argv[fault_arg + 2] : "";
        const bool tcp_fault = fault == "slow-read" || fault == "slow-write" || fault == "packet-loss";
        require(argc == fault_arg || (!use_gpu && tcp_fault && !fault_socket.empty()),
                "Invalid PACE fault mode or socket");
        std::ifstream input(argv[1]);
        require(input.good(), "Cannot read SDK config");
        const std::string config((std::istreambuf_iterator<char>(input)), {});
        InitParams params{RoleType::HYBRID, nullptr, "full"};
        auto meta = MetaClient::Create(config, params);
        require(meta != nullptr, "MetaClient::Create failed");
        params.storage_configs = meta->GetStorageConfig();
        GpuPool full_pool(use_gpu);
        GpuPool second_pool(use_gpu && !aft_gdr);
        const auto& state_pool = aft_gdr ? full_pool : second_pool;
        std::vector<std::vector<uint8_t>> fault_data;
        BlockBuffers fault_buffers;
        if (!fault.empty()) {
            fault_data.emplace_back(full_block_bytes, 0x5A);
            fault_buffers = buffers(fault_data, nullptr);
        }
        auto full = full_pool.createClient(config, params, second_pool);
        std::unique_ptr<TransferClient> separate_state;
        if (!use_gpu) {
            params.self_location_spec_name = "state";
            separate_state = TransferClient::Create(config, params);
        }
        auto* state = use_gpu ? full.get() : separate_state.get();
        require(full != nullptr && state != nullptr, "TransferClient::Create failed");
        const auto backend = static_cast<StorageType>(std::stoi(argv[2]));
        const int64_t nonce = std::chrono::system_clock::now().time_since_epoch().count();
        const std::string trace = "pace-smoke-" + std::to_string(nonce);
        const std::vector<int64_t> keys{nonce, nonce + 1, nonce + 2};
        auto [write_code, write] = meta->StartWrite(trace, keys, {}, {"Ffull", "Ffull", "FfullLstate"}, 60, 1);
        check(write_code, "StartWrite");
        require(write.locations.size() == keys.size() && count(write.locations) == keys.size(), "Incomplete allocation");

        // Allocated data is not visible before FinishWrite.
        auto [pending_code, pending] = meta->MatchLocation(trace, QueryType::QT_BATCH_GET, keys, {}, BlockMaskOffset{0}, 0, {});
        check(pending_code, "Pending match");
        require(count(pending) == 0, "Uncommitted writes became visible");
        verifyPool(*full, write, "full", full_block_bytes, full_pool);
        verifyPool(*state, write, "state", 65536, state_pool);
        check(meta->FinishWrite(trace, write.write_session_id, BlockMaskOffset{keys.size()}, write.locations), "FinishWrite");

        for (const auto query : {QueryType::QT_BATCH_GET, QueryType::QT_PREFIX_MATCH}) {
            auto [code, locations] = meta->MatchLocation(trace, query, keys, {}, BlockMaskOffset{0}, 0,
                                                        std::vector<std::string>(keys.size(), "full"));
            check(code, "MatchLocation");
            require(count(locations) == 3, "Prefix/batch query lost keys");
        }
        auto [read_code, read_locations] = meta->MatchLocation(trace, QueryType::QT_BATCH_GET, keys, {}, BlockMaskOffset{0}, 0, {});
        check(read_code, "Read published URIs");
        WriteLocation published;
        published.locations = std::move(read_locations);
        verifyPool(*full, published, "full", full_block_bytes, full_pool, false);
        verifyPool(*state, published, "state", 65536, state_pool, false);

        auto [offset_code, offset_match] = meta->MatchLocation(
            trace, QueryType::QT_BATCH_GET, keys, {}, BlockMaskOffset{1}, 0, {});
        check(offset_code, "Batch match with local prefix");
        // Batch matching preserves key positions but may also return the local prefix.
        require(offset_match.size() == keys.size(),
                "Batch offset query did not preserve key positions");
        for (size_t i = 1; i < keys.size(); ++i) {
            require(std::is_permutation(offset_match[i].begin(), offset_match[i].end(),
                                        published.locations[i].begin(), published.locations[i].end(),
                                        [](const auto& left, const auto& right) {
                                            return left.spec_name == right.spec_name && left.uri == right.uri;
                                        }), "Batch offset query changed key-to-URI positions");
        }
        auto [offset_backend_code, offset_backend] = meta->GetCacheLocationsByBackend(
            trace, keys, {}, BlockMaskOffset{1}, std::vector<std::string>(keys.size(), "full"),
            backend, BackendSelectStrategy::LSS_WEIGHTED_RANDOM);
        check(offset_backend_code, "Backend query with local prefix");
        require(offset_backend.size() == keys.size() && offset_backend[0].empty(),
                "Backend offset query did not preserve the skipped key position");
        for (size_t i = 1; i < keys.size(); ++i) {
            require(offset_backend[i].size() == 1 && offset_backend[i][0].location_specs.size() == 1,
                    "Backend offset query lost a requested key");
            const auto& spec = offset_backend[i][0].location_specs[0];
            require(spec.spec_name == "full" && std::any_of(published.locations[i].begin(),
                    published.locations[i].end(), [&](const auto& expected) {
                        return expected.spec_name == spec.spec_name && expected.uri == spec.uri;
                    }), "Backend offset query changed key-to-URI positions");
        }
        auto [reuse_code, reuse] = meta->StartWrite(trace, keys, {}, {"Ffull", "Ffull", "FfullLstate"}, 60, 1);
        check(reuse_code, "Replica reuse");
        require(count(reuse.locations) == 0, "min_replica_count=1 did not skip existing replicas");
        if (!reuse.write_session_id.empty()) {
            check(meta->FinishWrite(trace, reuse.write_session_id, BlockMaskOffset{0}, reuse.locations), "Finish skipped write");
        }
        if (!fault.empty() && fault != "slow-write") {
            exerciseFault(*full, published.locations.at(0).at(0).uri, fault_buffers, fault, fault_socket);
            verifyPool(*full, published, "full", full_block_bytes, full_pool, false);
        }
        auto [replica_code, replica] = meta->StartWrite(trace, keys, {}, {"Ffull", "Ffull", "FfullLstate"}, 60, 2);
        check(replica_code, "Allocate second replica");
        require(count(replica.locations) == 3, "min_replica_count=2 was ignored");
        verifyPool(*full, replica, "full", full_block_bytes, full_pool);
        verifyPool(*state, replica, "state", 65536, state_pool);
        check(meta->FinishWrite(trace, replica.write_session_id, BlockMaskOffset{keys.size()}, replica.locations), "Publish second replica");
        auto [two_code, two] = meta->StartWrite(trace, keys, {}, {"Ffull", "Ffull", "FfullLstate"}, 60, 2);
        check(two_code, "Reuse two replicas");
        require(count(two.locations) == 0, "Two existing replicas were not reused");
        if (!two.write_session_id.empty()) {
            check(meta->FinishWrite(trace, two.write_session_id, BlockMaskOffset{0}, two.locations), "Finish replica reuse");
        }

        auto sparse_keys = keys;
        sparse_keys.insert(sparse_keys.begin() + 1, nonce + 4);
        auto [backend_code, by_backend] = meta->GetCacheLocationsByBackend(
            trace, sparse_keys, {}, BlockMaskVector{false, false, true, false},
            std::vector<std::string>(sparse_keys.size(), "full"), backend,
            BackendSelectStrategy::LSS_WEIGHTED_RANDOM);
        check(backend_code, "GetCacheLocationsByBackend");
        require(by_backend.size() == 4 && by_backend[1].empty() && by_backend[2].empty(), "Backend miss/mask indices changed");
        require(!by_backend[0].empty() && !by_backend[3].empty(), "Backend query lost hits");
        for (const auto& group : by_backend) {
            for (const auto& location : group) {
                require(location.type == backend && present(location.location_specs), "Wrong backend identity");
            }
        }
        auto [host_code, hosts] =
            rtp_llm::kvcm::getSdkHostCacheState(*meta, trace, QueryType::QT_PREFIX_MATCH, keys, {"hbm"}, 0);
        check(host_code, "GetHostCacheState");
        require(hosts.empty(), "Payload writes unexpectedly announced an HBM host");

        // Abort must leave no visible metadata, and remove must preserve other indices.
        const std::vector<int64_t> aborted_keys{nonce + 5};
        auto [abort_code, aborted] = meta->StartWrite(trace, aborted_keys, {}, {"Ffull"}, 60, 1);
        check(abort_code, "Abort allocation");
        if (fault == "slow-write") {
            require(count(aborted.locations) == 1, "Incomplete fault allocation");
            exerciseFault(*full, aborted.locations.at(0).at(0).uri, fault_buffers, fault, fault_socket);
        } else {
            check(meta->FinishWrite(trace, aborted.write_session_id, BlockMaskOffset{0}, aborted.locations), "Abort finish");
        }
        auto [abort_match_code, abort_match] = meta->MatchLocation(trace, QueryType::QT_BATCH_GET, aborted_keys, {}, BlockMaskOffset{0}, 0, {});
        check(abort_match_code, "Abort match");
        require(count(abort_match) == 0, "Unpublished write became visible");
        const std::vector<int64_t> partial_keys{nonce + 6, nonce + 7, nonce + 8};
        auto [partial_code, partial] = meta->StartWrite(trace, partial_keys, {}, {"Ffull", "Ffull", "Ffull"}, 60, 1);
        check(partial_code, "Partial allocation");
        verifyPool(*full, partial, "full", full_block_bytes, full_pool);
        check(meta->FinishWrite(trace, partial.write_session_id, BlockMaskVector{true, false, true}, partial.locations), "Partial finish");
        auto [partial_match_code, partial_match] = meta->MatchLocation(trace, QueryType::QT_BATCH_GET, partial_keys, {}, BlockMaskOffset{0}, 0, {});
        check(partial_match_code, "Partial match");
        require(partial_match.size() == 3 && present(partial_match[0]) && !present(partial_match[1]) && present(partial_match[2]), "Partial success mask was not respected");
        check(meta->RemoveCache(trace, partial_keys, {}, BlockMaskOffset{0}), "Partial cleanup");
        waitForMatch(*meta, trace, partial_keys, [](const auto& locations) { return count(locations) == 0; });
        check(meta->RemoveCache(trace, {keys[1]}, {}, BlockMaskOffset{0}), "RemoveCache");
        waitForMatch(*meta, trace, keys, [](const auto& removed) {
            return removed.size() == 3 && present(removed[0]) && !present(removed[1]) && present(removed[2]);
        });
        check(meta->RemoveCache(trace, keys, {}, BlockMaskOffset{0}), "Cleanup");
        waitForMatch(*meta, trace, keys, [](const auto& locations) { return count(locations) == 0; });

        std::cout << "PASS: PACE bytes, two specs, multi-IOV, queries, backend, host state, masks, abort and removal\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
