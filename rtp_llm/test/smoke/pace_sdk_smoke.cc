#include <algorithm>
#include <chrono>
#include <cstdint>
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

#include "kvcm_client/meta_client.h"
#include "kvcm_client/transfer_client.h"

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

BlockBuffers buffers(std::vector<std::vector<uint8_t>>& data) {
    BlockBuffers result;
    for (auto& block : data) {
        // Deliberately split across an odd boundary to catch ordering/truncation.
        const size_t split = block.size() / 3 + 1;
        result.push_back({{{MemoryType::CPU, block.data(), split},
                           {MemoryType::CPU, block.data() + split, block.size() - split}}});
    }
    return result;
}

void verifyPool(TransferClient& transfer, WriteLocation& write, const std::string& spec_name, size_t bytes,
                bool save = true) {
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
    if (save) {
        auto saved = transfer.SaveKvCaches(uris, buffers(source));
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
    check(transfer.LoadKvCaches(actual, buffers(loaded)), "LoadKvCaches " + spec_name);
    require(loaded == expected, "Byte mismatch in " + spec_name);
    // URI/buffer count mismatches must be rejected before submitting I/O.
    require(transfer.LoadKvCaches(actual, {}) != ER_OK, "Invalid buffer count accepted");
}

}  // namespace

int main(int argc, char** argv) {
    using namespace kv_cache_manager;
    try {
        require(argc == 3, "Expected config path and backend type");
        std::ifstream input(argv[1]);
        require(input.good(), "Cannot read SDK config");
        const std::string config((std::istreambuf_iterator<char>(input)), {});
        InitParams params{RoleType::HYBRID, nullptr, "full"};
        auto meta = MetaClient::Create(config, params);
        require(meta != nullptr, "MetaClient::Create failed");
        params.storage_configs = meta->GetStorageConfig();
        auto full = TransferClient::Create(config, params);
        params.self_location_spec_name = "state";
        auto state = TransferClient::Create(config, params);
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
        verifyPool(*full, write, "full", 262144);
        verifyPool(*state, write, "state", 65536);
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
        verifyPool(*full, published, "full", 262144, false);
        verifyPool(*state, published, "state", 65536, false);
        auto [reuse_code, reuse] = meta->StartWrite(trace, keys, {}, {"Ffull", "Ffull", "FfullLstate"}, 60, 1);
        check(reuse_code, "Replica reuse");
        require(count(reuse.locations) == 0, "min_replica_count=1 did not skip existing replicas");
        if (!reuse.write_session_id.empty()) {
            check(meta->FinishWrite(trace, reuse.write_session_id, BlockMaskOffset{0}, reuse.locations), "Finish skipped write");
        }
        auto [replica_code, replica] = meta->StartWrite(trace, keys, {}, {"Ffull", "Ffull", "FfullLstate"}, 60, 2);
        check(replica_code, "Allocate second replica");
        require(count(replica.locations) == 3, "min_replica_count=2 was ignored");
        verifyPool(*full, replica, "full", 262144);
        verifyPool(*state, replica, "state", 65536);
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
        auto [host_code, hosts] = meta->GetHostCacheState(trace, QueryType::QT_PREFIX_MATCH, keys, {"hbm"}, 0);
        check(host_code, "GetHostCacheState");
        require(hosts.empty(), "Payload writes unexpectedly announced an HBM host");

        // Abort must leave no visible metadata, and remove must preserve other indices.
        const std::vector<int64_t> aborted_keys{nonce + 5};
        auto [abort_code, aborted] = meta->StartWrite(trace, aborted_keys, {}, {"Ffull"}, 60, 1);
        check(abort_code, "Abort allocation");
        check(meta->FinishWrite(trace, aborted.write_session_id, BlockMaskOffset{0}, aborted.locations), "Abort finish");
        auto [abort_match_code, abort_match] = meta->MatchLocation(trace, QueryType::QT_BATCH_GET, aborted_keys, {}, BlockMaskOffset{0}, 0, {});
        check(abort_match_code, "Abort match");
        require(count(abort_match) == 0, "Aborted write became visible");
        const std::vector<int64_t> partial_keys{nonce + 6, nonce + 7, nonce + 8};
        auto [partial_code, partial] = meta->StartWrite(trace, partial_keys, {}, {"Ffull", "Ffull", "Ffull"}, 60, 1);
        check(partial_code, "Partial allocation");
        verifyPool(*full, partial, "full", 262144);
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
