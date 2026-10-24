#include "rtp_llm/test/smoke/cache/CacheStoreSmokeEndpoint.h"
#include "rtp_llm/test/smoke/cache/CacheStoreDsv4Oracle.h"

#include <fstream>
#include <sstream>

#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"

namespace rtp_llm::cache_smoke {
namespace {
void mark(const std::filesystem::path& path) {
    std::ofstream out(path);
    out << "ready\n";
    require(static_cast<bool>(out), "cannot write DSV4 rank control file");
}
void waitFor(const std::filesystem::path& path, int64_t timeout) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout);
    while (!std::filesystem::exists(path)) {
        require(std::chrono::steady_clock::now() < deadline, "DSV4 rank control timeout: " + path.string());
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}
int run(int argc, char** argv) {
    require(argc == 29 || argc == 32, "DSV4 rank requires normal or CP configuration arguments");
    const bool sender = std::string(argv[1]) == "sender";
    require(sender || std::string(argv[1]) == "receiver", "invalid DSV4 rank role");
    const std::filesystem::path root(argv[2]);
    const int                   layers = std::stoi(argv[6]), heads = std::stoi(argv[7]), kv_heads = std::stoi(argv[8]);
    const int  head_dim = std::stoi(argv[9]), block_tokens = std::stoi(argv[10]), tokens = std::stoi(argv[11]);
    const int  pool_blocks = std::stoi(argv[12]), timeout = std::stoi(argv[13]);
    const int  tp_size = std::stoi(argv[16]), tp_rank = std::stoi(argv[17]), remote_tp = std::stoi(argv[18]);
    const bool cp2               = argc == 32 && std::stoi(argv[29]) == 2;
    const int  target_dp         = cp2 ? std::stoi(argv[30]) : 1;
    const int  decode_dp_rank    = cp2 ? std::stoi(argv[31]) : 0;
    const int  indexer_dim       = std::stoi(argv[22]);
    const int  fixed_pool_blocks = std::stoi(argv[26]), hca_state_pool_blocks = std::stoi(argv[27]);
    require(std::string(argv[14]) == "fp8" && std::string(argv[19]) == "dsv4" && std::stoi(argv[15]) == 0,
            "DSV4 smoke requires normal FP8 path");
    require(cp2 ? ((sender && tp_size == 2 && remote_tp == 1)
                   || (!sender && tp_size == 1 && remote_tp == 2 && target_dp == 2 && decode_dp_rank < target_dp)) :
                  (tp_size == remote_tp && (tp_size == 1 || tp_size == 2)),
            "DSV4 requires symmetric TP or production CP2-to-DP2 topology");
    require(tp_rank >= 0 && tp_rank < tp_size && tokens % block_tokens == 0, "invalid DSV4 rank or token alignment");
    Dsv4Oracle oracle{std::stoull(argv[5]), block_tokens, head_dim, indexer_dim, std::stoi(argv[25]) != 0};
    oracle.cp_enabled   = cp2;
    oracle.sliced       = cp2 && sender;
    oracle.page_sharded = cp2 && sender;
    oracle.rank         = sender ? tp_rank : 0;
    const auto stem     = [](const std::string& role, int rank, int size) {
        return role + (size == 1 ? "" : "." + std::to_string(rank));
    };
    const auto rank_stem = cp2 && !sender ? stem("receiver", decode_dp_rank, target_dp) :
                                            stem(sender ? "sender" : "receiver", tp_rank, tp_size);
    require(std::string(argv[3]) == "0" && std::string(argv[4]) == "auto",
            "smoke requires kernel-assigned production listener ports");
    initLogger();
    initRuntime(0, false, false, MlaOpsType::AUTO);

    ModelConfig model;
    model.num_layers                          = layers;
    model.hidden_size                         = heads * head_dim;
    model.data_type                           = DataType::TYPE_BF16;
    model.attn_config.head_num                = heads;
    model.attn_config.kv_head_num             = kv_heads;
    model.attn_config.size_per_head           = head_dim;
    model.attn_config.tokens_per_block        = block_tokens;
    model.attn_config.kernel_tokens_per_block = block_tokens;
    model.attn_config.indexer_head_dim        = indexer_dim;
    model.attn_config.sliding_window          = std::stoi(argv[23]);
    model.attn_config.kv_cache_dtype          = KvCacheDataType::FP8;
    projectDsv4Descs(model, argv[24]);
    model.hybrid_attention_config.hybrid_attention_types.assign(layers, HybridAttentionType::NONE);
    std::istringstream ratios(argv[28]);
    for (std::string ratio; std::getline(ratios, ratio, ',');)
        model.attn_config.layer_compress_ratios.push_back(std::stoi(ratio));
    require(model.attn_config.layer_compress_ratios.size() == static_cast<size_t>(layers),
            "DSV4 compression schedule differs");
    ParallelismConfig parallel;
    parallel.tp_size = tp_size;
    parallel.tp_rank = tp_rank;
    if (cp2) {
        parallel.role_type                          = sender ? RoleType::PREFILL : RoleType::DECODE;
        parallel.world_size                         = sender ? 2 : target_dp;
        parallel.world_rank                         = sender ? tp_rank : decode_dp_rank;
        parallel.dp_size                            = sender ? 1 : target_dp;
        parallel.dp_rank                            = sender ? 0 : decode_dp_rank;
        parallel.prefill_cp_config.method           = sender ? CPRotateMethod::ALL_GATHER : CPRotateMethod::PREFILL_CP;
        parallel.prefill_cp_config.kv_cache_sharded = true;
        parallel.prefill_cp_config.prefill_cp_size  = 2;
        require(parallel.get_attn_tp_size() == 1, "DSV4 CP attention TP must be one");
    }
    KVCacheConfig options;
    options.test_block_num      = pool_blocks;
    options.reserve_block_ratio = 0;
    options.reuse_cache         = cp2;
    auto config                 = CacheConfigCreator::createConfig(model, parallel, options);
    config.finalizeBlockNums(
        CacheConfigCreator::computeLocalBlockNum(config, model, RuntimeConfig{}, options, parallel), RuntimeConfig{});
    PDSepConfig manager_pd;
    if (cp2)
        manager_pd.role_type = parallel.role_type;
    auto manager = std::make_shared<KVCacheManager>(
        config, false, nullptr, options, parallel, RuntimeConfig{}, SpeculativeExecutionConfig{}, manager_pd);
    require(manager->init(), "DSV4 CacheManager init failed");
    const auto free_before = manager->freeBlocksNum();
    require(config.groups().size() == 7, "DSV4 production topology must have seven groups");
    for (const auto& group : config.groups()) {
        require(group.spec->block_payload_bytes() == oracle.payloadBytes(group.tag)
                    && group.kvBlockStrideBytes() == oracle.stride(group.tag),
                "DSV4 production packing differs tag=" + group.tag);
        require(group.policy.memory_placement
                    == (oracle.host(group.tag) ? CacheMemoryPlacement::HOST_PINNED : CacheMemoryPlacement::DEVICE),
                "DSV4 production residency differs tag=" + group.tag);
        const uint32_t capacity = group.tag == "hca_state" ? hca_state_pool_blocks :
                                  group.tag == "swa_kv" || group.tag.find("state") != std::string::npos ?
                                                             fixed_pool_blocks :
                                                             pool_blocks;
        require(group.block_num == capacity, "DSV4 production capacity differs tag=" + group.tag);
    }
    for (int layer = 0; layer < layers; ++layer) {
        const auto  expected = dsvExpectedTags(model.attn_config.layer_compress_ratios[layer]);
        const auto& actual   = config.topology().layer(layer).group_tags;
        require(std::set<std::string>(actual.begin(), actual.end()) == expected,
                "DSV4 production layer group membership differs");
    }
    auto converter = std::make_shared<ManagerConverter>(manager);
    for (const auto& [info, size] : converter->getAllBuffers()) {
        if (info.is_cuda)
            require(cudaMemset(info.addr, 0xa5, size) == cudaSuccess, "DSV4 DEVICE poison failed");
        else
            std::memset(info.addr, 0xa5, size);
    }
    require(cudaDeviceSynchronize() == cudaSuccess, "DSV4 poison sync failed");
    auto [resource, complete] = dsvAllocate(*manager, tokens, block_tokens, 1);
    auto& owned               = resource->cacheResource();
    require(owned.cacheKeys().size() == static_cast<size_t>(tokens / block_tokens), "DSV4 cache key count differs");
    for (const auto& group : config.groups()) {
        const auto& blocks = owned.blocks(group.tag);
        require(!blocks.empty() && blocks.size() <= owned.cacheKeys().size(), "DSV4 group slot count differs");
        for (size_t logical = 0; logical < blocks.size(); ++logical) {
            const bool   live = !isNullBlockIdx(blocks[logical]);
            const size_t tail = group.policy.active_tail_blocks;
            require(live == (tail == 0 || logical >= blocks.size() - tail), "DSV4 group materialization differs");
            if (sender && live)
                for (int layer : config.layerIdsForGroup(group.tag))
                    dsvWrite(manager->convertIndexToAddr(layer, group.tag, blocks[logical]).kv_addr,
                             oracle.payload(layer, group.tag, oracle.global(group.tag, logical)),
                             oracle.host(group.tag));
        }
    }
    CacheStoreInitParams params;
    params.listen_port          = 0;
    params.rdma_mode            = false;
    params.enable_metric        = false;
    params.device_id            = 0;
    params.thread_count         = 4;
    const auto listeners_before = listeningTcpPorts();
    auto       store            = NormalCacheStore::createNormalCacheStore(params);
    require(store != nullptr, "DSV4 NormalCacheStore init failed");
    const auto listen_port  = publishSmokeEndpoint(listeners_before, root, rank_stem);
    const auto source_ports = readSmokeSourcePorts(root, sender ? tp_size : remote_tp, timeout);
    manager->setCacheStore(store);
    auto makeRequests = [&](int peer_index) {
        std::vector<std::shared_ptr<RequestBlockBuffer>> requests;
        for (int layer = 0; layer < layers; ++layer) {
            for (const auto& tag : config.topology().layer(layer).group_tags) {
                auto         request   = std::make_shared<RequestBlockBuffer>("cache-smoke-1");
                const auto&  blocks    = owned.blocks(tag);
                const auto&  group     = config.group(tag);
                const size_t key_scale = group.seqSizePerBlock() / config.seq_size_per_block;
                require(key_scale > 0, "DSV4 key block scale must be positive");
                const bool compact =
                    cp2 && group.policy.cp_mapping == CpBlockMappingMode::COMPACT_LAST_RANK && key_scale > 1;
                const size_t logical_blocks = cp2 ? (compact ? owned.cacheKeys().size() :
                                                               (owned.cacheKeys().size() + key_scale - 1) / key_scale) :
                                                    blocks.size();
                const auto   plan           = buildCacheStorePlan(group.policy,
                                                      logical_blocks,
                                                      0,
                                                      true,
                                                      cp2 ? (sender ? tp_rank : (compact ? 1 : 0)) : 0,
                                                      cp2 ? (sender ? 2 : (compact ? 2 : 1)) : 1,
                                                      compact ? 1 : key_scale,
                                                      owned.cacheKeys().size());
                for (const auto& pair : plan) {
                    if (pair.offset_index < 0 || static_cast<size_t>(pair.offset_index) >= blocks.size())
                        continue;
                    if (cp2 && !sender && group.policy.group_type == CacheGroupType::FULL
                        && pair.offset_index % 2 != peer_index)
                        continue;
                    const auto block = blocks[static_cast<size_t>(pair.offset_index)];
                    if (isNullBlockIdx(block))
                        continue;
                    const auto key =
                        "kv_" + makeCacheKey(1, std::to_string(owned.cacheKeys().at(pair.key_index)), layer, tag);
                    auto parts = manager->convertIndexToBuffer(layer, tag, block);
                    if (cp2 && !sender && group.policy.group_type != CacheGroupType::FULL
                        && group.policy.cp_slice != CpBlockSliceMode::NONE) {
                        const CPSlotMapper mapper(1, 2, static_cast<int>(group.seqSizePerBlock()));
                        parts = mapper.sliceBlockForPeer(config, tag, std::move(parts), peer_index);
                    } else if (cp2 && !sender && group.policy.group_type != CacheGroupType::FULL && peer_index != 0) {
                        continue;
                    }
                    require(parts.size() == 1 && parts[0].addr && parts[0].size_bytes > 0,
                            "DSV4 production opaque block part count differs");
                    request->addBlock(key,
                                      std::shared_ptr<void>(parts[0].addr, [](void*) {}),
                                      static_cast<uint32_t>(parts[0].size_bytes),
                                      parts[0].is_cuda,
                                      true);
                }
                requests.push_back(request);
            }
        }
        return requests;
    };
    CheckResult checked;
    bool        api_ok = true;
    std::string api_error;
    if (sender) {
        checked = checkDsv4(*manager, owned, oracle);
        require(checked.matches, "DSV4 source-before-transfer mismatch");
        auto context = store->storeBuffers(makeRequests(0), timeout);
        context->waitDone();
        api_ok    = context->success();
        api_error = context->getErrorInfoString();
        require(api_ok, "DSV4 storeBuffers failed: " + api_error);
    }
    mark(root / (rank_stem + ".ready"));
    for (int rank = 0; rank < (cp2 ? 2 : tp_size); ++rank)
        waitFor(root / (stem("sender", rank, cp2 ? 2 : tp_size) + ".ready"), timeout);
    for (int rank = 0; rank < (cp2 ? target_dp : tp_size); ++rank)
        waitFor(root / (stem("receiver", rank, cp2 ? target_dp : tp_size) + ".ready"), timeout);
    if (!sender) {
        for (int peer = 0; peer < (cp2 ? 2 : 1); ++peer) {
            auto context = store->loadBuffers(
                makeRequests(peer),
                "127.0.0.1",
                source_ports[cp2 ? peer : tp_rank],
                0,
                timeout,
                [] { return false; },
                1,
                0);
            context->waitDone();
            api_ok    = context->success();
            api_error = context->getErrorInfoString();
            if (!api_ok)
                break;
        }
        checked = checkDsv4(*manager, owned, oracle);
        mark(root / (rank_stem + ".done"));
    } else {
        for (int rank = 0; rank < (cp2 ? target_dp : tp_size); ++rank)
            waitFor(root / (stem("receiver", rank, cp2 ? target_dp : tp_size) + ".done"), timeout);
    }
    const bool guards = dsvGuards(*manager, owned, oracle);
    manager->setCacheStore(nullptr);
    store.reset();
    manager->free(FreeInfo{resource, complete, 1});
    const auto    free_after = manager->freeBlocksNum();
    const bool    passed     = api_ok && checked.matches && guards && free_before == free_after;
    std::ofstream out(root / (rank_stem + ".result.json"));
    out << "{\"passed\":" << (passed ? "true" : "false") << ",\"pid\":" << getpid()
        << ",\"listen_port\":" << listen_port << ",\"port_selection\":\"kernel\""
        << ",\"backend\":\"normal_cache_store_tcp\",\"tp_size\":" << tp_size << ",\"tp_rank\":" << tp_rank
        << ",\"dp_rank\":" << decode_dp_rank << ",\"source_cp\":" << (cp2 ? 2 : 1)
        << ",\"layout\":\"dsv4\",\"seed\":" << oracle.seed << ",\"expected_signature\":" << quote(hex(checked.expected))
        << ",\"observed_signature\":" << quote(hex(checked.observed)) << ",\"bytes_checked\":" << checked.bytes
        << ",\"payload_matches\":" << (checked.matches ? "true" : "false")
        << ",\"api_ok\":" << (api_ok ? "true" : "false") << ",\"api_error\":" << quote(api_error)
        << ",\"guards_intact\":" << (guards ? "true" : "false") << ",\"free_before\":" << free_before
        << ",\"free_after\":" << free_after << ",\"production_resolved_config\":" << quote(config.debugString())
        << ",\"payload_regions\":" << checked.detail << ",\"logical_units\":" << checked.units << "}\n";
    require(static_cast<bool>(out), "cannot save DSV4 rank result");
    return passed ? 0 : 1;
}
}  // namespace
}  // namespace rtp_llm::cache_smoke
int main(int argc, char** argv) {
    try {
        return rtp_llm::cache_smoke::run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "CACHE_STORE_DSV4_SMOKE_FAILED: " << error.what() << '\n';
        return 1;
    }
}
