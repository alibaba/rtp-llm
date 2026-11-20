#include "rtp_llm/test/smoke/cache/CacheStoreSmokeEndpoint.h"
#include "rtp_llm/test/smoke/cache/CacheStoreNextOracle.h"

#include <fstream>
#include <sstream>

#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"

namespace rtp_llm::cache_smoke {
namespace {
void mark(const std::filesystem::path& path) {
    std::ofstream out(path);
    out << "ready\n";
    require(static_cast<bool>(out), "cannot write Next rank control file");
}
void waitFor(const std::filesystem::path& path, int64_t timeout) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout);
    while (!std::filesystem::exists(path)) {
        require(std::chrono::steady_clock::now() < deadline, "Next rank control timeout: " + path.string());
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}
int run(int argc, char** argv) {
    require(argc == 28, "Next rank requires 27 configuration arguments");
    const bool sender = std::string(argv[1]) == "sender";
    require(sender || std::string(argv[1]) == "receiver", "invalid Next role");
    const std::filesystem::path root(argv[2]);
    const int                   layers = std::stoi(argv[6]), heads = std::stoi(argv[7]), kv_heads = std::stoi(argv[8]);
    const int head_dim = std::stoi(argv[9]), block_tokens = std::stoi(argv[10]), tokens = std::stoi(argv[11]);
    const int pool_blocks = std::stoi(argv[12]), timeout = std::stoi(argv[13]);
    const int tp_size = std::stoi(argv[16]), tp_rank = std::stoi(argv[17]), remote_tp = std::stoi(argv[18]);
    const int key_heads = std::stoi(argv[22]), value_heads = std::stoi(argv[23]);
    const int linear_dim = std::stoi(argv[24]), conv_kernel = std::stoi(argv[25]);
    const int linear_pool_blocks = std::stoi(argv[27]);
    require(std::string(argv[14]) == "bf16" && std::string(argv[19]) == "next" && std::stoi(argv[15]) == 0,
            "Next smoke requires normal BF16 path");
    require(tp_size == remote_tp && (tp_size == 1 || tp_size == 2), "Next only permits symmetric TP1/TP2");
    require(tp_rank >= 0 && tp_rank < tp_size && tokens % block_tokens == 0, "invalid Next rank or pages");
    const NextOracle oracle{std::stoull(argv[5]),
                            tp_size,
                            tp_rank,
                            kv_heads,
                            head_dim,
                            block_tokens,
                            key_heads,
                            value_heads,
                            linear_dim,
                            conv_kernel,
                            2,
                            2};
    const auto       stem = [](const std::string& role, int rank, int size) {
        return role + (size == 1 ? "" : "." + std::to_string(rank));
    };
    const auto rank_stem = stem(sender ? "sender" : "receiver", tp_rank, tp_size);
    require(std::string(argv[3]) == "0" && std::string(argv[4]) == "auto",
            "smoke requires kernel-assigned production listener ports");
    initLogger();
    initRuntime(0, false, false, MlaOpsType::AUTO);

    ModelConfig model;
    model.num_layers                                      = layers;
    model.hidden_size                                     = heads * head_dim;
    model.data_type                                       = DataType::TYPE_BF16;
    model.attn_config.head_num                            = heads;
    model.attn_config.kv_head_num                         = kv_heads;
    model.attn_config.size_per_head                       = head_dim;
    model.attn_config.tokens_per_block                    = block_tokens;
    model.attn_config.kernel_tokens_per_block             = block_tokens;
    model.hybrid_attention_config.enable_hybrid_attention = true;
    auto& linear                                          = model.linear_attention_config;
    linear.linear_num_key_heads                           = key_heads;
    linear.linear_num_value_heads                         = value_heads;
    linear.linear_key_head_dim                            = linear_dim;
    linear.linear_value_head_dim                          = linear_dim;
    linear.linear_conv_kernel_dim                         = conv_kernel;
    linear.ssm_state_dtype                                = DataType::TYPE_BF16;
    linear.conv_state_dtype                               = DataType::TYPE_BF16;
    std::vector<std::string> layer_tags;
    std::istringstream       schedule(argv[26]);
    for (std::string tag; std::getline(schedule, tag, ',');) {
        require(tag == "full" || tag == "linear", "invalid Next layer tag");
        layer_tags.push_back(tag);
        KVCacheSpecDesc desc;
        desc.tag        = tag;
        desc.cache_type = tag == "linear" ? KVCacheSpecType::LinearAttention : KVCacheSpecType::MultiHeadAttention;
        if (tag == "linear") {
            CacheCapacityPolicyDesc capacity;
            capacity.explicit_block_num = linear_pool_blocks;
            desc.capacity               = capacity;
        }
        model.kv_cache_spec_descs.push_back({desc});
        model.hybrid_attention_config.hybrid_attention_types.push_back(tag == "linear" ? HybridAttentionType::LINEAR :
                                                                                         HybridAttentionType::NONE);
    }
    require(layer_tags.size() == static_cast<size_t>(layers), "Next layer schedule count differs");
    ParallelismConfig parallel;
    parallel.tp_size = tp_size;
    parallel.tp_rank = tp_rank;
    KVCacheConfig options;
    options.test_block_num      = pool_blocks;
    options.reserve_block_ratio = 0;
    options.reuse_cache         = false;
    auto config                 = CacheConfigCreator::createConfig(model, parallel, options);
    config.finalizeBlockNums(
        CacheConfigCreator::computeLocalBlockNum(config, model, RuntimeConfig{}, options, parallel), RuntimeConfig{});
    require(config.groups().size() == 2, "Next production topology must have FULL and Linear groups");
    require(config.group("full").kvBlockStrideBytes() == oracle.payload(3, "full", 0).size(),
            "Next FULL geometry differs");
    require(config.group("linear").spec->k_block_size_bytes() == oracle.ssmBytes()
                && config.group("linear").spec->v_block_size_bytes() == oracle.convBytes(),
            "Next SSM/conv geometry differs");
    auto manager = std::make_shared<KVCacheManager>(config, false, nullptr, options, parallel);
    require(manager->init(), "Next CacheManager init failed");
    const auto free_before = manager->freeBlocksNum();
    auto       converter   = std::make_shared<ManagerConverter>(manager);
    for (const auto& [info, size] : converter->getAllBuffers()) {
        require(info.is_cuda && cudaMemset(info.addr, 0xa5, size) == cudaSuccess,
                "Next requires real DEVICE backing and poison");
    }
    require(cudaDeviceSynchronize() == cudaSuccess, "Next poison sync failed");
    auto [resource, complete] = nextAllocate(*manager, tokens, block_tokens, 1);
    auto& owned               = resource->cacheResource();
    require(owned.cacheKeys().size() == static_cast<size_t>(tokens / block_tokens), "Next cache key count differs");
    for (const auto& group : config.groups()) {
        const auto& blocks = owned.blocks(group.tag);
        require(blocks.size() == owned.cacheKeys().size(), "Next group slot count differs");
        const auto plan = buildCacheStorePlan(group.policy, blocks.size(), 0, true, 0, 1, 1, owned.cacheKeys().size());
        std::set<size_t> planned;
        for (const auto& pair : plan)
            planned.insert(pair.offset_index);
        for (size_t logical = 0; logical < blocks.size(); ++logical) {
            const bool live = !isNullBlockIdx(blocks[logical]);
            require(live == (planned.count(logical) > 0), "Next production group plan differs from allocation");
            if (sender && live)
                for (int layer : config.layerIdsForGroup(group.tag))
                    writeBytes(manager->convertIndexToAddr(layer, group.tag, blocks[logical]).kv_addr,
                               oracle.payload(layer, group.tag, logical));
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
    require(store != nullptr, "Next NormalCacheStore init failed");
    const auto listen_port  = publishSmokeEndpoint(listeners_before, root, rank_stem);
    const auto source_ports = readSmokeSourcePorts(root, sender ? tp_size : remote_tp, timeout);
    manager->setCacheStore(store);
    auto makeLayer = [&](int layer) {
        const auto& tag     = layer_tags[layer];
        auto        request = std::make_shared<RequestBlockBuffer>("cache-smoke-1");
        const auto& blocks  = owned.blocks(tag);
        const auto& group   = config.group(tag);
        const auto  plan = buildCacheStorePlan(group.policy, blocks.size(), 0, true, 0, 1, 1, owned.cacheKeys().size());
        for (const auto& pair : plan) {
            const auto block = blocks.at(pair.offset_index);
            if (isNullBlockIdx(block))
                continue;
            const auto key = "kv_" + makeCacheKey(1, std::to_string(owned.cacheKeys().at(pair.key_index)), layer, tag);
            const auto parts = manager->convertIndexToBuffer(layer, tag, block);
            require(parts.size() == 1 && parts[0].addr && parts[0].size_bytes > 0,
                    "Next production opaque block part count differs");
            request->addBlock(key,
                              std::shared_ptr<void>(parts[0].addr, [](void*) {}),
                              static_cast<uint32_t>(parts[0].size_bytes),
                              parts[0].is_cuda,
                              true);
        }
        return request;
    };
    CheckResult checked;
    bool        api_ok = true;
    std::string api_error;
    if (sender) {
        checked = checkNext(*manager, owned, oracle);
        require(checked.matches, "Next source-before-transfer mismatch");
        std::vector<std::shared_ptr<RequestBlockBuffer>> requests;
        for (int layer = 0; layer < layers; ++layer)
            requests.push_back(makeLayer(layer));
        auto context = store->storeBuffers(requests, timeout);
        context->waitDone();
        api_ok    = context->success();
        api_error = context->getErrorInfoString();
        require(api_ok, "Next storeBuffers failed: " + api_error);
    }
    mark(root / (rank_stem + ".ready"));
    for (const auto& role : {"sender", "receiver"})
        for (int rank = 0; rank < tp_size; ++rank)
            waitFor(root / (stem(role, rank, tp_size) + ".ready"), timeout);
    if (!sender) {
        std::vector<std::shared_ptr<RequestBlockBuffer>> requests;
        for (int layer = 0; layer < layers; ++layer)
            requests.push_back(makeLayer(layer));
        auto context =
            store->loadBuffers(requests, "127.0.0.1", source_ports[tp_rank], 0, timeout, [] { return false; }, 1, 0);
        context->waitDone();
        api_ok    = context->success();
        api_error = context->getErrorInfoString();
        checked   = checkNext(*manager, owned, oracle);
        mark(root / (rank_stem + ".done"));
    } else {
        for (int rank = 0; rank < tp_size; ++rank)
            waitFor(root / (stem("receiver", rank, tp_size) + ".done"), timeout);
    }
    const bool guards = nextGuards(*manager, owned, oracle);
    manager->setCacheStore(nullptr);
    store.reset();
    manager->free(FreeInfo{resource, complete, 1});
    const auto    free_after = manager->freeBlocksNum();
    const bool    passed     = api_ok && checked.matches && guards && free_before == free_after;
    std::ofstream out(root / (rank_stem + ".result.json"));
    out << "{\"passed\":" << (passed ? "true" : "false") << ",\"pid\":" << getpid()
        << ",\"listen_port\":" << listen_port << ",\"port_selection\":\"kernel\""
        << ",\"backend\":\"normal_cache_store_tcp\",\"tp_size\":" << tp_size << ",\"tp_rank\":" << tp_rank
        << ",\"layout\":\"next\",\"seed\":" << oracle.seed << ",\"expected_signature\":" << quote(hex(checked.expected))
        << ",\"observed_signature\":" << quote(hex(checked.observed)) << ",\"bytes_checked\":" << checked.bytes
        << ",\"payload_matches\":" << (checked.matches ? "true" : "false")
        << ",\"api_ok\":" << (api_ok ? "true" : "false") << ",\"api_error\":" << quote(api_error)
        << ",\"guards_intact\":" << (guards ? "true" : "false") << ",\"free_before\":" << free_before
        << ",\"free_after\":" << free_after << ",\"production_resolved_config\":" << quote(config.debugString())
        << ",\"payload_regions\":" << checked.detail << ",\"logical_units\":" << checked.units << "}\n";
    require(static_cast<bool>(out), "cannot save Next rank result");
    return passed ? 0 : 1;
}
}  // namespace
}  // namespace rtp_llm::cache_smoke
int main(int argc, char** argv) {
    try {
        return rtp_llm::cache_smoke::run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "CACHE_STORE_NEXT_SMOKE_FAILED: " << error.what() << '\n';
        return 1;
    }
}
