#include "rtp_llm/test/smoke/cache/CacheStoreSmokeEndpoint.h"
#include "rtp_llm/test/smoke/cache/CacheSmokeSupport.h"

#include <chrono>
#include <filesystem>
#include <future>
#include <limits>
#include <thread>
#include <unistd.h>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ErrorCodeUtil.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm::cache_smoke {
namespace {

void mark(const std::filesystem::path& path) {
    std::ofstream out(path);
    out << "ready\n";
    require(static_cast<bool>(out), "cannot write readiness control file");
}

void waitFor(const std::filesystem::path& path, int64_t timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!std::filesystem::exists(path)) {
        require(std::chrono::steady_clock::now() < deadline, "control readiness timeout: " + path.string());
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

DataType dtype(const std::string& value) {
    if (value == "bf16")
        return DataType::TYPE_BF16;
    if (value == "fp16")
        return DataType::TYPE_FP16;
    if (value == "fp32")
        return DataType::TYPE_FP32;
    if (value == "int8")
        return DataType::TYPE_INT8;
    if (value == "fp8")
        return DataType::TYPE_FP8_E4M3;
    throw std::runtime_error("unknown dtype: " + value);
}

bool guardsIntact(KVCacheManager& manager, const KVCacheResource& resource, const PayloadOracle& oracle) {
    const auto&   blocks = resource.blocks("default");
    std::set<int> live(blocks.begin(), blocks.end());
    for (uint32_t layer = 0; layer < manager.cacheConfig().layer_num; ++layer) {
        for (uint32_t block = 0; block < manager.cacheConfig().group("default").block_num; ++block) {
            if (live.count(block))
                continue;
            auto address = manager.convertIndexToAddr(layer, "default", block);
            for (bool scale : {false, true}) {
                if (scale && !oracle.scales)
                    continue;
                const auto size  = oracle.payload(layer, 0, scale).size();
                const auto bytes = readBytes(scale ? address.kv_scale_addr : address.kv_addr, size);
                if (!std::all_of(bytes.begin(), bytes.end(), [](uint8_t value) { return value == 0xa5; }))
                    return false;
            }
        }
    }
    return true;
}

CheckResult checkCpMlaPayload(
    KVCacheManager& manager, const KVCacheResource& resource, const PayloadOracle& oracle, bool sender, int cp_rank) {
    CheckResult result;
    const auto& blocks = resource.blocks("default");
    for (uint32_t layer = 0; layer < manager.cacheConfig().layer_num; ++layer) {
        for (size_t local = 0; local < blocks.size(); ++local) {
            const size_t global   = sender ? local * 2 + cp_rank : local;
            const auto   expected = oracle.payload(layer, global, false);
            const auto   address  = manager.convertIndexToAddr(layer, "default", blocks[local]);
            const auto   actual   = readBytes(address.kv_addr, expected.size());
            result.matches &= expected == actual;
            result.expected = signature(expected.data(), expected.size(), result.expected);
            result.observed = signature(actual.data(), actual.size(), result.observed);
            result.bytes += actual.size();
            if (result.detail.size() > 1)
                result.detail += ',';
            result.detail += "{\"layer\":" + std::to_string(layer) + ",\"group\":\"default\",\"logical_block\":"
                             + std::to_string(global) + ",\"physical_block\":" + std::to_string(blocks[local])
                             + ",\"bytes\":" + std::to_string(actual.size())
                             + ",\"expected\":" + quote(hex(signature(expected.data(), expected.size())))
                             + ",\"observed\":" + quote(hex(signature(actual.data(), actual.size())))
                             + ",\"matches\":" + (expected == actual ? "true" : "false") + "}";
            const bool   fp8          = oracle.element_bytes == 1;
            const size_t latent       = oracle.latent_dim * oracle.element_bytes;
            const size_t inline_scale = fp8 ? (oracle.latent_dim / 128) * sizeof(float) : 0;
            const std::vector<std::pair<std::string, std::pair<size_t, size_t>>> components =
                fp8 ?
                    std::vector<std::pair<std::string, std::pair<size_t, size_t>>>{
                        {"latent", {0, latent}},
                        {"inline_scale", {latent, inline_scale}},
                        {"RoPE", {latent + inline_scale, oracle.rope_dim * 2}}} :
                    std::vector<std::pair<std::string, std::pair<size_t, size_t>>>{
                        {"latent", {0, latent}}, {"RoPE", {latent, oracle.rope_dim * 2}}};
            for (const auto& [name, slice] : components) {
                std::vector<uint8_t> exp, obs;
                for (int token = 0; token < oracle.block_tokens; ++token) {
                    const size_t offset = token * oracle.mlaStride() + slice.first;
                    exp.insert(exp.end(), expected.begin() + offset, expected.begin() + offset + slice.second);
                    obs.insert(obs.end(), actual.begin() + offset, actual.begin() + offset + slice.second);
                }
                if (result.units.size() > 1)
                    result.units += ',';
                result.units += "{\"layer\":" + std::to_string(layer) + ",\"group\":\"default\",\"logical_block\":"
                                + std::to_string(global) + ",\"global_head\":0,\"component\":" + quote(name)
                                + ",\"scale\":" + (name == "inline_scale" ? "true" : "false")
                                + ",\"bytes\":" + std::to_string(exp.size())
                                + ",\"expected\":" + quote(hex(signature(exp.data(), exp.size())))
                                + ",\"observed\":" + quote(hex(signature(obs.data(), obs.size())))
                                + ",\"matches\":" + (exp == obs ? "true" : "false") + "}";
            }
        }
    }
    result.detail += ']';
    result.units += ']';
    return result;
}

CheckResult checkCpMhaPayload(
    KVCacheManager& manager, const KVCacheResource& resource, const PayloadOracle& oracle, bool sender, int cp_rank) {
    CheckResult result;
    const auto& blocks = resource.blocks("default");
    for (uint32_t layer = 0; layer < manager.cacheConfig().layer_num; ++layer) {
        for (size_t local = 0; local < blocks.size(); ++local) {
            const size_t global  = sender ? local * 2 + cp_rank : local;
            const auto   address = manager.convertIndexToAddr(layer, "default", blocks[local]);
            for (bool scale : {false, true}) {
                if (scale && !oracle.scales)
                    continue;
                const auto expected = oracle.payload(layer, global, scale);
                const auto actual   = readBytes(scale ? address.kv_scale_addr : address.kv_addr, expected.size());
                result.matches &= expected == actual;
                result.expected = signature(expected.data(), expected.size(), result.expected);
                result.observed = signature(actual.data(), actual.size(), result.observed);
                result.bytes += actual.size();
                if (result.detail.size() > 1)
                    result.detail += ',';
                result.detail += "{\"layer\":" + std::to_string(layer) + ",\"group\":\"default\",\"logical_block\":"
                                 + std::to_string(global) + ",\"physical_block\":" + std::to_string(blocks[local])
                                 + ",\"scale\":" + (scale ? "true" : "false")
                                 + ",\"bytes\":" + std::to_string(actual.size())
                                 + ",\"expected\":" + quote(hex(signature(expected.data(), expected.size())))
                                 + ",\"observed\":" + quote(hex(signature(actual.data(), actual.size())))
                                 + ",\"matches\":" + (expected == actual ? "true" : "false") + "}";
                const size_t head_bytes = actual.size() / (2 * oracle.heads);
                for (int component = 0; component < 2; ++component) {
                    for (int head = 0; head < oracle.heads; ++head) {
                        const size_t offset = (component * oracle.heads + head) * head_bytes;
                        const bool   equal  = std::equal(
                            expected.begin() + offset, expected.begin() + offset + head_bytes, actual.begin() + offset);
                        if (result.units.size() > 1)
                            result.units += ',';
                        result.units += "{\"layer\":" + std::to_string(layer)
                                        + ",\"group\":\"default\",\"logical_block\":" + std::to_string(global)
                                        + ",\"global_head\":" + std::to_string(oracle.first_global_head + head)
                                        + ",\"component\":" + quote(component == 0 ? "K" : "V") + ",\"scale\":"
                                        + (scale ? "true" : "false") + ",\"bytes\":" + std::to_string(head_bytes)
                                        + ",\"expected\":" + quote(hex(signature(expected.data() + offset, head_bytes)))
                                        + ",\"observed\":" + quote(hex(signature(actual.data() + offset, head_bytes)))
                                        + ",\"matches\":" + (equal ? "true" : "false") + "}";
                    }
                }
            }
        }
    }
    result.detail += ']';
    result.units += ']';
    return result;
}

int run(int argc, char** argv) {
    require(
        argc == 22 || argc == 23,
        "rank arguments: role root port target_port seed layers heads kv_heads head_dim block tokens pool deadline dtype mutations tp_size tp_rank remote_tp_size layout latent_dim rope_dim");
    const bool sender = std::string(argv[1]) == "sender";
    require(sender || std::string(argv[1]) == "receiver", "unknown rank role");
    const std::filesystem::path root(argv[2]);
    const auto                  seed               = std::stoull(argv[5]);
    const int                   layers             = std::stoi(argv[6]);
    const int                   heads              = std::stoi(argv[7]);
    const int                   kv_heads           = std::stoi(argv[8]);
    const int                   head_dim           = std::stoi(argv[9]);
    const int                   block_tokens       = std::stoi(argv[10]);
    const int                   tokens             = std::stoi(argv[11]);
    const int                   pool_blocks        = std::stoi(argv[12]);
    const int64_t               timeout            = std::stoll(argv[13]);
    const auto                  type               = dtype(argv[14]);
    const int                   initial_fault_mode = std::stoi(argv[15]);
    const int                   fault_mode         = initial_fault_mode;
    require(fault_mode >= 0 && fault_mode <= 2, "unsupported CacheStore smoke fault mode");
    const int  tp_size        = std::stoi(argv[16]);
    const int  tp_rank        = std::stoi(argv[17]);
    const int  remote_tp_size = std::stoi(argv[18]);
    const bool mla            = std::string(argv[19]) == "mla";
    const int  latent_dim     = std::stoi(argv[20]);
    const int  rope_dim       = std::stoi(argv[21]);
    const bool cp2            = argc == 23 && std::stoi(argv[22]) == 2;
    require(!cp2
                || (fault_mode == 0
                    && ((sender && tp_size == 2 && remote_tp_size == 1)
                        || (!sender && tp_size == 1 && remote_tp_size == 2))),
            "CP2 smoke requires prefill CP2 to decode TP1");
    require(mla || std::string(argv[19]) == "mha", "unknown layout");
    const char* lifecycle_env = std::getenv("CACHE_SMOKE_REQUEST_LIFECYCLE");
    const bool  lifecycle     = lifecycle_env && std::string(lifecycle_env) == "1";
    require(!lifecycle || (!mla && !cp2 && tp_size == remote_tp_size && tokens == 1024),
            "request lifecycle fixture requires symmetric MHA at 1K");
    require(!lifecycle || seed < std::numeric_limits<uint64_t>::max(), "lifecycle seed overflow");
    require(mla || type == DataType::TYPE_BF16 || type == DataType::TYPE_INT8,
            "MHA smoke only registers BF16 and INT8");
    if (mla) {
        require(type == DataType::TYPE_BF16 || type == DataType::TYPE_FP8_E4M3, "MLA smoke supports BF16 and FP8");
        require(latent_dim > 0 && rope_dim > 0 && (type != DataType::TYPE_FP8_E4M3 || latent_dim % 128 == 0),
                "invalid MLA packing dimensions");
    }
    require(tp_size > 0 && tp_rank >= 0 && tp_rank < tp_size, "invalid rank identity");
    require(((tp_size == 1 || tp_size == 2) && (remote_tp_size == 1 || remote_tp_size == 2))
                || (tp_size == 2 && remote_tp_size == 4) || (tp_size == 4 && remote_tp_size == 2),
            "requires TP1/TP2 or TP2/TP4 topology");
    require(heads % tp_size == 0 && kv_heads % tp_size == 0, "fixture requires divisible global heads");
    const auto stem = [](const std::string& role, int rank, int size) {
        return role + (size == 1 ? "" : "." + std::to_string(rank));
    };
    const auto rank_stem = stem(sender ? "sender" : "receiver", tp_rank, tp_size);
    require(std::string(argv[3]) == "0" && std::string(argv[4]) == "auto",
            "smoke requires kernel-assigned production listener ports");
    initLogger();
    initRuntime(0, false, false, MlaOpsType::AUTO);

    ModelConfig model;
    model.num_layers                   = layers;
    model.hidden_size                  = heads * head_dim;
    model.data_type                    = type;
    model.attn_config.head_num         = heads;
    model.attn_config.kv_head_num      = kv_heads;
    model.attn_config.size_per_head    = head_dim;
    model.attn_config.tokens_per_block = block_tokens;
    if (cp2)
        model.attn_config.kernel_tokens_per_block = block_tokens;
    model.attn_config.use_mla       = mla;
    model.attn_config.kv_lora_rank  = latent_dim;
    model.attn_config.rope_head_dim = rope_dim;
    if (mla) {
        require(head_dim > rope_dim, "MLA attention head must include positive nope and RoPE dimensions");
        model.attn_config.nope_head_dim   = head_dim - rope_dim;
        model.attn_config.v_head_dim      = head_dim - rope_dim;
        model.attn_config.rope_config.dim = rope_dim;
    }
    model.kv_cache_spec_descs.resize(layers);
    for (auto& descs : model.kv_cache_spec_descs) {
        KVCacheSpecDesc desc;
        desc.tag        = "default";
        desc.cache_type = mla ? KVCacheSpecType::MultiHeadLatentAttention : KVCacheSpecType::MultiHeadAttention;
        desc.dtype      = type;
        descs.push_back(desc);
    }
    ParallelismConfig parallel;
    parallel.tp_size = tp_size;
    parallel.tp_rank = tp_rank;
    if (cp2) {
        parallel.world_size               = tp_size;
        parallel.world_rank               = tp_rank;
        parallel.role_type                = sender ? RoleType::PREFILL : RoleType::DECODE;
        parallel.prefill_cp_config.method = sender ? CPRotateMethod::ALL_GATHER : CPRotateMethod::PREFILL_CP;
        // DecodeRpcServer::prepareGenerateContext only advertises prefill_cp_size=2
        // when the decode side also enables this flag.
        parallel.prefill_cp_config.kv_cache_sharded = true;
        parallel.prefill_cp_config.prefill_cp_size  = 2;
        require(parallel.get_attn_tp_size() == (sender ? 1 : tp_size), "CP2 MLA attention TP differs");
    }
    KVCacheConfig options;
    options.test_block_num      = pool_blocks;
    options.reserve_block_ratio = 0;
    options.reuse_cache         = false;
    auto config                 = CacheConfigCreator::createConfig(model, parallel, options);
    config.finalizeBlockNums(
        CacheConfigCreator::computeLocalBlockNum(config, model, RuntimeConfig{}, options, parallel), RuntimeConfig{});
    PDSepConfig manager_pd;
    if (cp2)
        manager_pd.role_type = parallel.role_type;
    auto manager = std::make_shared<KVCacheManager>(
        config, false, nullptr, options, parallel, RuntimeConfig{}, SpeculativeExecutionConfig{}, manager_pd);
    require(manager->init(), "manager init failed");
    const auto    free_before  = manager->freeBlocksNum();
    const bool    scales       = !mla && (type == DataType::TYPE_INT8 || type == DataType::TYPE_FP8_E4M3);
    const int     attention_tp = cp2 && sender ? 1 : tp_size;
    PayloadOracle oracle{seed,
                         mla ? 1 : kv_heads / attention_tp,
                         head_dim,
                         block_tokens,
                         getTypeSize(type),
                         scales,
                         mla ? 0 : (cp2 && sender ? 0 : tp_rank * (kv_heads / attention_tp)),
                         mla,
                         latent_dim,
                         rope_dim};
    require(config.group("default").kvBlockStrideBytes() == oracle.payload(0, 0, false).size(),
            "independent cache payload geometry differs from production resolved spec");
    require(config.group("default").kvScaleStrideBytes() == (scales ? oracle.payload(0, 0, true).size() : 0),
            "independent scale geometry differs from production resolved spec");
    auto                 converter = std::make_shared<ManagerConverter>(manager);
    CacheStoreInitParams params;
    params.listen_port          = 0;
    params.rdma_mode            = false;
    params.enable_metric        = false;
    params.device_id            = 0;
    params.thread_count         = 4;
    const auto listeners_before = listeningTcpPorts();
    auto       store            = NormalCacheStore::createNormalCacheStore(params);
    require(store != nullptr, "production NormalCacheStore init failed");
    const auto listen_port  = publishSmokeEndpoint(listeners_before, root, rank_stem);
    const auto source_ports = readSmokeSourcePorts(root, sender ? tp_size : remote_tp_size, timeout);
    manager->setCacheStore(store);

    std::vector<std::string> request_reports;
    bool                     all_passed = true;
    for (int round = 0; round < (lifecycle ? 2 : 1); ++round) {
        const int64_t     request_number = round + 1;
        const std::string request_key    = "cache-smoke-" + std::to_string(request_number);
        const auto        control_root   = root / ("request-" + std::to_string(request_number));
        std::filesystem::create_directories(control_root);
        oracle.seed          = seed + round;
        const int fault_mode = round == 0 ? initial_fault_mode : 0;
        for (const auto& [info, size] : converter->getAllBuffers()) {
            require(info.is_cuda, "Cache smoke requires actual DEVICE backing");
            require(cudaMemset(info.addr, 0xa5, size) == cudaSuccess, "poison DEVICE backing failed");
        }
        require(cudaDeviceSynchronize() == cudaSuccess, "poison synchronization failed");

        auto resource = std::make_shared<BatchKVCacheResource>();
        resource->resetBatchSize(1);
        resource->initGroups(config.topologyPtr());
        auto input             = std::make_shared<GenerateInput>();
        input->input_ids       = torch::arange(tokens, torch::kInt32);
        input->generate_config = std::make_shared<GenerateConfig>();
        auto complete          = std::make_shared<CompleteTokenIds>(1, 1, tokens + block_tokens, block_tokens);
        complete->init(input);
        MallocInfo allocation{resource, complete};
        allocation.request_id          = request_number;
        allocation.reuse_cache         = false;
        allocation.enable_cache_lookup = false;
        const auto allocated           = manager->malloc(allocation);
        require(allocated.success,
                "production malloc failed status=" + std::to_string(static_cast<int>(allocated.status)));
        auto& owned = resource->cacheResource();
        require(owned.blocks("default").size() == static_cast<size_t>(tokens / block_tokens / (cp2 && sender ? 2 : 1)),
                "production allocated logical block count differs from complete token sequence");
        if (cp2)
            require(owned.cacheKeys().size() == static_cast<size_t>(tokens / block_tokens),
                    "CP source/target full cache key namespace differs");
        if (sender) {
            for (int layer = 0; layer < layers; ++layer) {
                for (size_t logical = 0; logical < owned.blocks("default").size(); ++logical) {
                    auto address = manager->convertIndexToAddr(layer, "default", owned.blocks("default")[logical]);
                    const size_t global = cp2 ? logical * 2 + tp_rank : logical;
                    writeBytes(address.kv_addr, oracle.payload(layer, global, false));
                    if (scales)
                        writeBytes(address.kv_scale_addr, oracle.payload(layer, global, true));
                }
            }
        }

        auto make_layer = [&](int layer, int peer_count, int peer_index, bool source) {
            auto        request = std::make_shared<RequestBlockBuffer>(request_key);
            const auto& blocks  = owned.blocks("default");
            for (size_t logical = 0; logical < blocks.size(); ++logical) {
                const size_t global = cp2 && source ? logical * 2 + tp_rank : logical;
                if (cp2 && !source && static_cast<int>(global % 2) != peer_index)
                    continue;
                const auto token_key = cp2 ? std::to_string(owned.cacheKeys().at(global)) : std::to_string(global + 1);
                const auto key       = makeCacheKey(1, token_key, layer, "default");
                auto       parts =
                    (source || cp2) ?
                              manager->convertIndexToBuffer(layer, "default", blocks[logical]) :
                              manager->convertIndexToBuffer(layer, "default", blocks[logical], peer_count, peer_index);
                if (source && !mla) {
                    // ExecOps publishes each manager-owned K/V block as two keyed halves.
                    // The manager's unsliced converter returns the whole KV tensor.
                    require(parts.size() == (scales ? 2u : 1u), "source KV/scale part count differs");
                    std::vector<BlockInfo> halves;
                    for (auto part : parts) {
                        require(part.size_bytes % 2 == 0, "source KV/scale block is not divisible into K/V");
                        part.size_bytes /= 2;
                        halves.push_back(part);
                        part.addr = static_cast<uint8_t*>(part.addr) + part.size_bytes;
                        halves.push_back(part);
                    }
                    parts = std::move(halves);
                }
                const std::vector<std::string> names =
                    mla    ? std::vector<std::string>{"kv_"} :
                    scales ? std::vector<std::string>{"k_", "v_", "k_scale_", "v_scale_"} :
                             std::vector<std::string>{"k_", "v_"};
                require(parts.size() == names.size(),
                        cp2 && !mla && !source ?
                            "DecodeRpcServer CP MHA whole-block converter returned " + std::to_string(parts.size())
                                + " part(s), but its K/V request builder requires " + std::to_string(names.size()) :
                            "production block part count differs");
                for (size_t index = 0; index < parts.size(); ++index) {
                    const auto& part = parts[index];
                    require(part.addr && part.size_bytes > 0 && part.is_cuda, "invalid production block part");
                    request->addBlock(names[index] + key,
                                      std::shared_ptr<void>(part.addr, [](void*) {}),
                                      static_cast<uint32_t>(part.size_bytes),
                                      true,
                                      true);
                }
            }
            return request;
        };

        CheckResult checked;
        bool        api_ok = true;
        std::string api_error;
        bool        fault_rejected        = false;
        bool        fault_bytes_unchanged = false;
        std::string fault_error;
        if (sender) {
            checked = cp2 ? (mla ? checkCpMlaPayload(*manager, owned, oracle, true, tp_rank) :
                                   checkCpMhaPayload(*manager, owned, oracle, true, tp_rank)) :
                            checkPayload(*manager, owned, oracle);
            require(checked.matches, "source-before-publication mismatch");
            std::vector<std::shared_ptr<RequestBlockBuffer>> layers_to_store;
            for (int layer = 0; layer < layers; ++layer)
                layers_to_store.push_back(make_layer(layer, 1, 0, true));
            auto context = store->storeBuffers(layers_to_store, timeout);
            context->waitDone();
            api_ok    = context->success();
            api_error = context->getErrorInfoString();
            require(api_ok, "production storeBuffers failed: " + api_error);
        }
        mark(control_root / (rank_stem + ".ready"));
        for (const auto& role : {"sender", "receiver"}) {
            const int role_size = (std::string(role) == "sender") == sender ? tp_size : remote_tp_size;
            for (int rank = 0; rank < role_size; ++rank)
                waitFor(control_root / (stem(role, rank, role_size) + ".ready"), timeout);
        }
        if (!sender) {
            if (fault_mode != 0) {
                require(tp_size == 1 && remote_tp_size == 1, "fault fixture requires TP1-to-TP1");
                const auto destination = manager->convertIndexToBuffer(0, "default", owned.blocks("default")[0]);
                require(!destination.empty(), "fault destination unavailable");
                const auto before = readBytes(destination[0].addr, destination[0].size_bytes);
                auto       bad    = std::make_shared<RequestBlockBuffer>(request_key);
                const auto key    = "k_" + makeCacheKey(1, "1", 0, "default");
                bad->addBlock(key,
                              std::shared_ptr<void>(destination[0].addr, [](void*) {}),
                              static_cast<uint32_t>(destination[0].size_bytes / 2 + 1),
                              true,
                              true);
                auto fault_context = store->loadBuffers(
                    {bad}, "127.0.0.1", fault_mode == 1 ? 0 : source_ports[0], 0, timeout, [] { return false; }, 1, 0);
                fault_context->waitDone();
                fault_error           = fault_context->getErrorInfoString();
                fault_rejected        = !fault_context->success();
                fault_bytes_unchanged = readBytes(destination[0].addr, destination[0].size_bytes) == before;
                require(fault_rejected && fault_bytes_unchanged,
                        "production load fault did not reject cleanly or modified destination");
            }
            // On a receiver, tp_size is decode TP and remote_tp_size is prefill TP.
            const int peer_count = cp2 ? 2 : remote_tp_size >= tp_size ? remote_tp_size / tp_size : 1;
            const int first_peer = cp2                       ? 0 :
                                   remote_tp_size >= tp_size ? tp_rank * peer_count :
                                                               tp_rank / (tp_size / remote_tp_size);
            for (int peer_index = 0; peer_index < peer_count; ++peer_index) {
                std::vector<std::shared_ptr<RequestBlockBuffer>> layers_to_load;
                for (int layer = 0; layer < layers; ++layer)
                    layers_to_load.push_back(make_layer(layer, peer_count, peer_index, false));
                const int partition_count = cp2 ? 1 : tp_size > remote_tp_size ? tp_size / remote_tp_size : 1;
                const int partition_id    = cp2 ? 0 : tp_size > remote_tp_size ? tp_rank % partition_count : 0;
                auto      context         = store->loadBuffers(
                    layers_to_load,
                    "127.0.0.1",
                    source_ports[first_peer + peer_index],
                    0,
                    timeout,
                    [] { return false; },
                    partition_count,
                    partition_id);
                context->waitDone();
                if (!context->success()) {
                    api_ok    = false;
                    api_error = context->getErrorInfoString();
                    break;
                }
            }
            checked = cp2 ? (mla ? checkCpMlaPayload(*manager, owned, oracle, false, 0) :
                                   checkCpMhaPayload(*manager, owned, oracle, false, 0)) :
                            checkPayload(*manager, owned, oracle);
            mark(control_root / (rank_stem + ".done"));
        } else {
            for (int rank = 0; rank < remote_tp_size; ++rank)
                waitFor(control_root / (stem("receiver", rank, remote_tp_size) + ".done"), timeout);
        }
        bool        expired_rejected = false, expired_bytes_unchanged = false;
        std::string expired_error;
        if (lifecycle) {
            require(api_ok && checked.matches, "lifecycle transfer failed before request end");
            if (sender) {
                store->markRequestEnd(request_key);
                mark(control_root / (rank_stem + ".ended"));
            }
            const int sender_count = sender ? tp_size : remote_tp_size;
            for (int rank = 0; rank < sender_count; ++rank)
                waitFor(control_root / (stem("sender", rank, sender_count) + ".ended"), timeout);
            if (!sender) {
                // Independent probe destination stays owned by the real async load.
                const size_t bytes = oracle.payload(0, 0, false).size() / 2;
                auto         probe = torch::empty({static_cast<int64_t>(bytes)},
                                          torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
                writeBytes(probe.data_ptr(), std::vector<uint8_t>(bytes, 0x69));
                auto expired = std::make_shared<RequestBlockBuffer>(request_key);
                expired->addBlock("k_" + makeCacheKey(1, "1", 0, "default"),
                                  std::shared_ptr<void>(probe.data_ptr(), [probe](void*) {}),
                                  static_cast<uint32_t>(bytes),
                                  true,
                                  true);
                auto completion = std::make_shared<std::promise<std::pair<bool, CacheStoreErrorCode>>>();
                auto done       = completion->get_future();
                store->load(
                    expired,
                    [completion](bool ok, CacheStoreErrorCode error) { completion->set_value({ok, error}); },
                    "127.0.0.1",
                    source_ports[tp_rank],
                    0,
                    timeout,
                    1,
                    0);
                require(done.wait_for(std::chrono::milliseconds(timeout)) == std::future_status::ready,
                        "expired request callback did not complete");
                const auto [ok, error]  = done.get();
                expired_rejected        = !ok && error == CacheStoreErrorCode::LoadBufferTimeout;
                expired_error           = ErrorCodeToString(transCacheStoreErrorCode(error));
                expired_bytes_unchanged = readBytes(probe.data_ptr(), bytes) == std::vector<uint8_t>(bytes, 0x69);
                mark(control_root / (rank_stem + ".expired_checked"));
            } else {
                for (int rank = 0; rank < remote_tp_size; ++rank)
                    waitFor(control_root / (stem("receiver", rank, remote_tp_size) + ".expired_checked"), timeout);
            }
        }
        const bool guards         = guardsIntact(*manager, owned, oracle);
        const auto free_allocated = manager->freeBlocksNum();
        if (!lifecycle) {
            manager->setCacheStore(nullptr);
            store.reset();
        }
        manager->free(FreeInfo{resource, complete, request_number});
        const auto free_after = manager->freeBlocksNum();
        const bool passed     = api_ok && checked.matches && guards && free_before == free_after
                            && (fault_mode == 0 || sender || (fault_rejected && fault_bytes_unchanged))
                            && (!lifecycle || sender || (expired_rejected && expired_bytes_unchanged));
        if (lifecycle) {
            mark(control_root / (rank_stem + ".released"));
            for (const auto& role : {"sender", "receiver"}) {
                const int size = (std::string(role) == "sender") == sender ? tp_size : remote_tp_size;
                for (int rank = 0; rank < size; ++rank)
                    waitFor(control_root / (stem(role, rank, size) + ".released"), timeout);
            }
        }
        std::ostringstream out;
        out << "{\"passed\":" << (passed ? "true" : "false") << ",\"pid\":" << getpid()
            << ",\"request_number\":" << request_number << ",\"request_key\":" << quote(request_key)
            << ",\"request_end_checked\":" << (lifecycle ? "true" : "false")
            << ",\"expired_rejected\":" << (expired_rejected ? "true" : "false")
            << ",\"expired_bytes_unchanged\":" << (expired_bytes_unchanged ? "true" : "false")
            << ",\"expired_error\":" << quote(expired_error) << ",\"listen_port\":" << listen_port
            << ",\"port_selection\":\"kernel\""
            << ",\"backend\":\"normal_cache_store_tcp\",\"seed\":" << oracle.seed << ",\"tp_size\":" << tp_size
            << ",\"tp_rank\":" << tp_rank << ",\"remote_tp_size\":" << remote_tp_size
            << ",\"first_global_head\":" << oracle.first_global_head << ",\"head_count\":" << oracle.heads
            << ",\"expected_signature\":" << quote(hex(checked.expected))
            << ",\"observed_signature\":" << quote(hex(checked.observed)) << ",\"bytes_checked\":" << checked.bytes
            << ",\"payload_matches\":" << (checked.matches ? "true" : "false")
            << ",\"api_ok\":" << (api_ok ? "true" : "false") << ",\"api_error\":" << quote(api_error)
            << ",\"fault_mode\":" << fault_mode << ",\"fault_rejected\":" << (fault_rejected ? "true" : "false")
            << ",\"fault_bytes_unchanged\":" << (fault_bytes_unchanged ? "true" : "false")
            << ",\"fault_error\":" << quote(fault_error) << ",\"guards_intact\":" << (guards ? "true" : "false")
            << ",\"free_before\":" << free_before << ",\"free_allocated\":" << free_allocated
            << ",\"free_after\":" << free_after << ",\"production_resolved_config\":" << quote(config.debugString())
            << ",\"payload_regions\":" << checked.detail << ",\"logical_units\":" << checked.units << "}\n";
        require(static_cast<bool>(out), "cannot save rank result");
        request_reports.push_back(out.str());
        std::ofstream raw(control_root / (rank_stem + ".result.json"));
        raw << out.str();
        raw.close();
        require(static_cast<bool>(raw), "cannot save request result");
        all_passed &= passed;
    }
    manager->setCacheStore(nullptr);
    store.reset();
    auto combined = request_reports.front();
    combined.erase(combined.find_last_of('}'));
    combined += ",\"lifecycle\":" + std::string(lifecycle ? "true" : "false") + ",\"all_requests_passed\":"
                + (all_passed ? "true" : "false") + ",\"store_initializations\":1,\"request_runs\":[";
    for (size_t i = 0; i < request_reports.size(); ++i) {
        if (i)
            combined += ',';
        combined += request_reports[i];
    }
    combined += "]}";
    std::ofstream out(root / (rank_stem + ".result.json"));
    out << combined << '\n';
    out.close();
    require(static_cast<bool>(out), "cannot save lifecycle result");
    return all_passed ? 0 : 1;
}
}  // namespace
}  // namespace rtp_llm::cache_smoke

int main(int argc, char** argv) {
    try {
        return rtp_llm::cache_smoke::run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "CACHE_STORE_SMOKE_FAILED: " << error.what() << '\n';
        return 1;
    }
}
