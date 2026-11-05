#pragma once

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <cstdlib>
#include <string>

namespace rtp_llm::test {

// Time-shared prototype: retain published CPU pages after the PP workers exit,
// then load them over the production TCP RPC transport into cold DP workers.
class V41PDDummyStore {
public:
    explicit V41PDDummyStore(uint32_t port) {
        const auto flag = std::getenv("DSV41_PD_DUMMY_DECODE");
        if (!flag || std::string(flag) != "1") {
            throw std::invalid_argument("dummy PD store requires DSV41_PD_DUMMY_DECODE=1");
        }
        CacheStoreInitParams params;
        params.listen_port                  = port;
        params.rdma_mode                    = false;
        params.enable_metric                = false;
        params.messager_io_thread_count     = 1;
        params.messager_worker_thread_count = 2;
        store_                              = NormalCacheStore::createNormalCacheStore(params);
        if (!store_) {
            throw std::runtime_error("dummy PD TCP listener failed");
        }
    }

    std::vector<std::string> publish(const ModelConfig&                                          model,
                                     const ParallelismConfig&                                    parallelism,
                                     const std::map<std::pair<int, std::string>, torch::Tensor>& raw,
                                     const std::map<std::string, torch::Tensor>&                 tables,
                                     int64_t                                                     request_id,
                                     int                                                         length) {
        RuntimeConfig runtime;
        KVCacheConfig options;
        options.test_block_num     = 8;
        options.seq_size_per_block = options.kernel_seq_size_per_block = 256;
        const auto config                                              = CacheConfigCreator::createConfig(
            model, parallelism, runtime, options, std::nullopt, SpeculativeExecutionConfig{});
        std::vector<std::string> keys;
        for (const auto& [binding, tensor] : raw) {
            const auto& [layer, tag]            = binding;
            const auto&                   group = config.groupForLayer(layer, tag);
            torch_ext::PyCacheStoreInputs inputs;
            inputs.input_lengths_host    = torch::tensor({length}, torch::kInt32);
            inputs.prefix_lengths_host   = torch::zeros({1}, torch::kInt32);
            inputs.host_kv_cache_offset  = tables.at(tag).slice(0, 0, 1).contiguous();
            inputs.request_id            = torch::tensor({request_id}, torch::kInt64);
            inputs.request_pd_separation = torch::ones({1}, torch::kBool);
            inputs.cache_keys            = torch::full({1, 1}, 87654, torch::kInt64);
            torch_ext::LayerKVCache surface(tensor, group.seq_size_per_block, layer, -1, tag);
            runtimeWriteCacheStore(inputs, surface, config, store_, 0, parallelism.tp_rank, 4, nullptr);
            if (!group.read_replica_source_layer_id.has_value()) {
                keys.push_back("kv_" + makeCacheKey(0, "87654", config.global_layer_begin + layer, tag));
            }
        }
        // Publication is asynchronous. A loopback load is the completion fence;
        // actual decoder loads still use a separate listener/process below.
        return keys;
    }

    void load(int64_t request_id, uint32_t peer_port, const std::map<std::string, torch::Tensor>& destinations) {
        auto buffers = std::make_shared<RequestBlockBuffer>(std::to_string(request_id));
        for (const auto& [key, tensor] : destinations) {
            if (!tensor.is_contiguous() || tensor.scalar_type() != torch::kUInt8) {
                throw std::invalid_argument("dummy PD destinations must be contiguous bytes");
            }
            auto owner = std::make_shared<torch::Tensor>(tensor);
            buffers->addBlock(
                key, std::shared_ptr<void>(owner, tensor.data_ptr()), tensor.nbytes(), tensor.is_cuda(), true);
        }
        auto context = store_->loadBuffers({buffers}, "127.0.0.1", peer_port, 0, 30000, [] { return false; }, 1, 0);
        context->waitDone();
        if (!context->success()) {
            throw std::runtime_error(context->getErrorInfoString());
        }
    }

    void close() {
        if (store_) {
            store_->stopTransport();
            store_.reset();
        }
    }

private:
    std::shared_ptr<NormalCacheStore> store_;
};

}  // namespace rtp_llm::test
