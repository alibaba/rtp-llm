#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheAllocationLease.h"
#include "rtp_llm/cpp/cache/PPTopologyValidator.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPExecutor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/V41PPLocalCache.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {
void registerExecCtxOps(pybind11::module& m);

namespace test {

// Test-only handles: all ownership comes from actual live native streams.
class V41PPAllocation {
public:
    V41PPAllocation(std::shared_ptr<KVCacheManager> manager, GenerateStreamPtr stream, int64_t generation):
        manager_(std::move(manager)), stream_(std::move(stream)), generation_(generation) {
        lease_ = stream_->streamCacheResource().retainAllocation();
        request_id_ = stream_->streamId();
        incarnation_ = stream_->streamCacheResource().cacheIncarnation();
    }

    KVCacheAllocationLease::BlockTables tables() const {
        if (!lease_) {
            throw std::invalid_argument("test allocation lease has been released");
        }
        return lease_->blockTables();
    }

    std::vector<int64_t> identity() const {
        return {request_id_, generation_, incarnation_};
    }

    void releaseOwner() {
        if (stream_) {
            stream_->releaseResource();
            stream_.reset();
        }
    }

    void releaseLease() {
        lease_.reset();
    }

    const std::shared_ptr<KVCacheManager>& manager() const {
        return manager_;
    }

    KVCacheAllocationLeasePtr retain() const {
        if (!lease_) {
            throw std::invalid_argument("test allocation lease has been released");
        }
        return lease_;
    }

private:
    std::shared_ptr<KVCacheManager> manager_;
    GenerateStreamPtr              stream_;
    KVCacheAllocationLeasePtr      lease_;
    int64_t                        generation_;
    int64_t                        request_id_;
    int64_t                        incarnation_;
};

class V41NativePPTransportTest {
public:
    V41NativePPTransportTest(ModelConfig config, const ParallelismConfig& parallelism):
        model_config_(std::move(config)), parallelism_(parallelism), pp_rank_(parallelism.pp_rank) {
        static std::once_flag runtime_once;
        std::call_once(runtime_once, [&]() {
            AUTIL_ROOT_LOG_CONFIG();
            initRuntime(parallelism.local_rank, false, false, MlaOpsType::AUTO);
        });
        RTP_LLM_CHECK(parallelism.pp_size == 2 && parallelism.tp_size == 4 && parallelism.dp_size == 1
                      && parallelism.world_size == 8 && model_config_.num_layers == 40
                      && model_config_.gen_num_per_cycle == 0);
        model_config_.model_type = "deepseek_v41";
        model_config_.input_vocab_size = model_config_.vocab_size;
        model_config_.special_tokens.eos_token_id = -1;
        model_config_.data_type = DataType::TYPE_BF16;
        RuntimeConfig runtime;
        runtime.warm_up = false;
        runtime.max_generate_batch_size = 4;
        runtime.fifo_scheduler_config.max_context_batch_size = 4;
        runtime.use_batch_decode_scheduler = false;
        KVCacheConfig cache_options;
        cache_options.test_block_num = 12 + parallelism.world_rank;
        cache_options.seq_size_per_block = cache_options.kernel_seq_size_per_block = 256;
        cache_options.reuse_cache = false;
        const auto cache = CacheConfigCreator::createConfig(
            model_config_, parallelism, runtime, cache_options, std::nullopt, SpeculativeExecutionConfig{});
        PDSepConfig role;
        role.role_type = RoleType::PREFILL;
        manager_ = std::make_shared<KVCacheManager>(cache,
                                                   false,
                                                   nullptr,
                                                   cache_options,
                                                   parallelism,
                                                   runtime,
                                                   SpeculativeExecutionConfig{},
                                                   role,
                                                   CacheStoreConfig{},
                                                   false,
                                                   std::make_shared<PPCacheCapacityNegotiator>(1.5, true));
        if (!manager_->init()) {
            throw std::runtime_error("native PP test cache manager initialization failed");
        }
        EngineInitParams params;
        params.model_config_ = model_config_;
        params.parallelism_config = parallelism;
        params.runtime_config = runtime;
        params.kv_cache_config = cache_options;
        params.pd_sep_config = role;
        params.hw_kernel_config.enable_cuda_graph = false;
        params.gpt_weights.layers.resize(40);
        params.py_model = params.py_eplb = params.py_sp_model = params.weight_manager = py::none();
        executor_ = std::make_unique<PPExecutor>(params, manager_, false);
    }

    std::shared_ptr<V41PPAllocation> allocate(int64_t request_id, int length) {
        requireOpen();
        if (request_id < 0 || length <= 0 || length >= model_config_.max_seq_len) {
            throw std::invalid_argument("native PP test request is out of bounds");
        }
        ResourceContext context;
        context.cache_manager = manager_;
        context.reuse_cache = false;
        auto input = std::make_shared<GenerateInput>();
        input->input_ids = torch::zeros({length}, torch::kInt32);
        input->request_id = request_id;
        input->generate_config = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 1;
        input->generate_config->top_k = 1;
        auto stream = std::make_shared<NormalGenerateStream>(input, model_config_, RuntimeConfig{}, context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        stream->setIsContextStream(true);
        const auto status = stream->streamCacheResource().initKVBlock();
        if (!status.ok()) {
            throw std::runtime_error(status.ToString());
        }
        return std::make_shared<V41PPAllocation>(manager_, std::move(stream), ++generation_);
    }

    torch_ext::KVCache cache() const {
        requireOpen();
        return torch_ext::KVCache(manager_->getMainModelGroupedCacheLayerLayout());
    }

    V41PPCacheBatchPtr prepareLocal(const torch::Tensor& spans,
                                    const std::vector<std::shared_ptr<V41PPAllocation>>& allocations) {
        requireOpen();
        if (!local_cache_) {
            local_cache_ = std::make_unique<V41PPLocalCache>(manager_, model_config_, parallelism_);
        }
        std::vector<KVCacheAllocationLeasePtr> leases;
        for (const auto& allocation : allocations) {
            if (!allocation || allocation->manager() != manager_) {
                throw std::invalid_argument("native PP local fixture requires this manager's root allocation");
            }
            leases.push_back(allocation->retain());
        }
        return local_cache_->prepare(spans, leases);
    }

    void commitLocal(const V41PPCacheBatchPtr& batch) {
        requireOpen();
        requireLocal();
        local_cache_->commit(batch);
    }

    void abortLocal(const V41PPCacheBatchPtr& batch) {
        requireOpen();
        requireLocal();
        local_cache_->abort(batch);
    }

    void retireLocal(const std::array<int64_t, 3>& source) {
        requireOpen();
        requireLocal();
        local_cache_->retire(source);
    }

    size_t activeLocal() const {
        return local_cache_ ? local_cache_->activeRequests() : 0;
    }

    void checkLocalConfig(const ModelConfig& model, const ParallelismConfig& parallelism) {
        requireOpen();
        V41PPLocalCache candidate(manager_, model, parallelism);
    }

    void sendLocal(const std::map<std::string, torch::Tensor>& tensors, const V41PPCacheBatchPtr& batch) {
        if (pp_rank_ != 0) {
            throw std::invalid_argument("native PP local test send requires the first stage");
        }
        retainLocalRound(batch);
        executor_->asyncSendTensors(PPIntermediateTensors{tensors}, tickets_);
    }

    std::map<std::string, torch::Tensor> receiveLocal(const V41PPCacheBatchPtr& batch) {
        if (pp_rank_ != 1) {
            throw std::invalid_argument("native PP local test receive requires the second stage");
        }
        retainLocalRound(batch);
        return executor_->receiveTensors(tickets_).tensors;
    }

    void send(const std::map<std::string, torch::Tensor>& tensors,
              const std::vector<std::shared_ptr<V41PPAllocation>>& allocations) {
        if (pp_rank_ != 0) {
            throw std::invalid_argument("native PP test send requires the first stage");
        }
        retainRound(allocations);
        executor_->asyncSendTensors(PPIntermediateTensors{tensors}, tickets_);
    }

    std::map<std::string, torch::Tensor>
    receive(const std::vector<std::shared_ptr<V41PPAllocation>>& allocations) {
        if (pp_rank_ != 1) {
            throw std::invalid_argument("native PP test receive requires the second stage");
        }
        retainRound(allocations);
        return executor_->receiveTensors(tickets_).tensors;
    }

    void wait() {
        requireOpen();
        executor_->waitAll(tickets_, "V4.1 test named-tensor handoff");
        // CUDA ticket waits enqueue stream ordering. This test needs completed
        // hardware work before inspecting byte/capacity evidence on the host.
        cudaSyncAndCheck();
    }

    void finish() {
        requireOpen();
        if (!tickets_.empty()) {
            throw std::invalid_argument("native PP test must drain tickets before releasing round leases");
        }
        // Receiver import/consumers may run after wait(); drain them separately.
        cudaSyncAndCheck();
        round_leases_.clear();
    }

    size_t freeBlocks() const {
        requireOpen();
        return manager_->freeBlocksNum();
    }

    uint32_t capacity() const {
        requireOpen();
        return manager_->cacheConfig().block_num;
    }

    size_t pendingTickets() const {
        return tickets_.size();
    }

    size_t retainedLeases() const {
        return round_leases_.size();
    }

    void close() {
        finish();
        local_cache_.reset();
        executor_.reset();
        manager_->stopMetricsReporting();
        manager_.reset();
    }

private:
    void requireLocal() const {
        if (!local_cache_) {
            throw std::invalid_argument("native PP local fixture has not prepared a dispatch");
        }
    }

    void requireOpen() const {
        if (!manager_ || !executor_) {
            throw std::invalid_argument("native PP test fixture is closed");
        }
    }

    void retainRound(const std::vector<std::shared_ptr<V41PPAllocation>>& allocations) {
        requireOpen();
        if (!tickets_.empty() || !round_leases_.empty()) {
            throw std::invalid_argument("native PP test has an unfinished round");
        }
        std::vector<KVCacheAllocationLeasePtr> leases;
        for (const auto& allocation : allocations) {
            if (!allocation || allocation->manager() != manager_) {
                throw std::invalid_argument("native PP test requires this rank's actual allocation");
            }
            leases.push_back(allocation->retain());
        }
        round_leases_ = std::move(leases);
    }

    void retainLocalRound(const V41PPCacheBatchPtr& batch) {
        requireOpen();
        if (!local_cache_ || !local_cache_->ownsBatch(batch) || !tickets_.empty() || !round_leases_.empty()) {
            throw std::invalid_argument("native PP local test requires its admitted batch and a drained round");
        }
        round_leases_ = batch->leases();
    }

    ModelConfig                           model_config_;
    ParallelismConfig                    parallelism_;
    int64_t                               pp_rank_;
    int64_t                               generation_ = 0;
    std::shared_ptr<KVCacheManager>         manager_;
    std::unique_ptr<PPExecutor>            executor_;
    std::unique_ptr<V41PPLocalCache>       local_cache_;
    PPTickets                             tickets_;
    std::vector<KVCacheAllocationLeasePtr> round_leases_;
};

}  // namespace test
}  // namespace rtp_llm

PYBIND11_MODULE(libth_v41_native_pp_transport_test, m) {
    rtp_llm::registerExecCtxOps(m);
    py::class_<rtp_llm::V41PPCacheBatch, rtp_llm::V41PPCacheBatchPtr>(m, "LocalBatch")
        .def_property_readonly("block_tables", [](const rtp_llm::V41PPCacheBatch& batch) {
            std::vector<rtp_llm::KVCacheAllocationLease::BlockTables> tables;
            for (const auto& binding : batch.bindings()) {
                tables.push_back(binding.lease->blockTables());
            }
            return tables;
        })
        .def_property_readonly("local_generations", [](const rtp_llm::V41PPCacheBatch& batch) {
            std::vector<int64_t> generations;
            for (const auto& binding : batch.bindings()) {
                generations.push_back(binding.local_generation);
            }
            return generations;
        })
        .def_property_readonly("new_requests", [](const rtp_llm::V41PPCacheBatch& batch) {
            std::vector<bool> fresh;
            for (const auto& binding : batch.bindings()) {
                fresh.push_back(binding.newly_admitted);
            }
            return fresh;
        })
        .def_property_readonly("kernel_block_table", &rtp_llm::V41PPCacheBatch::kernelBlockTable)
        .def_property_readonly("canonical_columns", [](const rtp_llm::V41PPCacheBatch& batch) {
            std::map<std::string, size_t> columns;
            for (const auto& group : batch.bindings().at(0).lease->topologyPtr()->groups()) {
                columns.emplace(group.tag, group.canonical_idx);
            }
            return columns;
        });
    py::class_<rtp_llm::test::V41PPAllocation, std::shared_ptr<rtp_llm::test::V41PPAllocation>>(m, "Allocation")
        .def_property_readonly("block_tables", &rtp_llm::test::V41PPAllocation::tables)
        .def_property_readonly("identity", &rtp_llm::test::V41PPAllocation::identity)
        .def("release_owner", &rtp_llm::test::V41PPAllocation::releaseOwner)
        .def("release_lease", &rtp_llm::test::V41PPAllocation::releaseLease);
    py::class_<rtp_llm::test::V41NativePPTransportTest>(m, "NativePPTransport")
        .def(py::init<rtp_llm::ModelConfig, const rtp_llm::ParallelismConfig&>())
        .def("allocate", &rtp_llm::test::V41NativePPTransportTest::allocate)
        .def("prepare_local", &rtp_llm::test::V41NativePPTransportTest::prepareLocal,
             py::arg("spans"),
             py::arg("root_allocations") = std::vector<std::shared_ptr<rtp_llm::test::V41PPAllocation>>{})
        .def("commit_local", &rtp_llm::test::V41NativePPTransportTest::commitLocal)
        .def("abort_local", &rtp_llm::test::V41NativePPTransportTest::abortLocal)
        .def("retire_local", &rtp_llm::test::V41NativePPTransportTest::retireLocal)
        .def_property_readonly("active_local", &rtp_llm::test::V41NativePPTransportTest::activeLocal)
        .def("check_local_config", &rtp_llm::test::V41NativePPTransportTest::checkLocalConfig)
        .def("send_local", &rtp_llm::test::V41NativePPTransportTest::sendLocal)
        .def("receive_local", &rtp_llm::test::V41NativePPTransportTest::receiveLocal)
        .def_property_readonly("cache", &rtp_llm::test::V41NativePPTransportTest::cache)
        .def_property_readonly("free_blocks", &rtp_llm::test::V41NativePPTransportTest::freeBlocks)
        .def_property_readonly("capacity", &rtp_llm::test::V41NativePPTransportTest::capacity)
        .def_property_readonly("pending_tickets", &rtp_llm::test::V41NativePPTransportTest::pendingTickets)
        .def_property_readonly("retained_leases", &rtp_llm::test::V41NativePPTransportTest::retainedLeases)
        .def("send", &rtp_llm::test::V41NativePPTransportTest::send)
        .def("receive", &rtp_llm::test::V41NativePPTransportTest::receive)
        .def("wait", &rtp_llm::test::V41NativePPTransportTest::wait)
        .def("finish", &rtp_llm::test::V41NativePPTransportTest::finish)
        .def("close", &rtp_llm::test::V41NativePPTransportTest::close);
}
