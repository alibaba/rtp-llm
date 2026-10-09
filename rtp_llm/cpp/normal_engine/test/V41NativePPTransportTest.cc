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
        model_config_(std::move(config)), pp_rank_(parallelism.pp_rank) {
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
        executor_.reset();
        manager_->stopMetricsReporting();
        manager_.reset();
    }

private:
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

    ModelConfig                           model_config_;
    int64_t                               pp_rank_;
    int64_t                               generation_ = 0;
    std::shared_ptr<KVCacheManager>         manager_;
    std::unique_ptr<PPExecutor>            executor_;
    PPTickets                             tickets_;
    std::vector<KVCacheAllocationLeasePtr> round_leases_;
};

}  // namespace test
}  // namespace rtp_llm

PYBIND11_MODULE(libth_v41_native_pp_transport_test, m) {
    rtp_llm::registerExecCtxOps(m);
    py::class_<rtp_llm::test::V41PPAllocation, std::shared_ptr<rtp_llm::test::V41PPAllocation>>(m, "Allocation")
        .def_property_readonly("block_tables", &rtp_llm::test::V41PPAllocation::tables)
        .def_property_readonly("identity", &rtp_llm::test::V41PPAllocation::identity)
        .def("release_owner", &rtp_llm::test::V41PPAllocation::releaseOwner)
        .def("release_lease", &rtp_llm::test::V41PPAllocation::releaseLease);
    py::class_<rtp_llm::test::V41NativePPTransportTest>(m, "NativePPTransport")
        .def(py::init<rtp_llm::ModelConfig, const rtp_llm::ParallelismConfig&>())
        .def("allocate", &rtp_llm::test::V41NativePPTransportTest::allocate)
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
