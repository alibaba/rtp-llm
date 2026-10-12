#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheAllocationLease.h"
#include "rtp_llm/cpp/cache/PPTopologyValidator.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/schedulers/PPScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPExecutor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/V41PPLocalCache.h"
#include "rtp_llm/cpp/normal_engine/pipeline/V41PPAdmission.h"
#include "rtp_llm/cpp/models/V41InputRejected.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagProcessor.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/cuda/Bf16GemmOp.h"

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

// A real caller-owned global input, retained across coordinated admission.
// Deliberately permit malformed/foreign inputs to reach the production WORLD
// transaction instead of rejecting them in this test fixture.
class V41PPGlobalInputs {
public:
    using TensorMember = torch::Tensor GptModelInputs::*;

    static const std::map<std::string, TensorMember>& fields() {
        static const std::map<std::string, TensorMember> members{
            {"spans", &GptModelInputs::v41_cache_spans},
            {"ids", &GptModelInputs::combo_tokens},
            {"windows", &GptModelInputs::engram_token_windows},
            {"lengths", &GptModelInputs::input_lengths},
            {"prefixes", &GptModelInputs::prefix_lengths},
            {"decode_lengths", &GptModelInputs::sequence_lengths},
            {"positions", &GptModelInputs::combo_position_ids},
            {"table", &GptModelInputs::kv_cache_kernel_block_id},
            {"physical_table", &GptModelInputs::kv_cache_block_id},
            {"zero_blocks", &GptModelInputs::kv_cache_blocks_to_zero},
            {"copy_mapping", &GptModelInputs::kv_cache_update_mapping},
            {"mask", &GptModelInputs::attention_mask},
            {"lm_indexes", &GptModelInputs::lm_output_indexes},
        };
        return members;
    }

    std::map<std::string, torch::Tensor> tensors() const {
        std::map<std::string, torch::Tensor> result;
        for (const auto& [name, member] : fields()) {
            if ((inputs.*member).defined()) {
                result.emplace(name, inputs.*member);
            }
        }
        return result;
    }

    void setTensor(const std::string& name, const torch::Tensor& value) {
        inputs.*fields().at(name) = value;
    }

    void control(const std::string& name) {
        if (name == "shutdown") {
            inputs.shutdown = true;
        } else if (name == "warmup") {
            inputs.warmup = true;
        } else if (name == "verify") {
            inputs.is_target_verify = true;
        } else if (name == "multimodal") {
            inputs.multimodal_features = std::vector<torch::Tensor>{torch::ones({1})};
        } else if (name == "pp_frame") {
            inputs.pp_intermediates.emplace("hidden_states", torch::ones({1}));
        } else {
            throw std::invalid_argument("unknown global input test control");
        }
    }

    GptModelInputs inputs{};
    V41PPCacheBatchPtr batch;
    py::object dispatch = py::none();
    PreparedPPModelInputsPtr model_dispatch;
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
        scheduler_runtime_ = runtime;
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
        return torch_ext::KVCache(manager_->getMainModelGroupedCacheLayerLayout(),
                                  manager_->allLayerCacheBase().topologyPtr());
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

    std::shared_ptr<V41PPGlobalInputs>
    globalInputs(const std::map<std::string, torch::Tensor>& tensors,
                 const std::vector<std::shared_ptr<V41PPAllocation>>& allocations) {
        requireOpen();
        auto result = std::make_shared<V41PPGlobalInputs>();
        for (const auto& [name, value] : tensors) {
            result->setTensor(name, value);
        }
        for (const auto& allocation : allocations) {
            result->inputs.v41_cache_leases.push_back(allocation ? allocation->retain() : nullptr);
        }
        return result;
    }

    V41PPCacheBatchPtr prepareGlobal(const std::shared_ptr<V41PPGlobalInputs>& global) {
        requireOpen();
        if (!admission_) {
            admission_ = std::make_unique<V41PPAdmission>(manager_, model_config_, parallelism_);
        }
        global->batch = admission_->prepare(global->inputs);
        return global->batch;
    }

    V41PPCacheBatchPtr prepareGlobalWorker(const std::shared_ptr<V41PPGlobalInputs>& global, const py::object& hook) {
        requireOpen();
        if (!admission_) {
            admission_ = std::make_unique<V41PPAdmission>(manager_, model_config_, parallelism_);
        }
        // Keep the descriptor provisional. If another stage rejects, its
        // hidden native leases must also drop before returning to the caller.
        py::object dispatch = py::none();
        V41PPAdmission::WorkerPreflight preflight;
        if (!hook.is_none()) {
            preflight = [&](const GptModelInputs& inputs, const V41PPCacheBatchPtr& batch) {
                try {
                    dispatch = hook(splitInputs(batch, inputs));
                } catch (const py::error_already_set& exception) {
                    const py::object rejected =
                        py::module_::import("rtp_llm.models_py.model_desc.v41_errors").attr("V41InputRejected");
                    if (exception.matches(rejected)) {
                        throw V41InputRejected(exception.what());
                    }
                    throw;
                }
            };
        }
        global->batch = admission_->prepareWithWorker(global->inputs, preflight);
        global->dispatch = std::move(dispatch);
        return global->batch;
    }

    void attachModel(const py::object& worker, const torch::Tensor& head) {
        requireOpen();
        if (executor_->model_) {
            throw std::invalid_argument("native PP fixture already has a model");
        }
        Weights weights;
        weights.layers.resize(40);
        if (pp_rank_ == 1) {
            auto local_head = std::make_shared<DenseWeights>();
            const int64_t width = head.size(0) / 4;
            local_head->kernel = head.narrow(0, parallelism_.tp_rank * width, width).contiguous();
            weights.lm_head = std::move(local_head);
        }
        GptModelInitParams params{
            weights,
            Executor::genModelDescription(model_config_, parallelism_, EPLBConfig{}, MoeConfig{}),
            manager_->getMainModelGroupedCacheLayerLayout()};
        params.parallelism_config = parallelism_;
        params.max_seq_len = model_config_.max_seq_len;
        params.hidden_size = model_config_.hidden_size;
        params.tokens_per_block = params.kernel_tokens_per_block = 256;
        params.cache_manager = manager_;
        params.hc_mult = 4;
        params.hw_kernel_config.enable_cuda_graph = false;
        executor_->model_ = std::make_unique<PyWrappedModel>(
            params, worker, false, false, false, DSparkCallPhase::NONE, false, false);
        foreign_model_ = std::make_unique<PyWrappedModel>(
            params, worker, false, false, false, DSparkCallPhase::NONE, false, false);
    }

    V41PPCacheBatchPtr prepareGlobalModel(const std::shared_ptr<V41PPGlobalInputs>& global, bool full_outputs) {
        requireOpen();
        if (!executor_->model_) {
            throw std::invalid_argument("native PP fixture has no attached model");
        }
        if (!admission_) {
            admission_ = std::make_unique<V41PPAdmission>(manager_, model_config_, parallelism_);
        }
        global->inputs.need_all_logits = global->inputs.need_all_hidden_states = full_outputs;
        PreparedPPModelInputsPtr prepared;
        auto batch = admission_->prepareWithWorker(
            global->inputs, [&](const GptModelInputs& inputs, const V41PPCacheBatchPtr& local) {
                prepared = executor_->model_->prepareV41PP(inputs, local);
            });
        global->batch = batch;
        global->model_dispatch = std::move(prepared);
        return batch;
    }

    py::dict executeGlobalModel(const PreparedPPModelInputsPtr& prepared, int64_t round,
                                 const py::object& incoming, bool foreign) {
        PPIntermediateTensors input;
        PPIntermediateTensors output;
        if (!incoming.is_none()) {
            input.tensors = incoming.cast<std::map<std::string, torch::Tensor>>();
        }
        auto* model = foreign ? foreign_model_.get() : executor_->model_.get();
        auto result = model->forwardV41PP(prepared, round,
                                                       incoming.is_none() ? nullptr : &input,
                                                       pp_rank_ == 0 ? &output : nullptr);
        py::dict values;
        values["payload"] = output.tensors;
        values["hidden"] = result.hidden_states;
        values["all_hidden"] = result.all_hidden_states;
        values["logits"] = result.logits;
        values["all_logits"] = result.all_logits;
        return values;
    }

    void retireGlobalModel(const std::array<int64_t, 3>& identity) {
        executor_->model_->retireV41PP(identity);
        retireGlobal(identity);
    }

    void releaseModelDescriptor(const std::shared_ptr<V41PPGlobalInputs>& global) {
        auto prepared = std::move(global->model_dispatch);
        py::gil_scoped_release release;
        std::thread([owned = std::move(prepared)]() mutable { owned.reset(); }).join();
    }

    void commitGlobal(const V41PPCacheBatchPtr& batch) {
        requireOpen();
        requireAdmission();
        admission_->commit(batch);
    }

    void initializeGlobal(const V41PPCacheBatchPtr& batch) {
        requireOpen();
        requireAdmission();
        admission_->initializeFresh(batch);
    }

    void abortGlobal(const V41PPCacheBatchPtr& batch) {
        requireOpen();
        requireAdmission();
        admission_->abort(batch);
    }

    int64_t beginGlobalExecution(const V41PPCacheBatchPtr& batch) {
        requireOpen();
        requireAdmission();
        return admission_->beginExecution(batch);
    }

    void completeGlobalExecution(const V41PPCacheBatchPtr& batch, bool successful) {
        requireOpen();
        requireAdmission();
        admission_->completeExecution(batch, successful);
    }

    void retireGlobal(const std::array<int64_t, 3>& source) {
        requireOpen();
        requireAdmission();
        admission_->retire(source);
    }

    size_t activeGlobal() const {
        return admission_ ? admission_->activeRequests() : 0;
    }

    bool globalFailed() const {
        return admission_ && admission_->failed();
    }

    torch_ext::PyModelInputs splitGlobalInputs(const std::shared_ptr<V41PPGlobalInputs>& source) {
        requireOpen();
        requireAdmission();
        if (!admission_->ownsBatch(source->batch)) {
            throw std::invalid_argument("native PP global fixture requires its coordinated batch");
        }
        // Split the actual accepted global inputs, preserving their local page
        // table and hidden leases. Do not manufacture an independent fixture.
        GptModelInputs global = source->inputs;
        return splitInputs(source->batch, std::move(global));
    }

    torch_ext::PyModelInputs localInputs(const V41PPCacheBatchPtr& batch, torch_ext::PyModelInputs inputs) {
        requireOpen();
        if (!ownsBatch(batch)) {
            throw std::invalid_argument("native PP local inputs require this owner's admitted batch");
        }
        inputs.v41_cache_spans = torch::empty({static_cast<int64_t>(batch->bindings().size()), 6}, torch::kInt64);
        auto* rows = inputs.v41_cache_spans.data_ptr<int64_t>();
        for (const auto& binding : batch->bindings()) {
            std::copy(binding.span.begin(), binding.span.end(), rows);
            rows += 6;
        }
        inputs.v41_cache_leases = batch->leases();
        return inputs;
    }

    torch_ext::PyModelInputs splitLocalInputs(const V41PPCacheBatchPtr& batch,
                                               const torch::Tensor& ids,
                                               const torch::Tensor& windows) {
        requireOpen();
        if (!local_cache_ || !local_cache_->ownsBatch(batch) || ids.device().type() != torch::kCPU
            || ids.scalar_type() != torch::kInt32 || ids.dim() != 1 || !ids.is_contiguous()
            || windows.device().type() != torch::kCPU || windows.scalar_type() != torch::kInt32
            || windows.sizes().vec() != std::vector<int64_t>{ids.numel(), 4} || !windows.is_contiguous()) {
            throw std::invalid_argument("native PP CP fixture requires its admitted batch and complete host tokens");
        }
        GptModelInputs global;
        global.combo_tokens = ids.clone();
        global.input_lengths = torch::empty({static_cast<int64_t>(batch->bindings().size())}, torch::kInt32);
        global.prefix_lengths = torch::empty_like(global.input_lengths);
        global.sequence_lengths = torch::empty({0}, torch::kInt32);
        global.engram_token_windows = windows.clone();
        int64_t total = 0;
        for (size_t row = 0; row < batch->bindings().size(); ++row) {
            const auto& span = batch->bindings()[row].span;
            global.input_lengths.data_ptr<int32_t>()[row] = span[5];
            global.prefix_lengths.data_ptr<int32_t>()[row] = span[4];
            total += span[5];
        }
        if (total != ids.numel()) {
            throw std::invalid_argument("native PP CP fixture token count disagrees with spans");
        }
        global.kv_cache_kernel_block_id = batch->kernelBlockTable().permute({1, 0, 2}).contiguous().cuda();
        return splitInputs(batch, std::move(global));
    }

    torch_ext::PyModelInputs splitInputs(const V41PPCacheBatchPtr& batch, GptModelInputs global) {
        torch_ext::PyContextParallelParams cp;
        ZigZagProcessor processor(parallelism_, false);
        processor.handleInputs(global, cp);
        torch_ext::PyModelInputs local;
        local.input_ids = global.combo_tokens;
        local.combo_position_ids = global.combo_position_ids;
        local.engram_token_windows = global.engram_token_windows;
        for (const auto& group : manager_->allLayerCacheBase().topology().groups()) {
            torch_ext::PyAttentionInputs attention;
            attention.is_prefill = true;
            attention.input_lengths = global.input_lengths;
            attention.prefix_lengths = global.prefix_lengths;
            attention.sequence_lengths = global.sequence_lengths;
            attention.combo_position_ids = global.combo_position_ids;
            attention.context_parallel_info = cp;
            attention.cu_seqlens = torch::cat({torch::zeros({1}, torch::kInt32),
                                             global.input_lengths.cpu().cumsum(0).to(torch::kInt32)});
            attention.kv_cache_kernel_block_id_device = global.kv_cache_kernel_block_id[group.canonical_idx];
            attention.kv_cache_kernel_block_id = attention.kv_cache_kernel_block_id_device.cpu();
            local.attention_inputs_by_tag.emplace(group.tag, std::move(attention));
        }
        return localInputs(batch, std::move(local));
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

    void startScheduler(int64_t chunk_size) {
        requireOpen();
        if (!executor_->model_ || scheduler_ || chunk_size <= 0 || chunk_size > 256) {
            throw std::invalid_argument("native PP scheduler requires its attached model and a short chunk");
        }
        scheduler_runtime_.max_generate_batch_size = 2;
        auto& fifo = scheduler_runtime_.fifo_scheduler_config;
        fifo.max_context_batch_size = 2;
        fifo.max_batch_tokens_size = 512;
        fifo.cp_force_single_prefill = false;
        fifo.enable_fast_gen = true;
        fifo.fast_gen_max_context_len = chunk_size;
        if (parallelism_.world_rank == 0) {
            PDSepConfig role;
            role.role_type = RoleType::PREFILL;
            scheduler_ = std::make_unique<PPScheduler>(scheduler_runtime_, model_config_, role,
                                                       parallelism_, ModelSpecificConfig{},
                                                       SpeculativeExecutionConfig{}, manager_, nullptr);
        }
    }

    std::vector<int64_t> enqueueScheduled(const std::vector<std::vector<int32_t>>& requests,
                                          bool full_outputs, bool pd_separation) {
        requireOpen();
        if (!scheduler_) {
            throw std::invalid_argument("only the scheduler root can enqueue a test request");
        }
        ResourceContext context;
        context.cache_manager = manager_;
        context.reuse_cache = false;
        std::vector<int64_t> ids;
        for (const auto& tokens : requests) {
            if (tokens.empty() || tokens.size() > 256) {
                throw std::invalid_argument("native PP scheduler test requires short nonempty prompts");
            }
            auto input = std::make_shared<GenerateInput>();
            input->input_ids = torch::tensor(tokens, torch::kInt32);
            input->request_id = ++scheduler_request_id_;
            input->generate_config = std::make_shared<GenerateConfig>();
            input->generate_config->max_new_tokens = 1;
            input->generate_config->top_k = 1;
            input->generate_config->is_streaming = false;
            input->generate_config->return_logits = true;
            input->generate_config->return_hidden_states = true;
            input->generate_config->return_all_hidden_states = full_outputs;
            input->generate_config->pd_separation = pd_separation;
            auto stream = std::make_shared<NormalGenerateStream>(
                input, model_config_, scheduler_runtime_, context, nullptr);
            const auto status = scheduler_->enqueue(stream);
            if (!status.ok()) {
                throw std::runtime_error(status.ToString());
            }
            scheduled_streams_.push_back(stream);
            ids.push_back(input->request_id);
        }
        return ids;
    }

    py::dict schedulerTick() {
        requireOpen();
        ScheduleOutput scheduled;
        if (scheduler_) {
            // Explicit ticks model the engine's schedule trigger without a
            // second host thread competing for the qualified CUDA stream.
            scheduler_->schedule_trigger_ = true;
            auto result = scheduler_->schedule();
            if (!result.ok()) {
                throw std::runtime_error(result.status().ToString());
            }
            scheduled = std::move(result.value());
        }
        const auto status = executor_->process(scheduled);
        if (!status.ok()) {
            throw std::runtime_error(status.ToString());
        }
        cudaSyncAndCheck();
        py::dict result;
        result["active_requests"] = executor_->v41_pp_admission_->activeRequests();
        result["identity_count"] = executor_->v41_request_identities_.size();
        size_t retained = 0;
        for (const auto& slot : executor_->slots_) {
            retained += slot.v41_cache_batch ? 1 : 0;
        }
        result["retained_slots"] = retained;
        result["free_blocks"] = freeBlocks();
        result["shutdown"] = executor_->shutdownCompleted();
        return result;
    }

    py::list scheduledOutputs() {
        py::list result;
        for (auto it = scheduled_streams_.begin(); it != scheduled_streams_.end();) {
            const auto& stream = *it;
            if (!stream->isFinished() && !stream->hasError()) {
                ++it;
                continue;
            }
            py::dict output;
            output["request_id"] = stream->streamId();
            output["outstanding"] = stream->ppOutstandingResults();
            auto next = stream->nextOutput(1);
            if (!next.ok()) {
                output["error"] = next.status().ToString();
                output["invalid_params"] = next.status().code() == ErrorCode::INVALID_PARAMS;
            } else {
                const auto& generated = next.value().generate_outputs.front();
                output["tokens"] = stream->completeTokenIdsVec();
                output["hidden"] = generated.hidden_states;
                output["logits"] = generated.logits;
                output["all_hidden"] = generated.all_hidden_states;
            }
            result.append(std::move(output));
            it = scheduled_streams_.erase(it);
        }
        return result;
    }

    void shutdownScheduler() {
        executor_->notifyShutdown();
    }

    void close() {
        finish();
        scheduler_.reset();
        scheduled_streams_.clear();
        admission_.reset();
        local_cache_.reset();
        executor_.reset();
        foreign_model_.reset();
        manager_->stopMetricsReporting();
        manager_.reset();
    }

private:
    void requireAdmission() const {
        if (!admission_) {
            throw std::invalid_argument("native PP fixture has not entered global admission");
        }
    }

    bool ownsBatch(const V41PPCacheBatchPtr& batch) const {
        return (local_cache_ && local_cache_->ownsBatch(batch)) || (admission_ && admission_->ownsBatch(batch));
    }

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
        if (!ownsBatch(batch) || !tickets_.empty() || !round_leases_.empty()) {
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
    std::unique_ptr<PyWrappedModel>        foreign_model_;
    std::unique_ptr<V41PPLocalCache>       local_cache_;
    std::unique_ptr<V41PPAdmission>        admission_;
    PPTickets                             tickets_;
    std::vector<KVCacheAllocationLeasePtr> round_leases_;
    RuntimeConfig scheduler_runtime_;
    std::unique_ptr<PPScheduler> scheduler_;
    std::vector<GenerateStreamPtr> scheduled_streams_;
    int64_t scheduler_request_id_ = 5000;
};

}  // namespace test
}  // namespace rtp_llm

PYBIND11_MODULE(libth_v41_native_pp_transport_test, m) {
    rtp_llm::registerExecCtxOps(m);
    py::register_exception<rtp_llm::V41InputRejected>(m, "NativeInputRejected", PyExc_RuntimeError);
    py::class_<rtp_llm::PreparedPPModelInputs, rtp_llm::PreparedPPModelInputsPtr>(m, "PreparedModelInputs");
    m.def("native_lm_logits", [](const torch::Tensor& hidden, const torch::Tensor& head) {
        return torch_ext::cublas_gemm_bf16_bf16_fp32(hidden.to(torch::kBFloat16), head);
    });
    py::class_<rtp_llm::test::V41PPGlobalInputs, std::shared_ptr<rtp_llm::test::V41PPGlobalInputs>>(m, "GlobalInputs")
        .def_property_readonly("tensors", &rtp_llm::test::V41PPGlobalInputs::tensors)
        .def_readonly("batch", &rtp_llm::test::V41PPGlobalInputs::batch)
        .def_readonly("dispatch", &rtp_llm::test::V41PPGlobalInputs::dispatch)
        .def_readonly("model_dispatch", &rtp_llm::test::V41PPGlobalInputs::model_dispatch)
        .def("set_tensor", &rtp_llm::test::V41PPGlobalInputs::setTensor)
        .def("control", &rtp_llm::test::V41PPGlobalInputs::control)
        .def_property_readonly("lease_count", [](const rtp_llm::test::V41PPGlobalInputs& global) {
            return global.inputs.v41_cache_leases.size();
        });
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
        .def_property_readonly("initialized", &rtp_llm::V41PPCacheBatch::initialized)
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
        .def("global_inputs", &rtp_llm::test::V41NativePPTransportTest::globalInputs,
             py::arg("tensors"),
             py::arg("root_allocations") = std::vector<std::shared_ptr<rtp_llm::test::V41PPAllocation>>{})
        .def("prepare_global", &rtp_llm::test::V41NativePPTransportTest::prepareGlobal)
        .def("prepare_global_worker", &rtp_llm::test::V41NativePPTransportTest::prepareGlobalWorker)
        .def("attach_model", &rtp_llm::test::V41NativePPTransportTest::attachModel)
        .def("prepare_global_model", &rtp_llm::test::V41NativePPTransportTest::prepareGlobalModel)
        .def("execute_global_model", &rtp_llm::test::V41NativePPTransportTest::executeGlobalModel,
             py::arg("prepared"), py::arg("round"), py::arg("incoming"), py::arg("foreign") = false)
        .def("retire_global_model", &rtp_llm::test::V41NativePPTransportTest::retireGlobalModel)
        .def("release_model_descriptor", &rtp_llm::test::V41NativePPTransportTest::releaseModelDescriptor)
        .def("commit_global", &rtp_llm::test::V41NativePPTransportTest::commitGlobal)
        .def("initialize_global", &rtp_llm::test::V41NativePPTransportTest::initializeGlobal)
        .def("begin_global_execution", &rtp_llm::test::V41NativePPTransportTest::beginGlobalExecution)
        .def("complete_global_execution", &rtp_llm::test::V41NativePPTransportTest::completeGlobalExecution)
        .def("abort_global", &rtp_llm::test::V41NativePPTransportTest::abortGlobal)
        .def("retire_global", &rtp_llm::test::V41NativePPTransportTest::retireGlobal)
        .def_property_readonly("active_global", &rtp_llm::test::V41NativePPTransportTest::activeGlobal)
        .def_property_readonly("global_failed", &rtp_llm::test::V41NativePPTransportTest::globalFailed)
        .def("split_global_inputs", &rtp_llm::test::V41NativePPTransportTest::splitGlobalInputs)
        .def("local_inputs", &rtp_llm::test::V41NativePPTransportTest::localInputs)
        .def("split_local_inputs", &rtp_llm::test::V41NativePPTransportTest::splitLocalInputs)
        .def("send_local", &rtp_llm::test::V41NativePPTransportTest::sendLocal)
        .def("receive_local", &rtp_llm::test::V41NativePPTransportTest::receiveLocal)
        .def_property_readonly("cache", &rtp_llm::test::V41NativePPTransportTest::cache)
        .def_property_readonly("free_blocks", &rtp_llm::test::V41NativePPTransportTest::freeBlocks)
        .def_property_readonly("capacity", &rtp_llm::test::V41NativePPTransportTest::capacity)
        .def_property_readonly("pending_tickets", &rtp_llm::test::V41NativePPTransportTest::pendingTickets)
        .def_property_readonly("retained_leases", &rtp_llm::test::V41NativePPTransportTest::retainedLeases)
        .def("start_scheduler", &rtp_llm::test::V41NativePPTransportTest::startScheduler)
        .def("enqueue_scheduled", &rtp_llm::test::V41NativePPTransportTest::enqueueScheduled,
             py::arg("requests"), py::arg("full_outputs"), py::arg("pd_separation") = false)
        .def("scheduler_tick", &rtp_llm::test::V41NativePPTransportTest::schedulerTick)
        .def("scheduled_outputs", &rtp_llm::test::V41NativePPTransportTest::scheduledOutputs)
        .def("shutdown_scheduler", &rtp_llm::test::V41NativePPTransportTest::shutdownScheduler)
        .def("send", &rtp_llm::test::V41NativePPTransportTest::send)
        .def("receive", &rtp_llm::test::V41NativePPTransportTest::receive)
        .def("wait", &rtp_llm::test::V41NativePPTransportTest::wait)
        .def("finish", &rtp_llm::test::V41NativePPTransportTest::finish)
        .def("close", &rtp_llm::test::V41NativePPTransportTest::close);
}
