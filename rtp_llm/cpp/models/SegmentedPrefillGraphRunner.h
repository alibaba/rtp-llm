#pragma once

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include <algorithm>
#include <tuple>
#include <vector>
#include <pybind11/stl.h>

namespace rtp_llm {

// Only the explicit segmented backend constructs this adapter. Existing full,
// decode and embedding runners retain their own input and padding contracts.
class SegmentedPrefillGraphRunner final: public GraphBase {
public:
    SegmentedPrefillGraphRunner(const GptModelInitParams& params, const GraphParams& graph_params, py::object model):
        GraphBase(std::move(model)),
        cache_manager_(params.cache_manager),
        buckets_(params.hw_kernel_config.generation_prefill_capture_token_buckets),
        max_requests_(params.hw_kernel_config.generation_prefill_cuda_graph_max_requests),
        position_factor_(graph_params.position_id_len_factor),
        dtype_(graph_params.model_data_type),
        vocab_size_(params.description.vocab_size) {}

    ~SegmentedPrefillGraphRunner() override {
        py::gil_scoped_acquire gil;
        if (dirty_) {
            // Python construction can fail before runner_ is assigned. Its
            // quarantined graph may already reference these allocated pages.
            (void)runner_.release();
            (void)py_instance_.release();
            retainDummyPages();
            return;
        }
        try {
            if (runner_ && !runner_.is_none()) {
                runner_.attr("close")();
                runner_ = py::object();
            }
            if (resource_) {
                cache_manager_->free(FreeInfo{resource_, tokens_, -1});
            }
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("segmented prefill teardown failed; retaining in-flight resources: %s", e.what());
            (void)runner_.release();
            // A failed drain must not return reserved pages to the allocator.
            retainDummyPages();
        }
    }

    void initCapture() override {
        py::gil_scoped_acquire gil;
        RTP_LLM_CHECK_WITH_INFO(cache_manager_ != nullptr, "segmented prefill requires bound cache pools");
        RTP_LLM_CHECK_WITH_INFO(py::hasattr(py_instance_, "create_segmented_prefill_runner"),
                                "segmented prefill requires a model adapter");
        RTP_LLM_CHECK_WITH_INFO(!buckets_.empty(), "segmented prefill requires token buckets");
        const int capacity = *std::max_element(buckets_.begin(), buckets_.end());
        auto      input    = std::make_shared<GenerateInput>();
        input->input_ids   = torch::zeros({capacity}, torch::kInt32);
        tokens_ = std::make_shared<CompleteTokenIds>(1, 1, capacity, cache_manager_->cacheConfig().seq_size_per_block);
        tokens_->init(input);
        resource_ = std::make_shared<BatchKVCacheResource>();
        resource_->resetBatchSize(1);
        resource_->initGroups(cache_manager_->cacheConfig().topologyPtr());
        MallocInfo allocation;
        allocation.batch_kv_cache_resource      = resource_;
        allocation.complete_token_ids           = tokens_;
        allocation.request_id                   = -1;
        allocation.reuse_cache                  = false;
        allocation.enable_cache_lookup          = false;
        allocation.enable_remove_skipped_blocks = false;
        const auto result                       = cache_manager_->malloc(allocation);
        RTP_LLM_CHECK_WITH_INFO(result.success, "segmented prefill could not reserve allocator-owned dummy pages");
        auto service_probe = std::make_shared<BatchKVCacheResource>();
        service_probe->resetBatchSize(1);
        service_probe->initGroups(cache_manager_->cacheConfig().topologyPtr());
        allocation.batch_kv_cache_resource = service_probe;
        const auto service_result          = cache_manager_->malloc(allocation);
        RTP_LLM_CHECK_WITH_INFO(service_result.success,
                                "segmented prefill reserved pages leave insufficient cache for one max-bucket request");
        cache_manager_->free(FreeInfo{service_probe, tokens_, -1});
        dirty_  = true;
        runner_ = py_instance_.attr("create_segmented_prefill_runner")(
            buckets_, max_requests_, py::cpp_function([this](int bucket) { return dummyInputs(bucket); }));
        dirty_ = false;
    }

    bool canRun(const PyModelInputs& inputs,
                CudaGraphState&      state,
                CudaGraphCheckMode   mode = CudaGraphCheckMode::FORWARD) override {
        py::gil_scoped_acquire gil;
        auto       result = runner_.attr("can_run")(inputs, mode == CudaGraphCheckMode::PREPARE).cast<py::tuple>();
        const int  bucket = result[0].cast<int>();
        const auto reason = result[1].cast<std::string>();
        state.current_batch_size        = inputs.attention_inputs.input_lengths.numel();
        state.current_real_graph_bs     = max_requests_;
        state.graph_token_capacity      = bucket;
        state.graph_request_capacity    = max_requests_;
        state.generation_prefill_status = GenerationPrefillCudaGraphStatus::GRAPH_INPUT_SHAPE_MISMATCH;
        for (int i = 0; i <= static_cast<int>(GenerationPrefillCudaGraphStatus::GRAPH_INPUT_SHAPE_MISMATCH); ++i) {
            auto status = static_cast<GenerationPrefillCudaGraphStatus>(i);
            if (reason == generationPrefillCudaGraphStatusString(status)) {
                state.generation_prefill_status = status;
                break;
            }
        }
        return bucket > 0;
    }

    PyModelOutputs forward(const PyModelInputs& inputs, CudaGraphState&) override {
        py::gil_scoped_acquire gil;
        return runner_.attr("forward")(inputs).cast<PyModelOutputs>();
    }
    void waitForPreviousForward() {
        py::gil_scoped_acquire gil;
        runner_.attr("wait_for_previous")();
    }
    // This backend begins its fixed-bank lease in forward, after native H2D
    // flush. The native prepare call only performs read-only admission.
    void prepareAttentionInputs(const PyModelInputs&, CudaGraphState&, bool = false) override {}
    void setPositionEncoding(torch::Tensor) override {}
    void setTokenTypeEmbedding(torch::Tensor) override {}
    void setInputEmbeddingScalar(float) override {}
    bool captureSessionMayBeDirty() const override {
        return dirty_;
    }

private:
    void retainDummyPages() {
        (void)new std::tuple<std::shared_ptr<KVCacheManager>, BatchKVCacheResourcePtr, CompleteTokenIdsPtr>(
            cache_manager_, resource_, tokens_);
    }

    PyModelInputs dummyInputs(int bucket) const {
        const auto    cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
        const auto    gpu = cpu.device(torch::kCUDA);
        PyModelInputs inputs;
        inputs.input_ids = torch::randint(std::max<size_t>(1, vocab_size_), {bucket}, gpu);
        if (position_factor_ > 0) {
            inputs.combo_position_ids = torch::arange(bucket, gpu).unsqueeze(1).repeat({1, position_factor_}).flatten();
        }
        for (const auto& group : cache_manager_->cacheConfig().topology().groups()) {
            PyAttentionInputs attention;
            attention.dtype                           = dtype_;
            attention.is_prefill                      = true;
            attention.input_lengths                   = torch::full({1}, bucket, cpu);
            attention.prefix_lengths                  = torch::zeros({1}, cpu);
            attention.sequence_lengths                = torch::empty({0}, cpu);
            attention.cu_seqlens                      = torch::tensor({0, bucket}, cpu);
            attention.cu_seqlens_device               = attention.cu_seqlens.to(torch::kCUDA);
            attention.cu_kv_seqlens_device            = attention.cu_seqlens_device;
            attention.input_lengths_device            = attention.input_lengths.to(torch::kCUDA);
            attention.prefix_lengths_device           = attention.prefix_lengths.to(torch::kCUDA);
            attention.padding_offset                  = torch::zeros({bucket}, gpu);
            attention.combo_position_ids              = inputs.combo_position_ids;
            attention.context_total_kv_length         = bucket;
            attention.total_tokens                    = bucket;
            const auto& ids                           = resource_->kernelBlocks(0, group.tag);
            attention.kv_cache_kernel_block_id        = torch::tensor(ids, cpu).unsqueeze(0);
            attention.kv_cache_kernel_block_id_device = attention.kv_cache_kernel_block_id.to(torch::kCUDA);
            attention.kv_cache_block_id        = torch::tensor(resource_->blocks(0, group.tag), cpu).unsqueeze(0);
            attention.kv_cache_block_id_device = attention.kv_cache_block_id.to(torch::kCUDA);
            inputs.attention_inputs_by_tag.emplace(group.tag, std::move(attention));
        }
        inputs.attention_inputs = inputs.attention_inputs_by_tag.begin()->second;
        return inputs;
    }

    std::shared_ptr<KVCacheManager> cache_manager_;
    BatchKVCacheResourcePtr         resource_;
    CompleteTokenIdsPtr             tokens_;
    std::vector<int>                buckets_;
    int                             max_requests_;
    int                             position_factor_;
    c10::ScalarType                 dtype_;
    size_t                          vocab_size_;
    py::object                      runner_;
    bool                            dirty_ = false;
};

}  // namespace rtp_llm
