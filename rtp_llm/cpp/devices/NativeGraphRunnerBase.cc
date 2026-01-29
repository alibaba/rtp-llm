#include "rtp_llm/cpp/devices/NativeGraphRunnerBase.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"

#define CUDAGRAPH_LOG RTP_LLM_LOG_DEBUG

namespace rtp_llm {

template<typename Input, typename Output>
std::optional<typename CudaGraphExecutorCache<Input, Output>::GraphItem>
CudaGraphExecutorCache<Input, Output>::get(BatchState const& state) {
    auto it = mMap.find(state);
    if (it == mMap.end()) {
        return std::nullopt;
    }
    mCache.splice(mCache.begin(), mCache, it->second);
    return *(it->second);
}

template<typename Input, typename Output>
void CudaGraphExecutorCache<Input, Output>::put(BatchState const&                                                state,
                                                const typename CudaGraphExecutorCache<Input, Output>::GraphItem& pair) {
    auto it = mMap.find(state);
    if (it != mMap.end()) {
        mCache.erase(it->second);
    }
    mCache.emplace_front(pair);
    mMap[state] = mCache.begin();

    if (static_cast<size_t>(mMap.size()) > mCapacity) {
        auto lastState = std::get<0>(mCache.back());
        mCache.pop_back();
        mMap.erase(lastState);
    }
}

template<typename Input, typename Output>
NativeGraphRunnerBase<Input, Output>::NativeGraphRunnerBase(DeviceBase* device): device_(device) {
    CUDAGRAPH_LOG("map cap: %d", device_->initParams().hw_kernel_config.num_native_cuda_graph);
    map_ = std::make_unique<CudaGraphExecutorCache<Input, Output>>(
        device_->initParams().hw_kernel_config.num_native_cuda_graph);
}

template<>
GptModelInputs NativeGraphRunnerBase<GptModelInputs, GptModelOutputs>::prepareInputBuffer(const GptModelInputs& old) {
    RTP_LLM_CHECK_WITH_INFO(old.lora_model_input == nullptr, "Native graph with lora_model_input not supported");
    RTP_LLM_CHECK_WITH_INFO(old.multimodal_features == std::nullopt,
                            "Native graph with multimodal_features not supported");
    RTP_LLM_CHECK_WITH_INFO(old.input_embeddings == std::nullopt && old.input_embeddings_locs == nullptr,
                            "Native graph with input_embeddings not supported");

    auto input             = old;
    input.combo_tokens     = device_->allocateBufferLike(*old.combo_tokens, AllocationType::HOST);
    input.input_lengths    = device_->allocateBufferLike(*old.input_lengths, AllocationType::HOST);
    input.sequence_lengths = device_->allocateBufferLike(*old.sequence_lengths, AllocationType::HOST);
    input.lm_output_indexes =
        old.lm_output_indexes ? device_->allocateBufferLike(*old.lm_output_indexes, AllocationType::HOST) : nullptr;
    input.lm_output_lengths =
        old.lm_output_lengths ? device_->allocateBufferLike(*old.lm_output_lengths, AllocationType::HOST) : nullptr;
    input.prefix_lengths =
        old.prefix_lengths ? device_->allocateBufferLike(*old.prefix_lengths, AllocationType::HOST) : nullptr;
    input.combo_tokens_type_ids = old.combo_tokens_type_ids ?
                                      device_->allocateBufferLike(*old.combo_tokens_type_ids, AllocationType::HOST) :
                                      nullptr;
    input.combo_position_ids =
        old.combo_position_ids ? device_->allocateBufferLike(*old.combo_position_ids, AllocationType::HOST) : nullptr;
    input.last_hidden_states =
        old.last_hidden_states ? device_->allocateBufferLike(*old.last_hidden_states, AllocationType::HOST) : nullptr;
    input.lora_ids = old.lora_ids ? device_->allocateBufferLike(*old.lora_ids, AllocationType::HOST) : nullptr;
    input.lora_input_lengths =
        old.lora_input_lengths ? device_->allocateBufferLike(*old.lora_input_lengths, AllocationType::HOST) : nullptr;
    input.attention_mask =
        old.attention_mask ? device_->allocateBufferLike(*old.attention_mask, AllocationType::HOST) : nullptr;
    input.kv_cache_block_id =
        old.kv_cache_block_id ? (old.kv_cache_block_id->shape().size() == 3 ?
                                     device_->allocateBuffer({DataType::TYPE_INT32,
                                                              {old.kv_cache_block_id->shape()[0],
                                                               old.kv_cache_block_id->shape()[1],
                                                               static_cast<size_t>(device_->initParams().max_seq_len)},
                                                              AllocationType::HOST}) :
                                     device_->allocateBuffer({DataType::TYPE_INT32,
                                                              {old.kv_cache_block_id->shape()[0],
                                                               static_cast<size_t>(device_->initParams().max_seq_len)},
                                                              AllocationType::HOST})) :
                                nullptr;
    input.kv_cache_layer_to_group =
        old.kv_cache_layer_to_group ? device_->allocateBufferLike(*old.kv_cache_layer_to_group, AllocationType::HOST) :
                                      nullptr;
    input.kv_cache_group_types    = old.kv_cache_group_types ?
                                        device_->allocateBufferLike(*old.kv_cache_group_types, AllocationType::HOST) :
                                        nullptr;
    input.kv_cache_update_mapping = nullptr;
    input.multimodal_features     = std::nullopt;
    input.text_tokens_mask =
        old.text_tokens_mask ? device_->allocateBufferLike(*old.text_tokens_mask, AllocationType::HOST) : nullptr;
    input.mm_features_locs =
        old.mm_features_locs ? device_->allocateBufferLike(*old.mm_features_locs, AllocationType::HOST) : nullptr;
    input.input_embeddings      = std::nullopt;
    input.input_embeddings_locs = nullptr;
    input.request_id = old.request_id ? device_->allocateBufferLike(*old.request_id, AllocationType::HOST) : nullptr;
    input.request_pd_separation = old.request_pd_separation ?
                                      device_->allocateBufferLike(*old.request_pd_separation, AllocationType::HOST) :
                                      nullptr;
    input.cache_keys = old.cache_keys ? device_->allocateBufferLike(*old.cache_keys, AllocationType::HOST) : nullptr;
    return input;
}

template<>
void NativeGraphRunnerBase<GptModelInputs, GptModelOutputs>::copy(GptModelInputs* dst, const GptModelInputs& src) {
    device_->copy({*dst->combo_tokens, *src.combo_tokens});
    device_->copy({*dst->input_lengths, *src.input_lengths});
    device_->copy({*dst->sequence_lengths, *src.sequence_lengths});
    if (src.lm_output_indexes)
        device_->copy({*dst->lm_output_indexes, *src.lm_output_indexes});
    if (src.lm_output_lengths)
        device_->copy({*dst->lm_output_lengths, *src.lm_output_lengths});
    if (src.prefix_lengths)
        device_->copy({*dst->prefix_lengths, *src.prefix_lengths});
    if (src.combo_tokens_type_ids)
        device_->copy({*dst->combo_tokens_type_ids, *src.combo_tokens_type_ids});
    if (src.combo_position_ids)
        device_->copy({*dst->combo_position_ids, *src.combo_position_ids});
    if (src.last_hidden_states)
        device_->copy({*dst->last_hidden_states, *src.last_hidden_states});
    if (src.lora_ids)
        device_->copy({*dst->lora_ids, *src.lora_ids});
    if (src.lora_input_lengths)
        device_->copy({*dst->lora_input_lengths, *src.lora_input_lengths});
    if (src.attention_mask)
        device_->copy({*dst->attention_mask, *src.attention_mask});

    if (src.kv_cache_block_id) {
        const auto& shape = src.kv_cache_block_id->shape();
        if (shape.size() == 2) {
            const int bs = static_cast<int>(shape[0]);
            for (int i = 0; i < bs; i++) {
                std::memcpy(dst->kv_cache_block_id->index(i)->data(),
                            src.kv_cache_block_id->index(i)->data(),
                            src.kv_cache_block_id->index(i)->sizeBytes());
            }
        } else if (shape.size() == 3) {
            // Hybrid layout: [group, batch, max_blocks]
            const int group = static_cast<int>(shape[0]);
            for (int g = 0; g < group; ++g) {
                std::memcpy(dst->kv_cache_block_id->index(g)->data(),
                            src.kv_cache_block_id->index(g)->data(),
                            src.kv_cache_block_id->index(g)->sizeBytes());
            }
        }
    }

    if (src.kv_cache_layer_to_group)
        device_->copy({*dst->kv_cache_layer_to_group, *src.kv_cache_layer_to_group});
    if (src.kv_cache_group_types)
        device_->copy({*dst->kv_cache_group_types, *src.kv_cache_group_types});

    if (src.text_tokens_mask)
        device_->copy({*dst->text_tokens_mask, *src.text_tokens_mask});
    if (src.mm_features_locs)
        device_->copy({*dst->mm_features_locs, *src.mm_features_locs});
    if (src.request_id)
        device_->copy({*dst->request_id, *src.request_id});
    if (src.request_pd_separation)
        device_->copy({*dst->request_pd_separation, *src.request_pd_separation});
    if (src.cache_keys)
        device_->copy({*dst->cache_keys, *src.cache_keys});
}

template<typename Input, typename Output>
Output NativeGraphRunnerBase<Input, Output>::run(size_t                       prefill_bs,
                                                 size_t                       decode_bs,
                                                 Input                        input,
                                                 std::function<Output(Input)> forward) {
    bool       hasCudaGraph = false;
    bool       doCapture    = false;
    BatchState state        = {prefill_bs, decode_bs, 0, 0};
    CUDAGRAPH_LOG("state: %s", state.debugString().c_str());
    if (!state.hasPrefill()) {
        CUDAGRAPH_LOG("This IS decode, maybe graph already captured ...");
        auto cudaGraphOpt = map_->get(state);
        if (cudaGraphOpt.has_value()) {
            CUDAGRAPH_LOG("Found it !");
            hasCudaGraph = true;
        } else {
            CUDAGRAPH_LOG("Not found");
            doCapture = true;
        }
    }

    if (hasCudaGraph) {
        auto [_, cudagraph, input_buffer, output_buffer] = *map_->get(state);
        copy(&input_buffer, input);
        device_->syncAndCheck();
        cudagraph->replay();
        device_->syncAndCheck();
        CUDAGRAPH_LOG("cudagraph launch done");
        return output_buffer;
    }

    auto output = forward(input);

    if (doCapture) {
        CUDAGRAPH_LOG("Capturing graph for %s", state.debugString().c_str());
        auto cudaGraph    = makeExecutor();
        auto input_buffer = prepareInputBuffer(input);
        copy(&input_buffer, input);
        Output output_buffer;

        {
            cudaGraph->captureBegin();

            auto out      = forward(input_buffer);
            output_buffer = out;

            cudaGraph->captureEnd();
        }

        map_->put(state, {state, cudaGraph, input_buffer, output_buffer});
        CUDAGRAPH_LOG(map_->debugString("capture done"));
    }
    return output;
}

INSTANTIATE_NATIVE_GRAPH_RUNNER_BASE(GptModelInputs, GptModelOutputs)

}  // namespace rtp_llm