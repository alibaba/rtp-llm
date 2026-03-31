#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <limits>
#include <utility>
#include "c10/core/DeviceType.h"
#include "c10/core/ScalarType.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"
#if USING_CUDA
#include "rtp_llm/cpp/cuda/ops/StandaloneOps.h"
#include "ATen/cuda/CUDAContext.h"
#endif

using namespace std;

namespace rtp_llm {

absl::StatusOr<GptModelInputs> NormalBatchStreamProcessor::gatherModelInput(const StreamGroups& stream_groups) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    auto context_streams = stream_groups.contextStreams();
    auto decode_streams  = stream_groups.decodeStreams();
    RTP_LLM_LOG_DEBUG(
        "context_streams size = %d, decode_streams size = %d", context_streams.size(), decode_streams.size());
    GptModelInputs model_input;
    const size_t   current_tokens_size      = stream_groups.modelExecuteTokenSize();
    const size_t   total_batch_size         = stream_groups.totalModelBatchSize();
    const size_t   total_decode_batch_size  = stream_groups.totalDecodeBatchSize();
    const size_t   total_context_batch_size = stream_groups.totalContextBatchSize();
    const size_t   total_block_copy_num     = stream_groups.totalBlockUpdateCopyNum();
    const size_t   max_blocks_num           = stream_groups.curBlocksNum();
    const size_t   multimodal_features_len  = stream_groups.mmFeaturesLen();

    const bool has_multimodal_input = is_multimodal_ && stream_groups.has_multimodal_input();
    const bool need_cal_position_id = (mm_position_ids_style_ != PositionIdsStyle::DEFAULT) || has_positional_encoding_;

    size_t num_layers = 0;
    if (model_input.kv_cache_layer_to_group.defined()) {
        num_layers = model_input.kv_cache_layer_to_group.numel();
    } else {
        num_layers = layer_to_kv_cache_group_id_.size();
    }

    // Use pinned_memory(true) in TensorOptions to leverage PyTorch's CachingHostAllocator,
    // which reuses pinned memory blocks across calls instead of cudaHostAlloc/Free each time.
    static const auto pinned_i32  = torch::TensorOptions(torch::kInt32).pinned_memory(true);
    static const auto pinned_i64  = torch::TensorOptions(torch::kInt64).pinned_memory(true);
    static const auto pinned_bool = torch::TensorOptions(torch::kBool).pinned_memory(true);

    model_input.combo_tokens = torch::empty({(int64_t)current_tokens_size}, pinned_i32);
    if (max_blocks_num) {
        model_input.kv_cache_kernel_block_id = torch::zeros({(int64_t)kv_cache_group_nums_,
                                                             (int64_t)total_batch_size,
                                                             (int64_t)(max_blocks_num * kernel_blocks_per_kv_block_)},
                                                            pinned_i32);
        model_input.kv_cache_block_id        = torch::zeros(
            {(int64_t)kv_cache_group_nums_, (int64_t)total_batch_size, (int64_t)max_blocks_num}, pinned_i32);
        model_input.kv_cache_layer_to_group = torch::empty({(int64_t)num_layers_}, pinned_i32);
        model_input.kv_cache_group_types    = torch::empty({(int64_t)kv_cache_group_nums_}, pinned_i32);
        model_input.kv_cache_update_mapping = torch::empty({(int64_t)total_block_copy_num, 2}, pinned_i32);
        model_input.cache_keys = torch::empty({(int64_t)total_context_batch_size, (int64_t)max_blocks_num}, pinned_i64);
    }
    model_input.request_id            = torch::empty({(int64_t)total_context_batch_size}, pinned_i64);
    model_input.request_pd_separation = torch::empty({(int64_t)total_context_batch_size}, pinned_bool);
    model_input.input_lengths         = torch::empty({(int64_t)total_batch_size}, pinned_i32);
    model_input.sequence_lengths      = torch::empty({(int64_t)total_decode_batch_size}, pinned_i32);
    model_input.lm_output_indexes     = torch::empty({(int64_t)total_batch_size}, pinned_i32);
    model_input.lm_output_lengths     = torch::empty({(int64_t)total_batch_size}, pinned_i32);
    model_input.prefix_lengths        = torch::empty({(int64_t)total_context_batch_size}, pinned_i32);
    if (need_cal_position_id) {
        model_input.combo_position_ids =
            torch::empty({(int64_t)(current_tokens_size * position_id_len_factor_)}, pinned_i32);
    }
    if (has_multimodal_input) {
        model_input.text_tokens_mask = torch::empty({(int64_t)current_tokens_size}, pinned_i32);
        model_input.mm_features_locs = torch::empty({(int64_t)multimodal_features_len}, pinned_i32);
    }
    model_input.kv_block_stride_bytes     = block_stride_bytes_;
    model_input.kv_scale_stride_bytes     = scale_stride_bytes_;
    model_input.seq_size_per_block        = seq_size_per_block_;
    model_input.kernel_seq_size_per_block = kernel_seq_size_per_block_;
    model_input.pd_separation             = role_type_ == RoleType::PREFILL;
    model_input.warmup                    = warm_up_;
    model_input.decode_entrance           = decode_entrance_;
    model_input.is_fake_stream            = stream_groups.isFakeStream();

    int* merged_tokens      = model_input.combo_tokens.data_ptr<int32_t>();
    int* input_lengths      = model_input.input_lengths.data_ptr<int32_t>();
    int* sequence_lengths   = model_input.sequence_lengths.data_ptr<int32_t>();
    int* lm_output_indexes  = model_input.lm_output_indexes.data_ptr<int32_t>();
    int* lm_output_lengths  = model_input.lm_output_lengths.data_ptr<int32_t>();
    int* prefix_lengths     = model_input.prefix_lengths.data_ptr<int32_t>();
    int* combo_position_ids = need_cal_position_id ? model_input.combo_position_ids.data_ptr<int32_t>() : nullptr;
    int* merged_text_mask   = has_multimodal_input ? model_input.text_tokens_mask.data_ptr<int32_t>() : nullptr;
    int* mm_features_locs   = has_multimodal_input ? model_input.mm_features_locs.data_ptr<int32_t>() : nullptr;
    int  batch_idx          = 0;
    int  input_vocab_size   = input_vocab_size_ ? input_vocab_size_ : vocab_size_;

    if (model_input.kv_cache_layer_to_group.defined()) {
        std::memcpy(model_input.kv_cache_layer_to_group.data_ptr(),
                    layer_to_kv_cache_group_id_.data(),
                    static_cast<size_t>(num_layers) * sizeof(int32_t));
    }

    if (model_input.kv_cache_group_types.defined()) {
        auto* dst = model_input.kv_cache_group_types.data_ptr<int32_t>();
        for (size_t g = 0; g < kv_cache_group_nums_; ++g) {
            dst[g] = static_cast<int32_t>(kv_cache_group_types_[g]);
        }
    }

    auto*      kv_cache_update_mapping = model_input.kv_cache_update_mapping.defined() ?
                                             (BlockIdPair*)model_input.kv_cache_update_mapping.data_ptr() :
                                             nullptr;
    const auto add_cache_update_copy   = [&](const auto& update_mapping) {
        size_t update_copy_num = update_mapping.size();
        std::memcpy(kv_cache_update_mapping, update_mapping.data(), update_copy_num * sizeof(BlockIdPair));
        kv_cache_update_mapping += update_copy_num;
    };

    if (merged_text_mask) {
        std::fill(merged_text_mask, merged_text_mask + current_tokens_size, 1);
    }

    for (const auto& stream : decode_streams) {
        model_input.need_all_logits = model_input.need_all_logits || stream->calculateLoss();
        auto current_batch_size     = stream->currentBatchSize();

        auto& kv_cache = *stream->kvCachePtr();
        RTP_LLM_LOG_DEBUG("decode kv_cache: %s", kv_cache.debugString().c_str());
        RTP_LLM_LOG_DEBUG("decode stream: %s", stream->debugString().c_str());

        for (auto i = 0; i < current_batch_size; ++i) {
            model_input.trace_ids.push_back(stream->traceId());

            auto currentTokens = stream->currentExecuteTokens(i);
            if (currentTokens[0] >= input_vocab_size) {
                std::ostringstream error_msg;
                error_msg << "stream [" << stream->streamId() << "] token_id " << currentTokens[0]
                          << " exceed vocab_size " << input_vocab_size;
                return absl::InvalidArgumentError(error_msg.str());
            }
            merged_tokens[batch_idx]    = currentTokens[0];
            input_lengths[batch_idx]    = stream->inputLength();
            sequence_lengths[batch_idx] = stream->seqLength() - 1;  // need remove
            if (need_cal_position_id) {
                stream->generateNextPositionId(combo_position_ids + batch_idx * position_id_len_factor_);
            }
            lm_output_indexes[batch_idx] = batch_idx;
            lm_output_lengths[batch_idx] = 1;
            if (max_blocks_num) {
                RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_kernel_block_id.dim() == 3,
                                        "hybrid kv_cache_kernel_block_id must be 3-D");
                RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_block_id.dim() == 3,
                                        "hybrid kv_cache_block_id must be 3-D");
                const size_t batch           = model_input.kv_cache_kernel_block_id.size(1);
                int32_t*     kernel_dst_base = model_input.kv_cache_kernel_block_id.data_ptr<int32_t>();
                int32_t*     store_dst_base  = model_input.kv_cache_block_id.data_ptr<int32_t>();
                for (int gid = 0; gid < kv_cache.groupNums(); ++gid) {
                    auto&    kernel_blocks = kv_cache.kernelBlocks(i, gid);
                    int32_t* kernel_dst    = kernel_dst_base
                                          + (static_cast<size_t>(gid) * batch + static_cast<size_t>(batch_idx))
                                                * max_blocks_num * kernel_blocks_per_kv_block_;
                    std::memcpy(kernel_dst, kernel_blocks.data(), kernel_blocks.size() * sizeof(int32_t));

                    auto&    physical_blocks = kv_cache.blocks(i, gid);
                    int32_t* store_dst =
                        store_dst_base
                        + (static_cast<size_t>(gid) * batch + static_cast<size_t>(batch_idx)) * max_blocks_num;
                    std::memcpy(store_dst, physical_blocks.data(), physical_blocks.size() * sizeof(int32_t));
                }
            }
            batch_idx += 1;
        }

        if (max_blocks_num) {
            add_cache_update_copy(stream->streamCacheResource().getKVBlockUpdateMapping());
        }

        stream->step();
    }

    std::vector<torch::Tensor> gathered_mm_features;
    int                        token_idx          = batch_idx;
    int                        cum_output_seq_len = batch_idx;
    int                        mm_feature_index   = 0;

    for (const auto& stream : context_streams) {
        // context stream也需要batch运行是为了perf test的场景
        model_input.need_all_logits = model_input.need_all_logits || stream->calculateLoss();
        auto current_batch_size     = stream->currentBatchSize();

        auto& kv_cache = *stream->kvCachePtr();
        if (enable_detail_log_) {
            RTP_LLM_LOG_DEBUG("context kv_cache: %s", kv_cache.debugString().c_str());
            RTP_LLM_LOG_DEBUG("context stream: %s", stream->debugString().c_str());
        } else {
            RTP_LLM_LOG_TRACE("context kv_cache: %s", kv_cache.debugString().c_str());
            RTP_LLM_LOG_TRACE("context stream: %s", stream->debugString().c_str());
        }

        // TODO(xinfei.sxf) deal with adjusted common seq len.
        for (auto i = 0; i < current_batch_size; ++i) {
            model_input.trace_ids.push_back(stream->traceId());

            auto input_tokens = stream->currentExecuteTokens(i);
            auto input_masks  = stream->textTokensMask();
            memcpy(merged_tokens + token_idx, input_tokens.data(), input_tokens.size() * sizeof(int));
            cum_output_seq_len += input_tokens.size();

            for (int index = 0; index < input_tokens.size(); ++index) {
                if (input_tokens[index] >= input_vocab_size && (index >= input_masks.size() || input_masks[index])) {
                    std::ostringstream error_msg;
                    error_msg << "stream [" << stream->streamId() << "] token_id " << input_tokens[index]
                              << " exceed vocab_size " << input_vocab_size;
                    return absl::InvalidArgumentError(error_msg.str());
                }
            }

            input_lengths[batch_idx]                            = input_tokens.size();
            prefix_lengths[batch_idx - total_decode_batch_size] = stream->prefixLength();
            lm_output_indexes[batch_idx]                        = cum_output_seq_len - 1;
            lm_output_lengths[batch_idx]                        = 1;

            if (has_multimodal_input) {
                std::vector<torch::Tensor> mm_features = stream->multimodalFeatures();
                torch::Tensor              mm_locs     = stream->multimodalLocations();
                if (mm_locs.defined()) {
                    auto* mm_locs_data = mm_locs.data_ptr<int>();
                    for (int i = 0; i < mm_locs.numel(); ++i) {
                        mm_features_locs[mm_feature_index] = mm_locs_data[i] + token_idx - stream->reuseLength();
                        mm_feature_index++;
                    }
                    for (auto& mm_feature : mm_features) {
                        if (!mm_feature.is_cuda()) {
                            gathered_mm_features.emplace_back(mm_feature.to(torch::kCUDA));
                        } else {
                            gathered_mm_features.emplace_back(mm_feature);
                        }
                    }
                    auto text_token_mask = stream->textTokensMask();
                    memcpy(merged_text_mask + token_idx, text_token_mask.data(), text_token_mask.size() * sizeof(int));
                }
            }

            if (need_cal_position_id) {
                auto context_pos_ids = stream->generateContextPositionIds();
                int  reuse_offset    = stream->reuseLength() * position_id_len_factor_;
                memcpy(combo_position_ids + token_idx * position_id_len_factor_,
                       context_pos_ids.data_ptr<int>() + reuse_offset,
                       (context_pos_ids.numel() - reuse_offset) * sizeof(int));
            }
            if (max_blocks_num) {
                RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_kernel_block_id.dim() == 3,
                                        "hybrid kv_cache_kernel_block_id must be 3-D");
                RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_block_id.dim() == 3,
                                        "hybrid kv_cache_block_id must be 3-D");
                const size_t batch           = model_input.kv_cache_kernel_block_id.size(1);
                int32_t*     kernel_dst_base = model_input.kv_cache_kernel_block_id.data_ptr<int32_t>();
                int32_t*     store_dst_base  = model_input.kv_cache_block_id.data_ptr<int32_t>();
                for (int gid = 0; gid < kv_cache.groupNums(); ++gid) {
                    auto&    kernel_blocks = kv_cache.kernelBlocks(i, gid);
                    int32_t* kernel_dst    = kernel_dst_base
                                          + (static_cast<size_t>(gid) * batch + static_cast<size_t>(batch_idx))
                                                * max_blocks_num * kernel_blocks_per_kv_block_;
                    std::memcpy(kernel_dst, kernel_blocks.data(), kernel_blocks.size() * sizeof(int32_t));

                    auto&    physical_blocks = kv_cache.blocks(i, gid);
                    int32_t* store_dst =
                        store_dst_base
                        + (static_cast<size_t>(gid) * batch + static_cast<size_t>(batch_idx)) * max_blocks_num;
                    std::memcpy(store_dst, physical_blocks.data(), physical_blocks.size() * sizeof(int32_t));
                }
                if (role_type_ == RoleType::PREFILL && stream->hasCacheKeys()) {
                    std::memcpy(model_input.cache_keys.data_ptr<int64_t>()
                                    + (batch_idx - total_decode_batch_size) * model_input.cache_keys.size(1),
                                stream->cacheKeys(i).data(),
                                stream->cacheKeys(i).size() * sizeof(int64_t));
                }
            }
            *(model_input.request_id.data_ptr<int64_t>() + (batch_idx - total_decode_batch_size)) = stream->streamId();
            *(reinterpret_cast<bool*>(model_input.request_pd_separation.data_ptr())
              + (batch_idx - total_decode_batch_size)) = stream->queryPdSep();
            batch_idx += 1;
            token_idx += input_tokens.size();
        }

        if (max_blocks_num) {
            add_cache_update_copy(stream->streamCacheResource().getKVBlockUpdateMapping());
        }

        stream->step();
    }

    if (is_multimodal_ && gathered_mm_features.size() > 0) {
        model_input.multimodal_features = std::move(gathered_mm_features);
    }
    return model_input;
}

absl::StatusOr<SamplerInputs> NormalBatchStreamProcessor::gatherSamplerInput(
    const StreamGroups& stream_groups, const GptModelInputs& model_inputs, const GptModelOutputs& model_output) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    RTP_LLM_CHECK(!stream_groups.empty());
    auto all_streams          = stream_groups.allStreams();
    auto total_batch_size_in  = stream_groups.totalSamplerBatchSizeIn();
    auto total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    bool return_all_probs     = stream_groups.needReturnAllProbs();

    SamplerInputs sampler_inputs =
        allocateSamplerInputs(stream_groups, total_batch_size_in, total_batch_size_out, model_inputs.sequence_lengths);
    setCommonSamplerInputs(sampler_inputs, all_streams);

    setLogitsProcessorInputs(sampler_inputs, all_streams);

    size_t total_decode_batch_size_in = 0;
    int    batch_idx                  = 0;
    bool   return_logits              = false;
    bool   calculate_softmax_probs    = false;
    bool   need_tiling                = false;
    for (auto& stream : all_streams) {
        auto complete_token_ids = stream->completeTokenIds();
        auto complete_seq_len   = complete_token_ids.size(1);
        auto seq_len            = stream->seqLength();
        auto current_batch_size = stream->currentBatchSize();
        auto sampler_batch_size =
            stream->needTilingForSampling() ? stream->nextBatchSize() : stream->currentBatchSize();

        for (int i = 0; i < sampler_batch_size; ++i) {
            int cur_batch = std::min(i, current_batch_size - 1);
            memcpy(sampler_inputs.token_ids.data_ptr<int32_t>() + ((batch_idx) * (sampler_inputs.step + 1)),
                   complete_token_ids.data_ptr<int32_t>() + cur_batch * complete_seq_len,
                   seq_len * sizeof(int));
            reinterpret_cast<bool*>(sampler_inputs.finished_mask.data_ptr())[batch_idx] = stream->isDoneWithoutLock(i);
            batch_idx += 1;
        }
        need_tiling |= stream->needTilingForSampling();
        if (!stream->isContextStream()) {
            total_decode_batch_size_in += sampler_batch_size;
        }
        return_logits |= stream->returnLogits();
        calculate_softmax_probs |= stream->calculateSoftmaxProbs();
        RTP_LLM_LOG_DEBUG("stream [%ld], sampler inputs token ids = [%s]",
                          stream->streamId(),
                          tensorDebugStringWithData<int32_t>(sampler_inputs.token_ids).c_str());
    }

    auto vocab_size           = (size_t)model_output.logits.size(1);
    sampler_inputs.vocab_size = vocab_size;
    if (return_all_probs) {
        sampler_inputs.all_probs = torch::zeros({(int64_t)total_batch_size_in, (int64_t)vocab_size},
                                                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    }

    // copy logits when needs tiling or returning logits
    torch::Tensor logits_tensor;
    if (need_tiling) {
        logits_tensor =
            torch::empty({(int64_t)total_batch_size_in, (int64_t)vocab_size}, model_output.logits.options());
        // copy decode batch logits
        if (total_decode_batch_size_in > 0) {
            logits_tensor.narrow(0, 0, total_decode_batch_size_in)
                .copy_(model_output.logits.narrow(0, 0, total_decode_batch_size_in));
        }
        // tile context batch logits
        size_t input_offset = total_decode_batch_size_in, logits_offset = total_decode_batch_size_in;
        for (auto& stream : stream_groups.contextStreams()) {
            auto sampler_batch_size =
                stream->needTilingForSampling() ? stream->nextBatchSize() : stream->currentBatchSize();
            for (int i = 0; i < sampler_batch_size; ++i) {
                logits_tensor[input_offset].copy_(model_output.logits[logits_offset]);
                input_offset += 1;
            }
            logits_offset += 1;
        }
    } else if (return_logits || calculate_softmax_probs) {
        logits_tensor = model_output.logits.clone();
    } else {
        logits_tensor = model_output.logits;
    }
    sampler_inputs.logits = logits_tensor;

    RTP_LLM_LOG_DEBUG("sampler inputs logits [%s]",
                      tensorDebugStringWithData<float>(sampler_inputs.logits.cpu(), 10).c_str());

    RTP_LLM_LOG_DEBUG("gatherSamplerInput done");
    return std::move(sampler_inputs);
}

SamplerInputs NormalBatchStreamProcessor::allocateSamplerInputs(const StreamGroups&  stream_groups,
                                                                size_t               total_batch_size_in,
                                                                size_t               total_batch_size_out,
                                                                const torch::Tensor& sequence_lengths,
                                                                size_t               propose_step) const {
    // TODO(xinfei.sxf) don't sample for chunk stream
    SamplerInputs sampler_inputs;
    sampler_inputs.step             = stream_groups.maxSeqLen() + propose_step;
    sampler_inputs.batch_size       = total_batch_size_in;
    sampler_inputs.batch_size_out   = total_batch_size_out;
    auto bs                         = (int64_t)total_batch_size_in;
    sampler_inputs.sequence_lengths = torch::empty({bs}, torch::kInt32);
    sampler_inputs.logits_processor_states_ptr.reset();
    sampler_inputs.input_lengths  = torch::empty({bs}, torch::kInt32);
    sampler_inputs.num_beams_in   = torch::empty({bs}, torch::kLong);
    sampler_inputs.num_beams_out  = torch::empty({bs}, torch::kLong);
    static const auto pinned_int  = torch::TensorOptions(torch::kInt).pinned_memory(true);
    static const auto pinned_i32  = torch::TensorOptions(torch::kInt32).pinned_memory(true);
    static const auto pinned_f32  = torch::TensorOptions(torch::kFloat32).pinned_memory(true);
    static const auto pinned_bool = torch::TensorOptions(torch::kBool).pinned_memory(true);

    sampler_inputs.top_k                = torch::empty({bs}, pinned_int);
    sampler_inputs.top_p                = torch::empty({bs}, pinned_f32);
    sampler_inputs.temperature          = torch::empty({bs}, pinned_f32);
    sampler_inputs.repetition_penalty   = torch::empty({bs}, pinned_f32);
    sampler_inputs.presence_penalty     = torch::empty({bs}, pinned_f32);
    sampler_inputs.frequency_penalty    = torch::empty({bs}, pinned_f32);
    sampler_inputs.no_repeat_ngram_size = torch::empty({bs}, pinned_i32);
    sampler_inputs.do_sample            = torch::empty({bs}, pinned_bool);
    sampler_inputs.finished_mask        = torch::empty({bs}, torch::kBool);
    if (stream_groups.needReturnCumLogProbs()) {
        sampler_inputs.cum_log_probs = torch::empty({(int64_t)total_batch_size_in}, torch::kFloat32);
    }
    sampler_inputs.token_ids =
        torch::empty({(int64_t)total_batch_size_in, (int64_t)(sampler_inputs.step + 1)}, torch::kInt32);
    sampler_inputs.generator.resize(total_batch_size_in);
    return sampler_inputs;
}

void NormalBatchStreamProcessor::setCommonSamplerInputs(SamplerInputs&                sampler_inputs,
                                                        std::list<GenerateStreamPtr>& all_streams,
                                                        bool                          score_batch,
                                                        size_t                        propose_step) const {
    int*      input_lengths        = sampler_inputs.input_lengths.data_ptr<int32_t>();
    int*      sequence_lengths     = sampler_inputs.sequence_lengths.data_ptr<int32_t>();
    uint64_t* num_beams_in         = reinterpret_cast<uint64_t*>(sampler_inputs.num_beams_in.data_ptr<int64_t>());
    uint64_t* num_beams_out        = reinterpret_cast<uint64_t*>(sampler_inputs.num_beams_out.data_ptr<int64_t>());
    uint32_t* top_k                = reinterpret_cast<uint32_t*>(sampler_inputs.top_k.data_ptr<int32_t>());
    float*    top_p                = sampler_inputs.top_p.data_ptr<float>();
    float*    temperature          = sampler_inputs.temperature.data_ptr<float>();
    float*    repetition_penalty   = sampler_inputs.repetition_penalty.data_ptr<float>();
    float*    presence_penalty     = sampler_inputs.presence_penalty.data_ptr<float>();
    float*    frequency_penalty    = sampler_inputs.frequency_penalty.data_ptr<float>();
    int32_t*  no_repeat_ngram_size = sampler_inputs.no_repeat_ngram_size.data_ptr<int32_t>();
    bool*     do_sample            = reinterpret_cast<bool*>(sampler_inputs.do_sample.data_ptr());

    int batch_idx = 0;
    for (auto& stream : all_streams) {
        int sampler_batch_size;
        if (score_batch) {
            sampler_batch_size = stream->scoreLen();
        } else if (stream->needTilingForSampling()) {
            sampler_batch_size = stream->nextBatchSize();
        } else {
            sampler_batch_size = stream->currentBatchSize();
        }
        if (sampler_inputs.cum_log_probs.defined()) {
            const auto& cum_log_probs = stream->cumLogProbs();
            memcpy(sampler_inputs.cum_log_probs.data_ptr<float>() + batch_idx,
                   cum_log_probs.data_ptr<float>(),
                   cum_log_probs.numel() * sizeof(float));
        }
        for (int i = 0; i < sampler_batch_size; ++i) {
            input_lengths[batch_idx]      = stream->inputLength();
            sequence_lengths[batch_idx]   = stream->seqLength() + propose_step;
            num_beams_in[batch_idx]       = stream->currentNumBeams();
            num_beams_out[batch_idx]      = stream->nextNumBeams();
            top_k[batch_idx]              = stream->generateConfig()->top_k;
            top_p[batch_idx]              = stream->generateConfig()->top_p;
            temperature[batch_idx]        = stream->generateConfig()->temperature;
            repetition_penalty[batch_idx] = stream->generateConfig()->repetition_penalty;
            presence_penalty[batch_idx]   = stream->generateConfig()->presence_penalty;
            frequency_penalty[batch_idx]  = stream->generateConfig()->frequency_penalty;
            do_sample[batch_idx]          = stream->generateConfig()->do_sample;
            if (!do_sample[batch_idx]) {
                top_k[batch_idx]       = 1;
                top_p[batch_idx]       = 1;
                temperature[batch_idx] = 1;
            }
            no_repeat_ngram_size[batch_idx]     = stream->generateConfig()->no_repeat_ngram_size.value_or(0);
            sampler_inputs.generator[batch_idx] = stream->getGenerator();
            batch_idx += 1;
        }
    }
}

void NormalBatchStreamProcessor::setLogitsProcessorInputs(SamplerInputs&                sampler_inputs,
                                                          std::list<GenerateStreamPtr>& all_streams,
                                                          bool                          score_batch) const {
    LogitsProcessorStatesPtr state_ptr = std::make_shared<LogitsProcessorStates>();
    std::for_each(all_streams.begin(), all_streams.end(), [&state_ptr, idx = 0](auto& stream) mutable {
        for (const auto& processor : stream->getAllLogitsProcessorPtr()) {
            state_ptr->insert(processor, idx, idx + stream->currentBatchSize());
        }
        idx += stream->currentBatchSize();
    });
    sampler_inputs.logits_processor_states_ptr = state_ptr;
}

absl::Status NormalBatchStreamProcessor::dispatch(const StreamGroups& stream_groups,
                                                  const MergedOutput& merge_outputs) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    const auto& sampler_output    = merge_outputs.sampler_output;
    const auto& new_all_token_ids = sampler_output.token_ids;
    RTP_LLM_LOG_DEBUG("new_all_token_ids = [%s]", tensorDebugStringWithData<int32_t>(new_all_token_ids).c_str());
    const size_t total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    RTP_LLM_CHECK(total_batch_size_out == (size_t)new_all_token_ids.size(0));
    int  batch_idx_in     = 0;
    int  batch_idx_out    = 0;
    int  token_offset     = 0;
    bool return_all_probs = stream_groups.needReturnAllProbs();
    auto new_tokens_all   = torch::empty({(int64_t)total_batch_size_out, 1}, torch::kInt32);

    for (auto& stream : stream_groups.allStreams()) {
        auto cur_batch_size  = stream->currentBatchSize();
        auto next_batch_size = stream->nextBatchSize();
        auto token_size      = stream->currentExecuteTokenSize();

        dispatchSingleStream(
            stream, merge_outputs, batch_idx_in, batch_idx_out, token_offset, return_all_probs, new_tokens_all);

        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
        token_offset += token_size;
    }

    RTP_LLM_LOG_DEBUG("dispatch done");
    return absl::OkStatus();
}

void NormalBatchStreamProcessor::dispatchSingleStream(GenerateStreamPtr    stream,
                                                      const MergedOutput&  merge_outputs,
                                                      int                  batch_idx_in,
                                                      int                  batch_idx_out,
                                                      int                  token_offset,
                                                      bool                 return_all_probs,
                                                      const torch::Tensor& new_tokens_all) const {

    const auto&  model_output      = merge_outputs.model_output;
    const auto&  sampler_output    = merge_outputs.sampler_output;
    const auto&  new_all_token_ids = sampler_output.token_ids;
    const size_t token_stride      = new_all_token_ids.size(1);

    auto cur_batch_size  = stream->currentBatchSize();
    auto next_batch_size = stream->nextBatchSize();
    auto token_size      = stream->currentExecuteTokenSize();

    auto batch_new_all_token_ids = new_all_token_ids.narrow(0, batch_idx_out, next_batch_size);

    bool has_beam_search = stream->currentNumBeams() > 1 || stream->nextNumBeams() > 1;
    bool has_var_batch   = stream->currentBatchSize() != stream->nextBatchSize();

    // construct mapping from output batches to input batches
    torch::Tensor src_batch_indices;
    if (has_beam_search) {
        // beam search
        src_batch_indices = sampler_output.beam_index.narrow(0, batch_idx_out, next_batch_size);
    } else if (has_var_batch) {
        // from context stream to decode straem, there might be other cases in future
        src_batch_indices = torch::zeros({(int64_t)next_batch_size}, torch::kInt32);
    }
    const auto get_src_idx = [&](int32_t dst_idx) {
        return src_batch_indices.defined() ? src_batch_indices.data_ptr<int32_t>()[dst_idx] : dst_idx;
    };

    // construct update info
    torch::Tensor batch_hidden_states;
    if (stream->generateConfig()->return_hidden_states) {
        batch_hidden_states = model_output.hidden_states.narrow(0, batch_idx_in, cur_batch_size);
    }

    torch::Tensor batch_logits;
    if (stream->returnLogits() || stream->calculateSoftmaxProbs() || has_beam_search) {
        batch_logits = model_output.logits.narrow(0, batch_idx_in, cur_batch_size);
    }

    torch::Tensor all_probs;
    if (return_all_probs) {
        all_probs = sampler_output.all_probs.narrow(0, batch_idx_out, next_batch_size);
    };

    torch::Tensor batch_cum_log_probs;
    if (sampler_output.cum_log_probs.defined()) {
        batch_cum_log_probs = sampler_output.cum_log_probs.narrow(0, batch_idx_out, next_batch_size);
    }

    torch::Tensor loss;
    if (stream->calculateLoss()) {
        auto all_logits_tensor = model_output.all_logits.narrow(0, token_offset, token_size - 1);
        auto tokens            = stream->currentExecuteTokens(0);
        auto label_tensor =
            torch::from_blob(const_cast<int*>(tokens.data() + 1), {(int64_t)(tokens.size() - 1)}, torch::kInt32)
                .to(torch::kCUDA);
        auto labels_int64 = label_tensor.toType(torch::kInt64);
        loss = torch::cross_entropy_loss(all_logits_tensor, labels_int64, torch::nullopt, at::Reduction::None)
                   .to(torch::kFloat32);
    }

    torch::Tensor all_hidden_states;
    if (stream->needReturnHiddenStates()) {
        all_hidden_states = model_output.all_hidden_states.narrow(0, token_offset, token_size);
    }

    auto new_tokens = new_tokens_all.narrow(0, batch_idx_out, next_batch_size);
    for (size_t i = 0; i < next_batch_size; ++i) {
        new_tokens.data_ptr<int32_t>()[i] =
            new_all_token_ids.data_ptr<int32_t>()[(batch_idx_out + i) * token_stride + token_stride - 1];
    }

    torch::Tensor current_softmax_result;
    if (stream->calculateSoftmaxProbs()) {
        auto batch_softmax_input = batch_logits.to(torch::kFloat32).contiguous();
#if USING_CUDA
        cudaSoftmaxInplace(batch_softmax_input, at::cuda::getCurrentCUDAStream().stream());
#else
        batch_softmax_input = torch::softmax(batch_softmax_input, -1);
#endif
        auto batch_softmax_tensor = batch_softmax_input.cpu();
        current_softmax_result    = torch::empty({(int64_t)next_batch_size, 1}, torch::kFloat32);
        for (int i = 0; i < next_batch_size; ++i) {
            current_softmax_result[i][0] = batch_softmax_tensor[get_src_idx(i)][new_tokens.data_ptr<int32_t>()[i]];
        }
    }

    for (int i = 0; i < cur_batch_size; ++i) {
        if (sampler_output.success.defined() && !(sampler_output.success.data_ptr<bool>()[batch_idx_in + i])) {
            stream->setStop(ErrorCode::UNKNOWN_ERROR, "sampler generate token id failed");
        }
    }

    RTP_LLM_LOG_DEBUG("stream [%ld], new_tokens size = [%ld]", stream->streamId(), new_tokens.numel());

    stream->update({has_beam_search ? batch_new_all_token_ids : new_tokens,
                    1,
                    batch_hidden_states,
                    batch_logits,
                    current_softmax_result,
                    batch_cum_log_probs,
                    all_probs,
                    loss,
                    src_batch_indices,
                    all_hidden_states});
}

}  // namespace rtp_llm
