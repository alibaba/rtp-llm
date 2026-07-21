#include "rtp_llm/cpp/normal_engine/NormalOutputDispatcher.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"
#include "rtp_llm/models_py/bindings/cuda/ops/StandaloneOps.h"
#include "ATen/cuda/CUDAContext.h"
#endif

namespace rtp_llm {
namespace {

const bool kAsyncDebugEnabled = []() {
    const char* env = std::getenv("RTP_LLM_ASYNC_DEBUG");
    return env != nullptr && std::string(env) == "1";
}();

bool asyncDebugEnabled() {
    return kAsyncDebugEnabled;
}

torch::Tensor copyToPinnedCpuAsync(const torch::Tensor& tensor, bool& need_sync) {
    if (!tensor.defined() || !tensor.is_cuda()) {
        return tensor;
    }

    auto cpu_tensor = torch::empty(
        tensor.sizes(), torch::TensorOptions().dtype(tensor.scalar_type()).device(torch::kCPU).pinned_memory(true));
    cpu_tensor.copy_(tensor, /*non_blocking=*/true);
    need_sync = true;
    return cpu_tensor;
}

void syncPinnedCpuCopies(bool need_sync) {
    if (!need_sync) {
        return;
    }
    // Keep D2H waiting explicit here instead of hiding it inside Tensor::cpu().
    // The copy launch returns quickly; only this worker thread blocks on its
    // stream while the main engine thread can continue issuing CUDA work.
    cuda_graph::graphGetCurrentStream().synchronize();
}

torch::Tensor makeHostInt64IndexTensor(const std::vector<int64_t>& values, bool pinned) {
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    if (pinned) {
        options = options.pinned_memory(true);
    }
    auto tensor = torch::empty({static_cast<int64_t>(values.size())}, options);
    if (!values.empty()) {
        std::memcpy(tensor.data_ptr<int64_t>(), values.data(), values.size() * sizeof(int64_t));
    }
    return tensor;
}

torch::Tensor copyHostIndexToDeviceAsync(const torch::Tensor& host_tensor, const torch::Device& device) {
    if (!device.is_cuda()) {
        return host_tensor;
    }
    RTP_LLM_CHECK(host_tensor.is_pinned());
    return host_tensor.to(device, torch::kInt64, /*non_blocking=*/true, /*copy=*/true);
}

}  // namespace

absl::Status NormalOutputDispatcher::dispatch(const StreamGroups& stream_groups,
                                              const MergedOutput& merge_outputs) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    const auto&  model_output         = merge_outputs.model_output;
    const auto&  sampler_output       = merge_outputs.sampler_output;
    const size_t total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    RTP_LLM_CHECK(total_batch_size_out == (size_t)sampler_output.token_ids.size(0));

    auto all_streams              = stream_groups.allStreams();
    bool need_return_logprobs     = false;
    int  max_content_top_logprobs = 0;
    for (const auto& stream : all_streams) {
        if (stream->shouldComputeLogprobs()) {
            need_return_logprobs     = true;
            max_content_top_logprobs = std::max(max_content_top_logprobs, stream->generateConfig()->top_logprobs);
        }
    }
    bool any_beam_search = false;
    if (sampler_output.token_ids.defined() && sampler_output.token_ids.size(1) > 1) {
        for (const auto& stream : all_streams) {
            if (stream->currentNumBeams() > 1 || stream->nextNumBeams() > 1) {
                any_beam_search = true;
                break;
            }
        }
    }
    torch::Tensor token_ids_for_copy;
    if (sampler_output.token_ids.defined()) {
        if (any_beam_search) {
            token_ids_for_copy = sampler_output.token_ids;
        } else {
            // Slice the last column on-device so the D2H is only [B, 1] int32.
            const int64_t last_col = sampler_output.token_ids.size(1) - 1;
            token_ids_for_copy     = sampler_output.token_ids.narrow(1, last_col, 1).contiguous();
        }
    }

    // Map requested output rows back to the compact raw-input snapshot. This
    // mapping covers context tiling and beam reordering without doing any
    // full-vocabulary work for rows that did not request logprobs.
    std::vector<int64_t> expected_raw_row_indices;
    std::vector<int64_t> requested_output_row_indices;
    std::vector<int64_t> requested_output_model_rows;
    int64_t              model_batch_idx  = 0;
    int64_t              output_batch_idx = 0;
    if (need_return_logprobs) {
        for (const auto& stream : all_streams) {
            const int64_t cur_batch_size  = stream->currentBatchSize();
            const int64_t next_batch_size = stream->nextBatchSize();
            if (stream->shouldComputeLogprobs()) {
                for (int64_t i = 0; i < cur_batch_size; ++i) {
                    expected_raw_row_indices.push_back(model_batch_idx + i);
                }

                const bool    has_beam_search = stream->currentNumBeams() > 1 || stream->nextNumBeams() > 1;
                const bool    has_var_batch   = cur_batch_size != next_batch_size;
                torch::Tensor src_batch_indices;
                if (has_beam_search) {
                    RTP_LLM_CHECK(sampler_output.beam_index.defined());
                    RTP_LLM_CHECK(!sampler_output.beam_index.is_cuda());
                    src_batch_indices = sampler_output.beam_index.narrow(0, output_batch_idx, next_batch_size);
                }
                for (int64_t i = 0; i < next_batch_size; ++i) {
                    int64_t src_idx = i;
                    if (src_batch_indices.defined()) {
                        src_idx = src_batch_indices.data_ptr<int32_t>()[i];
                    } else if (has_var_batch) {
                        // Context-to-decode tiling duplicates the single model row.
                        src_idx = 0;
                    }
                    RTP_LLM_CHECK(src_idx >= 0 && src_idx < cur_batch_size);
                    requested_output_row_indices.push_back(output_batch_idx + i);
                    requested_output_model_rows.push_back(model_batch_idx + src_idx);
                }
            }
            model_batch_idx += cur_batch_size;
            output_batch_idx += next_batch_size;
        }
    }

    torch::Tensor compact_token_logprobs;
    torch::Tensor compact_top_logprob_token_ids;
    torch::Tensor compact_top_logprobs;
    // Keep every pinned H2D source alive through the batch's existing D2H
    // synchronization. All index copies and consumers run on the same stream.
    std::vector<torch::Tensor> index_host_buffers;
    auto make_device_index = [&index_host_buffers](const std::vector<int64_t>& values, const torch::Device& device) {
        auto host_tensor   = makeHostInt64IndexTensor(values, device.is_cuda());
        auto device_tensor = copyHostIndexToDeviceAsync(host_tensor, device);
        index_host_buffers.emplace_back(std::move(host_tensor));
        return device_tensor;
    };
    if (need_return_logprobs) {
        RTP_LLM_CHECK(!expected_raw_row_indices.empty());
        RTP_LLM_CHECK(!requested_output_row_indices.empty());

        const int64_t real_vocab_size = all_streams.front()->vocabSize();
        RTP_LLM_CHECK(real_vocab_size > 0);
        for (const auto& stream : all_streams) {
            RTP_LLM_CHECK(stream->vocabSize() == real_vocab_size);
        }

        torch::Tensor raw_logits      = sampler_output.raw_logprobs_logits;
        torch::Tensor raw_row_indices = sampler_output.raw_logprobs_row_indices;
        if (!raw_logits.defined()) {
            // Compatibility for direct dispatcher tests/callers. Production
            // normal sampling always carries the pre-sampler compact snapshot.
            RTP_LLM_CHECK(model_output.logits.defined());
            RTP_LLM_CHECK(model_output.logits.dim() == 2);
            RTP_LLM_CHECK(real_vocab_size <= model_output.logits.size(1));
            raw_row_indices = makeHostInt64IndexTensor(expected_raw_row_indices, model_output.logits.is_cuda());
            auto raw_row_indices_device = copyHostIndexToDeviceAsync(raw_row_indices, model_output.logits.device());
            index_host_buffers.emplace_back(raw_row_indices);
            raw_logits = model_output.logits.narrow(1, 0, real_vocab_size).index_select(0, raw_row_indices_device);
        }

        RTP_LLM_CHECK(raw_logits.dim() == 2);
        RTP_LLM_CHECK(raw_logits.size(1) >= real_vocab_size);
        raw_logits = raw_logits.narrow(1, 0, real_vocab_size);
        RTP_LLM_CHECK(raw_row_indices.defined());
        RTP_LLM_CHECK(!raw_row_indices.is_cuda());
        RTP_LLM_CHECK(raw_row_indices.scalar_type() == torch::kInt64);
        raw_row_indices = raw_row_indices.contiguous();
        RTP_LLM_CHECK(raw_row_indices.dim() == 1);
        RTP_LLM_CHECK(raw_row_indices.size(0) == raw_logits.size(0));
        RTP_LLM_CHECK(raw_row_indices.size(0) == static_cast<int64_t>(expected_raw_row_indices.size()));

        std::vector<int64_t> compact_index_by_model_row(model_batch_idx, -1);
        for (int64_t compact_idx = 0; compact_idx < raw_row_indices.size(0); ++compact_idx) {
            const int64_t model_row = raw_row_indices.data_ptr<int64_t>()[compact_idx];
            RTP_LLM_CHECK(model_row >= 0 && model_row < model_batch_idx);
            RTP_LLM_CHECK(compact_index_by_model_row[model_row] == -1);
            compact_index_by_model_row[model_row] = compact_idx;
        }
        for (const int64_t expected_row : expected_raw_row_indices) {
            RTP_LLM_CHECK(compact_index_by_model_row[expected_row] >= 0);
        }

        std::vector<int64_t> output_to_compact_indices;
        output_to_compact_indices.reserve(requested_output_model_rows.size());
        for (const int64_t model_row : requested_output_model_rows) {
            const int64_t compact_idx = compact_index_by_model_row[model_row];
            RTP_LLM_CHECK(compact_idx >= 0);
            output_to_compact_indices.push_back(compact_idx);
        }

        auto output_to_compact     = make_device_index(output_to_compact_indices, raw_logits.device());
        auto requested_output_rows = make_device_index(requested_output_row_indices, sampler_output.token_ids.device());
        auto selected_token_ids    = sampler_output.token_ids.select(1, sampler_output.token_ids.size(1) - 1)
                                      .index_select(0, requested_output_rows);
        if (raw_logits.is_cuda() && !selected_token_ids.is_cuda()) {
            selected_token_ids = selected_token_ids.pin_memory();
            index_host_buffers.emplace_back(selected_token_ids);
        }
        auto sampled_token_ids =
            selected_token_ids.to(raw_logits.device(), torch::kInt64, /*non_blocking=*/true, /*copy=*/true)
                .contiguous();
        // An invalid sampled ID is reported by GenerateStream::update. Clamp
        // only the lookup index here so a padded/invalid ID cannot turn the
        // bookkeeping gather into an asynchronous device-side OOB failure.
        auto sampled_token_lookup_ids = sampled_token_ids.clamp(0, real_vocab_size - 1);

        torch::Tensor row_max;
        torch::Tensor row_shifted_logsumexp;
#if USING_CUDA
        if (raw_logits.is_cuda()) {
            auto stats_options    = raw_logits.options().dtype(torch::kFloat32).requires_grad(false);
            row_max               = torch::empty({raw_logits.size(0)}, stats_options);
            row_shifted_logsumexp = torch::empty({raw_logits.size(0)}, stats_options);
            invokeMtpRowLogSoftmaxStats(raw_logits,
                                        row_max,
                                        row_shifted_logsumexp,
                                        real_vocab_size,
                                        cuda_graph::graphGetCurrentStream().stream());
        } else
#endif
        {
            // CPU fallback keeps the reference implementation. CUDA uses a
            // row reduction above so half/bfloat16 inputs never materialize a
            // full FP32 [requested_rows, vocab] temporary.
            auto fp32_logits      = raw_logits.to(torch::kFloat32);
            row_max               = std::get<0>(fp32_logits.max(/*dim=*/-1));
            row_shifted_logsumexp = torch::logsumexp(fp32_logits - row_max.unsqueeze(1), /*dim=*/-1);
        }
        auto output_row_max               = row_max.index_select(0, output_to_compact).unsqueeze(1);
        auto output_row_shifted_logsumexp = row_shifted_logsumexp.index_select(0, output_to_compact).unsqueeze(1);

        auto flat_selected_indices = output_to_compact * real_vocab_size + sampled_token_lookup_ids;
        auto selected_logits =
            raw_logits.reshape({-1}).index_select(0, flat_selected_indices).to(torch::kFloat32).reshape({-1, 1});
        compact_token_logprobs = (selected_logits - output_row_max) - output_row_shifted_logsumexp;

        const int64_t max_top_logprobs =
            std::min<int64_t>(std::max<int64_t>(max_content_top_logprobs, 0), real_vocab_size);
        if (max_top_logprobs > 0) {
            auto topk_result        = raw_logits.topk(max_top_logprobs, -1, true, true);
            auto input_top_logprobs = (std::get<0>(topk_result).to(torch::kFloat32) - row_max.unsqueeze(1))
                                      - row_shifted_logsumexp.unsqueeze(1);
            compact_top_logprobs = input_top_logprobs.index_select(0, output_to_compact).unsqueeze(1).contiguous();
            compact_top_logprob_token_ids =
                std::get<1>(topk_result).to(torch::kInt32).index_select(0, output_to_compact).unsqueeze(1).contiguous();
        } else {
            const int64_t requested_output_count = static_cast<int64_t>(requested_output_row_indices.size());
            compact_top_logprobs =
                torch::empty({requested_output_count, 1, 0}, raw_logits.options().dtype(torch::kFloat32));
            compact_top_logprob_token_ids =
                torch::empty({requested_output_count, 1, 0}, raw_logits.options().dtype(torch::kInt32));
        }
    }

    // token IDs, success, and all compact logprob results share one D2H wait.
    // Each CUDA tensor is staged into pinned memory before the synchronization;
    // per-stream dispatch below only slices CPU tensors.
    bool          need_d2h_sync   = false;
    torch::Tensor token_ids_cpu   = copyToPinnedCpuAsync(token_ids_for_copy, need_d2h_sync);
    torch::Tensor success_cpu     = copyToPinnedCpuAsync(sampler_output.success, need_d2h_sync);
    compact_token_logprobs        = copyToPinnedCpuAsync(compact_token_logprobs, need_d2h_sync);
    compact_top_logprob_token_ids = copyToPinnedCpuAsync(compact_top_logprob_token_ids, need_d2h_sync);
    compact_top_logprobs          = copyToPinnedCpuAsync(compact_top_logprobs, need_d2h_sync);
    syncPinnedCpuCopies(need_d2h_sync);
    RTP_LLM_LOG_DEBUG("new_all_token_ids = [%s]", tensorDebugStringWithData<int32_t>(token_ids_cpu).c_str());

    int  batch_idx_in       = 0;
    int  batch_idx_out      = 0;
    int  token_offset       = 0;
    int  logprobs_batch_idx = 0;
    bool return_all_probs   = stream_groups.needReturnAllProbs();
    auto new_tokens_all     = torch::empty({(int64_t)total_batch_size_out, 1}, torch::kInt32);

    for (const auto& stream : all_streams) {
        auto       cur_batch_size          = stream->currentBatchSize();
        auto       next_batch_size         = stream->nextBatchSize();
        auto       token_size              = stream->currentExecuteTokenSize();
        const bool return_content_logprobs = stream->shouldComputeLogprobs();

        dispatchSingleStream(stream,
                             merge_outputs,
                             batch_idx_in,
                             batch_idx_out,
                             token_offset,
                             return_all_probs,
                             new_tokens_all,
                             token_ids_cpu,
                             success_cpu,
                             return_content_logprobs,
                             logprobs_batch_idx,
                             compact_token_logprobs,
                             compact_top_logprob_token_ids,
                             compact_top_logprobs);

        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
        token_offset += token_size;
        if (return_content_logprobs) {
            logprobs_batch_idx += next_batch_size;
        }
    }

    RTP_LLM_LOG_DEBUG("dispatch done");
    return absl::OkStatus();
}

void NormalOutputDispatcher::dispatchSingleStream(GenerateStreamPtr    stream,
                                                  const MergedOutput&  merge_outputs,
                                                  int                  batch_idx_in,
                                                  int                  batch_idx_out,
                                                  int                  token_offset,
                                                  bool                 return_all_probs,
                                                  const torch::Tensor& new_tokens_all,
                                                  const torch::Tensor& token_ids_cpu,
                                                  const torch::Tensor& success_cpu,
                                                  bool                 return_content_logprobs,
                                                  int                  logprobs_batch_idx,
                                                  const torch::Tensor& token_logprobs_cpu,
                                                  const torch::Tensor& top_logprob_token_ids_cpu,
                                                  const torch::Tensor& top_logprobs_cpu) const {

    const auto&  model_output      = merge_outputs.model_output;
    const auto&  sampler_output    = merge_outputs.sampler_output;
    const auto&  new_all_token_ids = token_ids_cpu;
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

    torch::Tensor token_logprobs;
    torch::Tensor top_logprob_token_ids;
    torch::Tensor top_logprobs;
    if (return_content_logprobs) {
        RTP_LLM_CHECK(token_logprobs_cpu.defined());
        RTP_LLM_CHECK(top_logprob_token_ids_cpu.defined());
        RTP_LLM_CHECK(top_logprobs_cpu.defined());
        RTP_LLM_CHECK(!token_logprobs_cpu.is_cuda());
        RTP_LLM_CHECK(!top_logprob_token_ids_cpu.is_cuda());
        RTP_LLM_CHECK(!top_logprobs_cpu.is_cuda());
        const int64_t requested_top_logprobs =
            std::min<int64_t>(std::max<int64_t>(stream->generateConfig()->top_logprobs, 0), top_logprobs_cpu.size(2));
        token_logprobs        = token_logprobs_cpu.narrow(0, logprobs_batch_idx, next_batch_size).contiguous();
        top_logprob_token_ids = top_logprob_token_ids_cpu.narrow(0, logprobs_batch_idx, next_batch_size)
                                    .narrow(2, 0, requested_top_logprobs)
                                    .contiguous();
        top_logprobs = top_logprobs_cpu.narrow(0, logprobs_batch_idx, next_batch_size)
                           .narrow(2, 0, requested_top_logprobs)
                           .contiguous();
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
        if (success_cpu.defined() && !(success_cpu.data_ptr<bool>()[batch_idx_in + i])) {
            if (asyncDebugEnabled()) {
                const auto& state = stream->getNormalAsyncDeviceState();
                RTP_LLM_LOG_ERROR("[async-debug] sampler success=false: stream=%ld pd_sep=%d status=%s "
                                  "pending=%d seq_len=%d state_next_real=%d batch_idx_in=%d token_stride=%zu",
                                  stream->streamId(),
                                  stream->queryPdSep(),
                                  StreamStateToString(stream->getStatus()).c_str(),
                                  stream->hasPendingAsyncBookkeeping(),
                                  stream->seqLength(),
                                  state.next_real_seq_len,
                                  batch_idx_in + i,
                                  token_stride);
            }
            stream->reportError(ErrorCode::UNKNOWN_ERROR, "sampler generate token id failed");
        }
    }

    RTP_LLM_LOG_DEBUG("stream [%ld], new_tokens size = [%ld]", stream->streamId(), new_tokens.numel());

    const int32_t logprobs_offset =
        stream->generateConfig()->return_logprobs ? stream->logprobsContentOffset(new_tokens, 1) : 0;
    stream->update({has_beam_search ? batch_new_all_token_ids : new_tokens,
                    1,
                    batch_hidden_states,
                    batch_logits,
                    current_softmax_result,
                    batch_cum_log_probs,
                    all_probs,
                    loss,
                    src_batch_indices,
                    all_hidden_states,
                    true,
                    false,
                    token_logprobs,
                    top_logprob_token_ids,
                    top_logprobs,
                    -1,
                    logprobs_offset});
}

}  // namespace rtp_llm
