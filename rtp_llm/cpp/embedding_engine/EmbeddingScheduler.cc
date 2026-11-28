#include "rtp_llm/cpp/embedding_engine/EmbeddingScheduler.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <algorithm>
#include <cstdint>
#include <mutex>

using namespace std;
namespace rtp_llm {

namespace {

// Cap co-batched unfused prefill attention score matrices ([B, H, S, S] FP32)
// when padded mixed-length batches would otherwise explode quadratically.
// A single valid stream still runs standalone when new_streams is empty.
constexpr uint64_t kMaxUnfusedPrefillAttentionScoresBytes = 512ULL * 1024ULL * 1024ULL;

bool usesUnfusedPrefillAttention(const ModelConfig& model_config, const FMHAConfig& fmha_config) {
    if (model_config.data_type == DataType::TYPE_FP32) {
        return true;
    }
    const bool all_dense_fmha_disabled = !fmha_config.enable_open_source_fmha
                                         && !fmha_config.enable_flashinfer_trt_fmha_v2
                                         && !fmha_config.enable_paged_flashinfer_trt_fmha_v2
                                         && !fmha_config.enable_flashinfer_trtllm_gen
                                         && fmha_config.disable_flashinfer_native;
    return all_dense_fmha_disabled;
}

size_t getStreamBatchSize(const EmbeddingStreamPtr& stream) {
    if (!stream || !stream->embeddingInput()) {
        return 1;
    }
    const int64_t batch_size = stream->batchSize();
    return batch_size > 0 ? static_cast<size_t>(batch_size) : 1;
}

size_t getStreamMaxSeqLen(const EmbeddingStreamPtr& stream) {
    if (!stream) {
        return 0;
    }
    const auto input = stream->embeddingInput();
    if (!input || !input->input_lengths.defined() || input->input_lengths.numel() <= 0) {
        return static_cast<size_t>(std::max<int64_t>(0, stream->inputLength()));
    }
    const auto     lengths  = input->input_lengths.to(torch::kCPU, torch::kInt32).contiguous();
    const int32_t* data     = lengths.data_ptr<int32_t>();
    int64_t        max_len  = 0;
    const int64_t  num_elem = lengths.numel();
    for (int64_t i = 0; i < num_elem; ++i) {
        max_len = std::max<int64_t>(max_len, data[i]);
    }
    if (num_elem == 1) {
        max_len = std::max<int64_t>(max_len, stream->inputLength());
    }
    return static_cast<size_t>(std::max<int64_t>(0, max_len));
}

bool exceedsMaxTokenBudget(size_t batch_size, size_t max_seq_len, unsigned __int128 max_token_budget) {
    const unsigned __int128 product =
        static_cast<unsigned __int128>(batch_size) * static_cast<unsigned __int128>(max_seq_len);
    return product > max_token_budget;
}

bool exceedsUnfusedAttentionScoreBudget(size_t batch_size, size_t head_num, size_t max_seq_len) {
    const size_t            effective_heads = std::max<size_t>(1, head_num);
    const unsigned __int128 bytes           = static_cast<unsigned __int128>(batch_size)
                                    * static_cast<unsigned __int128>(effective_heads)
                                    * static_cast<unsigned __int128>(max_seq_len)
                                    * static_cast<unsigned __int128>(max_seq_len)
                                    * static_cast<unsigned __int128>(sizeof(float));
    return bytes > static_cast<unsigned __int128>(kMaxUnfusedPrefillAttentionScoresBytes);
}

}  // namespace

EmbeddingScheduler::EmbeddingScheduler(const ModelConfig&                 model_config,
                                       const ConcurrencyConfig&           concurrency_config,
                                       const RuntimeConfig&               runtime_config,
                                       const kmonitor::MetricsReporterPtr metrics_reporter,
                                       const FMHAConfig&                  fmha_config):
    model_config_(model_config),
    concurrency_config_(concurrency_config),
    runtime_config_(runtime_config),
    fmha_config_(fmha_config),
    metrics_reporter_(metrics_reporter) {}

EmbeddingScheduler::~EmbeddingScheduler() {
    (void)stop();
    RTP_LLM_LOG_INFO("destory EmbeddingScheduler");
}

absl::Status EmbeddingScheduler::stop() {
    RTP_LLM_LOG_INFO("stop EmbeddingScheduler");
    lock_guard<mutex> lock(lock_);
    stop_ = true;
    cond_.notify_all();
    return absl::OkStatus();
}

absl::Status EmbeddingScheduler::enqueue(EmbeddingStreamPtr stream) {
    lock_guard<mutex> lock(lock_);
    waiting_streams_.emplace_back(stream);
    cond_.notify_all();
    return absl::OkStatus();
}

absl::StatusOr<list<EmbeddingStreamPtr>> EmbeddingScheduler::scheduleNew() {
    unique_lock<mutex> lock(lock_);
    cond_.wait(lock, [this] { return stop_ || !waiting_streams_.empty(); });
    std::list<EmbeddingStreamPtr> new_streams;
    const int64_t                 max_context_batch_size =
        std::max<int64_t>(0, runtime_config_.fifo_scheduler_config.max_context_batch_size);
    const int64_t           model_max_seq_len = std::max<int64_t>(0, model_config_.max_seq_len);
    const unsigned __int128 max_token_budget =
        static_cast<unsigned __int128>(max_context_batch_size) * static_cast<unsigned __int128>(model_max_seq_len);
    const int64_t max_batch_size              = std::max<int64_t>(0, runtime_config_.max_generate_batch_size);
    const bool    enforce_unfused_score_guard = usesUnfusedPrefillAttention(model_config_, fmha_config_);
    const size_t  head_num                    = model_config_.attn_config.head_num;

    unsigned __int128 total_len   = 0;
    size_t            batch_size  = 0;
    size_t            max_seq_len = 0;
    auto              it          = waiting_streams_.begin();
    while (it != waiting_streams_.end()) {
        const auto& stream        = *it;
        const auto  stream_tokens = static_cast<unsigned __int128>(std::max<int64_t>(0, stream->inputLength()));
        if (total_len + stream_tokens > max_token_budget) {
            break;
        }
        const size_t stream_bs      = getStreamBatchSize(stream);
        const size_t stream_max_seq = getStreamMaxSeqLen(stream);
        if (!new_streams.empty()) {
            const size_t candidate_bs      = batch_size + stream_bs;
            const size_t candidate_max_seq = std::max(max_seq_len, stream_max_seq);
            if (max_batch_size > 0 && candidate_bs > static_cast<size_t>(max_batch_size)) {
                break;
            }
            if (exceedsMaxTokenBudget(candidate_bs, candidate_max_seq, max_token_budget)) {
                break;
            }
            if (enforce_unfused_score_guard
                && exceedsUnfusedAttentionScoreBudget(candidate_bs, head_num, candidate_max_seq)) {
                break;
            }
        }
        stream->setStart();
        new_streams.push_back(stream);
        total_len += stream_tokens;
        batch_size += stream_bs;
        max_seq_len = std::max(max_seq_len, stream_max_seq);
        it          = waiting_streams_.erase(it);
    }
    // if new streams is empty, meaning that first stream is too big
    if (waiting_streams_.size() != 0 && new_streams.size() == 0) {
        it                 = waiting_streams_.begin();
        const auto& stream = *it;
        stream->setError("long prompt error, not scheduled");
        it = waiting_streams_.erase(it);
    }
    reportMetrics(new_streams.size());
    return new_streams;
}

void EmbeddingScheduler::reportMetrics(size_t new_stream_size) {
    if (metrics_reporter_) {
        RtpLLMSchedulerMetricsCollector collector;
        collector.wait_stream_size    = waitingStreamsSize();
        collector.running_stream_size = new_stream_size;
        metrics_reporter_->report<RtpLLMSchedulerMetrics, RtpLLMSchedulerMetricsCollector>(nullptr, &collector);
    }
}

int EmbeddingScheduler::waitingStreamsSize() {
    return waiting_streams_.size();
}

}  // namespace rtp_llm
