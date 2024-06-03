#pragma once
#include <assert.h>
#include <cstdint>
#include <mutex>
#include <string>
#include <optional>
#include <queue>
#include <condition_variable>
#include "src/fastertransformer/core/Buffer.h"
#include "autil/TimeUtility.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/dataclass/StreamCacheResource.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "autil/SynchronizedQueue.h"
#include "maga_transformer/cpp/system_prompt/SystemPrompt.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "absl/status/statusor.h"
#include "kmonitor/client/MetricsReporter.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"


namespace ft = fastertransformer;

namespace rtp_llm {

class GenerateStream {
public:
    GenerateStream(const std::shared_ptr<GenerateInput>& query, const ft::GptInitParameter& params, const ResourceContext& resource_context, kmonitor::MetricsReporterPtr metrics_reporter);
    virtual ~GenerateStream() {
        reportMetric();
        generate_outputs_queue_.wakeup();
    }

public:
    // Exported to python world.
    void                                cancel();
    absl::StatusOr<GenerateOutputs>     nextOutput();

    // Only used in C++ world.
    
    bool isContextStream() const;
    int tileNum() const;
    std::vector<int> contextTokens() const;
    std::vector<int> currentExecuteTokens() const;

    virtual int tryReleaseKVBlock(int nums) {
        return stream_cache_resource_.tryReleaseKVBlock(nums);
    }
    virtual bool initKVBlock() {
        if (is_context_stream_) {
            wait_time_us_ = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
        }
        is_context_stream_ = true;
        return stream_cache_resource_.initKVBlock();
    }
    virtual bool incrKVBlock() {
        return stream_cache_resource_.incrKVBlock();
    }
    int nextNeedBlockNums() const {
        return stream_cache_resource_.needKVCacheBlockNums();
    }

    std::shared_ptr<GenerateInput> generateInput() const;

    ft::SpecialTokens specialTokens() const {
        return special_tokens_;
    }

    const ResourceContext& resourceContext() const {
        return stream_cache_resource_.resourceContext();
    }

    size_t maxSeqLen() const;

    virtual void releaseResource() {
        if (need_release_resource_) {
            stream_cache_resource_.releaseResource();
        }
    }

    bool isStreaming() const {
        return generate_input_->generate_config->is_streaming;
    }

    int64_t streamId() const {
        return generate_input_->request_id;
    }

    int inputLength() const {
        return generate_input_->inputLength();
    }

    int loraId() const {
        return generate_input_->lora_id;
    }

    // TODO(xinfei.sxf) consider reuse when fallback
    int contextLength() const {
        return seq_length_- reuse_length_;
    }

    int needKVCacheBlockNums() const {
        return stream_cache_resource_.needKVCacheBlockNums();
    }

    size_t maxBlockSize() const {
        return stream_cache_resource_.maxBlockSize();
    }

    const ft::BufferPtr& completeTokenIds() {
        return complete_token_ids_;
    }

    // TODO(xinfei.sxf) batch?
    std::vector<int> completeTokenIdsVec() {
        return fastertransformer::buffer2vector<int>(*complete_token_ids_, seq_length_);
    }

    int currentExecuteTokenSize() {
        return currentExecuteTokens().size();
    }

    int batchSize() const;

    int prefixLength() const {
        return generate_input_->prefix_length;
    }

    std::shared_ptr<GenerateConfig>& generateConfig() {
        return generate_input_->generate_config;
    }

    int numBeams() const {
        return generate_input_->generate_config->num_beams;
    }

    int numReturnSequences() const {
        return generate_input_->generate_config->num_return_sequences;
    }

    std::optional<ft::BufferPtr>& imageEmbeddings() {
        return generate_input_->image_embeddings;
    }

    std::optional<int> lora_id() {
        return generate_input_->lora_id;
    }

    int reuseLength() const {
        return reuse_length_;
    }

    int seqLength() const {
        return seq_length_;
    }

    // for test
    void setSeqLength(int seq_length) {
        seq_length_ = seq_length;
    }

    void setIsContextStream(bool is_context_stream) {
        is_context_stream_ = is_context_stream;
    }

    void updatePrefix(const std::shared_ptr<SystemPrompt>& system_prompt);

    void setStop(const std::string& err_msg) {
        std::lock_guard<std::mutex> lock(output_mutex_);
        FT_LOG_WARNING("stop stream: %d %s", streamId(), err_msg.c_str());
        generate_status_.status = GenerateState::STOPPED;
        generate_status_.error_info = err_msg;
    }

    void stopAndRelease(const std::string& err_msg) {
        setStop(err_msg);
        releaseResource();
    }

    void setFinished() {
        std::lock_guard<std::mutex> lock(output_mutex_);
        generate_status_.status = GenerateState::FINISHED;
    }

    bool setRunning() {
        std::lock_guard<std::mutex> lock(output_mutex_);
        if (stoppedWithoutLock()) {
            return false;
        }
        // TODO(xinfei.sxf) reportWaitTime();
        generate_status_.status = GenerateState::RUNNING;
        for (int i = 0; i < tileNum(); ++i) {
            sub_generate_status_[i].status = GenerateState::RUNNING;
        }
        return true;
    }

    void setFinishedWithoutLock() {
        generate_status_.status = GenerateState::FINISHED;
    }

    bool stoppedWithoutLock() {
        return generate_status_.status == GenerateState::STOPPED;
    }

    bool stopped() {
        std::lock_guard<std::mutex> lock(output_mutex_);
        return generate_status_.status == GenerateState::STOPPED;
    }

    std::string stopReason() {
        std::lock_guard<std::mutex> lock(output_mutex_);
        return generate_status_.error_info;
    }

    bool finished() {
        std::lock_guard<std::mutex> lock(output_mutex_);
        return generate_status_.status == GenerateState::FINISHED;
    }

    void check_timeout() {
        auto running_time_ms = (autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_) / 1000;
        auto timeout_ms = generate_input_->generate_config->timeout_ms;
        if (timeout_ms > 0 && timeout_ms < running_time_ms) {
            stopAndRelease("query has been running " + std::to_string(running_time_ms) + " ms, "
               + "timeout_ms = " + std::to_string(timeout_ms) + ", it's timeout");
         }
    }

    // void setLoss(th::Tensor& loss) {
    //     output_.loss = loss
    // }

    void setReuseLength(int reuse_length) {
        reuse_length_ = reuse_length;
    }

    void setKVCache(const BatchKVCacheBlockAddr &kv_cache_block_addr, int reuse_length) {
        stream_cache_resource_.setKVCache(kv_cache_block_addr);
        reuse_length_ = reuse_length;
    }

    // for test
    StreamCacheResource& streamCacheResource() {
        return stream_cache_resource_;
    }

    const BatchKVCacheBlockAddr& kvCache() const {
        return stream_cache_resource_.kvCache();
    }

    bool needFinish() {
        return seq_length_ >= std::min(max_seq_len_, generate_input_->generate_config->max_new_tokens + generate_input_->inputLength()) || needFinishBySPTokens();
    }

    bool needFinishBySPTokens() const;

    bool matchEosToken() const;

    bool matchStopWordsList() const;

    void setSeqLength(uint seq_length) {
        seq_length_ = seq_length;
    };

    void update(ft::BufferPtr& new_tokens,
                int num_new_tokens,
                bool finished,
                const ft::BufferPtr& hidden_states,
                const ft::Buffer& logits,
                const ft::BufferPtr& cum_log_probs,
                bool not_update_output = false);

    void updateOutput(bool finished,
                      const ft::BufferPtr& hidden_states,
                      const ft::Buffer& logits,
                      const ft::BufferPtr& cum_log_probs);

    void setMetricsReporter(kmonitor::MetricsReporterPtr metrics_reporter) {
        metrics_reporter_ = metrics_reporter;
    }

    void reportMetric();

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "GenerateStream {"
                     << "generate_input:" << generate_input_->debugString()
                     << ", max_seq_len:" << max_seq_len_
                     << ", input_length:" << inputLength()
                     << ", seq_length:" << seq_length_
                     << ", reuse_length:" << reuse_length_
                     << ", batch_size:" << batch_size_
                     << "}";
        return debug_string.str();
    }

    const ft::BufferPtr& cumLogProbs() const {
        return cum_log_probs_;
    }

protected:
    ft::DeviceBase* device_;
    std::shared_ptr<GenerateInput>      generate_input_;
    std::shared_ptr<GenerateOutputs>    generate_outputs_;
    GenerateStatus                      generate_status_;
    std::vector<GenerateStatus>         sub_generate_status_;
    int                                 max_seq_len_;
    int                                 seq_length_;
    ft::BufferPtr                       complete_token_ids_;
    int64_t                             begin_time_us_;
    int64_t                             wait_time_us_ = 0;
    int64_t                             first_token_time_us_ = 0;
    std::mutex                          output_mutex_;
    std::condition_variable             update_cv_;
    StreamCacheResource                 stream_cache_resource_;
    SystemPromptParams                  prompt_param_;
    bool                                is_context_stream_     = true;
    size_t                              iter_count_            = 0;
    size_t                              batch_size_            = 1;
    int                                 reuse_length_          = 0;
    bool                                done_                  = false;
    bool                                cancelled_             = false;
    bool                                released_              = false;
    bool                                need_release_resource_ = true;
    kmonitor::MetricsReporterPtr        metrics_reporter_      = nullptr;
    ft::SpecialTokens                   special_tokens_;
    ft::BufferPtr                       cum_log_probs_;

    autil::SynchronizedQueue<GenerateOutputs>  generate_outputs_queue_;

    friend class StreamCacheResource;
};

typedef std::shared_ptr<GenerateStream> GenerateStreamPtr;
} // namespace rtp_llm
