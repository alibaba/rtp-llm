#pragma once

#include "absl/status/statusor.h"
#include "autil/TimeUtility.h"
#include "autil/SynchronizedQueue.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/logits_processor/ThinkModeLogitsProcessor.h"
#include "rtp_llm/cpp/logits_processor/TreeLogitsProcessor.h"
#include "rtp_llm/cpp/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/position_ids_generator/PositionIdsGenerator.h"
#include <iterator>
#include <mutex>



namespace rtp_llm {

// WARNGING: buffer in generate stream should all be host to avoid gpu buffer hold more time (except kv cache)

struct StreamUpdateInfo {
    const rtp_llm::BufferPtr new_tokens;
    int                 num_new_tokens;
    const rtp_llm::BufferPtr hidden_states;
    const rtp_llm::BufferPtr logits;
    const rtp_llm::BufferPtr softmax_probs;
    const rtp_llm::BufferPtr cum_log_probs;
    const rtp_llm::BufferPtr all_probs;
    const rtp_llm::BufferPtr loss;
    // for mtp
    const rtp_llm::BufferPtr all_hidden_states;
    bool update_remote_generate = true;
    bool force_update_info = false;
};
struct SpeculativeExecutorStreamOutput {
public:
    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "SpeculativeExecutorStreamOutput { propose_step : " << propose_step;
        if (tokens) {
            debug_string << ", tokens: " << tokens->debugStringWithData<int32_t>();
        }
        if (logits) {
            debug_string << ", logits: " << logits->debugStringWithData<int32_t>();
        }
        if (hidden_states) {
            debug_string << ", hidden_states: " << hidden_states->debugStringWithData<int32_t>();
        }
        if (all_probs) {
            debug_string << ", all_probs" << all_probs->debugStringWithData<int32_t>();
        }
        if (softmax_probs) {
            debug_string << ", softmax_probs" << softmax_probs->debugStringWithData<float>();
        }
        debug_string << "}";
        return debug_string.str();
    }

public:
    size_t             propose_step  = 0;
    rtp_llm::BufferPtr tokens        = nullptr;  // selected tokens
    rtp_llm::BufferPtr logits        = nullptr;
    rtp_llm::BufferPtr hidden_states = nullptr;
    rtp_llm::BufferPtr loss          = nullptr;
    rtp_llm::BufferPtr all_probs     = nullptr;
    rtp_llm::BufferPtr softmax_probs = nullptr;
};
using SpeculativeExecutorStreamOutputPtr = std::shared_ptr<SpeculativeExecutorStreamOutput>;
    
class GenerateStream;

using GenerateStreamPtr = std::shared_ptr<GenerateStream>;

class GenerateStream {
public:
    GenerateStream(const std::shared_ptr<GenerateInput>& query, const rtp_llm::GptInitParameter& params,
                   const ResourceContext& resource_context, kmonitor::MetricsReporterPtr metrics_reporter,
                   size_t extra_reserve_token_num = 0);
    virtual ~GenerateStream() {
        reportMetric();
        releaseResource();
    }

public:
    // Exported to python world.
    virtual void cancel();

    virtual ErrorResult<GenerateOutputs>     nextOutput() = 0;
    virtual bool hasOutput() {return false;}

    virtual void updateOutput(const StreamUpdateInfo& update_info) = 0;
    void update(const StreamUpdateInfo& update_info);

    virtual size_t scoreLen() const {
        return 1;
    }

    // Only used in C++ world.
    int reuseBlockSize() const;
    void fakeInitKVBlock();
    virtual absl::StatusOr<int> initKVBlock(int token_capacity, size_t reserve_step = 0);
    virtual absl::StatusOr<int> incrKVBlock(int token_capacity, size_t reserve_step = 0);
    virtual int tryReleaseKVBlock(int nums);
    virtual void releaseResource();
    int nextNeedBlockNums(size_t reserve_step) const;
    void setNeedReleaseResource(bool need_release_resource);
    void incrFallbackBlock(int fallback_blocks);
    bool hasCacheKeys() const;
    const std::vector<int64_t>& cacheKeys(int32_t batch_id = 0) const;

    std::shared_ptr<GenerateInput> generateInput() const;
    std::shared_ptr<GenerateConfig>& generateConfig() const;
    std::vector<int> textTokensMask() const;
    bool isStreaming() const;
    int64_t streamId() const;
    int loraId() const;
    std::string adapterName() const;
    rtp_llm::SpecialTokens specialTokens() const;

    int tileNum() const;
    int batchSize() const;
    int numBeams() const;
    int numReturnSequences() const;
    bool calculateLoss() const;
    bool calculateSoftmaxProbs() const;
    bool returnLogits() const;
    bool returnCumLogProbs() const;
    bool genTimeline() const;
    void setGenTimeline(bool gen_timeline);
    bool updatePrefix(const std::shared_ptr<SystemPrompt>& system_prompt);
    size_t maxSeqLen() const;
    int inputLength() const;
    int seqLength() const;
    // NOTE: In generatestream, set seq len must use setSeqLength api, we need to save start_check_seq_length_
    // for checking EOS and stop words
    void setSeqLength(int seq_length);
    int adjustedCommonLen() const;
    int seqSizePerBlock() const;
    int contextLength() const;
    int prefixLength() const;
    int inputPrefixLength() const;
    int reuseLength() const;
    int initialReuseLength() const;
    size_t maxTokenNum() const;
    void setReuseLength(int reuse_length);
    void setInitialReuseLength(int initial_reuse_length);
    int fallbackPrefixLength() const;
    void setFallbackPrefixLength(int fallback_prefix_length);
    void incLastOutputPos();

    absl::StatusOr<int> acquireCapacity(int token_capacity);
    int currentChunkLen() const;
    void resetChunkLen(int chunck_len, int max_chunk_len);

    bool isContextStream() const;
    bool isChunkStream() const;
    const rtp_llm::BufferPtr& cumLogProbs() const;

    const rtp_llm::BufferPtr& completeTokenIds();
    std::vector<int> completeTokenIdsVec(int batch_idx = 0);
    std::vector<int> commonCompleteTokenIdsVec(int batch_idx = 0);
    int currentExecuteTokenSize();
    std::vector<int> currentExecuteTokens(int batch_idx = 0) const;

    void step();

    std::vector<torch::Tensor> multimodalFeatures() const;
    int multimodalFeaturesLength() const;
    rtp_llm::BufferPtr multimodalLocations() const;
    std::vector<std::vector<int>> multimodalIntervals() const;

    int64_t getTimeoutMs() const;
    void checkTimeout();
    void setStop(ErrorCode error_code, const std::string& error_msg);
    void setStopWithoutLock(ErrorCode error_code, const std::string& error_msg);
    void stopAndRelease(ErrorCode error_code, const std::string& error_msg);
    ErrorInfo statusInfo();
    bool isDoneWithoutLock(int batch_id) const;
    void setPaused();
    bool setRunning();
    bool stoppedWithoutLock();
    virtual bool stopped();
    bool paused();
    std::string stopReason();
    virtual bool finished();
    bool running();
    bool waiting();
    bool finishedWithoutLock();
    void cancelIfNotRunning();
    void setFinishedWithoutLock();
    bool needRemoteGenerate() const;
    void setRemoteGenerate();
    size_t iterCount() const;

    const ResourceContext& resourceContext() const;
    void setKVCache(const BatchKVCacheResource &kv_cache_resource);
    void setLoss(const rtp_llm::Buffer& loss);
    void setSoftmaxProbs(const rtp_llm::Buffer& softmax_probs, int start_pos);
    const BatchKVCacheResource& kvCache() const;
    size_t maxBlockSize() const;

    bool needFinish();
    bool needFinishBySPTokens();
    void matchEosToken();
    void matchEosToken(int batch_id);
    void matchStopWordsList();
    void matchStopWordsList(int batch_id);

    void setMetricsReporter(kmonitor::MetricsReporterPtr metrics_reporter);
    void reportMetric();
    std::string debugString() const;

    void resetBeginTime(int64_t begin_time_us);

    // for test
    void setIsContextStream(bool is_context_stream);
    rtp_llm::BufferPtr getLoss();
    rtp_llm::BufferPtr getLastHiddenStates() const;
    void setLastHiddenStates(rtp_llm::BufferPtr hidden_states) {
        last_hidden_states_ = hidden_states;
    };
    rtp_llm::BufferPtr getSoftmaxProbs();
    StreamCacheResource& streamCacheResource();
    void setPerfTest(bool perf_test_);

    absl::Status releaseSequenceKVCache(size_t total_seq_len, size_t release_seq_len) {
        return stream_cache_resource_->releaseSequenceKVCache(total_seq_len, release_seq_len);
    }

    void CopyOnWrite(const GenerateStream& other_stream, bool copy_loss = true, bool share = false);

    void setReturnAllProbs(bool return_all_probs) {
        return_all_probs_ = return_all_probs;
    }

    bool getReturnAllProbs() {
        return return_all_probs_;
    }

    void beamSearchKvCacheUpdate(const rtp_llm::BufferPtr& beam_idx);
    void beamSearchLogitProcessorUpdate(const rtp_llm::BufferPtr& beam_idx);
    void updateLogitProcessorStatus(const StreamUpdateInfo& update_info);

    rtp_llm::BufferPtr generateContextPositionIds(rtp_llm::DeviceBase* device);

    void generateNextPositionId(int32_t* now_pos);

    int64_t vocabSize() const {
        return vocab_size_;
    }

    size_t outputTokenLen() const {
        return seqLength() - inputLength();
    }

    void setReturnLastHiddenStates(bool flag) {
        return_all_hidden_states_ = flag;
    }

    bool forceDisableSpRun() const {
        return generate_input_->generate_config->force_disable_sp_run;
    }

    bool disableSpRun() const {
        return numBeams() > 1 || forceDisableSpRun();
    }

    bool needReturnHiddenStates() {
        return return_all_hidden_states_;
    }

    bool waitForRemoteGenerate() {
        std::unique_lock<std::mutex> lock(*output_mutex_);
        return cv_->wait_for(lock, std::chrono::seconds(2), [this] {
            return need_remote_generate_;
        });
    }

    void setNeedRemoteGenerate(bool need_remote_generate) {
        need_remote_generate_ = need_remote_generate;
    }

    std::vector<int> getLatestTokens(size_t token_num);

    void incBatchWithPrefillTimes(int32_t times);
    void incBatchWithPrefillLen(int32_t len);


    std::shared_ptr<CompleteTokenIds> getCompleteTokenIds() const {
        return complete_token_ids_;
    }

    void setAccepedBounsToken(bool acceped_bouns_token) {
        acceped_bouns_token_ = acceped_bouns_token;
    }

    bool getAccepedBounsToken() {
        return acceped_bouns_token_;
    }

    size_t spEditSearchIndex() const {
        return sp_edit_search_index_;
    }

    void setSpEditRun(bool is_sp_edit_run) {
        sp_edit_run_ = is_sp_edit_run;
    }

    void incSpEditSearchIndex(size_t accepted_num) {
        if (sp_edit_run_) {
            sp_edit_search_index_ += accepted_num;
        }
    }

    bool spEditFirstTime() const {
        return sp_edit_first_time_;
    }

    void setSpEditFirstTime(bool sp_edit_first_time) {
        sp_edit_first_time_ = sp_edit_first_time;
    }

    void setProposeToken(int propose_token) {
        propose_token_ = propose_token;
    }

    int getProposeToken() {
        return propose_token_;
    }

    void setContainProposeToken(bool contain_propose_token) {
        contain_propose_token_ = contain_propose_token;
    }

    bool getContainProposeToken() {
        return contain_propose_token_;
    }

    void setMtpTokenIndex(int mtp_token_index) {
        mtp_token_index_ = mtp_token_index;
    }

    size_t getMtpTokenIndex() {
        return mtp_token_index_;
    }

    bool containSpOutputBuffer() {
        return sp_output_buffer_ != nullptr;
    }

    size_t getProposeStep() const {
        if (propose_stream_ && propose_stream_->sp_output_buffer_->propose_step > 0) {
            return propose_stream_->sp_output_buffer_->propose_step;
        }
        return 0;
    }

    ThinkModeLogitsProcessorPtr getThinkLogitsProcessor() {
        return think_logits_processor_ptr_;
    }

    TreeLogitsProcessorPtr getTreeLogitsProcessor() {
        return tree_logits_processor_ptr_;
    }

    void initializeLogitsProcessorList();
    std::vector<BaseLogitsProcessorPtr> getAllLogitsProcessorPtr() const {
        return logits_processor_list_;
    }

    rtp_llm::BufferPtr getProposeTokens() const {
        if (propose_stream_ && propose_stream_->sp_output_buffer_->tokens > 0) {
            return propose_stream_->sp_output_buffer_->tokens;
        }
        return nullptr;
    }

    rtp_llm::BufferPtr getScoreTokens() {
        if (score_stream_ && score_stream_->sp_output_buffer_->tokens != nullptr) {
            return score_stream_->sp_output_buffer_->tokens;
        }
        return nullptr;
    }

    void setSPOutputBuffer(SpeculativeExecutorStreamOutputPtr sp_output_buffer) {
        sp_output_buffer_ = sp_output_buffer;
    }

    SpeculativeExecutorStreamOutputPtr getSPOutputBuffer() {
        return sp_output_buffer_;
    }

    GenerateStreamPtr getProposeStream() {
        return propose_stream_;
    }

    bool containProposeStream() {
        return propose_stream_ != nullptr;
    }

    bool containScorestream() {
        return score_stream_ != nullptr;
    }

    GenerateStreamPtr getScoreStream() {
        return score_stream_;
    }

    void setProposeStream(GenerateStreamPtr stream) {
        propose_stream_ = stream;
    }

    void setScoreStream(GenerateStreamPtr stream) {
        score_stream_ = stream;
    }

public:
    struct TimeInfo {
        int64_t begin_time_us;
        int64_t wait_time_us;
        int64_t first_token_time_us;
        int64_t first_token_rt_us;
    };
    TimeInfo getTimeInfo();
    bool queryPdSep() const;

protected:
    rtp_llm::DeviceBase* device_;
    std::shared_ptr<GenerateInput>      generate_input_;
    std::shared_ptr<GenerateStatus>     generate_status_;
    std::vector<GenerateStatus>         sub_generate_status_;
    int                                 max_seq_len_;
    int64_t                             vocab_size_;
    std::shared_ptr<CompleteTokenIds>   complete_token_ids_;
    int64_t                             begin_time_us_;
    int64_t                             last_pause_us_ = 0;
    int64_t                             pause_time_us_ = 0;
    int64_t                             wait_time_us_ = 0;
    std::shared_ptr<StreamCacheResource> stream_cache_resource_;
    std::shared_ptr<bool>               is_context_stream_;
    size_t                              iter_count_             = 0;
    size_t                              last_output_pos_        = 0;
    int                                 initial_reuse_length_   = 0;
    int                                 reuse_length_           = 0;
    int                                 reuse_mm_length_        = 0;
    int                                 fallback_blocks_        = 0;
    int                                 fallback_times_         = 0;
    int                                 fallback_prefix_length_ = 0;
    // TOOD(xinfei.sxf) fix state
    bool                                done_                   = false;
    bool                                released_               = false;
    bool                                need_release_resource_  = true;

    bool                                enable_fast_gen_        = false;
    bool                                return_all_probs_       = false;
    int                                 current_chunk_len_      = 0;
    int                                 last_chunk_len_         = 0;
    int                                 max_chunk_len_          = 0;

    bool                                last_block_aligned_     = false;
    volatile bool                       need_remote_generate_   = false;
    bool                                use_cache_store_        = false;

    bool                                gen_timeline_           = false;

    // The number of times this stream has been interfered by prefills
    int32_t                             batch_with_prefill_times_ = 0;
    int32_t                             batch_with_prefill_len_ = 0;

    kmonitor::MetricsReporterPtr        metrics_reporter_;
    rtp_llm::SpecialTokens              special_tokens_;
    rtp_llm::BufferPtr                  cum_log_probs_;
    rtp_llm::BufferPtr                  all_probs_;
    rtp_llm::BufferPtr                  softmax_probs_;
    rtp_llm::BufferPtr                  loss_;
    rtp_llm::BufferPtr                  last_hidden_states_;
    int                                 loss_index_ = 0;
    std::shared_ptr<std::mutex>         output_mutex_;
    std::shared_ptr<std::condition_variable> cv_;

    GenerateStreamPtr                   propose_stream_        = nullptr;
    GenerateStreamPtr                   score_stream_          = nullptr;

    size_t                             propose_step_ = 0;
    size_t                             score_len_    = 0;
    bool                               acceped_bouns_token_   = false;
    int                                sp_edit_search_index_  = 0;
    bool                               sp_edit_first_time_    = true;
    bool                               sp_edit_run_           = false;
    int                                propose_token_         = 0;
    bool                               contain_propose_token_ = false;
    int                                mtp_token_index_       = 0;
    SpeculativeExecutorStreamOutputPtr sp_output_buffer_      = nullptr;

    bool return_all_hidden_states_ = false;

    std::optional<rtp_llm::BufferPtr>        context_position_ids_;
    PositionIdsStyle                    mm_position_ids_style_;

    rtp_llm::DataType dtype_;
    size_t hidden_size_;
    
    ThinkModeLogitsProcessorPtr         think_logits_processor_ptr_;
    TreeLogitsProcessorPtr              tree_logits_processor_ptr_;
    std::vector<BaseLogitsProcessorPtr> logits_processor_list_;

    // just for bool test
    bool perf_test_ = false;
    friend class StreamCacheResource;
};

typedef std::shared_ptr<GenerateStream> GenerateStreamPtr;

} // namespace rtp_llm
