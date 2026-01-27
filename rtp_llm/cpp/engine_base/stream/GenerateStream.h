#pragma once

#include "absl/status/statusor.h"
#include "autil/TimeUtility.h"
#include "autil/SynchronizedQueue.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/models/position_ids/PositionIdsGenerator.h"
#include <iterator>
#include <mutex>

namespace rtp_llm {

// WARNGING: buffer in generate stream should all be host to avoid gpu buffer hold more time (except kv cache)

struct StreamUpdateInfo {
    const rtp_llm::BufferPtr new_tokens;
    int                      num_new_tokens;
    const rtp_llm::BufferPtr hidden_states;
    const rtp_llm::BufferPtr logits;
    const rtp_llm::BufferPtr softmax_probs;
    const rtp_llm::BufferPtr cum_log_probs;
    const rtp_llm::BufferPtr all_probs;
    const rtp_llm::BufferPtr loss;
    const rtp_llm::BufferPtr src_batch_indices;
    // for mtp
    const rtp_llm::BufferPtr all_hidden_states;
    bool                     update_remote_generate = true;
    bool                     force_update_info      = false;
};

struct StreamSpecUpdateInfo {
    const rtp_llm::BufferPtr new_tokens;
    int                      num_new_tokens;

    int                      draft_token;
    const rtp_llm::BufferPtr draft_hidden_states;
    const rtp_llm::BufferPtr draft_token_probs;

    bool update_remote_generate = true;
    bool force_update_info      = false;
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

    // hold tensors from grpc
    std::vector<torch::Tensor> tensors_holder;
};
using SpeculativeExecutorStreamOutputPtr = std::shared_ptr<SpeculativeExecutorStreamOutput>;

class GenerateStream;

using GenerateStreamPtr = std::shared_ptr<GenerateStream>;

class GenerateStream {
public:
    GenerateStream(const std::shared_ptr<GenerateInput>& query,
                   const ModelConfig&                    model_config,
                   const RuntimeConfig&                  runtime_config,
                   const ResourceContext&                resource_context,
                   kmonitor::MetricsReporterPtr          metrics_reporter,
                   size_t                                extra_reserve_token_num = 0,
                   bool                                  pert_test               = false);
    virtual ~GenerateStream() {
        reportMetric();
        releaseResource();
    }

public:
    void setIsFakeStream(bool is_fake) {
        is_fake_stream_ = is_fake;
    }

    bool isFakeStream() const {
        return is_fake_stream_;
    }

    // Exported to python world.
    virtual void cancel();

    virtual ErrorResult<GenerateOutputs> nextOutput() = 0;
    virtual bool                         hasOutput() {
        return false;
    }

    virtual void updateOutput(const StreamUpdateInfo& update_info) = 0;
    void         update(const StreamUpdateInfo& update_info);
    void         specUpdate(const StreamSpecUpdateInfo& update_info);
    bool         updateKvCacheBlocks(const rtp_llm::BufferPtr& src_batch_indices);

    virtual size_t scoreLen() const {
        return score_len_ == 0 ? 1 : score_len_;
    }

    void setScoreLen(size_t score_len) {
        score_len_ = score_len;
    }

    // Only used in C++ world.
    int                  reuseBlockSize() const;
    void                 fakeInitKVBlock();
    virtual absl::Status initKVBlock(size_t reserve_step = 0);
    virtual absl::Status incrKVBlock(size_t reserve_step = 0);
    virtual int          tryReleaseKVBlock(int nums);
    virtual void         releaseResource();
    int                  nextNeedBlockNums(size_t reserve_step) const;
    void                 setAllowReleaseResource(bool allow_release_resource);
    void                 setNeedReleaseResource(bool need_release_resource);
    bool                 needReleaseResource() const;
    bool                 hasCacheKeys() const;
    const CacheKeysType& cacheKeys(int32_t batch_id = 0) const;

    std::shared_ptr<GenerateInput>   generateInput() const;
    std::shared_ptr<GenerateConfig>& generateConfig() const;
    std::vector<int>                 textTokensMask() const;
    bool                             isStreaming() const;
    int64_t                          streamId() const;
    int                              loraId() const;
    std::string                      adapterName() const;
    rtp_llm::SpecialTokens           specialTokens() const;

    int batchSize(int output_len) const;
    int currentBatchSize() const;
    int nextBatchSize() const;
    int maxBatchSize() const;

    int  numBeams(int output_len) const;
    int  currentNumBeams() const;
    int  nextNumBeams() const;
    int  maxNumBeams() const;
    bool hasNumBeams() const;

    bool needTilingForSampling() const;

    int    numReturnSequences() const;
    bool   calculateLoss() const;
    bool   calculateSoftmaxProbs() const;
    bool   returnLogits() const;
    bool   returnCumLogProbs() const;
    bool   genTimeline() const;
    int    profileStep() const;
    void   setGenTimeline(bool gen_timeline);
    bool   updatePrefix(const std::shared_ptr<SystemPrompt>& system_prompt);
    size_t maxSeqLen() const;
    int    inputLength() const;
    int    seqLength() const;
    // NOTE: In generatestream, set seq len must use setSeqLength api, we need to save start_check_seq_length_
    // for checking EOS and stop words
    void   setSeqLength(int seq_length);
    int    adjustedCommonLen() const;
    int    seqSizePerBlock() const;
    int    contextLength() const;
    int    prefixLength() const;
    int    inputPrefixLength() const;
    int    reuseLength() const;
    int    initialReuseLength() const;
    size_t maxTokenNum() const;
    void   setReuseLength(int reuse_length);
    void   setLocalReuseLength(int length);
    void   setRemoteReuseLength(int length);
    int    localReuseLength() const;
    int    remoteReuseLength() const;
    void   setInitialReuseLength(int initial_reuse_length);
    void   incLastOutputPos();

    bool                      isContextStream() const;
    const rtp_llm::BufferPtr& cumLogProbs() const;

    const rtp_llm::BufferPtr&         completeTokenIds();
    std::shared_ptr<CompleteTokenIds> completeTokenIdsPtr() const {
        return complete_token_ids_;
    }
    std::vector<int> completeTokenIdsVec(int batch_idx = 0);
    std::vector<int> commonCompleteTokenIdsVec(int batch_idx = 0);
    int              currentExecuteTokenSize();
    std::vector<int> currentExecuteTokens(int batch_idx = 0) const;

    void step();
    void spStep();

    std::vector<torch::Tensor>    multimodalFeatures() const;
    int                           multimodalFeaturesLength() const;
    rtp_llm::BufferPtr            multimodalLocations() const;
    std::vector<std::vector<int>> multimodalIntervals() const;

    int64_t      getTimeoutMs() const;
    void         checkTimeout();
    void         setStop(ErrorCode error_code, const std::string& error_msg);
    void         setStopWithoutLock(ErrorCode error_code, const std::string& error_msg);
    void         stopAndRelease(ErrorCode error_code, const std::string& error_msg);
    ErrorInfo    statusInfo();
    bool         isDoneWithoutLock(int batch_id) const;
    void         setPaused();
    bool         setRunning();
    bool         stoppedWithoutLock();
    virtual bool stopped();
    bool         paused();
    std::string  stopReason();
    virtual bool finished();
    bool         running();
    bool         waiting();
    bool         finishedWithoutLock();
    void         cancelIfNotRunning();
    void         setFinishedWithoutLock();
    bool         isRemoteRunningWithoutLock();
    bool         needRemoteGenerate() const;
    bool         setRemoteGenerate();
    size_t       iterCount() const;
    size_t       spIterCount() const;
    void         setSpIterCount(int sp_iter_count);

    const ResourceContext&      resourceContext() const;
    void                        setKVCache(const BatchKVCacheResource& kv_cache_resource);
    void                        setLoss(const rtp_llm::Buffer& loss);
    void                        setSoftmaxProbs(const rtp_llm::Buffer& softmax_probs, int start_pos);
    const BatchKVCacheResource& kvCache() const;
    BatchKVCacheResource&       kvCacheMutable();
    BatchKVCacheResourcePtr     kvCachePtr();
    size_t                      curBlocksNum() const;

    bool needFinish();
    bool needFinishBySPTokens();
    void matchEosToken();
    void matchEosToken(int batch_id);
    void matchStopWordsList();
    void matchStopWordsList(int batch_id);

    void        setMetricsReporter(kmonitor::MetricsReporterPtr metrics_reporter);
    void        reportMetric();
    std::string debugString() const;

    void resetBeginTime(int64_t begin_time_us);

    // for test
    void               setIsContextStream(bool is_context_stream);
    rtp_llm::BufferPtr getLoss();
    rtp_llm::BufferPtr getLastHiddenStates() const;
    void               setLastHiddenStates(rtp_llm::BufferPtr hidden_states) {
        last_hidden_states_ = hidden_states;
    };
    rtp_llm::BufferPtr   getSoftmaxProbs();
    StreamCacheResource& streamCacheResource();
    void                 setPerfTest(bool perf_test_);
    bool                 isPerfTest() const {
        return perf_test_;
    }

    void CopyOnWrite(const GenerateStream& other_stream, bool copy_loss = true, bool share = false);

    void setReturnAllProbs(bool return_all_probs) {
        return_all_probs_ = return_all_probs;
    }

    bool getReturnAllProbs() {
        return return_all_probs_;
    }

    void updateLogitProcessorMultiSeqStatus(const rtp_llm::BufferPtr& src_batch_indices);
    void updateLogitProcessorStatus(const StreamUpdateInfo& update_info);

    rtp_llm::BufferPtr generateContextPositionIds(rtp_llm::DeviceBase* device);

    void generateNextPositionId(int32_t* now_pos, rtp_llm::DeviceBase* device);

    rtp_llm::BufferPtr getContextPositionIds() const {
        return context_position_ids_.has_value() ? context_position_ids_.value() : nullptr;
    }

    void setContextPositionIds(rtp_llm::BufferPtr context_position_ids) {
        context_position_ids_ = context_position_ids;
    }

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
        return hasNumBeams() || forceDisableSpRun();
    }

    bool needReturnHiddenStates() {
        return return_all_hidden_states_;
    }

    bool waitForRemoteGenerate();

    void setNeedRemoteGenerate(bool need_remote_generate) {
        std::lock_guard<std::mutex> lock(*output_mutex_);
        need_remote_generate_ = need_remote_generate;
        cv_->notify_one();
    }

    void setNeedRemoteGenerateWithoutLock(bool need_remote_generate) {
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

    void setProposeToken(const std::vector<int>& propose_token) {
        propose_token_ = propose_token;
    }

    std::vector<int>& getProposeToken() {
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

    bool forceSpAccept() const {
        return generate_input_->generate_config->force_sp_accept;
    }

    int64_t interRequestId() const {
        return generate_input_->generate_config->inter_request_id;
    }

    std::string traceId() const {
        return generate_input_->generate_config->trace_id;
    }

    std::vector<BaseLogitsProcessorPtr> getAllLogitsProcessorPtr() const {
        return logits_processor_list_;
    }

    at::Generator getGenerator() {
        return generator_;
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

    bool reuseCache() const {
        return generate_input_->generate_config->reuse_cache;
    }

    bool enable3FS() const {
        return generate_input_->generate_config->enable_3fs;
    }

    bool enableMemoryBlockCache() const {
        return generate_input_->generate_config->enable_memory_block_cache;
    }

    void fillSubGenerateStatus(StreamState state);
    void resizeSubGenerateStatus(size_t new_size);

public:
    struct TimeInfo {
        int64_t begin_time_us;
        int64_t wait_time_us;
        int64_t first_token_time_us;
        int64_t first_token_rt_us;
    };
    TimeInfo getTimeInfo();
    bool     queryPdSep() const;

protected:
    rtp_llm::DeviceBase*                 device_;
    std::shared_ptr<GenerateInput>       generate_input_;
    std::shared_ptr<GenerateStatus>      generate_status_;
    std::vector<GenerateStatus>          sub_generate_status_;
    int                                  max_seq_len_;
    int64_t                              vocab_size_;
    std::shared_ptr<CompleteTokenIds>    complete_token_ids_;
    int64_t                              begin_time_us_;
    int64_t                              last_pause_us_ = 0;
    int64_t                              pause_time_us_ = 0;
    int64_t                              wait_time_us_  = 0;
    std::shared_ptr<StreamCacheResource> stream_cache_resource_;
    std::shared_ptr<bool>                is_context_stream_;
    size_t                               iter_count_           = 0;
    size_t                               sp_iter_count_        = 0;
    size_t                               last_output_pos_      = 0;
    int                                  initial_reuse_length_ = 0;
    int                                  reuse_length_         = 0;
    int                                  local_reuse_length_   = 0;
    int                                  remote_reuse_length_  = 0;
    int                                  reuse_mm_length_      = 0;
    // TOOD(xinfei.sxf) fix state
    bool done_                   = false;
    bool released_               = false;
    bool allow_release_resource_ = true;
    bool need_release_resource_  = false;

    bool return_all_probs_ = false;

    bool          last_block_aligned_   = false;
    volatile bool need_remote_generate_ = false;

    bool gen_timeline_ = false;

    // The number of times this stream has been interfered by prefills
    int32_t batch_with_prefill_times_ = 0;
    int32_t batch_with_prefill_len_   = 0;

    kmonitor::MetricsReporterPtr             metrics_reporter_;
    rtp_llm::SpecialTokens                   special_tokens_;
    rtp_llm::BufferPtr                       cum_log_probs_;
    rtp_llm::BufferPtr                       all_probs_;
    rtp_llm::BufferPtr                       softmax_probs_;
    rtp_llm::BufferPtr                       loss_;
    rtp_llm::BufferPtr                       last_hidden_states_ = nullptr;
    int                                      loss_index_         = 0;
    std::shared_ptr<std::mutex>              output_mutex_;
    std::shared_ptr<std::condition_variable> cv_;

    GenerateStreamPtr propose_stream_ = nullptr;
    GenerateStreamPtr score_stream_   = nullptr;

    size_t                             propose_step_         = 0;
    size_t                             score_len_            = 0;
    bool                               acceped_bouns_token_  = false;
    int                                sp_edit_search_index_ = 0;
    bool                               sp_edit_first_time_   = true;
    bool                               sp_edit_run_          = false;
    std::vector<int>                   propose_token_;
    bool                               contain_propose_token_ = false;
    int                                mtp_token_index_       = 0;
    SpeculativeExecutorStreamOutputPtr sp_output_buffer_      = nullptr;

    bool return_all_hidden_states_ = false;

    std::optional<rtp_llm::BufferPtr> context_position_ids_;
    PositionIdsStyle                  mm_position_ids_style_;

    rtp_llm::DataType dtype_;
    size_t            hidden_size_;

    std::vector<BaseLogitsProcessorPtr> logits_processor_list_;
    at::Generator                       generator_;

    // just for bool test
    bool perf_test_ = false;
    friend class StreamCacheResource;
    bool is_fake_stream_ = false;
};

typedef std::shared_ptr<GenerateStream> GenerateStreamPtr;

}  // namespace rtp_llm
