#pragma once

#include "absl/status/statusor.h"
#include "autil/TimeUtility.h"
#include "autil/SynchronizedQueue.h"
#include "kmonitor/client/MetricsReporter.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "maga_transformer/cpp/stream/StreamCacheResource.h"
#include "maga_transformer/cpp/stream/CompleteTokenIds.h"
#include "maga_transformer/cpp/system_prompt/SystemPrompt.h"
#include "maga_transformer/cpp/position_ids_generator/PositionIdsGenerator.h"

namespace ft = fastertransformer;

namespace rtp_llm {

// WARNGING: buffer in generate stream should all be host to avoid gpu buffer hold more time (except kv cache)

struct StreamUpdateInfo {
    const ft::BufferPtr new_tokens;
    int                 num_new_tokens;
    const ft::BufferPtr hidden_states;
    const ft::BufferPtr logits;
    const ft::BufferPtr softmax_probs;
    const ft::BufferPtr cum_log_probs;
    const ft::BufferPtr all_probs;
    const ft::BufferPtr loss;
};

class GenerateStream {
public:
    GenerateStream(const std::shared_ptr<GenerateInput>& query, const ft::GptInitParameter& params,
                   const ResourceContext& resource_context, kmonitor::MetricsReporterPtr metrics_reporter);
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
    ft::SpecialTokens specialTokens() const;

    int tileNum() const;
    int batchSize() const;
    int numBeams() const;
    int numReturnSequences() const;
    bool calculateLoss() const;
    bool calculateSoftmaxProbs() const;
    bool returnLogits() const;
    bool returnCumLogProbs() const;

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
    const ft::BufferPtr& cumLogProbs() const;

    const ft::BufferPtr& completeTokenIds();
    std::vector<int> completeTokenIdsVec(int batch_idx = 0);
    std::vector<int> commonCompleteTokenIdsVec(int batch_idx = 0);
    int currentExecuteTokenSize();
    std::vector<int> currentExecuteTokens(int batch_idx = 0) const;

    void step();

    std::vector<torch::Tensor> multimodalFeatures() const;
    int multimodalFeaturesLength() const;
    ft::BufferPtr multimodalLocations() const;
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
    void setLoss(const ft::Buffer& loss);
    void setSoftmaxProbs(const ft::Buffer& softmax_probs, int start_pos);
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
    ft::BufferPtr getLoss();
    ft::BufferPtr getSoftmaxProbs();
    StreamCacheResource& streamCacheResource();
    void setPerfTest(bool perf_test_);

    absl::Status releaseSequenceKVCache(size_t total_seq_len, size_t release_seq_len) {
        return stream_cache_resource_.releaseSequenceKVCache(total_seq_len, release_seq_len);
    }

    void CopyOnWrite(const GenerateStream& other_stream, bool copy_loss = true);

    void setReturnAllProbs(bool return_all_probs) {
        return_all_probs_ = return_all_probs;
    }

    bool getReturnAllProbs() {
        return return_all_probs_;
    }

    void setAccepedBounsToken(bool acceped_bouns_token) {
        acceped_bouns_token_ = acceped_bouns_token;
    }

    bool getAccepedBounsToken() {
        return acceped_bouns_token_;
    }

    void beamSearchKvCacheUpdate(ft::BufferPtr beam_idx);

    ft::BufferPtr generateContextPositionIds(ft::DeviceBase* device);

    void generateNextPositionId(int32_t* now_pos);

    int64_t vocabSize() const {
        return vocab_size_;
    }

    size_t outputTokenLen() const {
        return seqLength() - inputLength();
    }

    size_t spEditSearchIndex() const {
        return sp_edit_search_index_;
    }

    void incSpEditSearchIndex(size_t accepted_num) {
        if (sp_edit_run_) {
            sp_edit_search_index_ += accepted_num;
        }
    }

    void setSpEditRun(bool is_sp_edit_run) {
        sp_edit_run_ = is_sp_edit_run;
    }

    bool spEditFirstTime() const {
        return sp_edit_first_time_;
    }

    void setSpEditFirstTime(bool sp_edit_first_time) {
        sp_edit_first_time_ = sp_edit_first_time;
    }

    bool forceDisableSpRun() const {
        return generate_input_->generate_config->force_disable_sp_run;
    }

    bool disableSpRun() const {
        return numBeams() > 1 || forceDisableSpRun();
    }

    std::vector<int> getLatestTokens(size_t token_num);

    void incBatchWithPrefillTimes(int32_t times);
    void incBatchWithPrefillLen(int32_t len);


    const std::vector<StreamThinkInfo> streamThinkInfo() {
        return think_infos_;
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
    ft::DeviceBase* device_;
    std::shared_ptr<GenerateInput>      generate_input_;
    GenerateStatus                      generate_status_;
    std::vector<GenerateStatus>         sub_generate_status_;
    int                                 max_seq_len_;
    bool                                acceped_bouns_token_ = false;
    int64_t                             vocab_size_;
    std::shared_ptr<CompleteTokenIds>   complete_token_ids_;
    int64_t                             begin_time_us_;
    int64_t                             last_pause_us_ = 0;
    int64_t                             pause_time_us_ = 0;
    int64_t                             wait_time_us_ = 0;
    StreamCacheResource                 stream_cache_resource_;
    bool                                is_context_stream_      = true;
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

    int                                 sp_edit_search_index_   = 0;
    bool                                sp_edit_first_time_     = true;
    bool                                sp_edit_run_            = false;

    bool                                last_block_aligned_     = false;
    bool                                need_remote_generate_   = false;
    bool                                use_cache_store_        = false;

    // The number of times this stream has been interfered by prefills
    int32_t                             batch_with_prefill_times_ = 0;
    int32_t                             batch_with_prefill_len_ = 0;

    kmonitor::MetricsReporterPtr        metrics_reporter_;
    ft::SpecialTokens                   special_tokens_;
    ft::BufferPtr                       cum_log_probs_;
    ft::BufferPtr                       all_probs_;
    ft::BufferPtr                       softmax_probs_;
    ft::BufferPtr                       loss_;
    int                                 loss_index_ = 0;
    std::shared_ptr<std::mutex>         output_mutex_;

    std::optional<ft::BufferPtr>        context_position_ids_;
    PositionIdsStyle                    mm_position_ids_style_;

    std::vector<StreamThinkInfo>        think_infos_;

    // just for bool test
    bool perf_test_ = false;
    friend class StreamCacheResource;
};

typedef std::shared_ptr<GenerateStream> GenerateStreamPtr;

} // namespace rtp_llm
