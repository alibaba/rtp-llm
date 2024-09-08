#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
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
#include "src/fastertransformer/utils/assert_utils.h"

namespace ft = fastertransformer;

namespace rtp_llm {

// WARNGING: buffer in generate stream should all be host to avoid gpu buffer hold more time (except kv cache)
class GenerateStream {
public:
    GenerateStream(const std::shared_ptr<GenerateInput>& query, const ft::GptInitParameter& params,
                   const ResourceContext& resource_context, kmonitor::MetricsReporterPtr metrics_reporter);
    virtual ~GenerateStream() {
        reportMetric();
    }

public:
    // Exported to python world.
    void                                cancel();

    virtual absl::StatusOr<GenerateOutputs>     nextOutput() = 0;
    virtual void updateOutput(
                      const ft::BufferPtr& new_tokens,
                      const ft::BufferPtr& hidden_states,
                      const ft::BufferPtr& logits,
                      const ft::BufferPtr& cum_log_probs,
                      const ft::BufferPtr& all_probs) = 0;

    void update(const ft::BufferPtr&    new_tokens,
                int   num_new_tokens,
                const ft::BufferPtr& hidden_states,
                const ft::BufferPtr& logits,
                const ft::BufferPtr& cum_log_probs,
                const ft::BufferPtr& all_probs);


    virtual size_t scoreLen() const {
        return 1;
    }

    // Only used in C++ world.
    virtual absl::StatusOr<int> initKVBlock(int token_capacity, size_t reserve_step = 0);
    virtual absl::StatusOr<int> incrKVBlock(int token_capacity, size_t reserve_step = 0);
    virtual int tryReleaseKVBlock(int nums);
    virtual void releaseResource();
    int nextNeedBlockNums(size_t reserve_step) const;
    void incrFallbackBlock(int fallback_blocks);

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

    void updatePrefix(const std::shared_ptr<SystemPrompt>& system_prompt);
    size_t maxSeqLen() const;
    int inputLength() const;
    int seqLength() const;
    // NOTE: In generatestream, set seq len must use setSeqLength api, we need to save start_check_seq_length_
    // for checking EOS and stop words
    void setSeqLength(int seq_length);
    int commonLen() const;
    int adjustedCommonLen() const;
    void resetCommonLen();
    int seqSizePerBlock() const;
    int contextLength() const;
    int prefixLength() const;
    int inputPrefixLength() const;
    int reuseLength() const;
    void setReuseLength(int reuse_length);
    int fallbackPrefixLength() const;
    void setFallbackPrefixLength(int fallback_prefix_length);

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
    std::vector<int> contextTokens(int batch_idx) const;
    std::vector<int> currentExecuteTokens(int batch_idx) const;

    void step();

    std::optional<std::vector<torch::Tensor>>& multimodalFeatures() const;
    int multimodalFeaturesLength() const;
    std::optional<ft::BufferPtr>& multimodalLocations() const;

    void checkTimeout();
    bool checkTokenId(int token_id);
    void setStop(const std::string& err_msg, absl::StatusCode err_code = absl::StatusCode::kInternal);
    void stopAndRelease(const std::string& err_msg, absl::StatusCode err_code = absl::StatusCode::kInternal);
    bool isDoneWithoutLock(int batch_id) const;
    void setPaused();
    bool setRunning();
    bool stoppedWithoutLock();
    bool stopped();
    bool paused();
    std::string stopReason();
    bool finished();
    void setFinishedWithoutLock();
    size_t iterCount() const;

    const ResourceContext& resourceContext() const;
    void setKVCache(const BatchKVCacheBlockAddr &kv_cache_block_addr);
    void setLoss(const ft::Buffer& loss);
    const BatchKVCacheBlockAddr& kvCache() const;
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

    // for test
    void setIsContextStream(bool is_context_stream);
    ft::BufferPtr getLoss();
    StreamCacheResource& streamCacheResource();
    void setPerfTest(bool perf_test_);

    absl::Status releaseSequenceKVCache(size_t total_seq_len, size_t release_seq_len) {
        return stream_cache_resource_.releaseSequenceKVCache(total_seq_len, release_seq_len);
    }

    void CopyOnWrite(const GenerateStream& other_stream);

    void setNeedReleaseResource(bool need_release_resource) {
        need_release_resource_ = need_release_resource;
        stream_cache_resource_.setNeedReleaseResource(need_release_resource);
    }

    void setReturnAllProbs(bool return_all_probs) {
        return_all_probs_ = return_all_probs;
    }

    bool getReturnAllProbs() {
        return return_all_probs_;
    }

protected:
    ft::DeviceBase* device_;
    std::shared_ptr<GenerateInput>      generate_input_;
    GenerateStatus                      generate_status_;
    std::vector<GenerateStatus>         sub_generate_status_;
    int                                 max_seq_len_;
    int                                 seq_length_;
    int                                 start_check_seq_length_ = 0;
    int64_t                             vocab_size_;
    ft::BufferPtr                       complete_token_ids_;
    int64_t                             begin_time_us_;
    int64_t                             last_pause_us_ = 0;
    int64_t                             pause_time_us_ = 0;
    int64_t                             wait_time_us_ = 0;
    int64_t                             first_token_time_us_ = 0;
    StreamCacheResource                 stream_cache_resource_;
    SystemPromptParams                  prompt_param_;
    bool                                is_context_stream_      = true;
    size_t                              iter_count_             = 0;
    size_t                              last_output_pos_        = 0;
    int                                 reuse_length_           = 0;
    int                                 fallback_blocks_        = 0;
    int                                 fallback_times_         = 0;
    int                                 fallback_prefix_length_ = 0;
    bool                                done_                   = false;
    bool                                cancelled_              = false;
    bool                                released_               = false;
    bool                                need_release_resource_  = true;

    bool                                enable_fast_gen_        = false;
    bool                                return_all_probs_       = false;
    int                                 current_chunk_len_      = 0;
    int                                 last_chunk_len_         = 0;
    int                                 max_chunk_len_          = 0;

    int                                 common_len_             = 0;

    kmonitor::MetricsReporterPtr        metrics_reporter_       = nullptr;
    ft::SpecialTokens                   special_tokens_;
    ft::BufferPtr                       cum_log_probs_;
    ft::BufferPtr                       loss_;
    int                                 loss_index_ = 0;
    std::shared_ptr<std::mutex>         output_mutex_;
    std::shared_ptr<autil::SynchronizedQueue<GenerateOutputs>>  generate_outputs_queue_;

    // just for bool test
    bool perf_test_ = false;
    friend class StreamCacheResource;
};

typedef std::shared_ptr<GenerateStream> GenerateStreamPtr;

} // namespace rtp_llm
