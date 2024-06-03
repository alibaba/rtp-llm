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
    GenerateStream(const std::shared_ptr<GenerateInput>& query, const ft::GptInitParameter& params,
                   const ResourceContext& resource_context, kmonitor::MetricsReporterPtr metrics_reporter);
    virtual ~GenerateStream() {
        reportMetric();
        generate_outputs_queue_.wakeup();
    }

public:
    // Exported to python world.
    void                                cancel();
    absl::StatusOr<GenerateOutputs>     nextOutput();

    // Only used in C++ world.
    virtual bool initKVBlock();
    virtual bool incrKVBlock();
    virtual int tryReleaseKVBlock(int nums);
    virtual void releaseResource();
    int nextNeedBlockNums() const;
    int needKVCacheBlockNums() const;
    
    std::shared_ptr<GenerateInput> generateInput() const;
    std::shared_ptr<GenerateConfig>& generateConfig();
    std::optional<ft::BufferPtr>& imageEmbeddings();
    bool isStreaming() const;
    int64_t streamId() const;
    int loraId() const;
    ft::SpecialTokens specialTokens() const;
    
    int tileNum() const;
    int batchSize() const;
    int numBeams() const;
    int numReturnSequences() const;

    void updatePrefix(const std::shared_ptr<SystemPrompt>& system_prompt);
    size_t maxSeqLen() const;
    int inputLength() const;
    int seqLength() const;
    int contextLength() const;
    int prefixLength() const;
    int reuseLength() const;
    void setReuseLength(int reuse_length);

    bool isContextStream() const;
    const ft::BufferPtr& cumLogProbs() const;

    const ft::BufferPtr& completeTokenIds();
    std::vector<int> completeTokenIdsVec(int batch_id = 0);
    int currentExecuteTokenSize();
    std::vector<int> contextTokens() const;
    std::vector<int> currentExecuteTokens() const;

    void checkTimeout();
    void setStop(const std::string& err_msg);
    void stopAndRelease(const std::string& err_msg);
    bool isDoneWithoutLock(int batch_id) const;
    void setPaused();
    bool setRunning();
    bool stoppedWithoutLock();
    bool stopped();
    std::string stopReason();
    bool finished();
    void setFinishedWithoutLock();

    const ResourceContext& resourceContext() const;
    void setKVCache(const BatchKVCacheBlockAddr &kv_cache_block_addr);
    const BatchKVCacheBlockAddr& kvCache() const;
    size_t maxBlockSize() const;

    bool needFinish();
    bool needFinishBySPTokens();
    void matchEosToken();
    void matchEosToken(int batch_id);
    void matchStopWordsList();
    void matchStopWordsList(int batch_id);

    void update(ft::BufferPtr& new_tokens,
                int num_new_tokens,
                const ft::Buffer& hidden_states,
                const ft::Buffer& logits,
                const ft::Buffer& cum_log_probs,
                bool not_update_output = false);
    void updateOutput(const ft::Buffer& hidden_states,
                      const ft::Buffer& logits,
                      const ft::Buffer& cum_log_probs);

    void setMetricsReporter(kmonitor::MetricsReporterPtr metrics_reporter);
    void reportMetric();
    std::string debugString() const;

    // for test
    void setSeqLength(int seq_length);
    void setIsContextStream(bool is_context_stream);
    StreamCacheResource& streamCacheResource();

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
    int64_t                             last_pause_us_ = 0;
    int64_t                             pause_time_us_ = 0;
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
