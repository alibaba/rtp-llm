#pragma once

#include <queue>
#include "maga_transformer/cpp/dataclass/MagaInitParameter.h"
#include "maga_transformer/cpp/schedulers/SchedulerBase.h"
#include "maga_transformer/cpp/embedding_engine/EmbeddingStream.h"

namespace rtp_llm {

class EmbeddingScheduler {
public:
    explicit EmbeddingScheduler(const MagaInitParams& config);

    ~EmbeddingScheduler();
    
    absl::Status enqueue(EmbeddingStreamPtr stream);

    absl::StatusOr<std::list<EmbeddingStreamPtr>> scheduleNew();

    absl::Status stop();

public:
    // for test
    int waitingStreamsSize();
    int runningStreamsSize();

private:
    GptInitParameter&                   config_;
    std::list<EmbeddingStreamPtr>       waiting_streams_;
    std::list<EmbeddingStreamPtr>       running_streams_;
    std::atomic<bool>                   stop_               = false;
    std::mutex                          lock_;
    std::condition_variable             cond_;

};

}  // namespace rtp_llm
