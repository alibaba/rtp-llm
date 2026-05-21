#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <utility>

#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"

namespace rtp_llm {

// 用于 memory connector match
class MemoryAsyncMatchContext: public AsyncMatchContext {
public:
    explicit MemoryAsyncMatchContext(size_t                matched_block_count,
                                     int                   start_read_block_index = -1,
                                     int                   read_block_num         = 0,
                                     std::shared_ptr<void> read_copy_plan         = nullptr):
        matched_block_count_(matched_block_count),
        start_read_block_index_(start_read_block_index),
        read_block_num_(read_block_num),
        read_copy_plan_(std::move(read_copy_plan)) {}
    ~MemoryAsyncMatchContext() override = default;

public:
    void   waitDone() override;
    bool   done() const override;
    bool   success() const override;
    size_t matchedBlockCount() const override;
    int    startReadBlockIndex() const;
    int    readBlockNum() const;
    std::shared_ptr<void> readCopyPlan() const;
    void                  clearReadCopyPlan();

private:
    size_t                matched_block_count_{0};
    int                   start_read_block_index_{-1};
    int                   read_block_num_{0};
    std::shared_ptr<void> read_copy_plan_;
};

// 用于 memory connector read/write
class MemoryAsyncContext: public AsyncContext {
public:
    explicit MemoryAsyncContext(const std::function<void(bool)>& done_callback): done_callback_(done_callback) {}
    ~MemoryAsyncContext() override = default;

public:
    void waitDone() override;
    bool done() const override;
    bool success() const override;
    void setBroadcastResult(const std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>>& result);

private:
    std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>> broadcast_result_;
    std::function<void(bool)>                                               done_callback_;
    std::atomic<bool>                                                       already_done_{false};
};

}  // namespace rtp_llm
