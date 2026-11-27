#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"

namespace rtp_llm {
namespace {

using RpcResult = BroadcastResult<FunctionRequestPB, FunctionResponsePB>;

class RtpRankBroadcast final: public RankTransferBroadcast {
public:
    explicit RtpRankBroadcast(size_t blocks): block_count_(blocks) {}
    void attach(std::shared_ptr<RpcResult> result) noexcept {
        result_ = std::move(result);
    }
    void waitDone() override {
        result_->waitDone(0, false);
    }
    void onDone(DoneCallback callback) override {
        result_->onDone(std::move(callback));
    }
    bool done() const override {
        return result_->done();
    }
    std::string failureSummary() const override {
        return result_->success() ? std::string{} : "multi-rank transfer RPC failed: " + result_->failureSummary();
    }
    std::vector<int64_t> completionTimesUs() const override {
        return result_->completionTimesUs();
    }
    std::vector<int64_t> completionWallTimesUs() const override {
        return result_->completionWallTimesUs();
    }
    std::vector<RankTransferReply> replies() const override {
        const auto                     snapshots = result_->rankSnapshots();
        std::vector<RankTransferReply> output;
        output.reserve(snapshots.size());
        for (const auto& snapshot : snapshots) {
            TransferRankSnapshot rank_snapshot;
            rank_snapshot.rank           = snapshot.rank;
            rank_snapshot.server_address = snapshot.server_address;
            switch (snapshot.dispatch_state) {
                case BroadcastDispatchState::NOT_SUBMITTED:
                    rank_snapshot.dispatch_state = TransferRpcDispatchState::NOT_SUBMITTED;
                    break;
                case BroadcastDispatchState::POSSIBLY_SUBMITTED:
                    rank_snapshot.dispatch_state = TransferRpcDispatchState::POSSIBLY_SUBMITTED;
                    break;
                case BroadcastDispatchState::SUBMITTED:
                    rank_snapshot.dispatch_state = TransferRpcDispatchState::SUBMITTED;
                    break;
            }
            rank_snapshot.dispatch_error = snapshot.dispatch_error;
            rank_snapshot.cq_received    = snapshot.cq_received;
            rank_snapshot.cq_ok          = snapshot.cq_ok;
            rank_snapshot.grpc_code      = snapshot.grpc_code;
            rank_snapshot.grpc_message   = snapshot.grpc_message;
            if (!snapshot.cq_received || !snapshot.cq_ok || !snapshot.grpc_code
                || *snapshot.grpc_code != grpc::StatusCode::OK || !snapshot.response) {
                output.push_back({std::move(rank_snapshot), std::nullopt});
                continue;
            }
            rank_snapshot.response_received = true;
            const auto&    response         = *snapshot.response;
            TransferResult rank_result{ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "invalid transfer response"),
                                       CacheCopyError::RPC_FAILED};
            if (response.has_mem_response()) {
                const auto& memory_response = response.mem_response();
                if (memory_response.has_error_info()) {
                    rank_snapshot.worker_error_code    = memory_response.error_info().error_code();
                    rank_snapshot.worker_error_message = memory_response.error_info().error_message();
                }
                bool valid = MemoryOperationResponsePB::Code_IsValid(memory_response.code())
                             && static_cast<size_t>(memory_response.block_errors_size()) == block_count_;
                if (!MemoryOperationResponsePB::Code_IsValid(memory_response.code())) {
                    rank_snapshot.response_error =
                        "invalid worker copy error code: " + std::to_string(memory_response.code());
                } else {
                    rank_snapshot.worker_copy_error = BlockTransferRequestConverter::decodeCode(memory_response.code());
                    if (static_cast<size_t>(memory_response.block_errors_size()) != block_count_) {
                        rank_snapshot.response_error =
                            "worker block error count mismatch: expected=" + std::to_string(block_count_)
                            + " actual=" + std::to_string(memory_response.block_errors_size());
                    }
                }
                for (int index = 0; valid && index < memory_response.block_errors_size(); ++index) {
                    valid = MemoryOperationResponsePB::Code_IsValid(memory_response.block_errors(index));
                    if (!valid) {
                        rank_snapshot.response_error =
                            "invalid worker block error code at index=" + std::to_string(index)
                            + " code=" + std::to_string(memory_response.block_errors(index));
                    }
                }
                if (valid) {
                    try {
                        rank_result =
                            BlockTransferRequestConverter::decodeTransferResult(memory_response, block_count_);
                        if (rank_result.ok() && !rank_result.buffer_safe) {
                            rank_result.status = ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN,
                                                           "worker success has no buffer-safety proof");
                            rank_result.copy_error = CacheCopyError::RPC_FAILED;
                        }
                        rank_snapshot.block_errors   = rank_result.block_errors;
                        rank_snapshot.buffer_safe    = rank_result.buffer_safe;
                        rank_snapshot.response_valid = true;
                    } catch (const std::exception& error) {
                        rank_snapshot.response_error = "transfer response decode failed: " + std::string(error.what());
                        rank_result = {ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, rank_snapshot.response_error),
                                       CacheCopyError::RPC_FAILED};
                    } catch (...) {
                        rank_snapshot.response_error = "unknown transfer response decode failure";
                        rank_result                  = {ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN,
                                                 "unknown transfer response decode failure"),
                                                        CacheCopyError::RPC_FAILED};
                    }
                }
            } else {
                rank_snapshot.response_error = "transfer response contains no memory response";
            }
            output.push_back({std::move(rank_snapshot), std::move(rank_result)});
        }
        return output;
    }

private:
    size_t                     block_count_;
    std::shared_ptr<RpcResult> result_;
};

RankTransferTransport rtpTransport(std::vector<GroupSetPtr> groups, std::shared_ptr<BroadcastManager> manager) {
    RankTransferTransport transport;
    if (!manager)
        return transport;
    transport.addresses = manager->workerAddresses();
    transport.prepare   = [manager,
                         groups = std::move(groups)](const TransferTask& task) -> std::optional<RankTransferRequest> {
        FunctionRequestPB message;
        if (!BlockTransferRequestConverter::encodeTransfer(*message.mutable_mem_request(), task, groups)) {
            return std::nullopt;
        }
        RankTransferRequest request;
        for (const auto& address : manager->workerAddresses()) {
            request.peers.push_back({request.peers.size(), address});
        }
        request.send =
            [manager, message = std::move(message), id = task.requestId(), blocks = task.descriptors().size()](
                std::chrono::milliseconds remaining) mutable -> std::shared_ptr<RankTransferBroadcast> {
            const auto timeout = static_cast<int>(remaining.count());
            message.mutable_mem_request()->set_timeout_ms(timeout);
            auto adapter = std::make_shared<RtpRankBroadcast>(blocks);
            auto result  = manager->broadcast<FunctionRequestPB, FunctionResponsePB>(
                message,
                timeout,
                [](const std::shared_ptr<RpcService::Stub>&    stub,
                   const std::shared_ptr<grpc::ClientContext>& context,
                   const FunctionRequestPB&                    rpc_request,
                   grpc::CompletionQueue*                      queue) {
                    return stub->AsyncExecuteFunction(context.get(), rpc_request, queue);
                },
                id,
                MemoryOperationRequestPB_CopyDirection_Name(message.mem_request().copy_direction()),
                false);
            if (!result)
                return nullptr;
            adapter->attach(std::move(result));
            return adapter;
        };
        return request;
    };
    return transport;
}
}  // namespace

MultiRankBlockTransferEngine::MultiRankBlockTransferEngine(std::vector<GroupSetPtr>                       groups,
                                                           std::shared_ptr<BroadcastManager>              manager,
                                                           std::shared_ptr<BlockTreeCacheMetricsReporter> reporter):
    MultiRankBlockTransferEngine(groups, rtpTransport(groups, std::move(manager)), std::move(reporter)) {}
}  // namespace rtp_llm
