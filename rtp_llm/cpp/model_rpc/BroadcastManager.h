#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

template<typename RequestPB, typename ResponsePB>
class BroadcastResult {
public:
    using DoneCallback = std::function<void()>;

    struct WorkerRpcContext {
        std::shared_ptr<RpcService::Stub>    stub;
        std::shared_ptr<grpc::ClientContext> client_context;
        RequestPB                            request;
        ResponsePB                           response;
        grpc::Status                         status;
        std::string                          server_addr;
        int                                  timeout_ms;
    };

public:
    explicit BroadcastResult(const std::vector<std::shared_ptr<WorkerRpcContext>>& worker_rpc_contexts):
        worker_contexts_(worker_rpc_contexts), finished_(worker_rpc_contexts.size(), false) {}
    ~BroadcastResult() = default;

public:
    bool done() const {
        return already_done_.load(std::memory_order_acquire);
    }

    bool waitDone(int timeout_ms) {
        std::unique_lock<std::mutex> lock(wait_done_mutex_);
        const auto completed = [this] { return already_done_.load(std::memory_order_acquire); };
        if (timeout_ms > 0) {
            if (!wait_done_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), completed)) {
                return false;
            }
        } else {
            wait_done_cv_.wait(lock, completed);
        }
        const std::string fatal_error = fatal_error_;
        lock.unlock();
        if (!fatal_error.empty()) {
            RTP_LLM_FAIL("%s", fatal_error.c_str());
        }
        return true;
    }

    void waitDone() {
        (void)waitDone(/*timeout_ms=*/0);
    }

    bool success() const {
        return all_request_success_.load(std::memory_order_acquire);
    }

    void onDone(DoneCallback callback) {
        if (!callback) {
            return;
        }
        bool run_now = false;
        {
            std::lock_guard<std::mutex> lock(wait_done_mutex_);
            if (already_done_.load(std::memory_order_acquire)) {
                run_now = true;
            } else {
                callbacks_.push_back(std::move(callback));
            }
        }
        if (run_now) {
            callback();
        }
    }

    std::vector<ResponsePB> responses() const {
        std::unique_lock<std::mutex> lock(wait_done_mutex_);
        std::vector<ResponsePB>      responses;
        responses.reserve(worker_contexts_.size());
        for (const auto& worker_rpc_context : worker_contexts_) {
            responses.push_back(worker_rpc_context->response);
        }
        return responses;
    }

    void finishRank(size_t rank, bool cq_event_ok) {
        std::vector<DoneCallback> callbacks;
        {
            std::lock_guard<std::mutex> lock(wait_done_mutex_);
            if (already_done_.load(std::memory_order_relaxed)) {
                return;
            }
            if (rank >= worker_contexts_.size()) {
                fatal_error_ = "broadcast rpc cq tag rank out of range: rank=" + std::to_string(rank);
                finishLocked(callbacks);
            } else if (finished_[rank]) {
                return;
            } else {
                finished_[rank] = true;
                ++finished_count_;
                const auto& ctx = worker_contexts_[rank];
                if (!cq_event_ok) {
                    fatal_error_ = "broadcast rpc cq event failed, rank=" + std::to_string(rank)
                                   + " addr=" + ctx->server_addr;
                } else if (ctx->status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
                    fatal_error_ = "broadcast rpc timeout, timeout_ms=" + std::to_string(ctx->timeout_ms)
                                   + " rank=" + std::to_string(rank) + " err="
                                   + std::to_string(ctx->status.error_code()) + "(" + ctx->status.error_message()
                                   + ") addr=" + ctx->server_addr;
                } else if (!ctx->status.ok()) {
                    RTP_LLM_LOG_WARNING("broadcast rpc failed, rank=%zu err=%d(%s) addr=%s",
                                        rank,
                                        ctx->status.error_code(),
                                        ctx->status.error_message().c_str(),
                                        ctx->server_addr.c_str());
                    grpc_status_failure_seen_ = true;
                }
                if (finished_count_ == static_cast<int>(worker_contexts_.size())) {
                    finishLocked(callbacks);
                }
            }
        }
        wait_done_cv_.notify_all();
        for (auto& callback : callbacks) {
            callback();
        }
    }

private:
    friend class BroadcastManager;

    void finishLocked(std::vector<DoneCallback>& callbacks) {
        finished_count_ = static_cast<int>(worker_contexts_.size());
        all_request_success_.store(fatal_error_.empty() && !grpc_status_failure_seen_, std::memory_order_release);
        already_done_.store(true, std::memory_order_release);
        callbacks.swap(callbacks_);
    }

    std::vector<std::shared_ptr<WorkerRpcContext>> worker_contexts_;
    std::vector<bool>                              finished_;
    int                                            finished_count_{0};
    std::atomic<bool>                              already_done_{false};
    std::atomic<bool>                              all_request_success_{false};
    bool                                           grpc_status_failure_seen_{false};
    std::string                                    fatal_error_;
    mutable std::mutex                             wait_done_mutex_;
    std::condition_variable                        wait_done_cv_;
    std::vector<DoneCallback>                      callbacks_;
};

class BroadcastManager {
public:
    explicit BroadcastManager(const std::vector<std::string>& worker_addrs): worker_addrs_(worker_addrs) {}
    ~BroadcastManager() {
        if (completion_queue_) {
            completion_queue_->Shutdown();
        }
        if (completion_thread_.joinable()) {
            completion_thread_.join();
        }
        rpc_pool_.reset();
    }

public:
    bool init() {
        if (worker_addrs_.empty()) {
            RTP_LLM_LOG_WARNING("init failed, worker_addrs is empty");
            return false;
        }

        rpc_pool_ = std::make_shared<RPCPool>();
        completion_queue_ = std::make_unique<grpc::CompletionQueue>();
        completion_thread_ = std::thread([this] { pollCompletions(); });
        return true;
    }

    template<typename RequestPB, typename ResponsePB, typename RpcCall>
    std::shared_ptr<BroadcastResult<RequestPB, ResponsePB>>
    broadcast(const std::vector<RequestPB>& requests, int timeout_ms, const RpcCall& rpc_call) const {
        const auto worker_size = worker_addrs_.size();
        if (requests.size() != worker_size) {
            RTP_LLM_LOG_WARNING(
                "broadcast failed, requests size mismatch, req: %zu, worker size: %zu", requests.size(), worker_size);
            return nullptr;
        }
        if (!rpc_pool_ || !completion_queue_) {
            RTP_LLM_LOG_WARNING("broadcast failed, manager is not initialized");
            return nullptr;
        }

        using CtxT = typename BroadcastResult<RequestPB, ResponsePB>::WorkerRpcContext;
        using ResultT = BroadcastResult<RequestPB, ResponsePB>;
        std::vector<std::shared_ptr<CtxT>> contexts(worker_size);
        const auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms);

        for (int rank = 0; rank < worker_size; ++rank) {
            const auto& addr        = worker_addrs_[rank];
            auto        conn_status = rpc_pool_->getConnection(addr);
            if (!conn_status.ok()) {
                RTP_LLM_LOG_WARNING("broadcast: getConnection failed rank=%d addr=%s", rank, addr.c_str());
                return nullptr;
            }

            contexts[rank]      = std::make_shared<CtxT>();
            auto& ctx           = contexts.at(rank);
            ctx->stub           = conn_status.value().stub;
            ctx->request        = requests.at(rank);
            ctx->server_addr    = addr;
            ctx->timeout_ms     = timeout_ms;
            ctx->client_context = std::make_shared<grpc::ClientContext>();
            ctx->client_context->set_deadline(deadline);
        }

        auto result = std::make_shared<ResultT>(std::move(contexts));
        for (int rank = 0; rank < worker_size; ++rank) {
            auto& ctx = result->worker_contexts_.at(rank);
            try {
                auto reader = rpc_call(ctx->stub, ctx->client_context, ctx->request, completion_queue_.get());
                RTP_LLM_CHECK_WITH_INFO(reader != nullptr, "broadcast rpc call returned null reader, rank=%d", rank);
                auto tag = std::make_unique<CompletionTag<ResultT>>(result, static_cast<size_t>(rank));
                reader->Finish(&ctx->response, &ctx->status, tag.get());
                (void)tag.release();
            } catch (const std::exception& error) {
                RTP_LLM_FAIL("broadcast rpc dispatch failed, rank=%d err=%s", rank, error.what());
            } catch (...) {
                RTP_LLM_FAIL("broadcast rpc dispatch failed, rank=%d unknown error", rank);
            }
        }

        return result;
    }

    size_t workerNum() const {
        return worker_addrs_.size();
    }

private:
    struct CompletionTagBase {
        virtual ~CompletionTagBase() = default;
        virtual void complete(bool ok) = 0;
    };

    template<typename ResultT>
    struct CompletionTag final: CompletionTagBase {
        CompletionTag(std::shared_ptr<ResultT> result, size_t rank): result(std::move(result)), rank(rank) {}

        void complete(bool ok) override {
            result->finishRank(rank, ok);
        }

        std::shared_ptr<ResultT> result;
        size_t                   rank;
    };

    void pollCompletions() {
        void* raw_tag = nullptr;
        bool  ok      = false;
        while (completion_queue_->Next(&raw_tag, &ok)) {
            std::unique_ptr<CompletionTagBase> tag(static_cast<CompletionTagBase*>(raw_tag));
            try {
                tag->complete(ok);
            } catch (const std::exception& error) {
                RTP_LLM_LOG_ERROR("broadcast completion callback failed: %s", error.what());
            } catch (...) {
                RTP_LLM_LOG_ERROR("broadcast completion callback failed: unknown error");
            }
        }
    }

    std::vector<std::string> worker_addrs_;
    std::shared_ptr<RPCPool>                  rpc_pool_;
    std::unique_ptr<grpc::CompletionQueue>    completion_queue_;
    std::thread                               completion_thread_;
};

}  // namespace rtp_llm
