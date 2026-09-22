#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorLog.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// Shared by DecodeRpcServer and its CPU-only lifecycle tests. WorkerContext must
// retain the client context, submitted/completed flags and final grpc::Status.
// Keep this guard alive only while the referenced contexts, queues and key live.
template<typename WorkerContext>
struct KvWorkerRpcCleanup {
    std::vector<WorkerContext>&            contexts;
    std::vector<grpc::CompletionQueue>&    queues;
    std::chrono::system_clock::time_point deadline;
    const std::string&                    request_key;
    bool                                  has_allocation;

    ~KvWorkerRpcCleanup() {
        // Keep the caller's allocation alive until workers confirm that
        // writes ended. Cancelling these RPCs would discard that confirmation.
        if (!has_allocation) {
            for (auto& context : contexts) {
                try {
                    RTP_LLM_LOG_INFO("[KV_RPC] RPC_CANCEL request=[%s] context=%p reason=cleanup_without_allocation",
                                     request_key.c_str(), context.client_context.get());
                } catch (...) {}
                context.client_context->TryCancel();
            }
        }
        for (auto& queue : queues) {
            queue.Shutdown();
        }
        for (auto& queue : queues) {
            void* tag;
            bool  ok;
            auto  result = queue.AsyncNext(&tag, &ok, deadline);
            while (result == grpc::CompletionQueue::GOT_EVENT) {
                contexts[reinterpret_cast<uintptr_t>(tag)].completed = ok;
                result = queue.AsyncNext(&tag, &ok, deadline);
            }
            if (result != grpc::CompletionQueue::SHUTDOWN) {
                RTP_LLM_LOG_ERROR("request [%s] KV worker retirement timed out; cannot release allocation; "
                                  "[KV_RPC] RPC_CLEANUP_ABORT reason=cq_retirement_timeout has_allocation=%d",
                                  request_key.c_str(), has_allocation);
                std::abort();
            }
        }
        for (size_t rank = 0; rank < contexts.size(); ++rank) {
            const auto& context = contexts[rank];
            if (has_allocation && context.submitted && (!context.completed || !context.status.ok())) {
                if (context.completed) {
                    logKvRpcFailure(request_key, "cleanup", rank, context.status, *context.client_context);
                }
                RTP_LLM_LOG_ERROR(
                    "request [%s] KV worker %zu completion unconfirmed: %s; cannot release allocation; "
                    "[KV_RPC] RPC_CLEANUP_ABORT context=%p has_allocation=1 submitted=1 completed=%d grpc_code=%d",
                    request_key.c_str(), rank, context.status.error_message().c_str(), context.client_context.get(),
                    context.completed, context.completed ? context.status.error_code() : -1);
                std::abort();
            }
        }
    }
};

}  // namespace rtp_llm
