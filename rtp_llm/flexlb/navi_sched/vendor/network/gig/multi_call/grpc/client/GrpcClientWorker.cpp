#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc/client/GrpcClientWorker.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamClosure.h"
#include "autil/EnvUtil.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, GrpcClientWorker);

GrpcClientWorker::GrpcClientWorker(size_t threadNum) {
    int niceValue = autil::EnvUtil::getEnv("GRPC_IO_NICE_INC", 0);
    for (size_t i = 0; i < threadNum; i++) {
        CompletionQueueStatusPtr cqs(
            new CompletionQueueStatus(CompletionQueuePtr(new grpc::CompletionQueue())));
        _completionQueues.push_back(cqs);
        _workThreads.push_back(autil::Thread::createThread(
            std::bind(&GrpcClientWorker::workLoop, this, i, cqs), "grpc_client_io", niceValue));
    }
    AUTIL_LOG(INFO, "grpc client worker [%p] started, threads [%lu], nice value [%d].", this,
              threadNum, niceValue);
}

GrpcClientWorker::~GrpcClientWorker() {
    stop();
    for (auto alarm : _shutdownAlarms) {
        delete alarm;
    }
}

void GrpcClientWorker::stop() {
    for (const auto &cqs : _completionQueues) {
        autil::ScopedReadWriteLock lock(cqs->enqueueLock, 'w');
        cqs->stopped = true;
    }
    for (const auto &cqs : _completionQueues) {
        _shutdownAlarms.push_back(
            new ::grpc::Alarm(cqs->cq.get(), gpr_now(GPR_CLOCK_MONOTONIC), nullptr));
    }
    _completionQueues.clear();
    _workThreads.clear();
    AUTIL_LOG(INFO, "grpc client worker [%p] stopped", this);
}

const CompletionQueueStatusPtr &GrpcClientWorker::getCompletionQueue(size_t allocId) {
    size_t index = allocId % _completionQueues.size();
    return _completionQueues[index];
}

void GrpcClientWorker::workLoop(size_t idx, CompletionQueueStatusPtr cqsPtr) {
    auto &cq = *cqsPtr->cq;
    AUTIL_LOG(INFO, "client work loop started, worker [%p] idx [%lu] cq [%p]", this, idx, &cq);
    for (;;) {
        void *tag;
        bool ok = false;
        // AUTIL_LOG(INFO, "worker cq: %p", &cq);
        if (!cq.Next(&tag, &ok)) {
            break;
        }
        // AUTIL_LOG(INFO, "next %s", ok ? "true" : "false" );
        if (tag) {
            auto closure = static_cast<GigStreamClosureBase *>(tag);
            closure->run(ok);
        } else {
            // shutdown alarm
            autil::ScopedReadWriteLock lock(cqsPtr->enqueueLock, 'w');
            cqsPtr->stopped = true;
            cqsPtr->cq->Shutdown();
            AUTIL_LOG(INFO, "cq: %p shutdown", cqsPtr->cq.get());
        }
    }
    AUTIL_LOG(INFO, "client work loop stopped, worker [%p] idx [%lu] cq [%p]", this, idx, &cq);
}

} // namespace multi_call
