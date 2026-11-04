#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc/server/GrpcServerWorker.h"

#include <grpc++/security/server_credentials.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc/server/GrpcServerStreamHandler.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/rpc/GigRpcServer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamClosure.h"
#include "autil/EnvUtil.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, GrpcServerWorker);

GrpcServerWorker::GrpcServerWorker(GigRpcServer &owner)
    : _owner(owner)
    , _randomGenerator(0)
    , _callBackThreadPool(nullptr) {
}

GrpcServerWorker::~GrpcServerWorker() {
    stop();
    for (auto alarm : _shutdownAlarms) {
        delete alarm;
    }
    delete[] _threadPids;
    AUTIL_LOG(INFO, "grpc server worker stopped");
}

int32_t GrpcServerWorker::getPort() const {
    return _listenPort;
}

const GigAgentPtr &GrpcServerWorker::getAgent() const {
    return _owner.getAgent();
}

std::vector<int32_t> GrpcServerWorker::getIoPids() const {
    auto count = _completionQueues.size();
    std::vector<int32_t> pids;
    pids.resize(count);
    while (true) {
        bool failed = false;
        for (size_t i = 0; i < count; i++) {
            auto pid = _threadPids[i];
            if (pid <= 0) {
                failed = true;
                break;
            }
            pids[i] = pid;
        }
        if (failed) {
            usleep(100);
        } else {
            return pids;
        }
    }
}

bool GrpcServerWorker::init(const GrpcServerDescription &desc) {
    const auto &port = desc.port;
    auto threadNum = desc.ioThreadNum;
    const auto &secureConfig = desc.secureConfig;

    if (0 == threadNum) {
        AUTIL_LOG(ERROR, "gig grpc ioThreadNum is 0");
        return false;
    }
    // TODO
    // _callBackThreadPool = new arpc::common::LockFreeThreadPool(
    //         threadNum, 1000, arpc::common::WorkItemQueueFactoryPtr(),
    //         "GigServerCB");
    // if (!_callBackThreadPool->start()) {
    //     AUTIL_LOG(ERROR, "start callback thread pool failed");
    //     return false;
    // }
    auto spec = "0.0.0.0:" + port;
    if (!secureConfig.pemRootCerts.empty()) {
        auto option = getSslOptions(secureConfig);
        _builder.AddListeningPort(spec, grpc::SslServerCredentials(option), &_listenPort);
    } else {
        _builder.AddListeningPort(spec, grpc::InsecureServerCredentials(), &_listenPort);
    }
    _builder.SetMaxReceiveMessageSize(DEFAULT_GRPC_MAX_MESSAGE_SIZE);
    _builder.SetMaxSendMessageSize(DEFAULT_GRPC_MAX_MESSAGE_SIZE);
    _builder.RegisterAsyncGenericService(&_genericService);
    if (threadNum <= 0) {
        AUTIL_LOG(ERROR, "invalid thread num %lu", threadNum);
        return false;
    }
    for (size_t i = 0; i < threadNum; i++) {
        ServerCompletionQueuePtr cq(_builder.AddCompletionQueue().release());
        ServerCompletionQueueStatusPtr cqs(new ServerCompletionQueueStatus(cq));
        _completionQueues.push_back(cqs);
    }
    _server = _builder.BuildAndStart();
    int niceValue = autil::EnvUtil::getEnv("GRPC_IO_NICE_INC", 0);
    if (!_server) {
        AUTIL_LOG(ERROR, "start grpc server failed, port [%s], threadNum [%lu]", port.c_str(),
                  threadNum);
        return false;
    } else {
        auto count = _completionQueues.size();
        _threadPids = new int32_t[count];
        for (size_t i = 0; i < count; i++) {
            _threadPids[i] = -1;
        }
        for (size_t i = 0; i < count; i++) {
            _workThreads.push_back(autil::Thread::createThread(
                std::bind(&GrpcServerWorker::workLoop, this, i, _completionQueues[i]),
                "grpc_server_io", niceValue));
        }
    }
    _desc = desc;
    AUTIL_LOG(INFO, "start grpc server success, port [%s], threadNum [%lu], nice value [%d]", port.c_str(),
              threadNum, niceValue);
    return true;
}

grpc::SslServerCredentialsOptions
GrpcServerWorker::getSslOptions(const SecureConfig &secureConfig) const {
    grpc::SslServerCredentialsOptions option;
    option.pem_root_certs = secureConfig.pemRootCerts;
    grpc::SslServerCredentialsOptions::PemKeyCertPair pair;
    pair.private_key = secureConfig.pemPrivateKey;
    pair.cert_chain = secureConfig.pemCertChain;
    option.pem_key_cert_pairs.push_back(pair);
    option.client_certificate_request = GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY;
    return option;
}

void GrpcServerWorker::stop() {
    AUTIL_LOG(DEBUG, "grpc server [%p] stop begin", this);
    if (!_server) {
        return;
    }
    if (_callBackThreadPool) {
        _callBackThreadPool->stop();
    }
    for (const auto &cqs : _completionQueues) {
        autil::ScopedReadWriteLock lock(cqs->enqueueLock, 'w');
        cqs->stopped = true;
    }
    AUTIL_LOG(DEBUG, "grpc server [%p] queue stopped", this);
    auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(100);
    _server->Shutdown(deadline);
    // _server->Shutdown();
    for (const auto &cqs : _completionQueues) {
        _shutdownAlarms.push_back(
            new ::grpc::Alarm(cqs->cq.get(), gpr_now(GPR_CLOCK_MONOTONIC), nullptr));
    }
    _completionQueues.clear();
    _workThreads.clear();
    DELETE_AND_SET_NULL(_callBackThreadPool);
    AUTIL_LOG(DEBUG, "grpc server [%p] stop end", this);
}

void GrpcServerWorker::workLoop(size_t idx, ServerCompletionQueueStatusPtr cqsPtr) {
    _threadPids[idx] = syscall(SYS_gettid);
    auto &cqs = *cqsPtr;
    auto &cq = *cqs.cq;
    AUTIL_LOG(INFO, "server work loop started, worker [%p] idx [%lu] cq [%p]", this, idx, &cq);
    {
        GrpcServerStreamHandlerPtr handler(new GrpcServerStreamHandler(this, cqsPtr));
        handler->setCallBackThreadPool(_callBackThreadPool);
        handler->init();
    }
    for (;;) {
        void *tag;
        bool ok = false;
        AUTIL_LOG(DEBUG, "[%p] grpc server loop begin", this);
        if (!cq.Next(&tag, &ok)) {
            break;
        }
        if (tag) {
            auto closure = static_cast<GigStreamClosureBase *>(tag);
            closure->run(ok);
        } else {
            // shutdown alarm
            autil::ScopedReadWriteLock lock(cqsPtr->enqueueLock, 'w');
            cqs.stopped = true;
            cq.Shutdown();
            AUTIL_LOG(INFO, "cq: %p shutdown", cqsPtr->cq.get());
        }
        AUTIL_LOG(DEBUG, "[%p] grpc server loop end", this);
    }
    AUTIL_LOG(INFO, "server work loop stopped, worker [%p] idx [%lu] cq [%p]", this, idx, &cq);
}

} // namespace multi_call
