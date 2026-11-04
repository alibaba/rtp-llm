#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCServerAdapter.h"

#include <assert.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/descriptor.h>
#include <iosfwd>
#include <memory>
#include <netinet/in.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <string>
#include <unordered_set>
#include <vector>

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageCodec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/PacketArg.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCServer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCServerWorkItem.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/Tracer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/UtilFun.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"
#include "autil/EnvUtil.h"
#include "autil/LockFreeThreadPool.h"
#include "autil/WorkItem.h"

using namespace std;
using namespace anet;
ARPC_BEGIN_NAMESPACE(arpc);
ARPC_DECLARE_AND_SETUP_LOGGER(RPCServerClosure);

RPCServerAdapter::RPCServerAdapter(RPCServer *pRpcServer) {
    _pRpcServer = pRpcServer;
    atomic_set(&_clientConnNum, 0);
    pthread_rwlockattr_t attr;
    pthread_rwlockattr_init(&attr);
    pthread_rwlock_init(&_lock, &attr);
    // TODO: this config could be expose on rpc server
    _skipThreadPool = autil::EnvUtil::getEnv("ARPC_SKIP_THREAD_POOL", _skipThreadPool);
}

RPCServerAdapter::~RPCServerAdapter() { pthread_rwlock_destroy(&_lock); }

ErrorCode RPCServerAdapter::doPushWorkItem(RPCServerWorkItem *pWorkItem, CodecContext *pContext, Tracer *tracer) {
    if (__builtin_expect(!!(_pRpcServer->isStopped()), 1)) {
        delete pWorkItem;
        return ARPC_ERROR_SERVER_CLOSED;
    }
    if (pWorkItem == NULL) {
        ARPC_LOG(ERROR,
                 "new RPCServerWorkItem return NULL, "
                 "%s:%s",
                 pContext->rpcMethodDes->service()->name().c_str(),
                 pContext->rpcMethodDes->name().c_str());
        return ARPC_ERROR_NEW_NOTHROW;
    }

    pWorkItem->setArena(pContext->arena);

    if (_skipThreadPool) {
        pWorkItem->process();
        pWorkItem->destroy();
        return ARPC_ERROR_NONE;
    }

    // TODO: drop work item maybe unacceptable
    autil::ThreadPoolBasePtr threadPoolPtr = _pRpcServer->GetServiceThreadPool(pContext->rpcService);

    if (threadPoolPtr == NULL) {
        ARPC_LOG(ERROR,
                 "get service thread pool failed, "
                 "%s:%s",
                 pContext->rpcMethodDes->service()->name().c_str(),
                 pContext->rpcMethodDes->name().c_str());
        delete pWorkItem;
        return ARPC_ERROR_PUSH_WORKITEM;
    }

    tracer->SetServerQueueSize(threadPoolPtr->getItemCount());

    autil::ThreadPool::ERROR_TYPE code = threadPoolPtr->pushWorkItem((autil::WorkItem *)pWorkItem, false);

    if (code != autil::ThreadPool::ERROR_NONE) {
        ARPC_LOG(ERROR,
                 "drop work item with request"
                 "thread pool errorcode=%d "
                 "%s:%s"
                 "thread pool name: %s, queue size: %lu, thread num: %lu",
                 code,
                 pContext->rpcMethodDes->service()->name().c_str(),
                 pContext->rpcMethodDes->name().c_str(),
                 (threadPoolPtr->getName()).c_str(),
                 threadPoolPtr->getQueueSize(),
                 threadPoolPtr->getThreadNum())
        delete pWorkItem;
        return ARPC_ERROR_PUSH_WORKITEM;
    }

    return ARPC_ERROR_NONE;
}

ARPC_END_NAMESPACE(arpc);
