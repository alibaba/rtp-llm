#include "aios/network/rdma/accl/RdmaMempoolFactory.h"

#include "aios/network/rdma/accl/RdmaMempool.h"

ARPC_BEGIN_NAMESPACE(arpc);

AUTIL_LOG_SETUP(arpc, RdmaMempoolFactory);

autil::ReadWriteLock         RdmaMempoolFactory::mempoolLock;
std::shared_ptr<RdmaMempool> RdmaMempoolFactory::mempool;
std::atomic<int>             RdmaMempoolFactory::aliveCount(0);

#ifdef USE_RDMA
bool RdmaMempoolFactory::enable = true;
#else
bool RdmaMempoolFactory::enable = false;
#endif

bool RdmaMempoolFactory::Init() {
    autil::ScopedWriteLock lock(mempoolLock);
    if (!enable) {
        AUTIL_LOG(WARN, "RDMA not enabled in compilation.")
        return true;
    }

    aliveCount++;
    if (mempool) {  // already inited
        return true;
    }

    RdmaMempoolConfig config;
    mempool = std::make_shared<RdmaMempool>();
    if (mempool->Init(config)) {
        return true;
    }

    // init failed
    mempool.reset();
    aliveCount--;
    AUTIL_LOG(WARN, "mem pool failed to init with config.")
    return false;
}

std::shared_ptr<RdmaMempool> RdmaMempoolFactory::GetMempool() {
    autil::ScopedReadLock lock(mempoolLock);
    return mempool;
}

void RdmaMempoolFactory::SetMempool(const std::shared_ptr<RdmaMempool>& pool) {
    autil::ScopedWriteLock lock(mempoolLock);
    mempool = pool;
}

void RdmaMempoolFactory::Stop() {
    auto mempool = GetMempool();
    if (mempool) {
        mempool->Stop();
        AUTIL_LOG(WARN, "rdma mempool stoped");
    }
}

void RdmaMempoolFactory::Shutdown() {
    autil::ScopedWriteLock lock(mempoolLock);
    if (!enable) {
        return;
    }
    aliveCount--;
    if (aliveCount.load() == 0) {
        mempool.reset();
    }
}

void RdmaMempoolFactory::EnableInit(bool e) {
    enable = e;
}

ARPC_END_NAMESPACE(arpc);