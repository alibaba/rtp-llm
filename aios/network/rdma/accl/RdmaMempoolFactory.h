#pragma once
#include "aios/network/arpc/arpc/CommonMacros.h"
#include "autil/Lock.h"
#include "autil/Log.h"

ARPC_BEGIN_NAMESPACE(arpc);

class RdmaMempool;

class RdmaMempoolFactory {
public:
    static std::shared_ptr<RdmaMempool> GetMempool();

    // 需要成对调用
    static bool Init();
    static void Stop();
    static void Shutdown();

    // only for test
    static void SetMempool(const std::shared_ptr<RdmaMempool>& mempool);
    static void EnableInit(bool enable);

private:
    static std::atomic<int>             aliveCount;
    static autil::ReadWriteLock         mempoolLock;
    static std::shared_ptr<RdmaMempool> mempool;
    static bool                         enable;

private:
    AUTIL_LOG_DECLARE();
};

ARPC_END_NAMESPACE(arpc);