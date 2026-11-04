#pragma once

#include <memory>
#include <mutex>

#include "autil/Log.h"
#include "autil/LoopThread.h"
#include "autil/NoCopyable.h"

namespace anet {
class Transport;
}

namespace arpc {

class ANetRPCChannelManager;
class ANetRPCChannel;
class ClientMetricReporter;

class PooledChannelManager : public autil::NoCopyable {
public:
    PooledChannelManager(int ioThreadNum = 1);
    virtual ~PooledChannelManager();

private:
    PooledChannelManager(const PooledChannelManager &);
    PooledChannelManager &operator=(const PooledChannelManager &);

public:
    virtual std::shared_ptr<ANetRPCChannel> getRpcChannel(const std::string &addr);
    void setTimeout(int timeout) { _timeout = timeout; }
    void setQueueSize(int queueSize) { _queueSize = queueSize; }

    anet::Transport *GetTransport();
    void SetMetricReporter(const std::shared_ptr<ClientMetricReporter> &metricReporter);

private:
    std::shared_ptr<ANetRPCChannel> createRpcChannel(const std::string &spec);
    void cleanBrokenChannelLoop();

private:
    static std::string convertToSpec(const std::string &addr);

private:
    std::mutex _mu;
    int _timeout = 5000;
    size_t _queueSize = 50ul;
    std::unique_ptr<anet::Transport> _transport;
    std::unique_ptr<arpc::ANetRPCChannelManager> _rpcChannelManager;
    std::map<std::string, std::shared_ptr<ANetRPCChannel>> _rpcChannelPool;
    autil::LoopThreadPtr _loopThreadPtr;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace arpc
