#pragma once
#include <assert.h>
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/flexlb/navi_sched/vendor/autil/autil/Lock.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/CommonMacros.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/MessageCodec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCServerAdapter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/RPCServerWorkItem.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/ThreadPoolDescriptor.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/proto/rpc_extensions.pb.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/util/Log.h"
#include "autil/Lock.h"
#include "autil/LockFreeThreadPool.h"

ARPC_BEGIN_NAMESPACE(arpc);

/**
 * Default time span to break idle connections.
 */
const int MAX_IDLE_TIME = 900000; // 15 minutes

/**
 * Default backlog for server's listen socket
 */
const int LISTEN_BACKLOG = 256;

/**
 * @ingroup ServerClasses
 * manage rpc call map
 * provider listen interface for user
 * manage thread pool
 * @attention It is not thread safe. It is supposed that only one thread would
 * be responsible to create and maintain the lifecycle of the instance for
 * this class.
 */
class RPCServer {
public:
    typedef std::pair<std::pair<std::shared_ptr<RPCService>, RPCService *>, RPCMethodDescriptor *> ServiceMethodPair;
    typedef std::map<std::string, autil::ThreadPoolBasePtr> ThreadPoolMap;
    typedef std::map<RPCService *, autil::ThreadPoolBasePtr> ServiceThreadPoolMap;
    typedef std::unordered_map<uint32_t, ServiceMethodPair> RPCCallMap;

public:
    /**
     * The only constructor.
     */
    RPCServer(size_t threadNum, size_t queueSize);
    virtual ~RPCServer();

public:
    virtual ServiceMethodPair GetRpcCall(const CallId &callId) const;

    /**
     * Start listening a specific port according to the address string.
     * @todo user want to pass port of 0, then he wants to know which port is
     * actually listen after Listen be called
     * @param address A string to indicate listen address. Format is like this:
     *                <proto>:<addr>:port. A real example: "tcp:0.0.0.0:7893".
     *                Another example for domain socket:
     *                "domainstream:/tmp/ds-chunk.socket"
     * @param timeout default post packet timeout (ms) for all the accepted socket.
     *        will affect all the reply sending back to the client.
     * @param maxIdleTime This parameter determines after how long time idle, server
     *        will break the idle connections intentinally.
     * @attention This function can get called multiple times so that one ARPCServer
     *            can monitor more than one port.
     */
    virtual bool Listen(const std::string &address,
                        int timeout = 5000,
                        int maxIdleTime = MAX_IDLE_TIME,
                        int backlog = LISTEN_BACKLOG) = 0;

    /**
     * Stops RPCServer. Will terminate all the ongoing threads and destroy all
     * the port listening.
     * @return This function will always return true.
     */
    virtual bool Close() = 0;

    virtual int GetListenPort() = 0;

    /**
     * Register protobuf service to this ARPCServer, optionally assigned
     * it to a specific thread pool.
     * @param rpcService The service object pointer which is defined by protobuf
     * generated code. Can map to the "service" defination in .proto file.
     * @param threadPoolDescriptor Describes thread pool that this service will
     *        map to. If not indicated, this service will be binded to default
     *        thread pool.
     */
    bool RegisterService(RPCService *rpcService, ThreadPoolDescriptor threadPoolDescriptor = ThreadPoolDescriptor());
    bool RegisterService(const std::shared_ptr<RPCService> &rpcService,
                         const std::string &methodName,
                         ThreadPoolDescriptor threadPoolDescriptor);

    bool RegisterService(RPCService *rpcService, const autil::ThreadPoolBasePtr &pool);

    bool RegisterThreadPool(const autil::ThreadPoolBasePtr &pool);
    /**
     * Get thread pool shared pointer by its name.
     */
    autil::ThreadPoolBasePtr GetThreadPool(const std::string &threadPoolName = DEFAULT_TREAHDPOOL_NAME) const;

    /**
     * Get all thread pool names
     */
    std::vector<std::string> GetThreadPoolNames() const;

    /**
     * Get thread pool shared pointer by RPC service pointer.
     * One or more services will map to one thread pool.
     */
    autil::ThreadPoolBasePtr GetServiceThreadPool(RPCService *rpcService) const;

    version_t GetVersion() { return _version; }

    virtual void dump(std::ostringstream &out);

    void StopServer();

    void RecoverServer();

    size_t GetThreadPoolNum() {
        autil::ScopedReadLock lock(_threadPoolLock);
        return _threadPoolMap.size();
    }

    bool isStopped() { return _isStopped; }

    MessageCodec *GetMessageCodec() { return _messageCodec; }
    void stopThreadPools();
    void removeServiceThreadPool(RPCService *rpcService);

    RPCCallMap GetRPCCallMap() const;
    void SetRPCCallMap(const RPCCallMap &newMap);
    void unRegisterService(const std::shared_ptr<RPCService> &rpcService);

protected:
    bool addAndStartThreadPool(const ThreadPoolDescriptor &desc);
    bool doRegisterService(const std::shared_ptr<RPCService> &rpcService, const std::string &methodName);
    virtual bool doRegisterService(RPCService *rpcService);
    void reconstructThreadPoolMap();

protected:
    bool _isStopped;
    RPCServerAdapter *_serverAdapter;
    MessageCodec *_messageCodec;

    mutable autil::ReadWriteLock _mutex;
    RPCCallMap _rpcCallMap;

    ThreadPoolMap _threadPoolMap;
    mutable autil::ReadWriteLock _threadPoolLock;
    ServiceThreadPoolMap _serviceThreadPoolMap;
    mutable autil::ReadWriteLock _serviceThreadPoolLock;

    version_t _version;
    size_t _defaultThreadNum;
    size_t _defaultQueueSize;
    mutable autil::ReadWriteLock _sharedThreadPoolLock;
    std::string _sharedThreadPoolName;
    bool _poolOverride = false;
};

TYPEDEF_PTR(RPCServer);

ARPC_END_NAMESPACE(arpc);

