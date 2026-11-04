#pragma once

#define READ_WRITE_SIZE 8192
#include <sstream>
#include <stddef.h>
#include <stdint.h>
#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/channelpool.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connectionpriority.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/iconnection.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/iocomponent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipacketstreamer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/orderedpacketqueue.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/threadcond.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/threadmutex.h"

namespace anet {
class IPacketHandler;
class Packet;
class IOComponent;
class IServerStreamer;
class IServerAdapter;
class Socket;

struct ConnStat {
    std::string localAddr;
    std::string peerAddr;
    int64_t totalRxBytes{0};   // total bytes read
    int64_t totalTxBytes{0};   // total bytes write
    int64_t packetPosted{0};   // total packet posted
    int64_t packetHandled{0};  // total packet handled
    int64_t packetTimeout{0};  // total packet timeout
    int64_t queueSize{0};      // queue size on last writeData
    int64_t callWriteCount{0}; // call write total count
    int64_t callReadCount{0};  // call read total count
};

class Connection : public IConnection {
    friend class TCPConnectionTest_testWriteData_Test;
    friend class TCPConnectionTest_testReadData_Test;
    friend class TCPConnectionTest_testLock_Test;
    friend class TCPConnectionTest_testMemLeak_Test;
    friend class ConnectionTest_testPostPacket_Test;
    friend class ConnectionTest_testHandlePacket_Test;
    friend class ConnectionTest_testCheckQueueTimeout_Test;
    friend class ConnectionTest_testCheckChannelPoolTimeout_Test;
    friend class ConnectionTest_testCloseHook_Test;
    friend class ConnectionTest_testHTTPTimeoutAfterSend_Test;
    friend class ConnectionTest_testGetPacketPostedCount_Test;
    friend class ConnectionTest_testGetPacketHandledCount_Test;
    friend class ConnectionTest_testUpdateQueueStatus_Test;
    friend class ConnectionTest_testInitReadWriteBufSizeFromEnv_Test;
    friend class DirectTCPConnectionTest_testReadData_Test;
    friend class DirectTCPConnectionTest_testReadDataAbnormal_Test;
    friend class DirectTCPConnectionTest_testWriteData_Test;
    friend class DirectTCPConnectionTest_testWriteDataAbnormal_Test;

public:
    /* Callback state used by R/W thread
     *
     * CB_RUNNING will be set just before R/W thread transfer control to user's callback,
     * and reset to CB_IDLE immediately after the callback finishes.
     */
    enum RWCBState {
        RW_CB_IDLE,
        RW_CB_RUNNING,
    };

    Connection(Socket *socket, IPacketStreamer *streamer, IServerAdapter *serverAdapter);

    virtual ~Connection();

    void setServer(bool isServer) { _isServer = isServer; }

    bool isServer() { return _isServer; }

    void setIOComponent(IOComponent *ioc) { _iocomponent = ioc; }

    IOComponent *getIOComponent() { return _iocomponent; }

    void setDefaultPacketHandler(IPacketHandler *ph) { _defaultPacketHandler = ph; }

    IPacketHandler *getDefaultPacketHandler() { return _defaultPacketHandler; }

    /**
     * async interface
     *
     * @param packet
     * @param packetHandler
     * @param args
     * @param block whether block if output queue is full.
     */
    bool postPacket(Packet *packet, IPacketHandler *packetHandler = NULL, void *args = NULL, bool block = false);

    /**
     * Send a request packet to server, then waiting for the replay packet.
     * If the return value is NULL, caller is responsible for freeing the
     * request packet; If the return value is NOT NULL, ANet will free the
     * request packet, and caller is responsible for freeing the reply packet.
     *
     * @param packet the request to be sent to server
     * @param block whether block if output queue is full.
     * @return the value returned is a pointer to a reply packet received
     * from server, or NULL if fail to send the reply.
     */
    Packet *sendPacket(Packet *packet, bool block = false);

    bool handlePacket(Packet *header);

    int64_t getPacketPostedCount() { return _stats.packetPosted; }
    int64_t getPacketHandledCount() { return _stats.packetHandled; }
    int64_t getPacketTimeoutCount() { return _stats.packetTimeout; }

    bool checkTimeout(int64_t now);

    void closeHook();
    void openHook();

    virtual bool writeData() = 0;

    virtual bool readData() = 0;

    virtual void setWriteFinishClose(bool v) {}

    virtual void setWriteFinishShutdown(bool v) {}

    void setQueueTimeout(int queueTimeout) { _queueTimeout = queueTimeout; }

    virtual void clearOutputBuffer() { ; }

    virtual void clearInputBuffer() { ; }

    virtual int64_t getInputBufferSpaceAllocated() { return 0; }
    virtual int64_t getOutputBufferSpaceAllocated() { return 0; }
    int64_t getDataWriteBytes() { return _stats.totalTxBytes; }
    int64_t getDataReadBytes() { return _stats.totalRxBytes; }

    bool setQueueLimit(size_t limit);
    size_t getQueueLimit();

    void setQueueConsumeTimeout(int64_t timeout) { _queueConsumeTimeout = timeout; }

    virtual void close();
    virtual void addRef();
    virtual void subRef();
    virtual int getRef();

    virtual bool isClosed();
    virtual bool isConnected();

    void wakeUpSender();
    void beforeCallback();
    void afterCallback();

    void resetContext();
    StreamingContext *getContext() { return _context; }

    void setServer(IServerAdapter *serverAdapter) { _serverAdapter = serverAdapter; }

    int setPriority(CONNPRIORITY priority);

    CONNPRIORITY getPriority();

    bool setKeepAliveParameter(int idleTime, int keepInterval, int cnt);

    /* manually reconnect to the target, will always return error when
     * auto reconnection is enabled. */
    RECONNErr reconnect();

    char *getLocalAddr(char *dest, int len);
    char *getIpAndPortAddr(char *dest, int len);
    Socket *getSocket() { return _socket; }

    void dump(const char *lead, std::ostringstream &buf) {
        std::string leading(lead);

        buf << leading << "Connection Addr: " << this << "\t"
            << "Is Server: " << _isServer << "\t"
            << "Closed: " << _closed << std::endl;
        buf << leading << "Socket: " << _socket << "\t"
            << "Priority: " << getPriority() << "\t"
            << "DefaultPacketHandler: " << _defaultPacketHandler << std::endl;
        buf << leading << "Streamer: " << _streamer << "\t"
            << "ServerAdapter: " << _serverAdapter << std::endl;
        buf << leading << "ChannelPool: " << &_channelPool << std::endl;
        buf << leading << "Queue Timeout: " << _queueTimeout << "ms"
            << "\t"
            << "Queue Limit: " << _queueLimit << std::endl;
        buf << leading << "Waiting Threads: " << _waitingThreads << std::endl;
        buf << leading << "Packet Posted: " << _stats.packetPosted << "\t"
            << "Packet Handled: " << _stats.packetHandled << "\t"
            << "Packet Timeout: " << _stats.packetTimeout << std::endl;
        buf << leading << "Bytes Write: " << _stats.totalTxBytes << "\t"
            << "Bytes Read: " << _stats.totalRxBytes << std::endl;
    }

    RWCBState getRWCBState() { return _rwCbState; }

    ConnStat getConnStat();

protected:
    IPacketHandler *_defaultPacketHandler;
    bool _isServer;
    IOComponent *_iocomponent;
    Socket *_socket;
    IPacketStreamer *_streamer;     // Packet Streamer
    IServerAdapter *_serverAdapter; // Server Adapter

    OrderedPacketQueue _outputQueue; // Output Queue
    OrderedPacketQueue _myQueue;     // temporary queue used in write data
    ThreadCond _outputCond;          // output condition
    ChannelPool _channelPool;
    int _queueTimeout;  // tiemout spesification (millseconds)
    size_t _queueLimit; // max pending set packet allowed
    int64_t _queueConsumeTimeout; // waiting timeout in outputQueue (microsecond/μs)
    int _waitingThreads;
    bool _closed;
    StreamingContext *_context;

    ConnStat _stats;

    ThreadMutex _addrMutex;
    char *_socketAddr;
    size_t _socketAddrLen;
    uint64_t _jobId;
    uint32_t _insId;
    uint32_t _qosId;
    int64_t _lasttime;

    size_t _readWriteBufSize{READ_WRITE_SIZE};

private:
    RWCBState _rwCbState;
};
} // namespace anet

