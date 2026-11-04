#pragma once
#include <stdint.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connection.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/databuffer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ilogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/iocomponent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/log.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/socket.h"

namespace anet {

class DataBuffer;
class Socket;
class IPacketStreamer;
class IServerAdapter;

class TCPConnection : public Connection {
    friend class TCPConnectionTest_testWriteData_Test;

public:
    TCPConnection(Socket *socket, IPacketStreamer *streamer, IServerAdapter *serverAdapter);

    ~TCPConnection();

    /*
     * 写出数据
     *
     * @return 是否成功
     */
    bool writeData();

    /*
     * 读入数据
     *
     * @return 读入数据
     */
    bool readData();

    /*
     * 设置写完是否主动关闭
     */
    void setWriteFinishClose(bool v) { _writeFinishClose = v; }
    void setWriteFinishShutdown(bool v) { _writeFinishShutdown = v; }

    /*
     * 清空output的buffer
     */
    void clearOutputBuffer();

    /*
     * 清空output的buffer
     */
    void clearInputBuffer() { _input.clear(); }

    void addInputBufferSpaceAllocated(int64_t size);
    void addOutputBufferSpaceAllocated(int64_t size);
    int64_t getInputBufferSpaceAllocated() { return _inputBufferSpaceAllocated; };
    int64_t getOutputBufferSpaceAllocated() { return _outputBufferSpaceAllocated; };

    void shrinkInputBuffer();
    void shrinkOutputBuffer();

    int64_t getMaxRecvPacketSize() { return _maxRecvPacketSize; }
    int64_t getMaxSendPacketSize() { return _maxSendPacketSize; }

    void clearMaxRecvPacketSize() { _maxRecvPacketSize = 0; }
    void clearMaxSendPacketSize() { _maxSendPacketSize = 0; }

protected:
    DataBuffer _output;         // 输出的buffer
    DataBuffer _input;          // 读入的buffer
    PacketHeader _packetHeader; // 读入的packet header
    bool _gotHeader;            // packet header已经取过
    bool _writeFinishClose;     // 写完断开
    bool _writeFinishShutdown;  // 写完shutdown
    int64_t _inputBufferSpaceAllocated;
    int64_t _outputBufferSpaceAllocated;

    /*
     * These two variables are used to track the max recv/send packet size,
     * and will be reset to 0 at every shrink interval in timeout thread.
     */
    int64_t _maxRecvPacketSize;
    int64_t _maxSendPacketSize;
};

} // namespace anet

