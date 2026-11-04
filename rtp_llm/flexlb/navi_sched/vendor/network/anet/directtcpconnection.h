#pragma once

#include <stddef.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/directpacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/tcpconnection.h"

namespace anet {

class DirectPacketStreamer;
class DirectStreamingContext;
class IPacketStreamer;
class IServerAdapter;
class Socket;

class DirectTCPConnection : public TCPConnection {
public:
    DirectTCPConnection(Socket *socket, IPacketStreamer *streamer, IServerAdapter *serverAdapter);
    ~DirectTCPConnection();
    bool writeData() override;
    bool readData() override;

private:
    void clearWritingPacket();
    void finishPacketWrite();
    ssize_t sendBuffer(int &writeCnt, int &error);
    ssize_t sendPayload(int &writeCnt, int &error);

private:
    DirectPacketStreamer *_directStreamer;
    DirectStreamingContext *_directContext;
    size_t _payloadLeftToWrite = 0UL;
    DirectPayload _writingPayload;
    DirectPacket *_writingPacket = nullptr;
};

} // namespace anet

