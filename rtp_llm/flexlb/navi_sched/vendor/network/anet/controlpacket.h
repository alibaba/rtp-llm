#pragma once
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"

namespace anet {

class ControlPacket : public Packet {
public:
    enum {
        CMD_BAD_PACKET = 1,
        CMD_TIMEOUT_PACKET,
        CMD_CONNECTION_CLOSED,      // packet-level: triggered when post packet at a closed connection
        CMD_RECEIVE_NEW_CONNECTION, // connection-level: triggered when tcpacceptor accepts a new connection
        CMD_CLOSE_CONNECTION,       // connection-level: triggered when close the given connection
        CMD_END
    };

    static ControlPacket BadPacket;
    static ControlPacket TimeoutPacket;
    static ControlPacket ConnectionClosedPacket;
    static ControlPacket ReceiveNewConnectionPacket;
    static ControlPacket CloseConnectionPacket;

public:
    /**
     * @param c command code
     * @param freeDelete a flag indicating if this object will be delete when
     * free() is called by user.
     */
    ControlPacket(int c, bool freeDelete = true);

    bool isRegularPacket() { return false; }

    void free();

    void countDataLen() {}

    bool encode(DataBuffer *output) { return false; }

    bool decode(DataBuffer *input, PacketHeader *header) { return false; }

    int getCommand() { return _command; }

    /**
     * get description of this packet
     **/
    const char *what() { return whatCmd(_command); }

public:
    static const char *whatCmd(int command) {
        switch (command) {
        case CMD_BAD_PACKET:
            return "Bad Packet received";
            break;
        case CMD_TIMEOUT_PACKET:
            return "Packet Timeout";
            break;
        case CMD_CONNECTION_CLOSED:
            return "Drop packet due to closing connection";
            break;
        case CMD_RECEIVE_NEW_CONNECTION:
            return "Receive new connection";
        case CMD_CLOSE_CONNECTION:
            return "Close connection";
        default:
            return "NULL";
        }
    }

private:
    int _command;
    bool _freeDelete;
};

} // namespace anet

