#pragma once

namespace anet {
class Packet;

class IPacketHandler {
public:
    enum HPRetCode {
        KEEP_CHANNEL = 0,
        CLOSE_CHANNEL = 1,
        FREE_CHANNEL = 2
    };

    /**
     * This function is used in Client. When ANET receive response,
     * it will call handlePacket() to deal with response packet.
     * User use ANET send requests and use handlePacket() deal
     * with response. This function is Called in
     * Connection::handlePacket().
     * In some exceptions, ControlPacket(such as TimeoutPacket,
     * BadPacket) will be received. Obviously, you should consider
     * all these exceptions in you implementation.
     *
     * @param packet response packet or ControlPacket
     * @param args it's used to indicate which request packet this
     * respose packet corresponding. It's the same as args you used
     * in Connection::posePacket().
     * @return not used temporary.
     */
    virtual HPRetCode handlePacket(Packet *packet, void *args) = 0;

    virtual ~IPacketHandler() {}
};
} // namespace anet

