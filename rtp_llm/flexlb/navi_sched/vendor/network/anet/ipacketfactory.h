#pragma once

namespace anet {

class Packet;

/**
 * IPacketFactory is used to construct IPacketStreamer.
 * When ANET read response form peer, it will use IPacketFactory
 * to construct a new packet.
 */
class IPacketFactory {
public:
    virtual ~IPacketFactory() {}

    /**
     * The function is called when ANET need to construct a new
     * Packet. Which type of Packet will be generated in ANET is
     * indicated in this function.
     *
     * @param pcode type of Packet
     * @return address of new Packet. NULL if no Packet generated.
     */
    virtual Packet *createPacket(int pcode) = 0;
};
} // namespace anet

