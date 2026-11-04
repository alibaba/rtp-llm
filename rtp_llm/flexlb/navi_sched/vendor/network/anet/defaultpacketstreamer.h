#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipacketstreamer.h"

namespace anet {

class DefaultPacketStreamer : public IPacketStreamer {
public:
    DefaultPacketStreamer(IPacketFactory *factory);

    ~DefaultPacketStreamer();

    virtual StreamingContext *createContext();

    /**
     * get information about the packet in input data buffer
     *
     * @param input  input data buffer
     * @param header packet header
     * @return retrun true if we get the packet header; return false if not.
     */
    bool getPacketInfo(DataBuffer *input, PacketHeader *header, bool *broken);

    /**
     * decode the data in input buffer to get a packet
     *
     * @param input
     * @param header
     * @return return a packet if we get it. return NULL if not
     */
    Packet *decode(DataBuffer *input, PacketHeader *header);

    /**
     * encode a packet into output data buffer
     *
     * @param packet packet to be encoded
     * @param output output data buffer
     * @return return true if we finished encode. return false if not
     */
    bool encode(Packet *packet, DataBuffer *output);

    bool processData(DataBuffer *dataBuffer, StreamingContext *context);
};
} // namespace anet

