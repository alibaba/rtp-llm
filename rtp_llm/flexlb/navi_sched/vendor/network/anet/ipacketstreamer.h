#pragma once
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/controlpacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipacketfactory.h"

namespace anet {
class DataBuffer;
class StreamingContext;
class IPacketFactory;

class IPacketStreamer {
    friend class DefaultPacketStreamerTest_testGetPacketInfo_Test;

public:
    IPacketStreamer(IPacketFactory *factory) {
        _factory = factory;
        _existPacketHeader = true;
    }

    virtual ~IPacketStreamer() {}

    virtual StreamingContext *createContext() = 0;

    /*
     * Determine whether data in DataBuffer is enough.
     * If return true,  ANET will process DataBuffer and construct
     * a Packet. If return false, ANET will read more data from
     * socket.
     *
     * @param input  data source
     * @param header some necessary information will be recorded in
     * header.
     * @return Return true if there's enough data, else return false.
     */
    virtual bool getPacketInfo(DataBuffer *input, PacketHeader *header, bool *broken) = 0;

    /*
     * 对包的解码
     *
     * @param input
     * @param header
     * @return 解码后的数据包
     */
    virtual Packet *decode(DataBuffer *input, PacketHeader *header) = 0;

    /*
     * 对Packet的组装
     *
     * @param packet 数据包
     * @param output 组装后的数据流
     * @return 是否成功
     */
    virtual bool encode(Packet *packet, DataBuffer *output) = 0;

    /*
     * 是否有数据包头
     */
    virtual bool existPacketHeader() { return _existPacketHeader; }

    /**
     * Read Data from DataBuffer and construct Packet. The Packet
     * is recorded in StreamingContext. This function is added
     * because muliti steps is needed when constructing a packet
     * from DataBuffer.
     * This virtual function is implemented in DefaultPacketStreamer.

     * @param dataBuffer source data
     * @param context The context when you processData.
     * @return true return true when processData successfully and a
     * packet will be generated. Return false when processData
     * failed.
     */
    virtual bool processData(DataBuffer *dataBuffer, StreamingContext *context) = 0;

protected:
    IPacketFactory *_factory; // 产生packet
    bool _existPacketHeader;  // 是否有packet header, 如http有自己协议就不需要输出头信息
};
} // namespace anet

