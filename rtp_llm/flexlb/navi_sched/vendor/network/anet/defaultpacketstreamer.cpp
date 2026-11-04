#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/defaultpacketstreamer.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/controlpacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/databuffer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/debug.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ilogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipacketfactory.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipacketstreamer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/log.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/packet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/streamingcontext.h"

namespace anet {

DefaultPacketStreamer::DefaultPacketStreamer(IPacketFactory *factory) : IPacketStreamer(factory) {}

DefaultPacketStreamer::~DefaultPacketStreamer() {}

StreamingContext *DefaultPacketStreamer::createContext() { return new StreamingContext; }

bool DefaultPacketStreamer::getPacketInfo(DataBuffer *input, PacketHeader *header, bool *broken) {
    if (_existPacketHeader) {
        if (input->getDataLen() < PACKET_INFO_LEN) {
            return false;
        }
        int flag = input->readInt32();
        header->_chid = input->readInt32();
        header->_pcode = input->readInt32();
        header->_dataLen = input->readInt32();
        if (flag != ANET_PACKET_FLAG || header->_dataLen < 0 || header->_dataLen > ANET_MAX_PACKET_SIZE) {
            *broken = true;
            ANET_LOG(WARN,
                     "Broken Packet Detected! ANET FLAG(%08X) VS Packet(%08X), Packet Length(%d)",
                     ANET_PACKET_FLAG,
                     flag,
                     header->_dataLen);
            return false; // if broken the head must be wrong
        }
    } else if (input->getDataLen() == 0) {
        return false;
    }
    return true;
}

Packet *DefaultPacketStreamer::decode(DataBuffer *input, PacketHeader *header) {
    ANET_LOG(ERROR, "SHOULD NOT INVOKE DefaultPacketStreamer::decode(...)!");
    assert(false);
    return NULL;
    //     Packet *packet = _factory->createPacket(header->_pcode);
    //     if (packet != NULL) {
    //         if (!packet->decode(input, header)) {
    //             packet->free();
    //             packet = NULL;
    //         }
    //     } else {
    //         input->drainData(header->_dataLen);
    //     }
    //     return packet;
}

bool DefaultPacketStreamer::encode(Packet *packet, DataBuffer *output) {
    PacketHeader *header = packet->getPacketHeader();

    // 为了当encode失败恢复时用
    size_t oldLen = output->getDataLen();
    // dataLen的位置
    ssize_t dataLenOffset = -1;
    int headerSize = 0;

    // pcode offset
    size_t pcodeOffset = 0;
    if (_existPacketHeader) {
        output->writeInt32(ANET_PACKET_FLAG);
        output->writeInt32(header->_chid);

        pcodeOffset = output->getDataLen();
        output->writeInt32(header->_pcode);

        dataLenOffset = output->getDataLen();
        output->writeInt32(0);
        headerSize = 4 * sizeof(int32_t);
    }
    // 写数据
    if (packet->encode(output) == false) {
        ANET_LOG(ERROR, "encode error");
        output->stripData(output->getDataLen() - oldLen);
        return false;
    }
    // 计算包长度
    size_t dataLen = output->getDataLen() - oldLen - headerSize;
    if (dataLen > ANET_MAX_PACKET_SIZE) {
        ANET_LOG(WARN, "dataLen:[%lu] exceed ANET_MAX_PACKET_SIZE", dataLen);
        output->stripData(output->getDataLen() - oldLen);
        return false;
    }
    header->_dataLen = static_cast<int32_t>(dataLen);

    // Assert the invalid situation
    DBGASSERT(output->getDataLen() >= oldLen + headerSize);

    // 最终把长度回到buffer中
    if (dataLenOffset >= 0) {
        output->fillInt32(dataLenOffset, header->_dataLen);
        /* It is possible that the pcode field is updated in packet->encode()
         * logic, so we have to refresh the value below. */
        output->fillInt32(pcodeOffset, header->_pcode);
    }

    return true;
}

bool DefaultPacketStreamer::processData(DataBuffer *dataBuffer, StreamingContext *context) {
    Packet *packet = context->getPacket();
    if (NULL == packet) {
        PacketHeader header;
        bool brokenFlag = false;
        // we did not get packet header;
        if (!getPacketInfo(dataBuffer, &header, &brokenFlag)) {
            context->setBroken(brokenFlag);
            return !brokenFlag && !context->isEndOfFile();
        }
        packet = _factory->createPacket(header._pcode);
        assert(packet);
        packet->setPacketHeader(&header);
        context->setPacket(packet);
        dataBuffer->ensureFree(header._dataLen);
    }

    PacketHeader *header = packet->getPacketHeader();
    if (dataBuffer->getDataLen() < header->_dataLen) {
        context->setBroken(context->isEndOfFile());
        return !context->isEndOfFile();
    }

    if (!packet->decode(dataBuffer, header)) {
        ControlPacket *cmd = new ControlPacket(ControlPacket::CMD_BAD_PACKET);
        assert(cmd);
        cmd->setPacketHeader(header);
        context->setPacket(cmd);
        /*fix ticket #145*/
        // packet->free();
    } else
        DBGASSERT(packet->getDataLen() >= 0);

    context->setCompleted(true);
    return true;
}

} // namespace anet

/////////////
