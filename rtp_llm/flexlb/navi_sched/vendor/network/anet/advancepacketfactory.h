#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipacketfactory.h"

BEGIN_ANET_NS();
template <class T>
class AdvancePacketFactory : public IPacketFactory {
public:
    AdvancePacketFactory();

    ~AdvancePacketFactory();

    virtual Packet *createPacket(int pcode);
};

/*****************************************************************************/
/*****************************************************************************/
template <class T>
AdvancePacketFactory<T>::AdvancePacketFactory() {}

template <class T>
AdvancePacketFactory<T>::~AdvancePacketFactory() {}

template <class T>
Packet *AdvancePacketFactory<T>::createPacket(int pcode) {
    /* We assume that no method id can be greater than 32768 for now.
     * Below is a bad assertion since server will treat pcode as
     * service id + method id, which is likely to trigger this assertion.*/
    // DBGASSERT(pcode >= 0);

    /* When pcode == 0, it normally means that this comes form the path for
     * client to allocate the packet and try to initiate a RPC. In this case
     * We should create AdvancePacket. */
    if (AdvancePacket<T>::IsAdvancePacket(pcode))
        return new AdvancePacket<T>;
    else
        return new T;
}

END_ANET_NS();

