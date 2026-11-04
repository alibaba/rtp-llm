#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connectionpriority.h"

namespace anet {
class Packet;
class IPacketHandler;
class IServerAdapter;

/* This class defines the interface of a connection class, which is used
 * to encapsulate packet based communication based on socket stream. */
class IConnection {
public:
    /* This function will defines the async packet post interface, a
     * packet handler object is used to define the call back function.
     * Timeout is passed in and if it is -1, the msg won't timeout.
     * @param packet: the packet to post
     * @param packetHandler: callback function
     * @param timeout: timeout value for the msg
     * @param blockWhenQueueFull: defines if blocks the caller when queue
     *        is full
     * @ret - true if the packet is queued successfully.
     */
    virtual bool postPacket(Packet *packet, IPacketHandler *packetHandler, void *args, bool blockWhenQueueFull) = 0;

    /* Synchronously send a packet to the connection with timeout.
     * @param packet: packet to send
     * @param timeout: timeout value for the send operation
     * @param blockWhenQueueFull: as is
     * @ret - reply packet or NULL if failed to send
     */
    virtual Packet *sendPacket(Packet *packet, bool blockWhenQueueFull) = 0;

    /* The interface to handle packet for server or client.
     * For server connection, user needs to pass in the server adapter
     * call back function.
     * @param header - the packet to handle.
     * TODO: one of the enhancement to ANET, we don't need to tell the
     * difference between server and client connections.
     */
    virtual bool handlePacket(Packet *header) = 0;

    /* Set this connection as server connection, together with the server
     * adapter. If server adapter is NULL, we would consider this connection
     * is not for server.
     * TODO: we should make connection decouple from server or client.
     */
    virtual void setServer(IServerAdapter *serverAdapter) = 0;

    /* Set the connection priority. All packets send from this connection
     * will derive the priority being set.
     */
    virtual int setPriority(CONNPRIORITY priority) = 0;

    /* Virtual destructor. */
    virtual ~IConnection(){};
};
} // namespace anet

