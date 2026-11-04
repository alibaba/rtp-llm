#pragma once

#include <cstdlib>

namespace anet {
/* This class defines the interface of the Reactor/Proactor process
 * such as Transport in anet, io_service in asio and event_base in libevent.
 */
class ITransport {
public:
    /* This function will blocks the caller thread and checking forever if
     * any new event comes in the monitored handles.
     * @ret - the handlers being invoked during run().
     */
    virtual void run() = 0;

    /* Run the event loop for one time and then exit
     * @ret - the handlers being invoked during runOnce().
     */
    virtual void runOnce() = 0;

    /* Stop the event loop and all the framework threads.
     * All the threads that blocking in run() and runOnce() should get
     * returned immediately.
     * Note: this function won't wait for all the threads to exit, instead
     * it just send signals to all the threads and then return.
     */
    virtual bool stop() = 0;

    /* Reset the event processing framework to make it ready for 2nd run(),
     * runOnce() after Stop() is invoked.
     */
    virtual void reset() = 0;

    /*
     * This function will start a thread to check forever if
     * any new event comes in the monitored handles.
     */
    virtual void runInThread() = 0;

    virtual ~ITransport() {}
};

} // namespace anet

