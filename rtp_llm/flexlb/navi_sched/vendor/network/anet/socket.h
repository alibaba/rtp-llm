#pragma once
#include <errno.h>
#include <sstream>
#include <stdint.h>
#include <string>
#include <sys/socket.h>
#include <vector>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/addrspec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/atomic.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connectionpriority.h"

struct sockaddr_in;

namespace anet {
const int LISTEN_BACKLOG = 256;

class IOComponent;

class Socket {
public:
    Socket(int family = AF_INET);
    virtual ~Socket();

    bool setAddress(const char *address, const int port);
    /*
     * Set domain socket address. */
    bool setUnixAddress(const char *unixPath);

    bool setAddrSpec(const char *addrspec);

    bool connect();
    bool reconnect();

    void close();
    void shutdown();

    bool createUDP();

    /*
     * Set socket fd and ip address
     * @param socketHandle: socket fd
     * @param hostAddress: host address
     */
    void setUp(int socketHandle, struct sockaddr *hostAddress, int family = AF_INET);

    /*
     * get socket fd
     *
     * @return fd
     */
    int getSocketHandle();

    virtual ssize_t write(const void *data, size_t len);
    virtual ssize_t writev(const std::vector<iovec> &iovs);
    virtual ssize_t read(void *data, size_t len);

    bool setKeepAlive(bool on) { return setIntOption(SO_KEEPALIVE, on ? 1 : 0); }

    bool setKeepAliveParameter(int idleTime, int keepInterval, int cnt);

    bool setReuseAddress(bool on) { return setIntOption(SO_REUSEADDR, on ? 1 : 0); }

    bool setSoLinger(bool doLinger, int seconds);

    bool setTcpNoDelay(bool noDelay);

    bool setIntOption(int option, int value);

    bool setSoBlocking(bool on);

    bool checkSocketHandle();

    int setPriority(CONNPRIORITY priority);

    int getPriority();

    virtual int getSoError();

    char *getAddr(char *dest, int len, bool active = false);
    int getPort(bool active = true);
    virtual char *getRemoteAddr(char *dest, int len);

    static int getLastError() { return errno; }

    inline bool isAddressValid() { return addr.isValidState(); }

    int getProtocolType(void) { return addr.getProtocolType(); }

    /* Functions restructed from ServerSocket class */
    Socket *accept();
    bool listen(int backlog);

    /* below functions are for statics purpose */
    int64_t getBindErrCnt() { return atomic_read(&_bindErrCnt); }
    int64_t getConnectErrCnt() { return atomic_read(&_connectErrCnt); }
    int64_t getWriteErrCnt() { return atomic_read(&_writeErrCnt); }
    int64_t getReadErrCnt() { return atomic_read(&_readErrCnt); }
    int64_t getAcceptErrCnt() { return atomic_read(&_acceptErrCnt); }
    int64_t getListenErrCnt() { return atomic_read(&_listenErrCnt); }
    static int64_t getAccpetConnCnt() { return atomic_read(&_globalAcceptConnCnt); }

    void dump(const char *leadchar, std::ostringstream &buf) {
        std::string leading(leadchar);
        buf << leading << "socket FD: " << _socketHandle << std::endl;
        char tmp[256];
        getAddr(tmp, 256);
        buf << leading << "address: " << tmp << std::endl;
        buf << leading << "Bind Err: " << getBindErrCnt() << "\t" << "Connect Err: " << getConnectErrCnt() << "\t"
            << "Write Err: " << getWriteErrCnt() << std::endl;
        buf << leading << "Read Err: " << getReadErrCnt() << "\t" << "Accept Err: " << getAcceptErrCnt() << "\t"
            << "Listen Err: " << getListenErrCnt() << std::endl;
    }

    bool bindLocalAddress(const char *localIp, uint16_t localPort);
    int getProtocolFamily();

protected:
    int _socketHandle;
    AddrSpec addr;
    std::string _addrSpec;

private:
    atomic64_t _bindErrCnt;
    atomic64_t _connectErrCnt;
    atomic64_t _writeErrCnt;
    atomic64_t _readErrCnt;
    atomic64_t _acceptErrCnt;
    atomic64_t _listenErrCnt;
    static atomic64_t _globalAcceptConnCnt;

    int rebindconn();
    void clearCnt();
    void bindErrInc() { atomic_inc(&_bindErrCnt); }
    void connectErrInc() { atomic_inc(&_connectErrCnt); }
    void writeErrInc() { atomic_inc(&_writeErrCnt); }
    void readErrInc() { atomic_inc(&_readErrCnt); }
    void acceptErrInc() { atomic_inc(&_acceptErrCnt); }
    static void acceptConnInc() { atomic_inc(&_globalAcceptConnCnt); }
    static void acceptConnDec() { atomic_dec(&_globalAcceptConnCnt); }
    void listernErrInc() { atomic_inc(&_listenErrCnt); }

    void printIPAddr(char *dest, int len, sockaddr_storage &address);
    bool getSockAddr(sockaddr_storage &addr, bool active = true);
};

} // namespace anet

