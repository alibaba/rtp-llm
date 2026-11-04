#pragma once

namespace anet {

#define MAX_SOCKET_EVENTS 256

class IOComponent;
class Socket;

class IOEvent {

public:
    bool _readOccurred;  // 读发生
    bool _writeOccurred; // 写发生
    bool _errorOccurred; // 错误发生
    IOComponent *_ioc;   // 回传参数
};

class EPollSocketEvent {

public:
    /*
     * 构造函数
     */
    EPollSocketEvent();

    /*
     * 析造函数
     */
    ~EPollSocketEvent();

    /*
     * 增加Socket到事件中
     *
     * @param socket 被加的socket
     * @param enableRead: 设置是否可读
     * @param enableWrite: 设置是否可写
     * @return  操作是否成功, true  成功, false  失败
     */
    bool addEvent(Socket *socket, bool enableRead, bool enableWrite, void *data);

    /*
     * 设置删除Socket到事件中
     *
     * @param socket 被加的socket
     * @param enableRead: 设置是否可读
     * @param enableWrite: 设置是否可写
     * @return  操作是否成功, true  成功, false  失败
     */
    bool setEvent(Socket *socket, bool enableRead, bool enableWrite, void *data);

    /*
     * 删除Socket到事件中
     *
     * @param socket 被删除socket
     * @return  操作是否成功, true  成功, false  失败
     */
    bool removeEvent(Socket *socket, void *data);

    /*
     * 得到读写事件。
     *
     * @param timeout  超时时间(单位:ms)
     * @param events  事件数组
     * @param cnt   events的数组大小
     * @return 事件数, 0为超时
     */
    int getEvents(int timeout, IOEvent *events, int cnt);

    void wakeUp();

private:
    int _iepfd;    // epoll的fd
    int _pipes[2]; // pipe for wakeup
    //    ThreadMutex _mutex;  // 对fd操作加锁
};
} // namespace anet

