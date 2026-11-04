#pragma once
#include <unordered_map>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/iocomponent.h"

namespace anet {
class UDPConnection;

class UDPComponent : public IOComponent {

public:
    /**
     * 构造函数，由Transport调用。
     *
     * @param owner:      Transport
     * @param socket:     Socket
     * @param streamer:   数据包的双向流，用packet创建，解包，组包。
     * @param serverAdapter:  用在服务器端，当Connection初始化及Channel创建时回调时用
     */
    UDPComponent(Transport *owner, Socket *socket, IPacketStreamer *streamer, IServerAdapter *serverAdapter);

    /*
     * 析构函数
     */
    ~UDPComponent();

    /*
     * 初始化
     *
     * @return 是否成功
     */
    bool init(bool isServer = false);

    /*
     * 关闭
     */
    void close();

    /*
     * 当有数据可写到时被Transport调用
     *
     * @return 是否成功, true - 成功, false - 失败。
     */
    bool handleWriteEvent();

    /*
     * 当有数据可读时被Transport调用
     *
     * @return 是否成功, true - 成功, false - 失败。
     */
    bool handleReadEvent();

private:
    std::unordered_map<int, UDPConnection *> _connections; // UDP连接集合
    IPacketStreamer *_streamer;                                 // streamer
    IServerAdapter *_serverAdapter;                             // serveradapter
};
} // namespace anet

