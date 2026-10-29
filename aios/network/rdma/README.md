# RDMA accl 封装层说明

本目录是对内部 accl-barex 接口的 C++ 封装层 (mempool / channel / client / server 等), 供 RDMA 传输与 cache_store 等实现使用.

注意: 公开源码树不包含 accl-barex 实现. 默认配置(use_rdma 关闭)下本目录编译为 no-accl 桩实现; 真实 RDMA 传输链路需要内源 overlay 及内部 accl-barex 仓库.

## 代码结构
accl/ 目录下的文件为对接 accl 的封装类 (RdmaMempool, RdmaChannelClient/Server, AcclClient/ServerWrapper 等), 其 unittest/ 子目录为基于 mock 接口实现的各个类的单元测试和基于实际 rdma 环境的各种测试.

## 类生命周期设计
1. 保证accl能够跟随server /client 生命周期创建和销毁. 因为Accl内部用了一些单例(eg: XDeviceManager), 因此需要保证数据链路一定要先于其他首先显式销毁.
2. channel和connection可以随时调用. 但是channel和connection本身会保证如果accl销毁, 不会因此触发问题.
3. 保证request相关的生命周期与accl无关
4. RdmaMempool的生命周期是最长的.

## RdmaChannelClient StateChange
* 不支持reConnect;

### 状态转换:
#### CHANNEL_INIT -> CHANNEL_BROKEN
1. 传入了不可用的AcclClientWrapper
2. 调用AcclClientWrapper的Connect直接失败.

#### CHANENL_INIT -> CHANNEL_CONNECTING -> CHANNEL_BROKEN
1. 调用AcclClientWrapper的Connect返回成功
2. 连接失败, 触发OnConnectFailed

#### CHANNEL_INIT -> CHANNEL_CONNECTING -> CHANNEL_CONNECTED -> CHANNEL_BROKEN
1. 调用AcclClientWrapper的Connect返回成功
2. 连接成功, 触发OnConnectSuccess
3. 主动或被动断连, 触发OnConnectFailed

#### * -> CHANNEL_BROKEN
1. 外部直接调用Close方法, 强制触发断连
