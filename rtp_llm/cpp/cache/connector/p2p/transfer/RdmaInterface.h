#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/proto/service.pb.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <map>

namespace rtp_llm {

struct RemoteBuffer {
    int64_t                                       addr;
    uint32_t                                      len;
    std::shared_ptr<std::map<uint32_t, uint32_t>> nic_rkeys;
    RemoteBuffer(int64_t addr, uint32_t len, const std::shared_ptr<std::map<uint32_t, uint32_t>>& nic_rkeys):
        addr(addr), len(len), nic_rkeys(nic_rkeys) {}
};

// RDMA 内存管理器接口类
// 提供 RDMA MR 管理的接口
class IRdmaMemoryManager {
public:
    virtual ~IRdmaMemoryManager() = default;

    // 注册 buffer 为 RDMA MR
    // @param buffer: 要注册的 BufferPtr
    // @param aligned_size: 对齐大小（可选，默认 0）
    // @return: 是否注册成功
    virtual bool regUserMr(const BufferPtr& buffer, uint64_t aligned_size = 0) = 0;

    // 注销 buffer 的 RDMA MR
    // @param buffer: 要注销的 BufferPtr
    // @return: 是否注销成功
    virtual bool deregUserMr(const BufferPtr& buffer) = 0;

    // 查找 buffer 的 MR 信息
    // @param buffer: 要查找的 BufferPtr
    // @param mem_info: 输出的 MR 信息（accl::barex::memp_t*）
    // @return: 是否成功找到
    virtual std::shared_ptr<RemoteBuffer> findMemoryMr(const BufferPtr& buffer) = 0;
};

// RDMA 连接接口类
// 提供连接状态管理和数据读取接口
class IRdmaConnection {
public:
    virtual ~IRdmaConnection() = default;

    // 读取接口
    // @param local_remote_buffers: 本地 buffer 和远程 BlockBufferInfo 的配对列表
    // @param callback: 完成回调，参数为是否成功
    // @param deadline_ms: 超时时间（绝对时间戳，毫秒）
    virtual void read(const std::vector<std::pair<BufferPtr, std::shared_ptr<RemoteBuffer>>>& local_remote_buffers,
                      std::function<void(bool)>                                               callback,
                      uint64_t                                                                deadline_ms) = 0;
};

// RDMA 基础接口类
// 提供 RDMA MR 管理的通用接口
class IRdmaMemoryInterface {
public:
    virtual ~IRdmaMemoryInterface() = default;

    // 注册 buffer 为 RDMA MR
    // @param buffer: 要注册的 BufferPtr
    // @param aligned_size: 对齐大小（可选，默认 0）
    // @return: 是否注册成功
    virtual bool regUserMr(const BufferPtr& buffer, uint64_t aligned_size = 0) = 0;

    // 查找 buffer 的 MR 信息
    // @param buffer: 要查找的 BufferPtr
    // @param adopted: 是否为已采用的缓冲区
    // @param mem_info: 输出的 MR 信息（accl::barex::memp_t*）
    // @return: 是否成功找到
    virtual std::shared_ptr<RemoteBuffer> findMemoryMr(const BufferPtr& buffer, bool adopted) = 0;
};

// RDMA 客户端接口类
// 提供客户端连接管理接口
class IRdmaClient {
public:
    virtual ~IRdmaClient() = default;

    // 获取到指定地址的 RDMA 连接（连接复用，轮询机制）
    // @param ip: 目标 IP 地址
    // @param port: 目标端口（RDMA 端口）
    // @return: RDMA 连接指针，失败返回 nullptr
    // 注意：连接是复用的，多个调用者可以共享同一连接
    virtual std::shared_ptr<IRdmaConnection> getConnection(const std::string& ip, uint32_t port) = 0;
};

// RDMA 服务器接口类
// 提供服务器端监听接口
class IRdmaServer {
public:
    virtual ~IRdmaServer() = default;

    // 获取监听端口
    // @return: 监听端口号
    virtual uint32_t getListenPort() const = 0;
};

std::shared_ptr<IRdmaMemoryManager> createRdmaMemoryManager();

// 工厂方法：创建并初始化 RDMA 客户端实例
// @param memory_manager: RDMA 内存管理器
// @param io_thread_count: IO 线程数量
// @param worker_thread_count: 工作线程数量，默认 0
// @param rdma_connections_per_host: 每个主机的 RDMA 连接数量，默认 2
// @param connect_timeout_ms: 连接超时时间（毫秒），默认 250ms
// @return: RDMA 客户端接口指针，初始化失败返回 nullptr
std::shared_ptr<IRdmaClient> createRdmaClient(const std::shared_ptr<IRdmaMemoryManager>& memory_manager,
                                              int                                        io_thread_count,
                                              int                                        worker_thread_count       = 0,
                                              uint32_t                                   rdma_connections_per_host = 8,
                                              int                                        connect_timeout_ms = 250);

// 工厂方法：创建并初始化 RDMA 服务器实例
// @param memory_manager: RDMA 内存管理器
// @param listen_port: 监听端口
// @param io_thread_count: IO 线程数量
// @param worker_thread_count: 工作线程数量，默认 0
// @return: RDMA 服务器接口指针，初始化失败返回 nullptr
std::shared_ptr<IRdmaServer> createRdmaServer(const std::shared_ptr<IRdmaMemoryManager>& memory_manager,
                                              uint32_t                                   listen_port,
                                              int                                        io_thread_count,
                                              int                                        worker_thread_count = 0);

}  // namespace rtp_llm
