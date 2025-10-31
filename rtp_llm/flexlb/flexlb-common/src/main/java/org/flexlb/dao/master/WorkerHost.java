package org.flexlb.dao.master;

import lombok.Getter;

/**
 * WorkerHost - 工作节点主机信息
 * 统一的主机信息表示，用于服务发现和工作节点管理
 *
 * @author saichen.sm
 * @since 2025/4/24
 */
@Getter
public class WorkerHost {

    /**
     * 主机IP地址
     */
    private final String ip;
    /**
     * HTTP端口
     */
    private final int httpPort;
    /**
     * gRPC端口（通常为 httpPort + 1）
     */
    private final int grpcPort;
    /**
     * C++ HTTP服务端口
     */
    private final int httpServerPort;
    /**
     * 机房/站点信息
     */
    private final String site;
    /**
     * 工作组名称
     */
    private final String group;

    /**
     * 完整构造函数
     *
     * @param ip             主机IP地址
     * @param httpPort       HTTP端口
     * @param grpcPort       gRPC端口
     * @param httpServerPort C++ HTTP服务端口
     * @param site           机房/站点信息
     * @param group          工作组名称
     */
    public WorkerHost(String ip, int httpPort, int grpcPort, int httpServerPort, String site, String group) {
        this.ip = ip;
        this.httpPort = httpPort;
        this.grpcPort = grpcPort;
        this.httpServerPort = httpServerPort;
        this.site = site != null ? site : "";
        this.group = group != null ? group : "";
    }

    /**
     * 简化构造函数（用于服务发现场景）
     *
     * @param ip   主机IP地址
     * @param port 主端口
     * @param site 机房/站点信息
     */
    public WorkerHost(String ip, int port, String site) {
        this(ip, port, port + 1, port + 5, site, "");
    }

    /**
     * 最简构造函数（用于基础服务发现场景）
     *
     * @param ip   主机IP地址
     * @param port 主端口
     */
    public WorkerHost(String ip, int port) {
        this(ip, port, "");
    }

    /**
     * 获取IP:端口格式的字符串
     *
     * @return IP:Port格式字符串
     */
    public String getIpPort() {
        return ip + ":" + httpPort;
    }

    /**
     * 获取主端口（通常是HTTP端口）
     *
     * @return 主端口号
     */
    public int getPort() {
        return httpPort;
    }

    /**
     * 创建WorkerHost实例
     *
     * @param ip   主机IP
     * @param port 主机端口
     * @return WorkerHost实例
     */
    public static WorkerHost of(String ip, int port) {
        return new WorkerHost(ip, port);
    }

    /**
     * 创建WorkerHost实例
     *
     * @param ip   主机IP
     * @param port 主机端口
     * @param site 站点信息
     * @return WorkerHost实例
     */
    public static WorkerHost of(String ip, int port, String site) {
        return new WorkerHost(ip, port, site);
    }
}
