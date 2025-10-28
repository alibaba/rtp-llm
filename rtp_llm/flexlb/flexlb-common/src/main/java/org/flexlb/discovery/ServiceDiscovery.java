package org.flexlb.discovery;

import org.flexlb.dao.master.WorkerHost;

import java.util.List;

/**
 * ServiceDiscovery - 服务发现接口
 *
 * @author saichen.sm
 */
public interface ServiceDiscovery {

    /**
     * 根据服务地址同步获取主机列表
     *
     * @param address 服务地址
     * @return 主机列表
     */
    List<WorkerHost> getHosts(String address);

    /**
     * 监听服务地址的主机变化
     *
     * @param address  服务地址
     * @param listener 主机变化监听器
     */
    void listen(String address, ServiceHostListener listener);

    /**
     * 停止所有监听
     */
    void shutdown();
}
