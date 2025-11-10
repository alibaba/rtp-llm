package org.flexlb.discovery;

import org.flexlb.dao.master.WorkerHost;

import java.util.List;

/**
 * ServiceHostListener - 服务主机变化监听器
 * 当服务主机列表发生变化时触发回调
 *
 * @author saichen.sm
 */
@FunctionalInterface
public interface ServiceHostListener {

    /**
     * 当主机列表发生变化时触发
     *
     * @param hosts 新的主机列表
     */
    void onHostsChanged(List<WorkerHost> hosts);
}
